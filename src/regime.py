# Bayesian persistent regime-switching model for the spread (notebook 06).
#
# Observation equation (AR(1) around a regime-dependent mean):
#     y_t = (1 - phi) mu_{s_t} + phi y_{t-1} + eps_t,     eps_t ~ N(0, sigma2)
# State equation (persistent probit, driven by weather and seasonality):
#     z_t = alpha_0 + w_t' alpha + h_t' kappa + delta s_{t-1} + eta_t,  eta_t ~ N(0, 1)
#     s_t = 1[z_t > 0]
# Identification: mu_0 < mu_1. Priors are conjugate, so the Gibbs sampler draws
#   1. the state path s_{1:T} by forward-filtering backward-sampling (FFBS),
#   2. latent probit utilities z_t (truncated normal, Albert-Chib),
#   3. probit coefficients gamma = (alpha_0, alpha, kappa, delta),
#   4. ordered regime means (mu_0, mu_1),
#   5. AR coefficient phi (truncated to (-1, 1)),
#   6. sigma2 (inverse gamma).

import math

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm


#####################
# 1. Data preparation
#####################

def make_seasonal_features(dates, n_harmonics=2):
    """Fourier terms sin/cos(2 pi k d / 365.25), k = 1..n_harmonics, of the day of year d."""
    day = dates.dayofyear.to_numpy(dtype=float)
    features = {}
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * (day - 1.0) / 365.25
        features[f'season_sin_{k}'] = np.sin(angle)
        features[f'season_cos_{k}'] = np.cos(angle)
    return pd.DataFrame(features, index=dates)


def prepare_regime_data(df, spread_col, weather_cols, n_harmonics=2):
    """
    Arrays for the sampler: y_t, y_{t-1}, and the standardised state-equation
    regressors W_t = [weather_t, seasonality_t] (the intercept and s_{t-1} are
    added inside the sampler).

    Returns y, y_lag, W, dates, state_feature_names
    """
    data = df[[spread_col] + list(weather_cols)].dropna()
    y_full = data[spread_col].to_numpy(dtype=float)
    dates = data.index[1:]

    state = pd.concat([data[weather_cols].iloc[1:],
                       make_seasonal_features(dates, n_harmonics)], axis=1)
    std = state.std().replace(0, 1.0)
    state = (state - state.mean()) / std

    return y_full[1:], y_full[:-1], state.to_numpy(dtype=float), dates, list(state.columns)


#####################
# 2. Numerical helpers
#####################

def _sym(A):
    return 0.5 * (A + A.T)


def _draw_ordered_pair(mean, cov, rng):
    """Draw (mu0, mu1) ~ N(mean, cov) subject to mu0 < mu1 (via a = mu0, d = mu1 - mu0 > 0)."""
    A = np.array([[1.0, 0.0], [-1.0, 1.0]])
    m, V = A @ mean, _sym(A @ cov @ A.T)
    sd_d = np.sqrt(max(V[1, 1], 1e-12))
    d = truncnorm.rvs(-m[1] / sd_d, np.inf, loc=m[1], scale=sd_d, random_state=rng)
    mean_a = m[0] + V[0, 1] / V[1, 1] * (d - m[1])
    var_a = max(V[0, 0] - V[0, 1] ** 2 / V[1, 1], 1e-12)
    a = rng.normal(mean_a, np.sqrt(var_a))
    return np.array([a, a + d])


def _draw_mvn_last_positive(mean, cov, rng):
    """Draw from N(mean, cov) subject to the last element being positive (delta > 0)."""
    mean, cov = np.asarray(mean), _sym(np.asarray(cov))
    m_d, v_d = mean[-1], cov[-1, -1]
    sd_d = np.sqrt(max(v_d, 1e-12))
    d = truncnorm.rvs(-m_d / sd_d, np.inf, loc=m_d, scale=sd_d, random_state=rng)
    if len(mean) == 1:
        return np.array([d])
    V_od = cov[:-1, -1]
    cond_mean = mean[:-1] + V_od / v_d * (d - m_d)
    cond_cov = _sym(cov[:-1, :-1] - np.outer(V_od, V_od) / v_d)
    return np.r_[rng.multivariate_normal(cond_mean, cond_cov), d]


def hpdi(chain, prob=0.95):
    """Highest posterior density interval of a chain."""
    chain = np.sort(np.asarray(chain))
    n = len(chain)
    m = int(np.floor(prob * n))
    if m < 1:
        return np.nan, np.nan
    i = np.argmin(chain[m:] - chain[:n - m])
    return chain[i], chain[i + m]


def geweke_test(chain, frac1=0.1, frac2=0.5):
    """Geweke-style z-test comparing the means of the first 10% and last 50% of a chain."""
    chain = np.asarray(chain)
    n = len(chain)
    first, last = chain[:int(frac1 * n)], chain[int((1.0 - frac2) * n):]
    se = np.sqrt(first.var() / len(first) + last.var() / len(last))
    z = (first.mean() - last.mean()) / se if se > 0 else np.nan
    return z, 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan


#####################
# 3. Hidden Markov part: FFBS
#####################

def _log_emission(y, y_lag, mu, phi, sigma2):
    """log p(y_t | s_t = j), shape (T, 2)."""
    const = -0.5 * np.log(2.0 * np.pi * sigma2)
    return np.column_stack([const - 0.5 * (y - (1.0 - phi) * mu[j] - phi * y_lag) ** 2 / sigma2
                            for j in (0, 1)])


def _transition_probs(W, gamma):
    """P(s_t = 1 | s_{t-1} = i, W_t) for i = 0, 1; shape (T, 2)."""
    base = gamma[0] + W @ gamma[1:-1]
    return np.clip(np.column_stack([norm.cdf(base), norm.cdf(base + gamma[-1])]),
                   1e-12, 1 - 1e-12)


def _logaddexp(a, b):
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def ffbs_state_draw(y, y_lag, W, mu, phi, sigma2, gamma, rng, initial_state=0):
    """
    Draw the full state path s_{1:T} by forward filtering, backward sampling.

    Returns (s, filtered P(s_t = 1 | y_{1:t})).
    """
    T = len(y)
    le = _log_emission(y, y_lag, mu, phi, sigma2).tolist()
    p1 = _transition_probs(W, gamma)
    # lt[t][i][j] = log P(s_t = j | s_{t-1} = i)
    lt = np.stack([np.log1p(-p1), np.log(p1)], axis=2).tolist()

    # Forward filter
    log_alpha = np.empty((T, 2))
    a0 = le[0][0] + lt[0][initial_state][0]
    a1 = le[0][1] + lt[0][initial_state][1]
    c = _logaddexp(a0, a1)
    a0, a1 = a0 - c, a1 - c
    log_alpha[0] = a0, a1
    for t in range(1, T):
        n0 = le[t][0] + _logaddexp(a0 + lt[t][0][0], a1 + lt[t][1][0])
        n1 = le[t][1] + _logaddexp(a0 + lt[t][0][1], a1 + lt[t][1][1])
        c = _logaddexp(n0, n1)
        a0, a1 = n0 - c, n1 - c
        log_alpha[t] = a0, a1

    # Backward sampling
    la = log_alpha.tolist()
    u = rng.random(T)
    s = np.empty(T, dtype=int)
    s[-1] = int(u[-1] < math.exp(la[-1][1]))
    for t in range(T - 2, -1, -1):
        nxt = s[t + 1]
        l0 = la[t][0] + lt[t + 1][0][nxt]
        l1 = la[t][1] + lt[t + 1][1][nxt]
        s[t] = int(u[t] < 1.0 / (1.0 + math.exp(min(l0 - l1, 700.0))))

    return s, np.exp(log_alpha[:, 1])


#####################
# 4. Conditional posteriors
#####################

def _design(W, s, initial_state=0):
    """Probit design matrix [1, W_t, s_{t-1}]."""
    return np.column_stack([np.ones(len(s)), W, np.r_[initial_state, s[:-1]]])


def sample_z(W, s, gamma, rng, initial_state=0):
    """Latent utilities: z_t > 0 if s_t = 1, z_t <= 0 if s_t = 0."""
    mean = _design(W, s, initial_state) @ gamma
    a = np.where(s == 1, -mean, -np.inf)
    b = np.where(s == 1, np.inf, -mean)
    return truncnorm.rvs(a, b, loc=mean, scale=1.0, random_state=rng)


def sample_gamma(W, s, z, g0, G0, rng, initial_state=0, delta_positive=True):
    """Probit coefficients, prior gamma ~ N(g0, G0); optionally delta > 0."""
    Q = _design(W, s, initial_state)
    G0_inv = np.linalg.inv(G0)
    G1 = np.linalg.inv(G0_inv + Q.T @ Q)
    g1 = G1 @ (G0_inv @ g0 + Q.T @ z)
    return _draw_mvn_last_positive(g1, G1, rng) if delta_positive \
        else rng.multivariate_normal(g1, _sym(G1))


def sample_mu(y, y_lag, s, phi, sigma2, m_mu0, V_mu0, rng):
    """Ordered regime means from  y_t - phi y_{t-1} = (1 - phi) mu_{s_t} + e_t."""
    X = (1.0 - phi) * np.column_stack([1 - s, s])
    V0_inv = np.linalg.inv(V_mu0)
    V1 = np.linalg.inv(V0_inv + X.T @ X / sigma2)
    m1 = V1 @ (V0_inv @ m_mu0 + X.T @ (y - phi * y_lag) / sigma2)
    return _draw_ordered_pair(m1, V1, rng)


def sample_phi(y, y_lag, s, mu, sigma2, m_phi0, V_phi0, rng):
    """AR coefficient from  y_t - mu_{s_t} = phi (y_{t-1} - mu_{s_t}) + e_t, truncated to (-1, 1)."""
    r, x = y - mu[s], y_lag - mu[s]
    V1 = 1.0 / (1.0 / V_phi0 + x @ x / sigma2)
    m1 = V1 * (m_phi0 / V_phi0 + x @ r / sigma2)
    sd = np.sqrt(V1)
    return truncnorm.rvs((-1.0 - m1) / sd, (1.0 - m1) / sd, loc=m1, scale=sd, random_state=rng)


def sample_sigma2(y, y_lag, s, mu, phi, nu0, d0, rng):
    """sigma2 ~ IG-2(nu0 + T, d0 + SSR)."""
    resid = y - (1.0 - phi) * mu[s] - phi * y_lag
    return (d0 + resid @ resid) / rng.chisquare(nu0 + len(y))


#####################
# 5. Gibbs sampler
#####################

def _initial_values(y, y_lag, W):
    s = (y > np.median(y)).astype(int)
    mu = np.sort([y[s == 0].mean(), y[s == 1].mean()])
    x, r = y_lag - y_lag.mean(), y - y.mean()
    phi = float(np.clip((x @ r) / (x @ x), -0.95, 0.95))
    sigma2 = max(np.var(y - (1.0 - phi) * mu[s] - phi * y_lag), 1e-6)
    gamma = np.zeros(W.shape[1] + 2)
    gamma[-1] = 1.0                                  # start with positive persistence
    return s, mu, phi, sigma2, gamma


def gibbs_regime_switching(y, y_lag, W, nos=5000, nob=5000, nod=1,
                           m_mu0=None, V_mu0=None, m_phi0=0.8, V_phi0=0.2 ** 2,
                           g0=None, G0=None, nu0=5.0, d0=1.0,
                           delta_positive=True, initial_state=0, seed=42, verbose=True):
    """
    Gibbs sampler for the persistent regime-switching model.

    nos : draws kept, nob : burn-in, nod : thinning.
    Default priors: mu ~ N([q25, q75], 10 var(y) I), phi ~ N(0.8, 0.2^2),
    gamma ~ N(0, I) (with the intercept and delta at variance 4).

    Returns a dict with draws of mu (nos x 2), phi, sigma2, gamma
    (nos x k, last column = delta), and the posterior mean state probability.
    """
    rng = np.random.default_rng(seed)
    T, k = len(y), W.shape[1] + 2
    if m_mu0 is None:
        m_mu0 = np.percentile(y, [25, 75])
    if V_mu0 is None:
        V_mu0 = np.eye(2) * 10.0 * np.var(y)
    if g0 is None:
        g0 = np.zeros(k)
    if G0 is None:
        G0 = np.eye(k)
        G0[0, 0] = G0[-1, -1] = 4.0

    s, mu, phi, sigma2, gamma = _initial_values(y, y_lag, W)
    total = nob + nos * nod
    draws = {'mu': np.zeros((nos, 2)), 'phi': np.zeros(nos), 'sigma2': np.zeros(nos),
             'gamma': np.zeros((nos, k))}
    state_prob, filtered_prob, keep = np.zeros(T), np.zeros(T), 0

    for it in range(total):
        s, filt = ffbs_state_draw(y, y_lag, W, mu, phi, sigma2, gamma, rng, initial_state)
        z = sample_z(W, s, gamma, rng, initial_state)
        gamma = sample_gamma(W, s, z, g0, G0, rng, initial_state, delta_positive)
        mu = sample_mu(y, y_lag, s, phi, sigma2, m_mu0, V_mu0, rng)
        phi = sample_phi(y, y_lag, s, mu, sigma2, m_phi0, V_phi0, rng)
        sigma2 = sample_sigma2(y, y_lag, s, mu, phi, nu0, d0, rng)

        if it >= nob and (it - nob) % nod == 0:
            draws['mu'][keep], draws['phi'][keep] = mu, phi
            draws['sigma2'][keep], draws['gamma'][keep] = sigma2, gamma
            state_prob += s
            filtered_prob += filt
            keep += 1
        if verbose and (it + 1) % 2000 == 0:
            print(f'  iteration {it + 1:,} / {total:,}')

    draws['state_prob'] = state_prob / nos
    draws['filtered_prob'] = filtered_prob / nos
    return draws


def posterior_summary(draws, state_feature_names):
    """Mean, std, median and 95% HPD interval of every parameter."""
    chains = {'mu0': draws['mu'][:, 0], 'mu1': draws['mu'][:, 1],
              'phi': draws['phi'], 'sigma2': draws['sigma2']}
    names = ['alpha0'] + [f'alpha: {c}' for c in state_feature_names] + ['delta']
    chains.update({name: draws['gamma'][:, j] for j, name in enumerate(names)})

    rows = []
    for name, chain in chains.items():
        lo, hi = hpdi(chain)
        z, p = geweke_test(chain)
        rows.append({'parameter': name, 'mean': chain.mean(), 'std': chain.std(),
                     'hpd 2.5%': lo, 'hpd 97.5%': hi, 'geweke z': z, 'geweke p': p})
    return pd.DataFrame(rows).set_index('parameter')


#####################
# 6. Identification check on simulated data
#####################

def simulate_regime_switching(T, phi, mu, sigma2, p_stay, rng):
    """Simulate the observation equation with a symmetric 2-state Markov chain."""
    s = np.empty(T, dtype=int)
    s[0] = rng.integers(2)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() < p_stay else 1 - s[t - 1]
    y = np.empty(T)
    y[0] = mu[s[0]] + rng.normal(0, np.sqrt(sigma2 / (1 - phi ** 2)))
    for t in range(1, T):
        y[t] = (1 - phi) * mu[s[t]] + phi * y[t - 1] + rng.normal(0, np.sqrt(sigma2))
    return y, s


def fixed_state_sampler(y, y_lag, s, n_iter, m_mu0, V_mu0, m_phi0, V_phi0, nu0, d0, rng):
    """Sample (mu, phi, sigma2) with the TRUE state path held fixed."""
    mu = np.sort([y[s == 0].mean(), y[s == 1].mean()])
    phi, sigma2 = 0.5, max(np.var(y) * 0.5, 1e-4)
    out = {'mu': np.zeros((n_iter, 2)), 'phi': np.zeros(n_iter), 'sigma2': np.zeros(n_iter)}
    for i in range(n_iter):
        mu = sample_mu(y, y_lag, s, phi, sigma2, m_mu0, V_mu0, rng)
        phi = sample_phi(y, y_lag, s, mu, sigma2, m_phi0, V_phi0, rng)
        sigma2 = sample_sigma2(y, y_lag, s, mu, phi, nu0, d0, rng)
        out['mu'][i], out['phi'][i], out['sigma2'][i] = mu, phi, sigma2
    return out
