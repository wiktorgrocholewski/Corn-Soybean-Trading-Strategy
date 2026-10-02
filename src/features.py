# Features for the ML pipeline (notebooks 04-05).
#
# The functions follow the naming of the corn-soybean pair: `corn_col` is the
# independent leg x and `soy_col` the dependent leg y in  y = alpha + gamma * x.
# For other pairs pass the independent leg as corn_col.

import numpy as np
import pandas as pd

from .cointegration import rolling_ols


#####################
# 1. Rolling OLS hedge ratio (used for labeling)
#####################

def compute_hedge_ratio(df, corn_col, soy_col, window=504, verbose=True):
    """
    Rolling OLS hedge ratio with intercept, estimated on the past `window`
    trading days (504 ~ 2 years):  soy = alpha + gamma * corn + epsilon.
    By Engle-Granger superconsistency OLS estimates gamma well for a
    cointegrated pair despite autocorrelated residuals.

    Adds columns 'hedge_ratio', 'intercept', 'spread' (= soy - alpha - gamma * corn).
    """
    coef = rolling_ols(df[soy_col], df[corn_col], window)
    df['hedge_ratio'] = coef['beta']
    df['intercept'] = coef['alpha']
    df['spread'] = df[soy_col] - df['intercept'] - df['hedge_ratio'] * df[corn_col]

    if verbose:
        print(f"Rolling OLS ({window}d): valid rows {coef['beta'].notna().sum()} / {len(df)}, "
              f"hedge ratio {coef['beta'].min():.3f} to {coef['beta'].max():.3f}")
    return df


#####################
# 2. Three-state Kalman filter for hedge ratio, spread and z-scores
#####################

def _run_kalman_3state(corn, soy, delta, sigma2_s, R, phi):
    """One pass of the 3-state filter with fixed parameters (see compute_kalman_hedge)."""
    n = len(corn)
    F = np.diag([1.0, 1.0, phi])
    Q = np.diag([delta, delta, sigma2_s])
    x = np.array([0.0, 0.5, 0.0])
    P = np.diag([10.0, 10.0, 1.0])

    alphas, gammas, spreads = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    innovations, inn_variances = np.full(n, np.nan), np.full(n, np.nan)

    for t in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # Innovation
        H = np.array([1.0, corn[t], 1.0])
        v = soy[t] - H @ x_pred
        S = H @ P_pred @ H + R
        # Update
        K = P_pred @ H / S
        x = x_pred + K * v
        P = P_pred - np.outer(K, K) * S

        alphas[t], gammas[t], spreads[t] = x
        innovations[t], inn_variances[t] = v, S

    return alphas, gammas, spreads, innovations, inn_variances


def compute_kalman_hedge(df, corn_col, soy_col, delta=1e-5, R=1e-6,
                         burn=200, max_iter=10, tol=1e-5, verbose=True):
    """
    Hedge ratio, intercept, spread level and z-scores from a 3-state Kalman
    filter with a mean-reverting spread state.

    State-space model:
        State:       xi_t = [alpha_t, gamma_t, s_t]
        Transition:  xi_t = F xi_{t-1} + v_t,   F = diag(1, 1, phi),
                     v_t ~ N(0, diag(delta, delta, sigma2_s))
        Measurement: soy_t = [1, corn_t, 1] xi_t + w_t,   w_t ~ N(0, R)

    The standard 2-state filter (Chan 2013) treats the spread as iid noise, so a
    temporarily wide spread distorts gamma. Here the spread is its own AR(1)
    state. phi and sigma2_s are estimated iteratively from the filtered spread
    (after a burn-in), so no external half-life estimate is needed. delta is the
    only real hyperparameter: how fast the hedge ratio may drift (tune via CV).

    Adds columns:
        kf_hedge_ratio, kf_intercept — filtered gamma_t, alpha_t
        kf_spread     — filtered spread level s_t
        kf_innovation — prediction error v_t
        kf_z_score    — v_t / sqrt(S_t)  (surprise of today's observation)
        kf_level_z    — s_t / expanding std of s_t (distance from equilibrium)
    """
    corn, soy = df[corn_col].values, df[soy_col].values

    phi, sigma2_s = 0.992, 0.04
    for iteration in range(max_iter):
        _, _, spreads, _, _ = _run_kalman_3state(corn, soy, delta, sigma2_s, R, phi)
        s = spreads[burn:]
        phi_new = np.corrcoef(s[1:], s[:-1])[0, 1]
        sigma2_s_new = np.var(s[1:] - phi_new * s[:-1])
        converged = abs(phi_new - phi) < tol and abs(sigma2_s_new - sigma2_s) < tol
        phi, sigma2_s = phi_new, sigma2_s_new
        if converged:
            break

    alphas, gammas, spreads, innovations, inn_variances = \
        _run_kalman_3state(corn, soy, delta, sigma2_s, R, phi)

    df['kf_hedge_ratio'] = gammas
    df['kf_intercept'] = alphas
    df['kf_spread'] = spreads
    df['kf_innovation'] = innovations
    df['kf_z_score'] = innovations / np.sqrt(inn_variances)
    expanding_std = pd.Series(spreads, index=df.index).expanding(min_periods=50).std()
    df['kf_level_z'] = spreads / expanding_std.values

    if verbose:
        half_life = np.log(2) / -np.log(phi) if phi < 1 else np.inf
        print(f"Kalman (delta={delta:.0e}): converged in {iteration + 1} iterations, "
              f"phi={phi:.5f} (half-life {half_life:.0f} days), "
              f"gamma {np.nanmin(gammas[burn:]):.3f} to {np.nanmax(gammas[burn:]):.3f}, "
              f"innovation-z std {np.nanstd(df['kf_z_score'].values[burn:]):.3f} (ideal 1)")
    return df


#####################
# 3. Calendar features
#####################

def compute_calendar_features(df):
    """Month (1-12) and day of week (0-4). Feature importance later shows month overfits."""
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    return df


#####################
# 4. Spread volatility
#####################

def compute_spread_vol(df, corn_col, soy_col, hedge_ratio_col, span=100):
    """
    EWMA volatility of the tradeable spread P&L  d soy_t - gamma_{t-1} * d corn_t.
    Same quantity as labels.get_daily_vol, stored as a feature.
    """
    pnl = df[soy_col].diff() - df[hedge_ratio_col].shift(1) * df[corn_col].diff()
    df['spread_vol'] = pnl.ewm(span=span).std()
    return df
