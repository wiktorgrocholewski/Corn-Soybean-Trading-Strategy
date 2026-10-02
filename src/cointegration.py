# Stationarity / cointegration tests and hedge-ratio estimators.
#
# Tests:      adf_table, engle_granger, johansen_test, half_life, ecm
# Estimators: rolling_ols        — rolling OLS hedge ratio (strictly past window)
#             rolling_johansen   — first Johansen eigenvector on a rolling window
#             kalman_regression  — random-walk-coefficient Kalman filter
#                                  (Chan 2013, Ch. 3; the 2-state pair filter is
#                                   the special case X = [x, 1])
# The 3-state Kalman filter with an AR(1) spread state, used by the ML
# pipeline, lives in features.py.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


#####################
# 1. Tests
#####################

def adf_table(prices):
    """ADF test on the levels and first differences of every column."""
    rows = {}
    for col in prices.columns:
        s = prices[col].dropna()
        lvl, dif = adfuller(s), adfuller(s.diff().dropna())
        rows[col] = {'ADF (level)': lvl[0], 'p (level)': lvl[1],
                     'ADF (diff)': dif[0], 'p (diff)': dif[1]}
    return pd.DataFrame(rows).T


def engle_granger(y, x):
    """
    Engle-Granger / CADF test of y = alpha + beta * x + e.

    Uses statsmodels.coint, i.e. MacKinnon critical values for a residual-based
    test (plain ADF p-values on an estimated residual are too optimistic).
    """
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    stat, pvalue, _ = coint(y, x)
    return {'dependent': y.name, 'beta': ols.params.iloc[1], 'alpha': ols.params.iloc[0],
            't-stat': stat, 'p-value': pvalue}


def johansen_test(prices, det_order=0, k_ar_diff=1):
    """
    Johansen trace and max-eigenvalue tests (Chan 2013, Example 2.7).

    Returns
    -------
    table : pd.DataFrame — one row per null hypothesis r <= 0, 1, ...
    evec  : np.ndarray   — eigenvectors (columns), first column = most
                           mean-reverting combination
    """
    res = coint_johansen(prices, det_order=det_order, k_ar_diff=k_ar_diff)
    table = pd.DataFrame({
        'trace': res.lr1, 'trace 95%': res.cvt[:, 1], 'trace 99%': res.cvt[:, 2],
        'max-eig': res.lr2, 'max-eig 95%': res.cvm[:, 1],
    }, index=[f'r<={r}' for r in range(prices.shape[1])])
    return table, res.evec


def half_life(spread):
    """Half-life of mean reversion from  d s_t = c + lambda * s_{t-1} + e_t."""
    s = pd.Series(spread).dropna()
    fit = sm.OLS(s.diff().iloc[1:].values, sm.add_constant(s.shift(1).iloc[1:].values)).fit()
    lam = fit.params[1]
    return -np.log(2) / lam if lam < 0 else np.inf


def ecm(y, x, spread):
    """
    Error-correction model  dy_t = c + gamma * spread_{t-1} + delta * dx_t + e_t.
    gamma < 0 means y is pulled back towards the long-run relation.
    """
    data = pd.concat({'dy': y.diff(), 'dx': x.diff(), 'spread_lag': spread.shift(1)},
                     axis=1).dropna()
    fit = sm.OLS(data['dy'], sm.add_constant(data[['spread_lag', 'dx']])).fit()
    gamma = fit.params['spread_lag']
    return {'gamma': gamma, 'p-value': fit.pvalues['spread_lag'],
            'delta': fit.params['dx'],
            'half-life': -np.log(2) / gamma if gamma < 0 else np.inf}


#####################
# 2. Hedge-ratio estimators
#####################

def rolling_ols(y, x, window=252):
    """
    Rolling OLS  y = alpha + beta * x  with coefficients for day t estimated
    on [t - window, t) — strictly past data, so no look-ahead.

    Returns a DataFrame with columns 'alpha' and 'beta' (NaN for the first
    `window` rows).
    """
    yv, xv = np.asarray(y, float), np.asarray(x, float)
    n = len(yv)
    alpha, beta = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(window, n):
        X = np.column_stack([np.ones(window), xv[t - window:t]])
        alpha[t], beta[t] = np.linalg.lstsq(X, yv[t - window:t], rcond=None)[0]
    return pd.DataFrame({'alpha': alpha, 'beta': beta}, index=y.index)


def rolling_johansen(prices, window=252, normalize_on=None, det_order=0, k_ar_diff=1):
    """
    First Johansen eigenvector re-estimated every day on [t - window, t).

    normalize_on : column name. If given, the vector is scaled so that this
                   column has coefficient 1 (so the others are -hedge ratios).
                   If None, only the sign is fixed (first coefficient > 0), so
                   the spread does not flip sign when the solver does.

    Returns a DataFrame of weights with the same columns as `prices`.
    """
    values = prices.values
    n, k = values.shape
    weights = np.full((n, k), np.nan)
    j = None if normalize_on is None else list(prices.columns).index(normalize_on)
    for t in range(window, n):
        try:
            vec = coint_johansen(values[t - window:t], det_order, k_ar_diff).evec[:, 0]
        except np.linalg.LinAlgError:
            continue
        weights[t] = vec / vec[j] if j is not None else vec * np.sign(vec[0])
    return pd.DataFrame(weights, index=prices.index, columns=prices.columns)


def kalman_regression(y, X, delta=1e-5, obs_cov=1e-3, state_init=None, cov_init=0.1):
    """
    Kalman filter for a regression with random-walk coefficients:
        y_t = X_t' b_t + e_t,        e_t ~ N(0, obs_cov)
        b_t = b_{t-1} + w_t,         w_t ~ N(0, delta / (1 - delta) * I)

    Chan (2013), Ch. 3. Include a column of ones in X for a time-varying
    intercept. Returns the filtered coefficients b_{t|t} as a DataFrame with
    the columns of X (b_{t|t} uses y_t, so lag it by one day before trading
    on it).
    """
    yv, Xv = np.asarray(y, float), np.asarray(X, float)
    n, k = Xv.shape
    Q = delta / (1 - delta) * np.eye(k)
    b = np.zeros(k) if state_init is None else np.asarray(state_init, float)
    P = cov_init * np.eye(k)

    out = np.full((n, k), np.nan)
    for t in range(n):
        if t > 0:
            P = P + Q                              # predict (identity transition)
        h = Xv[t]
        S = h @ P @ h + obs_cov                    # innovation variance
        K = P @ h / S                              # Kalman gain
        b = b + K * (yv[t] - h @ b)                # update
        P = P - np.outer(K, h @ P)
        out[t] = b
    return pd.DataFrame(out, index=y.index, columns=getattr(X, 'columns', None))
