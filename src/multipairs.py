# Multi-pair diversification framework (Lee, Leung & Ning, 2023).
#
# Six pairs from four commodity ETFs, each traded with a z-score rule on its
# rolling-OLS spread. Capital is re-allocated across pairs every `rebal_freq`
# days by
#   EW  — equal weight,
#   MRB — Mean Reversion Budgeting: weight ~ minmax(mu/sigma) * minmax(loglik)
#         of an Ornstein-Uhlenbeck fit to each spread,
#   MRR — Mean Reversion Ranking: fixed weights by rank of the same score.
# P&L uses the frozen-beta, capital-normalised convention of src/backtest.py.

import numpy as np
import pandas as pd

from .backtest import performance
from .cointegration import rolling_ols


# =============================================================================
# 1. SPREAD COMPUTATION
# =============================================================================

def compute_all_spreads(prices, pairs, window=252):
    """
    Rolling-OLS spread  s1 - alpha - beta * s2  for every pair, with alpha and
    beta estimated on [t - window, t) (strictly past data).

    Returns
    -------
    spreads, betas, alphas : dicts {(s1, s2): pd.Series}
    """
    spreads, betas, alphas = {}, {}, {}
    for s1, s2 in pairs:
        coef = rolling_ols(prices[s1], prices[s2], window)
        spreads[(s1, s2)] = prices[s1] - coef['alpha'] - coef['beta'] * prices[s2]
        betas[(s1, s2)] = coef['beta']
        alphas[(s1, s2)] = coef['alpha']
    return spreads, betas, alphas


# =============================================================================
# 2. OU PARAMETER ESTIMATION
# =============================================================================

def estimate_ou_params(spread):
    """
    MLE of OU parameters and average log-likelihood for a spread window.
    Closed-form estimators of Lee, Leung & Ning (2023).

    Returns (mu, theta, sigma, ll); returns NaNs when the pair shows no
    mean reversion (mu_hat <= 0) so it can be assigned zero weight.
    """
    x  = np.asarray(spread, dtype=float)
    n  = len(x) - 1
    dt = 1  # daily

    Xx  = np.sum(x[:-1])
    Xy  = np.sum(x[1:])
    Xxx = np.sum(x[:-1] ** 2)
    Xxy = np.sum(x[:-1] * x[1:])
    Xyy = np.sum(x[1:] ** 2)

    denom = n * (Xxx - Xxy) - (Xx ** 2 - Xx * Xy)
    if abs(denom) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan

    theta = (Xy * Xxx - Xx * Xxy) / denom

    num_mu = Xxy - theta * Xx - theta * Xy + n * theta ** 2
    den_mu = Xxx - 2 * theta * Xx + n * theta ** 2
    if den_mu <= 0 or num_mu / den_mu <= 0:
        return np.nan, np.nan, np.nan, np.nan

    mu = -np.log(num_mu / den_mu) / dt
    if mu <= 0:
        return np.nan, np.nan, np.nan, np.nan

    e1 = np.exp(-mu * dt)
    e2 = np.exp(-2 * mu * dt)
    sigma2 = (2 * mu / (n * (1 - e2))) * (
        Xyy
        - 2 * e1 * Xxy
        + e2 * Xxx
        - 2 * theta * (1 - e1) * (Xy - e1 * Xx)
        + n * theta ** 2 * (1 - e1) ** 2
    )
    sigma = np.sqrt(max(sigma2, 1e-10))

    sigma_tilde = np.sqrt(sigma ** 2 * (1 - e2) / (2 * mu))
    ll = (
        -0.5 * np.log(2 * np.pi)
        - np.log(sigma_tilde)
        - (1 / (2 * n * sigma_tilde ** 2))
        * np.sum((x[1:] - x[:-1] * e1 - theta * (1 - e1)) ** 2)
    )

    return mu, theta, sigma, ll


# =============================================================================
# 3. PORTFOLIO WEIGHT COMPUTATION
# =============================================================================

def _minmax(arr):
    """Min-max to [0,1] over finite entries; non-finite -> 0."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0 or valid.max() == valid.min():
        return np.zeros_like(arr)
    norm = (arr - valid.min()) / (valid.max() - valid.min())
    return np.where(np.isfinite(norm), norm, 0.0)


def compute_mrb_weights(ou_params):
    """
    Mean Reversion Budgeting (MRB), Eq. (6.11).
        omega_i ∝ minmax(mu_i/sigma_i) * minmax(ll_i)
    Pairs with NaN OU params (no mean reversion) receive zero weight.
    """
    pairs  = list(ou_params.keys())
    mus    = np.array([ou_params[p][0] for p in pairs], dtype=float)
    sigmas = np.array([ou_params[p][2] for p in pairs], dtype=float)
    lls    = np.array([ou_params[p][3] for p in pairs], dtype=float)

    mu_r = np.where(sigmas > 0, mus / sigmas, np.nan)

    raw   = _minmax(mu_r) * _minmax(lls)
    total = raw.sum()

    if total <= 0:
        w = np.ones(len(pairs)) / len(pairs)   # fallback: equal weight
    else:
        w = raw / total

    return {p: float(w[i]) for i, p in enumerate(pairs)}


def compute_mrr_weights(ou_params):
    """
    Mean Reversion Ranking (MRR), Eq. (6.12).
        w_(k) = (N-1+2k) / (2N(N-1)),  k = 0..N-1   (0 = worst)

    Pairs with no mean reversion
    (mu_hat <= 0 / NaN OU fit) are EXCLUDED from the ranking and assigned
    zero weight — consistent with MRB and with the paper's stated rule that
    such pairs get zero weight. The rank formula is then applied only to the
    N' valid pairs (so the surviving weights still sum to 1).
    """
    pairs  = list(ou_params.keys())
    mus    = np.array([ou_params[p][0] for p in pairs], dtype=float)
    sigmas = np.array([ou_params[p][2] for p in pairs], dtype=float)
    lls    = np.array([ou_params[p][3] for p in pairs], dtype=float)

    valid = (np.isfinite(mus) & (mus > 0)
             & np.isfinite(sigmas) & (sigmas > 0)
             & np.isfinite(lls))
    w = np.zeros(len(pairs))
    valid_idx = np.where(valid)[0]
    N = len(valid_idx)

    if N == 0:
        # no mean-reverting pair this period -> fall back to equal weight
        return {p: 1.0 / len(pairs) for p in pairs}
    if N == 1:
        w[valid_idx[0]] = 1.0
        return {p: float(w[i]) for i, p in enumerate(pairs)}

    mu_r  = mus[valid] / sigmas[valid]
    score = _minmax(mu_r) * _minmax(lls[valid])

    # ascending ranks among valid pairs: 0 = worst, N-1 = best
    ranks = np.argsort(np.argsort(score))
    rw = np.array([(N - 1 + 2 * k) / (2 * N * (N - 1)) for k in ranks], dtype=float)
    rw /= rw.sum()

    for j, idx in enumerate(valid_idx):
        w[idx] = rw[j]

    return {p: float(w[i]) for i, p in enumerate(pairs)}


# =============================================================================
# 4. TRADING SIGNAL  (z-score on the rolling spread; used for SIGNAL only)
# =============================================================================

def compute_trading_signal(spread_window, K=1.0):
    """
    Entry signal from the rolling mean/SD of the spread.
    `spread_window` is length M+1: x[:-1] are the past M days, x[-1] is today.
    Returns +1 (long spread), -1 (short spread), or 0.
    """
    x   = np.asarray(spread_window, dtype=float)
    mu  = np.mean(x[:-1])
    sd  = np.std(x[:-1])
    cur = x[-1]

    if sd == 0:
        return 0
    if cur < mu - K * sd:
        return 1
    elif cur > mu + K * sd:
        return -1
    return 0


# =============================================================================
# 5. BACKTEST ENGINE  (capital-normalised, frozen-beta P&L)
# =============================================================================

def run_backtest(prices, spreads, betas, pairs,
                 K=1.0, M=63,
                 rebal_freq=63, ou_window=252,
                 method='MRR', initial_capital=1.0):
    """
    Backtest of the multi-pair diversification framework, returning a
    CAPITAL-NORMALISED daily return series so that the resulting Sharpe is
    directly comparable to the other strategies of the project.

    Conventions
    -----------
    * SIGNAL: z-score of the rolling-OLS spread vs its trailing M-day mean/SD.
    * P&L (frozen beta): on entry at t0 the hedge ratio beta_{t0} and the
      entry notional N_{t0} = |S1_{t0}| + |beta_{t0}|*|S2_{t0}| are frozen.
      While the position is held,
          pnl_t   = pos * (dS1_t - beta_{t0} * dS2_t)
          ret_t   = pnl_t / N_{t0}                       (per-unit-capital)
      This removes the P&L artefacts of a drifting rolling hedge ratio.
    * PORTFOLIO return: ret_t = sum_i w_i * ret_{i,t}, with w_i the EW/MRB/MRR
      weight in force during the holding period (weights sum to 1; a flat pair
      contributes 0 — its capital share earns the risk-free rate, taken as 0).
    * Positions are liquidated at each rebalance AFTER that day's P&L is booked.

    Parameters
    ----------
    prices  : pd.DataFrame  — ETF close prices (columns = tickers)
    spreads : dict          — {(s1,s2): pd.Series} from compute_all_spreads
    betas   : dict          — {(s1,s2): pd.Series} rolling hedge ratios
    pairs   : list          — list of (s1, s2) tuples
    K, M    : float, int    — entry threshold; rolling signal window (days)
    rebal_freq : int        — capital-rebalance frequency in trading days
    ou_window  : int        — OU estimation lookback at each rebalance
    method  : str           — 'EW', 'MRB', or 'MRR'
    initial_capital : float — linear scale on returns (Sharpe is invariant)

    Returns
    -------
    portfolio_returns : pd.Series   — daily capital-normalised returns
    weights_df        : pd.DataFrame — weights at each rebalance date
    """
    spread_df = pd.DataFrame({p: spreads[p] for p in pairs})
    beta_df   = pd.DataFrame({p: betas[p]   for p in pairs})

    valid_start = spread_df.dropna().index[0]
    spread_df = spread_df.loc[valid_start:]
    beta_df   = beta_df.loc[valid_start:]
    px        = prices.loc[valid_start:]
    dates     = spread_df.index
    T         = len(dates)

    P = len(pairs)

    # numpy views for the hot loop (identical logic, ~8x faster than .iloc)
    S  = spread_df.values                                    # [T, P] spreads
    B  = beta_df.values                                      # [T, P] hedge ratios
    S1 = np.column_stack([px[p[0]].values for p in pairs])   # [T, P] leg-1 prices
    S2 = np.column_stack([px[p[1]].values for p in pairs])   # [T, P] leg-2 prices

    rebal_set = set(range(ou_window, T, rebal_freq))

    ret_arr = np.zeros(T)
    weights_history = []
    cur_w  = np.full(P, 1.0 / P)
    pos    = np.zeros(P, dtype=int)
    fbeta  = np.full(P, np.nan)
    notion = np.full(P, np.nan)

    for t in range(M, T):
        # ---- 1. Book P&L for positions held INTO today (old weights in force)
        dr = 0.0
        for j in range(P):
            if pos[j] != 0 and np.isfinite(notion[j]) and notion[j] > 0:
                dS1 = S1[t, j] - S1[t - 1, j]
                dS2 = S2[t, j] - S2[t - 1, j]
                if np.isfinite(dS1) and np.isfinite(dS2):
                    dr += cur_w[j] * (pos[j] * (dS1 - fbeta[j] * dS2) / notion[j])
        ret_arr[t] = initial_capital * dr

        # ---- 2. Rebalance (new weights + liquidate) AFTER booking P&L
        if t in rebal_set:
            ou_params = {}
            for j, p in enumerate(pairs):
                w_spread = S[t - ou_window:t, j]
                w_spread = w_spread[np.isfinite(w_spread)]
                if len(w_spread) < 30:
                    ou_params[p] = (np.nan, np.nan, np.nan, np.nan)
                else:
                    ou_params[p] = estimate_ou_params(w_spread)

            if method == 'MRB':
                wd = compute_mrb_weights(ou_params)
            elif method == 'MRR':
                wd = compute_mrr_weights(ou_params)
            else:
                wd = {p: 1.0 / P for p in pairs}
            cur_w = np.array([wd[p] for p in pairs])
            weights_history.append({'date': dates[t],
                                    **{str(p): wd[p] for p in pairs}})
            pos[:] = 0; fbeta[:] = np.nan; notion[:] = np.nan

        # ---- 3. Signals / entries / exits for the END of today
        for j in range(P):
            win = S[t - M:t + 1, j]
            if np.any(np.isnan(win)):
                continue
            signal      = compute_trading_signal(win, K)
            mean_spread = win[:-1].mean()
            s_curr      = win[-1]
            if pos[j] == 0:
                beta_now = B[t, j]
                if signal != 0 and np.isfinite(beta_now):
                    notional = abs(S1[t, j]) + abs(beta_now) * abs(S2[t, j])
                    if notional > 0:
                        pos[j]    = int(signal)
                        fbeta[j]  = float(beta_now)    # frozen at entry
                        notion[j] = float(notional)    # frozen at entry
            elif pos[j] == 1 and s_curr > mean_spread:
                pos[j] = 0; fbeta[j] = np.nan; notion[j] = np.nan
            elif pos[j] == -1 and s_curr < mean_spread:
                pos[j] = 0; fbeta[j] = np.nan; notion[j] = np.nan

    portfolio_returns = pd.Series(ret_arr, index=dates)
    weights_df = (pd.DataFrame(weights_history).set_index('date')
                  if weights_history else pd.DataFrame())

    return portfolio_returns, weights_df


# =============================================================================
# 6. TRAIN / HOLDOUT EVALUATION
# =============================================================================

def _perf_on(returns, start=None, end=None, trim_warmup=True):
    """Performance on a date-sliced sub-series [start, end)."""
    r = returns
    if start is not None:
        r = r[r.index >= start]
    if end is not None:
        r = r[r.index < end]
    return performance(r, trim_warmup=trim_warmup)


def tune_on_dev(prices, pairs, holdout_start,
                K_grid=(1.0, 1.5, 2.0, 2.5),
                ols_grid=(126, 252, 504),
                rebal_grid=(21, 63),
                ou_grid=(63, 126, 252),
                M=63, tune_method='EW', verbose=True):
    """
    Joint grid search over (K, OLS window, rebal_freq, OU window) on the DEV
    set only (dates strictly before `holdout_start`), scored by the annualised
    Sharpe of the `tune_method` portfolio (default 'EW', allocation-neutral).

    Spreads/betas are recomputed once per OLS window. The holdout period is
    never read here, so the holdout evaluation stays out-of-sample.

    Dev Sharpe is measured over the ACTIVE dev window (warm-up trimmed) so the
    comparison is fair across OLS windows of different warm-up length.

    Returns
    -------
    best_params : dict {'K','ols_window','rebal_freq','ou_window'}
    table       : pd.DataFrame of every combination, sorted by dev Sharpe.
    """
    rows, best = [], None
    for ols in ols_grid:
        spreads, betas, _ = compute_all_spreads(prices, pairs, window=ols)
        for K in K_grid:
            for rb in rebal_grid:
                for ou in ou_grid:
                    returns, _ = run_backtest(prices, spreads, betas, pairs,
                                              K=K, M=M, rebal_freq=rb,
                                              ou_window=ou, method=tune_method)
                    s = _perf_on(returns, end=holdout_start, trim_warmup=True)['Sharpe']
                    row = {'K': K, 'ols_window': ols, 'rebal_freq': rb,
                           'ou_window': ou, 'dev_Sharpe': s}
                    rows.append(row)
                    if best is None or (np.isfinite(s) and s > best['dev_Sharpe']):
                        best = row
        if verbose:
            print(f'  grid done for OLS window={ols}')

    table = (pd.DataFrame(rows)
             .sort_values('dev_Sharpe', ascending=False, na_position='last')
             .reset_index(drop=True))
    best_params = {k: best[k] for k in ('K', 'ols_window', 'rebal_freq', 'ou_window')}
    if verbose:
        print(f"\nBest dev params (tuned on {tune_method}): {best_params}  "
              f"dev Sharpe = {best['dev_Sharpe']:.3f}")
    return best_params, table


def evaluate_holdout(prices, pairs, params, holdout_start, M=63):
    """
    With FROZEN `params` = {K, ols_window, rebal_freq, ou_window}, evaluate
    EW / MRB / MRR once. Dev stats use the active window (warm-up trimmed);
    holdout stats use the full holdout window. The strategy runs
    continuously, so a position opened in dev that is still open at
    `holdout_start` contributes its holdout-day P&L (trailing-window
    estimation => no look-ahead).

    Returns dict with a dev/holdout comparison `table` and per-method `series`.
    """
    spreads, betas, _ = compute_all_spreads(prices, pairs, window=params['ols_window'])
    series, dev_perf, hold_perf = {}, {}, {}
    for method in ['EW', 'MRB', 'MRR']:
        returns, _ = run_backtest(prices, spreads, betas, pairs,
                                  K=params['K'], M=M,
                                  rebal_freq=params['rebal_freq'],
                                  ou_window=params['ou_window'], method=method)
        series[method]    = returns
        dev_perf[method]  = _perf_on(returns, end=holdout_start, trim_warmup=True)
        hold_perf[method] = _perf_on(returns, start=holdout_start, trim_warmup=False)

    table = pd.DataFrame({
        m: {'train Sharpe':   dev_perf[m]['Sharpe'],
            'test Sharpe':    hold_perf[m]['Sharpe'],
            'Ann. return':    hold_perf[m]['Ann. return'],
            'Ann. vol':       hold_perf[m]['Ann. vol'],
            'Max drawdown':   hold_perf[m]['Max drawdown'],
            'Total return':   hold_perf[m]['Total return']}
        for m in ['EW', 'MRB', 'MRR']
    }).T
    return {'table': table, 'series': series, 'holdout_start': holdout_start,
            'dev_perf': dev_perf, 'holdout_perf': hold_perf}


def allocation_weights(spread_df, dates, method='EW', rebal_freq=63, ou_window=63):
    """
    Step-function capital weights over `dates`, re-estimated every
    `rebal_freq` days from OU fits to the trailing `ou_window` days of each
    spread (columns of spread_df). Used to combine per-pair return series
    produced by another signal (notebook 05).
    """
    cols = list(spread_df.columns)
    W = pd.DataFrame(index=dates, columns=cols, dtype=float)
    current = {c: 1.0 / len(cols) for c in cols}
    for i, date in enumerate(dates):
        if i % rebal_freq == 0 and method != 'EW':
            ou = {}
            for c in cols:
                window = spread_df[c].loc[:date].dropna().iloc[-ou_window:].values
                ou[c] = estimate_ou_params(window) if len(window) >= 30 else (np.nan,) * 4
            current = compute_mrb_weights(ou) if method == 'MRB' else compute_mrr_weights(ou)
        W.loc[date] = [current[c] for c in cols]
    return W
