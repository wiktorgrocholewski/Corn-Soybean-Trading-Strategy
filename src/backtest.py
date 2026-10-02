# Signals, backtests and performance metrics.
#
# Conventions shared by every strategy in the project:
#   * positions[t] is the position held from the close of day t to the close
#     of day t+1, decided with information available at the close of day t.
#   * The hedge (units of each asset per unit of spread) is FROZEN when a
#     trade is opened and held until it is closed or flipped. Letting a
#     rolling hedge ratio drift inside a trade creates P&L that could not be
#     earned without daily re-hedging (AFML Sec. 2.4.1).
#   * Returns are P&L divided by the gross capital of the position at entry,
#     sum_i |units_i| * price_i, so Sharpe ratios are comparable across
#     strategies. The risk-free rate is taken as 0.

import numpy as np
import pandas as pd

from .betsize import bet_size_from_prob, select_entry_dates


#####################
# 1. Z-score signals (classical mean-reversion rules)
#####################

def rolling_zscore(spread, window):
    """z_t = (s_t - mean(s_{t-w..t-1})) / std(s_{t-w..t-1}); the window excludes today."""
    past = spread.shift(1).rolling(window)
    z = (spread - past.mean()) / past.std()
    return z.replace([np.inf, -np.inf], np.nan)


def zscore_positions(z, entry, exit=0.0):
    """
    Mean-reversion position from a z-score:
      flat  -> long  (+1) if z <= -entry,  short (-1) if z >= +entry
      long  -> flat  once z >= -exit;       short -> flat once z <= +exit
      a crossing of the opposite entry band flips the position directly.
    NaN z-scores force a flat position.
    """
    pos, out = 0, np.zeros(len(z))
    for i, zi in enumerate(np.asarray(z, float)):
        if np.isnan(zi):
            pos = 0
        elif zi >= entry:
            pos = -1
        elif zi <= -entry:
            pos = 1
        elif (pos == 1 and zi >= -exit) or (pos == -1 and zi <= exit):
            pos = 0
        out[i] = pos
    return pd.Series(out, index=z.index)


def log_to_units(weights, prices):
    """
    Convert log-price weights to units of each asset.

    A spread  s = sum_i w_i log P_i  moves by  sum_i w_i dP_i / P_i, i.e. like
    a portfolio holding w_i DOLLARS of asset i, which is w_i / P_i units.
    """
    return weights / prices


#####################
# 2. Frozen-hedge spread backtest (classical strategies and ML multi-pair)
#####################

def spread_returns(prices, units, positions):
    """
    Daily returns of trading a spread with the hedge frozen at entry.

    Parameters
    ----------
    prices    : pd.DataFrame (T x k) — asset prices
    units     : pd.DataFrame (T x k) — hedge in units of each asset per unit
                of spread, as known at the close of each day
    positions : pd.Series (T)        — target position (+1 / 0 / -1)

    Returns
    -------
    pd.Series of daily returns (P&L / gross capital committed at entry).
    """
    P = prices.values.astype(float)
    U = units.reindex(index=prices.index, columns=prices.columns).values.astype(float)
    target = positions.reindex(prices.index).fillna(0.0).values

    ret = np.zeros(len(P))
    held, frozen, capital = 0.0, None, np.nan
    for t in range(len(P)):
        # 1. P&L of the position held into today
        if t > 0 and held != 0.0:
            ret[t] = held * frozen @ (P[t] - P[t - 1]) / capital
        # 2. Trade at today's close if the target changed
        if target[t] != held:
            if target[t] == 0.0:
                held, frozen, capital = 0.0, None, np.nan
            elif np.all(np.isfinite(U[t])) and np.all(np.isfinite(P[t])):
                gross = np.sum(np.abs(U[t]) * P[t])
                if gross > 0:
                    held, frozen, capital = target[t], U[t].copy(), gross
            # else: hedge not available today -> keep the current position
    return pd.Series(ret, index=prices.index)


def zscore_strategy(prices, units, z, entry, exit=0.0, allowed=None):
    """
    Z-score mean-reversion strategy: positions from zscore_positions, then
    frozen-hedge returns. `allowed` (bool Series) forces a flat position on
    days where trading is not permitted (e.g. an unstable hedge ratio).
    Returns (returns, positions).
    """
    positions = zscore_positions(z, entry, exit)
    if allowed is not None:
        ok = allowed.reindex(positions.index).eq(True)       # NaN -> not allowed
        positions = positions.where(ok, 0.0)
    return spread_returns(prices, units, positions), positions


#####################
# 3. Performance metrics
#####################

def count_trades(positions):
    """Number of trades opened (entries from flat plus direct flips)."""
    p = np.asarray(positions, float)
    prev = np.r_[0.0, p[:-1]]
    return int(np.sum((p != 0) & (p != prev)))


def performance(returns, positions=None, periods=252, trim_warmup=False):
    """
    Summary statistics of a daily return series.

    Sharpe = mean / std * sqrt(252) (rf = 0); total return and max drawdown
    are computed on the compounded equity curve. `trim_warmup` drops the
    leading days before the first non-zero return.
    """
    r = returns.dropna()
    if trim_warmup:
        nz = np.flatnonzero(r.to_numpy())
        r = r.iloc[nz[0]:] if len(nz) else r
    std = r.std()
    equity = (1 + r).cumprod()
    out = {
        'Sharpe':       r.mean() / std * np.sqrt(periods) if std > 0 else np.nan,
        'Ann. return':  r.mean() * periods,
        'Ann. vol':     std * np.sqrt(periods),
        'Total return': equity.iloc[-1] - 1 if len(r) else np.nan,
        'Max drawdown': (equity / equity.cummax() - 1).min() if len(r) else np.nan,
        'In market':    (r != 0).mean(),
    }
    if positions is not None:
        out['Trades'] = count_trades(positions.reindex(r.index).fillna(0))
    return out


def split_performance(returns, positions, test_start):
    """
    Performance on the training period (warm-up trimmed) and on the test
    period (from test_start on) as a two-row DataFrame.
    """
    train = returns.index < test_start
    return pd.DataFrame({
        'train': performance(returns[train], positions[train], trim_warmup=True),
        'test':  performance(returns[~train], positions[~train]),
    }).T


#####################
# 4. ML single-pair backtest (rebalance-and-replace)
#####################

def run_backtest(prob_pos, hedge_ratio, corn, soy,
                 all_dates=None, trade_freq='daily', method='prob',
                 notional=None):
    """
    Single-position backtest of the ML spread strategy.

    At each rebalance date the previous spread trade is closed and a new one
    opened with the current target size and the hedge ratio observed at that
    moment; between rebalances position and hedge ratio are held flat. Daily
    P&L uses the previous close's position and gamma (look-ahead-free):
        PnL_t = m_{t-1} * [(soy_t - soy_{t-1}) - gamma_{t-1} * (corn_t - corn_{t-1})]

    Parameters
    ----------
    prob_pos    : pd.Series — P(y = +1) from the model, indexed by candidate
                  rebalance dates
    hedge_ratio : pd.Series — hedge ratio gamma on all trading days
    corn, soy   : pd.Series — prices on all trading days
    all_dates   : pd.DatetimeIndex, optional — output dates (default:
                  prob_pos.index intersected with the price indices)
    trade_freq  : 'daily' or 'monthly' (first trading day of each month)
    method      : 'prob' — size from the AFML Sec. 10.3 probability formula
                  'sign' — size +/-1 from the sign of (p - 0.5)
    notional    : reference capital for returns. Default: gross spread
                  notional on the first output date, soy_0 + gamma_0 * corn_0

    Returns
    -------
    dict with 'position', 'gamma', 'daily_pnl', 'cum_pnl', 'daily_ret',
    'cum_ret', 'notional', 'rebalance_dates'
    """
    if all_dates is None:
        all_dates = prob_pos.index.intersection(corn.index).intersection(soy.index)
    all_dates = pd.DatetimeIndex(sorted(all_dates))

    rebalance = select_entry_dates(prob_pos.index, trade_freq)
    rebalance = rebalance[rebalance.isin(all_dates)]

    probs = prob_pos.loc[rebalance].values
    if method == 'prob':
        sizes = bet_size_from_prob(probs)
    elif method == 'sign':
        sizes = np.where(probs >= 0.5, 1.0, -1.0)
    else:
        raise ValueError(f"method must be 'prob' or 'sign', got {method!r}")

    position = pd.Series(sizes, index=rebalance).reindex(all_dates, method='ffill').fillna(0.0)
    gamma = hedge_ratio.loc[rebalance].reindex(all_dates, method='ffill').fillna(0.0)

    dsoy, dcorn = soy.loc[all_dates].diff(), corn.loc[all_dates].diff()
    daily_pnl = (position.shift(1).fillna(0.0)
                 * (dsoy - gamma.shift(1).fillna(0.0) * dcorn)).fillna(0.0)

    if notional is None:
        notional = float(soy.loc[all_dates].iloc[0]
                         + hedge_ratio.loc[all_dates].iloc[0] * corn.loc[all_dates].iloc[0])

    return {
        'position':        position,
        'gamma':           gamma,
        'daily_pnl':       daily_pnl,
        'cum_pnl':         daily_pnl.cumsum(),
        'daily_ret':       daily_pnl / notional,
        'cum_ret':         daily_pnl.cumsum() / notional,
        'notional':        notional,
        'rebalance_dates': rebalance,
    }


def market_neutrality(daily_pnl, corn, soy, label=''):
    """
    Regress daily strategy P&L on the daily price changes of both legs.

    A market-neutral spread strategy should have near-zero loadings on both
    legs; significant loadings mean the hedge is wrong or inconsistently
    applied. Valid even on in-sample windows, since it only measures the
    correlation between P&L and leg moves.
    """
    idx = daily_pnl.index.intersection(corn.index).intersection(soy.index)
    y = daily_pnl.loc[idx].values
    dcorn, dsoy = corn.loc[idx].diff().values, soy.loc[idx].diff().values
    mask = np.isfinite(y) & np.isfinite(dcorn) & np.isfinite(dsoy)
    y, dcorn, dsoy = y[mask], dcorn[mask], dsoy[mask]

    X = np.column_stack([np.ones(len(y)), dcorn, dsoy])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    se = np.sqrt(np.diag((resid @ resid) / (n - k) * np.linalg.inv(X.T @ X)))
    tvals = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0

    if label:
        print(f'=== {label} ===')
    print(f'  n = {n}  |  beta_corn = {beta[1]:+.4f} (t = {tvals[1]:+.2f})  '
          f'beta_soy = {beta[2]:+.4f} (t = {tvals[2]:+.2f})  R^2 = {r2:.3f}')
    return {'beta_corn': beta[1], 'beta_soy': beta[2],
            't_corn': tvals[1], 't_soy': tvals[2], 'r2': r2, 'n': n}
