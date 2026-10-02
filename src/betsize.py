# Bet sizing for the ML strategy, following AFML Chapter 10.
#
# Two approaches, both producing a target position series in [-1, +1]:
#
#   compute_position_prob    (§10.3 + §10.4)
#       Translate P(y = +1) into a signed bet size via a z-score test
#       against a coin flip, then average across all concurrently-active
#       bets (defined by triple-barrier lifespans).
#
#   compute_position_budget  (§10.2 budgeting)
#       Size purely from bet concurrency:
#           m_t = c_{t,l} / max(c_l) - c_{t,s} / max(c_s).
#       Only each bet's SIDE enters (from sign(p - 0.5)); magnitudes
#       are discarded. Expanding max by default, for an honest backtest.
#
# Both support two trading frequencies:
#
#   trade_freq='daily'    — new bets may enter every trading day, and
#                            the target position is re-evaluated daily.
#   trade_freq='monthly'  — new bets may enter only on the first trading
#                            day of each month, and the target position
#                            is evaluated only on those dates, held flat
#                            in between. Matches a fund that physically
#                            rebalances once a month.


import numpy as np
import pandas as pd
from scipy.stats import norm


# ===================================================================
# 1. Bet sizing from predicted probability (AFML §10.3, binary case)
# ===================================================================

def bet_size_from_prob(prob_pos):
    """
    Translate P(y = +1) into a signed bet size in [-1, +1] (AFML §10.3).

    Binary-case formula:
        side    = +1 if prob_pos >= 0.5 else -1
        p_max   = max(prob_pos, 1 - prob_pos)
        z       = (p_max - 0.5) / sqrt(p_max * (1 - p_max))
        size    = side * (2 * Phi(z) - 1)

    Properties:
        prob_pos = 0.5   -> size = 0   (indifferent, no bet)
        prob_pos -> 0    -> size -> -1 (full short)
        prob_pos -> 1    -> size -> +1 (full long)
        Monotonic in prob_pos; no calibration assumed beyond ordering.

    Parameters
    ----------
    prob_pos : array-like of P(y = +1), values in [0, 1]

    Returns
    -------
    np.ndarray of the same shape, bet sizes in [-1, +1]
    """
    p = np.asarray(prob_pos, dtype=float)
    side  = np.where(p >= 0.5, 1.0, -1.0)
    p_max = np.where(p >= 0.5, p, 1.0 - p)

    # Clip off 0.5 by a hair to avoid 0/0 in the z statistic at p=0.5
    # (size_mag = 0 in that limit anyway — this is just numerical).
    p_max = np.clip(p_max, 0.5 + 1e-6, 1.0 - 1e-6)
    z     = (p_max - 0.5) / np.sqrt(p_max * (1.0 - p_max))
    return side * (2.0 * norm.cdf(z) - 1.0)


# ===================================================================
# 2. Entry-date selection (trading frequency)
# ===================================================================

def select_entry_dates(candidate_index, trade_freq='daily'):
    """
    Pick entry dates from a set of candidates.

    'daily'   : every candidate is an entry.
    'monthly' : the first candidate in each calendar month is an entry
                (subsequent candidates in the same month are ignored).

    Parameters
    ----------
    candidate_index : pd.DatetimeIndex
        Days on which an entry would in principle be allowed.
    trade_freq : {'daily', 'monthly'}

    Returns
    -------
    pd.DatetimeIndex — sorted subset of candidate_index.
    """
    idx = pd.DatetimeIndex(candidate_index)

    if trade_freq == 'daily':
        return idx
    elif trade_freq == 'monthly':
        s = pd.Series(idx, index=idx)
        monthly = s.groupby([s.index.year, s.index.month]).first()
        return pd.DatetimeIndex(monthly.values).sort_values()
    else:
        raise ValueError(f"trade_freq must be 'daily' or 'monthly', got {trade_freq!r}")


# ===================================================================
# 3. Position series: §10.3 probability-based (+ §10.4 averaging)
# ===================================================================

def compute_position_prob(prob_pos, t1, all_dates, trade_freq='daily'):
    """
    Target position series from probability-based bet sizing (§10.3) with
    active-bet averaging (§10.4).

    Procedure:
        1. Pick entry dates per trade_freq.
        2. For each entry, compute a per-bet size via bet_size_from_prob.
        3. Pick evaluation dates: all_dates for daily, monthly rebalance
           dates for monthly (the fund can only adjust the portfolio
           once a month).
        4. At each evaluation date, position = mean of sizes of all
           currently-active bets. AFML convention: bet i is active at
           time t iff entry_i <= t < t1_i.
        5. For monthly mode, forward-fill the evaluation-date positions
           to the full all_dates index (step function held between
           rebalances).

    Parameters
    ----------
    prob_pos : pd.Series — P(y = +1), indexed by candidate entry dates
    t1 : pd.Series — barrier exit timestamp for each entry, same index
    all_dates : pd.DatetimeIndex — output dates for the position series
    trade_freq : 'daily' or 'monthly'

    Returns
    -------
    pd.Series — target position in [-1, +1], indexed by all_dates.
    """
    # 1. Entry dates (filter to those with a valid t1)
    entries = select_entry_dates(prob_pos.index, trade_freq)
    entries = entries[t1.reindex(entries).notna()]

    # 2. Per-bet sizes
    sizes = pd.Series(
        bet_size_from_prob(prob_pos.loc[entries].values),
        index=entries,
    )
    t1_e = t1.loc[entries]

    # 3. Evaluation dates: daily = all_dates; monthly = monthly rebalance
    if trade_freq == 'monthly':
        eval_dates = select_entry_dates(all_dates, 'monthly')
    else:
        eval_dates = all_dates

    # 4. Average active bets at each evaluation date
    position = pd.Series(0.0, index=eval_dates)
    entries_arr = sizes.index.values
    t1_arr      = t1_e.values
    sizes_arr   = sizes.values

    for loc in eval_dates:
        active = (entries_arr <= loc) & (loc < t1_arr)
        if active.any():
            position.loc[loc] = sizes_arr[active].mean()

    # 5. For monthly, forward-fill to all_dates (held between rebalances)
    if trade_freq == 'monthly':
        position = position.reindex(all_dates, method='ffill').fillna(0.0)

    return position


# ===================================================================
# 4. Position series: §10.2 concurrency-based budgeting
# ===================================================================

def compute_position_budget(prob_pos, t1, all_dates,
                            trade_freq='daily', max_mode='expanding'):
    """
    Target position from the §10.2 budgeting rule:

        m_t = c_{t,l} / max_{s<=t}{c_{s,l}} - c_{t,s} / max_{s<=t}{c_{s,s}}

    where c_{t,l} and c_{t,s} are counts of concurrent long and short
    bets at time t. Each bet's side is sign(prob_pos - 0.5) at its entry
    — magnitudes are discarded.

    The max in the denominator is expanding (running max up to time t)
    by default, to keep the backtest honest. Set max_mode='full' to use
    the full-sample max, matching AFML literally at the cost of minor
    look-ahead.

    For trade_freq='monthly', concurrency is measured and normalized
    only at monthly rebalance dates (the fund only observes its
    portfolio state once a month), then held flat between rebalances.

    Parameters
    ----------
    prob_pos : pd.Series — P(y = +1), indexed by candidate entry dates
    t1 : pd.Series — barrier exits, same index
    all_dates : pd.DatetimeIndex — output dates
    trade_freq : 'daily' or 'monthly'
    max_mode : 'expanding' (honest) or 'full' (matches AFML literally)

    Returns
    -------
    pd.Series — target position in [-1, +1], indexed by all_dates.
    """
    # 1. Entry dates and sides
    entries = select_entry_dates(prob_pos.index, trade_freq)
    entries = entries[t1.reindex(entries).notna()]

    sides = np.where(prob_pos.loc[entries].values >= 0.5, 1, -1)
    t1_e  = t1.loc[entries]

    # 2. Evaluation dates
    if trade_freq == 'monthly':
        eval_dates = select_entry_dates(all_dates, 'monthly')
    else:
        eval_dates = all_dates

    # 3. Concurrency at each evaluation date
    eval_arr    = eval_dates.values
    c_long_arr  = np.zeros(len(eval_dates), dtype=float)
    c_short_arr = np.zeros(len(eval_dates), dtype=float)

    for entry, exit_, side in zip(entries, t1_e.values, sides):
        span = (eval_arr >= entry) & (eval_arr < exit_)
        if side > 0:
            c_long_arr[span] += 1.0
        else:
            c_short_arr[span] += 1.0

    c_long  = pd.Series(c_long_arr,  index=eval_dates)
    c_short = pd.Series(c_short_arr, index=eval_dates)

    # 4. Normalize by max (expanding or full)
    if max_mode == 'expanding':
        max_l = c_long.cummax().replace(0.0, np.nan)
        max_s = c_short.cummax().replace(0.0, np.nan)
    elif max_mode == 'full':
        ml = c_long.max(); ms = c_short.max()
        max_l = pd.Series(ml if ml > 0 else np.nan, index=eval_dates)
        max_s = pd.Series(ms if ms > 0 else np.nan, index=eval_dates)
    else:
        raise ValueError(f"max_mode must be 'expanding' or 'full', got {max_mode!r}")

    long_share  = (c_long  / max_l).fillna(0.0)
    short_share = (c_short / max_s).fillna(0.0)
    position    = long_share - short_share

    # 5. For monthly, forward-fill to all_dates
    if trade_freq == 'monthly':
        position = position.reindex(all_dates, method='ffill').fillna(0.0)

    return position