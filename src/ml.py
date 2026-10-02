# Machine-learning pipeline for spread trading (notebooks 04-05).
#
# For a pair (dep, ind) with  dep = alpha + gamma * ind + spread:
#   pair_frame      — features: rolling-OLS hedge (for labels), 3-state Kalman,
#                     calendar, spread volatility, plus the given weather columns
#   pair_labels     — triple-barrier labels on the frozen-gamma P&L + uniqueness weights
#   assemble_xy     — feature matrix X, labels y, label end times t1, weights w
#   select_features — MDI-top-N union MDA-top-N, veto features with negative MDA
#   tune_xgb / tune_pipeline — grid searches scored by walk-forward neg log-loss
#   SeqBootstrapRF  — random forest on sequentially bootstrapped bags (AFML 4.5)
# For corn-soybean, dep = soybean and ind = corn.

from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .cv import WalkForwardPurgedCV, temporal_split
from .features import (compute_calendar_features, compute_hedge_ratio,
                       compute_kalman_hedge, compute_spread_vol)
from .labels import (apply_pt_sl_on_t1, get_avg_uniqueness, get_bins,
                     get_daily_vol, get_vertical_barrier, seq_bootstrap)


# Columns that are never features: raw prices and estimation intermediates
# (the hedge ratio itself would let the model read the price level).
PRICE_PREFIXES = ('close_', 'open_', 'high_', 'low_', 'volume_')
INTERMEDIATE_COLS = {'hedge_ratio', 'intercept', 'spread', 'kf_hedge_ratio', 'kf_intercept'}

# XGBoost grid: max_depth x learning_rate x n_estimators x min_child_weight (525 configs)
XGB_GRID = list(product([2, 3, 4, 5, 6], [0.01, 0.05, 0.1],
                        [50, 75, 100, 200, 300], [1, 5, 10, 15, 20, 25, 30]))


#####################
# 1. Features, labels, design matrix
#####################

def pair_frame(df, dep_col, ind_col, weather_cols, delta, ols_window=504,
               vol_span=100, verbose=False):
    """Feature frame of one pair (Kalman features depend on delta; OLS does not)."""
    d = df[[dep_col, ind_col] + list(weather_cols)].copy()
    d = compute_hedge_ratio(d, ind_col, dep_col, window=ols_window, verbose=verbose)
    d = compute_kalman_hedge(d, ind_col, dep_col, delta=delta, verbose=verbose)
    d = compute_calendar_features(d)
    d = compute_spread_vol(d, ind_col, dep_col, 'kf_hedge_ratio', span=vol_span)
    return d


def pair_labels(d, dep_col, ind_col, num_days, pt_sl, vol_span=100):
    """
    Triple-barrier labels for an entry on every day with a valid OLS hedge
    ratio: symmetric barriers at pt_sl x EWMA vol of the spread P&L, vertical
    barrier after num_days trading days, gamma frozen at entry.
    Returns (labels, weights) — labels has columns t1, ret, bin.
    """
    dep, ind, hedge = d[dep_col], d[ind_col], d['hedge_ratio']
    valid = d.dropna(subset=['hedge_ratio']).index
    vol = get_daily_vol(ind, dep, hedge, span=vol_span)
    t1 = get_vertical_barrier(valid, num_days=num_days, t_events=valid)
    touches = apply_pt_sl_on_t1(ind, dep, hedge, t1, vol, pt_sl=[pt_sl, pt_sl])
    labels = get_bins(touches, ind, dep, hedge)
    weights = get_avg_uniqueness(labels, d.index)
    return labels, weights


def feature_columns(d):
    """All candidate feature columns of a pair frame."""
    return [c for c in d.columns
            if not c.startswith(PRICE_PREFIXES) and c not in INTERMEDIATE_COLS]


def assemble_xy(d, labels, weights, features=None):
    """
    Design matrix on the labelled dates. y = 1 if the frozen-gamma P&L of a
    long-spread trade was positive at the first barrier touch, else 0.
    Rows with any NaN are dropped.
    """
    features = feature_columns(d) if features is None else list(features)
    X = d.loc[labels.index, features]
    y = labels['bin'].map({-1: 0, 1: 1})
    t1 = labels['t1']
    w = weights.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna() & t1.notna() & w.notna()
    return X[mask], y[mask].astype(int), t1[mask], w[mask]


def select_features(mdi, mda, top_n=20):
    """Union of the MDI and MDA top-N, minus features with negative MDA."""
    union = set(mdi.head(top_n).index) | set(mda.head(top_n).index)
    selected = sorted(f for f in union if mda.loc[f, 'mean'] >= 0)
    vetoed = sorted(f for f in union if mda.loc[f, 'mean'] < 0)
    return selected, vetoed


#####################
# 2. Model tuning
#####################

def make_xgb(params):
    """XGBoost classifier from a dict with max_depth, learning_rate, n_estimators, min_child_weight."""
    return XGBClassifier(max_depth=int(params['max_depth']),
                         learning_rate=float(params['learning_rate']),
                         n_estimators=int(params['n_estimators']),
                         min_child_weight=int(params['min_child_weight']),
                         eval_metric='logloss', random_state=42)


def cv_neg_logloss(model, X, y, t1, w, n_periods=3):
    """Mean train and test neg log-loss over walk-forward purged CV folds."""
    train, test = [], []
    for tr, te in WalkForwardPurgedCV(n_periods=n_periods, t1=t1).split(X):
        m = clone(model)
        m.fit(X.iloc[tr].values, y.iloc[tr].values, sample_weight=w.iloc[tr].values)
        train.append(-log_loss(y.iloc[tr].values, m.predict_proba(X.iloc[tr].values)))
        test.append(-log_loss(y.iloc[te].values, m.predict_proba(X.iloc[te].values)))
    return np.mean(train), np.mean(test)


def tune_xgb(X, y, t1, w, grid=XGB_GRID):
    """Grid search over XGBoost hyperparameters; returns all configs sorted by test neg log-loss."""
    rows = []
    for md, lr, ne, mcw in grid:
        params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': ne, 'min_child_weight': mcw}
        tr, te = cv_neg_logloss(make_xgb(params), X, y, t1, w)
        rows.append({**params, 'train_nll': tr, 'test_nll': te, 'gap': tr - te})
    return pd.DataFrame(rows).sort_values('test_nll', ascending=False).reset_index(drop=True)


def tune_pipeline(df, dep_col, ind_col, weather_cols, features, xgb_params,
                  delta_grid=(1e-6, 1e-5, 1e-4), num_days_grid=(100, 150, 200),
                  pt_sl_grid=(1.5, 2.0, 2.5), n_holdout=375):
    """
    Grid search over the pipeline hyperparameters that require rebuilding
    features/labels: Kalman delta, vertical barrier num_days, barrier width
    pt_sl. Scored on the dev set only (the holdout is split off first).
    """
    frames = {delta: pair_frame(df, dep_col, ind_col, weather_cols, delta) for delta in delta_grid}
    base = frames[delta_grid[0]]           # labels use the OLS hedge, independent of delta
    model = make_xgb(xgb_params)

    rows = []
    for num_days, pt_sl in product(num_days_grid, pt_sl_grid):
        labels, weights = pair_labels(base, dep_col, ind_col, num_days, pt_sl)
        for delta in delta_grid:
            X, y, t1, w = assemble_xy(frames[delta], labels, weights, features)
            sp = temporal_split(X, y, t1, sample_weight=w, n_holdout=n_holdout, verbose=False)
            tr, te = cv_neg_logloss(model, sp['X_dev'], sp['y_dev'], sp['t1_dev'], sp['w_dev'])
            rows.append({'delta': delta, 'num_days': num_days, 'pt_sl': pt_sl,
                         'n_events': len(labels), 'train_nll': tr, 'test_nll': te,
                         'gap': tr - te})
    return pd.DataFrame(rows).sort_values('test_nll', ascending=False).reset_index(drop=True)


#####################
# 3. Random forest with sequential bootstrap (AFML Ch. 4 & 6)
#####################

class SeqBootstrapRF:
    """
    Bagged decision trees, each trained on a sequentially bootstrapped bag.

    With heavily overlapping labels (mean uniqueness ~0.09) a standard
    bootstrap draws near-identical bags, so the trees are highly correlated.
    Bags are drawn in proportion to uniqueness and sized
    avg_uniqueness x n_events (AFML Sec. 4.5).
    """

    def __init__(self, n_trees=200, max_depth=5, min_leaf_frac=0.05, min_bag=50):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf_frac = min_leaf_frac
        self.min_bag = min_bag

    def fit(self, X, y, sample_weight, t1, trading_days):
        self.bag_size_ = max(int(sample_weight.mean() * len(t1)), self.min_bag)
        self.trees_ = []
        for i in range(self.n_trees):
            bag = seq_bootstrap(t1, trading_days, s_length=self.bag_size_, random_state=i)
            tree = DecisionTreeClassifier(
                criterion='entropy', max_features='sqrt', max_depth=self.max_depth,
                min_samples_leaf=max(1, int(self.min_leaf_frac * self.bag_size_)),
                class_weight='balanced', random_state=i)
            tree.fit(X.values[bag], y.values[bag], sample_weight=sample_weight.values[bag])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        for tree in self.trees_:
            p = tree.predict_proba(X.values)
            for j, cls in enumerate(tree.classes_):   # a bag may contain one class only
                proba[:, int(cls)] += p[:, j]
        return proba / len(self.trees_)


#####################
# 4. Probability -> position
#####################

def band_positions(prob, tau=0.15):
    """+1 if p > 0.5 + tau, -1 if p < 0.5 - tau, else flat (model used as a directional filter)."""
    p = np.asarray(prob, float)
    return pd.Series(np.where(p > 0.5 + tau, 1.0, np.where(p < 0.5 - tau, -1.0, 0.0)),
                     index=prob.index)
