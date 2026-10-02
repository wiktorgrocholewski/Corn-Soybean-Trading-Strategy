# Trading the corn–soybean spread

Research project of the Tilburg Investment Club (TIC), Spring 2026 —
Wiktor Grocholewski, Roos de Brabander, Antonio Harley, Artem Zeziulin.

**Question.** Corn and soybean compete for the same land and are driven by the same demand and weather shocks, so
their prices should not drift apart forever. Can the spread between them — and, more generally, between grain
commodities — be traded profitably, and does weather in the growing regions help to time it?

**Answer (short).** Not robustly. We built strategies of increasing complexity — z-score rules on cointegrated
pairs and baskets, a portfolio of pairs with OU-based allocation, triple-barrier machine learning with weather
features, and a Bayesian weather-driven regime-switching model. Once every parameter is chosen on training data only
and P&L is computed as tradeable, capital-normalised returns, none of them delivers an out-of-sample Sharpe ratio that
is distinguishable from zero. The write-up is in [`report/report.tex`](report/report.tex); the main numbers are
collected in [Results](#results).

---

## Repository layout

```
├── data/
│   ├── raw/          downloaded data: ETF and futures prices, daily weather per region
│   ├── processed/    datasets built by notebook 01 (all other notebooks load these)
│   └── cache/        cached tuning results of notebook 05
├── notebooks/        the analysis, in the order of the project (see below)
├── src/              all functions used by the notebooks
├── report/
│   ├── report.tex    the report (LaTeX) + references.bib
│   ├── figures/      figures saved by the notebooks
│   └── tables/       result tables saved by the notebooks (numbers quoted in the report)
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run the notebooks in order; each one starts with a short description of the question it answers. Notebook 01
rebuilds `data/processed/` from `data/raw/` (set `DOWNLOAD = True` to re-download everything, which needs internet and
will shift the results slightly). Runtime: notebooks 01–04 and 06 take a few minutes each; notebook 05 tunes six
pairs (~15 min the first time, then cached in `data/cache/`).

## Notebooks

The notebooks follow the chronology of the project, from the simplest model to the most complex, and do not overlap:
every test or strategy lives in exactly one notebook. Section numbers refer to the report.

| # | Notebook | Question | Methods | Report |
|---|---|---|---|---|
| 01 | `01_data.ipynb` | What data do we use? | Yahoo Finance (futures, ETFs), Open-Meteo weather, 30-day rolling weather aggregates | §2 |
| 02 | `02_pair_cointegration.ipynb` | Is corn–soybean cointegrated, and can a z-score rule trade it? | ADF, Engle–Granger, Johansen, ECM, half-life; rolling OLS (A); static Johansen / Kalman / rolling Johansen on log prices (B) | §3 |
| 03 | `03_multi_commodity.ipynb` | Does adding wheat and cotton help? | Johansen baskets (C, D); portfolio of six pairs with EW / MRB / MRR allocation (E) | §4 |
| 04 | `04_ml_triple_barrier.ipynb` | Can a classifier with weather features time the spread? | 3-state Kalman filter, triple-barrier labels, uniqueness weights, walk-forward purged CV, MDI/MDA selection, XGBoost, sequential-bootstrap RF, bet sizing, backtest | §5 |
| 05 | `05_ml_multi_pair.ipynb` | Does the ML model work on other pairs / as a portfolio? | Notebook 04 per pair + OU allocation of notebook 03 | §5 |
| 06 | `06_bayesian_regime.ipynb` | Do weather-driven regimes explain the spread? | Persistent-probit regime-switching AR(1), Gibbs sampler with FFBS; identification and bimodality checks | §6 |

Who did what: the classical strategies (02, 03 C–D) are Artem's (price levels, baskets) and Antonio's (log prices,
Kalman, rolling Johansen); the pairs portfolio (03 E) is Roos's, corrected by Wiktor; the data pipeline, ML pipeline
(04, 05) and regime model (06) are Wiktor's, with Roos on the first data and ML versions.

## `src/` modules

| Module | Contents |
|---|---|
| `data.py` | Universe (tickers, weather locations), download, rolling weather, `build_*` / `load_*` datasets, `save_figure` / `save_table` |
| `cointegration.py` | `adf_table`, `engle_granger`, `johansen_test`, `half_life`, `ecm`; hedge estimators `rolling_ols`, `rolling_johansen`, `kalman_regression` |
| `backtest.py` | `rolling_zscore`, `zscore_positions`, `spread_returns` (frozen hedge, capital-normalised), `performance`, `split_performance`; ML `run_backtest`, `market_neutrality` |
| `features.py` | Rolling OLS hedge, 3-state Kalman filter with AR(1) spread state, calendar and spread-volatility features |
| `labels.py` | Triple-barrier labels with frozen hedge, average uniqueness, sequential bootstrap (AFML Ch. 3–4) |
| `cv.py` | Dev/holdout split with purging, walk-forward purged CV, purged k-fold, `cv_score`, MDA and MDI importance (AFML Ch. 7–8) |
| `ml.py` | Pair pipeline (`pair_frame`, `pair_labels`, `assemble_xy`), feature selection, XGBoost and pipeline grid searches, `SeqBootstrapRF`, `band_positions` |
| `betsize.py` | Probability-based bet sizing and concurrency budgeting (AFML Ch. 10) |
| `multipairs.py` | OU estimation, MRB / MRR weights, multi-pair backtest, dev tuning and holdout evaluation (Lee, Leung & Ning 2023) |
| `regime.py` | Bayesian persistent regime-switching model: data preparation, FFBS, conditional posteriors, Gibbs sampler, diagnostics |

## Data

| File | Content | Used in |
|---|---|---|
| `raw/futures_{corn,soybean}.csv` | ZC=F, ZS=F front-month futures, 2005–2026 | 02 |
| `raw/etf_{corn,soybean,wheat,cotton}.csv` | CORN, SOYB, WEAT (Teucrium), COTN.L (WisdomTree cotton, London) | 03–06 |
| `raw/weather_{us,brazil,wheat,cotton}.csv` | 15 daily variables × 3 locations per region, Open-Meteo archive | 04–06 |
| `processed/futures_corn_soybean.csv` | futures closes on common days | 02 |
| `processed/full_dataset.csv` | ETF closes (outer join) + 180 rolling weather features | 03–06 |

Caveats: the futures series is a continuous front-month series from Yahoo and jumps at every contract roll; the
ETFs start in 2011. The cotton ETN used in the first version (iPath BAL) was delisted in 2023, so cotton is the
London-listed COTN.L. The raw cotton weather for "nagpur" was downloaded with the wrong longitude sign (a point near
Cuba); the coordinate is fixed in `src/data.py` but the file has not been re-downloaded.

## Results

Out-of-sample Sharpe ratios (capital-normalised daily returns, risk-free rate 0). All parameters were chosen on the
training / dev sample only. Full tables are in `report/tables/`.

| Strategy | Notebook | Test period | Train Sharpe | Test Sharpe |
|---|---|---|---|---|
| A — pair, price levels, rolling OLS hedge | 02 | Oct 2019 – Feb 2026 (futures) | −0.01 | −0.27 |
| B1 — pair, log prices, static Johansen | 02 | 〃 | 0.16 | 0.58 |
| B2 — pair, log prices, Kalman hedge | 02 | 〃 | 0.64 | 0.01 |
| B3 — pair, log prices, rolling Johansen | 02 | 〃 | 0.40 | 0.10 |
| C — rolling Johansen baskets (5 baskets) | 03 | Nov 2021 – Apr 2026 (ETFs) | 0.14 … 0.98 | −0.65 … 0.35 |
| D — 4-commodity log basket (static / Kalman / rolling) | 03 | 〃 | 0.63 … 0.81 | −0.28 … −0.03 |
| E — portfolio of six pairs, EW / MRB / MRR | 03 | 〃 | 0.38 / 0.07 / 0.24 | 0.42 / 0.38 / 0.46 |
| XGBoost corn–soybean, probability / sign sizing | 04 | Jan 2024 – Jul 2025 | CV accuracy 55% | 0.76 / 0.41 |
| XGBoost on six pairs, EW / MRB / MRR portfolio | 05 | Dec 2023 – Sep 2025 | — | 0.51 / 0.68 / 0.66 |
| Bayesian regime switching | 06 | — | sampler does not converge; regimes not identified | not traded |

With test samples of 1.5–6 years, the standard error of an annualised Sharpe ratio is roughly 0.4–0.9, so none of
these numbers is significantly different from zero — and the best of many tried specifications is biased upwards.
Transaction costs are ignored.

## Methodological choices worth knowing

* **Frozen hedge.** The hedge ratio is fixed when a trade is opened. A daily-updated rolling hedge creates P&L that
  could only be earned by re-hedging every day (AFML §2.4.1); earlier versions of the classical strategies selected
  parameters on such non-tradeable P&L.
* **Log-price spreads are dollar-weighted.** A spread in log prices is a portfolio of *dollar* positions, so its
  hedge is converted to units with `backtest.log_to_units` before computing price P&L.
* **Walk-forward CV instead of purged k-fold** for the ML pipeline, because the rolling OLS and Kalman features are
  estimated sequentially: training on later folds would use features built from test-fold prices.
* **Purging at the holdout boundary.** Dev observations whose triple-barrier label is resolved inside the holdout are
  dropped before the final fit.
* **One evaluation of the test set.** Every threshold, window and hyperparameter is chosen on training data. In the
  first versions several of them were chosen after looking at test results, which inflated the reported numbers.

## Project history

The project started (Feb–Mar 2026) with front-month futures, weather from three arbitrary locations, an ECM-residual
spread and an XGBoost classifier on ~130 features; its apparent backtest Sharpe of ~2 came from scoring overlapping
daily labels as daily returns. In parallel we looked at Brazilian planted-area statistics (CONAB), which show soybean
area replacing first-crop corn. These first-iteration notebooks, as well as duplicated data downloads and
cointegration tests, were removed in the clean-up of September 2026; they remain available in the git history
(commit `6fda2d9` and earlier).

## References

* E. P. Chan (2013), *Algorithmic Trading: Winning Strategies and Their Rationale*, Wiley.
* M. López de Prado (2018), *Advances in Financial Machine Learning*, Wiley.
* H. Lee, T. Leung, B. Ning (2023), "A diversification framework for multiple pairs trading strategies", *Risks* 11(5), 93.
* R. F. Engle, C. W. J. Granger (1987), "Co-integration and error correction", *Econometrica* 55(2), 251–276.
* S. Johansen (1991), "Estimation and hypothesis testing of cointegration vectors in Gaussian VAR models", *Econometrica* 59(6), 1551–1580.
* J. H. Albert, S. Chib (1993), "Bayesian analysis of binary and polychotomous response data", *JASA* 88(422), 669–679.
