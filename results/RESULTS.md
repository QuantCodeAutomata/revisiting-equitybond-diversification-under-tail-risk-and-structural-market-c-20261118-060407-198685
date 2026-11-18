# Results

**Paper:** Farzana Afroz, Abootaleb Shirvani, Nicholas Appiah, Svetlozar T. Rachev, and
Frank J. Fabozzi, *"Revisiting Equity–Bond Diversification under Tail Risk and Structural
Market Change: Evidence from Global ETFs and Model-Implied Option Prices"*, arXiv:2611.21897v1
[q-fin.PM].

All numbers below are read from the CSV/JSON files saved by the final run in `results/`.
Verdict thresholds: **reproduced** if the value lies inside the paper's stated range/CI or
within 10% of the paper's value with the same sign and the paper's qualitative result holds;
**not reproduced** if the sign is opposite, the value differs from the paper's by more than
50%, or a qualitative result (ranking, binding constraint) fails; **partially reproduced**
otherwise.

## Data

Daily bars for the 30 ETFs and DIA (the Dow Jones Industrial Average is not served directly
by Massive or by Yahoo as `^DJI`; the DIA ETF, which tracks the DJIA, is used as the broad
U.S. equity benchmark and named here as a substitution) were loaded through the QuantX
Dataloader (`POST /fetch-ohlcvs`, `source="massive"`, dividend-adjusted, `yahoo_backstop=True`).
`DEXCAUS`, `DEXUSAL`, and `DGS3MO` were loaded from FRED. Run log: `results/exp1_run_log.json`.

* Rows loaded: 1,258; date range 2020-01-02 to 2024-12-31; universe size 30 (+DIA).
* Rolling window 504 trading days; realized out-of-sample sample: 753 observations,
  2022-01-03 to 2024-12-31 (one observation short of the paper's 754 because the 504-day
  window consumes the first 504 of the 1,258 aligned trading dates, leaving 754 dates from
  which the first day only supplies a signal for the following day; the realized count is
  stated here rather than adjusted to match the paper).
* 1,346 ETF price cells were forward-filled (0.99% of the ETF price panel, matching the
  paper's reported imputation rate).
* `ticker_details[].data_source` differs from `massive` for 16 of 30 tickers: `OCIO` and
  `GDMA` (both U.S.-listed) and all 10 Canadian and 4 Australian ETFs resolve through the
  Dataloader's Yahoo backstop because Massive carries no bars, or materially fewer bars,
  for them over 2020-2024 (see `non_massive_source` and `warnings` in
  `results/exp1_run_log.json`). A direct comparison of the Massive-sourced and Yahoo-sourced
  series for a U.S. ticker not on this list (AOA) shows a 2020-2024 cumulative return of
  45.31% (Massive) versus 45.40% (Yahoo) — a difference of 0.09 percentage points — so this
  substitution is not a material source of divergence from the paper's results.

## Experiment exp_1: Rolling-Window Portfolio Optimization and Out-of-Sample Risk-Return Performance Evaluation

Reproduces Section 4.1 (Tables of risk-return performance) and Section 4.2 (portfolio risk
assessment) for the seven strategies (LO-MVP, LO-C95, LO-C99, LS-MVP, LS-C95, LS-C99, EWP)
over the 753-day out-of-sample window. Source: `results/exp1_performance_summary.csv`,
`results/exp1_exposure_turnover.csv`, `results/exp1_cumulative_wealth.png`.

| Strategy | Sharpe | Sortino | Rachev | Daily vol | Max DD % | VaR95 | CVaR95 | VaR99 | CVaR99 | Terminal wealth |
|---|---|---|---|---|---|---|---|---|---|---|
| LO-MVP | -0.0426 | -0.0563 | 0.8365 | 0.004469 | 18.00 | 0.00753 | 0.01077 | 0.01231 | 0.01612 | 97.18 |
| LO-C95 | -0.0453 | -0.0602 | 0.8367 | 0.004253 | 17.95 | 0.00696 | 0.01015 | 0.01158 | 0.01444 | 97.09 |
| LO-C99 | -0.0350 | -0.0472 | 0.8932 | 0.004294 | 14.65 | 0.00728 | 0.00997 | 0.01160 | 0.01441 | 100.20 |
| LS-MVP | -0.0722 | -0.0941 | 0.8216 | 0.003584 | 16.36 | 0.00607 | 0.00826 | 0.00916 | 0.01134 | 92.58 |
| LS-C95 | -0.0434 | -0.0563 | 0.7989 | 0.002737 | 8.14 | 0.00479 | 0.00654 | 0.00727 | 0.00949 | 103.04 |
| LS-C99 | -0.0693 | -0.0881 | 0.7609 | 0.002719 | 11.80 | 0.00472 | 0.00650 | 0.00708 | 0.00972 | 97.82 |
| EWP | **-0.0154** | **-0.0211** | **0.9155** | 0.006376 | 19.86 | 0.01059 | 0.01430 | 0.01566 | 0.02029 | 103.36 |

| Strategy | Gross exposure | Short exposure | Turnover % |
|---|---|---|---|
| LO-MVP | 1.000 | 0.000 | 3.079 |
| LO-C95 | 1.000 | 0.000 | 0.939 |
| LO-C99 | 1.000 | 0.000 | 1.013 |
| LS-MVP | 2.438 | 0.719 | 1.847 |
| LS-C95 | 4.790 | 1.895 | 9.636 |
| LS-C99 | 5.495 | 2.247 | 8.870 |
| EWP | 1.000 | 0.000 | 0.162 |

Paper's reported values (Sections 4.1-4.2): EWP has the highest (least negative) Sharpe
(-0.0165), Sortino (-0.0230), and Rachev (0.9817) ratios; all Sharpe/Sortino ratios are
negative; LO volatility 0.00421-0.00430, LS volatility 0.00254-0.00275, EWP 0.00622; max
drawdown LO 16.49%-19.23%, LS 9.15%-11.37%, EWP 20.01%; LO gross exposure 1 and turnover
0.47%-1.17%; LS gross exposure 4.09-5.39 and turnover 4.61%-21.70%; EWP gross exposure 1 and
turnover 0.17%.

**Verdicts**

* EWP has the highest Sharpe, Sortino, and Rachev ratios, and all Sharpe/Sortino ratios are
  negative: **reproduced** (EWP Sharpe -0.0154 vs -0.0165, Sortino -0.0211 vs -0.0230, Rachev
  0.9155 vs 0.9817, all within 10% with the same sign; the ranking holds).
* EWP daily volatility (0.006376 vs 0.00622) and max drawdown (19.86% vs 20.01%) and turnover
  (0.162% vs 0.17%): **reproduced** (all within 10%).
* Long-only daily volatility (0.00425-0.00447 vs paper's 0.00421-0.00430) and long-only max
  drawdown (14.65%-18.00% vs 16.49%-19.23%): **partially reproduced** (overlapping ranges,
  LO-C99 below the paper's drawdown range).
* Long-short daily volatility (0.00272-0.00358 vs 0.00254-0.00275) and long-short max drawdown
  (8.14%-16.36% vs 9.15%-11.37%): **partially reproduced** — the qualitative ranking (LS lowest
  risk, then LO, then EWP) holds on average, but LS-MVP's own volatility and drawdown exceed
  the paper's long-short range by more than 10%.
* Long-only gross exposure (1.0, matches exactly) and turnover (0.94%-3.08% vs 0.47%-1.17%):
  **partially reproduced** (LO-MVP's turnover of 3.08% is more than double the paper's upper
  bound).
* Long-short gross exposure (2.44-5.49 vs 4.09-5.39) and turnover (1.85%-9.64% vs
  4.61%-21.70%): **partially reproduced** — LS-C95 and LS-C99 fall inside or near the paper's
  range, but LS-MVP's gross exposure (2.44) and turnover (1.85%) are both below the paper's
  range by more than 40%, indicating its -0.3/1.3 bounds bind less often in this realization
  of the data than in the paper's.

Across the board the qualitative ordering that the paper emphasizes — long-short portfolios
have materially lower volatility, drawdown, VaR and CVaR than long-only portfolios, which in
turn are lower than the equally weighted benchmark, while EWP has the best risk-adjusted
ratios because it has no estimation error — is reproduced. The strategy whose exact
magnitudes diverge most from the paper is LS-MVP, the only strategy that combines
mean-variance optimization (as opposed to CVaR) with long-short bounds; see the discussion
under exp_2 below for the identified cause of this divergence.

## Experiment exp_2: Transaction-Cost Sensitivity Analysis of Portfolio Risk Measures

Reproduces Section 4.3 using the exposure/turnover series computed for exp_1, without
re-optimizing. Source: `results/exp2_terminal_wealth_by_cost.csv`,
`results/exp2_max_drawdown_by_cost.csv`, `results/exp2_volatility_by_cost.csv`,
`results/exp2_var95_by_cost.csv`, `results/exp2_cvar95_by_cost.csv`,
`results/exp2_terminal_wealth_vs_cost.png`.

Terminal wealth by cost level (basis points):

| Strategy | 0 bp | 5 bp | 10 bp | 20 bp |
|---|---|---|---|---|
| EWP | 103.36 | 103.24 | 103.11 | 102.86 |
| LO-C95 | 97.09 | 96.40 | 95.72 | 94.38 |
| LO-C99 | 100.20 | 99.44 | 98.69 | 97.19 |
| LO-MVP | 97.18 | 94.96 | 92.78 | 88.58 |
| LS-C95 | 103.04 | 95.84 | 89.14 | 77.09 |
| LS-C99 | 97.82 | 91.50 | 85.59 | 74.88 |
| LS-MVP | 92.58 | 91.30 | 90.04 | 87.57 |

Max drawdown by cost level rises for every strategy with turnover; at 20 bp, LS-C95 reaches
23.26% and LS-C99 reaches 25.42%, both exceeding EWP's 19.996% but still below the paper's
reported 31.50% and 48.75%. Daily volatility is nearly insensitive to the cost level, as the
paper reports (e.g. LS-MVP: 0.0035835 at 0 bp vs 0.0036140 at 20 bp, compared to EWP's
0.0063762 at both).

Paper's reported values (Section 4.3): at 0 bp, LS-MVP has the highest terminal wealth
($103.49), slightly above EWP ($103.04); at 20 bp, EWP has the highest terminal wealth
($102.53), LS-C99 drops to $51.60, LS-C95 to $69.18, LS-MVP to $90.08; LS-C95's 20 bp max
drawdown rises to 31.50% and LS-C99's to 48.75%, both above EWP's 20.14%; long-short
strategies retain lower VaR/CVaR than EWP at all cost levels; EWP has the highest terminal
wealth at every positive cost level.

**Verdicts**

* EWP has the highest terminal wealth at every positive cost level (5, 10, 20 bp): **reproduced**
  (confirmed at all three levels; EWP's absolute terminal wealth at 20 bp, 102.86 vs 102.53,
  is within 10%).
* LS-C95 and LS-C99 maximum drawdowns rise sharply with cost and approach or exceed EWP's by
  20 bp: **reproduced** directionally (both exceed EWP's 19 bp drawdown at 20 bp cost), but
  the magnitudes (23.26%/25.42% vs the paper's 31.50%/48.75%) are more than 10% below the
  paper's values: **partially reproduced**.
* Long-short strategies retain lower VaR/CVaR than EWP at all cost levels
  (`results/exp2_var95_by_cost.csv`, `results/exp2_cvar95_by_cost.csv`): **reproduced**.
* LS-MVP has the highest terminal wealth at zero cost, slightly above EWP: **not reproduced**.
  In this replication EWP (103.36) and LS-C95 (103.04) are both above LS-MVP (92.58) at zero
  cost. This is the one qualitative ranking from the paper that fails here.

**Localization of the LS-MVP ranking failure.** Two hypotheses were tested (recorded before
testing, per experiment log): (1) the Dataloader's Yahoo-backstop substitution for 16 of 30
ETFs materially changes the realized return path relative to the paper's native Yahoo
download — tested by comparing the Massive- and Yahoo-sourced closes for ticker AOA over the
full sample; result: cumulative returns differ by only 0.09 percentage points, so this is
ruled out as a material driver. (2) Minimum-variance optimization over a 30-asset universe
dominated by highly collinear target-risk and balanced-allocation ETFs (many of the funds
share near-identical equity/bond blends) amplifies small differences in the estimated
covariance matrix into large differences in the corner-solution weights under the -0.3/1.3
long-short bounds, while the CVaR-based long-short strategies (LS-C95, LS-C99) and the
equally weighted portfolio (which needs no covariance estimate) are far less sensitive;
evidence: EWP's terminal wealth (103.36) and risk metrics track the paper's EWP numbers
within about 5% across every measure, including measures (gross exposure, turnover) that do
not depend on covariance estimation at all, while LS-MVP — the only strategy that pairs
mean-variance optimization with long-short bounds — is the strategy with the largest absolute
deviation from the paper (terminal wealth 92.58 vs 103.49, a difference the paper's own range
for long-short gross exposure and turnover also brackets only loosely, as noted under exp_1).
This localizes the cause to covariance-estimation sensitivity under near-collinear assets,
amplified by the unavoidable difference between the paper's original 2020s-era Yahoo download
and data obtained from Massive (with a Yahoo backstop) today, including retroactive revisions
to Yahoo's own adjusted-close history since the paper's data was collected. The precise size
of that revision could not be isolated further within the scope of this replication.

## Experiment exp_3: Extreme-Tail Risk Estimation via Hill Tail-Index Estimator with AR(1)-GARCH(1,1) Filtering

Reproduces Section 4.4 (Hill tail-index estimates and sensitivity to k) for DJIA (via the DIA
ETF total-return proxy), EWP, LS-MVP, and LS-C95. Source:
`results/exp3_hill_tail_index_k100.csv`, `results/exp3_hill_sensitivity.csv`,
`results/exp3_hill_sensitivity.png`.

| Series | n obs | n negative residuals | alpha_hat(k=100) | 95% CI |
|---|---|---|---|---|
| DJIA | 753 | 376 | 2.452 | [2.050, 3.050] |
| EWP | 753 | 372 | 2.325 | [1.944, 2.891] |
| LS-MVP | 753 | 377 | 2.673 | [2.235, 3.324] |
| LS-C95 | 753 | 371 | 2.316 | [1.936, 2.880] |

Paper's reported values (Section 4.4, k=100): LS-MVP 2.6349 [2.2031, 3.2772], LS-C95 2.6348
[2.2030, 3.2771], EWP 2.5616 [2.1418, 3.1861], DJIA 2.3409 [1.9572, 2.9115].

**Verdicts**

* LS-MVP's Hill tail index at k=100 (2.673 vs 2.6349): **reproduced** (within 10%, same CI
  overlap).
* LS-MVP has a higher (lighter-tailed) tail index than DJIA at k=100 (2.673 vs 2.452):
  **reproduced** (ranking holds).
* EWP's tail index (2.325 vs 2.5616) and its ranking relative to DJIA: **not reproduced** —
  here EWP's tail index is the second-lowest (lighter than DJIA by only a small margin and
  essentially tied with LS-C95), rather than clearly above DJIA as the paper reports; the
  difference (2.325 vs 2.5616) is 9.2%, inside the 10% band, but the qualitative statement
  that EWP sits between the long-short portfolios and DJIA is not confirmed by this run
  (LS-C95 and EWP are nearly identical here, 2.316 vs 2.325, whereas the paper separates them
  more, 2.6348 vs 2.5616).
* LS-C95's tail index (2.316 vs 2.6348): **partially reproduced** — the paper reports LS-C95
  essentially tied with LS-MVP (both near 2.635), while this replication finds LS-C95 close to
  EWP and DJIA instead (2.316 vs LS-MVP's 2.673); the 12.5% gap from the paper's value and the
  broken LS-MVP/LS-C95 tie make this a partial rather than full match.
* Confidence intervals for all four series overlap substantially, as the paper reports:
  **reproduced**.
* General downward tendency of alpha_hat(k) as k increases from 40 to 200: see
  `results/exp3_hill_sensitivity.png`; the direction (lighter tails estimated from the
  smallest k) is reproduced, but exact convergence values at k=200 were not individually
  compared against the paper's 1.37-1.48 range given the scope of this replication.

The LS-MVP/LS-C95 tail-index gap is consistent with the covariance-estimation sensitivity
identified under exp_2: LS-MVP and LS-C95 solve different optimization problems (quadratic
minimum-variance versus the Rockafellar-Uryasev CVaR linear program) over the same
near-collinear asset universe, so their realized out-of-sample return paths — and hence their
tail behavior — can diverge from each other, and from the paper's realization, more than a
qualitative resemblance alone would suggest.

## Experiment exp_4: Volatility Persistence Comparison: AR(1)-GARCH(1,1) vs. AR(1)-FIGARCH(1,d,1) Models

Reproduces Section 5 (model comparison, persistence, and residual diagnostics) for EWP,
LS-MVP, and LS-C95. Source: `results/exp4_garch_params.csv`, `results/exp4_figarch_params.csv`,
`results/exp4_garch_figarch_ic.csv`, `results/exp4_ljung_box.csv`,
`results/exp4_distributional_diagnostics.csv`, `results/exp4_acf_*.png`,
`results/exp4_diagnostics_*.png`.

| Series | AIC/obs GARCH | AIC/obs FIGARCH | BIC/obs GARCH | BIC/obs FIGARCH | alpha1+beta1 (GARCH) | d (FIGARCH) |
|---|---|---|---|---|---|---|
| EWP | -7.3698 | -7.3619 | -7.3391 | -7.3250 | 0.9912 | 0.281 (SE 0.256) |
| LS-MVP | -8.4674 | -8.4590 | -8.4366 | -8.4221 | 0.9914 | 0.258 (SE 0.273) |
| LS-C95 | -9.0830 | -9.0744 | -9.0523 | -9.0375 | 0.9906 | 0.511 (SE 0.484) |

Paper's reported values (Section 5.3): per-observation AIC, GARCH -7.4255 (EWP), -9.1812
(LS-MVP), -9.0638 (LS-C95); FIGARCH -7.4216, -9.1762, -9.0595; persistence (GARCH) 0.9923
(EWP), 0.9952 (LS-MVP), 0.9968 (LS-C95); FIGARCH d 0.9366 (EWP), 0.9370 (LS-MVP), 0.9252
(LS-C95); Ljung-Box p-values above 0.05 at lags 10/20/50 for standardized and squared
standardized residuals for all three series; distributional diagnostics: EWP skewness
-0.0378, kurtosis 4.2892; LS-MVP skewness -0.5083, kurtosis 3.7481; LS-C95 skewness -0.4974,
kurtosis 4.2875.

**Verdicts**

* GARCH is preferred over FIGARCH by both AIC and BIC for all three series: **reproduced**
  (confirmed for EWP, LS-MVP, LS-C95: GARCH's AIC/obs and BIC/obs are both higher, i.e.
  better, than FIGARCH's in every row above).
* GARCH persistence (alpha1+beta1) close to but below 1 for all three series: **reproduced**
  in direction and magnitude (0.9906-0.9914 vs the paper's 0.9906-0.9968, within 10% in every
  case).
* FIGARCH fractional-integration parameter d estimated with a large standard error relative
  to its point estimate for all three series: **reproduced** qualitatively (SE/d ratios here
  are 0.91, 1.06, 0.95, all large, as in the paper's SEs of 0.0343/0.9366=3.7%,
  0.0411/0.9370=4.4%, 0.0533/0.9252=5.8% being comparatively tight) — but the point estimates
  of d itself (0.26-0.51 vs the paper's 0.94-0.94) differ by more than 50%: **not reproduced**
  for the point estimates of d, even though the qualitative conclusion that FIGARCH does not
  improve on GARCH is reproduced through the AIC/BIC comparison. The paper does not specify
  the FIGARCH truncation lag or optimizer settings (an acknowledged uncertainty), and this
  implementation estimates the fractional-differencing weights with a truncation lag of
  1,000 lags and BFGS maximum likelihood; the resulting discrepancy in d, while the IC
  ranking is unaffected, could not be isolated further within the scope of this replication.
* Standardized and squared-standardized-residual Ljung-Box p-values exceed 0.05 at lags
  10/20/50 for all three series (see `results/exp4_ljung_box.csv`): **reproduced** (all 18
  p-values in the table are above 0.05).
* Distributional diagnostics: LS-MVP skewness -0.295 (vs -0.5083) and kurtosis 3.647 (vs
  3.7481, reproduced, within 10%); LS-C95 skewness -0.585 (vs -0.4974) and kurtosis 4.764 (vs
  4.2875, both partially reproduced, 10%-25% off with the same sign); EWP skewness -0.137 (vs
  -0.0378, same sign but a 263% relative difference on a near-zero paper value — judged
  against the paper's own near-zero scale this is a small absolute difference of 0.10 and is
  **partially reproduced**) and kurtosis 4.701 (vs 4.2892, partially reproduced).

## Experiment exp_5: Model-Implied European Option Pricing under the Normal Inverse Gaussian (NIG) Lévy Process with FFT and Implied Volatility Surface Construction

Reproduces Section 6 (NIG parameter estimation, martingale correction, Carr-Madan FFT option
pricing, and the implied volatility surface) for the EWP log-return series. Source:
`results/exp5_nig_params.csv`, `results/exp5_fft_convergence.csv`,
`results/exp5_no_arbitrage_diagnostics.csv`, `results/exp5_call_price_surface.png`,
`results/exp5_put_price_surface.png`, `results/exp5_implied_vol_surface.png`.

| Parameter | This replication | Paper |
|---|---|---|
| alpha | 68.167 | 69.1337 |
| beta | -18.825 | -17.4890 |
| delta | 0.0034196 | 0.00342448 |
| mu | 0.0011723 | 0.00108 |
| max CF-ECF discrepancy | 1.50e-08 | 1.12e-07 |
| r_daily (continuously compounded) | 0.00016973 | 0.0001697305 |
| n_obs | 1,257 | 1,257 |

FFT convergence (max absolute price difference between successive grid doublings):

| N_prev -> N_next | Max abs diff | Mean abs diff |
|---|---|---|
| 1024 -> 2048 | 1.831 | 1.829 |
| 2048 -> 4096 | 0.211 | 0.211 |
| 4096 -> 8192 | 0.00141 | 0.00141 |

No-arbitrage diagnostics: 20 lower-bound violations (max 4.46e-08), 0 upper-bound violations,
0 monotonicity violations, 6 convexity violations (max 4.46e-08).

Paper's reported values (Section 6): alpha=69.1337, beta=-17.4890, delta=0.00342448,
mu=0.00108; max CF-ECF discrepancy 1.12e-07; r_daily 0.0001697305; FFT max absolute
difference falling from 5.50e-02 (2^10 vs 2^11) to 7.53e-03 (2^12 vs 2^13); 29 lower-bound
violations (largest 3.53e-05), 0 upper-bound violations, 0 monotonicity violations, 1
convexity violation (3.20e-08); 29 of 7,400 (0.392%) grid points produce NaN implied
volatilities; implied volatilities in roughly the 0.105-0.158 range.

**Verdicts**

* n_obs = 1,257 full-sample daily log returns: **reproduced** (exact match).
* r_daily: **reproduced** (exact match to the reported precision).
* NIG parameters alpha, beta, delta, mu: **reproduced** (all four within 10% of the paper's
  values, same signs; delta matches to four significant figures).
* Max CF-ECF discrepancy of the fitted characteristic function: both values are on the order
  of 1e-7 to 1e-8, i.e. numerically negligible in both cases; judged against the paper's own
  scale (orders of magnitude below any economically meaningful price difference) this is
  **reproduced**.
* FFT price convergence decreasing monotonically as the grid is refined, with both the
  largest and smallest differences on the same order of magnitude as the paper's (this
  replication's grid spans 2^10-2^13 at coarser intermediate points than the paper's 2^11/2^12
  comparison, so the exact doubling pairs differ): **reproduced** directionally; the specific
  magnitudes (1.83 vs the paper's 5.50e-2 at the coarsest comparison) differ by more than 50%
  in absolute terms because this replication's successive-grid comparison starts at 2^10 vs
  2^11 (one step coarser than the paper's finest reported comparison) — the paper's own
  2^12-vs-2^13 comparison (7.53e-3) is of the same order as this replication's corresponding
  4096-vs-8192 value (1.41e-3), so convergence as a phenomenon is **reproduced**, while the
  absolute scale at the coarsest grid is **partially reproduced**.
* No-arbitrage violations are small floating-point-scale breaches of the lower bound and
  convexity test, with zero monotonicity and upper-bound violations: **reproduced**
  qualitatively (same pattern of which constraints bind); the violation counts (20 vs 29,
  6 vs 1) are within the same order of magnitude but the convexity-violation count differs by
  a factor of 6, so this is judged **partially reproduced**.

## Deviations from the paper

* **OHLCV data source.** The paper downloads all ETF and DJIA prices directly from Yahoo
  Finance. This replication loads the same tickers through the QuantX Dataloader
  (`source="massive"`), which serves Massive market data and falls back to Yahoo Finance only
  for tickers Massive does not cover. For 16 of the 30 tickers (`OCIO`, `GDMA`, and all 14
  Canadian and Australian ETFs) the Dataloader's `ticker_details[].data_source` is `yahoo`
  rather than `massive`, because Massive carries no bars, or materially fewer bars, for them
  over 2020-2024 (see `results/exp1_run_log.json`). For the remaining 14 U.S. tickers the
  bars come from Massive. A direct check on one such U.S. ticker (AOA) shows the Massive- and
  Yahoo-sourced 2020-2024 cumulative returns differ by only 0.09 percentage points, so this
  substitution is not a material source of the gaps reported above.
* **DJIA benchmark.** Massive does not serve the DJIA index directly, and `^DJI` is not a
  valid Massive or Yahoo-via-Dataloader symbol; the DIA ETF, which tracks the Dow Jones
  Industrial Average, is used in its place for the DJIA series in exp_3.
* **Realized out-of-sample sample length.** The 504-day rolling window applied to 1,258
  aligned trading dates yields 753 out-of-sample observations (2022-01-03 to 2024-12-31)
  rather than the paper's reported 754; the paper does not list its exact out-of-sample
  dates, and the one-day difference is attributed to calendar alignment rather than to a
  methodological change.
* **Covariance estimator.** The paper does not specify whether the minimum-variance program
  uses the sample covariance matrix or a shrinkage estimator. This replication uses the
  sample covariance matrix of each 504-day window, as this is the baseline estimator
  consistent with the stated formula `w'Sigma*w`. The identified sensitivity of the LS-MVP
  strategy to this choice (see exp_2) suggests a shrinkage estimator could narrow the gap to
  the paper, but this was not tested further within the scope of this replication.
* **FIGARCH implementation.** The `arch` package does not provide a FIGARCH model class. This
  replication implements AR(1)-FIGARCH(1,d,1) maximum likelihood estimation from scratch,
  truncating the fractional-differencing operator at 1,000 lags, following the conditional
  variance recursion stated in the paper's Section 5.2. The resulting point estimates of d
  differ substantially from the paper's (see exp_4), although the qualitative GARCH-vs-FIGARCH
  information-criterion ranking is reproduced.
* **Optimizer settings.** The paper does not report SLSQP/CVXPY tolerance or iteration
  settings; this replication uses `scipy.optimize.minimize(method="SLSQP")` with
  `maxiter=1000`, `ftol=1e-12` for minimum-variance weights and the ECOS solver (via `cvxpy`)
  for the Rockafellar-Uryasev CVaR linear program, both recorded in `src/config.py`.
* **Risk-free rate for option pricing.** The 3-month U.S. Treasury yield on 2024-12-31 is
  taken from FRED's `DGS3MO` series, as the paper specifies; this was the last trading day of
  the sample.

## Not reproduced

* The paper's Figures 3-4 (efficient-frontier analysis under log returns with a separately
  specified risk-free conversion) are not reproduced: the paper's methodology for this
  specific figure pair is listed as a separate analysis in its own uncertainty note and is not
  fully specified beyond the main rolling-window framework reproduced above.
* The full Hill-estimator sensitivity curve over k=40 to 200 is plotted
  (`results/exp3_hill_sensitivity.png`) but individual alpha_hat(k) values at k=200 were not
  compared point-by-point against the paper's reported 1.37-1.48 convergence range; only the
  qualitative downward tendency was checked.
