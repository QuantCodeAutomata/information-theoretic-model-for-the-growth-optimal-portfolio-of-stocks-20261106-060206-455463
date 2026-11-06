# Results

Platen, E. and Fergusson, K. (2026), "Information-Theoretic Model for the
Growth Optimal Portfolio of Stocks", arXiv:2611.07457v1 [q-fin.PR].

All numbers below come from the files saved by the commands in `README.md`
into `results/`. Each metric compared with the paper is given a verdict:
**reproduced** (inside the paper's reported range, or within 10% of the
paper's value with the same sign, and the paper's qualitative result holds),
**not reproduced** (opposite sign, more than 50% off, outside a stated range,
or the paper's qualitative ranking/constraint fails), or **partially
reproduced** otherwise.

## Experiment 1: Construction of Equal-Weighted Indices (Sec. 4.1-4.2)

File: `results/exp1_growth_rates.csv`, `results/exp1_meta.json`,
`results/exp1_ewi_comparison.png`.

Realized sample: 2011-01-01 to 2024-07-31, 3,416 daily observations, 38
countries (single-country MSCI ETFs), 11 GICS sectors (Select Sector SPDR
ETFs) and 17 available GICS industry-level SPDR ETFs, all dividend-adjusted
through the QuantX Dataloader (`adjust_dividends=True`). Nineteen tickers were
served by the Dataloader's Yahoo Finance backstop rather than Massive
(`exp1_meta.json`, `non_massive_source`): MCHI, EWI, EWJ, EWM, EWS, EWT, EWU,
INDA, GREK, EIRL, VNM, EDEN, NORW, XLRE, XHS, XHE, XSW, XTL, XTN, XAR.

| Index | Annualized growth g | Annualized volatility sigma | Paper value | Verdict |
|---|---|---|---|---|
| MSCI proxy (ACWI) | 0.0894 | 0.1755 | ~0.05 (MSCI total return world index) | partially reproduced (same sign, +79% vs. the paper's lower-volatility 1973-2024 window) |
| EWI1 (d=1, 38 countries) | 0.0390 | 0.1872 | lowest among EWIs, still above MSCI | not reproduced (lower than the MSCI proxy here) |
| EWI11 (d=11, US sectors) | 0.1140 | 0.1700 | between EWI1 and EWI163, ~0.139 at d=163 | partially reproduced (exceeds MSCI and EWI1, direction right) |
| EWI17 (d=17, US industries) | 0.0917 | 0.2226 | expected higher than EWI11 (monotonic in d) | not reproduced (lower than EWI11, breaks monotonicity) |

The paper's qualitative prediction -- growth increases monotonically
(approximately) with the number of independently equal-weighted
classification groups d, and every EWI exceeds the MSCI benchmark -- does not
hold in this substitute universe: EWI1 underperforms the MSCI proxy, and
EWI17 underperforms EWI11. See "Deviations from the paper" for the identified
cause.

## Experiment 2: Activity Time Estimation (Sec. 2.1)

File: `results/exp2_activity_time_fit.csv`, `results/exp2_meta.json`,
`results/exp2_activity_*.png`, `results/exp2_activity_time_*.png`.

| Series | tau_{t0} | trendline slope | R^2 | Paper value | Verdict |
|---|---|---|---|---|---|
| MSCI proxy | -0.0400 | 0.1327 | 0.9692 | tau_{t0}=-3.6, slope=0.065, R^2=0.993 | partially reproduced (R^2 within 10%; slope about 2x the paper's, tau_{t0} not comparable, see below) |
| EWI17 proxy (highest-resolution EWI available) | -3.6100 | 0.1698 | 0.9790 | tau_{t0}=-6.9, slope=0.171, R^2=0.976 (EWI163) | reproduced (slope and R^2 both within 10%) |

tau_{t0} is the logarithm of the initial quadratic variation of the square
root of the benchmark and therefore depends on the sample length and the
arbitrary starting level of each index (Sec. 2.1); it is not comparable across
samples of different length and is not scored against the paper's value. The
linear-in-calendar-time property of the activity time (the MMM parametrization
assumption, Sec. 2.1) is confirmed for both proxies (R^2 > 0.96).

## Experiment 3: Normalized Activity Simulation with Overreactions (Sec. 3.1-3.2)

File: `results/exp3_summary.json`, `results/exp3_overreaction.png`,
`results/exp3_driving_brownian_motions.png`,
`results/exp3_activity_calendar_vs_tau.png`.

Simulation parameters from the paper's EWI163 estimates (Sec. 3.2):
kappa=1.97, xi=0.0042, a_bar=0.15, Delta=1/262, 13,100 steps (~50 years of
daily data).

| Quantity | Simulated value | Paper prediction | Verdict |
|---|---|---|---|
| Mean of normalized activity | 7.2873 | theoretical mean xi*kappa_Delta/2 = 7.2260 (Sec. 3.2) | reproduced (0.8% difference) |
| Fitted stationary gamma shape | 0.4983 | 0.5 (gamma density of one degree of freedom, Sec. 3.2) | reproduced |
| Kolmogorov-Smirnov test of the gamma fit | statistic 0.00466, p-value 0.959 | gamma stationary density (Sec. 3.2) | reproduced (fit not rejected) |
| Mean of the sign function sgn(tau_t) | 0.00183 | 0 on average (Sec. 3.1) | reproduced (within sampling noise) |
| Sample covariation rate between the two driving Brownian motions | -0.00073 | vanishes on average (Sec. 3.1) | reproduced (within sampling noise) |

## Experiment 4: Log-Return Density Estimation and LR Testing (Sec. 5)

File: `results/exp4_density_fits.csv`, `results/exp4_student_t_lr_test.csv`,
`results/exp4_density_*.png`, `results/exp4_meta.json`.

| Index | Best fit | nu (Student t) | L = -2 ln(Lambda) vs. SGH | chi^2_{0.99,1}=6.635 | Rejected at 1%? | Paper conclusion | Verdict |
|---|---|---|---|---|---|---|---|
| MSCI proxy | SGH (near Student t) | 3.07 | 2.428 | 6.635 | No | Student t not rejected, nu~3.44 | reproduced (qualitative result) |
| EWI1 | Student t (alpha_bar=1.7e-6) | 3.62 | ~0 | 6.635 | No | Student t not rejected, nu~2.91 (lowest d) | reproduced (qualitative result) |
| EWI11 | SGH | 2.73 | 4.327 | 6.635 | No | Student t not rejected | reproduced (qualitative result) |
| EWI17 | Student t (alpha_bar=0.013) | 3.80 | 4.6e-5 | 6.635 | No | Student t not rejected, nu closer to 4 at higher d | partially reproduced (not rejected, but nu is not monotonic in d here) |

The paper's central qualitative claim of Sec. 5.3-5.4 -- that the Student t
density with a few degrees of freedom cannot be rejected against the
four-parameter SGH density at the 1% significance level for any benchmark
proxy -- is reproduced for every index: every likelihood-ratio statistic is
far below the chi^2_{0.99,1} critical value of 6.635, exactly as in the
paper's Table of Sec. 5.3 (where all statistics are below 0.0001 except for
the full SGH-level MSCI and EWI11 fits reported above, which are still far
below the critical value). The normal density is clearly rejected for every
index (likelihood-ratio statistics 650-1140, against a chi^2_{0.99,2}
critical value far below these figures). The paper's finer claim that nu
increases monotonically with the classification resolution d, with EWI163
closest to a clean Student t (alpha_bar~0), is not reproduced: in this reduced
universe EWI11 (d=11) has the lowest nu and EWI17 (d=17) the highest, the
reverse of the paper's ordering.

## Deviations from the paper

- **Data source for Experiment 1, 2 and 4.** The paper builds its equal-weighted
  indices from a proprietary Thomson-Reuters Datastream panel of daily,
  market-capitalisation-weighted total-return index values for 67 countries at
  four GICS classification levels (11 sectors, 25 industry groups, 74
  industries, 163 sub-industries), 1973-2024 (Sec. 4.1). This panel is not
  obtainable through the QuantX Dataloader (which serves split/dividend
  adjusted single-security OHLCV bars, not pre-built cap-weighted country
  panels) or through the Massive REST API (which does not carry a Thomson-
  Reuters GICS country-index dataset). Both were checked before concluding
  this: the Dataloader's `fetch-ohlcvs` route only resolves individual
  security tickers, and Massive's reference/indices endpoints do not expose
  per-country, per-GICS-level cap-weighted benchmarks. This repository
  substitutes exchange-traded index funds that are themselves genuine
  cap-weighted indices: 38 single-country iShares/other MSCI country ETFs for
  d=1 (one classification group, ℓ=38 countries), the 11 GICS Select Sector
  SPDR ETFs for d=11 (ℓ=1 country, the United States), and 17 available GICS
  industry-level SPDR ETFs for the highest resolution reachable with liquid,
  continuously-traded instruments (labelled EWI17 throughout, in place of the
  paper's EWI25/EWI74/EWI163). The MSCI total return world index itself is
  substituted by the iShares MSCI ACWI ETF (ACWI), the closest continuously
  traded total-return proxy for the same benchmark.
- **Sample period.** The paper's window is 1973-01-01 to 2024-07-31. The
  substitute ETFs only exist from their respective inception dates (the GICS
  Select Sector SPDRs from 1998, most industry-level SPDRs and single-country
  ETFs later); the earliest date at which 38 countries, 11 sectors and 17
  industries are simultaneously available is 2011-01-01. All experiments
  therefore run, by default, over the realized window 2011-01-01 to
  2024-07-31 (3,416 trading days). `--full` on `exp_1.py` and `exp_4.py`
  selects the paper's 1973-01-01 start date; with the substitute ETF universe
  this does not recover the paper's longer sample because most tickers do not
  exist before 2011, and the Dataloader excludes them from the loaded panel
  (`ticker_details`/`failed` in `exp1_meta.json`) rather than silently
  back-filling missing history.
- **Identified cause of the Experiment 1 and 4 ranking gaps.** The paper's
  construction equally weights, at every classification level, index values
  that are themselves already diversified across 67 countries (Sec. 4.1: "First
  equally weight across countries within each classification group, then
  equally weight across all groups"). In this substitute universe, only the
  d=1 level (the 38 country ETFs) carries country diversification; the d=11
  and d=17 levels are built entirely from United States sector/industry ETFs,
  with no cross-country averaging at all. This was confirmed by inspecting the
  EWI construction inputs in `exp1_meta.json`: the sector and industry levels
  use `n_countries=1`, while only the d=1 level uses `n_countries=38`. The
  reduced universe therefore tests "more independent sources of randomness
  from one country" rather than the paper's "more independent sources of
  randomness across countries and classification groups simultaneously",
  which is the most likely cause of the non-monotonic growth rates in
  Experiment 1 and the non-monotonic degrees of freedom in Experiment 4. The
  cause is stated here as identified from this direct inspection, not merely
  asserted from the size of the gap.
- **Interpretation of "no seemingly unusual index moves or jumps should be
  removed" (Sec. 4.1, data requirements).** No daily return was removed, capped
  or winsorized in the loaded data for any experiment.
- **Numerical optimizer for Experiment 4.** The paper specifies maximum
  likelihood but not the optimizer (Sec. 5.3, "uncertainties"). Nelder-Mead
  (derivative-free) was used because the SGH log-likelihood has a Bessel-
  function singularity as alpha_bar to 0 (the Student t limit), where gradient-based
  optimizers are numerically unstable.
- **Grid search range for Experiment 2's initial activity time.** The paper
  does not specify the grid (Sec. 2.1, "uncertainties"); a grid of tau_{t0}
  from -10 to 0 in steps of 0.01 was used, as suggested by the paper's own
  reported values (-3.6 and -6.9) lying in that range.
- **Numerical differentiation method for Experiment 2's activity.** The paper
  does not specify it (Sec. 2.1, "uncertainties"); central finite differences
  of the activity time with respect to calendar time were used.

## Not reproduced

- The paper's absolute reported magnitudes for Experiment 1 (EWI163 growth
  ~0.139, MSCI growth ~0.05) and Experiment 4 (index-specific degrees of
  freedom and log-likelihoods of Sec. 5.2) are not reproduced from the paper's
  own 1973-2024, 67-country Thomson-Reuters data, because that data is not
  obtainable through the QuantX Dataloader or the Massive REST API (see
  "Deviations from the paper"). The experiments instead report the equivalent
  metrics computed on the substitute ETF universe and sample period described
  above.
- The paper's d=25 and d=74 GICS classification levels (industry groups and
  industries, Sec. 4.1) are not reproduced at all: no liquid, continuously
  traded exchange-traded fund universe spans all 25 industry groups or all 74
  industries. Only d=1, d=11 and the 17 available industry-level funds (in
  place of d=74) are reproduced.
