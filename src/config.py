"""Paper-derived parameters for every experiment.

Source: Platen & Fergusson (2026), "Information-Theoretic Model for the
Growth Optimal Portfolio of Stocks", arXiv:2611.07457v1 [q-fin.PR].
All module-level constants carry a comment stating their meaning, unit and
the paper section that specifies them (or "not specified in the paper").
"""

# --------------------------------------------------------------------------
# Experiment 1: Construction of Equal-Weighted Indices (EWIs)
# --------------------------------------------------------------------------

# Proportional transaction cost per US dollar of turnover (Sec. 4.1, kappa-bar)
TRANSACTION_COST_KBAR = 0.004  # 40 basis points

# GICS classification depths used by the paper (Sec. 4.1): sectors, industry
# groups, industries, sub-industries, and the degenerate single global level.
EWI_LEVELS_PAPER = (1, 11, 25, 74, 163)

# Trading days per year used to annualise growth rates and volatilities
# (Sec. 4.1: "the average calendar time step size Delta corresponds to
# approximately 1/262").
TRADING_DAYS_PER_YEAR = 262

# Paper's sample window for the Thomson-Reuters GICS index data (Sec. 4.1).
SAMPLE_START_PAPER = "1973-01-01"
SAMPLE_END_PAPER = "2024-07-31"

# Country universe size used by the paper (Sec. 4.1): ell = 67 countries.
COUNTRY_UNIVERSE_SIZE_PAPER = 67

# The paper's Thomson-Reuters GICS sub-index panel for 67 countries at four
# classification depths (1973-2024) is a proprietary dataset not obtainable
# through the QuantX Dataloader (OHLCV bars only) or the Massive REST API
# (no international GICS-level cap-weighted sub-indices by country).
# The executed run substitutes exchange-traded index funds that are
# themselves cap-weighted indices, at three classification resolutions:
#  - single-country MSCI index ETFs for the d=1 (country-only) level,
#  - US GICS Select Sector SPDR ETFs for the d=11 (sector) level,
#  - US GICS industry-level SPDR ETFs for the highest resolution level.
# See data/data_loader.py for the exact tickers and results/RESULTS.md for
# the "Deviations from the paper" entry this substitution requires.
SAMPLE_START_RUN = "2011-01-01"  # earliest date with broad SPDR industry-ETF coverage
SAMPLE_END_RUN = "2024-07-31"  # paper's stated end date (Sec. 4.1)

# --------------------------------------------------------------------------
# Experiment 2: Activity time estimation (MMM parametrization, Sec. 2.1)
# --------------------------------------------------------------------------

# Grid search range and step for the initial activity time tau_{t0} used to
# maximise the R^2 of the linear trendline fitted to the activity time
# trajectory (Sec. 2.1; exact grid not specified in the paper -- the range
# below covers the paper's reported estimates of -3.6 (MSCI) and -6.9
# (EWI163) with margin).
TAU_T0_GRID_MIN = -10.0
TAU_T0_GRID_MAX = 0.0
TAU_T0_GRID_STEP = 0.01

# Paper's reported MMM-parametrization fit results (Sec. 2.1, p.7-8), used
# only as the comparison targets in results/RESULTS.md, not as inputs.
MSCI_TAU_T0_PAPER = -3.6
MSCI_TRENDLINE_SLOPE_PAPER = 0.065
MSCI_TRENDLINE_R2_PAPER = 0.993
EWI163_TAU_T0_PAPER = -6.9
EWI163_TRENDLINE_SLOPE_PAPER = 0.171
EWI163_TRENDLINE_R2_PAPER = 0.976

# --------------------------------------------------------------------------
# Experiment 3: Normalized activity simulation (Sec. 3.1-3.2)
# --------------------------------------------------------------------------

# Smoothing parameter kappa in (1,2) capturing overreactions (Sec. 3.1,
# Fig. 3.6-3.8 caption), estimated by the paper from the EWI163.
CIR_KAPPA = 1.97

# Scaling parameter xi of the normalized activity (Sec. 3.2, Fig. 3.7-3.8).
CIR_XI = 0.0042

# Constant average activity a-bar (Sec. 3.1).
CIR_A_BAR = 0.15

# Calendar time step size Delta = 1/262 for daily data (Sec. 4.1).
CIR_DELTA = 1.0 / TRADING_DAYS_PER_YEAR

# Simulation length: "approximately 50 years of daily data" (Sec. 3.2,
# Fig. 3.7-3.8; exact step count not specified in the paper).
CIR_SIMULATION_YEARS_PAPER = 50
CIR_SIMULATION_STEPS_PAPER = int(CIR_SIMULATION_YEARS_PAPER * TRADING_DAYS_PER_YEAR)
# Reduced step count actually executed to keep the reproduced run inside the
# available computation time (see results/RESULTS.md); use --full for the
# paper configuration.
CIR_SIMULATION_STEPS_RUN = CIR_SIMULATION_STEPS_PAPER

# Random seed for reproducibility (not specified in the paper).
CIR_RANDOM_SEED = 20261106

# --------------------------------------------------------------------------
# Experiment 4: SGH log-return density estimation and LR testing (Sec. 5)
# --------------------------------------------------------------------------

# Chi-squared critical values with 1 degree of freedom (Sec. 5.3), used to
# judge the likelihood-ratio statistic L = -2 ln(Lambda) of each 3-parameter
# nested density (variance-gamma, hyperbolic, NIG, Student t) against the
# 4-parameter SGH density.
CHI2_CRITICAL_99_DF1 = 6.635
CHI2_CRITICAL_95_DF1 = 3.841
CHI2_CRITICAL_90_DF1 = 2.706
LR_TEST_SIGNIFICANCE_LEVEL = 0.01  # Sec. 5.3: reject at the 1% level

# Degrees of freedom used in the chi-squared test when the 2-parameter
# normal density is compared against the 4-parameter SGH density.
CHI2_DF_NORMAL_VS_SGH = 2

# Optimizer used for maximum likelihood estimation (not specified in the
# paper beyond "maximum likelihood method", Sec. 5.3); Nelder-Mead is
# derivative-free and robust to the Bessel-function singularities of the
# SGH log-likelihood.
MLE_OPTIMIZER_METHOD = "Nelder-Mead"
MLE_MAX_ITER = 20000
