import numpy as np
from scipy import stats

from exp.exp_4 import analyze_series, log_returns


def test_log_returns_matches_definition():
    series = __import__("pandas").Series([100.0, 110.0, 99.0, 108.9])
    x = log_returns(series)
    expected = np.diff(np.log(series.to_numpy()))
    assert np.allclose(x, expected)


def test_analyze_series_student_t_not_rejected_for_fat_tailed_sample():
    """Regression test for Sec. 5.3-5.4: a Student t sample must not be
    rejected against the SGH nesting density at the 1% significance level,
    matching the paper's conclusion for every benchmark proxy."""
    rng = np.random.default_rng(42)
    x = stats.t.rvs(df=3.4, size=6000, random_state=rng) * 0.009
    result = analyze_series("synthetic", x)
    rows = {row["density"]: row for row in result["rows"]}
    assert rows["Student t"]["rejected_at_1pct"] is False
    assert rows["normal"]["lr_statistic"] > rows["Student t"]["lr_statistic"]
    # SGH must attain the highest log-likelihood among all nested cases.
    sgh_ll = rows["SGH"]["log_likelihood"]
    for key, row in rows.items():
        if key != "SGH":
            assert sgh_ll >= row["log_likelihood"] - 1e-3
