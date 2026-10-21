import numpy as np
import pytest

from quotelab.analytics import bootstrap_ci, holm_bonferroni, max_drawdown, ols_hac, paired_test
from quotelab.calibration import fit_fill_curve


def test_ols_hac_recovers_slope_and_widens_errors_under_autocorrelation():
    rng = np.random.default_rng(0)
    n = 20_000
    x = np.zeros(n)
    u = np.zeros(n)
    for i in range(1, n):  # persistent regressor and AR(1) errors
        x[i] = 0.9 * x[i - 1] + rng.standard_normal()
        u[i] = 0.9 * u[i - 1] + rng.standard_normal()
    y = 0.5 + 2.0 * x + u
    res = ols_hac(x, y, lags=40)
    assert res["beta"][1] == pytest.approx(2.0, abs=0.1)
    assert abs(res["t"][1]) < abs(res["naive_t"][1])  # naive errors are too small


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(1).normal(3.0, 1.0, 500)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and lo < 3.0 < hi


def test_paired_test_detects_shift_under_common_noise():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 100, 200)  # large shared market noise
    a = common + 1.0 + rng.normal(0, 1, 200)
    b = common + rng.normal(0, 1, 200)
    res = paired_test(a, b)
    assert res["p_value"] < 1e-6 and res["variance_reduction"] > 0.99


def test_holm_bonferroni():
    adj = holm_bonferroni([0.01, 0.04, 0.03, 0.2])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_max_drawdown():
    assert max_drawdown(np.array([0, 5, 3, 7, 1, 8])) == 6
    assert max_drawdown(np.array([1, 2, 3])) == 0


def test_fill_curve_mle_recovers_parameters():
    rng = np.random.default_rng(3)
    A, k = 1.3, 1.9
    delta = np.array([0.5, 1.5, 2.5, 3.5])
    exposure = np.full(4, 50_000.0)
    fills = rng.poisson(A * np.exp(-k * delta) * exposure)
    a_hat, k_hat = fit_fill_curve(delta, fills, exposure)
    assert a_hat == pytest.approx(A, rel=0.03) and k_hat == pytest.approx(k, rel=0.03)
