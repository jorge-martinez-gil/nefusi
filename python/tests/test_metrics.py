# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from nefusi import metrics


def test_pearson_perfect_and_negative():
    a = np.array([1.0, 2, 3, 4, 5])
    assert metrics.pearson(a, a) == pytest.approx(1.0)
    assert metrics.pearson(a, -a) == pytest.approx(-1.0)


def test_pearson_known_value():
    # affine transform => correlation 1
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = 2 * a + 0.5
    assert metrics.pearson(a, b) == pytest.approx(1.0)


def test_spearman_is_monotone_invariant():
    a = np.array([1.0, 2, 3, 4, 5])
    b = np.exp(a)  # monotone increasing
    assert metrics.spearman(a, b) == pytest.approx(1.0)


def test_rank_data_handles_ties():
    r = metrics.rank_data([10.0, 10.0, 20.0])
    assert list(r) == [1.5, 1.5, 3.0]


def test_kendall_tau_bounds_and_perfect():
    a = np.array([1.0, 2, 3, 4])
    assert metrics.kendall_tau(a, a) == pytest.approx(1.0)
    assert metrics.kendall_tau(a, a[::-1]) == pytest.approx(-1.0)


def test_error_metrics():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    assert metrics.mse(a, b) == pytest.approx(1.0)
    assert metrics.rmse(a, b) == pytest.approx(1.0)
    assert metrics.mae(a, b) == pytest.approx(1.0)


def test_evaluate_all_keys():
    a = np.random.default_rng(0).random(20)
    b = a + 0.01 * np.random.default_rng(1).random(20)
    out = metrics.evaluate_all(a, b)
    assert set(out) == {"pearson", "spearman", "kendall", "mse", "rmse", "mae"}


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    a = rng.random(40)
    b = a + 0.05 * rng.standard_normal(40)
    point, lo, hi = metrics.bootstrap_ci(a, b, "pearson", n_boot=300, seed=0)
    assert lo <= point <= hi
    assert -1.0 <= lo <= hi <= 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.pearson([1, 2, 3], [1, 2])
