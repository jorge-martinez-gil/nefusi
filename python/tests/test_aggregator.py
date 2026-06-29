# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from nefusi import NeuroFuzzyAggregator, load_builtin
from nefusi.optimize import differential_evolution

# small, fast optimisation budget for the test-suite
FIT = dict(popsize=18, maxiter=12)


def _toy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    gold = X.mean(axis=1)  # learnable monotone aggregation target
    return X, gold


def test_fit_predict_shapes():
    X, gold = _toy_data()
    model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, **FIT)
    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.all((pred >= 0) & (pred <= 1))


def test_learns_positive_correlation():
    X, gold = _toy_data()
    model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, popsize=24, maxiter=25)
    assert model.score(X, gold, "pearson") > 0.6


def test_determinism_same_seed():
    X, gold = _toy_data()
    a = NeuroFuzzyAggregator(random_state=7).fit(X, gold, **FIT)
    b = NeuroFuzzyAggregator(random_state=7).fit(X, gold, **FIT)
    assert np.allclose(a.theta_, b.theta_)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        NeuroFuzzyAggregator(n_measures=3).predict(np.zeros((2, 3)))


def test_n_measures_mismatch_raises():
    X, gold = _toy_data()
    with pytest.raises(ValueError):
        NeuroFuzzyAggregator(n_measures=5).fit(X, gold, maxiter=2)


def test_save_load_roundtrip(tmp_path):
    X, gold = _toy_data()
    model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, **FIT)
    p = tmp_path / "model.json"
    model.save(str(p))
    loaded = NeuroFuzzyAggregator.load(str(p))
    assert np.allclose(model.predict(X), loaded.predict(X))


def test_explain_and_rules_after_fit():
    X, gold = _toy_data()
    model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, **FIT)
    exp = model.explain(X[0])
    assert hasattr(exp, "score")
    assert isinstance(model.describe(), str)
    assert "FUNCTION_BLOCK" in model.to_fcl()
    assert all(r.active for r in model.rules_)


def test_pairwise_scheme_builds_more_rules():
    single = NeuroFuzzyAggregator(n_measures=3, rule_scheme="single")
    pairwise = NeuroFuzzyAggregator(n_measures=3, rule_scheme="pairwise")
    assert len(pairwise._build_templates(3)) > len(single._build_templates(3))


def test_builtin_dataset_fit_smoke():
    gold, X = load_builtin("mc-training")
    model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, popsize=24, maxiter=20)
    assert model.score(X, gold, "pearson") > 0.5


def test_de_minimises_sphere():
    res = differential_evolution(
        lambda v: float(np.sum(v ** 2)),
        [(-5, 5)] * 4, seed=0, maxiter=80,
    )
    assert res.fun < 1e-2
