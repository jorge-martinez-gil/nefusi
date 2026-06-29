# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from nefusi.fis import MamdaniFIS, Rule
from nefusi.membership import partition_from_breakpoints


@pytest.fixture
def fis():
    part = partition_from_breakpoints([0.2, 0.4, 0.3, 0.45, 0.55, 0.7, 0.6, 0.8])
    rules = [
        Rule({0: "high"}, "high"),
        Rule({1: "high"}, "high"),
        Rule({0: "medium"}, "medium"),
        Rule({1: "medium"}, "medium"),
        Rule({0: "low"}, "low"),
        Rule({1: "low"}, "low"),
    ]
    return MamdaniFIS(2, part, rules, measure_names=["a", "b"], n_points=201)


def test_monotonic_aggregation(fis):
    low = fis.infer([0.0, 0.0])
    mid = fis.infer([0.5, 0.5])
    high = fis.infer([1.0, 1.0])
    assert low < mid < high


def test_output_in_unit_interval(fis):
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.random(2)
        assert 0.0 <= fis.infer(x) <= 1.0


def test_disabled_rule_is_inactive():
    part = partition_from_breakpoints([0.2, 0.4, 0.3, 0.45, 0.55, 0.7, 0.6, 0.8])
    rules = [Rule({0: "high"}, "high"), Rule({1: "low"}, None)]
    fis = MamdaniFIS(2, part, rules, n_points=51)
    assert len(fis.active_rules()) == 1


def test_predict_batch_shape(fis):
    X = np.random.default_rng(1).random((7, 2))
    assert fis.predict(X).shape == (7,)


def test_explain_contains_all_fields(fis):
    exp = fis.explain([0.9, 0.1])
    assert 0.0 <= exp.score <= 1.0
    assert set(exp.memberships[0]) == {"low", "medium", "high"}
    assert 0.0 <= exp.confidence <= 1.0
    assert 0.0 <= exp.uncertainty <= 1.0
    total = sum(exp.measure_contributions.values())
    assert total == pytest.approx(1.0, abs=1e-6) or total == 0.0
    d = exp.to_dict()
    assert "active_rules" in d and "measure_contributions" in d
    assert isinstance(exp.format(), str)


def test_defuzz_methods_agree_on_direction():
    part = partition_from_breakpoints([0.2, 0.4, 0.3, 0.45, 0.55, 0.7, 0.6, 0.8])
    rules = [Rule({0: "high"}, "high"), Rule({0: "low"}, "low")]
    for method in MamdaniFIS.DEFUZZ_METHODS:
        fis = MamdaniFIS(1, part, rules, n_points=201, defuzz=method)
        assert fis.infer([0.95]) >= fis.infer([0.05])


def test_to_fcl_roundtrip_text(fis):
    fcl = fis.to_fcl()
    assert "FUNCTION_BLOCK" in fcl
    assert "RULEBLOCK" in fcl
    assert "score IS high" in fcl


def test_unknown_defuzz_raises():
    part = partition_from_breakpoints([0.2, 0.4, 0.3, 0.45, 0.55, 0.7, 0.6, 0.8])
    with pytest.raises(ValueError):
        MamdaniFIS(1, part, [], defuzz="nope")
