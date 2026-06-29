# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from nefusi import datasets


def test_list_and_load_all_builtins():
    keys = datasets.list_datasets()
    assert "mc-training" in keys
    for key in keys:
        gold, X = datasets.load_builtin(key)
        assert gold.ndim == 1
        assert X.ndim == 2
        assert X.shape[0] == gold.shape[0]
        assert X.shape[0] > 0


def test_measure_counts_match_registry():
    for key, info in datasets.BUILTIN_DATASETS.items():
        gold, X = datasets.load_builtin(key)
        assert X.shape[1] == info.n_measures


def test_values_normalised_range():
    gold, X = datasets.load_builtin("mc-training")
    assert gold.min() >= 0.0 and gold.max() <= 1.0
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        datasets.load_builtin("does-not-exist")


def test_load_custom_file(tmp_path):
    p = tmp_path / "toy.csv"
    p.write_text("0.5, 0.1, 0.2\n0.8, 0.7, 0.9\n\n# comment line ignored\n")
    gold, X = datasets.load_dataset(str(p))
    assert gold.shape == (2,)
    assert X.shape == (2, 2)
    assert np.allclose(gold, [0.5, 0.8])
