# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""NEFUSI — interpretable neuro-fuzzy aggregation of semantic similarity measures.

NEFUSI learns a transparent Mamdani fuzzy inference system that aggregates an
arbitrary number of base semantic-similarity measures into a single score that
maximises correlation with human judgements, while remaining fully explainable
(per-rule firing, membership degrees, per-measure contributions, uncertainty).

Quick start
-----------
>>> from nefusi import NeuroFuzzyAggregator, load_builtin
>>> gold, X = load_builtin("mc-training")
>>> model = NeuroFuzzyAggregator(random_state=0).fit(X, gold)
>>> round(model.score(X, gold, "pearson"), 3)  # doctest: +SKIP
0.87
>>> print(model.explain(X[0]).format())         # doctest: +SKIP
"""
from .aggregator import NeuroFuzzyAggregator
from .fis import Explanation, MamdaniFIS, Rule
from .membership import partition_from_breakpoints, trapmf, trimf
from .metrics import (
    bootstrap_ci,
    evaluate_all,
    kendall_tau,
    mae,
    mse,
    pearson,
    rmse,
    spearman,
)
from .optimize import differential_evolution
from .datasets import (
    BUILTIN_DATASETS,
    list_datasets,
    load_builtin,
    load_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "NeuroFuzzyAggregator",
    "MamdaniFIS",
    "Rule",
    "Explanation",
    "trapmf",
    "trimf",
    "partition_from_breakpoints",
    "pearson",
    "spearman",
    "kendall_tau",
    "mse",
    "rmse",
    "mae",
    "evaluate_all",
    "bootstrap_ci",
    "differential_evolution",
    "load_dataset",
    "load_builtin",
    "list_datasets",
    "BUILTIN_DATASETS",
    "__version__",
]
