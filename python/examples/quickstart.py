# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""NEFUSI quickstart: aggregate similarity measures in ~10 lines.

Run:  python examples/quickstart.py
"""
from nefusi import NeuroFuzzyAggregator, load_builtin

# A built-in benchmark: gold human scores + 4 base similarity measures.
gold, X = load_builtin("mc-training")

# Learn an interpretable fuzzy aggregator (reproducible via random_state).
model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, maxiter=40)

print("Pearson r :", round(model.score(X, gold, "pearson"), 4))
print("Spearman  :", round(model.score(X, gold, "spearman"), 4))
print("\nLearned rule base:")
print(model.describe())

print("\nExplanation for the first pair:")
print(model.explain(X[0]).format())
