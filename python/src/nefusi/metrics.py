# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""Correlation and error metrics for semantic similarity evaluation.

All metrics are implemented with NumPy only (no SciPy dependency) so that the
package installs cleanly in minimal environments.  The standard battery used
across the semantic-similarity literature is provided: Pearson's r, Spearman's
rho, Kendall's tau, plus MSE / RMSE / MAE.  A non-parametric bootstrap routine
yields confidence intervals for any of them.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

__all__ = [
    "pearson",
    "spearman",
    "kendall_tau",
    "mse",
    "rmse",
    "mae",
    "rank_data",
    "evaluate_all",
    "bootstrap_ci",
    "METRICS",
]


def _clean(a, b) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def pearson(a, b) -> float:
    """Pearson product-moment correlation coefficient."""
    a, b = _clean(a, b)
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def rank_data(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), with ties resolved by the mean rank."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # resolve ties to average rank
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(a, b) -> float:
    """Spearman rank correlation (Pearson on average ranks; ties handled)."""
    a, b = _clean(a, b)
    if a.size < 2:
        return float("nan")
    return pearson(rank_data(a), rank_data(b))


def kendall_tau(a, b) -> float:
    """Kendall's tau-b rank correlation (ties handled)."""
    a, b = _clean(a, b)
    n = a.size
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    ties_a = ties_b = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        prod = np.sign(da) * np.sign(db)
        concordant += int((prod > 0).sum())
        discordant += int((prod < 0).sum())
        ties_a += int((da == 0).sum())
        ties_b += int((db == 0).sum())
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - ties_a) * (n0 - ties_b))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def mse(a, b) -> float:
    a, b = _clean(a, b)
    return float(np.mean((a - b) ** 2))


def rmse(a, b) -> float:
    return float(np.sqrt(mse(a, b)))


def mae(a, b) -> float:
    a, b = _clean(a, b)
    return float(np.mean(np.abs(a - b)))


#: Registry of named metrics, all with signature ``f(gold, pred) -> float``.
METRICS: Dict[str, Callable] = {
    "pearson": pearson,
    "spearman": spearman,
    "kendall": kendall_tau,
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
}


def evaluate_all(gold, pred) -> Dict[str, float]:
    """Compute every registered metric and return a name -> value dict."""
    return {name: fn(gold, pred) for name, fn in METRICS.items()}


def bootstrap_ci(
    gold,
    pred,
    metric: str = "pearson",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval for a metric.

    Returns
    -------
    (point_estimate, lower, upper)
        The metric on the full sample and the ``(1 - alpha)`` percentile CI.
    """
    gold, pred = _clean(gold, pred)
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    n = gold.size
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = fn(gold[idx], pred[idx])
    stats = stats[np.isfinite(stats)]
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return float(fn(gold, pred)), lo, hi
