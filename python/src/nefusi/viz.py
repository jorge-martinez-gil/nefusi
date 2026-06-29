# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""Publication-quality visualisations for NEFUSI (matplotlib-optional).

These helpers turn the *transparent* internals of the fuzzy aggregator into
figures suitable for papers, slides and teaching:

* :func:`plot_membership_functions` -- the learned fuzzy partition;
* :func:`plot_explanation` -- per-measure contribution + aggregated output set
  for a single prediction;
* :func:`plot_convergence` -- the Differential Evolution learning curve;
* :func:`plot_scatter` -- predicted vs. human similarity with the fitted line.

``matplotlib`` is an *optional* dependency.  Importing this module without it
raises a helpful message only when a plotting function is actually called.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .fis import Explanation, MamdaniFIS
from .membership import trapmf


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "matplotlib is required for nefusi.viz; install with "
            "`pip install nefusi[plot]` or `pip install matplotlib`."
        ) from exc


def plot_membership_functions(fis: MamdaniFIS, ax=None, n: int = 400):
    """Plot the (shared) trapezoidal membership functions of the partition."""
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    u = np.linspace(0, 1, n)
    colors = {"low": "#d1495b", "medium": "#edae49", "high": "#00798c"}
    for name, corners in fis.terms.items():
        ax.plot(u, trapmf(u, *corners), lw=2.2,
                color=colors.get(name), label=name)
        ax.fill_between(u, trapmf(u, *corners), alpha=0.12,
                        color=colors.get(name))
    ax.set_xlabel("similarity value")
    ax.set_ylabel("membership degree")
    ax.set_title("NEFUSI learned fuzzy partition")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    return ax


def plot_explanation(explanation: Explanation, axes=None):
    """Plot per-measure contributions and the aggregated output fuzzy set."""
    plt = _require_mpl()
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax1, ax2 = axes

    names = explanation.measure_names
    contrib = [explanation.measure_contributions[i] for i in range(len(names))]
    ax1.barh(names, contrib, color="#00798c")
    ax1.set_xlabel("normalised contribution")
    ax1.set_title("Measure contributions")
    ax1.invert_yaxis()

    if explanation.universe is not None:
        ax2.plot(explanation.universe, explanation.aggregated_set,
                 color="#d1495b", lw=2)
        ax2.fill_between(explanation.universe, explanation.aggregated_set,
                         alpha=0.2, color="#d1495b")
        ax2.axvline(explanation.score, color="black", ls="--",
                    label=f"score = {explanation.score:.3f}")
        ax2.set_xlabel("output similarity")
        ax2.set_ylabel("aggregated membership")
        ax2.set_title("Aggregated output set")
        ax2.legend(frameon=False)
    return axes


def plot_convergence(history: Sequence[float], ax=None, objective: str = "pearson"):
    """Plot the optimiser learning curve (best objective per generation)."""
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(np.arange(len(history)), history, color="#00798c", lw=2)
    ax.set_xlabel("generation")
    ax.set_ylabel(f"best train {objective}")
    ax.set_title("Differential Evolution convergence")
    return ax


def plot_scatter(gold, pred, ax=None, title: str = "Predicted vs. human"):
    """Scatter of predicted vs. gold similarity with a least-squares line."""
    plt = _require_mpl()
    from .metrics import pearson, spearman
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    gold = np.asarray(gold); pred = np.asarray(pred)
    ax.scatter(gold, pred, s=30, alpha=0.7, color="#00798c", edgecolor="white")
    if len(gold) > 1:
        m, b = np.polyfit(gold, pred, 1)
        xs = np.linspace(gold.min(), gold.max(), 50)
        ax.plot(xs, m * xs + b, color="#d1495b", lw=2)
    ax.set_xlabel("human similarity")
    ax.set_ylabel("NEFUSI similarity")
    ax.set_title(f"{title}\nr={pearson(gold, pred):.3f}  "
                 f"rho={spearman(gold, pred):.3f}")
    return ax
