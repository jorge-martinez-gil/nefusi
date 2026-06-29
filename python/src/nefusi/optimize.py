# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""A self-contained Differential Evolution optimiser.

The original NEFUSI used the Differential Evolution (DE) implementation of the
MOEAFramework Java library.  To keep the Python port dependency-free (NumPy
only) we provide a compact, well-documented DE/rand/1/bin optimiser with a fixed
seed for full reproducibility.  It is more than adequate for tuning the few
dozen parameters of a fuzzy aggregator and makes the learning procedure
completely transparent and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["DEResult", "differential_evolution"]


@dataclass
class DEResult:
    """Outcome of a Differential Evolution run."""

    x: np.ndarray
    fun: float
    history: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    converged: bool = False


def differential_evolution(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    popsize: Optional[int] = None,
    maxiter: int = 100,
    F: float = 0.6,
    CR: float = 0.9,
    seed: Optional[int] = None,
    tol: float = 1e-8,
    patience: int = 25,
    callback: Optional[Callable[[int, float], None]] = None,
) -> DEResult:
    """Minimise ``func`` over box ``bounds`` with DE/rand/1/bin.

    Parameters
    ----------
    func : callable
        Objective ``f(x) -> float`` to *minimise*.
    bounds : sequence of (low, high)
        Per-dimension search bounds.
    popsize : int, optional
        Population size.  Defaults to ``max(20, 8 * dim)``.
    maxiter : int
        Maximum number of generations.
    F : float
        Differential weight (mutation scale).
    CR : float
        Crossover probability.
    seed : int, optional
        Random seed for reproducibility.
    tol : float
        Convergence tolerance on the best objective improvement.
    patience : int
        Stop after this many generations without improvement > ``tol``.
    callback : callable, optional
        Called as ``callback(generation, best_fun)`` each generation.

    Returns
    -------
    DEResult
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)
    if popsize is None:
        popsize = max(15, min(6 * dim, 60))

    pop = lo + rng.random((popsize, dim)) * (hi - lo)
    fitness = np.array([func(ind) for ind in pop])
    n_eval = popsize

    best_idx = int(np.argmin(fitness))
    best_f = float(fitness[best_idx])
    history = [best_f]
    stale = 0
    converged = False

    idx_all = np.arange(popsize)
    for gen in range(maxiter):
        for i in range(popsize):
            choices = idx_all[idx_all != i]
            r1, r2, r3 = rng.choice(choices, 3, replace=False)
            mutant = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), lo, hi)
            cross = rng.random(dim) < CR
            cross[rng.integers(dim)] = True  # ensure at least one gene changes
            trial = np.where(cross, mutant, pop[i])
            ft = func(trial)
            n_eval += 1
            if ft <= fitness[i]:
                pop[i] = trial
                fitness[i] = ft

        gen_best = int(np.argmin(fitness))
        gen_best_f = float(fitness[gen_best])
        if best_f - gen_best_f > tol:
            stale = 0
        else:
            stale += 1
        best_f = min(best_f, gen_best_f)
        history.append(best_f)
        if callback is not None:
            callback(gen, best_f)
        if stale >= patience:
            converged = True
            break

    best_idx = int(np.argmin(fitness))
    return DEResult(
        x=pop[best_idx].copy(),
        fun=float(fitness[best_idx]),
        history=history,
        n_evaluations=n_eval,
        converged=converged,
    )
