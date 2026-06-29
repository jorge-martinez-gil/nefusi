# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""Fuzzy membership functions and fuzzy partitions.

This module provides the building blocks used by the Mamdani fuzzy inference
engine in :mod:`nefusi.fis`.  Everything is implemented with NumPy only so the
package has no heavy dependencies and is trivial to install for researchers.

The membership functions are *vectorised*: they accept either a Python scalar or
a NumPy array as the evaluation point ``x`` and return the same shape.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "trapmf",
    "trimf",
    "scalar_trapmf",
    "TERM_NAMES",
    "partition_from_breakpoints",
    "n_breakpoints",
]

#: Canonical names of the three fuzzy terms used by the default NEFUSI partition.
#: They correspond to the ``poor`` / ``good`` / ``excellent`` terms of the
#: original Fuzzy Control Language (FCL) definition.
TERM_NAMES: Tuple[str, str, str] = ("low", "medium", "high")

#: Number of free breakpoints that parameterise the shared 3-term partition.
N_BREAKPOINTS = 8


def n_breakpoints() -> int:
    """Return the number of free breakpoints of the default 3-term partition."""
    return N_BREAKPOINTS


def trapmf(x, a: float, b: float, c: float, d: float):
    """Trapezoidal membership function with corners ``a <= b <= c <= d``.

    The function rises linearly from 0 at ``a`` to 1 at ``b``, stays at 1 on the
    plateau ``[b, c]`` and falls linearly back to 0 at ``d``.  Degenerate corners
    are handled gracefully so the same routine produces left shoulders
    (``a == b``) and right shoulders (``c == d``).

    Parameters
    ----------
    x : float or array-like
        Point(s) at which to evaluate the membership degree.
    a, b, c, d : float
        Trapezoid corners.

    Returns
    -------
    float or numpy.ndarray
        Membership degree(s) in ``[0, 1]`` with the same shape as ``x``.
    """
    scalar = np.isscalar(x)
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.zeros_like(xv)

    # Rising edge a -> b
    if b > a:
        m = (xv > a) & (xv < b)
        y[m] = (xv[m] - a) / (b - a)
    # Plateau b -> c (inclusive)
    y[(xv >= b) & (xv <= c)] = 1.0
    # Falling edge c -> d
    if d > c:
        m = (xv > c) & (xv < d)
        y[m] = (d - xv[m]) / (d - c)

    return float(y[0]) if scalar else y


def scalar_trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Fast scalar-only trapezoidal membership (used in hot optimisation loops)."""
    if x <= a:
        return 1.0 if a == b == 0.0 else 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    if x < d:
        return (d - x) / (d - c)
    return 1.0 if c == d == 1.0 and x >= c else 0.0


def trimf(x, a: float, b: float, c: float):
    """Triangular membership function (a trapezoid with ``b == c``)."""
    return trapmf(x, a, b, b, c)


def partition_from_breakpoints(
    breakpoints: Sequence[float],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Build a shared 3-term trapezoidal partition over ``[0, 1]``.

    The layout mirrors the original NEFUSI FCL definition, in which every input
    similarity measure *and* the output share the same partition shape::

        low    := (0, 1) (b1, 1) (b2, 0)
        medium := (b3, 0) (b4, 1) (b5, 1) (b6, 0)
        high   := (b7, 0) (b8, 1) (1, 1)

    Breakpoints are clipped to ``[0, 1]`` and sorted *within* each term, so any
    real-valued vector produced by an optimiser yields a valid partition.

    Parameters
    ----------
    breakpoints : sequence of 8 floats
        Free breakpoints ``[b1, ..., b8]``.

    Returns
    -------
    dict
        Mapping ``term name -> (a, b, c, d)`` trapezoid corners.
    """
    b = np.clip(np.asarray(breakpoints, dtype=float), 0.0, 1.0)
    if b.size != N_BREAKPOINTS:
        raise ValueError(
            f"expected {N_BREAKPOINTS} breakpoints, got {b.size}"
        )
    low = sorted([b[0], b[1]])
    med = sorted([b[2], b[3], b[4], b[5]])
    high = sorted([b[6], b[7]])
    return {
        "low": (0.0, 0.0, float(low[0]), float(low[1])),
        "medium": (float(med[0]), float(med[1]), float(med[2]), float(med[3])),
        "high": (float(high[0]), float(high[1]), 1.0, 1.0),
    }
