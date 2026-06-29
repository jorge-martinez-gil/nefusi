# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""Benchmark dataset loaders.

The package bundles the precomputed feature files shipped with the original
NEFUSI repository.  Each row is ``gold, m_1, m_2, ..., m_k`` where ``gold`` is
the (min-max normalised) human similarity judgement and ``m_i`` are the base
similarity measures to be aggregated.

A small registry describes each built-in benchmark.  ``load_dataset`` reads any
file in the same comma-separated format, so users can plug in their own data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DatasetInfo",
    "BUILTIN_DATASETS",
    "data_dir",
    "load_dataset",
    "load_builtin",
    "list_datasets",
]

_HERE = os.path.dirname(os.path.abspath(__file__))


def data_dir() -> str:
    """Absolute path to the bundled dataset directory."""
    return os.path.join(_HERE, "data")


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata describing a built-in benchmark."""

    key: str
    filename: str
    n_measures: int
    description: str
    reference: str

    @property
    def path(self) -> str:
        return os.path.join(data_dir(), self.filename)


#: Registry of bundled benchmarks.
BUILTIN_DATASETS: Dict[str, DatasetInfo] = {
    "mc-training": DatasetInfo(
        "mc-training", "mc-training.txt", 4,
        "Miller & Charles word-similarity benchmark (training split). "
        "Four base similarity measures aggregated against human judgements.",
        "Miller, G.A. & Charles, W.G. (1991). Contextual correlates of "
        "semantic similarity. Language and Cognitive Processes, 6(1), 1-28.",
    ),
    "mc-validation": DatasetInfo(
        "mc-validation", "mc-validation.txt", 7,
        "Miller & Charles word-similarity benchmark (validation split, "
        "seven base measures).",
        "Miller, G.A. & Charles, W.G. (1991). Contextual correlates of "
        "semantic similarity. Language and Cognitive Processes, 6(1), 1-28.",
    ),
    "geresid-training": DatasetInfo(
        "geresid-training", "geresid-training.txt", 4,
        "GeReSiD geographic relatedness/similarity benchmark (training split).",
        "Rodriguez, M.A. & Egenhofer, M.J. (2003) / GeReSiD dataset "
        "(geographic semantic relatedness and similarity).",
    ),
    "geresid-validation": DatasetInfo(
        "geresid-validation", "geresid-validation.txt", 4,
        "GeReSiD geographic relatedness/similarity benchmark (validation split).",
        "Rodriguez, M.A. & Egenhofer, M.J. (2003) / GeReSiD dataset "
        "(geographic semantic relatedness and similarity).",
    ),
}


def list_datasets() -> List[str]:
    """Return the keys of all built-in benchmarks."""
    return list(BUILTIN_DATASETS)


def load_dataset(
    path: str, gold_col: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a comma-separated similarity file.

    Parameters
    ----------
    path : str
        File path.  Each non-empty line is ``v0, v1, ..., vk``.
    gold_col : int
        Index of the gold/human column (default 0); all other columns are
        treated as base similarity measures.

    Returns
    -------
    (gold, X)
        ``gold`` is a 1-D array of human scores; ``X`` is a 2-D array of shape
        ``(n_samples, n_measures)``.
    """
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) < 2:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"no numeric rows found in {path}")
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=float)
    gold = arr[:, gold_col]
    X = np.delete(arr, gold_col, axis=1)
    return gold, X


def load_builtin(key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a bundled benchmark by key (see :func:`list_datasets`)."""
    if key not in BUILTIN_DATASETS:
        raise KeyError(
            f"unknown dataset {key!r}; available: {list_datasets()}"
        )
    return load_dataset(BUILTIN_DATASETS[key].path)
