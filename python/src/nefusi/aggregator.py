# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""The high-level NEFUSI neuro-fuzzy similarity aggregator.

:class:`NeuroFuzzyAggregator` exposes a scikit-learn-style API
(``fit`` / ``predict`` / ``score`` / ``explain``) around the Mamdani engine in
:mod:`nefusi.fis`.  It learns

* the breakpoints of a shared 3-term fuzzy partition, and
* the consequent term of every rule (rules may also be *pruned* for sparsity),

by maximising rank/linear correlation with human similarity judgements using the
Differential Evolution optimiser in :mod:`nefusi.optimize`.

The system generalises the original four-measure NEFUSI to an arbitrary number
of input similarity measures, which is the key capability needed to use the
method as a general *aggregator* of heterogeneous similarity signals.
"""
from __future__ import annotations

import itertools
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fis import Explanation, MamdaniFIS, Rule
from .membership import TERM_NAMES, N_BREAKPOINTS, partition_from_breakpoints
from .metrics import METRICS, evaluate_all
from .optimize import DEResult, differential_evolution

__all__ = ["NeuroFuzzyAggregator"]

_TERMS = list(TERM_NAMES)              # ["low", "medium", "high"]
_CONSEQUENT_CHOICES = _TERMS + [None]  # last index = pruned/disabled rule


class NeuroFuzzyAggregator:
    """Interpretable neuro-fuzzy aggregator of semantic similarity measures.

    Parameters
    ----------
    n_measures : int, optional
        Number of input similarity measures.  Inferred from the data in
        :meth:`fit` if omitted.
    rule_scheme : {"single", "pairwise"}
        ``"single"`` builds one rule per (measure, term) pair — a fully
        transparent additive aggregator.  ``"pairwise"`` additionally adds
        interaction rules over measure pairs (agreement on ``high`` / ``low``),
        echoing the conjunctive rules of the published NEFUSI design.
    allow_pruning : bool
        If ``True`` the optimiser may disable individual rules, yielding a
        sparser, more interpretable rule base.
    defuzz : str
        Defuzzification method (see :class:`nefusi.fis.MamdaniFIS`).
    n_points : int
        Resolution of the output universe.
    measure_names : list of str, optional
        Human-readable measure names.
    random_state : int, optional
        Seed controlling the optimiser.
    """

    def __init__(
        self,
        n_measures: Optional[int] = None,
        rule_scheme: str = "single",
        allow_pruning: bool = True,
        defuzz: str = "centroid",
        n_points: int = 101,
        measure_names: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
    ):
        if rule_scheme not in ("single", "pairwise"):
            raise ValueError("rule_scheme must be 'single' or 'pairwise'")
        self.n_measures = n_measures
        self.rule_scheme = rule_scheme
        self.allow_pruning = allow_pruning
        self.defuzz = defuzz
        self.n_points = n_points
        self.measure_names = list(measure_names) if measure_names else None
        self.random_state = random_state

        # learned state (set by fit)
        self.theta_: Optional[np.ndarray] = None
        self.fis_: Optional[MamdaniFIS] = None
        self.result_: Optional[DEResult] = None
        self.history_: List[float] = []
        self._rule_templates: Optional[List[Dict[int, str]]] = None

    # -- model structure ---------------------------------------------------
    def _build_templates(self, n: int) -> List[Dict[int, str]]:
        """Antecedent templates (consequents are learned, not fixed)."""
        templates: List[Dict[int, str]] = []
        for i in range(n):
            for term in _TERMS:
                templates.append({i: term})
        if self.rule_scheme == "pairwise":
            for i, j in itertools.combinations(range(n), 2):
                templates.append({i: "high", j: "high"})
                templates.append({i: "low", j: "low"})
        return templates

    def _n_consequent_choices(self) -> int:
        return len(_CONSEQUENT_CHOICES) if self.allow_pruning else len(_TERMS)

    def _bounds(self, n: int) -> List[Tuple[float, float]]:
        templates = self._build_templates(n)
        n_choices = self._n_consequent_choices()
        # 8 breakpoints in [0, 1] then one selector per rule in [0, n_choices)
        bounds = [(0.0, 1.0)] * N_BREAKPOINTS
        bounds += [(0.0, n_choices - 1e-6)] * len(templates)
        return bounds

    def _decode(self, theta: np.ndarray, n: int) -> MamdaniFIS:
        templates = self._build_templates(n)
        partition = partition_from_breakpoints(theta[:N_BREAKPOINTS])
        rules = []
        for k, ant in enumerate(templates):
            sel = int(theta[N_BREAKPOINTS + k])
            sel = min(sel, self._n_consequent_choices() - 1)
            consequent = _CONSEQUENT_CHOICES[sel] if self.allow_pruning else _TERMS[sel]
            rules.append(Rule(antecedents=ant, consequent=consequent))
        names = self.measure_names or [f"m{i}" for i in range(n)]
        return MamdaniFIS(
            n_measures=n,
            term_partition=partition,
            rules=rules,
            measure_names=names,
            n_points=self.n_points,
            defuzz=self.defuzz,
        )

    # -- training ----------------------------------------------------------
    def fit(
        self,
        X,
        y,
        objective: str = "pearson",
        X_val=None,
        y_val=None,
        val_weight: float = 0.0,
        maxiter: int = 60,
        popsize: Optional[int] = None,
        F: float = 0.6,
        CR: float = 0.9,
        verbose: bool = False,
    ) -> "NeuroFuzzyAggregator":
        """Learn the fuzzy system from labelled data.

        Parameters
        ----------
        X : array (n_samples, n_measures)
            Base similarity measures, values in ``[0, 1]``.
        y : array (n_samples,)
            Human gold similarity scores.
        objective : {"pearson", "spearman", "kendall"} or {"mse", ...}
            Quantity to optimise.  Correlations are maximised; error metrics
            are minimised.
        X_val, y_val : arrays, optional
            Optional validation split used in a weighted multi-objective
            criterion (mirrors NEFUSI's train+test MOEA objective).
        val_weight : float in [0, 1]
            Weight of the validation term in the objective.
        maxiter, popsize, F, CR : DE hyper-parameters.
        verbose : bool
            Print progress.

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[1]
        if self.n_measures is None:
            self.n_measures = n
        elif self.n_measures != n:
            raise ValueError(
                f"n_measures={self.n_measures} but X has {n} columns"
            )
        if self.measure_names is None:
            self.measure_names = [f"m{i}" for i in range(n)]

        metric_fn = METRICS[objective]
        maximize = objective in ("pearson", "spearman", "kendall")
        use_val = X_val is not None and y_val is not None and val_weight > 0
        if use_val:
            X_val = np.atleast_2d(np.asarray(X_val, dtype=float))
            y_val = np.asarray(y_val, dtype=float).ravel()

        def objective_fn(theta: np.ndarray) -> float:
            fis = self._decode(theta, n)
            pred = fis.predict(X)
            train_score = metric_fn(y, pred)
            if not np.isfinite(train_score):
                return 1e6
            val = train_score
            if use_val:
                vpred = fis.predict(X_val)
                vscore = metric_fn(y_val, vpred)
                if not np.isfinite(vscore):
                    vscore = -1.0 if maximize else 1e6
                val = (1 - val_weight) * train_score + val_weight * vscore
            return -val if maximize else val

        bounds = self._bounds(n)
        cb = None
        if verbose:
            def cb(gen, best):  # noqa: E306
                sign = -1 if maximize else 1
                print(f"  gen {gen:3d}  best {objective}={sign * best:.4f}")

        self.result_ = differential_evolution(
            objective_fn,
            bounds,
            popsize=popsize,
            maxiter=maxiter,
            F=F,
            CR=CR,
            seed=self.random_state,
            callback=cb,
        )
        self.theta_ = self.result_.x
        self.history_ = [(-h if maximize else h) for h in self.result_.history]
        self.fis_ = self._decode(self.theta_, n)
        return self

    # -- inference & evaluation -------------------------------------------
    def _check_fitted(self):
        if self.fis_ is None:
            raise RuntimeError("aggregator is not fitted; call fit() first")

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self.fis_.predict(X)

    def score(self, X, y, metric: str = "pearson") -> float:
        self._check_fitted()
        return METRICS[metric](y, self.predict(X))

    def evaluate(self, X, y) -> Dict[str, float]:
        """Return the full battery of metrics on ``(X, y)``."""
        self._check_fitted()
        return evaluate_all(y, self.predict(X))

    def explain(self, x) -> Explanation:
        """Return a full explanation for a single input vector."""
        self._check_fitted()
        return self.fis_.explain(x)

    # -- introspection / export -------------------------------------------
    @property
    def rules_(self) -> List[Rule]:
        self._check_fitted()
        return self.fis_.active_rules()

    def describe(self) -> str:
        self._check_fitted()
        return self.fis_.describe_rules()

    def to_fcl(self) -> str:
        """Export the learned system as Fuzzy Control Language (FCL)."""
        self._check_fitted()
        return self.fis_.to_fcl()

    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "n_measures": self.n_measures,
            "rule_scheme": self.rule_scheme,
            "allow_pruning": self.allow_pruning,
            "defuzz": self.defuzz,
            "n_points": self.n_points,
            "measure_names": self.measure_names,
            "random_state": self.random_state,
            "theta": list(map(float, self.theta_)),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "NeuroFuzzyAggregator":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        model = cls(
            n_measures=d["n_measures"],
            rule_scheme=d["rule_scheme"],
            allow_pruning=d["allow_pruning"],
            defuzz=d["defuzz"],
            n_points=d["n_points"],
            measure_names=d["measure_names"],
            random_state=d["random_state"],
        )
        model.theta_ = np.asarray(d["theta"], dtype=float)
        model.fis_ = model._decode(model.theta_, d["n_measures"])
        return model
