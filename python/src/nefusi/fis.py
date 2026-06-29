# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""A small, transparent Mamdani fuzzy inference engine.

The engine aggregates an arbitrary number ``N`` of base similarity measures into
a single similarity score using a Mamdani-type fuzzy rule base.  Unlike the
original Java implementation (which was hard-wired to exactly four measures and
manipulated Fuzzy Control Language files via string substitution) this engine is

* general over ``N`` measures,
* fully introspectable, and
* able to return a complete *explanation* of every prediction
  (fuzzification degrees, per-rule firing strengths, the aggregated output fuzzy
  set, the contribution of each rule, and an uncertainty estimate).

Everything is pure NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .membership import scalar_trapmf, trapmf

__all__ = ["Rule", "Explanation", "MamdaniFIS"]


@dataclass(frozen=True)
class Rule:
    """A single Mamdani rule.

    Parameters
    ----------
    antecedents : dict
        Mapping ``measure index -> term name``.  Multiple entries are combined
        with the AND t-norm.
    consequent : str or None
        Output term name, or ``None`` if the rule is disabled (pruned).
    weight : float
        Rule weight in ``[0, 1]`` (certainty factor).
    """

    antecedents: Dict[int, str]
    consequent: Optional[str]
    weight: float = 1.0

    @property
    def active(self) -> bool:
        return self.consequent is not None

    def describe(self, measure_names: Optional[Sequence[str]] = None) -> str:
        if not self.active:
            return "(disabled)"
        parts = []
        for idx, term in self.antecedents.items():
            name = measure_names[idx] if measure_names else f"m{idx}"
            parts.append(f"{name} IS {term}")
        body = " AND ".join(parts)
        w = "" if abs(self.weight - 1.0) < 1e-9 else f" (w={self.weight:.2f})"
        return f"IF {body} THEN score IS {self.consequent}{w}"


@dataclass
class Explanation:
    """A complete, human-readable trace of one prediction."""

    score: float
    inputs: np.ndarray
    measure_names: List[str]
    memberships: Dict[int, Dict[str, float]]
    rule_firings: List[Tuple[Rule, float]]
    measure_contributions: Dict[int, float]
    confidence: float
    uncertainty: float
    universe: np.ndarray = field(repr=False, default=None)
    aggregated_set: np.ndarray = field(repr=False, default=None)

    def top_rules(self, k: int = 5) -> List[Tuple[Rule, float]]:
        """Return the ``k`` rules with the highest firing strength."""
        firing = [(r, s) for r, s in self.rule_firings if s > 0]
        firing.sort(key=lambda t: t[1], reverse=True)
        return firing[:k]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "inputs": list(map(float, self.inputs)),
            "measure_names": list(self.measure_names),
            "memberships": {
                self.measure_names[i]: degrees
                for i, degrees in self.memberships.items()
            },
            "active_rules": [
                {"rule": r.describe(self.measure_names), "firing": float(s)}
                for r, s in self.rule_firings
                if s > 0
            ],
            "measure_contributions": {
                self.measure_names[i]: float(c)
                for i, c in self.measure_contributions.items()
            },
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
        }

    def format(self, k: int = 5) -> str:
        lines = [f"NEFUSI prediction: score = {self.score:.4f}",
                 f"  confidence = {self.confidence:.3f}   "
                 f"uncertainty = {self.uncertainty:.3f}", "",
                 "  Input fuzzification:"]
        for i, degrees in self.memberships.items():
            deg = ", ".join(f"{t}={v:.2f}" for t, v in degrees.items())
            lines.append(f"    {self.measure_names[i]} = {self.inputs[i]:.3f}"
                         f"  ->  {deg}")
        lines.append("")
        lines.append(f"  Top firing rules:")
        for r, s in self.top_rules(k):
            lines.append(f"    [{s:.2f}] {r.describe(self.measure_names)}")
        lines.append("")
        lines.append("  Measure contributions:")
        for i, c in sorted(self.measure_contributions.items(),
                           key=lambda t: t[1], reverse=True):
            lines.append(f"    {self.measure_names[i]}: {c:.3f}")
        return "\n".join(lines)


class MamdaniFIS:
    """A Mamdani fuzzy inference system over ``N`` similarity measures.

    Parameters
    ----------
    n_measures : int
        Number of input similarity measures.
    term_partition : dict
        Mapping ``term name -> (a, b, c, d)`` trapezoid corners, shared by all
        inputs and the output (see :func:`nefusi.membership.partition_from_breakpoints`).
    rules : list of Rule
        The fuzzy rule base.
    measure_names : list of str, optional
        Human-readable names for the measures.
    n_points : int
        Resolution of the discretised output universe ``[0, 1]``.
    defuzz : {"centroid", "mom", "lom", "som", "bisector"}
        Defuzzification method.
    """

    DEFUZZ_METHODS = ("centroid", "mom", "lom", "som", "bisector")

    def __init__(
        self,
        n_measures: int,
        term_partition: Dict[str, Tuple[float, float, float, float]],
        rules: Sequence[Rule],
        measure_names: Optional[Sequence[str]] = None,
        n_points: int = 101,
        defuzz: str = "centroid",
    ):
        if defuzz not in self.DEFUZZ_METHODS:
            raise ValueError(f"unknown defuzz method {defuzz!r}")
        self.n_measures = int(n_measures)
        self.terms = dict(term_partition)
        self.rules = list(rules)
        self.measure_names = (
            list(measure_names)
            if measure_names is not None
            else [f"m{i}" for i in range(n_measures)]
        )
        self.n_points = int(n_points)
        self.defuzz = defuzz
        self.u = np.linspace(0.0, 1.0, self.n_points)
        # Pre-compute output membership arrays for every term.
        self._term_u = {
            name: trapmf(self.u, *corners) for name, corners in self.terms.items()
        }

    # -- fuzzification -----------------------------------------------------
    def membership(self, value: float, term: str) -> float:
        return scalar_trapmf(value, *self.terms[term])

    def fuzzify(self, value: float) -> Dict[str, float]:
        return {name: scalar_trapmf(value, *c) for name, c in self.terms.items()}

    # -- inference ---------------------------------------------------------
    def _aggregate(self, x: Sequence[float]):
        """Return (aggregated output set, list of (rule, firing strength))."""
        agg = np.zeros_like(self.u)
        firings: List[Tuple[Rule, float]] = []
        for rule in self.rules:
            if not rule.active:
                firings.append((rule, 0.0))
                continue
            strengths = [
                scalar_trapmf(x[idx], *self.terms[term])
                for idx, term in rule.antecedents.items()
            ]
            fs = (min(strengths) if strengths else 0.0) * rule.weight
            firings.append((rule, fs))
            if fs > 0.0:
                clipped = np.minimum(self._term_u[rule.consequent], fs)
                agg = np.maximum(agg, clipped)
        return agg, firings

    def _defuzzify(self, agg: np.ndarray) -> float:
        total = agg.sum()
        if total <= 0.0:
            return 0.0
        if self.defuzz == "centroid":
            return float((self.u * agg).sum() / total)
        if self.defuzz == "bisector":
            cumulative = np.cumsum(agg)
            half = cumulative[-1] / 2.0
            return float(self.u[np.searchsorted(cumulative, half)])
        peak = agg.max()
        max_idx = np.where(agg >= peak - 1e-12)[0]
        if self.defuzz == "mom":  # mean of maxima
            return float(self.u[max_idx].mean())
        if self.defuzz == "som":  # smallest of maxima
            return float(self.u[max_idx[0]])
        if self.defuzz == "lom":  # largest of maxima
            return float(self.u[max_idx[-1]])
        raise AssertionError("unreachable")

    def infer(self, x: Sequence[float]) -> float:
        """Aggregate one input vector into a single similarity score."""
        agg, _ = self._aggregate(x)
        return self._defuzzify(agg)

    def predict(self, X) -> np.ndarray:
        """Vectorised prediction over a 2-D array ``(n_samples, n_measures)``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self.infer(row) for row in X])

    # -- explainability ----------------------------------------------------
    def explain(self, x: Sequence[float]) -> Explanation:
        """Return a complete :class:`Explanation` for one input vector."""
        x = np.asarray(x, dtype=float)
        agg, firings = self._aggregate(x)
        score = self._defuzzify(agg)

        memberships = {i: self.fuzzify(x[i]) for i in range(self.n_measures)}

        # Per-measure contribution: total firing strength routed through each
        # measure, normalised to sum to 1 over measures that fired.
        raw = {i: 0.0 for i in range(self.n_measures)}
        for rule, fs in firings:
            if fs <= 0.0 or not rule.active:
                continue
            for idx in rule.antecedents:
                raw[idx] += fs
        s = sum(raw.values())
        contributions = (
            {i: v / s for i, v in raw.items()} if s > 0 else raw
        )

        # Confidence: peak activation of the aggregated output set (how strongly
        # *some* rule supports the answer).  Uncertainty: normalised spread of
        # the aggregated fuzzy set (wide support => ambiguous answer).
        confidence = float(agg.max())
        if agg.sum() > 0:
            mean = (self.u * agg).sum() / agg.sum()
            var = ((self.u - mean) ** 2 * agg).sum() / agg.sum()
            uncertainty = float(np.sqrt(var) / 0.2887)  # 0.2887 = std of U[0,1]
        else:
            uncertainty = 1.0

        return Explanation(
            score=score,
            inputs=x,
            measure_names=self.measure_names,
            memberships=memberships,
            rule_firings=firings,
            measure_contributions=contributions,
            confidence=confidence,
            uncertainty=min(uncertainty, 1.0),
            universe=self.u,
            aggregated_set=agg,
        )

    # -- export ------------------------------------------------------------
    def active_rules(self) -> List[Rule]:
        return [r for r in self.rules if r.active]

    def describe_rules(self) -> str:
        lines = []
        for n, r in enumerate(self.active_rules(), 1):
            lines.append(f"RULE {n}: {r.describe(self.measure_names)}")
        return "\n".join(lines)

    def to_fcl(self, block_name: str = "nefusi") -> str:
        """Export the rule base as IEC-61131-7 Fuzzy Control Language (FCL)."""
        def term_line(name, c):
            a, b, cc, d = c
            return f"\tTERM {name} := ({a:g},0) ({b:g},1) ({cc:g},1) ({d:g},0);"

        lines = [f"FUNCTION_BLOCK {block_name}", "", "VAR_INPUT"]
        for nm in self.measure_names:
            lines.append(f"\t{nm} : REAL;")
        lines += ["END_VAR", "", "VAR_OUTPUT", "\tscore : REAL;", "END_VAR", ""]
        for i, nm in enumerate(self.measure_names):
            lines.append(f"FUZZIFY {nm}")
            for t, c in self.terms.items():
                lines.append(term_line(t, c))
            lines.append("END_FUZZIFY\n")
        lines.append("DEFUZZIFY score")
        for t, c in self.terms.items():
            lines.append(term_line(t, c))
        method = {"centroid": "COG", "mom": "MOM", "lom": "LM",
                  "som": "RM", "bisector": "COGS"}[self.defuzz]
        lines += [f"\tMETHOD : {method};", "\tDEFAULT := 0;", "END_DEFUZZIFY", ""]
        lines += ["RULEBLOCK No1", "\tAND : MIN;", "\tACT : MIN;", "\tACCU : MAX;", ""]
        for n, r in enumerate(self.active_rules(), 1):
            ant = " AND ".join(
                f"{self.measure_names[i]} IS {t}" for i, t in r.antecedents.items()
            )
            lines.append(f"\tRULE {n} : IF {ant} THEN score IS {r.consequent};")
        lines += ["END_RULEBLOCK", "", "END_FUNCTION_BLOCK"]
        return "\n".join(lines)
