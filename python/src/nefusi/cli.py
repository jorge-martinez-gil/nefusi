# SPDX-FileCopyrightText: 2022 Jorge Martinez Gil
#
# SPDX-License-Identifier: MIT
"""Command-line interface for NEFUSI.

Examples
--------
List bundled benchmarks::

    nefusi datasets

Fit on a built-in benchmark and report all metrics::

    nefusi fit --dataset mc-training --seed 0

Fit on your own files and aggregate similarity measures::

    nefusi fit --train my_train.csv --test my_test.csv --seed 0

Explain a single prediction (row index into the dataset)::

    nefusi explain --dataset mc-training --row 0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from . import __version__
from .aggregator import NeuroFuzzyAggregator
from .datasets import BUILTIN_DATASETS, list_datasets, load_builtin, load_dataset
from .metrics import bootstrap_ci, evaluate_all


def _load(args):
    if args.dataset:
        gold, X = load_builtin(args.dataset)
    elif args.train:
        gold, X = load_dataset(args.train)
    else:
        raise SystemExit("error: provide --dataset KEY or --train FILE")
    return gold, X


def _fit_model(X, gold, args) -> NeuroFuzzyAggregator:
    model = NeuroFuzzyAggregator(
        rule_scheme=args.rule_scheme,
        allow_pruning=not args.no_pruning,
        defuzz=args.defuzz,
        random_state=args.seed,
    )
    model.fit(
        X, gold,
        objective=args.objective,
        maxiter=args.maxiter,
        verbose=args.verbose,
    )
    return model


def cmd_datasets(args) -> int:
    for key in list_datasets():
        info = BUILTIN_DATASETS[key]
        print(f"{key:20s}  {info.n_measures} measures   {info.description}")
    return 0


def cmd_fit(args) -> int:
    gold, X = _load(args)
    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} measures")
    model = _fit_model(X, gold, args)

    print("\n== Training performance ==")
    for k, v in evaluate_all(gold, model.predict(X)).items():
        print(f"  {k:9s}: {v:.4f}")
    r, lo, hi = bootstrap_ci(gold, model.predict(X), "pearson", seed=args.seed)
    print(f"  pearson 95% CI: [{lo:.3f}, {hi:.3f}]")

    if args.test:
        ygt, Xt = load_dataset(args.test)
        if Xt.shape[1] == X.shape[1]:
            print("\n== Held-out test performance ==")
            for k, v in evaluate_all(ygt, model.predict(Xt)).items():
                print(f"  {k:9s}: {v:.4f}")
        else:
            print(f"\n[skip] test set has {Xt.shape[1]} measures, "
                  f"model expects {X.shape[1]}")

    if args.export_fcl:
        with open(args.export_fcl, "w", encoding="utf-8") as fh:
            fh.write(model.to_fcl())
        print(f"\nExported learned system to {args.export_fcl}")
    if args.save:
        model.save(args.save)
        print(f"Saved model to {args.save}")
    return 0


def cmd_explain(args) -> int:
    gold, X = _load(args)
    model = _fit_model(X, gold, args)
    row = args.row
    exp = model.explain(X[row])
    if args.json:
        print(json.dumps(exp.to_dict(), indent=2))
    else:
        print(f"Gold similarity for row {row}: {gold[row]:.4f}")
        print(exp.format())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nefusi",
        description="Interpretable neuro-fuzzy aggregation of semantic "
                    "similarity measures.",
    )
    p.add_argument("--version", action="version", version=f"nefusi {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--dataset", choices=list_datasets(),
                        help="built-in benchmark key")
        sp.add_argument("--train", help="training file (comma-separated)")
        sp.add_argument("--test", help="held-out test file")
        sp.add_argument("--rule-scheme", default="single",
                        choices=["single", "pairwise"])
        sp.add_argument("--no-pruning", action="store_true",
                        help="disable rule pruning")
        sp.add_argument("--defuzz", default="centroid",
                        choices=["centroid", "mom", "lom", "som", "bisector"])
        sp.add_argument("--objective", default="pearson",
                        choices=["pearson", "spearman", "kendall",
                                 "mse", "rmse", "mae"])
        sp.add_argument("--maxiter", type=int, default=60)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--verbose", action="store_true")

    sp = sub.add_parser("datasets", help="list bundled benchmarks")
    sp.set_defaults(func=cmd_datasets)

    sp = sub.add_parser("fit", help="train and evaluate an aggregator")
    add_common(sp)
    sp.add_argument("--export-fcl", help="write learned system as FCL")
    sp.add_argument("--save", help="write learned model as JSON")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("explain", help="explain a single prediction")
    add_common(sp)
    sp.add_argument("--row", type=int, default=0, help="row index to explain")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_explain)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
