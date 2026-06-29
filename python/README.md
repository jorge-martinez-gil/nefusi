# NEFUSI (Python) — Interpretable Neuro-Fuzzy Aggregation of Semantic Similarity Measures

A pure-Python, NumPy-only reference implementation of **NEFUSI**: a transparent
**neuro-fuzzy system** that **aggregates many semantic-similarity measures into a
single score** that correlates with human judgement — while explaining *why*.

> Keywords: semantic similarity aggregation · neuro-fuzzy semantic similarity ·
> fuzzy semantic similarity · Mamdani semantic similarity · interpretable /
> explainable semantic similarity · semantic similarity ensemble · semantic
> textual similarity benchmark.

This package complements the original Java implementation. It generalises NEFUSI
from four fixed measures to **an arbitrary number of input measures**, adds a
**scikit-learn-style API**, **per-prediction explanations**, a **metrics +
benchmark suite**, a **CLI**, and **reproducible, seeded optimisation** — with
**zero heavy dependencies** (NumPy only).

## Why aggregate similarity measures, and why fuzzy logic?

No single similarity measure is best across domains. Combining them usually
helps — but a plain average is uninterpretable and a black-box regressor cannot
tell a researcher *which* measure mattered or *why*. A Mamdani fuzzy system
aggregates measures through human-readable rules ("IF measure A is high AND
measure B is high THEN similarity is high"), so every prediction comes with a
full, inspectable explanation. NEFUSI keeps that interpretability while
*learning* the membership functions and rules from data via Differential
Evolution.

## Install

```bash
pip install -e python/          # from the repository root
# optional plotting:  pip install -e "python/[plot]"
```

Requires Python ≥ 3.8 and NumPy. No SciPy, no compiler, no Java.

## 60-second example

```python
from nefusi import NeuroFuzzyAggregator, load_builtin

gold, X = load_builtin("mc-training")          # human scores + 4 measures
model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, maxiter=60)

model.score(X, gold, "pearson")                # correlation with humans
print(model.describe())                        # the learned fuzzy rule base
print(model.explain(X[0]).format())            # full explanation of one prediction
```

`explain()` returns membership degrees per measure, every rule's firing
strength, each measure's normalised contribution, a confidence score and an
uncertainty estimate.

## Command line

```bash
nefusi datasets                                  # list bundled benchmarks
nefusi fit --dataset geresid-training --test ... # fit + full metric battery
nefusi explain --dataset mc-training --row 0     # explain one prediction
nefusi fit --dataset mc-training --export-fcl learned.fcl   # export as FCL
```

## Reproducible benchmark

```bash
python examples/reproduce.py            # writes results/benchmark.md
```

Every figure is computed live; nothing is hard-coded. Metrics reported:
Pearson, Spearman, Kendall, MSE/RMSE/MAE, plus bootstrap confidence intervals.

## Bundled benchmarks

| key | measures | description |
|-----|----------|-------------|
| `mc-training` | 4 | Miller & Charles word similarity (train) |
| `mc-validation` | 7 | Miller & Charles word similarity (7 measures) |
| `geresid-training` | 4 | GeReSiD geographic relatedness (train) |
| `geresid-validation` | 4 | GeReSiD geographic relatedness (validation) |

Plug in your own data with `load_dataset("yourfile.csv")` (format:
`gold, m1, m2, ..., mk` per line).

## Scientific integrity

This is a **reference re-implementation** that is faithful to the NEFUSI
*methodology* but is **not a bit-exact reproduction** of the original
Java/MOEAFramework pipeline; absolute numbers therefore differ from the
published papers. In-sample correlations are optimistic — the meaningful figures
are the held-out and multi-objective ones in `results/benchmark.md`. No result
in this package is fabricated: run `examples/reproduce.py` to regenerate them.

## How it works

1. **Fuzzify** each measure into `low / medium / high` using a shared
   trapezoidal partition (8 learnable breakpoints).
2. **Apply** a Mamdani rule base (one rule per measure×term by default;
   `rule_scheme="pairwise"` adds interaction rules). Rules may be pruned for a
   sparser, more interpretable system.
3. **Aggregate** (max accumulation) and **defuzzify** (centroid / mean-of-maxima
   / …) to a single score in `[0, 1]`.
4. **Learn** breakpoints + rule consequents with seeded **Differential
   Evolution** maximising correlation with human judgements (optionally a
   weighted train + held-out multi-objective criterion, as in the paper).

## Citation

If you use NEFUSI, please cite (see `../CITATION.cff`):

- Martinez-Gil, J., Mokadem, R., Küng, J., Hameurlain, A. *A Novel Neurofuzzy
  Approach for Semantic Similarity Measurement.* DaWaK 2021, 192–203.
- Martinez-Gil, J., Chaves-Gonzalez, J.M. *Sustainable Semantic Similarity
  Assessment.* J. Intelligent & Fuzzy Systems 43(5):6163–6174 (2022).

## License

MIT — see `../LICENSE`.
