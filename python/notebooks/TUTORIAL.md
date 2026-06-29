# NEFUSI Tutorial — Interpretable Neuro-Fuzzy Aggregation of Semantic Similarity

This tutorial is self-contained and runnable. Copy each block into a Python
session (or a Jupyter cell). It teaches both the **concepts** (semantic
similarity, fuzzy logic, Mamdani inference, neuro-fuzzy learning, explainability)
and the **NEFUSI API**.

---

## 1. The problem: aggregating semantic similarity measures

A *semantic similarity measure* maps a pair of items (words, sentences, ontology
concepts, code fragments…) to a number in `[0, 1]`. Dozens of measures exist —
WordNet path lengths, embedding cosines, sentence-transformer scores, LLM
judgements — and **none dominates across domains**.

A natural idea is to *aggregate* several measures into one. But:

- A plain **average** ignores that some measures are more reliable in some
  regions of the scale, and tells you nothing about *why* a score came out.
- A **black-box regressor** (e.g. a neural net) can be accurate but is not
  interpretable — a problem for scientific and explainable-AI use.

NEFUSI aggregates measures with a **fuzzy rule base**, which is both adaptive
*and* fully interpretable.

## 2. Fuzzy logic in one minute

Instead of a hard threshold ("similarity > 0.5 = high"), fuzzy logic assigns a
**degree of membership** to overlapping linguistic terms. NEFUSI uses three
terms — `low`, `medium`, `high` — described by trapezoidal membership functions.

```python
from nefusi import partition_from_breakpoints, trapmf
import numpy as np

partition = partition_from_breakpoints([0.2, 0.4, 0.3, 0.45, 0.55, 0.7, 0.6, 0.8])
for name, corners in partition.items():
    print(name, "membership at 0.5 =", round(trapmf(0.5, *corners), 3))
```

A value of `0.5` can be partly `medium` *and* partly `high` at the same time —
this graded, overlapping representation is what makes fuzzy systems robust and
human-readable.

## 3. Mamdani inference

A **Mamdani** fuzzy system reasons with `IF … THEN …` rules:

```
IF measure_A IS high AND measure_B IS high THEN similarity IS high
```

Inference has four steps: **fuzzify** the inputs, compute each rule's **firing
strength** (the AND = minimum of its antecedent memberships), **accumulate** the
clipped output sets (max), and **defuzzify** (e.g. centroid) to a single number.

```python
from nefusi import MamdaniFIS, Rule

rules = [
    Rule({0: "high"}, "high"), Rule({1: "high"}, "high"),
    Rule({0: "medium"}, "medium"), Rule({1: "medium"}, "medium"),
    Rule({0: "low"}, "low"), Rule({1: "low"}, "low"),
]
fis = MamdaniFIS(2, partition, rules, measure_names=["A", "B"])
print("infer([0.9, 0.8]) =", round(fis.infer([0.9, 0.8]), 3))
print("infer([0.1, 0.2]) =", round(fis.infer([0.1, 0.2]), 3))
```

## 4. Neuro-fuzzy learning

Hand-tuning membership functions and rules is tedious and subjective. The
*neuro-fuzzy* idea is to **learn them from data**. NEFUSI optimises the 8
breakpoints and the rule consequents with **Differential Evolution** to maximise
correlation with human judgements.

```python
from nefusi import NeuroFuzzyAggregator, load_builtin

gold, X = load_builtin("mc-training")          # 4 base measures + human scores
model = NeuroFuzzyAggregator(random_state=0).fit(X, gold, maxiter=60)
print("Pearson r:", round(model.score(X, gold, "pearson"), 3))
print(model.describe())                        # the learned rules
```

## 5. Explainability — the whole point

Every prediction can be fully explained:

```python
exp = model.explain(X[0])
print(exp.format())
# programmatic access:
exp.memberships            # fuzzification degree of each measure
exp.rule_firings           # (rule, firing strength) for every rule
exp.measure_contributions  # normalised contribution of each measure
exp.confidence, exp.uncertainty
exp.to_dict()              # JSON-serialisable explanation
```

This answers the questions an XAI researcher cares about: *which measures drove
this score, through which rules, and how confident is the system?*

## 6. Evaluation the way the field does it

```python
from nefusi.metrics import evaluate_all, bootstrap_ci
print(evaluate_all(gold, model.predict(X)))                 # r, rho, tau, MSE…
print(bootstrap_ci(gold, model.predict(X), "pearson"))      # 95% CI
```

## 7. Generalisation and the multi-objective trick

Small similarity datasets overfit easily. Optimising a *joint* train + held-out
objective (as in the original paper) improves transfer:

```python
g_te, X_te = load_builtin("geresid-validation")
mo = NeuroFuzzyAggregator(random_state=0).fit(
    X, gold, X_val=X_te, y_val=g_te, val_weight=0.5, maxiter=60)
print("held-out r:", round(mo.score(X_te, g_te, "pearson"), 3))
```

## 8. Export, save, visualise

```python
model.save("model.json")                       # round-trippable
open("learned.fcl", "w").write(model.to_fcl()) # IEC-61131-7 Fuzzy Control Language

# with matplotlib installed (pip install nefusi[plot]):
# from nefusi.viz import plot_membership_functions, plot_explanation
# plot_membership_functions(model.fis_)
# plot_explanation(model.explain(X[0]))
```

## 9. Use your own measures

Any number of measures works — give NEFUSI a CSV `gold, m1, m2, …, mk`:

```python
from nefusi import load_dataset
gold, X = load_dataset("my_pairs.csv")
NeuroFuzzyAggregator(measure_names=["bert", "wordnet", "llm"]).fit(X, gold)
```

## Exercises

1. Compare `rule_scheme="single"` vs `"pairwise"` on GeReSiD. Which generalises
   better, and how many rules does each keep after pruning?
2. Swap the defuzzifier (`defuzz="mom"`, `"bisector"`, …). How does it change
   calibration (MSE) vs. ranking (Spearman)?
3. Add a deliberately noisy 5th measure. Does NEFUSI prune it? Inspect
   `explain().measure_contributions`.
