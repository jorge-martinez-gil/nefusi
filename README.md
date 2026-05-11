# NEFUSI — Neurofuzzy Semantic Similarity

<p align="center">
  <img src="https://img.shields.io/badge/Java-11+-orange?logo=java" alt="Java 11+"/>
  <img src="https://img.shields.io/badge/build-Maven-blue?logo=apachemaven" alt="Maven"/>
  <img src="https://img.shields.io/github/license/jorge-martinez-gil/nefusi" alt="License: MIT"/>
  <img src="https://img.shields.io/badge/funded%20by-NLnet%20%7C%20NGI%20Zero-lightgrey" alt="NLnet funded"/>
  <a href="https://github.com/jorge-martinez-gil/nefusi/actions/workflows/build.yml">
    <img src="https://github.com/jorge-martinez-gil/nefusi/actions/workflows/build.yml/badge.svg" alt="CI"/>
  </a>
</p>

> **NEFUSI** is a neuro-fuzzy approach for **semantic similarity measurement** that combines fuzzy logic with multi-objective evolutionary optimisation (MOEA). It automatically learns fuzzy membership functions and rule weights to maximise correlation with human similarity judgements.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start (Docker)](#quick-start-docker)
- [Build from Source](#build-from-source)
- [Reproducing the Results](#reproducing-the-results)
- [Project Structure](#project-structure)
- [Related Publications](#related-publications)
- [Funding](#funding)
- [License](#license)

## Overview
NEFUSI trains a fuzzy inference system whose membership functions and rule parameters are optimised by a **Differential Evolution (DE)** algorithm from the MOEAFramework library. The objective is to maximise Pearson correlation on a training set (Miller & Charles benchmark) while simultaneously maintaining generalisation on a held-out test set (GeReSiD benchmark).

Expected output after a successful run (≈5 min on a modern laptop):
```
Execution time in milliseconds : 277719
Training        Test
0.8611          0.6805
```

## Prerequisites
| Tool | Version |
|------|---------|
| Java | 11+ |
| Maven | 3.8+ |
| Docker | 20+ *(optional, for containerised run)* |

## Quick Start (Docker)
The fastest way to reproduce results with zero local Java/Maven setup:

```bash
git clone https://github.com/jorge-martinez-gil/nefusi.git
cd nefusi
docker compose up --build
```

The container compiles the project, runs NEFUSI, and prints the results to stdout.

To use custom datasets, mount your files:
```bash
docker run --rm \
  -v $(pwd)/my-datasets:/app/datasets \
  -v $(pwd)/my-fcl:/app/fcl \
  nefusi-nefusi
```

## Build from Source

### 1. Clone
```bash
git clone https://github.com/jorge-martinez-gil/nefusi.git
cd nefusi
```

### 2. Build
```bash
mvn package -DskipTests
```
This produces `target/nefusi-1.0.0-jar-with-dependencies.jar`.

### 3. Run
```bash
java -jar target/nefusi-1.0.0-jar-with-dependencies.jar
```
> **Note:** The working directory must contain the `datasets/` and `fcl/` folders (they are part of this repository under `src/`).

### Legacy manual compilation (without Maven)
```bash
# From the src/ directory:
javac -cp combined.jar nefusi/*.java
java -cp ".;combined.jar" nefusi.nefusi   # Windows
java -cp ".:combined.jar" nefusi.nefusi   # Linux/macOS
```

## Reproducing the Results
NEFUSI uses two benchmark datasets shipped in `src/datasets/`:
| File | Role |
|------|------|
| `mc-training.txt` | Miller & Charles training set |
| `geresid-validation.txt` | GeReSiD test/validation set |

The fuzzy rule template is in `src/fcl/tipperB.fcl`. The optimiser fills 62 free parameters using DE with 10,000 evaluations. Results may vary slightly across runs due to the stochastic nature of DE.

## Project Structure
```
nefusi/
├── src/
│   ├── nefusi/
│   │   ├── nefusi.java        # Main class: MOEA problem definition & entry point
│   │   └── Pair.java          # Helper: index-value pair for rank computation
│   ├── datasets/              # Benchmark datasets (MC, GeReSiD)
│   ├── fcl/                   # Fuzzy Control Language templates
│   └── combined.jar           # Bundled legacy dependencies (fallback)
├── pom.xml                    # Maven build descriptor
├── Dockerfile                 # Multi-stage Docker image
├── docker-compose.yml         # One-command reproducible run
└── .github/workflows/
    └── build.yml              # CI: build & artifact upload
```

## Related Publications
- Jorge Martinez-Gil, Riad Mokadem, Josef Küng, Abdelkader Hameurlain: **A Novel Neurofuzzy Approach for Semantic Similarity Measurement**. *DaWaK 2021*, pp. 192–203.
- Jorge Martinez-Gil, Jose Manuel Chaves-Gonzalez: **Sustainable Semantic Similarity Assessment**. *J. Intell. Fuzzy Syst.* 43(5): 6163–6174 (2022).

## Funding
NEFUSI is funded by the **NGI Zero Discovery** programme of the [NLnet Foundation](https://nlnet.nl/project/NEFUSI/) and the European Commission. Project number: **2021-04-069**.

## License
This project is licensed under the [MIT License](LICENSE).
