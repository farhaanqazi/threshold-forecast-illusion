# Threshold Forecast Illusion

Threshold Forecast Illusion is a research-oriented repository for studying whether thresholded
NHS waiting-list outcomes can appear highly predictable even when the underlying level series
are near-random-walk processes. The project combines a reproducible data pipeline,
time-based forecasting experiments, and honest persistence baselines around breach-risk
prediction.

> **License:** MIT · **Python:** 3.12

---

## Research question

NHS hospitals must keep at least 92% of patients within the 18-week RTT standard. This project
uses monthly public waiting-list extracts to examine a specific methodological question:

- can a binary breach indicator appear highly predictable even when the underlying level series
  are persistence-dominated?
- how much of the observed signal comes from label persistence rather than genuine forecasting
  skill?

The work is organised around careful comparisons between forecasting models and simple
persistence baselines, with particular attention to transition cases and thresholded outcomes.

---

## Project overview

The repository follows a practical workflow:

1. ingest and clean public NHS RTT extracts,
2. build a consistent analysis grain at provider × specialty × month,
3. engineer features and train forecasting/classification models,
4. compare results against persistence baselines,
5. reproduce the analysis through notebooks and tests.

Supporting design notes are documented in [architecture.md](architecture.md).

### Data grain

A single row is one **provider (hospital) × treatment-function (specialty) × month**.
The national figure reconciles end-to-end (e.g. ~7.05M waiting in Mar-2026). Two source
gotchas are handled explicitly: the `Total` column is empty on detail rows (the total is
re-derived from the wait-band counts), and the `C_999` treatment function is an
all-specialties summary line that would double-count if included, so it is excluded.

---

## Experimental setup

All models use time-based splits rather than random splits, and each result is compared against
an honest persistence baseline (next month = this month). A result is only treated as meaningful
when it improves on that baseline.

| Model | Target | Algorithm | Result |
|---|---|---|---|
| **Breach risk** | Will the specialty breach the 18-week standard next month? | XGBoost classifier | **Clear win** — ROC AUC ≈ 0.98; score = P(breach) × 100 |
| **Demand** | Next month's total waiting list | XGBoost (FVA hybrid) | Persistence-dominated at 1-month horizon → routed champion-challenger holds parity, never worse |
| **Waiting time** | Next month's % within 18 weeks | XGBoost / LightGBM | Persistence-dominated (MAE parity, marginally better RMSE) |

The key, deliberately reported finding: **level series are near-random-walk at a one-month
horizon, so they are persistence-dominated, while the binary breach outcome carries strong
signal.** That contrast — and a Forecast-Value-Added wrapper that guarantees the model is
never worse than the baseline — is the methodological core of the project.

---

## Repository layout

```
.
├── architecture.md          # Notes on the data and modeling workflow
├── notebooks/               # Reproducible analysis and experiments (run in order 01 → 07)
│   ├── 01_explore_raw_data.ipynb
│   ├── 02_unify_extracts.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_demand_model.ipynb
│   ├── 05_breach_model.ipynb
│   ├── 06_waiting_time_model.ipynb
│   └── 07_gold_star_schema.ipynb
├── src/tfi/                # Shared modeling utilities
│   ├── modeling.py          # Feature definitions, model constructors, evaluation helpers
│   └── gold.py              # Gold-layer data utilities
├── tests/                   # Regression and validation tests
│   ├── test_models.py
│   ├── test_calibration_slices.py
│   └── test_data_validation.py
├── conftest.py              # Shared fixtures for walk-forward evaluation
├── docs/
└── pyproject.toml
```

> **Data is not committed.** `data/` (raw 1.9 GB + derived) is git-ignored; the notebooks
> regenerate the Silver, feature, and Gold layers from the raw extracts. See
> [Getting the data](#getting-the-data).

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv venv --python 3.12
uv pip install -e ".[notebooks,dev]"
```

Register the Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name tfi --display-name "Python (tfi)"
```

### Getting the data

1. Download the monthly **Consultant-led RTT Waiting Times** CSVs from
   [NHS England](https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/)
   (the "Full CSV data file" for each month) into `data/raw/`.
2. Open the notebooks and run them in order **01 → 07** with the `Python (tfi)` kernel.
   This rebuilds `data/interim/`, `data/processed/`, and `data/gold/`.

---

## Running the tests

```bash
uv run pytest -q
```

The suite trains models on time-based splits and fails the build on: feature leakage, a
model losing to its persistence baseline, AUC below floor, train/test overfitting,
miscalibration, per-segment AUC drops, schema violations, or grain duplication.

---

## Outputs

The pipeline writes intermediate and derived data to the local `data/` tree, including parquet
and SQLite outputs for downstream analysis and reproducibility.

These outputs are intended for inspection, replication, and further experimental work rather than
for production deployment.

---

## Roadmap

- [x] Reproducible data preparation and feature engineering workflow
- [x] Forecasting experiments with persistence baselines
- [ ] Additional research analyses around transition subsets and thresholded outcomes
- [ ] Lightweight reproducibility tooling for rerunning experiments

---

## License

This project is licensed under the [MIT License](LICENSE).

Data © NHS England, published under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
