# Threshold Forecast Illusion

Threshold Forecast Illusion is a research repository examining how persistence affects time-series classification metrics. It began with an NHS waiting-list breach-prediction task in which a standard model achieved a high AUC, but a current-state persistence baseline achieved most of the same discrimination.

The central comparison is not forecasting versus no forecasting: persistence is a legitimate forecast. The question is how much incremental discrimination a trained model adds beyond a persistence baseline.

> **License:** MIT · **Python:** 3.12

---

## The Theorem

For the centered-threshold synthetic AR(1) setup used here, a continuous level series $X_t$ with lag-1 autocorrelation $\rho$ and a binary state derived by thresholding at the process center has a continuous-persistence ROC AUC of:

$$ \text{AUC} = \frac{1}{2} + \frac{2}{\pi} \arcsin\left(\frac{\rho}{\sqrt{2}}\right) $$

The relationship is validated numerically for $\rho$ from approximately $-0.7$ to $0.99$, with maximum absolute simulation error below 0.003 in the extended sweep. The derivation and validation assume a clean synthetic process; the formula is not claimed to describe every bounded, heterogeneous NHS series. The practical concern is that a high AUC can substantially reflect persistence rather than incremental model value.

---

## Project Structure & Roadmap

The research is executed in three distinct phases:

### Phase 1: Numerical Validation
- **Synthetic Simulation Engine:** A stationary AR(1) generator tests centered thresholding across negative, near-zero, and high autocorrelation.
- **Validation:** The empirical continuous-persistence AUC closely matches the analytic curve across the extended range.

### Phase 2: Persistence-Aware Evaluation
To expose how much a model adds beyond current state, the code includes:
- **Persistence comparisons:** Binary breach-state and continuous current-performance baselines.
- **Forecast Value Added (FVA):** Compares level forecasts against persistence using held-out time periods.
- **Transition diagnostics:** Reports ranking metrics on observations where the breach state changes.

### Phase 3: NHS Case Study
The current empirical study is limited to public NHS England RTT waiting-list extracts. No second-domain replication is included.

---

## Repository Layout

```text
.
├── implementation_plan.md   # Architectural blueprint and milestones
├── notebooks/               # Reproducible analysis and experiments
│   ├── 01_to_07_...         # The NHS Empirical Pipeline (Data extraction to modeling)
├── src/tfi/                 # Threshold Forecast Illusion core logic
│   ├── synthetic.py         # AR(1) generator and theoretical validation
│   ├── modeling.py          # Transition-Subset AUC and Full-Set FVA utilities
│   └── gold.py              # NHS data handlers
├── tests/                   # Strict regression and mathematical theorem tests
│   └── test_synthetic.py    # Gated mathematical validation test
├── results/                 # Committed analysis tables from the NHS study
│   ├── persistence_vs_model.csv
│   ├── theorem_vs_observed.csv
│   ├── nhs_auc_ci.csv
│   ├── balance_sweep.csv
│   ├── rho_distribution.csv
│   ├── rho_tercile_gap.csv
│   ├── synthetic_sweep_extended.csv
│   └── formula_aggregation.csv
├── pyproject.toml
└── README.md
```

---

## Setup & Execution

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv venv --python 3.12
uv pip install -e ".[notebooks,dev]"
```

Register the Jupyter kernel for running the analytical notebooks:

```bash
uv run python -m ipykernel install --user --name tfi --display-name "Python (TFI)"
```

### Running the Theorem Validation

The theorem validation test simulates high-volume centered-threshold AR(1) walks and checks that empirical AUC matches the analytic formula within a bootstrap standard-error tolerance.

```bash
uv run pytest tests/test_synthetic.py -s
```

### Getting the NHS Empirical Data

To replicate the healthcare domain findings (Notebooks 01 → 07):
1. Download the monthly **Consultant-led RTT Waiting Times** CSVs from [NHS England](https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/) into `data/raw/`.
2. Run notebooks **01 → 07** with the `Python (TFI)` kernel.

### Current NHS Findings

The walk-forward evaluation covers November 2025 through February 2026, with 14,172 pooled test rows:

- XGBoost AUC: **0.979**.
- Binary persistence AUC: **0.897**.
- Continuous persistence AUC: **0.960**.
- The model adds 0.0196 AUC over continuous persistence, while the binary-baseline gap is 0.0822.
- The model's 95% bootstrap interval does not overlap the binary-persistence interval; the intervals are row-bootstrap estimates and may be optimistic because rows repeat provider-specialty series.

The formula evaluated at the NHS median rho (0.807) predicts AUC 0.887, below the observed continuous-persistence AUC 0.960. Rebalancing the pooled data did not close that gap. Applying the formula per series and averaging also did not close it: the equal-weighted and row-weighted predictions were 0.860 and 0.862. The gap is therefore unresolved; the results do not establish a single mechanism for the NHS discrepancy.

The rho-tercile analysis is descriptive: the formula-observed gap was 0.158 for the low-rho group and 0.033 for the high-rho group. This shows that formula fit varies with series stickiness, but it does not by itself identify the cause.

The CSV tables in `results/` contain the reported values. The raw-to-feature pipeline and model implementation remain in the notebooks and `src/tfi/`; the exploratory scripts that produced the result tables are local scratch files under `_local/`.

---

## License

This project is licensed under the [MIT License](LICENSE).

Data © NHS England, published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
