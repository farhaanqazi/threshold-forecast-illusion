# Threshold Forecast Illusion

Threshold Forecast Illusion is a research repository dedicated to proving and measuring a fundamental time-series classification phenomenon: **thresholded outcomes derived from highly autocorrelated level series can appear highly predictable, even when the model possesses no genuine foresight.** 

This project transitions from an initial empirical observation on NHS waiting lists to a formal mathematical theorem, confirmed via synthetic simulation and replicated across multiple domains.

> **License:** MIT · **Python:** 3.12

---

## The Theorem

If a continuous level series $X_t$ follows a near-random-walk (or a highly persistent AR(1) process) with lag-1 autocorrelation $\rho$, and we attempt to classify a binary breach state $Y_t$ (where $Y_t = 1$ if $X_t \ge \tau$), a trivial persistence classifier (using only the lagged level $X_{t-1}$ as a score) will achieve a ROC AUC of exactly:

$$ \text{AUC} = \frac{1}{2} + \frac{2}{\pi} \arcsin\left(\frac{\rho}{\sqrt{2}}\right) $$

For a non-stationary random walk ($\rho \approx 1$), the AUC collapses to $\approx 0.99$. Complex machine learning models deployed on such tasks often report astronomical AUCs, masking the reality that they are merely memorizing the current level. We call this the **Threshold Forecast Illusion**.

---

## Project Structure & Roadmap

The research is executed in three distinct phases:

### Phase 1: Numerical Validation (Flagship)
- **Synthetic Simulation Engine:** A stationary AR(1) generator sweeps $\phi$ from 0.5 to 0.99.
- **Confirmation:** The empirical AUC of the persistence classifier is strictly matched against the analytic curve, confirming the mathematical proof within tight statistical tolerances.

### Phase 2: Corrective Metrics
To expose the illusion in real-world models, we implement two distinct corrective metrics:
- **Full-Set Forecast Value Added (FVA):** Evaluates whether a complex model adds any skill over the trivial persistence baseline on the full dataset ($\text{FVA} \approx 0$ means no true foresight).
- **Transition-Subset AUC:** Evaluates model accuracy exclusively on timesteps where the label flips ($Y_t \neq Y_{t-1}$). Under the illusion, this diagnostic collapses to chance ($\approx 0.5$).

### Phase 3: Empirical Generalization
The illusion is demonstrated on real-world datasets that fit the structural requirement (a continuous drifting quantity against a fixed policy cutoff).
- **Domain A (Healthcare):** Public NHS RTT waiting-list extracts (the original empirical pipeline).
- **Domain B (Environment):** Air Quality (PM2.5) indices evaluated against regulatory limits.

---

## Repository Layout

```text
.
├── implementation_plan.md   # Architectural blueprint and milestones
├── notebooks/               # Reproducible analysis and experiments
│   ├── 01_to_07_...         # The NHS Empirical Pipeline (Data extraction to modeling)
│   ├── 08_validate_theorem.ipynb
│   ├── 09_transition_metrics.ipynb
│   └── 10_second_domain_replication.ipynb
├── src/tfi/                 # Threshold Forecast Illusion core logic
│   ├── synthetic.py         # AR(1) generator and theoretical validation
│   ├── modeling.py          # Transition-Subset AUC and Full-Set FVA utilities
│   └── gold.py              # NHS data handlers
├── tests/                   # Strict regression and mathematical theorem tests
│   └── test_synthetic.py    # Gated mathematical validation test
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

Phase 1 (The Theorem) is gated by a strict mathematical test suite. The test simulates high-volume thresholded AR(1) walks and asserts that the empirical AUC matches the analytic formula within a strict standard-error tolerance.

```bash
uv run pytest tests/test_synthetic.py -s
```

### Getting the NHS Empirical Data

To replicate the healthcare domain findings (Notebooks 01 → 07):
1. Download the monthly **Consultant-led RTT Waiting Times** CSVs from [NHS England](https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/) into `data/raw/`.
2. Run notebooks **01 → 07** with the `Python (TFI)` kernel.

---

## License

This project is licensed under the [MIT License](LICENSE).

Data © NHS England, published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
