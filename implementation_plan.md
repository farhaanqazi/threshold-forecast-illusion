# Pivot to Formal Theorem: Threshold Forecast Illusion

This implementation plan is derived directly from the Coder Agent Brief. The project pivots from an empirical NHS observation into a formal, generalized theorem demonstrating how persistence classifiers achieve artificially high AUCs on thresholded near-random walks solely due to autocorrelation.

## Goal Description

To numerically confirm the analytic theorem:
`AUC = 1/2 + (2/pi) * arcsin(rho / sqrt(2))`
and to implement corrective metrics (Transition-Subset AUC and Full-Set FVA) that strip out the illusion of foresight. Finally, to replicate the finding on a second non-NHS domain (Air Quality index vs. regulatory limits).

> [!WARNING]
> **Strict Validation Milestone:** The validation milestone (Phase 1 success criterion) is that the empirical curve matches the analytic curve across the `phi` sweep. Nothing else gets built until that holds.

## Proposed Changes

We will create a new suite of research notebooks or scripts, completely isolated from the existing NHS data pipeline, to prove the generalized case first.

### Phase 1: Validating the Theorem

#### [NEW] `src/tfi/synthetic.py`
- **Task 1:** Generator is `X_t = phi*X_{t-1} + mu + eps`, `phi` sweepable. For stationary AR(1), `rho = phi`. Define a minimal dataframe interface (time, level, label).

#### [NEW] `notebooks/08_validate_theorem.ipynb`
- **Task 2:** Implement the Persistence Classifier (Score for `Y_t` is the lagged level `X_{t-1}`).
- **Task 3:** Sweep `phi` over {0.5, 0.7, 0.9, 0.95, 0.99}, overlay empirical AUC against `1/2 + (2/pi)*arcsin(phi/sqrt(2))`, balanced threshold, fixed seed, N >= 10k.
- *Success Criteria:* Empirical curve matches the analytic curve across the `phi` sweep.

---

### Phase 2: Corrective Metrics

#### [MODIFY] `src/tfi/modeling.py`
- **Task 4 (Transition-Subset AUC):** `transition-subset AUC` must return both full-set and transition AUC.
- **Task 5 (FVA):** FVA is full-set. Baseline = lagged-level score, never lagged label.

#### [NEW] `notebooks/09_transition_metrics.ipynb`
- Apply the new transition-subset and FVA metrics to the synthetic AR(1) data.

---

### Phase 3: Generalization

#### [NEW] `notebooks/10_second_domain_replication.ipynb`
- **Task 6:** Second domain is a thresholded persistent level series (air quality vs limit), not credit default.

## Verification Plan

### Automated Tests
- Test dynamic generation using a fixed seed, N >= 10k, and asserting that the empirical AUC matches `1/2 + (2/pi)*arcsin(phi/sqrt(2))` within standard error tolerance.
