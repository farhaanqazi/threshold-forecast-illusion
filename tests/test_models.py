"""Production gates for the three models — accuracy, leakage, overfitting, stability, outputs.

Each test encodes a deploy-blocking standard. A failure here means a model must not ship.
"""
import numpy as np

from tfi import modeling as M

# --- Thresholds (principled, justified against measured performance) ---
BREACH_MIN_AUC = 0.90          # absolute floor for ranking quality
BREACH_MIN_MARGIN = 0.03       # must beat the breach-persistence baseline by this much
MAX_OVERFIT_GAP = 0.05         # train AUC - test AUC
PARITY_TOLERANCE = 1.02        # FVA forecast may be at most 2% worse than persistence


def test_no_feature_leakage(df):
    """Features must reference the past, the target must reference the future."""
    d = df.sort_values(M.SERIES + ["period_date"])
    key = d.groupby(M.SERIES).size().idxmax()  # longest (continuous) series
    s = d[(d["provider_code"] == key[0]) & (d["specialty_code"] == key[1])]
    assert np.allclose(s["lag1_total"].to_numpy()[1:], s["total_waiting"].to_numpy()[:-1]), \
        "lag1_total does not equal the previous month's value"
    assert np.allclose(s["target_total_next"].to_numpy()[:-1], s["total_waiting"].to_numpy()[1:]), \
        "target does not equal the next month's value"


def test_breach_beats_baseline(breach_windows):
    """The breach model must out-rank persistence on every walk-forward window."""
    for w in breach_windows:
        assert w["test_auc"] >= w["baseline_auc"] + BREACH_MIN_MARGIN, \
            f"{w['test_month']:%Y-%m}: AUC {w['test_auc']:.3f} vs baseline {w['baseline_auc']:.3f}"


def test_breach_auc_floor(breach_windows):
    """The breach model must clear an absolute AUC floor on every window."""
    for w in breach_windows:
        assert w["test_auc"] >= BREACH_MIN_AUC, f"{w['test_month']:%Y-%m}: AUC {w['test_auc']:.3f}"


def test_breach_not_overfit(breach_windows):
    """Train-vs-test AUC gap must stay within the overfitting tolerance on every window."""
    for w in breach_windows:
        gap = w["train_auc"] - w["test_auc"]
        assert gap <= MAX_OVERFIT_GAP, f"{w['test_month']:%Y-%m}: gap {gap:.3f}"


def test_breach_probabilities_valid(breach_ho):
    """Risk scores must be finite probabilities in [0, 1]."""
    p = breach_ho["proba"]
    assert np.isfinite(p).all()
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_breach_diagnostics_cover_transitions_and_persistence():
    """The diagnostic helper should report persistence and transition-specific ranking metrics."""
    y_true = np.array([0, 1, 1, 0])
    proba = np.array([0.1, 0.8, 0.7, 0.2])
    current_breach = np.array([0, 1, 0, 1])

    metrics = M.breach_ranking_diagnostics(y_true, proba, current_breach, k=2)

    assert metrics["transition_mask"].tolist() == [False, False, True, True]
    assert metrics["overall_precision_at_k"] == 1.0
    assert metrics["transition_precision_at_k"] == 0.5


def test_demand_not_worse_than_persistence(demand_fva):
    """The deployed demand forecast must never be meaningfully worse than the champion."""
    assert demand_fva["hybrid_mae"] <= demand_fva["persistence_mae"] * PARITY_TOLERANCE


def test_wait_not_worse_than_persistence(wait_fva):
    """The deployed waiting-time forecast must never be meaningfully worse than the champion."""
    assert wait_fva["hybrid_mae"] <= wait_fva["persistence_mae"] * PARITY_TOLERANCE


def test_training_is_deterministic(df):
    """Fixed seed + single thread must reproduce identical predictions."""
    d = df.dropna(subset=["target_breach_next", "pct_within_18wk"]).head(15000)
    x, y = d[M.FEATURE_COLS], d["target_breach_next"].astype(int)
    a = M.classifier(n_estimators=40, n_jobs=1).fit(x, y).predict_proba(x)[:, 1]
    b = M.classifier(n_estimators=40, n_jobs=1).fit(x, y).predict_proba(x)[:, 1]
    assert np.array_equal(a, b)
