"""Calibration and per-segment gates for the breach-risk model.

A risk score is only useful if it is calibrated (a 70 means ~70% chance) and if it
holds up within segments, not just on average.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from hcip import modeling as M

MAX_ECE = 0.05            # expected calibration error
MAX_BRIER = 0.10          # Brier score
MIN_SPECIALTY_AUC = 0.85  # per-specialty floor
MIN_TERCILE_AUC = 0.90    # per provider-size floor


def test_breach_is_calibrated(breach_ho):
    y, p = breach_ho["y"], breach_ho["proba"]
    assert M.expected_calibration_error(y, p) <= MAX_ECE
    assert M.brier(y, p) <= MAX_BRIER


def test_breach_auc_per_specialty(breach_ho):
    """No major specialty may fall below the per-segment AUC floor."""
    te = breach_ho["test_df"].assign(proba=breach_ho["proba"], y=breach_ho["y"])
    top = te.groupby("specialty_name")["total_waiting"].sum().sort_values(ascending=False).head(10).index
    for specialty in top:
        sub = te[te["specialty_name"] == specialty]
        if sub["y"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["y"], sub["proba"])
        assert auc >= MIN_SPECIALTY_AUC, f"{specialty}: AUC {auc:.3f}"


def test_breach_auc_per_provider_size(breach_ho):
    """Performance must hold across small / mid / large providers."""
    te = breach_ho["test_df"].assign(proba=breach_ho["proba"], y=breach_ho["y"])
    size = te.groupby("provider_code")["total_waiting"].transform("sum")
    te = te.assign(tercile=pd.qcut(size, 3, labels=["small", "mid", "large"]))
    for tercile, sub in te.groupby("tercile", observed=True):
        if sub["y"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["y"], sub["proba"])
        assert auc >= MIN_TERCILE_AUC, f"{tercile}: AUC {auc:.3f}"
