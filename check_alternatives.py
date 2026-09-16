"""Rule out the ordinary explanations for the model's 0.98 ROC-AUC.

Writes results/alternative_checks.csv (columns: check, fold, metric, value, note)
and results/xgboost_settings.json, and prints a summary.

No new modelling choices: the same model, features, seed and walk-forward
split as the main run (`tfi.modeling.breach_fit_eval`).

Checks
  1. overfitting        train vs test ROC-AUC per fold, plus the capacity settings
  2. leakage            every feature at month t recomputed from data truncated at t
                        must equal the stored feature; target must be shift(-1)
  3. current_value_only the same XGBoost on one feature (current share within 18 weeks)
  4. class_imbalance    pointer to the existing PR-AUC and rebalancing numbers

Folds are named by feature month; each is scored on the following month's outcome.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tfi import modeling as M  # noqa: E402

RESULTS = Path(__file__).parent / "results"
N_FOLDS = 4
RAW_COLS = ["provider_code", "specialty_code", "period_date", "total_waiting",
            "breach_rate", "over_52wk", "over_104wk", "pct_within_18wk"]
CAPACITY_KEYS = ["max_depth", "learning_rate", "n_estimators", "subsample",
                 "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda",
                 "gamma", "random_state", "eval_metric"]


def fold_name(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def pooled_scores(windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    y = np.concatenate([w["y"] for w in windows]).astype(int)
    p = np.concatenate([w["proba"] for w in windows])
    return y, p


# ---------------------------------------------------------------------------
# Check 1: overfitting
# ---------------------------------------------------------------------------
def check_overfitting(windows: list[dict]) -> tuple[list[dict], dict]:
    rows, gaps = [], []
    for w in windows:
        f = fold_name(w["test_month"])
        gap = w["train_auc"] - w["test_auc"]
        gaps.append(gap)
        rows += [
            {"check": "overfitting", "fold": f, "metric": "train_roc_auc", "value": w["train_auc"],
             "note": f"model scored on its own {w['n_train']} training rows"},
            {"check": "overfitting", "fold": f, "metric": "test_roc_auc", "value": w["test_auc"],
             "note": f"{w['n_test']} test rows; same number as the main run"},
            {"check": "overfitting", "fold": f, "metric": "train_minus_test", "value": gap,
             "note": "positive = fits training rows better than test rows"},
        ]
    rows += [
        {"check": "overfitting", "fold": "pooled", "metric": "train_minus_test_mean", "value": float(np.mean(gaps)),
         "note": "mean gap across the four folds"},
        {"check": "overfitting", "fold": "pooled", "metric": "train_minus_test_range",
         "value": float(np.max(gaps) - np.min(gaps)), "note": "max gap minus min gap across folds"},
    ]

    params = windows[0]["clf"].get_params()
    # Effective values as trained (XGBoost leaves unset params as None in get_params()).
    booster_cfg = json.loads(windows[0]["clf"].get_booster().save_config())
    train_param = booster_cfg["learner"]["gradient_booster"]["tree_train_param"]
    def f32(key: str) -> float:  # booster stores float32; round away the representation noise
        return round(float(train_param[key]), 6)
    effective = {"max_depth": int(train_param["max_depth"]), "learning_rate": f32("learning_rate"),
                 "n_estimators": int(params["n_estimators"]), "subsample": f32("subsample"),
                 "colsample_bytree": f32("colsample_bytree"), "min_child_weight": f32("min_child_weight"),
                 "reg_alpha": f32("reg_alpha"), "reg_lambda": f32("reg_lambda"), "gamma": f32("gamma"),
                 "random_state": params["random_state"], "eval_metric": params["eval_metric"]}
    settings = {k: effective[k] for k in CAPACITY_KEYS}
    settings["n_trees_in_booster"] = int(booster_cfg["learner"]["gradient_booster"]["gbtree_model_param"]["num_trees"])
    settings["n_features"] = len(windows[0]["feature_cols"])
    settings["early_stopping"] = "not used (no validation set; fixed n_estimators)"
    settings["early_stopping_rounds"] = params.get("early_stopping_rounds")
    settings["fit_call"] = "tfi.modeling.breach_fit_eval -> tfi.modeling.classifier(n_estimators=200)"
    settings["train_rows_per_fold"] = {fold_name(w["test_month"]): int(w["n_train"]) for w in windows}
    for k, v in settings.items():
        if isinstance(v, (int, float, str)) and k != "fit_call":
            rows.append({"check": "overfitting", "fold": "settings", "metric": k, "value": v,
                         "note": "XGBoost capacity setting used by the main run"})
    return rows, settings


# ---------------------------------------------------------------------------
# Check 2: leakage
# ---------------------------------------------------------------------------
def recompute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Re-derive every FEATURE_COL from raw measures, mirroring notebook 03 exactly."""
    d = raw.sort_values(M.SERIES + ["period_date"]).reset_index(drop=True).copy()
    d["month"] = d["period_date"].dt.month
    d["quarter"] = d["period_date"].dt.quarter
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["over_52_share"] = d["over_52wk"] / d["total_waiting"]
    d["over_104_share"] = d["over_104wk"] / d["total_waiting"]
    grp = d.groupby(M.SERIES)
    for k in (1, 2, 3, 12):
        d[f"lag{k}_total"] = grp["total_waiting"].shift(k)
    d["lag1_breach"] = grp["breach_rate"].shift(1)
    d["lag12_breach"] = grp["breach_rate"].shift(12)
    for w in (3, 6, 12):
        d[f"roll{w}_total"] = grp["total_waiting"].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    d["roll3_std_total"] = grp["total_waiting"].transform(lambda s: s.rolling(3, min_periods=2).std())
    d["mom_change_total"] = d["total_waiting"] - d["lag1_total"]
    d["mom_pct_total"] = d["total_waiting"] / d["lag1_total"] - 1
    d["yoy_pct_total"] = d["total_waiting"] / d["lag12_total"] - 1
    d[["mom_pct_total", "yoy_pct_total"]] = d[["mom_pct_total", "yoy_pct_total"]].replace([np.inf, -np.inf], np.nan)
    return d


def check_leakage(df: pd.DataFrame, months: list) -> list[dict]:
    rows = []
    d = df.sort_values(M.SERIES + ["period_date"]).reset_index(drop=True)
    total_checked = 0
    for month in months:
        f = fold_name(month)
        truncated = d.loc[d["period_date"] <= month, RAW_COLS]
        recomputed = recompute_features(truncated)
        recomputed = recomputed[recomputed["period_date"] == month].set_index(M.SERIES).sort_index()
        stored = d[d["period_date"] == month].set_index(M.SERIES).sort_index()
        assert recomputed.index.equals(stored.index), f"{f}: series mismatch after truncation"
        bad_cols = []
        for col in M.FEATURE_COLS:
            a = stored[col].to_numpy(dtype=float)
            b = recomputed[col].to_numpy(dtype=float)
            if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                bad_cols.append(col)
        assert not bad_cols, f"{f}: features differ when future months are removed: {bad_cols}"
        n = len(stored)
        total_checked += n
        rows.append({"check": "leakage", "fold": f, "metric": "rows_recomputed_from_truncated_data", "value": n,
                     "note": f"all {len(M.FEATURE_COLS)} features equal stored values (rtol 1e-9, NaN-equal)"})

    # Target alignment over the whole frame.
    grp = d.groupby(M.SERIES)
    exp_pct = grp["pct_within_18wk"].shift(-1)
    exp_total = grp["total_waiting"].shift(-1)
    assert np.allclose(d["target_pct_within_18_next"], exp_pct, equal_nan=True), "target_pct is not shift(-1)"
    assert np.allclose(d["target_total_next"], exp_total, equal_nan=True), "target_total is not shift(-1)"
    has = exp_pct.notna()
    exp_breach = (exp_pct[has] < M.STANDARD).astype(int)
    assert d.loc[has, "target_breach_next"].notna().all(), "target_breach missing where next month exists"
    assert (d.loc[has, "target_breach_next"].astype(int).to_numpy() == exp_breach.to_numpy()).all(), \
        "target_breach_next is not (next month share < standard)"
    assert d.loc[~has, "target_breach_next"].isna().all(), "target_breach defined without a next month"

    last = d["period_date"].max()
    last_rows = d[d["period_date"] == last]
    assert last_rows["target_breach_next"].isna().all(), f"{fold_name(last)} rows carry a target"
    rows += [
        {"check": "leakage", "fold": "pooled", "metric": "rows_recomputed_from_truncated_data", "value": total_checked,
         "note": "sum over the four test months; passed"},
        {"check": "leakage", "fold": "pooled", "metric": "target_is_shift_minus_1_rows", "value": len(d),
         "note": "target equals next month's value within each provider-by-specialty series; passed"},
        {"check": "leakage", "fold": "pooled", "metric": f"{fold_name(last)}_rows_without_target", "value": len(last_rows),
         "note": f"all rows in the final month ({fold_name(last)}) have no target; passed"},
        {"check": "leakage", "fold": "pooled", "metric": "status", "value": "passed",
         "note": "features at t use only months <= t; target is month t+1"},
    ]
    return rows


# ---------------------------------------------------------------------------
# Check 3: current-value-only model
# ---------------------------------------------------------------------------
def check_current_value_only(df: pd.DataFrame, months: list, full_windows: list[dict]) -> list[dict]:
    single = [M.breach_fit_eval(df, m, feature_cols=["pct_within_18wk"]) for m in months]
    rows = []
    for w in single:
        rows.append({"check": "current_value_only", "fold": fold_name(w["test_month"]), "metric": "test_roc_auc",
                     "value": w["test_auc"], "note": "XGBoost, single feature: current share within 18 weeks"})
    y1, p1 = pooled_scores(single)
    y_full, p_full = pooled_scores(full_windows)
    assert np.array_equal(y1, y_full), "single-feature and full-model test sets differ"
    rows += [
        {"check": "current_value_only", "fold": "pooled", "metric": "roc_auc", "value": roc_auc_score(y1, p1),
         "note": "single feature; same folds, seed and settings as the full model"},
        {"check": "current_value_only", "fold": "pooled", "metric": "pr_auc", "value": average_precision_score(y1, p1),
         "note": "single feature"},
        {"check": "current_value_only", "fold": "pooled", "metric": "full_model_roc_auc", "value": roc_auc_score(y_full, p_full),
         "note": f"all {len(M.FEATURE_COLS)} features; reference from the same run"},
        {"check": "current_value_only", "fold": "pooled", "metric": "full_model_pr_auc",
         "value": average_precision_score(y_full, p_full), "note": "reference from the same run"},
        {"check": "current_value_only", "fold": "pooled", "metric": "roc_auc_share_of_full_model_above_chance",
         "value": (roc_auc_score(y1, p1) - 0.5) / (roc_auc_score(y_full, p_full) - 0.5),
         "note": "(single - 0.5) / (full - 0.5)"},
    ]
    return rows


# ---------------------------------------------------------------------------
# Check 4: class imbalance pointer
# ---------------------------------------------------------------------------
def check_class_imbalance() -> list[dict]:
    rows = []
    mm = pd.read_csv(RESULTS / "multi_metric_comparison.csv").set_index("predictor")
    rows.append({"check": "class_imbalance", "fold": "pooled", "metric": "positive_rate",
                 "value": mm.loc["xgboost", "positive_rate_baseline_PR_AUC"],
                 "note": "breach rate on the pooled test set = PR-AUC no-skill baseline (multi_metric_comparison.csv)"})
    for pred in ["xgboost", "binary_persistence", "continuous_persistence"]:
        rows.append({"check": "class_imbalance", "fold": "pooled", "metric": f"pr_auc_{pred}",
                     "value": mm.loc[pred, "PR_AUC"], "note": "multi_metric_comparison.csv"})
    bal = pd.read_csv(RESULTS / "balance_sweep.csv")
    for _, r in bal.iterrows():
        rows.append({"check": "class_imbalance", "fold": "pooled", "metric": f"rebalance_gap_at_rate_{r['target_breach_rate']:.2f}",
                     "value": r["gap_mean"],
                     "note": "observed minus formula continuous-persistence AUC at an imposed breach rate (balance_sweep.csv)"})
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = M.load_features()
    months = M.valid_target_months(df, "target_breach_next")[-N_FOLDS:]
    print(f"folds (feature months): {[fold_name(m) for m in months]}")

    windows = [M.breach_fit_eval(df, m) for m in months]
    rows, settings = check_overfitting(windows)
    rows += check_leakage(df, months)
    rows += check_current_value_only(df, months, windows)
    rows += check_class_imbalance()

    out = pd.DataFrame(rows, columns=["check", "fold", "metric", "value", "note"])
    out.to_csv(RESULTS / "alternative_checks.csv", index=False)
    with open(RESULTS / "xgboost_settings.json", "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)

    print("\n=== Check 1: overfitting ===")
    print(out[(out.check == "overfitting") & (out.fold != "settings")]
          .pivot(index="fold", columns="metric", values="value").to_string(float_format=lambda v: f"{v:.4f}"))
    print("settings:", {k: settings[k] for k in CAPACITY_KEYS}, "| early stopping:", settings["early_stopping"])
    print("\n=== Check 2: leakage ===")
    for _, r in out[out.check == "leakage"].iterrows():
        print(f"  {r.fold:>7}  {r.metric:<45} {fmt(r.value):>8}  {r.note}")
    print("\n=== Check 3: current-value-only model ===")
    for _, r in out[out.check == "current_value_only"].iterrows():
        print(f"  {r.fold:>7}  {r.metric:<45} {fmt(r.value):>8}  {r.note}")
    print("\n=== Check 4: class imbalance (existing numbers) ===")
    for _, r in out[out.check == "class_imbalance"].iterrows():
        print(f"  {r.fold:>7}  {r.metric:<45} {fmt(r.value):>8}")
    print(f"\nSaved: {RESULTS / 'alternative_checks.csv'}, {RESULTS / 'xgboost_settings.json'}")


if __name__ == "__main__":
    main()
