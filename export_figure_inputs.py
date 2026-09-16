"""Export the model-dependent inputs that the paper figures need into results/.

Everything `make_figures.py` draws comes from a results/ file. Most of those
files are produced by the main NHS run; the four written here need the
labelled feature frame or the fitted model rather than an existing table:

    results/pooled_predictions.csv   labels + scores on the pooled walk-forward test set
    results/breach_transitions.csv   2x2 this-month -> next-month breach transition counts
    results/example_series.csv       four departments' full monthly share-within-18-weeks series
    results/data_sample.csv          six labelled rows for the data-sample table

Same model, features, seed and walk-forward split as the main run
(`tfi.modeling.breach_fit_eval`, last four feature months with a label).

Month convention: `feature_month` is the month the features were observed;
`outcome_month` (feature month + 1) is the month whose breach state is the label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tfi import modeling as M  # noqa: E402

RESULTS = Path(__file__).parent / "results"
N_FOLDS = 4


def month_label(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def export_pooled_predictions(df: pd.DataFrame) -> pd.DataFrame:
    months = M.valid_target_months(df, "target_breach_next")[-N_FOLDS:]
    parts = []
    for month in months:
        w = M.breach_fit_eval(df, month)
        te = w["test_df"]
        parts.append(pd.DataFrame({
            "provider_code": te["provider_code"].to_numpy(),
            "specialty_code": te["specialty_code"].to_numpy(),
            "feature_month": month_label(month),
            "outcome_month": month_label(pd.Timestamp(month) + pd.DateOffset(months=1)),
            "y": w["y"].astype(int),
            "model_proba": w["proba"],
            "pct_within_18wk": te["pct_within_18wk"].to_numpy(),
            "binary_persistence": (te["pct_within_18wk"] < M.STANDARD).astype(int).to_numpy(),
            "continuous_persistence": (1.0 - te["pct_within_18wk"]).to_numpy(),
        }))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULTS / "pooled_predictions.csv", index=False)
    return out


def export_transitions(labelled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for this_state in (0, 1):
        sub = labelled[labelled["breach_this_month"] == this_state]
        for next_state in (0, 1):
            n = int((sub["breach_next_month"] == next_state).sum())
            rows.append({
                "this_month_state": this_state,
                "next_month_state": next_state,
                "count": n,
                "row_pct": 100.0 * n / len(sub) if len(sub) else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "breach_transitions.csv", index=False)
    return out


def choose_example_series(df: pd.DataFrame) -> pd.DataFrame:
    """Pick four departments with complete series that span the persistence patterns.

    Selection is deterministic: within each pattern the largest department (by
    mean waiting list) is taken so that the names are recognisable.
    """
    n_months = df["period_date"].nunique()
    g = df.dropna(subset=["pct_within_18wk"]).sort_values("period_date").groupby(M.SERIES)
    stats = g.agg(n_obs=("pct_within_18wk", "size"),
                  min_val=("pct_within_18wk", "min"),
                  max_val=("pct_within_18wk", "max"),
                  mean_val=("pct_within_18wk", "mean"),
                  std_val=("pct_within_18wk", "std"),
                  mean_total=("total_waiting", "mean"),
                  provider_name=("provider_name", "first"),
                  specialty_name=("specialty_name", "first"))
    crossings = g["pct_within_18wk"].apply(
        lambda s: int((np.sign(s.to_numpy() - M.STANDARD)[1:] != np.sign(s.to_numpy() - M.STANDARD)[:-1]).sum()))
    stats["n_cross"] = crossings
    stats = stats[stats["n_obs"] == n_months]

    def pick(mask: pd.Series, role: str) -> pd.Series:
        cands = stats[mask]
        if cands.empty:
            raise RuntimeError(f"No candidate series for role '{role}'.")
        row = cands.sort_values("mean_total", ascending=False).iloc[0].copy()
        row["role"] = role
        return row

    chosen = [
        pick(stats["max_val"] < 0.90, "persistently in breach"),
        pick(stats["min_val"] > 0.94, "persistently compliant"),
        pick((stats["n_cross"] == 1) & (stats["min_val"] < 0.88) & (stats["max_val"] > 0.95), "crosses the threshold"),
        pick((stats["n_cross"] >= 5) & stats["mean_val"].between(0.90, 0.94) & (stats["std_val"] < 0.03),
             "hovers near the threshold"),
    ]
    parts = []
    for row in chosen:
        prov, spec = row.name
        sub = df[(df["provider_code"] == prov) & (df["specialty_code"] == spec)].sort_values("period_date")
        parts.append(pd.DataFrame({
            "role": row["role"],
            "provider_code": prov,
            "provider_name": row["provider_name"],
            "specialty_code": spec,
            "specialty_name": row["specialty_name"],
            "period_date": sub["period_date"].dt.strftime("%Y-%m").to_numpy(),
            "pct_within_18wk": sub["pct_within_18wk"].to_numpy(),
        }))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULTS / "example_series.csv", index=False)
    return out


def export_data_sample(labelled: pd.DataFrame, example: pd.DataFrame) -> pd.DataFrame:
    """Six real labelled rows drawn from the four example departments.

    One steady non-breach row, one steady breach row, the month before and the
    month of the crossing department's flip, and one flip in each direction from
    the hovering department.
    """
    def series_rows(role: str) -> pd.DataFrame:
        key = example[example["role"] == role].iloc[0]
        sub = labelled[(labelled["provider_code"] == key["provider_code"])
                       & (labelled["specialty_code"] == key["specialty_code"])]
        return sub.sort_values("period_date")

    steady_ok = series_rows("persistently compliant")
    steady_breach = series_rows("persistently in breach")
    crossing = series_rows("crosses the threshold")
    hover = series_rows("hovers near the threshold")

    flip = crossing["breach_this_month"] != crossing["breach_next_month"]
    flip_pos = int(np.flatnonzero(flip.to_numpy())[0])
    hover_flip = hover[hover["breach_this_month"] != hover["breach_next_month"]]
    hover_pair = pd.concat([hover_flip[hover_flip["breach_this_month"] == 0].iloc[[0]],
                            hover_flip[hover_flip["breach_this_month"] == 1].iloc[[0]]]).sort_values("period_date")
    picks = [
        steady_ok.iloc[[0]],
        steady_breach.iloc[[0]],
        crossing.iloc[[max(flip_pos - 1, 0), flip_pos]],
        hover_pair,
    ]
    sample = pd.concat(picks).drop_duplicates(subset=M.SERIES + ["period_date"]).head(6)
    assert (sample["breach_this_month"] == 0).any(), "sample needs a non-breach row"
    assert (sample["breach_this_month"] != sample["breach_next_month"]).any(), "sample needs a flip"
    out = pd.DataFrame({
        "provider_name": sample["provider_name"].to_numpy(),
        "specialty_name": sample["specialty_name"].to_numpy(),
        "month": sample["period_date"].dt.strftime("%Y-%m").to_numpy(),
        "share_within_18wk_pct": (100.0 * sample["pct_within_18wk"]).round(1).to_numpy(),
        "breach_this_month": sample["breach_this_month"].to_numpy(),
        "breach_next_month": sample["breach_next_month"].to_numpy(),
    })
    out.to_csv(RESULTS / "data_sample.csv", index=False)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = M.load_features()
    labelled = df.dropna(subset=["target_breach_next", "pct_within_18wk"]).copy()
    labelled["breach_this_month"] = (labelled["pct_within_18wk"] < M.STANDARD).astype(int)
    labelled["breach_next_month"] = labelled["target_breach_next"].astype(int)

    preds = export_pooled_predictions(df)
    print(f"pooled_predictions.csv: {len(preds)} rows, feature months "
          f"{sorted(preds['feature_month'].unique())}")

    trans = export_transitions(labelled)
    print("breach_transitions.csv:")
    print(trans.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    example = choose_example_series(df)
    print("example_series.csv (provider | specialty | role):")
    for role, sub in example.groupby("role", sort=False):
        print(f"  {sub['provider_name'].iloc[0]} | {sub['specialty_name'].iloc[0]} | {role}")

    sample = export_data_sample(labelled, example)
    print("data_sample.csv:")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
