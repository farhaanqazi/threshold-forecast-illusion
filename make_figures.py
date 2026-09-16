"""Regenerate every data figure in the Prediction Illusion paper from results/.

Reads only files under results/ and writes 300 dpi PNGs to figures/.
No number on any figure is typed into this script.

Paper numbering (Figs 1 and 2 are schematics, drawn elsewhere):
    fig3_monthly.png        per-month ROC-AUC, model vs binary persistence
    fig4_metrics.png        pooled ROC-AUC and PR-AUC with bootstrap 95% CIs
    fig5_formula.png        arcsine formula curve, synthetic points, NHS gap
    fig6_tests.png          (a) rebalancing test, (b) rho-tercile test
Additional figures (file names descriptive; paper numbers assigned in the text):
    fig_example_series.png  four departments' share within 18 weeks over time
    fig_roc_curves.png      pooled ROC curves for the three predictors
    fig_transition_matrix.png  2x2 breach transition matrix
    fig_rho_histogram.png   lag-1 autocorrelation across series
    fig_calibration.png     reliability diagram with ECE

Inputs produced by the main NHS run: persistence_vs_model.csv, nhs_auc_ci.csv,
multi_metric_comparison.csv, theorem_vs_observed.csv, synthetic_sweep_extended.csv,
balance_sweep.csv, rho_tercile_gap.csv, rho_distribution.csv, model_calibration.csv.
Inputs produced by export_figure_inputs.py: pooled_predictions.csv,
example_series.csv, breach_transitions.csv.

Month convention: months in results/ are FEATURE months. A row "2026-02" holds
February 2026 observations and is scored on the March 2026 outcome. Axes that
show months say so explicitly.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DPI = 300

K = "#222222"   # black: observed values
G = "#888888"   # grey: persistence
B = "#1f4e79"   # dark blue: model
LIGHT = "#c8c8c8"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def formula(rho):
    return 0.5 + (2.0 / np.pi) * np.arcsin(np.asarray(rho) / np.sqrt(2.0))


def month_name(ym: str) -> str:
    return pd.Timestamp(ym + "-01").strftime("%b %Y")


def next_month_name(ym: str) -> str:
    return (pd.Timestamp(ym + "-01") + pd.DateOffset(months=1)).strftime("%b %Y")


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=DPI)
    plt.close(fig)
    print(f"  wrote figures/{name}")


# ---------------------------------------------------------------------------
# Fig 3: per-month ROC-AUC
# ---------------------------------------------------------------------------
def fig3_monthly() -> None:
    pm = read("persistence_vs_model.csv")
    monthly = pm[pm["month"] != "pooled"].reset_index(drop=True)
    pooled_cont = read("multi_metric_comparison.csv").set_index("predictor").loc["continuous_persistence", "ROC_AUC"]

    x = np.arange(len(monthly))
    ticks = [f"{month_name(m)}\n(scored on {next_month_name(m)})" for m in monthly["month"]]
    model, pers = monthly["xgboost_auc"].to_numpy(), monthly["persistence_auc"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(x, model, "o-", color=B, label="XGBoost model")
    ax.plot(x, pers, "s--", color=G, label="Binary persistence")
    ax.axhline(pooled_cont, color=K, lw=0.8, ls=":", label="Continuous persistence (pooled)")
    for xi, m, p in zip(x, model, pers):
        ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=B)
        ax.annotate(f"{p:.3f}", (xi, p), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color=G)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Feature month (features observed in this month, label from the next)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig3_monthly.png")


# ---------------------------------------------------------------------------
# Fig 4: pooled metrics with bootstrap CIs
# ---------------------------------------------------------------------------
def fig4_metrics() -> None:
    ci = read("nhs_auc_ci.csv").set_index("metric")
    mm = read("multi_metric_comparison.csv").set_index("predictor")
    preds = ["xgboost", "binary_persistence", "continuous_persistence"]
    labels = ["Model", "Binary\npersistence", "Continuous\npersistence"]

    roc = ci.loc[preds, "auc"].to_numpy()
    roc_err = [roc - ci.loc[preds, "ci_lower_95"].to_numpy(), ci.loc[preds, "ci_upper_95"].to_numpy() - roc]
    pr = mm.loc[preds, "PR_AUC"].to_numpy()
    pr_err = [pr - mm.loc[preds, "PR_AUC_CI_lower_95"].to_numpy(), mm.loc[preds, "PR_AUC_CI_upper_95"].to_numpy() - pr]
    no_skill = mm.loc["xgboost", "positive_rate_baseline_PR_AUC"]
    n_boot = int(mm.loc["xgboost", "n_bootstraps"])

    x, w = np.arange(len(preds)), 0.36
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.bar(x - w / 2, roc, w, color=B, label="ROC-AUC", yerr=roc_err, capsize=3, ecolor=K)
    ax.bar(x + w / 2, pr, w, color=G, label="PR-AUC", yerr=pr_err, capsize=3, ecolor=K)
    for i in range(len(preds)):
        ax.text(x[i] - w / 2, roc[i] + 0.008, f"{roc[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, pr[i] + 0.008, f"{pr[i]:.3f}", ha="center", fontsize=8)
    ax.axhline(no_skill, color=K, lw=0.8, ls=":")
    ax.text(x[-1] + w, no_skill + 0.003, f"PR-AUC no-skill\n(breach rate {no_skill:.3f})", fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("Score on pooled test set")
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0),
              title=f"Error bars: bootstrap 95% CI ({n_boot} resamples)", title_fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_metrics.png")


# ---------------------------------------------------------------------------
# Fig 5: formula curve, synthetic validation, NHS gap
# ---------------------------------------------------------------------------
def fig5_formula() -> None:
    theo = read("theorem_vs_observed.csv").set_index("reference")
    median_rho = theo.loc["median_rho", "rho"]
    pred = theo.loc["median_rho", "theorem_predicted_auc"]
    obs = theo.loc["median_rho", "observed_continuous_persistence_auc"]
    synth = read("synthetic_sweep_extended.csv")

    rho = np.linspace(-0.70, 0.99, 400)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(rho, formula(rho), color=B, label="Formula: AUC = 1/2 + (2/π) arcsin(ρ/√2)")
    ax.scatter(synth["empirical_rho"], synth["observed_AUC"], marker="x", s=28, color=K, lw=0.9, zorder=3,
               label=f"Synthetic AR(1) simulation (n = {int(synth['n'].iloc[0]):,} per point)")
    ax.plot([median_rho], [pred], "o", color=B, zorder=4, ms=6)
    ax.plot([median_rho], [obs], "s", color=K, zorder=4, ms=6)
    ax.vlines(median_rho, pred, obs, color=K, lw=0.9)
    ax.text(median_rho + 0.02, (pred + obs) / 2, f"gap {obs - pred:.3f}", fontsize=8, va="center")
    ax.annotate(f"Observed NHS continuous persistence: {obs:.3f}", (median_rho, obs),
                xytext=(-0.68, 0.96), fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=G, shrinkB=5))
    ax.annotate(f"Formula at NHS median ρ = {median_rho:.3f}: {pred:.3f}", (median_rho, pred),
                xytext=(-0.68, 0.86), fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=G, shrinkB=5))
    ax.set_xlabel("Lag-1 autocorrelation ρ")
    ax.set_ylabel("Continuous-persistence ROC-AUC")
    ax.set_ylim(0.15, 1.02)
    ax.set_xlim(-0.75, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig5_formula.png")


# ---------------------------------------------------------------------------
# Fig 6: rebalancing test and tercile test
# ---------------------------------------------------------------------------
def fig6_tests() -> None:
    bal = read("balance_sweep.csv")
    ter = read("rho_tercile_gap.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.1))
    # (a) rebalancing
    obs, frm = bal["observed_AUC_mean"].to_numpy(), bal["theorem_AUC"].to_numpy()
    xx = np.arange(len(bal))
    a1.bar(xx - 0.18, obs, 0.36, color=K, label="Observed")
    a1.bar(xx + 0.18, frm, 0.36, color=G, label="Formula")
    for i in range(len(bal)):
        a1.text(xx[i], max(obs[i], frm[i]) + 0.008, f"gap {obs[i] - frm[i]:.3f}", ha="center", fontsize=7)
    a1.set_xticks(xx)
    a1.set_xticklabels([f"{r:.2f}" for r in bal["target_breach_rate"]])
    a1.set_xlabel("Imposed breach rate")
    a1.set_ylabel("Continuous-persistence ROC-AUC")
    a1.set_ylim(0.80, 1.0)
    a1.set_title("(a) Rebalancing test", fontsize=9)

    # (b) terciles
    def label(group: str, rng: str) -> str:
        # rho_range is "min-max"; either bound may itself be negative.
        m = re.fullmatch(r"(-?\d+\.\d+)-(-?\d+\.\d+)", rng.strip())
        if m is None:
            raise ValueError(f"unexpected rho_range format: {rng!r}")
        lo, hi = float(m.group(1)), float(m.group(2))
        return f"{group.capitalize()}\n({lo:.2f} to {hi:.2f})"
    ob, fo = ter["observed_AUC"].to_numpy(), ter["theorem_AUC"].to_numpy()
    xx = np.arange(len(ter))
    a2.bar(xx - 0.18, ob, 0.36, color=K, label="Observed")
    a2.bar(xx + 0.18, fo, 0.36, color=G, label="Formula")
    for i in range(len(ter)):
        a2.text(xx[i], max(ob[i], fo[i]) + 0.008, f"gap {ob[i] - fo[i]:.3f}", ha="center", fontsize=7)
    a2.set_xticks(xx)
    a2.set_xticklabels([label(g, r) for g, r in zip(ter["group"], ter["rho_range"])], fontsize=7)
    a2.set_xlabel("Lag-1 autocorrelation tercile (ρ range)")
    a2.set_ylim(0.70, 1.03)
    a2.set_title("(b) By autocorrelation tercile", fontsize=9)

    h, l = a2.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig6_tests.png")


# ---------------------------------------------------------------------------
# Example series
# ---------------------------------------------------------------------------
def fig_example_series() -> None:
    ex = read("example_series.csv")
    ex["period_date"] = pd.to_datetime(ex["period_date"] + "-01")
    styles = {
        "persistently in breach": dict(color=K, ls="-", marker="o"),
        "persistently compliant": dict(color=B, ls="-", marker="s"),
        "crosses the threshold": dict(color=G, ls="-", marker="^"),
        "hovers near the threshold": dict(color=K, ls="--", marker="d"),
    }
    def tidy(name: str) -> str:
        return " ".join(w.upper() if w.lower() in {"nhs", "uk"} else w.capitalize() for w in name.split())

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for role, sub in ex.groupby("role", sort=False):
        sub = sub.sort_values("period_date")
        ax.plot(sub["period_date"], 100 * sub["pct_within_18wk"], ms=3.5, lw=1.2,
                label=f"{tidy(sub['provider_name'].iloc[0])}, {sub['specialty_name'].iloc[0]} ({role})",
                **styles[role])
    ax.axhline(92, color=K, lw=0.9, ls=":")
    ax.text(ex["period_date"].min(), 92.4, "92% standard", fontsize=7, va="bottom")
    ax.set_ylabel("Share of pathways within 18 weeks (%)")
    ax.set_xlabel("Observation month")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    save(fig, "fig_example_series.png")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
def fig_roc_curves() -> None:
    p = read("pooled_predictions.csv")
    y = p["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([0, 1], [0, 1], color=LIGHT, ls=":", lw=1, label="Chance")
    for col, lab, kw in [
        ("model_proba", "XGBoost model", dict(color=B, lw=1.4)),
        ("continuous_persistence", "Continuous persistence", dict(color=G, lw=1.4)),
        ("binary_persistence", "Binary persistence", dict(color=K, lw=1.0, marker="o", ms=4)),
    ]:
        fpr, tpr, _ = roc_curve(y, p[col])
        ax.plot(fpr, tpr, label=f"{lab} (AUC = {roc_auc_score(y, p[col]):.3f})", **kw)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              title=f"Pooled test set, n = {len(y):,}", title_fontsize=8)
    fig.tight_layout()
    save(fig, "fig_roc_curves.png")


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------
def fig_transition_matrix() -> None:
    t = read("breach_transitions.csv")
    counts = t.pivot(index="this_month_state", columns="next_month_state", values="count").loc[[1, 0], [1, 0]]
    pct = t.pivot(index="this_month_state", columns="next_month_state", values="row_pct").loc[[1, 0], [1, 0]]
    labels = ["Breach (<92%)", "Compliant (≥92%)"]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    fig.subplots_adjust(left=0.32, right=0.98, bottom=0.17, top=0.90)
    ax.imshow(pct.to_numpy(), cmap="Greys", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            v = pct.iloc[i, j]
            ax.text(j, i, f"{int(counts.iloc[i, j]):,}\n({v:.1f}% of row)", ha="center", va="center",
                    fontsize=10, color="white" if v > 55 else K)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("State next month (the label)")
    ax.set_ylabel("State this month (the feature)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Breach transitions, all labelled rows (n = {int(counts.to_numpy().sum()):,})", fontsize=9)
    save(fig, "fig_transition_matrix.png")


# ---------------------------------------------------------------------------
# Histogram of lag-1 rho
# ---------------------------------------------------------------------------
def fig_rho_histogram() -> None:
    rho = read("rho_distribution.csv")["rho"].to_numpy()
    median = read("theorem_vs_observed.csv").set_index("reference").loc["median_rho", "rho"]
    lo_cut, hi_cut = np.quantile(rho, [1 / 3, 2 / 3])
    min_obs = int(read("rho_distribution.csv")["n_obs"].min())

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.hist(rho, bins=np.arange(-0.7, 1.0001, 0.025), color=LIGHT, edgecolor="white", lw=0.4)
    ax.axvline(median, color=K, lw=1.2, label=f"Median ρ = {median:.3f}")
    ax.axvline(lo_cut, color=B, lw=0.9, ls="--", label=f"Tercile cut points: {lo_cut:.3f}, {hi_cut:.3f}")
    ax.axvline(hi_cut, color=B, lw=0.9, ls="--")
    ax.set_xlabel("Lag-1 autocorrelation ρ of share within 18 weeks")
    ax.set_ylabel("Number of series")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              title=f"{len(rho):,} series with ≥ {min_obs} observations", title_fontsize=8)
    fig.tight_layout()
    save(fig, "fig_rho_histogram.png")


# ---------------------------------------------------------------------------
# Calibration reliability diagram
# ---------------------------------------------------------------------------
def fig_calibration() -> None:
    cal = read("model_calibration.csv").dropna()
    n = cal["count"].sum()
    ece = float((cal["count"] / n * (cal["mean_predicted"] - cal["observed_frequency"]).abs()).sum())
    n_bins = len(read("model_calibration.csv"))
    edges = np.linspace(0, 1, n_bins + 1)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax2 = ax.twinx()
    ax2.bar((edges[:-1] + edges[1:]) / 2, cal["count"], width=1 / n_bins * 0.9, color=LIGHT, alpha=0.6, zorder=0)
    ax2.set_ylabel("Rows in bin", color=G)
    ax2.tick_params(axis="y", colors=G)
    ax2.spines["top"].set_visible(False)
    ax2.set_yscale("log")

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot([0, 1], [0, 1], color=K, ls=":", lw=0.9, label="Perfect calibration")
    ax.plot(cal["mean_predicted"], cal["observed_frequency"], "-", color=B, lw=1.0)
    ax.scatter(cal["mean_predicted"], cal["observed_frequency"], s=18 + 60 * np.sqrt(cal["count"] / cal["count"].max()),
               color=B, zorder=5, clip_on=False, label="Model (point size ~ √ rows in bin)")
    ax.text(0.03, 0.97, f"ECE = {ece:.3f}\n{n_bins} equal-width bins, n = {n:,}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8)
    ax.set_xlabel("Mean predicted breach probability")
    ax.set_ylabel("Observed breach rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_calibration.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    print("Part 1")
    fig3_monthly()
    fig4_metrics()
    fig5_formula()
    fig6_tests()
    print("Part 2")
    fig_example_series()
    fig_roc_curves()
    fig_transition_matrix()
    fig_rho_histogram()
    fig_calibration()
    print("done")


if __name__ == "__main__":
    main()
