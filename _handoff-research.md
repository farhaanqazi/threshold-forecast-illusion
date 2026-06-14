# Handoff — research track (read me first)

**You are now in the RESEARCH clone, branch `research`.** The sibling
dashboard repo (`healthcare-capacity-intelligence-platform`) is the
engineering track and is OFF-LIMITS for paper work. Confirm the branch
before editing anything.

## The user's real target
Building a profile to apply for an **ML-related PhD**. The goal is a
**publishable paper** (workshop / preprint / forecasting journal is fine —
not necessarily NeurIPS main track). The NHS project is a *means*, not the
end. Judge everything by: "does this produce a defensible research result?"

Chat preference: **plain English**, no jargon-soup. Repo files: **production-
grade and professional** (this handoff file is the one exception — it's a
working note, keep it gitignored / uncommitted).

## What the base project is (one line)
Medallion pipeline (Bronze→Silver→Gold star schema) over 19 months of NHS RTT
waiting-list data, grain = provider × specialty × month, + 3 models (demand,
waiting-time, breach) with a 14-gate test suite. See the dashboard repo's
README / architecture.md. Data lives in a shared `data/` (junction or copy).

## The research thesis (the whole point)
A contradiction in the existing results: the **level** series (demand,
waiting %) are near-random-walk → persistence-dominated (can barely beat
"next month = this month"). Yet the **binary breach** outcome scores
**AUC ≈ 0.98**. How does a near-random-walk become near-perfectly predictable
once you threshold it?

**Hypothesis (UNVERIFIED — verify first):** the breach label is extremely
sticky (a specialty breaching this month breaches next month), so AUC 0.98 is
mostly the classifier learning "are you breaching *now*" — i.e. persistence in
disguise. The regressors were compared to a persistence baseline; the breach
classifier likely was **not** compared to a *binary-persistence* baseline.

Three framings of one contribution:
1. **Illusion of predictability in thresholded KPIs** — binarizing a random
   walk at a policy threshold manufactures high AUC from label autocorrelation,
   not skill. Generalizes beyond NHS (SLA/churn/readmission/credit breaches).
2. **Forecast Value Added (FVA) for classification** — extend the levels FVA
   tradition to binary/threshold outcomes with a proper persistence classifier.
3. **Transitions, not states** — the decision-relevant problem is which
   *currently-compliant* specialties are about to tip over (and which recover).
   Restrict eval to the transition subset; easy AUC collapses.

## FIRST ACTIONS (in order)
1. Confirm branch is `research`; confirm `data/` resolves (junction/copy).
2. **Read** `notebooks/05_breach_model.ipynb` and `src/hcip/modeling.py` —
   find exactly how the breach target and its baseline were defined. This is
   the crux; do not assert the hypothesis before reading it.
3. Run the **pivotal diagnostic** (make-or-break, decides if there's a paper):
   - add a **binary-persistence baseline** (next-month breach = this-month
     breach), score the classifier against it on the same time-split;
   - re-evaluate on the **transition subset only** (cases that change state);
   - report AUC **and** a decision-relevant metric (precision@k of flagged
     at-risk specialties, or a net-benefit / decision curve).
   Outcome A: AUC collapses toward persistence + poor transition-AUC → that gap
   IS the paper (framing 1). Outcome B: model genuinely beats binary-
   persistence on transitions → positive paper (framing 3). Either is good.

## What would lift it from "applied note" to "research"
A small **analytical result**: under a random-walk-with-drift level process and
a fixed threshold, derive the AUC achievable by the *persistence* classifier as
a function of the gap-to-threshold distribution and noise — show high AUC is
expected with ZERO genuine skill. With the theory, one dataset suffices; without
it, need 2–3 other thresholded-KPI datasets to survive "single dataset" reviews.

## Honest risks
Negative-result / evaluation-critique papers are harder to place (the analytical
lemma de-risks this). Adjacent work exists (AUC base-rate critiques, "predicting
the present", persistence baselines in clinical prediction) — **do a real
literature check before committing**.

## Venue targets
International Journal of Forecasting (best cultural fit — FVA/honest-baseline
tradition), ML4H, a NeurIPS/ICML time-series workshop, KDD applied track.

## Workflow / git
This clone's `origin` → the dashboard repo. Fixes that belong only here: just
commit on `research`, they stay put (divergence is free). To share a fix to the
dashboard, push + cherry-pick onto its `main`. To pull dashboard progress in:
`git fetch origin && git merge origin/main`. Confirm branch before every edit.
