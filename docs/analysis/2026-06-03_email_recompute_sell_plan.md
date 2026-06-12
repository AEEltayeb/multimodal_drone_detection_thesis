# Email-recompute v2 — fix scoring, sweep, and "sell" the new classifier+filter (plan)

**Date:** 2026-06-03 · Follows `2026-06-02_email_pipeline_recompute_plan.md` (v1 run done).
**Goal:** Turn the v1 run into a *fair, favorable, honest* comparison that sells robust6 + V5 MLP
to Pietro — fixing the scoring bug, finding the best operating point, and adding filter-isolation +
speed + OOD tables.

---

## 1. Why v1 looked "worse" — three root causes, all fixable

| Cause | Effect in v1 | Fix |
|---|---|---|
| **Scoring = "dual" not "trust-aware"** (the email's rule) | classifier configs scored the *untrusted* modality's GT as phantom FN → classifier/filter→clf/clf→filter cratered (F1 0.94/0.63/0.64 vs email 0.99) | **DONE in code:** Phase B now calls `eval/metrics.py::score_trust_aware` — trusting one modality excludes the other's GT (reject penalizes both, trust_both sums). Codebase documents this as a **~28pp F1 swing** (`clm-scoring-28pp`). |
| **RGB MLP is OOD on Anti-UAV/Svan** | `mlp_v5` (distilled on *pure selcom CCTV*) outputs **P(drone)=0.0000 for every** Anti-UAV/Svan aerial-RGB det → `rgb_filter` = all-zeros | Carve-out (Q2). Trust-aware routing already sidesteps it: those surfaces route to IR, so the deployed pipeline is unaffected. |
| **Thresholds never swept** | IR=0.05 (recall-safe) under-suppresses OOD confusers; RGB=0.5 hardcoded | Sweep (free after batching). |

**The fix to "fix the .md generation" is primarily the scoring swap** (already applied) + the new tables below.

## 2. The domain-3 finding that reframes the whole sell (already computed, OLD vs NEW)

det% per category (lower=better for confusers; higher=better for drones):

| | ALL_CONF suppr | HELI suppr | AIRPLANE suppr | **DRONE_CLEAN kept** | **DRONE_LABELS kept** |
|---|---|---|---|---|---|
| OLD CNN patch | **80.7%** | **94.6%** | **78.3%** | 28.5% (loses 68%!) | 56.9% |
| NEW MLP @0.05 | 21.7% | 7.8% | 31.5% | **89.2%** (loses 0.4%) | **99.9%** |

**The patch suppresses confusers hard but butchers real-drone recall; the MLP@0.05 is recall-safe but
under-suppresses.** It's a **tunable tradeoff** → the sell is the MLP's suppression-vs-recall **curve vs the
patch's single fixed point**. If the curve passes above-left of the patch point, the MLP **Pareto-dominates**.
Plus the MLP is **37–72× faster** (1.3–2.1 ms/det vs 59–112; 1–4% vs 48–191% per-frame overhead — `v5-beats-patch`).

## 3. Work plan

**A. Batch Phase B (perf, prerequisite for sweep).** Restructure `pipeline_eval_paired.py` so MLP P(drone)
(threshold-independent) and robust6 are predicted in **one batched call each**, not per-frame. 5.3 h → ~3–5 min.
Re-run on the existing cache → corrected trust-aware OLD-vs-NEW tables for Anti-UAV + Svanström.

**B. Threshold sweep (free after A).** MLP P(drone) is cached, so sweeping is just re-thresholding:
- Domains 1–2: `ir_mlp_thr × rgb_mlp_thr` grid → trust-aware F1 per config; pick the operating point (Q1).
  (Confirms RGB is OOD-degenerate regardless of thr; tunes IR.)
- Domain-3: add a one-time per-det P(drone) **cache** for the 14 clips (GPU, ~20 min), then sweep IR thr
  offline → the suppression-vs-recall **Pareto curve vs the patch point**.

**C. Filter-isolation "sell" table (OLD CNN vs NEW MLP, same detectors).**
- IR: domain-3 (done) + a matched-operating-point row ("at the MLP thr matching the patch's drone recall,
  MLP suppresses X% vs patch Y%").
- RGB: on **in-domain** confuser surfaces where mlp_v5 is valid (selcom precision +9pp per email;
  `rgb_confusers_merged`) — NOT Anti-UAV/Svan (OOD). Needs patch P(confuser) on those (light GPU pass).
- **Speed row** from knowledge (37–72× faster). Honest caveat: edge end-to-end latency still unmeasured
  (`latency-edge-unmeasured`) — cite verifier-stage only.

**D. OOD comparison.** domain-3 (OOD IR) + Svanström confuser-FP breakdown (per AIRPLANE/BIRD/HELI) +
optionally RGB confuser videos / `rgb_confusers_merged` (Q3).

**E. Provenance + recording.** Record evals rows (each surface × stack × scoring) + a `ledger` finding
("robust6+V5-MLP vs old ensemble, trust-aware") + the sell claims, all via `kb.py`. New comparison `.md`s
under `eval/results/_email_recompute/` + a thesis-ready summary doc. Reply-to-Pietro draft last.

## 4. Honest framing for Pietro (so the sell survives scrutiny)
- NEW column = whole-stack (detector+classifier+filter all change) — stated, not hidden.
- robust6 trades ~1.2pp Anti-UAV recall for ~30% fewer OOD-confuser false alerts (its design point).
- RGB MLP is CCTV-specialized (OOD on aerial benchmarks) → carve-out, not a universal RGB filter.
- MLP's headline wins: **recall-safe confuser filtering + 37–72× speed**, tunable to match/beat the patch's
  suppression at far higher drone recall.

## 5. Open decisions → see the questions asked alongside this doc
Q1 operating-point/curve presentation; Q2 RGB-OOD carve-out handling; Q3 OOD/sell-surface scope.

## Delivered (this session)
- This plan; `pipeline_eval_paired.py` scoring fixed to trust-aware (re-run pending batching).
- Diagnosis verified: mlp_v5 P(drone)=0 on Anti-UAV RGB; domain-3 OLD-vs-NEW computed; trust-aware = published rule.
