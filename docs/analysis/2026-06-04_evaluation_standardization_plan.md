# Thesis evaluation-standardization plan (2026-06-04)

Goal: turn the ~30-table, ~8-surface sprawl into a coherent **initial → intervention → study → impact**
story for every production component, while **keeping all evaluation surfaces** (no data dropped from
main text). Decisions locked with the user:

| Decision | Choice |
|---|---|
| Narrative | **Master scorecard + per-component 4-beat drill-downs** |
| Surfaces | **Keep all; unify columns/format only** (don't demote any to appendix) |
| Versions in main tables | **3 max, story-picked** (not latest-3) |
| Metric cells | **P/R/F1 triple + fire-rate**, consistent everywhere |
| Table consolidation | **Keep per-component tables, standardized layout** (not surface×component matrices) |
| Scorecard anchor | **Per-stage best-fit surface** (each row labeled with its surface) |

The sprawl is cut by *organization*, not by cutting surfaces: (a) one standardized table per production
component instead of many scattered ones, (b) ≤3 rows each, (c) a single master scorecard for the overview,
(d) full traces/per-split/per-clip tables → appendix.

---

## 1. The canonical surface → metric vocabulary (define ONCE, in Methodology)

Add a "metric" column to the existing `tab:datasets` so the rule is fixed and never re-explained:

| Surface | Role | Headline metric(s) |
|---|---|---|
| in-distribution test (`rgb_dataset` / `ir_dset`) | does it still work where trained | P/R/F1 (IoU@0.5) |
| Svanström (paired) | **discriminating** (small drones + labelled confusers) | drone P/R/F1 (IoP@0.5) + per-class fire-rate |
| Anti-UAV RGBT | saturated sanity floor | P/R/F1 (IoU@0.5) |
| SelCom CCTV | deployment-partner | P/R/F1 (IoP@0.5) |
| Roboflow OOD (9) | OOD generalisation | P/R/F1 (drone) / fire-rate (confuser) |
| YouTube real-video (19) | operational | segment drone-F1 + confuser FPR |
| OOD confuser-zoo | false-alarm robustness | fire-rate |
| CBAM (IR held-out) | novel thermal confuser | F1 + FP count |

**Metric standard:** drone surfaces → P/R/F1; confuser-only surfaces → fire-rate (frame) or FPPI (video);
3 decimal places; mAP only in appendix. This vocabulary is stated once and every table obeys it.

---

## 2. Standard per-component table template (the unifying layout)

Every component's study table is ONE table, identical shape, two row-blocks split by `\midrule`:

```
                | <version-1> (initial) | <version-2> (study) | <version-3> (production)
 DRONE SURFACES |     P / R / F1         |    P / R / F1        |    P / R / F1
   in-dist test |   ...                  |   ...               |   ...
   Svanström    |   ...                  |   ...               |   ...
   Anti-UAV     |   ...                  |   ...               |   ...
   SelCom       |   ...                  |   ...               |   ...
   real-video   |   ...                  |   ...               |   ...
 ----------------------------------------------------------------------------
 CONFUSER SURF. |   fire / FPPI          |   fire / FPPI       |   fire / FPPI
   confuser-zoo |   ...                  |   ...               |   ...
   real-video   |   ...                  |   ...               |   ...
   CBAM (IR)    |   ...                  |   ...               |   ...
```

Columns = the ≤3 story-picked versions; rows = all surfaces (dashes where a metric is N/A). The shipped
version is bolded. Same column order, same surface order, in every component table → fully skimmable.

---

## 3. Master "Production Pipeline Scorecard" (Results headline, per-stage best-fit anchor)

One table near the top of Experimental Results — the whole arc at a glance:

| Stage (production) | What it fixes | Anchor surface | Before | After | Δ |
|---|---|---|---|---|---|
| RGB detector `ft4` | recall/clutter on CCTV-scale drones | SelCom | F1 ... | F1 ... | ... |
| IR detector `v3b` | thermal drone recall, confuser mining | ir_dset / Svanström-IR | F1 ... | F1 ... | ... |
| Trust classifier `robust6` | OOD false-alerts, feature leakage | OOD-confuser | fire ... | fire ... | −30% |
| RGB verifier `mlp_v5` | residual confuser hallucination | Svanström | F1 0.768→0.869; fire ... | ... | ... |
| IR verifier `mlp_v5_ir_aligned` | novel thermal confusers | CBAM (held-out) | F1 0.699→0.846; 48→15 FP | ... | +0.147 |

Each row labeled with its own anchor surface; full numbers live in that stage's standardized table.

---

## 4. Story-picked ≤3 versions per component (rest → appendix)

| Component | v1 = initial | v2 = the study/lesson | v3 = production | → appendix |
|---|---|---|---|---|
| RGB detector | baseline (`Yolo26n_trained`) high-recall/high-halluc | `retrained_v2` over-correction (recall collapse) | **`ft4`** | hardneg_v3more, selcom_ft2, selcom_960 |
| IR detector | V2 (broad-merge, weakest) | V5 (bulk-ingest **regression**, HITL lesson) | **v3b** | V3, V4, V6, Final |
| Trust classifier | sa32 (strongest hand-engineered) | fnfn (open-world extreme: safe-but-misses) | **robust6** | control40, lean13/17/19 |
| RGB verifier | bare `ft4` (no verifier) | patch v2 (predecessor) | **mlp_v5** | patch v1/v3/v4, fail-open gate |
| IR verifier | bare `v3b` ("ship none") | patch (useless on thermal) | **mlp_v5_ir_aligned** | dedicated grayscale-only model |

Each pick is chosen to carry the narrative beat, not recency. The IR V5 regression stays because it IS the
HITL methodological contribution.

---

## 5. The repeating 4-beat drill-down (per component section)

1. **Initial** — the starting model + its failure mode, one number on the discriminating surface.
2. **Intervention** — one sentence: what we changed and why.
3. **Study** — the standardized ≤3-version table (§2) + 2–3 sentences reading it.
4. **Impact** — production pick + headline before→after Δ (ties back to the scorecard row).

---

## 6. Current → new table mapping (what collapses)

| New artifact | Absorbs / replaces |
|---|---|
| Master scorecard (new) | — |
| RGB-detector standard table | `tab:rgb_comparison`, `tab:selcom`, `tab:ood_rgb_drone`, RGB rows of `tab:realvideo_master` |
| IR-detector standard table | `tab:ir_evolution` (→3 rows), Svanström/Anti-UAV-IR cells, `tab:ood_ir` |
| Trust-classifier standard table | `tab:classifiers`, `tab:robust6_pipeline`, `tab:cascade_classifier_drone/fpr` |
| RGB-verifier standard table | `tab:distill_verifier`; `tab:patch_sweep`/`tab:patch_audit`/`tab:failopen` → drill-down/appendix |
| IR-verifier standard table | `tab:ir_aligned`, `tab:ir_aligned_gray` |
| Cascade-suppression table | `tab:cum_confuser`, `tab:cumulative_svanstrom`, `tab:cascade_segment`, `tab:cascade_percategory` |
| → Appendix | full IR-version trace, Roboflow per-split, Anti-UAV full, per-clip video, dataset-composition tables |

Net: ~30 main-text tables → ~7 standardized tables + master scorecard; everything else to appendix
(surfaces all still present, just reorganized).

---

## 7. Execution phases (each ends with build + hygiene + audit)

- **Phase 0** — add surface→metric column to `tab:datasets`; lock the metric vocabulary in `sec:metrics`.
- **Phase 1** — build the master Production Pipeline Scorecard in `sec:cumulative` (regenerate from `evals.csv`, never hand-typed).
- **Phase 2** — build the 5 standardized per-component tables (`thesis_tools.py table` where possible); bold the shipped version.
- **Phase 3** — rewrite each component section to the 4-beat structure pointing at its table.
- **Phase 4** — move displaced rows/tables to appendix; re-point every `\ref`; delete only true duplicates.
- **Phase 5** — build (expect compile-clean), `hygiene` (0 undefined / 0 rule-#2), `audit` (every cell traces to `evals.csv`), diff for Overleaf port.

Integrity: tables (re)generated from `knowledge/evals.csv` so they can't drift; every result-number sentence keeps its `% [source:]`; no number invented (parked IR-Svanström/Anti-UAV-IR cells stay flagged).

---

## Open questions for you (next round)
1. **Cascade story**: the 4 cascade tables (cumulative/segment/per-category) — fold into the master scorecard + ONE "cascade on real video" table, or keep cascade as its own 4-beat "component"?
2. **Scorecard "before"**: define "before" as the *bare detector* (no downstream) for all stages, or as the *previous stage's output*? (cumulative vs per-stage marginal).
3. **Appendix depth**: full per-clip YouTube table (19 clips) — keep in appendix, or drop to a summary + pointer to the repo?

## Delivered
- This plan: `docs/analysis/2026-06-04_evaluation_standardization_plan.md`
- (No thesis edits yet — planning only; execution gated on the 3 open questions above.)
