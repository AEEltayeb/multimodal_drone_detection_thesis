# Thesis Evidence + Reorganization Plan

**Date:** 2026-06-01 · **Status: PLAN ONLY — nothing edited, no jobs started.**

**Decisions locked (this session):**
1. **Reorg = Full IMRAD 5-chapter** (conform to the `tesi_master` skeleton).
2. **Run all four data jobs** (P0 clean-paired eval, P0 dual-verifier, P1 IR-generalization + mlp_v5 cascade, P2 latency + baseline-640).
3. **Curated versions** — production pick per slot + the 1–2 justifying comparators + the one instructive failure; everything else → footnote/appendix.

This plan answers the core question — *how do we PROVE each model earns its keep?* — then specifies (A) the data to acquire, (B) the reorganization, (C) version curation.

---

## 1. Doctrine: how a component "earns its keep"

Every production component must be justified by an **ablation** that satisfies four constraints:

1. **With vs without** — the system is measurably better *with* the component than without it (or with the comparator it replaces).
2. **Right surface** — measured on data whose modality matches the component's job (the hygiene protocol below).
3. **Right unit** — per-detection (a detector/verifier's intrinsic quality), frame-level alert (a classifier/cascade *decision*), or segment (real-video deployment). The unit must match the claim.
4. **Net of cost** — the benefit survives the latency/complexity it adds.

A component that helps standalone but not **in composition** has not earned its keep. The verifier in particular must show it helps *after* the classifier has already fired (clf vs clf+filter), not just versus the bare detector.

### 1.1 Evaluation-hygiene protocol (becomes a Methodology subsection)

| Component type | Valid evaluation data | Rationale |
|---|---|---|
| **Dual-modality** (trust classifier, dual-verifier fusion) | **Paired RGB+IR only**: Svanström (discriminating) + Anti-UAV (saturated floor) | A modality decision is only testable when *both* real modalities are present for the *same* frame. |
| **Single-model / single-path** (one detector; one verifier; one branch) | That modality's data | RGB: Roboflow OOD, confuser-zoo, rgb_dataset. IR: thermal surfaces. No cross-modal contamination because only one path is exercised. |
| **Grayscale-IR** | (a) single-IR-path **fallback**, evaluated single-path *where it helps* (F1/FPPI); (b) **confuser-harvest source** for alignment | Grayscale is never substituted for thermal inside a paired eval. It is a fallback mode and a data source, not a thermal stand-in. |

**Units:** per-detection · frame-level alert · segment — matched to the claim.
**Scoring:** Svanström IoP@1280 · Anti-UAV IoU · confuser-only surfaces = fire/hallucination rate (no GT).

### 1.2 The one contamination this fixes
The headline dual-modality classifier ablation (`robust6-production-viable`, the 5000-frame run) used **true-thermal IR for drones but grayscale-fallback IR for confusers** — a modality split across positives/negatives, violating the paired-only rule. **P0-a fixes it** by re-running on paired Svanström (drones + filename-mined confusers, real thermal on both sides).

---

## 2. Target structure — Full IMRAD (7 → 5 chapters)

| IMRAD chapter | Built from current | Key moves |
|---|---|---|
| **1. Introduction** | Ch1 (as-is) | Keep Background, Problem, RQs, Contributions, Ethics, Outline. Refresh Outline to 5-ch. |
| **2. Background & Related Work** | Ch2 | As-is + prior-work comparison. |
| **3. Approach: System & Methodology** | Ch3 (Methodology) **+** Ch4 (Architecture) **+** Ch5 (HITL) | The "what we built / how we measure" chapter. Component **designs** + methodological **instruments** (datasets, IoP, scoring audit, MRI, reproducibility, HITL). Inline ablation results move to Ch4, leaving a **headline number + forward-ref** in each component section. |
| **4. Empirical Evaluation** | Ch6 (Experimental Results), restructured **RQ-first** | 4.1 Study Design (RQs + **hygiene protocol** + surfaces + **master ablation table**); 4.2 Results **by RQ**; 4.3 Discussion; 4.4 Threats to Validity. |
| **5. Conclusion** | Ch7 | RQ answers, production stack, future work. |
| Appendices | Datasets-in-Detail, Glossary **+ new "Version History & Full Ablation Inventory"** | Curated-out trials land here. |

**Ablation-placement principle (the crux of the IMRAD move):** Ch3 states each component's *design + one headline justification number* and forward-refs; Ch4 holds the *systematic ablation evidence organized by RQ*. Judgment calls kept in Ch3: the HITL IR-evolution trace (it is a process contribution) and the MRI feature-space figures (they are method). Their *metric tables* are cross-referenced from Ch4.

**Results-by-RQ map (Ch4.2):**
- **RQ1** (suppress confuser fire without retraining) ← cumulative confuser suppression; 3-stance detector vs cascade; **filter-helps-classifier** (clf vs clf+filter); mlp_v5 vs patch. *Uses P0-a, P0-b.*
- **RQ2** (in-distribution cost by surface + classifier) ← classifier ablation sa32/robust6/fnfn/control40 on **paired Svanström** (P0-a); per-frame-misleading vs segment-recovers; temporal as load-bearing.
- **RQ3** (scoring comparability) ← the 28-pp IoP/dual scoring-rule audit.
- **RQ4** (thermal→grayscale transfer) ← grayscale fallback (single-path); cross-modal alignment; IR aligned verifier + **dual-verifier fusion** (P0-b, P1-a).
- **Methodological thread** (HITL) ← IR V2→Final/v3b trace incl. the V5 regression.

---

## 3. Master "Component Ablation Summary" table (Ch4.1 spine)

One row per production component → the ablation that proves it → valid surface → unit → ledger id → status. This single table *is* the "earns its keep" proof, and every row traces to a ledger finding.

| # | Component (production) | Earns-keep ablation | Valid surface | Unit | Ledger id | Status |
|---|---|---|---|---|---|---|
| 1 | RGB `ft4` (vs baseline / retrained_v2) | confuser-fire ↓ without recall collapse | Svan IoP@1280 + confuser splits (RGB) | per-det | `retrainedv2-recall-collapse`, `ft4-backbone-freeze`, `antiuav-saturated` | ✅ have |
| 2 | `selcom_ft2_1280` + imgsz policy | CCTV recall recovery via resolution | selcom held-out, Svan | per-det | `selcom-imgsz-win`, `selcom960-cross-surface-winner`, `selcom-ood-confuser-damage` | ✅ (baseline-640 control = P2) |
| 3 | IR `v3b` | best IR via HITL; vs RGB on paired | **paired** Svan + Anti-UAV | per-det | `ir-version-progression` | ✅ |
| 4 | Grayscale-IR fallback | usable *where it helps*; −12.4pp drone F1 vs lowest FPPI | real-video **single IR path** | F1/FPPI seg | `ir-grayscale-fallback`, `grayscale-drone-recall` | ✅ conditional |
| 5 | Trust classifier sa32 / robust6 / fnfn | clf on-vs-off routes modality; robust6 leakage-robust | **paired Svan + Anti-UAV** | frame-alert + seg | `robust6-production-viable`, `trust-classifier-conditional`, `three-classifier-realvideo`, `control40-deprecated` | ⚠️ **contaminated → P0-a** |
| 6 | RGB verifier `mlp_v5` | clf vs **clf+filter**; beats patch | confuser-rich RGB + Svan | frame-alert + per-det | `v5-beats-patch`, `mlp-beats-patch-both-modalities`, `v5-rgbds-ceiling`, `verifier-recall-precision-decision` | ✅ have |
| 7 | IR verifier `mlp_v5_ir_aligned` | confuser-FP ↓ recall-safe (grayscale-harvest+align) | **held-out thermal** confuser | per-det/FP | `ir-grayscale-harvest-solves-thermal-verifier`, `gray-thermal-alignable`, `ir-mlp-aligned-warranted` | ⚠️ 1 set only → **P1-a** |
| 8 | Dual-verifier fusion (trust-first) | two verifiers vs single; IR-always-on recall-neutral | **paired** | frame-alert | `dual-verifier-fusion-rule` | ❌ **gap → P0-b** |
| 9 | Alert-gate / per-frame placement | per-frame right; alert-gating mlp_v5 = −4pp Svan F1 | Svan + real-video seg | segment | `v5-ship-per-frame`, `cascade-bird-vs-airplane-asymmetry` | ✅ |
| 10 | Temporal smoother | load-bearing recovery 0.586→0.826 | real-video **segment** | segment | `cascade-segment-recovers`, `cascade-perframe-misleading` | ✅ |
| 11 | Statistical feature selection (method) | leakage stat: 6 feats beat 32 | paired + OOD drone video | frame-alert | `fusion-feature-leakage`, `ft4-lean-trust-classifier`, `robust6-grayscale-ir-features-dead` | ✅ (re-confirm via P0-a) |
| 12 | Model MRI (instrument) | feature-space separability grounds verifier+classifier | feature-space | LDA/ANOVA/AUROC | `v5-lda-separability`, `ir-features-separable`, `mlp-v5-recall-drop-is-ood-coverage` | ✅ method |
| 13 | HITL (process) | co-evolution incl. V5 regression failure | held-out IR test split | P/R/F1 trace | `ir-version-progression` | ✅ method |
| — | End-to-end latency (cost side) | full-pipeline ms/frame on edge | deployment hardware | latency_ms | `latency-edge-unmeasured` | ⚠️ verifier-stage only → **P2-a** |

---

## 4. Data-acquisition plan

All four approved. **Extend existing scripts — do not write near-duplicates** (CLAUDE.md rule 1). Each job lists: goal · method/extends · output rows · claim it backs · compute. User runs the long jobs.

### P0-a — Clean paired Svanström classifier + verifier ablation  *(fixes a core claim)*
- **Goal:** re-prove the trust classifier (sa32 / robust6 / fnfn) and the **filter-helps-classifier** composition on **paired RGB + real-thermal**, using Svanström drones + filename-mined confusers (`IR_BIRD_/IR_AIRPLANE_/IR_HELICOPTER_`, no IoU filter for confusers; drones IoP@1280).
- **Extends:** `eval/pipeline_cache.py` (build a paired-Svan-confuser feature cache: RGB + **real thermal** per confuser frame) → `eval/pipeline_eval_offline.py` / `eval/overnight_ablation_full.py` (replay the cell matrix: bare / clf[·] / filter[mlp|patch] / clf→filter / filter→clf).
- **Output:** clean `evals` rows on a new `eval_config` (e.g. `svan_paired_confuser_1280`); **revise** `robust6-production-viable` to drop the modality-split caveat (or refute if the clean numbers differ).
- **Backs:** RQ1 (filter-helps-classifier), RQ2 (classifier cost on paired surface), master rows 5/6/11.
- **Assumption to verify first:** Svanström confuser scenes carry **both** RGB and IR streams. If IR-only, fall back to Anti-UAV paired + the paired-confuser subset that does have both, and state the reduced n.
- **Compute:** offline replay if paired-thermal-confuser feats already cached; else one GPU caching pass.

### P0-b — Dual-verifier fusion eval
- **Goal:** prove RGB `mlp_v5` + IR `mlp_v5_ir_aligned`, **trust-first** order, earns its keep vs single-verifier; confirm the **always-on IR verifier stays recall-neutral** under *both* sa32 and robust6.
- **Extends:** `ir_gui/pyside_engine.py::_mlp_trust_first` (trust-first already implemented) + a thin offline driver over `pipeline_eval_offline.py` for the `dual_verifier_pipeline` config; implement the router soft-weight extraction for the `trust_both` conflict.
- **Output:** `evals` rows per (router × verifier-config); a `ledger` row confirming/refuting `dual-verifier-fusion-rule` with **paired** evidence.
- **Backs:** RQ1, RQ4, master row 8. **Paired surfaces only.**

### P1-a — IR verifier generalization (≥1 more held-out thermal confuser set)
- **Goal:** harden "generalizes" beyond the single CBAM held-out set.
- **Extends:** `mri/train_aligned.py` + `eval/run_aligned.py` / `eval/ir_verifier_eval.py` (held-out gate) — add a second held-out thermal aerial-confuser dataset; keep it out of training.
- **Output:** `evals` row; strengthens `ir-grayscale-harvest-solves-thermal-verifier` (currently CBAM-only). **Backs** master row 7, RQ4.

### P1-b — mlp_v5 full-cascade real-video re-run
- **Goal:** the real-video cascade numbers (`pipe_video_*`) were measured with the **patch predecessor**; re-run with `mlp_v5` (RGB) + `aligned` (IR).
- **Extends:** the real-video pipeline harness (`eval/eval_video_temporal.py` / `pipe_video_*` driver), swapping patch→mlp_v5.
- **Output:** updated `evals` rows for `pipe_video_drone_iop` / `pipe_video_confuser`; resolves the open conclusion item. **Backs** RQ2, master rows 6/10.

### P2-a — End-to-end edge latency
- **Goal:** full-pipeline ms/frame (detector+classifier+verifier+temporal) on deployment hardware; only verifier-stage measured today.
- **Extends:** `eval/eval_pipeline_v5_quick.py` (already has verifier-stage + per-frame overhead) → full e2e on target hardware; record one `evals` row per pipeline/arch with `latency_ms`.
- **Output:** resolves `latency-edge-unmeasured`. **Backs** the cost side of every "earns its keep" claim.

### P2-b — Baseline RGB @ imgsz=640 on Svanström
- **Goal:** the pending resolution-floor **control** (retrained_v2 collapses to R=0.072 @640; baseline's own 640 number is missing).
- **Extends:** the existing Svanström RGB eval at `imgsz=640`, `baseline` weights.
- **Output:** completes the `svan_iop_640` baseline cell; resolves the §3.1 pending item. **Backs** master row 2 (resolution dependency).

**Gating:** P0-a and P0-b **must land before** Ch4's dual-modality results are rewritten (no writing the clean-paired claim before the clean number exists — run-then-write).

---

## 5. Version & trial curation map (Curated)

| Slot | In body | Footnote | Appendix | Cut |
|---|---|---|---|---|
| RGB detector | baseline, retrained_v2 (instructive collapse), ft4 (production), selcom_ft2_1280 (CCTV) | selcom_960 (cross-surface optimal, 1 mention), selcom_640 (operating point) | hardneg_v3more (intermediate) | lean detector one-offs |
| IR detector | **6-version HITL trace** V2→V4→V5(regression)→V6→Final/v3b | — | per-revision dataset-state detail | — |
| Patch verifier | v2 (shipped) + **one** version-ranking table (v1–v4) | mlp_v5 distill lineage (v2/v3/v4) = 1 sentence | full distill changelog | — |
| Trust classifier | sa32, robust6, fnfn (open-world), control40 (instructive reject) | lean13/17/10/19 as "the hand-tuning that motivated statistical selection" (1 para) | dual-clf-v3 | — |
| Verifiers | RGB mlp_v5, IR aligned (thermal+gray dual-scaler) | — | — | — |
| **Evals/trials** | the **curated ablation set = the master-table rows**, RQ-organized | — | **full 116-eval inventory** | — |

**Answer to "how many versions/trials to mention":** the body names ~4 RGB variants, the 6-point IR trace, 2 patch + the mlp lineage as one line, 4 classifiers, 2 verifiers — and reports the **master-table ablation set** (≈12–15 rows), *not* the full 116 evals. The Version-History appendix carries the rest, so nothing is hidden but the argument stays legible.

---

## 6. Methodology coverage (MRI + HITL)

- **Model MRI** → Ch3 as the feature-space **instrument** (`python -m mri`): hooks P3/P5, ROI-pools the 517-D embedding, emits LDA/ANOVA/leakage/activation evidence. Frame once as method; its *outputs* (separability, recall-drop diagnosis, leakage map) are cited from Ch4. It also doubles as an **audit instrument** (regenerating V5 evidence on the shipped corpus corrected two figures — `mri-v5-report-regen`).
- **HITL** → Ch3 as the dataset-curation **process** contribution; the IR V2→Final/v3b trace (incl. the **V5 regression** = bulk-ingestion-bypassing-review failure) is the case study. This is the orthogonal 5th methodological thread already declared in `sec:rqs`.

---

## 7. Execution sequence (via the thesis skill, gated)

1. **(done)** This plan — approved.
2. **Run P0 jobs** (user) → record `evals` + revise `robust6-production-viable` ledger. *Gate for step 5's dual-modality rewrite.*
3. `structure` → perform the 7→5 IMRAD consolidation as **pure moves** on `thesis_working.tex` (record `coherence` rows; keep the diff reviewable by not rewriting prose in this step).
4. `table` / `plot` → generate the **master Component-Ablation-Summary table** + figures from the P0/P1 evals (deterministic, from `evals.csv`).
5. `draft` → add the **Evaluation-Protocol** subsection; rewrite the dual-modality results with the **clean paired** numbers; apply the curation map; write Ch4 Data-Analysis + Discussion.
6. `audit` + `examiner` → verify every claim traces to a ledger/eval row; record `claims`/`review`.
7. `readability` + `humanify` → polish, certainty-calibrated.
8. `hygiene` + **compile** (`docs/build_thesis.ps1` — MiKTeX, pdflatex×3 + bibtex; **not** latexmk/biber).

Run P1/P2 jobs in parallel with steps 3–4; fold their numbers in at step 5 if ready, else leave them as documented future work in Threats/Future-Work (honest, not fabricated).

---

## 8. Risks / what NOT to do

- **Do not overwrite `thesis_working.tex`** — it holds un-ported edits (diff: ~1016 ins/453 del vs `thesis_chapters.tex`). The skill syncs only with explicit confirmation.
- **Paired-only for dual-modality.** Never let the RGB-only confuser-zoo or grayscale-IR drive a trust-classifier / fusion claim.
- **Run-then-write.** No clean-paired claim before P0-a produces the number. If a job is not run, it is future work, not a written result.
- **Keep the honest carve-outs**: the `rgb_dataset` recall ceiling (route to patch), selcom doubly-OOD grain of salt, Anti-UAV saturation, CBAM-in-training optimism (held-out gate is the real test).
- **IMRAD move = moves first, prose later.** Do the chapter consolidation as relocations (step 3), *then* rewrite (step 5), so the reorg diff is reviewable and reversible.
- **Provenance gaps stay flagged** (patch-v3 path, fnfn training, may10 imgsz, ir-gray-confuser-src) — cite only what's recoverable.

---

## 9. Open decisions deferred to you
- **Compute order** of P0/P1/P2 (which to launch first; P0-a is the load-bearing fix).
- Whether the **Version-History appendix** should include metric tables or just a model/version index.
- Whether to keep the 9-section component chapter (current Ch4) **whole** inside IMRAD-Ch3, or split the verifier sub-story across Ch3 (design) and Ch4 (results) more aggressively.

## Delivered
- `docs/analysis/2026-06-01_thesis_evidence_reorg_plan.md` (this plan)

_No thesis files edited; no data jobs started; no skill run. Awaiting your go on §7 / §9._
