# Thesis draft plan — missing contributions (report only, no tex edits)

**Date:** 2026-05-31 · **Mode:** `/thesis draft` → user chose *"just report, don't edit yet"* +
*V5 positioning = supersede patch as production*. This doc is the claim→evidence→target-section
plan for the next session to execute against `docs/thesis_working.tex`. Nothing in the tex was
changed.

Integrity rule reminder: every drafted sentence carries a `% [source: ...]` provenance comment.
Provenance strings below are pre-built from `knowledge/evals.csv` + `knowledge/ledger.csv`.

---

## ⚠️ Naming collision to resolve FIRST (blocks the V5 draft)

The thesis already uses the token **"V5"** in two unrelated places:

| "V5" in thesis today | What it actually is | Lines |
|---|---|---|
| IR detector revision **V5** | 5th HITL checkpoint of the IR YOLO; the regression case study | 153, 204, 685, 746, 757, 767, 772–775, 910, 1512 |
| "v5 patch verifier" (future work) | a *hypothetical retrained MobileNetV3* on airplane crops | 664, 1393, 1477, 1522 |

The contribution we want to add — `mlp_v5` — is **neither**: it is a *distilled feature-space MLP*
over FT4 YOLO ROI embeddings, a different architecture from the patch CNN. If we drop it in as
"V5" the reader cannot disambiguate three different things.

**Recommendation:** in thesis prose name it the **distilled ROI verifier (DRV)** (or
"feature-space verifier"), reserve the bare token "V5" for the IR revision, and **rewrite the four
"v5 patch verifier" future-work mentions** — that retrain-on-airplane-crops plan was *superseded*
by the distilled verifier we actually built. (Decision to confirm before drafting.)

---

## CARD 1 — Distilled ROI verifier (supersedes patch) ★ biggest gap, strongest evidence

**Ledger:** `v5-beats-patch` (supported), `v5-lda-separability` (supported),
`patch-v2-neutral-selcom` (supported), `v5-selcom-train-deploy-mismatch` (supported),
`embedding-distillation-cv` (partial, CV-only), `mri-v5-report-regen` (partial).

**Current thesis state:** patch verifier (§6, `\label{sec:patch_verifier_arch}`, lines 583–671)
is presented as *the* production alert gate; no successor exists. Contributions list (line 202)
names "a MobileNetV3 patch verifier" as the gate. Future work (1522) proposes a *patch* v5.

**What to draft (positioning = SUPERSEDE):**

1. **New subsection** after §6.4 Catch-Rate Audit (insert ~line 671, before the
   `\chapter{Dataset Curation}`): **"Distilled ROI Verifier (Production Successor)"**.
   - *Motivation* ties to the patch audit's own weakness: airplane catch 52%, median prob 0.540
     (Table tab:patch_audit) → the patch CNN is distribution-bound. Cite `patch-verifier-distribution-bound`.
   - *Architecture:* FT4 YOLO fused P3+P5 ROI features (517-D: 5 meta + p3@2×2 + p5@1×1) → MLP.
     Cite `v5-lda-separability`.
   - *Feature geometry justification:* the ROI features are **linearly separable** — LDA 0.949
     binary (drone vs confuser), 0.954 4-class by category. This grounds the architecture choice.
     `% [source: ledger=v5-lda-separability]`
   - *5-surface head-to-head vs patch v2* (the core table, numbers verified from evals.csv):

     | Surface | Config | Patch v2 F1 | DRV (mlp_v5) F1 | DRV halluc | Δ |
     |---|---|---|---|---|---|
     | Svanström paired (IoP@1280) | `svan_iop_1280_s9` | 0.768 (P0.686 R0.872) | **0.869** (P0.903 R0.838) | 0.037 | **+8.6pp** |
     | SelCom CCTV (IoP@1280) | `selcom_iop_1280` | 0.591 (=bare FT4) | **0.607** (P0.956 R0.444) | 0.019 | +1.5pp |
     | Confuser zoo (640) | `confuser_test_640` | — | FP=21, halluc **0.008** | — | ~97% cleaner |
     | Anti-UAV (IoU@640) | `antiuav_iou_640_s5` | — | 0.985 (P0.989 R0.981) | 0.010 | ties |

     `% [source: ledger=v5-beats-patch; eval=v5_svan_mlp,v5_svan_patch,v5_selcom_mlp,v5_confuser_mlp,v5_antiuav_mlp; run=eval/eval_v4_vs_patch.py; cache=eval/results/_v5_head_to_head_pure_1x8/comparison.md; config=svan_iop_1280_s9]`
   - *Latency:* 46–72× faster per detection than patch (claim text of `v5-beats-patch`); supports
     per-frame (not alert-gated) deployment. **Verify the exact latency table before writing** —
     evals.csv has no `latency_ms` populated for these rows; the 46–72× is from the ledger claim
     string + `project_v5_distillation_production` memory ("50× faster, 1–4% pipeline overhead").
   - *SelCom carve-out (honesty):* patch v2 on SelCom == bare FT4 (no CCTV exposure → votes
     "other"), so DRV must beat *bare*, and does (+1.5pp). `% [source: ledger=patch-v2-neutral-selcom; eval=v5_selcom_patch,v5_selcom_bare,v5_selcom_mlp]`
   - *Train-deploy lesson:* DRV's first SelCom run collapsed (F1 0.243, mixed 80/20 source);
     pure-CCTV source recovered to 0.607 (+37pp); the drone-weight bump is redundant once source
     matches. Frame as a methodological finding about verifier training data.
     `% [source: ledger=v5-selcom-train-deploy-mismatch; eval=v5_selcom_mixed_src,v5_selcom_pure3x5; config=selcom_iop_1280_persrc]`
   - *Precursor (optional, mark CV-only):* embedding distillation CV pilot — MLP 0.9955 / XGB
     0.9917 5-fold CV, meta-only already 0.939–0.966. **Phase-3 (held-out) pending** — do NOT
     present as a deployment number. `% [source: ledger=embedding-distillation-cv; eval=distill_cv_mlp,distill_cv_xgb; config=distill_cv_5fold]`
   - *Caveat to carry:* `mri-v5-report-regen` is **partial** — the original `mlp_v5_report.md` §2
     numbers came from a non-shipped corpus and the top-feature claim (confidence) was refuted
     (true top feature = P5 ch.154). Draft from the **regenerated** `mri/docs/mlp_v5_report_regen.md`,
     not the original report.

2. **Rewrite contributions list** (line 202): change "a MobileNetV3 patch verifier acts as a
   confuser-aware alert gate" → patch verifier is the *audited baseline gate*, **superseded** by
   the distilled ROI verifier as the production successor. Keep the surface-dependent-exchange
   framing intact.

3. **Rewrite the 4 "v5 patch" future-work mentions** (664, 1393, 1477, 1522): the retrain-patch-on-
   airplane-crops plan is *replaced* by the distilled verifier. The airplane gap is now addressed
   (or explicitly still open) by the DRV, not a hypothetical patch retrain. Re-check the confuser
   FP=21 result to state whether airplane is actually closed or still the weak link.

4. **Production stack update** (Conclusion §, line 1514 `Production Stack and Future Work` + Ch.1
   overview line 151 + abstract): the shipped verifier becomes the distilled ROI verifier, per
   frame. Cite `project_v5_distillation_production` decision.

**⚠️ Open item before this card is "production":** `v5_svan_mlp` extra field says *"CANDIDATE not
signed off; v5.2 train/eval in progress"*. Memory `project_v5_distillation_production` calls it
PRODUCTION CANDIDATE. Confirm with user whether to write it as *shipped* or *evaluated production
candidate* — this gates the strength of the abstract/contributions wording.

---

## CARD 2 — IR confuser verifier: SHIP NONE (symmetric negative result)

**Ledger:** `ir-verifier-conditional` (conditional). **Evals:** `ir_mlp_v5_cbam_thr05`,
`ir_mlp_v5_ir_dset_thr05`.

**Current thesis state:** absent. The IR detector section (§3, lines 495–510) and the
verifier-value discussion never state that *no* per-frame IR verifier ships.

**What to draft:** a short subsection (in §6 after the DRV card, or in Ch.5 Experiments near the
IR results 1143) — **"No Per-Frame IR Verifier: a Conditional-Value Result"**.
- IR detector v3b is already precise (raw F1 0.95–0.96) → no FP headroom for a verifier.
- On `ir_dset_final`: bare R0.971 F1 0.965; MLP@thr0.05 **lost 71 TP for 4 FP** (net-negative);
  patch also net-negative. `% [source: ledger=ir-verifier-conditional; eval=ir_mlp_v5_ir_dset_thr05; config=ir_final_640]`
- Helps **only** in confuser-saturated thermal scenes (CBAM set): bare P0.547 F1 0.699 48FP →
  MLP@0.05 13FP, recall unchanged, +0.187 F1. `% [source: ledger=ir-verifier-conditional; eval=ir_mlp_v5_cbam_thr05; config=cbam_ir_640]`
- **Thesis value:** this is the *symmetric counterpart* to Card 1 — a verifier pays off only when
  the detector hallucinates (RGB), and the precise IR detector proves the converse. Strengthens the
  "verifier value is conditional on detector hallucination" thesis (`trust-classifier-conditional`,
  `patch-verifier-distribution-bound`). Tool: `eval/ir_verifier_eval.py`.
- Keep `conditional`: do NOT write "an IR verifier is useless" — write the condition.

---

## CARD 3 — Scene-fingerprint overfitting in fusion classifiers

**Ledger:** `scene-fingerprint-overfit` (supported). **Evals:** none registered (claim is from a
feature-ablation sweep — *verify source before drafting; no evals.csv row backs the +18–26pp*).

**Current thesis state:** the Trust Classifier feature-set section (§3, `\label` near line 514,
"Feature Set" 514–530) describes the 32-feat set but does **not** discuss the leakage failure mode.

**What to draft:** a paragraph in the Feature Set subsection (~line 514–530) or the classifier
variants comparison (545–582) — **feature-leakage / sequence-split caveat**.
- Per-clip brightness scalars and `pos_x` act as **scene fingerprints** under a sequence-split:
  the classifier memorises which clip a detection came from instead of learning drone-vs-confuser.
- Dropping them (lean13 → lean10 feature set) recovers **+18–26pp** on held-out drone clips.
- Connects to `scene-aware-v3more` naming and to `three-classifier-realvideo` (why Svanström
  ordering predicts real-video). `% [source: ledger=scene-fingerprint-overfit]`
- **BLOCKER:** no `evals.csv` row carries the +18–26pp. Per integrity rule (1), either locate the
  ablation run and record an `evals` row first, or state the finding qualitatively without the
  pinned delta. Do not write "+18–26pp" until it traces to a row.

---

## Suggested execution order (next session)
1. Confirm naming decision (DRV vs reserve "V5") + production-vs-candidate wording for Card 1.
2. Record the missing `evals` rows for Card 3 (and DRV latency) so every number traces.
3. Draft Card 1 (new subsection + contributions/abstract/conclusion rewrites).
4. Draft Cards 2 & 3.
5. `git diff docs/thesis_working.tex` summary → user ports accepted edits to Overleaf.

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\docs\analysis\2026-05-31_thesis_v5_draft_plan.md` (this plan)
