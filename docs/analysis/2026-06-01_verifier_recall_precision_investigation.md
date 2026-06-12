# V5 Verifier Recall/Precision — Full Investigation & Decision

**Date:** 2026-06-01 · **Decision: ship baseline `mlp_v5` (full-veto). The recall drop is a still-image carve-out, deployment-benign; no runtime fix is warranted; the only real lever is diverse small-drone training data (future work).**

This consolidates the whole investigation: diagnosis → three negative results → a partial calibration fix → the ship decision. All numbers from recorded runs (scripts cited).

## 1. The concern
The per-detection MLP verifier (`mlp_v5`) suppresses false positives strongly but costs recall:
- **rgb_dataset_test** (general still-image benchmark): per-box R 0.888→0.694; frame-level R 0.934→0.840.
- **svanström** (per-box, offline): R 0.914→0.829, but precision 0.46→0.88.

## 2. Reframing — the drop is still-image-specific, not deployment-critical
On the **video deployment surface (svanström, 5000-frame ablation), `mlp_v5` frame-level alert recall ≈ 1.0** (a drone seen across frames survives even if some boxes are vetoed), with precision 0.46→0.88. The recall drop is a **per-box / sparse-still-frame** phenomenon (rgb_dataset), not a video-deployment loss. `eval/temporal_ablation.py`, `eval/overnight_ablation_full.py`.

## 3. Diagnosis (MRI / statistics) — an OOD coverage gap
`eval/diagnose_mlp_recall_drop.py`, `eval/_veto_vs_confuser.py`. The falsely-vetoed real drones are:
- **NOT low-confidence** (Δconf = 0.000) — the obvious hypothesis was wrong;
- **smaller** (log_area kept −0.075 vs vetoed −0.180) with a **distinct deep-feature signature** (per-neuron AUROC ≈ 0.89);
- **far from confusers** (centroid 16.48 vs kept-drones 11.05; vetoed-vs-confuser AUROC 0.876 ≥ kept 0.862) → an **OOD coverage gap, not drone↔confuser overlap**;
- **systematically** vetoed (consistent across frames), not flicker.
- Only **43 of 19,334** training drones are under-scored by the current MLP → the OOD drones are **absent from training** (the gap is *absence*, not weighting).

## 4. Fix attempts — three negative results
| attempt | script | rgb_dataset recall | side effect | verdict |
|---|---|---|---|---|
| **Fail-open** (OOD-abstain) | `eval/test_failopen_verifier.py`, `eval/eval_failopen_prepost.py` | 0.694→**0.870** ✅ | **svanström precision 0.887→0.631** 💥 | ❌ backfires on cluttered video |
| **Targeted re-weight** (×4 under-scored drones) | `eval/retrain_v5_targeted.py` | 0.694→0.696 | — | ❌ no-op (coverage gap = absence) |
| **2-of-3 temporal voting** | `eval/temporal_ablation.py` | −0.008 | (small precision win on cluttered video) | ❌ veto is systematic, not flicker |

## 5. Partial fix — expanded confuser reference
`eval/failopen_expanded_ref.py`. Building the OOD reference *with* svanström clutter (held-out) vs rgb_confusers-only, at matched recall 0.90:

| variant | R | P |
|---|---|---|
| bare | 0.915 | 0.460 |
| full-veto (mlp_v5) | 0.829 | **0.884** |
| fail-open, original ref | 0.900 | 0.486 |
| fail-open, **expanded ref** | 0.900 | **0.611** |

Expanding the reference recovers **~half** the lost precision (0.486→0.611, +12.5pp) → the backfire is **partly calibration**. But it does **not** reach full-veto (0.884) → the recall/precision trade is **partly fundamental** (genuine drone↔clutter overlap). (Index-parity split → 0.611 is an optimistic ceiling.)

## 6. Size-aware threshold (mild safe interim)
`eval/test_size_aware_threshold.py`. Lenient veto for small boxes: rgb_dataset R 0.694→0.734, svanström R 0.843→0.891, at a modest precision/FP cost (no catastrophe). A tunable knob if still-image recall ever matters.

## 7. Decision & rationale
**Ship `mlp_v5` full-veto.** On video (deployment) it is recall-safe (frame R ≈1.0) with a large precision gain; the recall drop is confined to a general still-image benchmark (frame R 0.84) and is the documented carve-out. Every runtime fix either backfires on the deployment surface (fail-open), is a no-op (re-weight), can't help (temporal), or is marginal (size-aware). The **only frontier-moving lever is diverse small/OOD-drone training data** (future work; validated on the IR side, `ir-recall-fixed-by-drone-diversity`). Expanded-reference is a free precision win on file if fail-open is ever desired.

## 8. The arc, for the thesis
*"We found the verifier's recall drop, proved via MRI it is an OOD small-drone coverage gap (separable from confusers, systematic), showed it is deployment-benign (video frame-recall ≈1.0), and demonstrated that runtime fixes do not solve it — fail-open backfires on cluttered surfaces (a one-class OOD score conflates OOD-drone with OOD-clutter; expanding the reference recovers only ~half), re-weighting is a no-op (the gap is data absence), and temporal voting cannot recover a systematic veto. The single real lever is closing the coverage gap with diverse training data. We therefore ship the verifier as-is and document the still-image carve-out."*

## Delivered
- `docs/analysis/2026-06-01_verifier_recall_precision_investigation.md` (this)
- Scripts: `eval/diagnose_mlp_recall_drop.py`, `eval/_veto_vs_confuser.py`, `eval/test_failopen_verifier.py`, `eval/eval_failopen_prepost.py`, `eval/retrain_v5_targeted.py`, `eval/temporal_ablation.py`, `eval/failopen_expanded_ref.py`, `eval/test_size_aware_threshold.py`
- Figures: `docs/analysis/images/failopen_{ood_hist,tradeoff,pca}.png`, `failopen_expanded_ref_svan.png`, `fusion_{lda_hist,pca_2d,feature_auroc,leakage_map}.png`, `v5_ir_activation_{drone,confuser}_example.png`
- Ledger: `mlp-v5-recall-drop-is-ood-coverage`, `verifier-recall-precision-decision` (this doc)
