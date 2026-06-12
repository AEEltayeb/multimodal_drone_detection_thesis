# Full-Pipeline Offline Eval — Verifier Matrix (Results)

**Date:** 2026-06-01 · Plan: [2026-05-31_full_pipeline_offline_eval_plan.md] · Scripts: `eval/pipeline_cache.py` (Phase A, GPU) + `eval/pipeline_eval_offline.py` (Phase B, **zero-GPU, re-runnable**). Raw: `eval/results/_offline_pipeline/offline_eval_results.{md,json}`.

Sampling: **1000 strided frames per surface** (stride = N/1000, not first-N). Offline: Phase A cached boxes + 517-D feats + patch score + GT once; Phase B replays all verifier variants with no GPU (re-runs in seconds).

## RGB verifier matrix (detector ft4) — bare vs patch_v2 vs mlp_v5
| surface | metric | bare | patch_v2@0.5 | **mlp_v5@0.25** |
|---|---|---|---|---|
| svanstrom (IoP) | F1 / halluc | 0.613 / 0.449 | 0.784 / 0.149 | **0.865 / 0.045** |
| rgb_confuser | FP (halluc) | 216 | 104 | **16** |
| selcom_val (IoP) | F1 (P) | 0.591 (0.86) | 0.591 (0.86) | **0.612 (0.95)** |
| antiuav_rgb | F1 | 0.987 | 0.987 | 0.987 (tie) |
| rgb_dataset_test | F1 (R) | 0.922 (0.888) | 0.894 (0.837) | 0.812 (0.694) ⚠ |

**Reproduces the V5 report** on a fresh strided sample: mlp_v5 wins svanström (+25pp over bare, +8pp over patch), cuts confuser FP 13.5× (216→16), lifts selcom precision 0.86→0.95 at no recall cost, ties on saturated antiuav. The **rgb_dataset_test recall carve-out persists** (R 0.89→0.69) — the known honest trade-off. ✅ mlp_v5 confirmed as RGB production verifier.

## IR verifier matrix (detector v3b) — bare vs ir_patch vs aligned_thermal
| surface | metric | bare | ir_patch@0.5 | **aligned_thr@0.05** |
|---|---|---|---|---|
| ir_dset_final | F1 (R) | 0.979 (0.969) | 0.957 (0.926) | **0.977 (0.965)** |
| ir_video | F1 | 0.975 | 0.973 | 0.975 (neutral) |
| antiuav_ir | F1 | 0.957 | 0.957 | 0.957 (tie) |
| **cbam (confuser-dense)** | F1 / halluc | 0.699 / 0.267 | 0.688 / 0.228 | **0.846 / 0.083** |

**Confirms the aligned verifier as IR production:** recall-safe on normal thermal (ir_dset dR −0.004, ir_video & antiuav neutral) and a **decisive win on confuser-dense CBAM: F1 0.699→0.846, FP 48→15 (−69%)** at recall 0.967→0.917. `ir_patch` is useless-to-harmful everywhere (loses TP, barely cuts FP) — matches the "patch weak on thermal" finding. Matches recorded `ir_aligned_cbam_heldout` (F1 0.846). ✅

## Grayscale-fallback matrix (v3b on RGB→gray) — bare vs patch vs aligned_gray
| surface | metric | bare | patch@0.5 | **aligned_gray@0.05** |
|---|---|---|---|---|
| gray_confuser | FP (halluc) | 143 | 119 | **20** (−86%) |
| gray_svan (IoP) | F1 / R / halluc | 0.548 / 0.548 / 0.189 | 0.591 / 0.548 / 0.128 | 0.273 / **0.164** / 0.017 ⚠ |

**New finding — the grayscale caveat is sharper than recorded.** `aligned_gray` crushes grayscale-confuser hallucination (143→20 FP, −86%) — great. **But on grayscale *drone* frames (gray_svan) it over-vetoes hard: recall collapses 0.548→0.164** to drive FP to ~0. The recorded gray-mode cost (dR −0.053 on a held-out mix) understates the hit on small-drone surveillance gray frames. **Implication:** the grayscale aligned verifier is a confuser-context tool, not safe per-frame on svan-like small-drone grayscale — gate it (alert-only / scene-routed) or raise its keep-threshold there. Worth a follow-up threshold sweep.

## Verdicts
1. **RGB:** ship mlp_v5 (confirmed, reproduced). ✅
2. **IR:** ship aligned_thermal — recall-safe + only thing that helps confuser-dense thermal. ✅
3. **Grayscale:** aligned_gray excellent for confuser rejection, **over-aggressive on small gray drones** → gate/tune, don't run per-frame unconditionally. ⚠ (new)
4. Harness is **offline & re-runnable** — Phase B replays from cache in seconds; thresholds/variants are free to sweep.

## Caveats
- antiuav/ir surfaces strided to 1000 (saturated → fine). svan IoP@1280, ir @640, conf 0.40 (IR) / 0.25 (RGB).
- Trust-classifier swap (sa32 vs robust6) NOT in this matrix — needs paired RGB+IR caching (next pass); covered separately by [2026-05-31_ft4_lean_trust_classifier.md].
- cbam n=180 (small, but held-out + the decisive confuser surface).

## Delivered
- `docs/analysis/2026-06-01_full_pipeline_offline_eval.md` (this)
- `eval/pipeline_cache.py`, `eval/pipeline_eval_offline.py`
- `eval/results/_offline_pipeline/cache/*.pkl` (11 surface caches), `offline_eval_results.{md,json}`
