# Soft-Veto Threshold Sweep — baseline RGB + IR + classifier + filters

**Scope:** ablate baseline RGB, IR (native + grayscale), sa32 classifier (argmax + soft-veto sweep), and both patch filters across three datasets. ~1000 frames per dataset via uniform stride. Single-frame stages only (no temporal).

**Soft-veto rule:** if RGB has ≥1 detection, keep RGB unless `P(reject_both) ≥ τ`; if RGB missed and classifier argmax ∈ {IR_only, both}, fall back to IR dets.

**Sweep grid:** τ ∈ {0.5, 0.7, 0.85, 0.95}

## Headline — recommended τ

**Mean-F1 winner across the three datasets: τ = 0.95** (mean F1 = 0.7897).

| τ | mean F1 across datasets |
|---:|---:|
| 0.5 | 0.7768 |
| 0.7 | 0.7801 |
| 0.85 | 0.7839 |
| 0.95 | 0.7897 ← |

**Operating-mode-aware recommendation (read this carefully):**

| Mode | Best stage | F1 antiuav | F1 svanstrom | F1 drone_video |
|---|---|---:|---:|---:|
| Paired (RGB + native IR) | `classifier_argmax` | **0.977** | **0.890** | n/a |
| RGB-only / grayscale fallback | `softveto @ τ=0.95` | n/a | n/a | **0.586** |

On paired data the full trust-aware classifier wins outright — modality arbitration routes Svanström RGB-confuser frames to the clean IR stream (RGB collapses to F1=0.43 alone, classifier rescues to F1=0.89). Soft-veto is *strictly worse* on paired data because it biases towards RGB even when IR is the correct trust target.

On RGB-only deployment the classifier sees identical grayscale-derived features for both modality branches — an OOD shift for a model trained on paired thermal+RGB. Argmax then over-rejects (`reject_both`) on legitimate drone frames. Soft-veto fail-open at τ=0.95 recovers those frames and boosts F1 from 0.477 (argmax) → 0.586 (+11 pp), matching the +11.6 pp gain reported in the design diagnostic.

**Use:** `classifier_argmax` when running paired; `softveto @ τ=0.95` when running grayscale / RGB-only. Same trained model, different decision rule.

## antiuav
- n_frames = 1005, scoring = IOU @ 0.5, paired = True

| Stage | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| rgb_only | 921 | 59 | 17 | 0.9398 | 0.9819 | 0.9604 |
| rgb_filter | 921 | 58 | 17 | 0.9408 | 0.9819 | 0.9609 |
| ir_native | 928 | 22 | 62 | 0.9768 | 0.9374 | 0.9567 |
| ir_native_filter | 928 | 22 | 62 | 0.9768 | 0.9374 | 0.9567 |
| ir_grayscale | 19 | 14 | 919 | 0.5758 | 0.0203 | 0.0391 |
| ir_grayscale_filter | 19 | 14 | 919 | 0.5758 | 0.0203 | 0.0391 |
| classifier_argmax | 1,846 | 70 | 18 | 0.9635 | 0.9903 | 0.9767 |
| classifier_argmax_filter | 1,846 | 69 | 18 | 0.9640 | 0.9903 | 0.9770 |
| softveto_0.5 | 980 | 60 | 12 | 0.9423 | 0.9879 | 0.9646 |
| softveto_0.5_filter | 980 | 59 | 12 | 0.9432 | 0.9879 | 0.9650 |
| softveto_0.7 | 980 | 60 | 12 | 0.9423 | 0.9879 | 0.9646 |
| softveto_0.7_filter | 980 | 59 | 12 | 0.9432 | 0.9879 | 0.9650 |
| softveto_0.85 | 980 | 60 | 12 | 0.9423 | 0.9879 | 0.9646 |
| softveto_0.85_filter | 980 | 59 | 12 | 0.9432 | 0.9879 | 0.9650 |
| softveto_0.95 ← | 980 | 60 | 12 | 0.9423 | 0.9879 | 0.9646 |
| softveto_0.95_filter | 980 | 59 | 12 | 0.9432 | 0.9879 | 0.9650 |

## svanstrom
- n_frames = 1026, scoring = IOP @ 0.5, paired = True

| Stage | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| rgb_only | 412 | 1,099 | 8 | 0.2727 | 0.9810 | 0.4267 |
| rgb_filter | 396 | 596 | 24 | 0.3992 | 0.9429 | 0.5609 |
| ir_native | 396 | 25 | 14 | 0.9406 | 0.9659 | 0.9531 |
| ir_native_filter | 389 | 24 | 21 | 0.9419 | 0.9488 | 0.9453 |
| ir_grayscale | 230 | 202 | 190 | 0.5324 | 0.5476 | 0.5399 |
| ir_grayscale_filter | 230 | 143 | 190 | 0.6166 | 0.5476 | 0.5801 |
| classifier_argmax | 806 | 185 | 14 | 0.8133 | 0.9829 | 0.8901 |
| classifier_argmax_filter | 783 | 149 | 37 | 0.8401 | 0.9549 | 0.8938 |
| softveto_0.5 | 416 | 171 | 5 | 0.7087 | 0.9881 | 0.8254 |
| softveto_0.5_filter | 400 | 135 | 21 | 0.7477 | 0.9501 | 0.8368 |
| softveto_0.7 | 416 | 173 | 5 | 0.7063 | 0.9881 | 0.8238 |
| softveto_0.7_filter | 400 | 136 | 21 | 0.7463 | 0.9501 | 0.8359 |
| softveto_0.85 | 417 | 174 | 3 | 0.7056 | 0.9929 | 0.8249 |
| softveto_0.85_filter | 401 | 136 | 19 | 0.7467 | 0.9548 | 0.8380 |
| softveto_0.95 ← | 417 | 182 | 3 | 0.6962 | 0.9929 | 0.8184 |
| softveto_0.95_filter | 401 | 140 | 19 | 0.7412 | 0.9548 | 0.8345 |

## drone_video_drone
- n_frames = 1359, scoring = IOP @ 0.5, paired = False

| Stage | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| rgb_only | 647 | 556 | 587 | 0.5378 | 0.5243 | 0.5310 |
| rgb_filter | 604 | 451 | 630 | 0.5725 | 0.4895 | 0.5277 |
| ir_grayscale | 476 | 185 | 758 | 0.7201 | 0.3857 | 0.5024 |
| ir_grayscale_filter | 413 | 166 | 821 | 0.7133 | 0.3347 | 0.4556 |
| classifier_argmax | 598 | 678 | 636 | 0.4687 | 0.4846 | 0.4765 |
| classifier_argmax_filter | 526 | 575 | 708 | 0.4777 | 0.4263 | 0.4505 |
| softveto_0.5 | 587 | 351 | 647 | 0.6258 | 0.4757 | 0.5405 |
| softveto_0.5_filter | 505 | 271 | 729 | 0.6508 | 0.4092 | 0.5025 |
| softveto_0.7 | 618 | 387 | 616 | 0.6149 | 0.5008 | 0.5520 |
| softveto_0.7_filter | 535 | 297 | 699 | 0.6430 | 0.4335 | 0.5179 |
| softveto_0.85 | 637 | 395 | 597 | 0.6172 | 0.5162 | 0.5622 |
| softveto_0.85_filter | 553 | 303 | 681 | 0.6460 | 0.4481 | 0.5292 |
| softveto_0.95 ← | 685 | 418 | 549 | 0.6210 | 0.5551 | 0.5862 |
| softveto_0.95_filter | 595 | 322 | 639 | 0.6489 | 0.4822 | 0.5532 |

## Scoring rule

All rows use **trust-aware (Rule A)** scoring per the email rule:

- `label=0` (reject_both): kept=[] on both sides; both modalities' GTs become FN.
- `label=1` (trust RGB): RGB dets scored vs RGB GT; IR GT excluded.
- `label=2` (trust IR): IR dets scored vs IR GT; RGB GT excluded.
- `label=3` (trust both): RGB dets vs RGB GT + IR dets vs IR GT (TPs and FNs sum across modalities — that's why classifier_argmax on Anti-UAV reports TP=1,846 vs RGB-only's 921).

This is the operationally relevant rule (the system isn't penalised for picking the right modality). The legacy strict RGB-side-only scoring is *not* used in this doc.

## Notes

- **Soft-veto rule**: applies on top of the classifier's `predict_proba` output. The rule only deviates from argmax when `argmax = reject_both` (class 0). At τ → 1.0 the rule reduces to *never veto*; at τ → 0 it reduces to argmax.
- **IR side fed to classifier**: native IR detector on paired datasets; `ir_grayscale` on RGB-only datasets (matches the production PySide pipeline's `_process_grayscale` mode at `ir_gui/fusion/pipeline.py:200`).
- **`*_filter` rows**: patch verifier (rgb_filter / ir_filter, threshold 0.70) applied on top of the kept detections — matches the deployed cascade.
- **The `ir_grayscale` row on Anti-UAV is intentionally near-zero** (R=0.020). On a confuser-free paired dataset where the real IR feed is available, ir_grayscale is the wrong fallback — it's shown here purely as a sanity check that the cross-modal fallback path doesn't accidentally fire.

## Delivered

- `docs/analysis/full_pipeline_ablations/softveto_ablation.md`
- `docs/analysis/full_pipeline_ablations/csv/softveto_ablation.csv`
