# Soft-Veto Threshold Sweep — baseline RGB + IR + classifier + filters

**Scope:** ablate baseline RGB, IR (native + grayscale), sa32 classifier (argmax + soft-veto sweep), and both patch filters across three datasets. ~1000 frames per dataset via uniform stride. Single-frame stages only (no temporal).

**Soft-veto rule:** if RGB has ≥1 detection, keep RGB unless `P(reject_both) ≥ τ`; if RGB missed and classifier argmax ∈ {IR_only, both}, fall back to IR dets.

**Sweep grid:** τ ∈ {0.5, 0.7, 0.85, 0.95, 0.99}

## Headline — recommended τ

**τ = 0.99** maximises mean F1 across the three datasets (mean F1 = 0.7483).

| τ | mean F1 across datasets |
|---:|---:|
| 0.5 | 0.6721 |
| 0.7 | 0.7016 |
| 0.85 | 0.7210 |
| 0.95 | 0.7332 |
| 0.99 | 0.7483 ← |

## drone_video_drone
- n_frames = 1359, scoring = IOP @ 0.5, paired = False

| Stage | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| rgb_only | 967 | 342 | 267 | 0.7387 | 0.7836 | 0.7605 |
| rgb_filter | 813 | 271 | 421 | 0.7500 | 0.6588 | 0.7015 |
| ir_grayscale | 476 | 185 | 758 | 0.7201 | 0.3857 | 0.5024 |
| ir_grayscale_filter | 413 | 166 | 821 | 0.7133 | 0.3347 | 0.4556 |
| classifier_argmax | 736 | 754 | 498 | 0.4940 | 0.5964 | 0.5404 |
| classifier_argmax_filter | 634 | 594 | 600 | 0.5163 | 0.5138 | 0.5150 |
| softveto_0.5 | 744 | 236 | 490 | 0.7592 | 0.6029 | 0.6721 |
| softveto_0.5_filter | 608 | 184 | 626 | 0.7677 | 0.4927 | 0.6002 |
| softveto_0.7 | 803 | 252 | 431 | 0.7611 | 0.6507 | 0.7016 |
| softveto_0.7_filter | 660 | 194 | 574 | 0.7728 | 0.5348 | 0.6322 |
| softveto_0.85 | 845 | 265 | 389 | 0.7613 | 0.6848 | 0.7210 |
| softveto_0.85_filter | 700 | 207 | 534 | 0.7718 | 0.5673 | 0.6539 |
| softveto_0.95 | 878 | 283 | 356 | 0.7562 | 0.7115 | 0.7332 |
| softveto_0.95_filter | 727 | 220 | 507 | 0.7677 | 0.5891 | 0.6667 |
| softveto_0.99 ← | 935 | 330 | 299 | 0.7391 | 0.7577 | 0.7483 |
| softveto_0.99_filter | 779 | 263 | 455 | 0.7476 | 0.6313 | 0.6845 |

## Notes

- Soft-veto only changes behaviour when the classifier's `argmax` is `reject_both` (class 0). At τ → 1.0 the rule reduces to *never veto*; at τ → 0 it reduces to argmax (and may even *force-keep* RGB).
- The IR side fed to the classifier is the native IR detector on paired datasets, and `ir_grayscale` on RGB-only datasets (matches the production PySide pipeline's `_process_grayscale` mode).
- `*_filter` rows apply the rgb_filter patch verifier on top of the kept detections (matches the deployed cascade).
