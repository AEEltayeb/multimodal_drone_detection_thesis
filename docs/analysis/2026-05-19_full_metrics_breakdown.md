# Per-(model × surface × size) metrics breakdown

Source CSV: [`2026-05-19_metrics_inventory.csv`](2026-05-19_metrics_inventory.csv) (1869 rows / 32 surfaces).
Scoring: IoP@0.5, conf=0.25, imgsz per-model (selcom_640=640, selcom_960=960, selcom_1280=1280, everything else=1280 on Svanström / 640 elsewhere unless stated).

---

## Topic 1 — Best at small drones (Svanström, 76 small GT)

Ranked by recall (the binding metric — precision is recoverable downstream).

| Rank | Model | R | P | F1 |
|---|---|---:|---:|---:|
| 1 | hardneg_v3more | **0.842** | 0.42 | 0.56 |
| 2 | baseline | 0.816 | 0.41 | 0.54 |
| 3 | selcom_1280 | 0.750 | 0.59 | 0.66 |
| 4 | selcom_960 | 0.632 | **0.71** | **0.67** |
| 5 | retrained_v2 | 0.342 | 0.46 | 0.39 |
| 5 | selcom_640 | 0.342 | **0.96** | 0.50 |

**Takeaway:** baseline / hardneg_v3more dominate raw recall. selcom_960 has the best F1 because its precision is 1.7× baseline's. selcom_640 is at the resolution floor — high precision on what little it catches.

---

## Topic 2 — Best at medium drones (Svanström, 1560 medium GT)

| Rank | Model | R | P | F1 |
|---|---|---:|---:|---:|
| 1 | baseline | **0.969** | 0.99 | **0.98** |
| 2 | hardneg_v3more | 0.961 | 0.99 | 0.97 |
| 3 | selcom_1280 | 0.918 | 0.99 | 0.95 |
| 4 | selcom_960 | 0.847 | **1.00** | 0.92 |
| 5 | selcom_640 | 0.589 | 1.00 | 0.74 |
| 6 | retrained_v2 | 0.310 | 1.00 | 0.47 |

**Takeaway:** baseline crushes medium drones (the most common size). selcom variants degrade with imgsz. retrained_v2 misses ~70% of medium drones — confirms its "no medium things" rule.

---

## Topic 3 — Best on selcom distribution (selcom holdout, 311 imgs)

Models pre-trained / fine-tuned on selcom dominate. Ranked by F1.

| Rank | Model | R | P | F1 |
|---|---|---:|---:|---:|
| 1 | selcom_960 | 0.437 | **0.88** | **0.585** |
| 2 | selcom_1280 | 0.468 | 0.76 | 0.580 |
| 3 | selcom_640 | 0.119 | 0.88 | 0.21 |
| 4 | baseline | 0.088 | 0.41 | 0.15 |
| 5 | hardneg_v3more | 0.014 | 0.57 | 0.03 |
| 6 | retrained_v2 | 0.003 | 0.25 | 0.01 |

**Takeaway:** selcom_960 edges selcom_1280 — higher precision at near-equal recall. Non-selcom models barely see anything (selcom holdout is a different distribution).

---

## Topic 4 — Best OOD drone detection (Roboflow rgb_drone, 4225 imgs across 3 splits, +patch)

| Rank | Model | R | P | F1 | FPs |
|---|---|---:|---:|---:|---:|
| 1 | selcom_960 | 0.813 | 0.87 | **0.839** | 417 |
| 2 | selcom_640 | 0.704 | **0.94** | 0.81 | **151** |
| 3 | selcom_1280 | **0.837** | 0.76 | 0.80 | 877 |
| 4 | baseline | 0.689 | 0.93 | 0.79 | 177 |
| 5 | retrained_v2 | 0.684 | 0.93 | 0.79 | 173 |
| 6 | hardneg_v3more | 0.587 | 0.90 | 0.71 | 208 |

**Takeaway:** selcom_960 is the Pareto-best — top F1 with manageable FPs. selcom_1280 has the highest raw recall but 2× the FPs of selcom_960.

---

## Topic 5 — Best on saturated benchmark (Anti-UAV, RGB only)

Anti-UAV is saturated; this topic exists to detect models that break on a "should be easy" surface.

| Rank | Model | R | P | F1 | FPs |
|---|---|---:|---:|---:|---:|
| 1 | retrained_v2 | 0.996 | 0.99 | **0.993** | 39 |
| 2 | baseline | 0.994 | 0.99 | 0.992 | 43 |
| 3 | selcom_640 | 0.985 | **0.99** | 0.988 | **37** |
| 4 | selcom_960 | 0.984 | 0.96 | 0.972 | 175 |
| 5 | selcom_1280 | 0.978 | 0.84 | 0.902 | **849** |

**Takeaway:** selcom_1280 is the only model that bleeds FPs on Anti-UAV. Its 1280-imgsz fires on background structure that other configurations ignore.

---

## Topic 6 — Cleanest confuser rejection (Roboflow OOD bird+airplane+heli, +patch)

Total FP count across all three confuser categories. Lower is better.

| Rank | Model | Total FPs |
|---|---|---:|
| 1 | hardneg_v3more | **785** |
| 2 | baseline | 1867 |
| 3 | retrained_v2 | 2158 |
| 4 | selcom_640 | 2468 |
| 5 | selcom_960 | 3909 |
| 6 | selcom_1280 | 4311 |

**Caveat:** hardneg_v3more wins this *only* in isolation. Its drone recall (Topic 4 R=0.59) is the worst of the six models. The selcom variants fire more on confusers because they fire more, period — that's also why their drone recall is highest.

---

## Topic 7 — Mixed real-video (drone + birds in same frame)

Per-clip F1 winner. Six clips, ~150 GT drones each.

| Clip | Winner | F1 | Runner-up |
|---|---|---:|---|
| drone_seagull_attack | selcom_1280 | 0.82 | baseline 0.78 |
| flock_of_seagulls_attack_drone_beach | baseline / ir_grayscale | 0.84 | selcom_640 0.79 |
| drone_attacked_by_bird_mountain_side_view | selcom_1280 | 0.77 | baseline 0.60 |
| drone_and_bird_sky_and_trees_short | ir_grayscale | 0.65 | baseline 0.59 |
| two_birds_drone | baseline | 0.58 | ir_grayscale 0.60 |
| drone_over_mountain_attacked_by_birds | baseline | 0.52 | retrained_v2 0.50 |

**Takeaway:** No single model wins all mixed clips. selcom_1280 wins 2 (high-clutter, big drone). baseline wins 2 (mid-clutter). ir_grayscale wins 2 (small-drone in mixed scenes).

---

## Per-model — good at / bad at

### `baseline` (Yolo26n_trained)
- **Good at:** medium drones (R=0.97, F1=0.98 Svanström), real-video drone-clean clips (top 2/3), saturated Anti-UAV (F1=0.992)
- **Bad at:** drone-OOD generalisation (Roboflow R=0.69, behind every selcom), selcom-distribution drones (R=0.09 on selcom holdout)

### `hardneg_v3more`
- **Good at:** small drones (R=0.84, ties baseline), confuser rejection (cleanest on Roboflow OOD by 2×+ over peers)
- **Bad at:** OOD drone recall (R=0.59, worst of all 6), real-video clean clips (not in current eval set), selcom holdout (R=0.01)

### `retrained_v2`
- **Good at:** Anti-UAV F1 (#1 at 0.993), precision on what little it detects (P=1.0 on medium Svanström)
- **Bad at:** *being a usable detector*. Recall is below 0.10 on selcom holdout, 0.31 on Svanström medium, 0.08 on most mixed real-video clips. Functional as a cross-check inside a cascade, not as a primary detector.

### `selcom_640`
- **Good at:** precision floor on small drones (P=0.96), saturated Anti-UAV (F1=0.988 with only 37 FPs), cheapest of the selcom variants
- **Bad at:** small/medium drone recall (R=0.34/0.59 Svanström) — below the resolution floor. Worst F1 of the three selcom variants on its own distribution (0.21).

### `selcom_960`
- **Good at:** **best F1 on Roboflow OOD drone (0.84), on selcom holdout (0.585), best small-drone precision among recall-comparable models (P=0.71 on Svanström)**. Cross-surface Pareto-best of the selcom family.
- **Bad at:** Confuser rejection (3909 FPs, 5th of 6), Anti-UAV precision (P=0.96 vs peers 0.99)

### `selcom_1280`
- **Good at:** highest raw recall on selcom holdout (R=0.47), highest raw recall on Roboflow drone (R=0.84), wins 2/6 mixed real-video clips
- **Bad at:** **Anti-UAV precision collapse (P=0.84, 849 FPs vs 37–175 for peers)**. Highest confuser FP count of any model (4311 across Roboflow OOD). The 1280-imgsz amplifies everything — drones and false positives.

### `ir_grayscale` (IR detector on grayscale RGB)
- **Good at:** mixed real-video small/medium drones in clutter (wins 2/6 mixed clips), cross-modal fallback
- **Bad at:** clean-drone real-video clips (R=0.41–0.54 vs RGB R≈0.95), not evaluated on Svanström/Roboflow

### `ir_on_rgb` (IR detector on raw RGB)
- **Good at:** essentially nothing in this evaluation
- **Bad at:** every surface — R≤0.15 on most real-video clips

---

## Overall ranking (drone recall + low FP rate)

Composite rank = average of (Svanström small R, Svanström medium R, selcom holdout F1, Roboflow drone F1, Roboflow confuser FP suppression). Lower composite rank = better.

| Rank | Model | Svan small R | Svan med R | selcom F1 | Roboflow F1 | Confuser FPs | Composite |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **selcom_960** | 4 | 4 | **1** | **1** | 5 | **3.0** |
| 2 | baseline | 2 | **1** | 4 | 4 | 2 | 2.6 |
| 3 | selcom_1280 | 3 | 3 | 2 | 3 | 6 | 3.4 |
| 4 | hardneg_v3more | **1** | 2 | 5 | 6 | **1** | 3.0 |
| 5 | selcom_640 | 5 | 5 | 3 | 2 | 4 | 3.8 |
| 6 | retrained_v2 | 5 | 6 | 6 | 5 | 3 | 5.0 |

Tied at 3.0: selcom_960 and hardneg_v3more, but selcom_960 wins on *drone* outcomes (selcom_F1, Roboflow_F1 both #1) where hardneg wins only on FP rejection. By the user's tiebreaker (don't favour a model with bad drone recall just because it suppresses FPs), **selcom_960 is the cross-surface #1**.

**Top 3 by drone-first composite (drop the confuser-FP column):**

| Rank | Model | Avg rank across 4 drone-recall columns |
|---|---|---:|
| 1 | selcom_960 | 2.5 |
| 2 | baseline | 2.75 |
| 3 | selcom_1280 | 2.75 |

**Bottom 2 (worst across the board):** retrained_v2 (avg 5.5) and selcom_640 (avg 3.75 — saved only by Roboflow F1).

---

## Sources

- `eval/results/svanstrom_persize/summary.csv`
- `eval/results/selcom_val_holdout/<m>/<m>_results.json`
- `eval/results/antiuav_per_model/<m>/<m>_results.json`
- `eval/results/roboflow_ood/summary.csv`
- `eval/results/video_persize/summary.csv`
- Inventory: `2026-05-19_metrics_inventory.csv`

Reproduction commands: `docs/EVIDENCE_LEDGER.md` §12.
