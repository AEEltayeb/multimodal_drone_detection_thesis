# Trust-Classifier Routing Robustness — Re-plan Around the Real Goal

**Date:** 2026-06-01 · **Trigger:** user re-anchored the goal — *the classifier must ROUTE correctly:
drone-in-RGB → trust_rgb, drone-in-grayscale/IR → trust_ir, both → trust_both, neither → reject — robust across
regimes.* That means the metric is **per-class P/R/F1 for trust_rgb/trust_ir, split by regime**, not the binary
trust-vs-reject we'd been screening on. · **Script:** `classifier/train_routing_robust.py` (Phase 1a) ·
**Data:** `fusion_dataset_lean19.csv` (8,871 rows; conf_sum derived). Method: statistics-first.

## Phase 1a result — the goal-aligned metric pinpoints the real failure

robust6, GroupShuffleSplit, **per-class P/R/F1 by regime** (test set):

| class | overall F1 | thermal R | **grayscale R** | grayscale P |
|---|---|---|---|---|
| reject | 0.894 | 0.959 | 0.881 | **0.440** ⚠ |
| **trust_rgb** | **0.482** ⚠ | 0.520 | **0.190** 🔴 | 1.000 |
| trust_ir | 0.883 | 0.879 | 0.500 (n=12) | 0.750 |
| trust_both | 0.981 | 1.000 | 0.913 | 0.905 |

**The failure is `trust_rgb` recall, and it is catastrophic on grayscale (R = 0.19).** When a drone is visible
**only in RGB** — the *dominant* grayscale case (499 of the dataset's 700 trust_rgb frames are grayscale) — the
classifier routes it to trust_rgb only **19%** of the time. Where do the other 81% go? Into **reject** (grayscale
reject precision is only 0.44 — half of what it rejects is actually a drone). 

**This IS the pipeline-level "grayscale over-rejection," now mechanistically explained:** robust6 can't confirm the
RGB drone strongly enough while its IR features are blind (Phase 0: IR features → chance on grayscale), so it
defaults to reject. trust_both (0.98) and thermal trust_ir (0.92) are already fine — they are *not* the problem.

## What this does to the plan

**1. `conf_sum` (as rgb_max_conf+ir_max_conf) is redundant — drop it.** robust7 vs robust6: macro-F1 0.807 vs
0.810 (a hair *worse*), trust_rgb F1 0.471 vs 0.482. robust6 already contains both max-confidences, so their sum
adds nothing to a tree model. (The *true* `conf_sum` = sum over **all** detections carries a count signal the max
doesn't; only worth testing in the re-mine, low priority.) **My earlier "add conf_sum" recommendation is retracted
on this evidence.**

**2. The RGB verifier score is now the *targeted* lever — and the evidence converges.** Phase 0 (verifier) showed
`rgb_verifier_pdrone` separates drone-from-confuser at **AUROC 0.949** and is always available (RGB is RGB even on
grayscale frames). Phase 1a shows the hole is **trust_rgb recall**. These meet exactly: a strong "RGB really has a
drone" feature should push grayscale RGB-drones out of *reject* and into *trust_rgb*. **Sharp, falsifiable
prediction:** adding `rgb_verifier_pdrone` lifts grayscale trust_rgb recall well above 0.19, without hurting
trust_both/trust_ir.

**3. Grayscale `trust_ir` is data-starved, not feature-limited.** Only **62** grayscale trust_ir frames exist
(n=12 in test). No feature fixes a 62-sample class. This is a *data* gap (mine more grayscale frames where IR-alone
catches the drone) — but it is also the *rare* case: on grayscale the drone is almost always in RGB, so getting
**trust_rgb** right is most of the grayscale goal. Flag it; don't block on it.

## Revised plan (statistics-first, goal-aligned)

| phase | action | status |
|---|---|---|
| 1a | per-class-by-regime eval; test conf_sum | **done** — found trust_rgb recall is the hole; conf_sum redundant |
| **1b** | **re-mine fusion data with `rgb_verifier_pdrone` + `ir_verifier_pdrone` (+ true conf_sum); train robust6 vs +rgb_verif vs +both_verif; report the same per-class-by-regime table** | **next (one GPU re-mine)** |
| 1c | if trust_rgb still weak, add the `*_diff` routing features (Phase 0 §4: conf_diff/area_diff drove single-modality routing) | conditional |
| 3 | full-pipeline ablation: confirm **classifier→filter** stays best (not filter→classifier) with the new feature; compare P/R/F1 vs sa32 + robust6 | after 1b |
| — | grayscale trust_ir **data** mine (separate track) | backlog |

**Win condition for 1b:** grayscale `trust_rgb` recall ↑ (from 0.19) and overall macro-F1 ↑, with no regression on
trust_both / thermal trust_ir / confuser fire-rate.

## "Best combo" right now (the user's question)
Among what is testable without the re-mine: **robust6** (macro P/R/F1 = 0.882 / 0.781 / 0.810) — the conf_sum
variant is redundant. The combo expected to win, **robust6 + rgb_verifier_pdrone**, needs the Phase-1b re-mine to
measure; Phase 1a tells us *exactly why* it should win and *what number to watch* (grayscale trust_rgb recall).

## Delivered
- `docs/analysis/2026-06-01_routing_robustness_replan.md` (this doc)
- `classifier/train_routing_robust.py` (per-class-by-regime routing trainer; auto-detects verifier columns for 1b)
- `classifier/fusion_models/routing_robust/routing_compare.json`, `trust_routing_best.joblib`
