# Video Test Evaluation — RGB, IR & Full Pipeline

**Date:** 2026-05-17  
**Scripts:** `eval/eval_video_tests.py`, `eval/eval_pipeline_video_tests.py`

---

## Datasets

**Source videos:** `G:\drone\drone detection video tests\rgb\{category}\`  
**Extracted datasets:** `datasets\drone detection video tests\rgb\{category}\{video}\`

- **Confusers (negative):** 10 videos, 1,250 frames (airplanes, birds, helicopters)
- **Drones (positive):** 9 videos, 1,359 frames (1,234 GT boxes, stride=3)

---

## 1. Per-Frame Detector Comparison (conf=0.25)

### 1.1 Aggregate — Drone Detection (9 videos, 1,359 frames)

| Model | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| selcom_1280 | 1002 | 61 | 231 | .943 | .813 | **.873** |
| baseline_trained | 924 | 27 | 309 | .972 | .749 | .846 |
| ir_final_gray | 605 | 26 | 628 | .959 | .491 | .649 |
| retrained_v2 | 559 | 17 | 674 | **.970** | .453 | .618 |

### 1.2 Aggregate — Confuser Rejection (10 videos, 1,250 frames)

| Model | FP frames | FP rate | FPPI |
|---|---|---|---|
| retrained_v2 | 212 | **.170** | .196 |
| ir_final_gray | 131 | .105 | .158 |
| baseline_trained | 461 | .369 | .512 |
| selcom_1280 | 516 | .413 | .709 |

---

## 2. Full Pipeline Evaluation

**Pipeline:** RGB YOLO -> IR YOLO (grayscale) -> Classifier -> Temporal (2/3) -> Patch Verifier Veto

### 2.1 Pipeline Components

| Component | Asset | Config |
|---|---|---|
| RGB YOLO | `baseline_trained` / `retrained_v2` / `selcom_1280` / `selcom_640` | conf=0.25 |
| IR YOLO | `models/IR_final_cleaned/weights/best.pt` | conf=0.40, grayscale input |
| Classifier | `classifier/fusion_models/scene_aware_v3more_32feat/model.joblib` | 32 features, 4-way trust |
| Temporal | `PerModalityTemporalState` | window=3, require=2, cooldown=0 (eval) |
| Patch Verifier | `classifier/runs/patches/confuser_filter4_rgb_v2_backup.pt` | thr=0.70, at alert time only |

### 2.2 Aggregate — Drone Detection Through Pipeline (bipartite IoP matching, conf=0.25)

Scoring: per-detection bipartite matching (IoP ≥ 0.5). Each detection matched to at most one GT, each GT to at most one detection.

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| baseline | RGB YOLO | 924 | 275 | 310 | .771 | .749 | .760 |
| baseline | IR (gray) | 605 | 149 | 629 | .802 | .490 | .609 |
| baseline | Classifier | 829 | 766 | 405 | .520 | .672 | .586 |
| retrained_v2 | RGB YOLO | 559 | 56 | 675 | .909 | .453 | .605 |
| retrained_v2 | IR (gray) | 605 | 149 | 629 | .802 | .490 | .609 |
| retrained_v2 | Classifier | 740 | 431 | 494 | .632 | .600 | .615 |
| selcom_1280 | RGB YOLO | 1002 | 542 | 232 | .649 | .812 | .721 |
| selcom_1280 | IR (gray) | 605 | 149 | 629 | .802 | .490 | .609 |
| selcom_1280 | Classifier | 815 | 986 | 419 | .453 | .660 | .537 |
| selcom_640 | RGB YOLO | 822 | 197 | 412 | .807 | .666 | .730 |
| selcom_640 | IR (gray) | 605 | 149 | 629 | .802 | .490 | .609 |
| selcom_640 | Classifier | 767 | 701 | 467 | .522 | .622 | .568 |

> **Note:** Classifier FP counts are high because the classifier merges RGB+IR detections when it trusts both modalities (label=3), doubling the detection count. This inflates per-detection FP but is operationally correct — the temporal gate collapses these into a single binary alert.

### 2.3 Aggregate — Confuser FP Through Pipeline (bipartite, per-detection)

| Model | RGB YOLO FP | IR FP | Classifier FP |
|---|---|---|---|
| baseline | 640 (.512) | 138 (.110) | 351 (.281) |
| retrained_v2 | 245 (.196) | 138 (.110) | 233 (.186) |
| selcom_1280 | 886 (.709) | 138 (.110) | 380 (.304) |
| selcom_640 | 325 (.260) | 138 (.110) | 259 (.207) |

### 2.4 Cascade FP Reduction — Raw YOLO to Final Alert (confusers)

| Model | Raw RGB FP | After Classifier | Final Alerts | Reduction |
|---|---|---|---|---|
| baseline | 640 | 351 | 22 | 96.6% |
| retrained_v2 | 245 | 233 | 16 | 93.5% |
| selcom_1280 | 886 | 380 | 19 | 97.9% |
| selcom_640 | 325 | 259 | 15 | 95.4% |

### 2.5 Alert Events

| Model | Category | Triggers | Vetoed | Passed | Veto Rate |
|---|---|---|---|---|---|
| baseline | Drone | 56 | 9 | 47 | 16.1% |
| baseline | Confuser | 36 | 14 | 22 | 38.9% |
| retrained_v2 | Drone | 49 | 9 | 40 | 18.4% |
| retrained_v2 | Confuser | 24 | 8 | 16 | 33.3% |
| selcom_1280 | Drone | 59 | 13 | 46 | 22.0% |
| selcom_1280 | Confuser | 29 | 10 | 19 | 34.5% |
| selcom_640 | Drone | 56 | 9 | 47 | 16.1% |
| selcom_640 | Confuser | 26 | 11 | 15 | 42.3% |

### 2.6 Segment-Based Temporal+Veto Metrics (3-frame segments, cooldown=0)

Each 3-frame segment scored as unit (binary). Segment positive if any frame has GT, fires if alert active on any frame.

#### Drone (456 segments)

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| baseline | Temporal 2/3 | 304 | 4 | 118 | .987 | .720 | **.833** |
| baseline | + Patch veto | 300 | 4 | 122 | .987 | .711 | **.826** |
| retrained_v2 | Temporal 2/3 | 274 | 6 | 148 | .979 | .649 | .781 |
| retrained_v2 | + Patch veto | 268 | 6 | 154 | .978 | .635 | .770 |
| selcom_1280 | Temporal 2/3 | 299 | 9 | 123 | .971 | .709 | .819 |
| selcom_1280 | + Patch veto | 295 | 8 | 127 | .974 | .699 | .814 |
| selcom_640 | Temporal 2/3 | 298 | 5 | 124 | .983 | .706 | .822 |
| selcom_640 | + Patch veto | 294 | 5 | 128 | .983 | .697 | .816 |

#### Confuser (420 segments)

| Model | Stage | FP segs | Total segs | Seg FPR |
|---|---|---|---|---|
| baseline | Temporal 2/3 | 77 | 420 | .183 |
| baseline | + Patch veto | 68 | 420 | .162 |
| retrained_v2 | Temporal 2/3 | 53 | 420 | .126 |
| retrained_v2 | + Patch veto | 50 | 420 | **.119** |
| selcom_1280 | Temporal 2/3 | 62 | 420 | .148 |
| selcom_1280 | + Patch veto | 57 | 420 | .136 |
| selcom_640 | Temporal 2/3 | 56 | 420 | .133 |
| selcom_640 | + Patch veto | 53 | 420 | .126 |

---

## 3. Per-Video Breakdown — Drone Detection (Segment P/R/F1, post-veto)

| Video (frames) | baseline P/R/F1 | retrained_v2 P/R/F1 | selcom_1280 P/R/F1 | selcom_640 P/R/F1 |
|---|---|---|---|---|
| drone_and_bird_sky_trees (114) | 1.00/.711/.831 | 1.00/.711/.831 | 1.00/.711/.831 | 1.00/.711/.831 |
| drone_attacked_bird_mtn (108) | 1.00/.516/.681 | 1.00/.290/.450 | 1.00/.645/.784 | 1.00/.581/.735 |
| drone_over_mtn_birds (68) | 1.00/.435/.606 | 1.00/.478/.647 | 1.00/.261/.414 | 1.00/.261/.414 |
| drone_seagull_attack (235) | .931/.818/.871 | .898/.667/.765 | .892/.879/.885 | .918/.849/.882 |
| drone_takeoff_ground (163) | 1.00/.944/.971 | .980/.926/.952 | .980/.907/.942 | 1.00/.926/.962 |
| drone_takeoff_short (116) | 1.00/.744/.853 | 1.00/.692/.818 | 1.00/.641/.781 | 1.00/.667/.800 |
| drone_takeoff_trees (166) | 1.00/.821/.902 | 1.00/.821/.902 | 1.00/.768/.869 | 1.00/.786/.880 |
| seagulls_beach (239) | 1.00/.646/.785 | 1.00/.477/.646 | 1.00/.646/.785 | 1.00/.554/.713 |
| two_birds_drone (150) | 1.00/.500/.667 | 1.00/.460/.630 | 1.00/.500/.667 | 1.00/.620/.765 |

---

## Models

| ID | Weights | imgsz | Preprocessing |
|---|---|---|---|
| baseline_trained | `RGB model/Yolo26n_trained/weights/best.pt` | 640 | None |
| retrained_v2 | `RGB model/Yolo26n_retrained_v2/weights/best.pt` | 640 | None |
| selcom_1280 | `RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt` | 1280 | None |
| selcom_640 | `RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt` | 640 | None |
| ir_final_gray | `models/IR_final_cleaned/weights/best.pt` | 640 | BGR->Gray->BGR |

---

## Scoring Methodology

- **Tier 1 (Detection):** Per-detection bipartite matching with IoP ≥ 0.5. Each detection matched to best-overlapping GT, each GT consumed at most once. Standard PASCAL-VOC style matching. Uses `score_detections()` from `eval/metrics.py`.
- **Tier 2 (Segments):** Binary per-segment. 3-frame segments matching the temporal 2-of-3 window. Segment positive if any frame has GT, fires if alert active on any frame. cooldown=0 for evaluation (cooldown=5 in production GUI to avoid operator spam).
- **Confuser FPR:** FP detections / total frames at Tier 1; FP segments / total segments at Tier 2.

---

## Output Files

| File | Contents |
|---|---|
| `eval/results/video_tests/video_tests_comparison.csv` | Per-frame eval (all models, all videos) |
| `eval/results/video_tests/video_tests_comparison.json` | Same, full detail |
| `eval/results/pipeline_video_tests/pipeline_comparison.csv` | Pipeline eval (all stages, P/R/F1 + segments) |
| `eval/results/pipeline_video_tests/pipeline_comparison.json` | Same, full detail |
| `eval/results/pipeline_video_tests/{cat}/{video}/{model}.json` | Per-video pipeline detail |

---

## Recreation

```powershell
# Per-frame detector eval (all models)
python eval/eval_video_tests.py

# Full pipeline eval (all RGB models x all categories)
python eval/eval_pipeline_video_tests.py

# Pipeline with specific models/categories
python eval/eval_pipeline_video_tests.py --rgb-models baseline_trained --categories drone
```

### Parameters

- RGB confidence: 0.25
- IR confidence: 0.40
- Patch verifier threshold: 0.70
- Temporal window: 3, require: 2, cooldown: 0 (eval) / 5 (production)
- IoP matching threshold: 0.5
- Device: CUDA GPU 0

- Device: CUDA GPU 0

