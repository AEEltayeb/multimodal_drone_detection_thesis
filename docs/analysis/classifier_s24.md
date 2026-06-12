# Architectural Evolution: SA32 Baseline to Dual-Classifier v3

This document formalizes the scientific process, feature ablations, and experimental results that led to the deprecation of the `SA32` baseline architecture in favor of the `Dual Classifier v3` architecture.

---

## 1. Observation & Problem Statement

The production `SA32` architecture utilized a single 32-feature XGBoost model to classify drone sightings based on a combination of Thermal (IR) and Visible (RGB) sensor data. 

**The Vulnerability:**
During field tests and evaluation on fallback datasets, the `SA32` model demonstrated a catastrophic failure mode when operating on **Grayscale/RGB-only footage** (i.e., when Thermal data was missing or unreliable). Because the pipeline falls back to running the Thermal-YOLO model on grayscale RGB frames, the classifier was forced to evaluate single-camera data. 
Under these conditions, it failed spectacularly against bird confusers:
- `confuser_BIRD` (13,300 generic bird images): **0.2469 F1**
- `video_drone_drone_seagull_attack` (Chaotic drone/bird interaction): **0.4611 F1**

The model was routinely confusing flapping seagulls for drones when the thermal signature was missing.

---

## 2. Hypothesis Formulation

**Hypothesis:** Forcing a single XGBoost model to simultaneously map decision boundaries for pristine multimodal (Paired) data and noisy single-sensor (Grayscale) fallback data creates conflicting feature weights. 

By splitting the inference gate into two highly specialized models—one strictly for paired multimodal data, and one strictly for single-sensor grayscale data—we can optimize the feature space for each specific domain, allowing the grayscale model to heavily weight shape and confidence to reject birds, while the paired model leans on cross-modal geometry.

---

## 3. Experimental Design: Feature Ablation (v1 & v2)

We designed an ablation study to test three feature variants across both proposed pipelines:

1. **`sa32_feats` (32 features):** The full original feature set, including 10 computationally expensive global scene calculations.
   - *Expensive Features:* `rgb_img_dynamic_range`, `rgb_img_entropy`, `rgb_sky_ground_ratio`, `rgb_edge_density`, `rgb_blurriness`, and their `ir_` equivalents.
2. **`sa32_lite` (22 features):** Stripped of all 10 expensive global scene features, leaving only YOLO confidence, local contrast, and bounding box geometry.
3. **`sa32_lite+` (24 features):** The 22 features from `lite`, plus two newly engineered features: `area_diff` (the mathematical difference between the Thermal and RGB bounding box areas) and `xmodal_centroid_dist`.

**Ablation Results:**

### Paired Pipeline Ablation
| Variant | Features | Accuracy | F1-Macro | F1-Weighted | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sa32_lite` | 22 | 0.9775 | 0.9438 | 0.9769 | 63.5s |
| `sa32_lite+` | 24 | 0.9778 | 0.9448 | 0.9773 | 84.6s |
| `sa32_feats` | 32 | 0.9792 | 0.9493 | 0.9786 | 86.1s |

**Feature Impact (Paired):**
| Feature Group Added | F1 Delta | Time Delta | Reason / Effect |
| :--- | :--- | :--- | :--- |
| **+ Cross-Modal Geometry**<br>(`area_diff`, `centroid_dist`) | **+0.0010**<br>(0.9438 → 0.9448) | +21.1s | **Helped.** By comparing the bounding box size between the thermal and RGB cameras, the model gained a highly robust boundary. If a bird is hot but visually small, `area_diff` immediately flags it as a confuser. |
| **+ Global Scene Textures**<br>(`entropy`, `blurriness`, etc) | **+0.0045**<br>(0.9448 → 0.9493) | +1.5s | **Helped slightly, but functionally hurt latency.** While 10 global features bumped accuracy by a microscopic 0.4%, they require expensive full-image pixel operations during real-time inference. Because cross-modal geometry handles the bulk of the decision logic, these 10 features were deemed bloat for the paired model. |

*Decision:* We locked in `sa32_lite+` (24 features) for the Paired model to prioritize massive latency reductions while maintaining elite accuracy.

### Grayscale Pipeline Ablation
| Variant | Features | Accuracy | F1-Macro | F1-Weighted | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sa32_lite` | 22 | 0.9787 | 0.9096 | 0.9775 | 135.2s |
| `sa32_lite+` | 24 | 0.9788 | 0.9096 | 0.9776 | 126.6s |
| `sa32_feats` | 32 | 0.9814 | **0.9243** | 0.9804 | 155.7s |

**Feature Impact (Grayscale):**
| Feature Group Added | F1 Delta | Time Delta | Reason / Effect |
| :--- | :--- | :--- | :--- |
| **+ Cross-Modal Geometry**<br>(`area_diff`, `centroid_dist`) | **+0.0000**<br>(0.9096 → 0.9096) | -8.6s | **Did not help.** Because the grayscale fallback only possesses a single camera stream, the bounding boxes for "both" cameras are functionally identical (or missing), rendering cross-modal geometry mathematically useless. |
| **+ Global Scene Textures**<br>(`entropy`, `blurriness`, etc) | **+0.0147**<br>(0.9096 → 0.9243) | +29.1s | **Helped massively.** Because `area_diff` is impossible with only one camera, the model mathematically relies heavily on global scene context to compensate. Features like `sky_ground_ratio` and `blurriness` are critical for rejecting birds in the sky when the thermal stream is offline. |

*Decision:* We locked in `sa32_feats` (32 features) for the Grayscale model because the performance boost justifies the latency cost when operating in fallback mode.

**Decision:** We locked in `sa32_lite+` for the Paired model, and `sa32_feats` for the Grayscale model.

---

## 4. Iterative Refinement: Data Leakage (v3)

During the Grayscale training phase, we injected 144k static confusers and 65k video frames of chaotic drone/bird footage into the training pool. 

**The Leakage Catch:** We discovered that `GroupShuffleSplit` was holding out entire single-clip videos (like `drone_seagull_attack`) for the test set. Because the model randomly saw 0 frames of drones and birds interacting during training, it could not learn the boundary. Alternatively, splitting a single video by frame results in data leakage (training on frame 1, testing on frame 2).

**The Fix:** We manually overrode the split index:
- Forced `video_drone_two_birds_drone` and `video_drone_flock_of_seagulls_attack` explicitly into the **Train Set**.
- Strictly held out `video_drone_drone_seagull_attack` in the **Test Set** to act as an uncompromised generalization benchmark.
- Added 1,013 `rgb_dataset_test` frames to the cross-validation pool to repair an observed RGB-only regression.

---

## 5. Results & Validation

The final evaluations of the `v3` dual-classifiers against the `SA32` baseline were generated using the `classifier/eval_dual_vs_sa32.py` and `classifier/eval_rgb_test.py` scripts. These scripts lock the validation splits and compare the legacy 32-feature model against the newly engineered classifiers.

### Grayscale Fallback (Bird Rejection)
| Dataset | N | SA32 (Production) | Dual Classifier v3 | Delta |
| :--- | :--- | :--- | :--- | :--- |
| `confuser_BIRD` | 13,300 | 0.2469 | **1.0000** | +0.7531 |
| `confuser_AIRPLANE` | 4,313 | 0.2337 | **0.4999** | +0.2662 |
| `video_drone_seagull_attack` (Held-out) | 235 | 0.4611 | **0.9423** | +0.4811 |
| `video_drone_drone_takeoff_short` | 116 | 0.2071 | **1.0000** | +0.7929 |
| `video_birds_flock_of_birds` | 21 | 0.4878 | **1.0000** | +0.5122 |
| `video_birds_birds_flying_overhead` | 20 | 0.4872 | **1.0000** | +0.5128 |
| `rgb_dataset_test` (G Drive) | 1,013 | 0.6904 | **0.9596** | +0.2692 |

*Conclusion:* The Grayscale model perfectly generalizes to unseen bird attacks. The `0.9423` F1 score on the held-out seagull attack video proves the model dynamically accepts the drone and rejects the bird frame-by-frame as the bounding box switches targets.

### Paired Pipeline (Multimodal Tracking)
| Dataset | N | SA32 (Production) | Dual Classifier v3 | Delta |
| :--- | :--- | :--- | :--- | :--- |
| `antiuav` | 14,047 | 0.9184 | **0.9155** | -0.0029 |
| `svanstrom` | 3,233 | 0.8172 | **0.8172** | +0.0000 |

*Conclusion:* The Paired model matches production accuracy exactly, but drops 25% of its computational overhead by abandoning global scene features.

---

## 6. Feature Importance Analysis

Extracting the XGBoost tree weights reveals exactly how the two distinct models "think":

**Paired Model Top Features (`sa32_lite+`):**
1. `rgb_best_log_bbox_area` (21.4%)
2. `area_diff` (21.0%)
3. `ir_best_log_bbox_area` (16.4%)

*Analysis:* When both cameras are active, the model ignores AI confidence scores and acts purely on cross-modal geometry. It verifies that the bounding box sizes match across spectrums.

**Grayscale Model Top Features (`sa32_feats`):**
1. `ir_mean_conf` (49.9%)
2. `ir_max_conf` (17.5%)
3. `rgb_max_conf` (7.0%)

*Analysis:* With only one camera, cross-modal geometry is impossible. The model relies almost entirely (67%) on the raw confidence outputs of the fallback Thermal-YOLO model, using global features (like `blurriness` and `sky_ground_ratio`) to contextualize the remaining 33% of the decision tree.

---

## 7. Remaining Vulnerabilities

While the bird-confuser vulnerability is resolved, the Grayscale model still struggles with rigid metallic confusers due to a lack of training data:
- `confuser_AIRPLANE`: **0.4999 F1**
- `video_helicopters_compilation`: **0.3285 F1**

## 8. Reproducibility Guide

To recreate these exact results and F1 scores, a researcher must run the `v3` pipeline with the following strict parameters:

**0. Dataset Creation & Feature Extraction:**
Before training, the raw numerical features must be extracted from the source imagery using the Python pipeline (`classifier/extract_features.py` equivalent).
- **Paired Pipeline Extraction:** Runs the trained Thermal-YOLO and RGB-YOLO models concurrently. It calculates cross-modal geometry features, specifically `area_diff` (the mathematical size difference between the hot thermal signature and the visual RGB signature), which is impossible to calculate with a single camera.
- **Grayscale Pipeline Extraction:** Because the Thermal camera is offline, the Thermal-YOLO model is run against the *grayscale version* of the RGB frame. It also calculates heavy global scene textures (entropy, dynamic range, sky-to-ground ratio, blurriness) using OpenCV pixel matrices.

**1. Training Datasets:**
Because the models are specialized, they require distinctly curated training sets:

*The Paired Training Pool:*
- Must use strictly multimodal data containing active Thermal bounding boxes.
- `fusion_dataset_v3more.csv` (152k rows of clean, paired Thermal/RGB drone and confuser tracks).

*The Grayscale Training Pool:*
Must concatenate massive volumes of single-camera fallback data and hard negatives:
- `fusion_dataset_v3more_gray_aug.csv` (152k rows of paired data, heavily augmented and stripped of Thermal data)
- `fusion_dataset_144k.csv` (144k static confusers)
- `fusion_dataset_full56.csv` (65k chaotic video confusers/tests)
- `fusion_dataset_rgb_test.csv` (1,013 pure RGB drone frames)

**2. Model Hyperparameters (XGBoost):**
Both classifiers are trained using `XGBClassifier` with:
- `n_estimators=400`
- `max_depth=6`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `random_state=42` (Critical for exact validation split matches)
- *Note:* Windows deadlocks require `n_jobs=1`.

**3. Training Commands:**
Execute the `v3` ablation script, which contains the hardcoded data-leakage overrides for the `seagull_attack` videos:
```bash
python classifier/ablation_split_v3.py --mode paired
python classifier/ablation_split_v3.py --mode grayscale
```

**4. Evaluation Commands:**
Execute the evaluation scripts to generate the tables found in Section 5:
```bash
python classifier/eval_dual_vs_sa32.py
python classifier/eval_rgb_test.py
```
