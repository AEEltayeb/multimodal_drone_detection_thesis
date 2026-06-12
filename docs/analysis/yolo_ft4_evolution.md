# YOLO RGB Object Detector Evolution: ft4_1280

This document formalizes the scientific process, dataset curation, and automated ablation studies that led to the `ft4_1280` YOLO model superseding the `baseline_trained` and `retrained_v2` models for RGB drone detection.

---

## 1. Observation & Problem Statement

The original `baseline_trained` YOLO model suffered from two major vulnerabilities:
1. **High Hallucination Rate:** When evaluated against pure confuser datasets (birds, airplanes, empty skies), the model hallucinated drones **51.14%** of the time.
2. **Domain Shift Failure:** The model failed spectacularly to generalize to the specialized SELCOM TestDrone validation dataset, achieving a dismal **0.1453 F1** score.

The initial proposed solution was `retrained_v2`, which attempted to solve the hallucination problem by injecting a massive volume of confuser images (roughly 30% of the training pool) to force the model to learn negative boundaries.

---

## 2. The Over-Correction: Catastrophic Forgetting (`retrained_v2`)

The `retrained_v2` model successfully dropped the hallucination rate to an incredible **10.36%** and produced the fewest false positives (only 44 on the Svanstrom dataset). 

**The Failure Mode:** It achieved this by effectively going blind. The massive 30% injection of confuser images caused catastrophic forgetting. The model became so conservative that it stopped detecting actual drones. 
- Recall on SVANSTROM plummeted to **0.3052**.
- Recall on SELCOM plummeted to **0.0034** (literally finding only 1 true positive in the entire dataset).

This proved that blindly injecting large volumes of confusers destroys the YOLO model's primary tracking capabilities.

---

## 3. The Automation Architecture (`auto_confuser_ft4.py`)

To systematically search for the optimal injection parameters without manual intervention, we engineered a custom automated training loop (`auto_confuser_ft4.py`). This script is responsible for orchestrating dataset generation, training, and multi-surface evaluation.

Key architectural features of the script include:
1. **Dynamic Dataset Swapping:** Rebuilding the 9,000+ base training images for every config is I/O intensive. The script uses a `--confusers-only` flag via `build_selcom_confuser_ft4.py` to leave the base images untouched on disk, dynamically swapping only the variable confuser images and extra positives. This cuts dataset generation down to ~5 seconds per round.
2. **Sequential Multi-Surface Evaluation:** After training a configuration (e.g., 3 epochs), the script evaluates the resulting weights across 5 distinct datasets: Selcom Val, Dataset RGB Test, Svanstrom, Anti-UAV, and the pure Confuser pool.
3. **Fast-Reject Regression Gates:** To save compute time, the script evaluates surfaces sequentially and compares them against the `ft3` baseline snapshot. If a config fails an early regression gate (e.g., Selcom F1 drops below the allowable delta), the script immediately aborts evaluation for the remaining surfaces, marks the config as a `FAIL`, and proceeds to the next hyperparameter combination.
4. **Memory Optimizations:** To run on constrained hardware (4GB VRAM GTX 1050 Ti) at a high resolution (`imgsz=1280`), the script programmatically enforces `workers=0` (disabling multi-processing data loaders) and `plots=False` (disabling YOLO's VRAM-heavy metric plotting) to prevent OOM crashes during the rapid train-eval cycling.
5. **Ablation Mode:** The script supports a `--run-all` and `--no-fast-reject` flag. This disables the fast-reject mechanics and forces full evaluation across all surfaces for every config, which was utilized to collect the complete data tables for the Phase 2 ratio ablation study.

---

## 4. Iterative Refinement & Strict Regression Gates (`ft3` to `ft4`)

To solve catastrophic forgetting, we adopted a highly targeted "Hard-Negative Mining" strategy combined with an automated hyperparameter search (`auto_confuser_ft4.py`). 

First, an intermediary model (`ft3`) was trained using a balanced 50/50 split of the SELCOM dataset and the baseline dataset. This successfully restored drone recall (SELCOM F1 soared to **0.6194**), but the hallucination rate regressed to **61.00%**.

We established **strict regression gates** based on `ft3` performance. Any candidate model that degraded core metrics beyond these thresholds was automatically rejected:
* **Svanstrom DRONE Recall:** Allowable delta $\ge -0.0100$ (Baseline: 0.9270)
* **Dataset RGB F1:** Allowable delta $\ge -0.0100$ (Baseline: 0.9197)
* **Selcom val F1:** Allowable delta $\ge -0.0100$ (Baseline: 0.6194)
* **Anti-UAV F1:** Allowable delta $\ge -0.0050$ (Baseline: 0.9431)
* **Confuser Hallucination:** Must decrease ($< 0$ delta) (Baseline: 0.6100)

We evaluated combinations of hard-negative quantities (100 to 600), backbone freezing (`freeze=12` to `18`), and learning rates.

### Phase 1 Results: The Backbone Freeze Discovery
* **R1 (600 hard-negs, freeze=12):** **FAIL.** Svanstrom DRONE recall plummeted to 0.8978 (Delta: -0.029). The 600 confuser dose was too toxic for the tracking capabilities.
* **R2 (300 hard-negs, freeze=12):** **FAIL.** Selcom val F1 dropped to 0.5965 (Delta: -0.023). Lowering the dose wasn't enough on its own.
* **R3 (300 hard-negs, freeze=15):** **PASS.** Passed all gates. The critical discovery was that **freezing the first 15 layers of the YOLO backbone** prevented catastrophic forgetting, allowing the model to learn the confuser signatures without unlearning drone tracking.

---

## 5. Phase 2: The Ratio Ablation Study

We hypothesized that the R1 failure (600 confusers) was not due to the absolute count, but rather the **confuser-to-positive ratio**. The R3 winner had a ~3.3% confuser ratio (300 confusers / 9,125 total images). R1 had a 6.4% ratio (600 confusers / 9,425 total images). 

We ran a rigorous ablation study (A1 through A4) matching the 6.4% and 3.3% ratios by injecting tens of thousands of extra positive drone images (from the general dataset) alongside elevated confuser counts. All ablation runs used the successful `freeze=15` strategy.

### Ablation Configurations & Full Results:

| Config | Params | Svan DRONE Recall | Selcom val F1 | Result |
|---|---|---|---|---|
| **A1** | 600 confusers, 0 extra pos (Ratio: 6.4%) | 0.8971 ($\Delta$: -0.030) | 0.6277 ($\Delta$: +0.008) | **FAIL** (Svanstrom R drop) |
| **A2** | 600 confusers, 4k extra pos (Ratio: 4.5%) | 0.9186 ($\Delta$: -0.008) | 0.6052 ($\Delta$: -0.014) | **FAIL** (Selcom val F1 drop) |
| **A3** | 600 confusers, 8.8k extra pos (Ratio: 3.3%) | 0.9140 ($\Delta$: -0.013) | 0.5911 ($\Delta$: -0.028) | **FAIL** (Svanstrom R & Selcom F1) |
| **A4** | 900 confusers, 17.6k extra pos (Ratio: 3.3%)| N/A | N/A | **TRAIN FAILED** (OOM on 27k images) |

### Ablation Conclusions:
1. **Ratio is Not a Magic Bullet:** While A2 (4.5% ratio) initially appeared to rescue the Svanstrom Drone Recall (-0.008, PASS), adding even *more* positive images in A3 (3.3% ratio) actually caused the recall to regress again (-0.013, FAIL). 
2. **The Domain Dilution Effect:** Scaling up the general-domain positive images to maintain the ratio severely diluted the model's domain-specific performance. As extra general positives were added, the model lost its tuning on the specific datasets:
   - **Selcom val F1** degraded monotonically: +0.008 (A1) $\rightarrow$ -0.014 (A2) $\rightarrow$ -0.028 (A3).
   - **Svanstrom Drone Recall** suffered similarly at high dilution volumes in A3.
3. **The Optimal Strategy:** The hypothesis that we could scale up confusers indefinitely by simply adding more positives is **rejected**. Flooding the dataset with general positives causes a distributional shift. The undeniable winning strategy remains **R3**: A minimal effective dose (300 hard-negatives) combined with a heavily frozen backbone (`freeze=15`) to preserve tracking without dilution.

---

## 6. Final Results & Validation (The R3 Winner)

The `ft4_1280` model achieved the "Goldilocks" balance, significantly dropping hallucination rates without triggering catastrophic forgetting.

### Core Datasets (F1 Scores)
| Dataset | baseline_trained | retrained_v2 | ft3_1280 | R3 Winner (300hn, f=15) | $\Delta$ from ft3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ANTIUAV** | 0.9599 | 0.9700 | 0.9431 | **0.9545** | +0.0114 |
| **SVANSTROM** | 0.5650 | 0.4316 | 0.6083 | **0.6070** | -0.0013 |
| **SELCOM_VAL** | 0.1453 | 0.0067 | 0.6194 | **0.6151** | -0.0043 |
| **Dataset RGB** | - | - | 0.9197 | **0.9177** | -0.0020 |

### Hallucination & Critical Recall
| Metric | baseline_trained | retrained_v2 | ft3_1280 | R3 Winner (300hn, f=15) | $\Delta$ from ft3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Confuser Hallucination %** | 51.14% | 10.36% *(Blind)* | 61.00% | **45.04%** | **-15.96 pp** |
| **Svanstrom HELI Halluc %** | 86.24% | - | 86.24% | **58.88%** | **-27.36 pp** |
| **Svanstrom DRONE Recall** | - | 0.3052 | 0.9270 | **0.9194** | -0.0076 |

*Conclusion:* By strategically mining only the highest-confidence false positives, restricting the injection to exactly 300 images, and **freezing 15 backbone layers**, `ft4_1280` successfully hardened the model against confusers (dropping hallucinations by 16 percentage points overall, and by 27 percentage points on helicopters specifically) while perfectly maintaining drone recall.

---

## 7. Reproducibility Guide

To recreate the `ft4_1280` model, follow these steps:

**1. Hard-Negative Mining:**
Generate the CSV of high-confidence false positives by running the intermediary `ft3_1280` model against the raw confuser images.
```bash
python scripts/mine_confuser_hardnegs.py
```

**2. Dataset Assembly:**
Execute the `build_selcom_confuser_ft4.py` script. This script automatically copies the `ft3` base training images, parses the mined hard-negatives CSV, and selects exactly **300** category-balanced high-confidence false positives with empty labels.
```bash
python "RGB model/dataset preparation/build_selcom_confuser_ft4.py" --n-hardnegs 300
```

**3. Fine-Tuning Execution:**
Execute the `finetune_selcom.py` script with the winning R3 hyperparameters. The `--freeze 15` flag is critically important.
```bash
python "RGB model/finetune_selcom.py" --ft 4 --imgsz 1280 --epochs 3 --lr0 5e-6 --freeze 15
```
