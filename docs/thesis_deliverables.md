# Thesis Deliverables Tracker

Living document. Update as chapters are written. Every claim must trace to `EVIDENCE_LEDGER.md`.

---

## Status Legend
- `[ ]` Not started
- `[/]` In progress
- `[x]` Done
- `[!]` Blocked / needs data

---

## 1. Chapters Status & Writing Outline

### Ch 1 — Introduction `[x]`
- `[x]` Structure in LaTeX
- `[x]` Write background (drone proliferation, security threat landscape)
- `[x]` Write problem statement (RGB hallucinates 94% on birds — cite CSV)
- `[x]` Write research objectives (5 objectives listed)
- `[x]` Write contributions list (6 contributions listed)
- `[x]` Write thesis outline (1 paragraph per chapter)

### Ch 2 — Literature Review `[ ]`
- `[x]` Section structure in LaTeX (7 sections)
- `[ ]` Write §2.1 Object Detection Architectures (YOLO family, single-stage vs two-stage)
- `[ ]` Write §2.2 Drone Detection in Aerial Surveillance (survey of radar/acoustic/RF/visual)
- `[ ]` Write §2.3 Thermal IR Imaging for Detection
- `[ ]` Write §2.4 Multi-Modal Sensor Fusion (early/late/decision-level)
- `[ ]` Write §2.5 Hard-Negative Mining
- `[ ]` Write §2.6 Confidence Calibration and Scoring Rules
- `[ ]` Write §2.7 Human-in-the-Loop ML
- `[ ]` Find and add all citations (see §4 References below)

### Ch 3 — System Architecture `[x]`
- `[x]` Section structure in LaTeX
- `[x]` Write §3.1 Architecture Overview (enumerated 5 components + S1→S3 numbers)
- `[x]` Write §3.2 Design Rationale (fail-open, baseline vs retrained_v2 argument)
- `[x]` Write §3.3 Trust-Aware Fusion (4-class formulation, ir_detected=49%)
- `[x]` Write §3.4 Alert-Gate Cascade (alert-gate-only vs filter-then-classify)
- `[x]` Write §3.5 Temporal Smoothing (5-of-6 window)
- `[x]` Write §3.6 Resolution Dependency (640×512 native, 13.3× improvement)
- `[ ]` Create Fig 3.1 (pipeline block diagram — still placeholder)
- `[ ]` Create Fig 3.2 (trust flow diagram — still placeholder)

### Ch 4 — Component Design & Training `[/]`
- `[x]` Section structure + data tables in LaTeX
- `[x]` Write §4.1 RGB Detector (architecture, training data=137k, 3 iterations ablation)
- `[x]` Write §4.2 IR Detector (architecture, finetuning, complementary strengths)
- `[x]` Write §4.3 Trust Classifier (40 features, XGBoost, 3 variants, OOD analysis)
- `[x]` Write §4.4 Confuser Patch Verifier (4-class CNN, 4 versions, v2 selected)
- `[x]` Write §4.5 Label Reviewer (tool design, motivation, keyboard workflow)
- `[ ]` Write §4.6 HITL IR Model Development (iterative loop, version history table ✅)
- `[ ]` Write §4.7 Resolution Sensitivity & Svanström Effect (table ✅)
- `[ ]` Create Fig 4.1 (RGB training curves)
- `[ ]` Create Fig 4.2 (IR training curves)
- `[ ]` Create Fig 4.3 (RGB vs IR night example)
- `[ ]` Take Fig 4.4 screenshot (label reviewer GUI)
- `[ ]` Create Fig 4.5 (HITL iterative loop diagram)

### Ch 5 — Out-of-Distribution Analysis `[ ]`
- `[x]` Section structure + grayscale table in LaTeX
- `[ ]` Write §5.1 Overview (3 OOD axes)
- `[ ]` Write §5.2 RGB OOD (confuser halluc, cross-domain Anti-UAV vs Svanström)
- `[ ]` Write §5.3 IR OOD (grayscale, cross-domain Svanström)
- `[ ]` Write §5.4 Implications for Fusion Architecture
- `[!]` Run IR drone detection on grayscale @ imgsz=1280 for Tab 5.3
- `[ ]` Generate Fig 5.1 (side-by-side confuser example)

### Ch 6 — Experimental Setup `[ ]`
- `[x]` Section structure in LaTeX
- `[ ]` Write §6.1 Datasets (table ✅, add conversion provenance)
- `[ ]` Write §6.2 Evaluation Metrics (IoP vs IoU)
- `[ ]` Write §6.3 Scoring Rule Audit (dual vs trust_aware)
- `[ ]` Write §6.4 Hardware and Software
- `[ ]` Write §6.5 Reproducibility Infrastructure

### Ch 7 — Results & Ablation `[ ]`
- `[x]` Section structure + all tables/figure placeholders in LaTeX
- `[ ]` Write §7.1 Cumulative Confuser Suppression + create Fig 6.1 (bar chart)
- `[ ]` Write §7.2 RGB Model Comparison + create Fig 6.4 (scatter)
- `[ ]` Write §7.3 Resolution Sensitivity + create Fig 6.6
- `[ ]` Write §7.4 Classifier Comparison (table ✅)
- `[ ]` Write §7.5 Patch Verifier Audit (table ✅)
- `[ ]` Write §7.6 Patch Threshold Sweep + create Fig 6.3
- `[ ]` Write §7.7 Scoring Rule Sensitivity + create Fig 6.7

### Ch 8 — Discussion `[ ]`
- `[x]` Section structure in LaTeX
- `[ ]` Write §8.1 Architecture vs Retraining (central thesis argument)
- `[ ]` Write §8.2 Open-World Deployment Tradeoff
- `[ ]` Write §8.3 Limitations
- `[ ]` Write §8.4 Threats to Validity
- `[ ]` Write §8.5 Future Work

### Ch 9 — Conclusion `[ ]`
- `[ ]` Write summary (1-2 pages, reference key numbers)

---

## 2. Dataset Conversion Provenance

Both primary datasets were originally video-based and required conversion to YOLO format.

### Svanström
- **Original**: `G:/drone/Drone-detection-dataset-must-cite/Drone-detection-dataset-master/Data/`
  - `Video_IR/` — IR .mp4 files
  - `Video_V/` — Visible .mp4 files
  - `*_LABELS.mat` — MATLAB ground truth (per-frame bboxes)
- **Conversion script**: `classifier/convert_svanstrom_paired.py`
  - Parses MATLAB `.mat` files (binary `__function_workspace__` extraction)
  - Extracts paired IR+Visible frames every 3rd frame (default `--sample-every 3`)
  - Converts `(x,y,w,h)` top-left absolute → YOLO `(cx,cy,w,h)` normalized
  - Outputs to `G:/drone/svanstrom_paired/{IR,RGB}/{images,labels}/`
  - Categories: DRONE (labeled), AIRPLANE/BIRD/HELICOPTER (no drone labels = confuser frames)
  - Native resolution: **640×512**
- **Output**: 28,710 paired frames (from `meta.json`)
- **Status**: `[x]` Conversion script exists and is reproducible

### Anti-UAV RGBT
- **Original**: `G:/drone/Anti-UAV-RGBT/` — contains .mp4 video files + JSON annotations
  - Also: `G:/drone/3rd_Anti-UAV_train_val/` (competition data)
- **Frame extraction**: `G:/drone/Anti-UAV-RGBT/framecut.py` (simple cv2 video→jpg)
- **YOLO conversion**: Two scripts found:
  - `scripts/dataset_preparation/04a_convert_antiuav_to_yolo.py` (current)
  - `RGB model/dataset preparation/anti_uav_rgbt_rgb_to_yolo.py` (legacy RGB-only)
  - `archive/scripts/dataset_old/convert_antiuav_to_yolo.py` (archived)
- **Output**: `G:/drone/Anti-UAV-RGBT_yolo_converted/{test,val}/` + `G:/drone/CST-AntiUAV_YOLO/`
- **Status**: `[x]` Conversion pipeline exists, multiple versions tracked

### ⚠️ Svanström Usage Audit (train/eval overlap)

| Component | In Training? | In Eval? | Clean? |
|---|---|---|---|
| **RGB detector** | ❌ No (trained on `G:/drone/dataset/dataset`, 137k imgs) | ✅ 28,710 | ✅ True OOD |
| **IR detector** | ⚠️ Yes — 18,639 `czoom_svan_*` crops in `IR_dset_final/train` | ✅ 28,710 | ⚠️ Target overlap |
| **Trust classifier** | ⚠️ Yes — all 28,710 rows in `fusion_dataset.csv` (75/25 seq-split) | ✅ 28,710 | ⚠️ Train seqs in eval |
| **Patch verifier** | ❌ No (trained on confuser crops only) | ✅ Applied | ✅ Clean |

**IR detail**: `IR_dset_final` contains 22,962 Svanström-sourced frames across splits:
- train: 18,639 (DRONE: 13,361 | AIRPLANE: 1,919 | BIRD: 2,008 | HELICOPTER: 1,351)
- val: 2,050 | test: 2,273

**Trust classifier detail**: `train_fusion.py` uses `GroupShuffleSplit(75/25)` with sequence-stratified splitting. The classifier's own test set saw ~6,254 Svanström frames. But `cumulative_halluc.py` evaluates on ALL 28,710 — including ~22k training sequences.

**Mitigation**: Anti-UAV results (acc=0.991) are fully clean. Confuser zoo is clean. Svanström numbers should be interpreted as in-distribution performance. Both are documented in thesis §8 Threats to Validity.

**No train/val/test split exists for `svanstrom_paired/`** — it's a single unsplit pool of 28,710 paired frames used as-is for evaluation.

---

## 3. Figures Needed

### Architecture Figures
- `[ ]` **Fig 3.1** — Full pipeline block diagram
- `[ ]` **Fig 3.2** — Trust classifier decision flow

### Training & Data Figures
- `[ ]` **Fig 4.1** — RGB model training curves (`RGB model/Yolo26n_trained/results.csv`)
- `[ ]` **Fig 4.2** — IR model training curves (`runs/corrective_finetune/finetune_v3b/results.csv`)
- `[ ]` **Fig 4.3** — RGB vs IR night example (Anti-UAV night sequences)
- `[ ]` **Fig 4.4** — Label reviewer GUI screenshot
- `[x]` **Fig 4.IR** — IR model evolution chart → `docs/figures/fig4_ir_evolution.pdf`
- `[ ]` **Fig 4.5** — HITL iterative model-data curation loop diagram
- `[ ]` **Fig 4.6** — Patch verifier 4-class architecture
- `[ ]` **Fig 4.7** — Trust classifier feature importance bar chart (exists: `fusion_no_fn_feature_importance.png`)

### Data Tables (in LaTeX)
- `[x]` **Tab 4.1** — RGB 3-way comparison (baseline/hardneg/retrained)
- `[x]` **Tab 4.2** — IR model evolution (seed→v3→v4→v5→final→v3b with mAP50)
- `[x]` **Tab 5.1** — IR on grayscale confuser halluc rates
- `[x]` **Tab 5.2** — Resolution sensitivity (640 vs 1280)
- `[!]` **Tab 5.3** — IR drone detection on grayscale (needs eval run)

### OOD Figures
- `[ ]` **Fig 5.1** — Side-by-side confuser: RGB fires, IR-grayscale rejects
- `[ ]` **Fig 5.2** — Cross-eval matrix (IR models across datasets)

### Results Figures
- `[x]` **Fig 6.1** — Cumulative confuser suppression bar chart → `docs/figures/fig6_1_cumulative_confuser.pdf`
- `[x]` **Fig 6.2** — Svanström by-category × by-stage → `docs/figures/fig6_2_svanstrom_by_category.pdf`
- `[x]` **Fig 6.3** — Patch threshold sweep → `docs/figures/fig6_3_threshold_sweep.pdf`
- `[ ]` **Fig 6.4** — RGB 3-way scatter (recall vs halluc) — `svanstrom_1280_by_category.csv`
- `[x]` **Fig 6.6** — Resolution sensitivity plot → `docs/figures/fig6_6_resolution.pdf`
- `[ ]` **Fig 6.7** — Scoring rule impact (28pp swing) — `E_scoring/` ablation
- `[ ]` **Fig 6.9** — Example confuser detections with bboxes

### Discussion Figures
- `[x]` **Fig 7.1** — OOD classifier comparison → `docs/figures/fig7_1_ood_classifier.pdf`
- `[ ]` **Fig 7.2** — Failure mode gallery

---

## 4. Key Claims & Evidence Mapping

| Claim | Evidence Source | Status |
|---|---|---|
| Multi-stage pipeline suppresses 98.4% of confuser FPs | `confuser_fusion_no_fn_model_v1.1/summary.json` | ✅ |
| Baseline RGB is the best drone detector (R=0.959) | `svanstrom_1280_by_category.csv` | ✅ |
| retrained_v2 collapses drone recall to 0.306 | Same CSV | ✅ |
| imgsz=1280 is required (0.07→0.959) | EVIDENCE_LEDGER §3.1 | ✅ |
| Svanström native resolution is 640×512 | `cv2.imread` check | ✅ |
| Scoring rule causes 28pp F1 swing | `E_scoring/` ablation | ✅ |
| Patch verifier v2 > v4 on every metric | `_patch_catch_audit/baseline_v2/summary.json` | ✅ |
| IR hallucinates less than RGB on confusers (22% vs 53%) | `confuser_test_hallucination.csv` | ✅ |
| Trust classifier top feature is `ir_detected` (49%) | `fusion_no_fn_metrics.json` | ✅ |
| RGB struggles with birds (94% halluc rate) | `svanstrom_1280_by_category.csv` | ✅ |
| IR on grayscale hallucinates 2.4× less than RGB | `confuser_test_hallucination.csv` | ✅ |
| IR model improved v3→v4 (0.900→0.955) via human review | `runs/IR_FT_*/results.csv` | ✅ |
| IR model generalises to grayscale (never trained on it) | `confuser_test_hallucination.csv` | ✅ |
| Svanström/Anti-UAV converted from video with provenance | `convert_svanstrom_paired.py`, `04a_convert_antiuav_to_yolo.py` | ✅ |
| IR drone detection on grayscale works well | needs eval run | `[!]` TODO |
| Temporal gate suppresses isolated firings | GUI demo runs | `[!]` needs quantification |
| Human-in-the-loop label review improved data quality | label reviewer logs | `[!]` needs documentation |
| Latency is acceptable for real-time deployment | EVIDENCE_LEDGER §8 | `[!]` all placeholders |

---

## 5. References To Find

### Core (must cite)
- `[ ]` YOLOv8/YOLOv11 architecture paper (Ultralytics)
- `[ ]` XGBoost paper (Chen & Guestrin, 2016)
- `[ ]` Anti-UAV benchmark dataset paper
- `[ ]` Svanström et al. drone detection dataset paper (original `must-cite` dataset)
- `[ ]` Transfer learning / fine-tuning survey for object detection

### Methodological
- `[ ]` Multi-modal sensor fusion for UAV detection (survey)
- `[ ]` Hard-negative mining in object detection (Shrivastava et al., 2016 or similar)
- `[ ]` Confidence calibration in neural networks (Guo et al., 2017)
- `[ ]` IoU vs IoP scoring metrics comparison
- `[ ]` Cascade classifiers / multi-stage detection pipelines

### Domain
- `[ ]` Counter-UAS systems survey (commercial/military)
- `[ ]` Thermal/IR imaging for drone detection
- `[ ]` Bird-drone discrimination in aerial surveillance
- `[ ]` Regulatory framework for drone detection (South Africa / international)

### Technical
- `[ ]` PySide6 / Qt for real-time GUI applications
- `[ ]` Feature importance interpretation in gradient boosting
- `[ ]` Ablation study methodology best practices
- `[ ]` Data-centric AI (Andrew Ng, 2021 — "Data-Centric AI Competition")
- `[ ]` Active learning / model-assisted labeling surveys
- `[ ]` Cross-modality transfer (thermal→grayscale domain gap literature)

---

## 6. Writing Process Rules

- **Every number** in the thesis must appear in `EVIDENCE_LEDGER.md` with source file and reproduction command
- **Every figure** must have a data source listed in §3 above
- **Frame the 3 RGB retrains** as intentional recall-precision tradeoff ablation, not trial-and-error
- **Lead with architecture**, not individual model performance
- **Disclose** the dual vs trust_aware scoring delta prominently (§6.3)
- **The label reviewer** is a novel contribution — document the workflow, not just the tool
- **The HITL IR development** is a novel contribution — the iterative model-assisted curation loop
- **Dataset conversion** must be documented: both Svanström and Anti-UAV were video→YOLO conversions
- **Svanström resolution (640×512)** explains the imgsz=1280 requirement — this is not a model defect
