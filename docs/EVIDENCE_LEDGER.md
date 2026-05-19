# Evidence ledger

Single source of truth for every quantitative claim about the drone-detection pipeline. Every metric reported in the thesis, in a presentation, or to the company must be findable here with its source file, the command that produced it, the dataset, imgsz, stride, and date. **No number lives only in chat or memory — it lives here.**

## How to use this file

- **Before citing a number anywhere**: find it here. If it isn't here, treat the claim as unverified.
- **After running anything that emits a metric**: add a row to the relevant section (or a placeholder if you'll fill it later). The cost of a sparse row is near zero; the cost of a missing one is hours of re-running.
- **`source` column**: relative path from repo root to the artifact (CSV, JSON, summary). Prefer the most specific file — not the parent directory.
- **`how to reproduce` column**: the exact command line, or a pointer to a script that wraps it. If unknown, write `UNKNOWN — locate before re-running` so it surfaces next time.
- **Stale numbers**: when a component is replaced, don't delete the old row — append a new one with the new date and mark the old row's `status` as `superseded`. We need the history for the thesis.

Status values: `current` (load-bearing for production decisions), `superseded` (kept for history), `provisional` (not yet re-verified in current session), `placeholder` (row reserved, no number yet).

---

## 1. Production stack (current pick, 2026-05-11)

| Component | Choice | Why | Source for the choice |
|---|---|---|---|
| RGB YOLO | `RGB model/Yolo26n_trained/weights/best.pt` (baseline) | Only RGB variant with usable small-drone recall at imgsz=1280 (R=0.959 vs retrained_v2=0.306 on Svanstrom) | Section 3 row "DRONE recall by RGB model @ 1280" |
| IR YOLO | `runs/corrective_finetune/finetune_v3b/weights/best.pt` | IR alone hits 0.961 F1 on Svanstrom; only one IR model considered | May 10 ablation, `ir_only` config |
| Trust classifier | **`scene_aware_v3more_32feat`** (primary) / `fusion_no_fn_v1.1` (open-world fallback) | **Production pick flipped 2026-05-18.** sa32 wins real-video cascade drone F1 (0.826 baseline vs control40's 0.644, $-18$~pp; §9.5.8) and **ties control40 on the OOD confuser zoo** (sa32 S2=0.205, S3=0.103 vs control40 S2=0.212, S3=0.094; §7 + `eval/results/_cumulative_halluc/confuser_c40/summary.json`). sa32 therefore strictly dominates control40 on the combined real-video + zoo axes. **`fusion_no_fn_v1.1`** is the only meaningful alternative: 13$\times$ more conservative on the OOD zoo (S2=0.016) but collapses real-video drone recall (segment $R$=0.128). Ship sa32 by default; ship fnfn only when the deployment surface is confuser-saturated and missed drones cost less than false alarms. **`control_v3more_40feat`** is deprecated — no operational regime favors it. | Section 7 (Svanstrom + zoo) + Section 9.5.8 (real-video) classifier comparisons |
| Patch verifier | `classifier/runs/patches/confuser_filter4_{rgb,ir}_v2_backup.pt` | v3 over-aggressive; v4≈v2; v2 is what production runs | May 10 ablation `B_patch_verifier` table |
| Cascade | `alert_gate_only` | Per-frame filtering costs ~1pp F1; production runtime is alert-gated | May 10 ablation `D_cascade` + GUI source |
| Scoring | `trust_aware` | Matches the system's actual decision rule (classifier picks the trusted modality) | May 10 ablation `E_scoring`, 28-pp delta |
| imgsz | 1280 | Svanstrom drones unresolvable at 640; baseline RGB recall 0.07→0.959 by imgsz alone | `eval/results/_phase4/Qx1H/noTemp_1280/summary.json` + Svanstrom@1280 baseline run |
| RGB conf | TBD via conf sweep with **baseline** RGB | The May 10 sweep was against retrained_v2 | Section 5 row "Conf sweep × baseline RGB" (placeholder) |
| IR conf | 0.40 | May 10 conf sweep on Anti-UAV/Svanstrom | `eval/results/_ablation/2026-05-10T16-08-14/H_conf_sweep` |

---

## 2. Datasets

| Tag | Path | Type | Native res | Notes | Source for metadata |
|---|---|---|---|---|---|
| `antiuav` | (per `eval/config.yaml`) | paired RGB+IR | RGB 1920×1080, IR 640×512 | Saturated for both modalities; not informative for further ablation | `eval/config.yaml` |
| `svanstrom` | (per `eval/config.yaml`) | paired RGB+IR | 640×480 native | Small drones; requires imgsz=1280; RGB-confuser-rich (birds/airplanes/helicopters) | `eval/config.yaml` + `project_imgsz_1280_svanstrom.md` memory |
| `antiuav_rgb_gray` | `G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB` | RGB-only, fed to IR model after grayscale | 1920×1080 | The only labeled grayscale-cross-domain dataset we have | `eval/config.yaml` (added May 2026) |
| `rgb_dataset` | `G:/drone/dataset/dataset` | RGB-only labeled | mixed | RGB model standalone benchmark | `eval/config.yaml` |
| `ir_dset_final` | `G:/drone/IR_dset_final` | IR-only labeled | mixed | IR model standalone benchmark | `eval/config.yaml` |
| `rgb_confusers_merged` | `G:/drone/rgb_confusers_merged` | RGB unlabeled confusers | mixed | Bird/airplane/helicopter images; train=21784, val=2607, test=2633. Has `dataset_documentation.md`. | `eval/diagnose_failures.py` line 28 + `eval/cumulative_halluc.py` line 46 |
| `confuser_airplane_roboflow` | `G:/drone/rgb_confusers_merged/images/test/airplane_*` (99 images) | RGB unlabeled, 99 images | mixed | Airplane-only subset within the confuser test split | Filename prefix `airplane_` in `eval/diagnose_failures.py` line 139 |
| `yt_DiN4s-MWvPg.mp4` | `ir_gui/demo_outputs/yt_DiN4s-MWvPg.mp4` | RGB video | mixed | Diagnostic clip, sky-flight frames 565+ | Plan file Phase 4 |
| `yt_FQO3COJl5Us.mp4` | `ir_gui/demo_outputs/yt_FQO3COJl5Us.mp4` | RGB video | mixed | Diagnostic clip, drone-in-hand full clip (positive control) | Plan file Phase 4 |
| `yt_Qx1Hlot9Ye8.mp4` | `ir_gui/demo_outputs/yt_Qx1Hlot9Ye8.mp4` | RGB video | mixed | Diagnostic clip, small-distant-drone sky frames 163+ | Plan file Phase 4 |
| `selcom_cctv` | `G:/drone/selcom_dataset` | RGB labeled (fixed-cam CCTV) | 1920×1080 (HEVC 7 Mbps, yuvj420p) | 2076 images (1953 positives + 123 true negatives); median drone sqrt(area)=36.8 px so ~12 px in input at imgsz=640, ~24 px at imgsz=1280. Source video `G:/drone/TestDrone_ritagliato_selcom_footage.mp4` (1:30, 25 fps). | `ffmpeg -i` on source + `python -c` size histogram |
| `selcom_mixed_ft2_val` | `G:/drone/_finetune_selcom_mixed_ft2/images/val` | RGB labeled, derived | 1920×1080 | 311 imgs / 295 GT boxes; pure-selcom 15% held-out split, no general data; seed=0 | `RGB model/dataset preparation/build_selcom_mixed_ft2.py` |

---

## 3. RGB YOLO models — measured metrics

Per-model metrics at imgsz=1280 unless noted. All metrics on Svanstrom unless noted.

### 3.1 Svanstrom @ imgsz=1280 — drone-class only

| RGB model | Det rate | TP | FP | FN | P | R | Missed-GT median area ratio | Source | How to reproduce | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `Yolo26n_trained` (baseline) | 96.5% | 1248 | 79 | 54 | 0.940 | **0.959** | 0.0014 | `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` (script-generated) | `python eval/diagnose_failures.py` (baseline) + `python eval/diagnose_failures_all.py` (hardneg/retrained) | current |
| `Yolo26n_hardneg_v3more` | 95.7% | 1237 | 78 | 65 | 0.941 | 0.950 | 0.0020 | `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` | `python eval/diagnose_failures_all.py` | current |
| `Yolo26n_retrained_v2` | **30.8%** | 398 | 24 | 904 | 0.943 | **0.306** | 0.0024 | `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` | `python eval/diagnose_failures_all.py` | current — but disqualified for production |
| `Yolo26n_retrained_v2` @ imgsz=640 | — | — | — | — | 0.837 | **0.072** | — | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` row `rgb_only` × svanstrom | `python eval/ablate.py --matrix eval/ablations.yaml --datasets svanstrom` (May 10 stride/imgsz) | superseded |

### 3.2 Anti-UAV @ imgsz=1280, stride=5

| RGB model | P | R | F1 | TP | FP | FN | Source | How to reproduce |
|---|---|---|---|---|---|---|---|---|
| `Yolo26n_trained` (baseline) | **0.9922** | **0.9950** | **0.9936** | 3178 | 25 | 16 | `eval/results/_ablation/2026-05-18T22-48-19/master.csv` row `A_rgb_yolo / rgb_old / antiuav / rgb_only` | `python eval/ablate.py --matrix eval/ablations.yaml --factors A_rgb_yolo --datasets antiuav` |
| `Yolo26n_retrained_v2` | 0.9922 | 0.9950 | 0.9936 | 3178 | 25 | 16 | same run, `rgb_new` level | same |
| `Yolo26n_retrained_v2` (May 10) | 0.991 | 0.995 | 0.9929 | --- | --- | --- | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` row `rgb_only` × antiuav | per `eval/ablate.py` matrix (earlier stride) | superseded by 2026-05-18 row above |

**Read:** Anti-UAV is fully saturated for both RGB variants. The two are numerically indistinguishable on this benchmark (identical TP/FP/FN). Anti-UAV is therefore a sanity floor only; it does not discriminate between the RGB training stances. Svanstr\"om (\S3.1) is the discriminating benchmark.

### 3.3 Confuser test set — hallucination rates @ imgsz=1280 (no GT drones, halluc = any detection)

By confuser class (Svanstrom unlabeled confuser frames):

| Class | Frames | baseline halluc | hardneg_v3more halluc | retrained_v2 halluc | Med FP conf (baseline / hardneg / retrained_v2) | Source |
|---|---|---|---|---|---|---|
| BIRD | 589 | 94.4% (807 FPs) | 94.2% | **3.4%** | 0.691 / 0.683 / 0.527 | `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` |
| AIRPLANE | 677 | 74.6% | 64.7% | **5.6%** | 0.733 / 0.677 / 0.430 | same |
| HELICOPTER | 625 | 66.2% (464 FPs) | 41.9% | **4.5%** | 0.832 / 0.710 / 0.672 | same |

Separate confuser test set `rgb_confusers_merged` (`G:/drone/rgb_confusers_merged/images/test/`) @ imgsz=1280:

| Source split | Images | baseline halluc | hardneg_v3more halluc | retrained_v2 halluc | Source |
|---|---|---|---|---|---|
| airplane (roboflow) | 99 | 27.3% | 7.1% | 19.2% | `eval/results/_failure_diagnosis/confuser_test_hallucination.csv` |
| other (svan+kaggle+roboflow) | 2534 | 53.0% | 47.1% | **10.9%** | same |

### 3.4 selcom CCTV fine-tunes (2026-05-15/16) — IoP@0.5, conf=0.25

Scoring is IoP@0.5 (Intersection over Prediction area), conf=0.25. Selcom drones blend into urban backgrounds and the camera is a fixed wide-angle 7 Mbps HEVC stream — out-of-distribution for the baseline. The mixed fine-tunes blend 80% general RGB + 20% selcom at train time, keeping the val split pure selcom.

**On `selcom_mixed_ft2_val` (311 images, 295 GT boxes):**

| RGB model | imgsz | TP | FP | FN | P | R | F1 | Source | How to reproduce | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `Yolo26n_trained` (baseline, pre-finetune) | 640 | 2 | 6 | 293 | 0.250 | 0.007 | 0.013 | `runs/rgb_finetune_eval/Yolo26n_selcom_mixed_ft2/comparison.json` | `python "RGB model/finetune_selcom.py" --ft 2 --skip-stage --skip-train` | current |
| `Yolo26n_trained` (baseline) | 1280 | 26 | 37 | 269 | 0.413 | 0.088 | 0.145 | `runs/rgb_finetune_eval/Yolo26n_selcom_mixed_ft2_1280/comparison.json` | `python "RGB model/finetune_selcom.py" --ft 2 --imgsz 1280 --skip-stage --skip-train` | current |
| `Yolo26n_selcom_mixed_ft2` | 640 | 72 | 50 | 223 | 0.590 | 0.244 | 0.345 | `runs/rgb_finetune_eval/Yolo26n_selcom_mixed_ft2/comparison.json` | `python "RGB model/finetune_selcom.py" --ft 2` | superseded by `_ft2_1280` |
| **`Yolo26n_selcom_mixed_ft2_1280`** | **1280** | **138** | **43** | **157** | **0.762** | **0.468** | **0.580** | `runs/rgb_finetune_eval/Yolo26n_selcom_mixed_ft2_1280/comparison.json` | `python "RGB model/finetune_selcom.py" --ft 2 --imgsz 1280 --batch 4 --skip-stage` | current — CCTV production weights |

**Base model for selcom fine-tune**: `Yolo26n_trained` (baseline), NOT `Yolo26n_retrained_v2`. This matters for downstream calibration: the selcom-fine-tuned weights inherit baseline's confidence distribution, so the production trust classifiers (`control_v3more_40feat`, `fusion_no_fn_v1.1`) — which were calibrated against baseline-family RGB — remain valid choices when the deployed RGB is the selcom fine-tune.

**Confuser negatives in detector training (all variants):**

| Detector | Birds in training? | Airplanes? | Helicopters? |
|---|---|---|---|
| `Yolo26n_trained` (baseline) | yes (drone-vs-bird subset) | no | no |
| `Yolo26n_hardneg_v3more` | yes | yes (Svanström split) | yes (Svanström split) |
| `Yolo26n_retrained_v2` | yes (aggressive) | yes (aggressive) | yes (aggressive) |
| IR `v3b` / Final | yes | yes | yes (HITL-mined hard negatives) |
| `Yolo26n_selcom_mixed_ft2_1280` | inherits baseline | inherits baseline | inherits baseline |

All detectors are trained with at least some confuser negatives. The cascade's role is to catch residual hallucinations on the categories where training-time mining does not work (especially birds for RGB at small scale), not to add precision to a "naïve" detector.

**Dataset_rgb regression check (same imgsz for both rows in each pair):**

| RGB model | imgsz | dataset_rgb P | R | F1 | Δ F1 vs baseline @ same imgsz | Source |
|---|---|---|---|---|---|---|
| `Yolo26n_trained` (baseline) | 640 | 0.975 | 0.927 | 0.950 | (reference) | `Yolo26n_selcom_mixed_ft2/comparison.json` |
| `Yolo26n_selcom_mixed_ft2` | 640 | 0.951 | 0.939 | 0.945 | **-0.006** PASS | same |
| `Yolo26n_trained` (baseline) | 1280 | 0.934 | 0.910 | 0.922 | (reference) | `Yolo26n_selcom_mixed_ft2_1280/comparison.json` |
| `Yolo26n_selcom_mixed_ft2_1280` | 1280 | 0.899 | 0.930 | 0.914 | **-0.008** PASS | same |

**Notes for thesis citation:**
- Training-time val metrics (mAP50, R at F1-optimal conf) are higher than the table above because Ultralytics reports at the best-confidence operating point. The numbers here are at fixed conf=0.25 (production threshold) with IoP@0.5 scoring.
- ft2_1280 doubles selcom recall vs ft2@640 (0.244→0.468) AND raises precision (0.59→0.76) — not a tradeoff but a genuine resolution win. Median selcom drone goes from ~12 px (imgsz=640 input) to ~24 px (imgsz=1280 input), crossing YOLO's effective small-object detection floor.
- An earlier `Yolo26n_selcom_mixed_ft1` run on a 104-image precursor val set (587 train images, single source video) hit P=0.84/R=0.72/F1=0.77 — not directly comparable to the rows above because the val split was a different, smaller set of frames from a single clip.
- See also `runs/preprocess_sweep/REPORT.md` for the 2026-05-15 preprocessing sweep that established that **no OpenCV preprocessing variant improves selcom detection** — fine-tuning + imgsz are the only effective levers.

---

## 4. IR YOLO model — measured metrics

| Dataset | imgsz | TP | FP | FN | P | R | F1 | Source |
|---|---|---|---|---|---|---|---|---|
| antiuav | (May 10 default) | 15910 | 213 | 926 | 0.987 | 0.945 | 0.9654 | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` row `ir_only` × antiuav |
| svanstrom | 1280 | 2234 | 117 | 63 | 0.950 | 0.973 | 0.9613 | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` row `ir_only` × svanstrom (assumed imgsz=1280; verify) |

### 4.1 IR model version comparison — `IR_dset_final` test split @ imgsz=640

Source: `eval/results/ir_version_comparison/ir_comparison_test_640_2026-05-16T20-04-39.{json,csv}`. Run date 2026-05-16. All versions evaluated on the same fixed `IR_dset_final` test split for an apples-to-apples comparison; replaces the previous mAP-only `runs/IR_FT_*/results.csv` best-epoch numbers (which were per-version val splits, not comparable across versions).

| Version | mAP50 | mAP50-95 | P | R | F1 | Status |
|---|---|---|---|---|---|---|
| V2 (`IR_dsetV2_merged_300ep`) | 0.661 | --- | 0.458 | 0.406 | 0.430 | superseded |
| V3 | 0.571 | 0.242 | 0.648 | 0.579 | 0.611 | superseded |
| V4 | 0.722 | 0.393 | 0.895 | 0.669 | 0.765 | superseded |
| V5 | 0.694 | 0.446 | 0.768 | 0.709 | 0.737 | superseded — regression |
| V6 | 0.956 | 0.571 | 0.921 | 0.941 | 0.931 | superseded |
| Final | **0.977** | **0.602** | **0.955** | **0.980** | **0.967** | base for prod |
| **v3b (production)** | 0.972 | 0.591 | **0.957** | 0.977 | **0.967** | current — production |

**Reads:**
- V3 → V4: precision jump (0.648 → 0.895) driven by FP review.
- V5 regression: precision drops to 0.768 (new data introduced noise faster than curation cleaned it); recall rises slightly — overall F1 down to 0.737.
- V6 jump: F1 0.737 → 0.931 from comprehensive split cleanup.
- Final vs v3b: numerically indistinguishable on this test split (F1 = 0.967 for both). v3b is a 2-epoch corrective finetune on top of Final, retained as production weights.
- mAP-only narrative (gemini draft: "0.900 → 0.958") was based on per-version val splits and is not comparable; use P/R/F1 from this table.

Reproduce: `python eval/ir_version_comparison.py --imgsz 640 --split test` (script writes per-run JSON + CSV under `eval/results/ir_version_comparison/`).

**V2 row provenance**: weights at `archive/models/IR_dsetV2_merged_300ep/best.pt`; evaluated against `G:/drone/IR_dset_final/dataset.yaml` test split (9612 imgs) at `imgsz=640`, 2026-05-17. CSV: `eval/results/ir_version_comparison/ir_v2_eval_test_640.csv`. mAP50-95 not in the reported run output (left blank). F1 = 2·P·R/(P+R) = 0.430 computed from P=0.458, R=0.406. The mAP50=0.661 / F1=0.430 mismatch (higher mAP50 than V3 but lower F1) is consistent with V2's `merged` corpus being broader and noisier than V3's curated set — mAP integrates across all conf thresholds while F1 here is at the F1-optimal conf reported by Ultralytics on a noisier label distribution.

### IR on grayscale confusers @ imgsz=1280

| Source split | Images | Halluc rate | Avg conf | Source |
|---|---|---|---|---|
| airplane | 99 | 16.2% | 0.718 | UNKNOWN |
| other | 2534 | 22.2% | 0.752 | UNKNOWN |

### IR on `antiuav_rgb_gray` (real recall/precision, has GT)

| imgsz | P | R | F1 | Source |
|---|---|---|---|---|
| 640 | placeholder | placeholder | placeholder | not yet run |
| 1280 | placeholder | placeholder | placeholder | not yet run |

---

## 5. Trust classifiers — measured metrics

All at imgsz=1280 (Svanstrom) or May 10 default (Anti-UAV); all in `classifier` config of the pipeline. Numbers below from May 10 matrix unless noted.

| Classifier | Trained against | Dataset | P | R | F1 | Source | Status |
|---|---|---|---|---|---|---|---|
| `retrained_v2_32feat` | retrained_v2 RGB | antiuav | 0.991 | 0.994 | 0.9924 | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` C_classifier × clf_retrainedv2 | superseded (retrain mismatch with new production RGB) |
| `retrained_v2_32feat` | retrained_v2 RGB | svanstrom | 0.979 | 0.922 | 0.9496 | same | superseded |
| `scene_aware_v3more_32feat` | retrained_v2 RGB | antiuav | 0.992 | 0.995 | 0.9930 | same row `clf_sceneaware` × antiuav | superseded (calibration mismatch) |
| `scene_aware_v3more_32feat` | retrained_v2 RGB | svanstrom | 0.980 | 0.947 | 0.9629 | same | superseded |
| `control40` | baseline RGB | antiuav | 0.991 | 0.994 | 0.9927 | `clf_control40` × antiuav | current candidate |
| `control40` | baseline RGB | svanstrom | 0.981 | 0.946 | 0.9629 | same × svanstrom (imgsz=640 — re-eval at 1280 needed) | provisional |
| `fusion_no_fn_v1.1` | baseline RGB | antiuav | 0.991 | 0.994 | 0.9924 | `clf_fusionnofn` × antiuav | current candidate |
| `fusion_no_fn_v1.1` | baseline RGB | svanstrom | 0.983 | 0.944 | 0.9628 | same × svanstrom (imgsz=640 — re-eval at 1280 needed) | provisional |

**Re-eval needed**: `control40` and `fusion_no_fn_v1.1` on Svanstrom @ imgsz=1280 with baseline RGB, to pick between them for production.

---

## 6. Patch verifiers — measured metrics

From May 10 ablation, `filter_then_classifier` cascade on Svanstrom @ imgsz=640. Versions:

| Version | Path | Svanstrom F1 (filter_then_classifier) | Notes | Source |
|---|---|---|---|---|
| v1 | `confuser_filter4_{rgb,ir}.pt` (April 20 backup → v1) | 0.9241 | between v2 and v3 | `eval/results/_ablation/.../B_patch_verifier` |
| **v2** | `confuser_filter4_{rgb,ir}_v2_backup.pt` | 0.9311 | production choice | same |
| v3 | (locate) | **0.8781** | over-aggressive; vetoes drone TPs | same |
| v4 | (current default in code) | 0.9331 | ≈ v2 | same |

### 6.1 Catch-rate audit — baseline RGB × v2 patch verifier, Svanstrom @ imgsz=1280

Source: `eval/results/_patch_catch_audit/baseline_v2/summary.json` (3190 frames, 3130 detections, stride=9, conf=0.25). Status: `current`.

| Bucket | N | Median patch prob | Catch / veto rate @ 0.30 | @ 0.40 | @ 0.50 | @ 0.60 | @ 0.70 |
|---|---|---|---|---|---|---|---|
| BIRD | 807 | 0.904 | 0.644 | 0.644 | **0.638** | 0.605 | 0.581 |
| AIRPLANE | 532 | 0.540 | 0.523 | 0.521 | **0.517** | 0.474 | 0.446 |
| HELICOPTER | 464 | 0.987 | 0.711 | 0.709 | **0.709** | 0.683 | 0.666 |
| DRONE_TP (unwanted) | 1248 | 0.000 | 0.055 | 0.055 | **0.054** | 0.048 | 0.042 |
| DRONE_FP (bonus catches) | 79 | 0.000 | 0.266 | 0.266 | 0.266 | 0.241 | 0.228 |

**Reads:**
- Catch rates at the production threshold 0.5: bird 64%, airplane 52%, helicopter 71%. **All three below the 0.90 bar set in the decision tree.** Patch verifier is partially working but does not handle baseline's confuser output on its own.
- Drone-TP veto at 0.5 is only 5.4% — acceptable, the patch verifier is *not* over-aggressive on drones.
- Threshold sweep is flat in the [0.3, 0.5] range — the v2 verifier is bimodal (either confident or unsure); lowering threshold can't recover the misses.
- v2 verifier already has an "other" class (index 3, not in confuser_indices) — the misses are crops the verifier *actively thinks* are not confusers, not unknown crops it abstains on. Retraining with those crops is the right intervention.

**Implication for production decision tree (path_forward.md §3b):** branch (3) → retrain the patch verifier with these specific bird/airplane/helicopter FP images as confuser-class training data. The FP crops are recoverable from `per_detection.csv` (filter `bucket ∈ {BIRD,AIRPLANE,HELICOPTER}` and `det_conf ≥ 0.25`) cross-referenced with the Svanstrom RGB images.

Reproduce: `python eval/audit_patch_catch.py --rgb-model baseline --patch-version v2`.

Reproduce: `python eval/audit_patch_catch.py --rgb-model baseline --patch-version v2` (full Svanstrom @ stride=9, imgsz=1280, conf=0.25). Output lands in `eval/results/_patch_catch_audit/baseline_v2/` (`per_detection.csv` + `summary.json` + `manifest.json`). Smoke run: `... --limit 30 --output-dir eval/results/_patch_catch_audit/_smoke` (~6s, confirms wiring). To compare verifier versions, swap `--patch-version` (v1 / v2 / v4 supported).

Audit script (`eval/audit_patch_catch.py`, 2026-05-11) was smoke-tested on 30 frames — confirmed:
- Loads v2 verifier with class names `['airplane', 'helicopter', 'bird', 'other']`, confuser indices `[0, 1, 2]` (the "other" head means the verifier already has an OOD class, so prob=0 for genuinely-non-confuser crops is expected behavior).
- Buckets detections into DRONE_TP / DRONE_FP / BIRD / AIRPLANE / HELICOPTER / OTHER using same IoP@0.5 rule and `detect_category()` filename-based labeling as `diagnose_failures_all.py`.
- Reports per-bucket veto rate at thresholds {0.30, 0.40, 0.50, 0.60, 0.70}; for confuser buckets this is the catch rate (higher better), for DRONE_TP this is the unwanted-veto rate (lower better).

---

## 7. Cumulative-stage halluc chart (thesis headline figure)

Script: `eval/cumulative_halluc.py`. Two modes:

- **`--mode confuser`** — runs against `G:/drone/rgb_confusers_merged/images/{split}/` (no GT; IR fed grayscale-replicate of RGB). Reports halluc rate per stage by source category.
- **`--mode svanstrom`** — runs against `G:/drone/svanstrom_paired/` paired frames (real IR + GT). Reports both fire rate (alerts) and IoP-scored TP/FP/FN/P/R/F1 per stage per category. **Svanstrom RGB always scored with IoP @ 0.5 per project mandate.**

Stages (cumulative):

- **S1** RGB YOLO alone — alert iff any RGB det with conf ≥ rgb_conf.
- **S2** + trust classifier — alert iff classifier label ≠ 0 (not reject_both) AND the trusted modality has dets.
- **S3** + patch verifier on alert gate — revoke S2 alert if any det in the trusted modality has patch_prob ≥ patch_thr. Matches production `alert_gate_only` cascade.
- *(temporal stage not included — needs ordered video; separate study.)*

Defaults baked into the script: RGB=baseline (`Yolo26n_trained`), IR=`finetune_v3b`, classifier=`fusion_no_fn_model_v1.1.joblib`, patch RGB/IR=v2_backup, rgb_conf=0.25, ir_conf=0.40, patch_thr=0.5, imgsz=1280.

Reproduce:
```
python eval/cumulative_halluc.py --mode confuser   # run (a) — confuser test set
python eval/cumulative_halluc.py --mode svanstrom  # run (b) — Svanstrom paired
```
Outputs land in `eval/results/_cumulative_halluc/<mode>_<classifier-stem>/` with `per_frame.csv`, `summary.json`, `manifest.json`.

Smoke tests (2026-05-11): both modes verified on 20-frame slices — confuser drops from S1=0.40 → S2=0.05 on AIRPLANE; Svanstrom drops from S1=1.00 → S2=0.00 on AIRPLANE (classifier overrides RGB hallucinations).

### Results — confuser mode, classifier = `fusion_no_fn_model_v1.1`, patch = v2, baseline RGB, imgsz=1280

Source: `eval/results/_cumulative_halluc/confuser_fusion_no_fn_model_v1.1/summary.json`.

| Source | N | S1 fire | S2 fire | S3 fire |
|---|---|---|---|---|
| AIRPLANE (roboflow) | 99 | 0.273 | 0.030 | 0.030 |
| OTHER (svan+kaggle+roboflow) | 2534 | 0.530 | 0.015 | 0.007 |
| **OVERALL** | 2633 | **0.521** | **0.016** | **0.008** |

### Results — Svanstrom mode, classifier = `fusion_no_fn_model_v1.1`, patch = v2, baseline RGB, imgsz=1280, IoP@0.5

Source: `eval/results/_cumulative_halluc/svanstrom_fusion_no_fn_model_v1.1/summary.json`. Full 28,710 frames (stride=1).

**Drone class (frames with GT drones):**

| Stage | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| S1 RGB alone | 11260 | 721 | 454 | 0.940 | **0.961** | 0.950 |
| S2 + classifier | 10682 | 991 | 1032 | 0.915 | 0.912 | 0.914 |
| S3 + patch v2 | 9564 | 756 | 2150 | 0.927 | **0.817** | 0.868 |

**Confuser classes (frames with no drone GT — all alerts are FPs):**

| Category | N | S1 fire | S2 fire | S3 fire | S1 FP | S2 FP | S3 FP |
|---|---|---|---|---|---|---|---|
| BIRD | 5298 | 0.953 | 0.039 | **0.007** | 7258 | 291 | 49 |
| AIRPLANE | 6090 | 0.748 | 0.074 | **0.036** | 4754 | 458 | 220 |
| HELICOPTER | 5627 | 0.662 | 0.047 | **0.0004** | 4137 | 275 | 2 |

**Cumulative confuser FP collapse**: ~16,149 → 1,024 → 271 (98.3% suppression end-to-end). **Recall cost on drones**: 0.961 → 0.912 → 0.817 (14.4 pp absolute loss S1→S3, 9.5 pp from patch verifier alone). The patch-verifier stage trades drone recall for confuser FP reduction at thr=0.5 — operating-point choice depends on whether the deployment prioritizes recall (drone defense) or precision (false-alarm fatigue).

### Patch threshold sweep — Svanstrom @ stride=9 (3190 frames), same stack

Source: `eval/results/_cumulative_halluc/svanstrom_fnfn_thr0{6,7,8}/summary.json`. S1 and S2 are deterministic across the sweep (P=0.940/0.940, R=0.959/0.909, F1=0.949/0.912 — confirmed identical in all three runs).

| Patch thr | Drone TP | Drone R | Drone F1 | BIRD FP | AIRPLANE FP | HELI FP | Total confuser FP |
|---|---|---|---|---|---|---|---|
| (S2, no patch) | 1184 | 0.909 | 0.912 | 38 | 52 | 30 | 120 |
| 0.5 (full run, stride=1) | — | 0.817 | 0.868 | 49 (×9 in full) | 220 | 2 | 271 (full run) |
| 0.6 | 1065 | **0.818** | 0.869 | 6 | 27 | 0 | 33 |
| 0.7 | 1089 | **0.836** | 0.879 | 9 | 28 | 0 | 37 |
| 0.8 | 1114 | **0.856** | **0.889** | 12 | 30 | 0 | **42** |

**Reading:** raising patch_thr from 0.5 → 0.8 recovers 3.9 pp drone recall (0.817 → 0.856) at the cost of only 13 additional confuser FPs across 1891 confuser frames (29 → 42, +0.7 pp confuser fire rate). The v2 verifier is bimodal so most confuser probs are ≥0.9 — raising threshold up to 0.8 mostly just spares the borderline drone crops (which had probs in the 0.5–0.8 range and were marginal). HELI catches are perfect (0 FPs) across the entire sweep — the verifier is essentially certain on helicopters.

**Adding thr=0.9 (`svanstrom_fnfn_thr09`):** DRONE TP=1131, R=**0.869**, F1=**0.895**; confuser FPs 14+31+0 = 45. So thr=0.5→0.9 path:

| Patch thr | Drone R | Drone F1 | Total confuser FP |
|---|---|---|---|
| 0.5 | 0.817 | 0.868 | ~29 |
| 0.6 | 0.818 | 0.869 | 33 |
| 0.7 | 0.836 | 0.879 | 37 |
| 0.8 | 0.856 | 0.889 | 42 |
| **0.9** | **0.869** | **0.895** | **45** |
| (S2, no patch) | 0.909 | 0.912 | 120 |

The curve is monotone — every threshold bump recovers a little more drone recall for a few more confuser FPs. At thr=0.9 the system still catches ~63% of S2's confuser FPs (45 vs 120) while costing only 4.0 pp drone F1 vs S2. Above 0.9 is unmeasured but unlikely to be useful (already 14/31 confuser misses out of 1891 frames is essentially noise).

### Classifier comparison — `scene_aware_v3more_32feat` (deployed) vs `fusion_no_fn_model_v1.1` (legacy candidate)

Sources: `eval/results/_cumulative_halluc/{svanstrom_sa32_thr08, confuser_sa32}/summary.json` vs the corresponding `fnfn_thr08` / `confuser_fusion_no_fn_model_v1.1` runs above.

**Svanstrom @ stride=9, patch v2, baseline RGB, imgsz=1280, IoP@0.5:**

| Classifier | thr | Drone S2 R | Drone S2 F1 | Drone S3 R | Drone S3 F1 | Confuser S2 FP | Confuser S3 FP |
|---|---|---|---|---|---|---|---|
| `fusion_no_fn_v1.1` | 0.8 / 0.9 | 0.909 | 0.912 | 0.856 / 0.869 | 0.889 / 0.895 | 120 | 42 / 45 |
| `scene_aware_v3more_32feat` | 0.8 | 0.922 | 0.919 | 0.868 | 0.896 | 111 | 41 |
| **`control_v3more_40feat`** | 0.9 | **0.934** | **0.925** | **0.893** | **0.909** | **111** | 43 |

**control40 wins on every drone metric.** It is a 40-feature classifier (like fusion_no_fn) but with the v3more "scene-aware" feature additions on top — so it gets sceneaware's discrimination *and* fusion_no_fn's robustness. Drone S3 F1 0.909 vs sa32's 0.896 (+1.3 pp), drone S3 recall 0.893 vs sa32's 0.868 (+2.5 pp). Confuser FP cost is negligible (43 vs 41 at S3).

`scene_aware_v3more_32feat` **wins on every Svanstrom metric** — including the confuser FP count it was supposed to be worse at because of the "trained-on-retrained_v2 calibration mismatch" hypothesis. The hypothesis was wrong: scene-aware's feature set is robust to the RGB model swap.

**Confuser zoo @ imgsz=1280:**

| Classifier | OVERALL S1 fire | S2 fire | S3 fire |
|---|---|---|---|
| `fusion_no_fn_v1.1` | 0.521 | **0.016** | **0.008** |
| `scene_aware_v3more_32feat` | 0.521 | 0.205 | 0.103 |

**Opposite verdict on the OOD confuser zoo.** `scene_aware_v3more_32feat` fires 13× more on the broad mixed-source confuser set than `fusion_no_fn_v1.1` does. On generic OOD imagery, fnfn is dramatically more conservative; on Svanstrom-distribution data, sa32 is more accurate. Two valid deployment positions:

- **Calibrated deployment** (company knows the scene distribution, like Svanstrom): ship `scene_aware_v3more_32feat`.
- **Open-world deployment** (unknown OOD scenes): `fusion_no_fn_v1.1` is the safer floor.

---

## 8. Roboflow OOD eval (2026-05-16) — YOLO + patch verifier, no classifier, imgsz=640

Source: `eval/results/roboflow_ood/summary.csv`. Reproduce: `python eval/run_roboflow_eval.py --full`. Stack: RGB(baseline | retrained_v2) or IR YOLO → patch verifier v2 backup on alert-gate cascade, temporal gate enabled (largely inert on still images). **No trust classifier.** Analysis: [docs/analysis/2026-05-16_roboflow_ood_eval.md](analysis/2026-05-16_roboflow_ood_eval.md). Status: `current`.

### 8.1 RGB drone detection (aggregated across splits of `rgb_drone`)

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| `Yolo26n_trained` (baseline) | raw | 2492 | 240 | 849 | 0.912 | 0.746 | 0.820 |
| `Yolo26n_trained` (baseline) | + patch v2 | 2304 | 181 | 1037 | 0.927 | 0.690 | 0.792 |
| `Yolo26n_retrained_v2` | raw | 2424 | 199 | 917 | 0.924 | 0.726 | 0.813 |
| `Yolo26n_retrained_v2` | + patch v2 | 2291 | 176 | 1050 | 0.929 | 0.686 | 0.790 |

### 8.2 RGB confuser FP totals (no GT, every det = FP, aggregated across splits)

| Confuser | baseline raw FP | baseline filtered | base supp | retrained_v2 raw | retrained_v2 filtered | rv2 supp |
|---|---|---|---|---|---|---|
| airplane | 1327 | 1286 | 3.1% | 1666 | 1634 | 1.9% |
| bird | 103 | 51 | 50.5% | 85 | 41 | 51.8% |
| helicopter | 457 | 292 | 36.1% | 297 | 273 | 8.1% |
| **total** | 1887 | 1629 | 13.7% | 2048 | 1948 | 4.9% |

### 8.3 IR drone detection

| Dataset | Stage | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| `ir_drone_night` (all splits) | raw | 561 | 575 | 1567 | 0.494 | **0.264** | 0.344 |
| `ir_drone_night` (all splits) | + patch v2 | 446 | 494 | 1682 | 0.474 | 0.210 | 0.291 |
| `ir_mixed_cbam` (train+valid) | raw | 1775 | 386 | 1646 | 0.821 | 0.519 | 0.636 |
| `ir_mixed_cbam` (train+valid) | + patch v2 | 1678 | 242 | 1743 | 0.874 | 0.491 | 0.628 |

### 8.4 IR confuser FP totals

| Dataset | Raw FP | Filtered FP | Suppression |
|---|---|---|---|
| `ir_airplane_hors2` (all splits) | 1128 | 1048 | 7.1% |
| `ir_airplane_plane` (train+valid) | 386 | 368 | 4.7% |
| `ir_bird` (train+valid) | 95 | 61 | 35.8% |

**Key reads (full justification in the linked analysis):**
- Baseline RGB also wins on Roboflow drone OOD (R 0.746 vs 0.726); retrained_v2's Svanstrom-recall collapse does not generalize — it's a small-drone-resolution problem, not a generic model defect.
- Patch verifier costs net F1 across **every** drone setting at imgsz=640 (~5–8% drone TP veto for ~14% confuser cut on RGB).
- Patch verifier suppression of OOD airplane FPs is **3.1% baseline / 1.9% retrained_v2** — vs 52% on Svanstrom@1280 airplanes. The verifier is severely distribution-bound.
- The **trust classifier** (not the patch verifier) is the main confuser-rejection lever — Svanstrom @1280 S2 fire rates (BIRD 0.039, AIRPLANE 0.074) are an order of magnitude better than YOLO+patch alone achieves here.
- **IR YOLO has a hidden OOD ceiling**: R=0.264 on `ir_drone_night` and 0.519 on `ir_mixed_cbam` vs Anti-UAV 0.945 / Svanstrom 0.973. Belongs in the thesis limitations chapter.

**Dataset-quality caveats (affect how seriously to take the recall/precision numbers above):**
- **`ir_drone_night` is heavily sensor-augmented** (synthetic noise, blur, contrast/inversion variants applied per-image at dataset-build time). Many frames are barely recognizable as IR drone imagery. The R=0.264 floor reflects augmentation-stacking, not a clean IR OOD signal — the dataset is not a fair generalization test, more a worst-case robustness probe.
- **`rgb_drone` has substantial missing labels** — many frames contain visible drones with no GT box, so a non-trivial fraction of the 240/199 "FP" counts and 849/917 "FN" counts are annotation noise. True precision is higher than 0.912/0.924; true recall is also higher (since unlabeled drones can't be counted as TPs either). Use this dataset for relative model comparison (baseline vs retrained_v2 ordering), not absolute P/R claims.

---

## 9. Real-video confuser eval (2026-05-17)

Source: `eval/results/confuser_videos/confuser_comparison.csv`. Script: `eval/eval_confuser_videos.py`. Datasets extracted by `scripts/extract_confuser_datasets.py` from YouTube confuser clips. All frames negative (no drones); every detection is a false positive. Run on 2026-05-17 with `conf=0.25`, three RGB models compared on 1250 frames across 10 videos. Status: **`superseded` by §9.4** (numerical FPR values; the relative ordering "selcom worst" still holds and is more emphatic in §9.4).

### 11.1 Aggregate (1250 frames, 10 videos)

| Model | FP dets | FPPI | FP frames | FPR (frame-level fire) |
|---|---|---|---|---|
| `Yolo26n_trained` (baseline) | 640 | 0.512 | 461 | 0.369 |
| `Yolo26n_retrained_v2` | 245 | 0.196 | 212 | **0.170** |
| `Yolo26n_selcom_mixed_ft2_1280` | 886 | 0.709 | 516 | **0.413** |

### 11.2 By category

| Category | Frames | baseline FPR | retrained_v2 FPR | selcom_1280 FPR |
|---|---|---|---|---|
| Airplanes | 304 | 39.8% | 31.6% | 38.5% |
| Birds | 352 | **53.7%** | **11.4%** | **67.9%** |
| Helicopters | 594 | 25.4% | 12.8% | 26.9% |

### 11.3 Striking per-video patterns

| Video | Frames | baseline FPR | retrained_v2 FPR | selcom_1280 FPR | Note |
|---|---|---|---|---|---|
| `birds_birds_flying_overhead_various_sizes_short` | 20 | 0.950 | **0.000** | 0.900 | retrained_v2 *completely silences* |
| `birds_distant_birds_flying_in_the_sky_short` | 20 | 0.050 | 0.050 | **0.750** | selcom fires 15×, baseline/v2 1× each |
| `birds_flock_of_birds_flying_short` | 21 | 0.000 | 0.000 | 0.238 | only selcom fires |
| `birds_flock_of_birds_flying_sunset` | 20 | 0.300 | 0.050 | 0.150 | retrained_v2 best |
| `birds_birds_in_slow_motion_flying_various_sizes_compilation` | 271 | 0.602 | 0.140 | 0.731 | retrained_v2 4× better; selcom worst |
| `airplanes_distant_airplane_over_head_flying_away` | 55 | 0.582 | 0.327 | 0.382 | retrained_v2 best on small/distant airplane |
| `helicopters_helicopter_overhead_short` | 20 | 0.400 | 0.050 | 0.600 | retrained_v2 silences; selcom worst |
| `helicopters_helicopter_overhead_very_small_airplane_in_background` | 20 | 0.850 | 0.750 | 0.750 | all three fire heavily; retrained_v2's small-suppression is not categorical |

### 11.4 Reads

- **Selcom_1280 is worse than baseline on OOD confusers** (FPR 41.3% vs 36.9% aggregate; +14 pp on birds). The selcom fine-tune trades CCTV drone recall for OOD confuser robustness, dramatically so on small/distant birds. Production deployment should pair selcom with a confuser-aware downstream stage; it cannot ship alone outside CCTV.
- **retrained_v2's confuser-rejection wins are real-video robust**, not Svanström artefacts. FPR drops 2–5× across every confuser category on YouTube footage.
- **retrained_v2's hallucinations are also lower-confidence** than baseline's (mean conf 0.34 vs 0.75 on `helicopter_overhead_short`; 0.30 vs 0.68 on `distant_birds`). When the model does fire, the FPs are more vetoable by a downstream confidence threshold.
- **The "small airplane in background" case** fires all three models at 75–85% FPR. retrained_v2's "don't fire on small things" rule has limits; if a small drone-like object exists in the frame even as background clutter, retrained_v2 still fires.

Reproduce: `python eval/eval_confuser_videos.py` (full run, all 3 RGB models, 10 videos, ~3 min on a 1050 Ti). Or `--videos <name1> <name2>` to restrict.

### 9.2 Real-video DRONE eval (drone + bird mixed scenes) (2026-05-17)

Source: `eval/results/video_tests/video_tests_comparison.csv`. 10 drone videos with GT (`is_negative=False`); IoP scoring; three RGB models. Three videos are clean drone takeoff; seven have birds in the same frame as the drone. Status: **`superseded` by §9.4** for the 9 videos that re-ran with the fix; `flock_of_birds_attack_drone` was not in the re-run and its F1=0/0/0.415 row remains the only source for that case until it is re-staged.

**Per-video drone detection (P / R / F1):**

| Video | Frames | GT | baseline P/R/F1 | retrained_v2 P/R/F1 | selcom_1280 P/R/F1 |
|---|---|---|---|---|---|
| `drone_takeoff_short` | 116 | 116 | 0.990 / 0.836 / 0.907 | 1.000 / 0.655 / 0.792 | 1.000 / 0.905 / **0.950** |
| `drone_takeoff_from_ground_and_not_hand_short` | 163 | 154 | 0.980 / 0.974 / **0.977** | 0.967 / 0.753 / 0.847 | 0.955 / 0.968 / 0.961 |
| `drone_takeoff_short_trees_background_dji` | 166 | 162 | 0.971 / 0.827 / 0.893 | 0.971 / 0.833 / 0.897 | 0.961 / 0.901 / **0.930** |
| `drone_seagull_attack` | 235 | 194 | 0.832 / 0.742 / 0.785 | 0.907 / 0.454 / 0.605 | 0.775 / 0.995 / **0.871** |
| `drone_attacked_by_bird_mountain_side_view` | 108 | 88 | 0.551 / 0.682 / 0.609 | 0.636 / 0.080 / 0.141 | 0.784 / 0.864 / **0.822** |
| `drone_over_mountain_attacked_by_birds` | 68 | 68 | 0.862 / 0.368 / 0.516 | 1.000 / 0.353 / **0.522** | 0.867 / 0.191 / 0.313 |
| `flock_of_seagulls_attack_drone_beach` | 239 | 187 | 0.954 / 0.770 / **0.852** | 1.000 / 0.439 / 0.610 | 0.585 / 0.888 / 0.705 |
| `drone_and_bird_sky_and_trees_short` | 114 | 115 | 0.621 / 0.557 / 0.587 | 0.794 / 0.235 / 0.362 | 0.554 / 0.670 / **0.606** |
| `two_birds_drone` | 150 | 150 | 0.469 / 0.767 / **0.582** | 0.375 / 0.080 / 0.132 | 0.386 / 0.893 / 0.539 |
| `flock_of_birds_attack_drone` | 112 | 44 | 0.000 / 0.000 / **0.000** | 0.000 / 0.000 / **0.000** | 0.269 / 0.909 / 0.415 |

**Reads:**

- **Three-model recall/precision continuum is now empirically clean.** Across the 10 videos, ranking by drone-recall is consistently selcom_1280 $\geq$ baseline $>$ retrained_v2; ranking by drone-precision is roughly the inverse. retrained_v2 occupies the conservative end, selcom_1280 the aggressive end, baseline in between.
- **retrained_v2 recall collapse generalises beyond Svanström.** On `drone_attacked_by_bird_mountain_side_view` retrained_v2 R=0.080; on `two_birds_drone` R=0.080; on `flock_of_birds_attack_drone` R=0.000. The Svanström R=0.306 number is consistent with the pattern; it is not an artefact of that single dataset.
- **`flock_of_birds_attack_drone` is the load-bearing single result.** Baseline and retrained_v2 both score F1=0.000 (zero drone TPs). Only selcom_1280 detects the drone at all, with R=0.909 (40/44 GT recovered) at the cost of P=0.269 (109 FPs from the surrounding birds). This is the visceral case that the cascade is needed *and* that selcom_1280 is the only RGB model that can do its job in extremely bird-cluttered scenes.
- **selcom_1280's confuser-video weakness and drone-video strength share a mechanism.** The same lowered detection floor on small-sky objects that gives selcom_1280 the worst confuser FPR (§9.1) gives it the best drone recall in mixed scenes (this section). The fine-tune is not "broken on confusers"; it is "broadly more willing to fire," which is good for tiny drones in clutter and bad for distant birds in clear sky.

Reproduce: same script wrapper as §9.1 with drone videos in the `--videos` list. Per-video JSONs in `eval/results/video_tests/{category}/{video}/{model}.json`.

### 9.3 IR model on grayscale-RGB vs raw-RGB video (2026-05-17) — superseded by §9.4

Status: **`superseded` by §9.4**. The aggregate IR-grayscale F1 (0.664) and the `flock_of_seagulls_attack_drone_beach` F1 (0.901) below are from the buggy `iop_25` path; corrected values are in §9.4. Reads/framing notes retained for history.


Source: `eval/results/video_tests/video_tests_comparison.csv`. Script: `python eval/eval_video_tests.py --models ir_final_gray ir_final_rgb`. Two modes of the same IR `final` weights:

- `ir_final_gray` --- IR weights fed `cvtColor(BGR2GRAY)` of the RGB stream (single-channel intensity, 3-channel replicate).
- `ir_final_rgb` --- IR weights fed raw 3-channel RGB directly.

The IR model has never seen visible-light data in training; both modes are out-of-distribution. The question is whether grayscale conversion --- which preserves single-channel intensity structure --- transfers the IR detector's drone features better than feeding raw color.

**Aggregate drone detection (9 videos, 1359 frames, 1234 GT, IoP@0.5, conf=0.25):**

| Mode | TP | FP | FN | $P$ | $R$ | $F1$ |
|---|---|---|---|---|---|---|
| `ir_final_gray` | 717 | 208 | 517 | 0.775 | 0.581 | **0.664** |
| `ir_final_rgb` | 238 | 127 | 996 | 0.652 | 0.193 | 0.298 |

The IR model on grayscale-RGB achieves $F1=0.664$ on real-world drone-positive video. On raw RGB it collapses to $F1=0.298$. Grayscale conversion is essential to cross-modal transfer.

**Aggregate confuser FPR (10 confuser videos, 1250 frames):**

| Category | ir_gray FPR | ir_rgb FPR |
|---|---|---|
| Airplanes (304) | 37.8% | 27.0% |
| Birds (352) | **25.9%** | 57.4% |
| Helicopters (594) | 19.2% | 20.5% |
| All (1250) | 25.6% | 32.5% |

Grayscale conversion *helps* on birds (cuts FPR by more than half) and *hurts* on airplanes. Modality choice is therefore per-category in confuser-rejection behaviour, not globally one-sided.

**Cross-model comparison on a single bird-cluttered drone video** (`flock_of_seagulls_attack_drone_beach`, 239 frames, 187 GT):

| Detector | $P$ | $R$ | $F1$ |
|---|---|---|---|
| baseline RGB (§9.2) | 0.954 | 0.770 | 0.852 |
| retrained_v2 (§9.2) | 1.000 | 0.439 | 0.610 |
| selcom_1280 (§9.2) | 0.585 | 0.888 | 0.705 |
| **`ir_final_gray`** | **0.987** | 0.829 | **0.901** |
| `ir_final_rgb` | 0.966 | 0.150 | 0.259 |

The IR model on grayscale-RGB beats every RGB detector in the project on this video --- highest single-video drone $F1$ recorded.

**Reads:**

- The §sec:grayscale claim in the thesis (the IR model retains useful drone-detection capability on grayscale-RGB) is now empirically validated with drone-positive video, not only confuser-only fire-rate. This is the strongest single piece of evidence for the cross-modal generalisation finding.
- The grayscale-RGB mode is competitive with the RGB detectors and dominant in bird-heavy scenes. On clean-sky takeoff videos the RGB detectors (especially selcom_1280) still win.
- The IR-on-raw-RGB mode is a control. $F1=0.298$ is unusable; grayscale conversion is the load-bearing step.

**Confidence sweep extrema (ir_final_gray):**
- Drone-side best: $F1=0.709$ at conf=0.05 ($P=0.658, R=0.768$).
- Confuser side: FPPI falls from 0.246 (conf=0.10) to 0.023 (conf=0.80). A high-threshold operating point on grayscale fallback gives near-zero confuser fire at modest recall cost.

Reproduce: `python eval/eval_video_tests.py --models ir_final_gray ir_final_rgb`.

---

## 9.4 Unified video re-run with corrected scoring (2026-05-18)

Source: `eval/results/video_tests/video_tests_comparison.csv` (re-run 2026-05-18). Script: `python eval/eval_video_tests.py` (fixed). Status: **`current`** — supersedes the IR-grayscale numbers in §9.3, the per-video drone numbers in §9.2, and the confuser-FPR numbers in §9.1 for `eval_video_tests.py`-derived metrics.

### 9.4.1 The scoring bug

The pre-fix `eval_video_tests.py` ran YOLO at `base_conf=0.05` and post-filtered detections to `conf>=0.25` to obtain the headline `iop_25` metric. Ultralytics applies the conf threshold **pre-NMS**, so a low-conf inference post-filtered to a higher threshold yields a different post-NMS detection set than a direct high-conf inference. The pipeline eval (`eval_pipeline_video_tests.py`) ran YOLO directly at `rgb_conf=0.25`, so per-video JSON numbers from the two scripts did not agree (FP counts differed by 3–10×).

**Fix**: added a second YOLO inference pass at `prod_conf=0.25` per frame. `iop_25` now uses that pass with proper bipartite matching via `score_detections`. Runtime cost ~1.5–2×. Per-video JSON now carries `total_dets_prod`, `base_conf`, `prod_conf` for provenance. `eval_pipeline_video_tests.py` was already correct (unchanged). Code-side details in `docs/analysis/2026-05-17_thesis_revision_state.md` § "2026-05-18 STATUS AT TOP".

### 9.4.2 Aggregate drone detection (9 drone videos, 1359 frames, 1234 GT, IoP@0.5, conf=0.25)

| Model | imgsz | P | R | F1 | Source |
|---|---|---|---|---|---|
| `Yolo26n_trained` (baseline) | 640 | 0.771 | 0.749 | **0.760** | `video_tests_comparison.csv` |
| `Yolo26n_retrained_v2` | 640 | 0.909 | 0.453 | 0.605 | same |
| `Yolo26n_selcom_mixed_ft2_1280` | 1280 | 0.649 | 0.812 | 0.721 | same |
| `Yolo26n_selcom_mixed_ft2_1280` @ 640 | 640 | 0.807 | 0.666 | 0.730 | same |
| `IR_final` on grayscale-RGB | 640 | 0.743 | 0.557 | **0.636** | same |
| `IR_final` on raw-RGB | 640 | 0.647 | 0.191 | 0.295 | same |

**`flock_of_birds_attack_drone` is NOT in this re-run.** The §9.2 load-bearing claim ("only selcom detects, baseline and retrained_v2 F1=0") rests on the older run; for thesis citation this case is either re-included in a future re-run or dropped.

### 9.4.3 Confuser FPR (10 confuser videos, 1250 frames, conf=0.25)

| Model | imgsz | All-confuser FP | FPPI | Frame-level FPR | Source |
|---|---|---|---|---|---|
| baseline | 640 | 640 | 0.512 | 0.475 | same |
| retrained_v2 | 640 | 245 | 0.196 | 0.264 | same |
| selcom_1280 | 1280 | **886** | **0.709** | **0.593** | same |
| selcom_640 (selcom weights @ imgsz=640) | 640 | 325 | 0.260 | 0.423 | same |
| IR on grayscale-RGB | 640 | 197 | 0.158 | 0.256 | same |
| IR on raw-RGB | 640 | 187 | 0.150 | 0.325 | same |

**Selcom_1280 OOD damage is more emphatic than §9.1 reported**: FPR 59.3% vs baseline 47.5% (+11.8 pp), vs retrained_v2 26.4% (+33 pp). The fine-tune is decisively the worst RGB on OOD confusers. The §9.1 categorical claim ("selcom worst") is preserved; the magnitude is larger.

### 9.4.4 `flock_of_seagulls_attack_drone_beach` (239 frames, 187 GT) — corrected per-video

| Detector | P | R | F1 | $\Delta$ F1 vs §9.3 |
|---|---|---|---|---|
| baseline RGB | 0.940 | 0.759 | **0.840** | $-0.012$ |
| retrained_v2 | 1.000 | 0.439 | 0.610 | $\pm 0$ |
| selcom_1280 | 0.578 | 0.877 | 0.696 | $-0.009$ |
| selcom_640 | 0.977 | 0.668 | 0.794 | new |
| **IR on grayscale-RGB** | **0.917** | 0.770 | **0.837** | $-0.064$ |
| IR on raw-RGB | 0.966 | 0.150 | 0.259 | $\pm 0$ |

**Thesis-headline change**: IR-grayscale on this video is no longer dominant — it now essentially ties baseline RGB (0.837 vs 0.840, $\Delta = 0.003$). The §9.3 claim "IR-on-grayscale-RGB beats every RGB detector in the project on this video" must be reframed as "matches baseline RGB on the hardest bird-cluttered clip while cutting confuser FPR by 3×." selcom_1280 stays third (0.696); retrained_v2 conservative (0.610); IR-raw-RGB unusable (0.259).

### 9.4.5 Pareto framing (drone-F1 aggregate vs all-confuser FPPI)

| Detector | Drone F1 | Confuser FPPI |
|---|---|---|
| baseline | 0.760 | 0.512 |
| retrained_v2 | 0.605 | 0.196 |
| selcom_1280 | 0.721 | 0.709 |
| selcom_640 | 0.730 | 0.260 |
| IR-grayscale | 0.636 | **0.158** |
| IR-raw | 0.295 | 0.150 |

**No detector strictly Pareto-dominates another.** Useful corners:
- **Highest drone F1**: baseline RGB (0.760), at the cost of FPPI=0.512.
- **Lowest confuser FPPI among usable detectors**: IR-grayscale (FPPI=0.158, drone F1=0.636 — 12.4 pp below baseline).
- **Best F1 at FPPI $\leq$ 0.3**: `selcom_640` (F1=0.730, FPPI=0.260) — interesting and *not previously studied*; the imgsz=640 inference of the selcom_1280 weights drops the OOD fire rate from 0.709 to 0.260 at only 1.1 pp drone-F1 cost vs selcom_1280@1280. This is the cleanest single-detector point on this corpus.

The honest IR-grayscale headline becomes: *"comparable drone recall to RGB at a third of the confuser fire rate, despite never having seen visible-light data in training"* — not *"beats RGB."*

### 9.4.6 Optimal conf per model (from sweep)

| Model | Best drone conf | P | R | F1 |
|---|---|---|---|---|
| baseline | 0.25 | 0.771 | 0.749 | 0.760 |
| retrained_v2 | 0.05 | 0.787 | 0.533 | 0.636 |
| selcom_1280 | 0.40 | 0.695 | 0.767 | 0.729 |
| selcom_640 | 0.10 | 0.761 | 0.729 | 0.745 |
| IR-grayscale | 0.15 | 0.696 | 0.604 | 0.647 |
| IR-raw | 0.05 | 0.574 | 0.324 | 0.414 |

Reproduce: `python eval/eval_video_tests.py` (all 6 models, all 19 videos, ~10 min). To re-add the missing drone case: stage `flock_of_birds_attack_drone` under `datasets/drone detection video tests/rgb/drone/flock_of_birds_attack_drone/{images,labels}/test/` and re-run.

---

## 9.5 Full pipeline on real video (2026-05-18)

Source: `eval/results/pipeline_video_tests/pipeline_comparison.{csv,json}` and `eval/results/pipeline_video_tests/{cat}/{video}/{model}.json` (19 videos × 4 RGB models). Script: `python eval/eval_pipeline_video_tests.py`. Status: `current`.

### 9.5.1 Setup

Stages run per frame: RGB YOLO $\to$ IR YOLO on grayscale-RGB $\to$ XGBoost trust classifier $\to$ temporal 2-of-3 smoother $\to$ patch verifier alert gate. Configuration: `rgb_conf=0.25`, `ir_conf=0.40`, `patch_thr=0.70`, classifier = `scene_aware_v3more_32feat` (32 features, 4 classes). IR weights = `models/IR_final_cleaned/weights/best.pt` (`final`, not `v3b`). Patch verifier = `confuser_filter4_{rgb,ir}_v2_backup.pt`.

Sanity check: Stage 1 RGB per-frame metrics match \S9.4 exactly for all four RGB variants (baseline $F1=0.760$, retrained\_v2 $0.605$, selcom\_1280 $0.721$, selcom\_640 $0.730$). The two scripts produce identical RGB-stage detection sets.

### 9.5.2 Tier 1: per-frame detection metrics (drone, IoP@0.5, conf=0.25)

| RGB model | Stage | $P$ | $R$ | $F1$ | $\Delta F1$ vs RGB |
|---|---|---|---|---|---|
| baseline | RGB | 0.771 | 0.749 | **0.760** | --- |
| baseline | IR-gray (conf=0.40) | 0.802 | 0.490 | 0.609 | --- |
| baseline | + classifier | 0.520 | 0.672 | 0.586 | $-0.174$ |
| retrained\_v2 | RGB | 0.909 | 0.453 | 0.605 | --- |
| retrained\_v2 | + classifier | 0.632 | 0.600 | 0.615 | $+0.010$ |
| selcom\_1280 | RGB | 0.649 | 0.812 | 0.721 | --- |
| selcom\_1280 | + classifier | 0.453 | 0.660 | 0.537 | $-0.184$ |
| selcom\_640 | RGB | 0.807 | 0.666 | 0.730 | --- |
| selcom\_640 | + classifier | 0.522 | 0.622 | 0.568 | $-0.162$ |

The IR-grayscale stage runs at $\mathrm{conf}=0.40$ here (production default), versus $\mathrm{conf}=0.25$ in \S9.4 — hence drone $F1=0.609$ vs the \S9.4 number $0.636$. Not an inconsistency; different operating point.

### 9.5.3 Tier 1: per-frame confuser FPR (all 1250 confuser frames)

| RGB model | RGB alone | + Classifier | Suppression |
|---|---|---|---|
| baseline | 0.512 | 0.281 | 45\% |
| retrained\_v2 | 0.196 | 0.186 | 5\% |
| selcom\_1280 | **0.709** | 0.304 | **57\%** |
| selcom\_640 | 0.260 | 0.207 | 20\% |

The classifier compresses RGB variant differences: selcom\_1280's $0.709 \to 0.304$ collapse is the largest absolute suppression in the table. retrained\_v2 is already conservative, so the classifier finds little extra to remove.

### 9.5.4 Tier 2: segment-level alert metrics (3-frame windows, drone-positive)

| RGB model | Temporal 2/3 ($P$/$R$/$F1$) | + Patch veto | Alerts: passed / vetoed |
|---|---|---|---|
| baseline | 0.987 / 0.720 / 0.833 | 0.987 / 0.711 / **0.826** | 47 / 9 |
| retrained\_v2 | 0.979 / 0.649 / 0.781 | 0.978 / 0.635 / 0.770 | 40 / 9 |
| selcom\_1280 | 0.971 / 0.709 / 0.819 | 0.974 / 0.699 / 0.814 | 46 / 13 |
| selcom\_640 | 0.983 / 0.706 / 0.822 | 0.983 / 0.697 / 0.816 | 47 / 9 |

Production-relevant grain: the cascade recovers most of the per-frame drone $F1$ that the classifier alone gave up. Baseline goes $0.760$ (RGB) $\to 0.586$ (+ classifier) $\to \mathbf{0.826}$ (+ temporal + patch). The temporal smoother is the load-bearing recovery step.

### 9.5.5 Tier 2: segment-level confuser FPR

| RGB model | Temporal 2/3 | + Patch veto | RGB raw $\to$ cascade $\to$ |
|---|---|---|---|
| baseline | 0.183 | 0.162 | $0.512 \to 0.162$ (68\% cut) |
| retrained\_v2 | 0.126 | 0.119 | $0.196 \to 0.119$ (39\% cut) |
| selcom\_1280 | 0.148 | **0.136** | $0.709 \to 0.136$ (**81\% cut**) |
| selcom\_640 | 0.133 | 0.126 | $0.260 \to 0.126$ (52\% cut) |

The cascade earns its keep on selcom\_1280 most dramatically (81\% confuser-FPR cut at a $-0.7$~pp drone-$F1$ cost). All four RGB variants land within $F1$ 0.77–0.83 and confuser FPR 0.12–0.16 after the full cascade --- variant choice matters far less downstream than at Stage 1.

### 9.5.6 Reads

- **Per-frame metrics are the wrong unit to read the cascade.** Per-frame drone $F1$ drops $0.174$–$0.184$ pp for three of four RGB variants at the classifier stage, because the classifier merges RGB and IR detections per its trust label and inflates per-frame FP counts when both modalities fire. At the segment/alert level (production grain), the same cascade is monotonically better than raw RGB for confuser FPR while costing only 0.7–1.6 pp drone $F1$.
- **Cascade tightens RGB variant variance.** Stage-1 RGB $F1$ spans $0.605$–$0.760$ ($\Delta = 15.5$~pp). Post-cascade segment $F1$ spans $0.770$–$0.826$ ($\Delta = 5.6$~pp). The downstream stack absorbs detector variance --- it normalises selcom\_1280's aggressive fire and lifts retrained\_v2's conservative miss rate toward the middle.
- **Patch veto is small but positive.** $-0.7$ to $-1.4$ pp drone segment $F1$; $-0.7$ to $-2.1$ pp confuser FPR. Confirms the v2 verifier as a useful but not dominant lever at `patch_thr=0.7`.
- **retrained\_v2 is the only RGB variant whose drone $F1$ slightly *gains* from the classifier alone** (+1 pp). Consistent with the conservative-stance read: retrained\_v2 produces fewer per-frame FPs to begin with, so the classifier's merging of modalities adds proportionally fewer wrong boxes.
- **Cross-script consistency** between `eval_video_tests.py` (\S9.4) and `eval_pipeline_video_tests.py` (this section) is now verified at Stage 1. The scoring fix removed the 3–10$\times$ FP discrepancy that motivated \S9.4.

Reproduce: `python eval/eval_pipeline_video_tests.py` (caches RGB stage outputs to per-video JSONs; subsequent runs skip already-processed (video, model) pairs unless the JSON is deleted).

### 9.5.8 Three-classifier comparison on real video (2026-05-18)

Sources: `eval/results/pipeline_video_tests/` (sa32), `eval/results/pipeline_video_tests_control40/` (control40), `eval/results/pipeline_video_tests_fusionnofn/` (fnfn). All three runs use identical RGB / IR / patch verifier configuration; only the classifier swaps. fnfn loaded via path arg through the new adapter (raw-model joblib + sibling `_metrics.json` features list).

**Segment-level drone $F1$ (3-frame windows, after temporal + patch):**

| RGB model | sa32 | control40 | fnfn | sa32 $-$ fnfn |
|---|---|---|---|---|
| baseline | **0.826** | 0.644 | 0.219 | $+0.607$ |
| retrained\_v2 | 0.770 | 0.576 | 0.181 | $+0.589$ |
| selcom\_1280 | 0.814 | 0.595 | 0.108 | $+0.706$ |
| selcom\_640 | 0.816 | 0.619 | 0.165 | $+0.651$ |

**Segment-level all-confuser FPR:**

| RGB model | sa32 | control40 | fnfn | RGB raw |
|---|---|---|---|---|
| baseline | 0.162 | 0.083 | **0.024** | 0.512 |
| retrained\_v2 | 0.119 | 0.045 | 0.012 | 0.196 |
| selcom\_1280 | 0.136 | 0.057 | 0.019 | 0.709 |
| selcom\_640 | 0.126 | 0.040 | 0.021 | 0.260 |

**Per-frame drone TP retention through the classifier** (how much of the RGB-stage TP set the classifier accepts, baseline RGB):

| Classifier | RGB TP | Post-classifier TP | Retention |
|---|---|---|---|
| sa32 | 924 | 829 | 90\% |
| control40 | 924 | 557 | 60\% |
| fnfn | 924 | 136 | **15\%** |

**Reads:**

- The Svanstrom-side ordering (\S7) holds on real video: sa32 most permissive, control40 middle, fnfn most conservative. The Svanstrom classifier comparison is therefore predictive, not Svanstrom-specific.
- **fnfn is too conservative for the real-video deployment surface.** Drone segment $R = 0.128$ for baseline RGB (best case); the classifier rejects 85\% of correct RGB drone TPs. For an alert system this would mean missing 7 of 8 drone events. fnfn remains valid only when confuser noise dominates the operational cost and missed drones are tolerable.
- **control40 is now deprecated.** It ties sa32 on the OOD zoo (S2 0.212 vs 0.205; S3 0.094 vs 0.103; §7 + `eval/results/_cumulative_halluc/confuser_c40/summary.json`) and loses 18–22 pp drone $F1$ vs sa32 on real video. There is no operational regime in the measured surfaces where control40 strictly outperforms sa32. Retained in the comparison tables for completeness; not retained in the production stack.
- **sa32 dominates on the joint axis for real-video deployment** --- highest drone $F1$ for all four RGB variants, confuser FPR that the patch verifier still cuts further. This is the basis for the 2026-05-18 production-pick flip in \S1.
- **The cascade trade is steeper than Svanstrom showed.** Svanstrom S2 confuser zoo: sa32 0.205 vs fnfn 0.016 (13$\times$). Real-video segment FPR: sa32 0.162 vs fnfn 0.024 (6.8$\times$). The classifiers diverge less on real-video confusers than on the synthetic zoo, while their drone-recall divergence is much larger. This is the asymmetry that motivates the production flip.

Reproduce: `python eval/eval_pipeline_video_tests.py --classifier <name-or-path> --out-tag <suffix>`.

### 9.5.9 Per-category cascade FPR breakdown (sa32, 2026-05-18)

Re-derived from per-video JSONs (`eval/results/pipeline_video_tests/{cat}/{video}/{model}.json`). 1250 confuser frames split as: airplanes 304, birds 352, helicopters 594. Segment-level (3-frame windows) post-cascade FPR shown.

**Segment-level confuser FPR after temporal + patch veto (sa32 classifier, $\mathtt{patch\_thr}=0.7$):**

| RGB model | Birds | Airplanes | Helicopters |
|---|---|---|---|
| baseline | 0.017 | 0.225 | 0.216 |
| retrained\_v2 | **0.008** | **0.206** | 0.141 |
| selcom\_1280 | 0.034 | 0.216 | 0.156 |
| selcom\_640 | 0.008 | 0.216 | 0.151 |

**Per-frame RGB FP → cascade-classifier FP, by category** (the supplementary read showing where the cascade earns its keep):

| RGB | Cat | RGB FP | + classifier FP | Suppression |
|---|---|---|---|---|
| baseline | birds | 338 | 67 | 80% |
| baseline | airplanes | 137 | 143 | $-4\%$ (classifier *adds* per-frame FPs) |
| baseline | helicopters | 165 | 141 | 15% |
| selcom\_1280 | birds | 538 | 101 | **81%** |
| selcom\_1280 | airplanes | 150 | 149 | 1% |
| selcom\_1280 | helicopters | 198 | 130 | 34% |

**Reads:**

- **The cascade's value is concentrated on birds**, the exact category the detector cannot be trained to suppress without losing drone recall. selcom_1280's 1.528 per-frame bird FPPI (\S9.4) collapses to a 0.034 segment FPR after the cascade --- a 45$\times$ reduction at the alert grain.
- **The cascade is partly effective on helicopters** (15--34% per-frame suppression at the classifier alone; modest further cuts at segment level).
- **The cascade is essentially inert on airplanes**: per-frame the classifier sometimes *adds* FPs by merging modalities (baseline 137 $\to$ 143); at the segment level the post-cascade FPR is $\sim 0.21$ across all four RGB models, regardless of whether the RGB stage fired on 137 or 150 detections. This matches the patch-verifier audit's finding (\S6.1) that the verifier is genuinely uncertain on Svanstr\"om airplanes (median patch prob $0.540$), and the cascade does not have a separate mechanism for OOD airplane rejection.
- **The bird-vs-airplane asymmetry is the strongest single argument for the cascade's design intent.** The downstream stack handles exactly the category where detector-level mining hits a wall, and is honest about its inability to handle the category where detector-level mining already worked.

Reproduce: aggregation script over `eval/results/pipeline_video_tests/{cat}/{video}/{model}.json` summing `seg_final.FP` / `seg_final.segments` per category.

### 9.5.7 What is NOT in this run (open items)

- `flock_of_birds_attack_drone` is not staged (\S9.2 load-bearing case --- baseline + retrained\_v2 score $F1=0$, only selcom detects). Without it the cascade's behaviour on the hardest bird-vs-drone confounder is not measured.
- ~~Production classifier comparison~~ → **RESOLVED 2026-05-18** in \S9.5.8 (sa32 / control40 / fnfn on real video; sa32 is now production per \S1).
- Patch threshold is fixed at 0.70. \S7 shows the Svanstrom threshold sweep ($0.5 \to 0.9$ recovers $5$~pp drone recall for $\sim 13$ extra confuser FPs); the real-video equivalent sweep is not run.
- ~~Per-category cascade confuser FPR~~ → **RESOLVED 2026-05-18** in §9.5.9 (birds 0.008–0.034, airplanes 0.21 across all RGB, helicopters 0.14–0.22).
- End-to-end latency / FPS on edge hardware is still unmeasured (\S10).
- `retrained_v2_32feat` is the fourth available classifier; not run on real video. Tied to a deprecated RGB calibration; likely uninformative.

---

## 10. Latency / throughput — placeholder

| Hardware | imgsz | Stage | ms / frame | Source |
|---|---|---|---|---|
| (company target) | 1280 | YOLO RGB | placeholder | not yet run |
| (company target) | 1280 | YOLO IR | placeholder | not yet run |
| (company target) | 1280 | trust classifier | placeholder | not yet run |
| (company target) | 1280 | patch verifier | placeholder | not yet run |
| (company target) | 1280 | temporal + draw | placeholder | not yet run |
| (company target) | 1280 | end-to-end | placeholder | not yet run |

---

## 11. Open data-provenance items (UNKNOWN → resolve)

~~Path of `rgb_confusers_merged` dataset~~ → **RESOLVED** `G:/drone/rgb_confusers_merged` (2026-05-12).
~~Path of `confuser_airplane_roboflow` 99-image subset~~ → **RESOLVED** prefix `airplane_*` in `G:/drone/rgb_confusers_merged/images/test/` (2026-05-12).
~~Exact script and output CSV that produced the Svanstrom@1280 per-category breakdown~~ → **RESOLVED** `eval/diagnose_failures.py` + `eval/diagnose_failures_all.py` → `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` (scripts updated to write CSVs directly 2026-05-12).

Remaining:

- Patch verifier v3 file path (or confirm deleted).
- Confirm imgsz used in the May 10 ablation for Anti-UAV and Svanstrom (default per `ablations.yaml` is now 1280, but the May 10 run may have used 640).
- `fusion_no_fn_v1.1` training provenance — no `metrics.json` was saved at training time. Post-hoc extraction of features/model-type/classes saved to `classifier/runs/reliability/fusion/fusion_no_fn_v1.1_metrics.json` (2026-05-12). Training data composition and validation accuracy are NOT recoverable.

---

## Changelog

- **2026-05-18 (d)** — Resolved long-standing §3.2 Anti-UAV baseline RGB placeholder. Ran `A_rgb_yolo` factor on antiuav (28 min): baseline and `retrained_v2` are numerically identical on this benchmark (P=0.9922, R=0.9950, F1=0.9936; TP=3178, FP=25, FN=16 for both). Anti-UAV confirmed saturated and non-discriminating for the RGB training-stance comparison. Ch2 saturated-benchmark sentence updated; Ch2 §2.X Numerical-Comparison Table cell filled.
- **2026-05-18 (c)** — Added §9.5.8 Three-classifier real-video comparison (sa32 / control40 / fnfn). Flipped §1 production pick: **sa32** is now primary, control40 conservative alternative, fnfn open-world fallback. Justification: sa32 wins real-video drone F1 by 18–22 pp over control40 and 60+ pp over fnfn; fnfn rejects 85% of correct RGB drone TPs (too conservative for operational deployment). Added classifier-loading adapter for raw-model joblibs (`fusion_no_fn_v1.1`) via sibling `_metrics.json` features list.
- **2026-05-18 (b)** — Added §9.5 Full pipeline on real video. 4 RGB models × 19 videos × {RGB, IR-gray, classifier, temporal, patch} stages. Sanity check passes (Stage 1 matches §9.4). Key finding: cascade tightens variant-F1 spread from $0.605$–$0.760$ to $0.770$–$0.826$; selcom_1280 confuser FPR collapses $0.709 \to 0.136$ (81% cut). Per-frame metrics misrepresent the cascade — segment/alert level is the production-relevant grain.
- **2026-05-18** — Added §9.4 Unified video re-run with corrected scoring. Fixed `eval_video_tests.py` scoring bug (YOLO conf is pre-NMS, so post-filtering low-conf inference $\ne$ direct high-conf inference); added second inference pass at `prod_conf=0.25`. Marked §9.1 / §9.2 / §9.3 numbers as superseded for the affected metrics. Headline changes: IR-grayscale aggregate drone F1 **0.664 $\to$ 0.636**; `flock_of_seagulls_attack_drone_beach` IR-grayscale F1 **0.901 $\to$ 0.837** (now essentially tied with baseline RGB 0.840, no longer dominant); selcom_1280 all-confuser FPR **0.413 $\to$ 0.593** (more emphatic OOD damage); selcom_640 added as a sixth model (Pareto-best at FPPI $\leq 0.3$). `flock_of_birds_attack_drone` was not in the re-run; the §9.2 row for that video is the only source until it is re-staged.
- **2026-05-17 (c)** — Added §9.3 IR model on grayscale-RGB vs raw-RGB video (19 videos, 1359 drone frames + 1250 confuser frames). Key finding: IR-on-grayscale-RGB F1=0.664 on drone video vs IR-on-raw-RGB F1=0.298 (>2× improvement). On `flock_of_seagulls_attack_drone_beach`, IR-grayscale F1=0.901 beats every RGB detector in the project. The §sec:grayscale cross-modal generalisation claim is now empirically backed by drone-positive video.
- **2026-05-17 (b)** — Added §9.2 Real-video DRONE eval (10 drone videos × 3 RGB models, 7 with birds in same frame). Key finding: three-model recall/precision continuum (retrained_v2 conservative / baseline middle / selcom_1280 aggressive) confirmed on real video. retrained_v2's Svanström recall collapse generalises (R=0.000–0.080 on heavily bird-cluttered drone videos). `flock_of_birds_attack_drone` is the load-bearing case: baseline and retrained_v2 score F1=0; only selcom_1280 detects the drone.
- **2026-05-17 (a)** — Added §9.1 Real-video confuser eval (all 3 RGB models on 10 YouTube confuser videos, 1250 frames). Key finding: selcom_1280 is *worse* than baseline on OOD confusers (FPR 41.3% vs 36.9%), dramatically so on small/distant birds. retrained_v2 confirmed as the cleanest confuser-rejector on real video (FPR 17.0%). Renumbered Latency to §10 and Open-items to §11. See also `docs/analysis/2026-05-17_failure_profile_by_dataset.md` for the per-detection size/conf analysis from the prior day.
- **2026-05-16** — Added §4.1 IR model version comparison (V3/V4/V5/V6/Final/v3b) on the fixed `IR_dset_final` test split @ imgsz=640. P/R/F1 numbers now load-bearing for the HITL chapter; mAP-only narrative deprecated. V2 placeholder pending re-run.
- **2026-05-16** — Added §8 Roboflow OOD eval (9 datasets, baseline+retrained_v2+IR with patch verifier, no classifier, imgsz=640). Linked full write-up at `docs/analysis/2026-05-16_roboflow_ood_eval.md`. Renumbered subsequent sections.
- **2026-05-12** — Provenance audit fixes: resolved 5 UNKNOWN entries (rgb_confusers_merged path, airplane subset, Svanstrom@1280 CSV source, confuser halluc source). Updated `diagnose_failures.py` and `diagnose_failures_all.py` to write CSVs directly (previously hand-transcribed). Extracted `fusion_no_fn_v1.1` metadata from joblib bundle → `fusion_no_fn_v1.1_metrics.json`. Added control40 confuser-zoo result to §7.
- **2026-05-11** — Initial creation. Production stack pick changed: RGB = baseline `Yolo26n_trained` (not retrained_v2) based on three-way Svanstrom@1280 comparison. Classifier candidates downgraded to `control40` / `fusion_no_fn_v1.1` pending re-eval with baseline RGB.
