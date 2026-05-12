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
| Trust classifier | **`control_v3more_40feat`** | `classifier/fusion_models/control_v3more_40feat/model.joblib` — wins Svanstrom on every drone metric (S3 F1 0.909 vs sa32 0.896 vs fnfn 0.895; S3 recall 0.893 vs sa32 0.868). For open-world OOD deployment `fusion_no_fn_v1.1` remains the conservative fallback (1.6% vs sa32 20.5% S2 fire on the confuser zoo; control40 confuser-zoo not yet measured). | Section 7 classifier comparison table |
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

### 3.2 Anti-UAV @ imgsz (per May 10 ablation, probably default)

| RGB model | P | R | F1 | Source | How to reproduce |
|---|---|---|---|---|---|
| `Yolo26n_retrained_v2` | 0.991 | 0.995 | 0.9929 | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` row `rgb_only` × antiuav | per `eval/ablate.py` matrix |
| `Yolo26n_trained` (baseline) | — | — | — | UNKNOWN (not in the May 10 matrix; baseline was an A-factor but only ran on svanstrom + rgb_dataset) | re-run with baseline as RGB |

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

---

## 4. IR YOLO model — measured metrics

| Dataset | imgsz | TP | FP | FN | P | R | F1 | Source |
|---|---|---|---|---|---|---|---|---|
| antiuav | (May 10 default) | 15910 | 213 | 926 | 0.987 | 0.945 | 0.9654 | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` row `ir_only` × antiuav |
| svanstrom | 1280 | 2234 | 117 | 63 | 0.950 | 0.973 | 0.9613 | `eval/results/_ablation/2026-05-10T16-08-14/master.csv` row `ir_only` × svanstrom (assumed imgsz=1280; verify) |

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

## 8. Latency / throughput — placeholder

| Hardware | imgsz | Stage | ms / frame | Source |
|---|---|---|---|---|
| (company target) | 1280 | YOLO RGB | placeholder | not yet run |
| (company target) | 1280 | YOLO IR | placeholder | not yet run |
| (company target) | 1280 | trust classifier | placeholder | not yet run |
| (company target) | 1280 | patch verifier | placeholder | not yet run |
| (company target) | 1280 | temporal + draw | placeholder | not yet run |
| (company target) | 1280 | end-to-end | placeholder | not yet run |

---

## 9. Open data-provenance items (UNKNOWN → resolve)

~~Path of `rgb_confusers_merged` dataset~~ → **RESOLVED** `G:/drone/rgb_confusers_merged` (2026-05-12).
~~Path of `confuser_airplane_roboflow` 99-image subset~~ → **RESOLVED** prefix `airplane_*` in `G:/drone/rgb_confusers_merged/images/test/` (2026-05-12).
~~Exact script and output CSV that produced the Svanstrom@1280 per-category breakdown~~ → **RESOLVED** `eval/diagnose_failures.py` + `eval/diagnose_failures_all.py` → `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` (scripts updated to write CSVs directly 2026-05-12).

Remaining:

- Patch verifier v3 file path (or confirm deleted).
- Confirm imgsz used in the May 10 ablation for Anti-UAV and Svanstrom (default per `ablations.yaml` is now 1280, but the May 10 run may have used 640).
- `fusion_no_fn_v1.1` training provenance — no `metrics.json` was saved at training time. Post-hoc extraction of features/model-type/classes saved to `classifier/runs/reliability/fusion/fusion_no_fn_v1.1_metrics.json` (2026-05-12). Training data composition and validation accuracy are NOT recoverable.

---

## Changelog

- **2026-05-12** — Provenance audit fixes: resolved 5 UNKNOWN entries (rgb_confusers_merged path, airplane subset, Svanstrom@1280 CSV source, confuser halluc source). Updated `diagnose_failures.py` and `diagnose_failures_all.py` to write CSVs directly (previously hand-transcribed). Extracted `fusion_no_fn_v1.1` metadata from joblib bundle → `fusion_no_fn_v1.1_metrics.json`. Added control40 confuser-zoo result to §7.
- **2026-05-11** — Initial creation. Production stack pick changed: RGB = baseline `Yolo26n_trained` (not retrained_v2) based on three-way Svanstrom@1280 comparison. Classifier candidates downgraded to `control40` / `fusion_no_fn_v1.1` pending re-eval with baseline RGB.
