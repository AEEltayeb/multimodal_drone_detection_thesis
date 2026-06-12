# Datasets registry — the "ONE datasets file" (v1 draft)

**Date:** 2026-06-05. **Purpose:** single authoritative map of every dataset the thesis depends on →
physical path, size, splits, provenance, scoring rule, and which models/evals consume it. This is the
engineering source-of-truth that (a) the thesis appendix `app:datasets` summarises in prose, and (b) the
eventual physical `datasets/` bundle will be built from.

> Numbers sourced from: `docs/thesis_working.tex` appendix `app:datasets` (tables `tab:datasets`,
> `tab:ds_rgb_components`, `tab:ds_confusers`, `tab:ds_ir_components`, `tab:youtube_*`),
> `eval/config.yaml` (physical roots), `knowledge/eval_configs.csv` (eval splits + scoring).
> Every number below is from a read source file, not memory.

---

## 0. The scatter problem (why this file exists)

The data lives in **three disconnected locations**, and no single file maps them:

| Location | What's there |
|---|---|
| `G:/drone/*` | The big benchmark corpora (Anti-UAV, Svanström, composite RGB, IR_dset_final, cutpaste, selcom fine-tunes, Roboflow OOD) — **~60+ directories** |
| `C:/Users/User/Desktop/*` | At least one cutpaste set (`cutpaste_drone_v4`, also mirrored on G:) |
| In-repo `datasets/` + `ir_gui/demo_outputs` | YouTube video tests, confuser videos, gold-label sets, neg sets, IR youtube clips |

**Cleanup signal:** G:/drone has ~60+ dataset dirs; the thesis cites **7 canonical corpora + Roboflow + YouTube**.
The gap = predecessors, eval-subsets, and superseded variants. A full G: audit (tagging each dir
canonical / derived / superseded / unused) is the precursor to a physical "ONE datasets folder."

---

## 1. Master registry — thesis-canonical corpora

| Tag | Modality | Role | Full size | Eval split(s) used | Physical root | Scoring | Thesis ref | Redistributable |
|---|---|---|---|---|---|---|---|---|
| **rgb_dataset** | RGB | RGB-detector train + in-dist test | 172,022 (137,506/17,307/17,209) | `rgb_dataset_test` 507 (stride-34) | `G:/drone/dataset/dataset` | IoU@0.5 | tab:datasets, tab:ds_rgb_components | ✅ public (mixed) |
| **ir_dset_final** | IR | IR-detector train + in-dist test | 129,130 | `ir_final_640` 9,612 @640 | `G:/drone/IR_dset_final` | IoU@0.5 | tab:datasets, tab:ds_ir_components | ✅ research-open |
| **rgb_confusers_merged** | RGB | confuser hard-neg train source + OOD confuser-zoo test | 27,024 (21,784/2,607/2,633) | `confuser_zoo_1280` 2,633 (no GT) | merged — components on G: (see §2) | fire-rate (no GT) | tab:datasets, tab:ds_confusers | ✅ public |
| **svanstrom** (paired) | RGB+IR | **discriminating benchmark** (small drones, labelled confusers) | 28,710 (stride-3, 279 seqs) | `svan_iop_1280` 28,710; `_s9` 3,190; `svan_ir_iop_640` 1,000 | `G:/drone/svanstrom_paired` | **IoP@0.5 RGB / IoU IR** | tab:datasets, app §Svanström | ✅ public (Svanström 2021) |
| **antiuav** (RGBT) | RGB+IR | saturated benchmark / sanity floor | 85,374 paired | `antiuav_iou_email` 85,374; `_640_s5` 17,075 | `G:/drone/Anti-UAV-RGBT_yolo_converted/test` | IoU@0.5 | tab:datasets, app §Anti-UAV | ✅ public (Jiang 2021) |
| **selcom_cctv** | RGB | deployment-partner CCTV fine-tune surface | 2,076 (1,953+ / 123−) | `selcom_iop_1280` 311 (295 GT) | `G:/drone/_finetune_selcom_mixed_ft2` (+ft1/mixed_ft1/ft3) | IoP@0.5 | app §SelCom | ⛔ **PROPRIETARY** |
| **Roboflow OOD (9-set)** | RGB+IR | OOD generalisation audit (no training contribution) | RGB-drone 3,341 + 3 confuser + 3 IR | `roboflow_{rgb_drone,ir_drone,ir_confuser}_640` | `G:/drone/academy_roboflow_*`, `*.yolo26-*` (see §2) | IoU@0.5 | app §Roboflow | ✅ Roboflow community |
| **YouTube real-video** | RGB (+IR) | real-world drone-positive (10) + confuser-only (10) clips | drone 7,151→1,633 extr; confuser 36,949→1,270 extr | `video_drone_iop` 1,359; `video_confuser` 1,250 | repo `datasets/drone detection video tests(_v2)`, `datasets/confuser_videos`; IR → `ir_gui/demo_outputs` | IoP@0.5 / FPR | tab:youtube_*, app §YouTube | ⚠️ fair-use excerpts |

---

## 2. Path resolution detail (incl. unresolved)

- **rgb_confusers_merged** — a *merged* corpus; components (per `tab:ds_confusers`): Svanström Airplane/Heli/Bird
  splits, `New_Dataset.v1i.yolo26_airplane-drone-heli-rgb`, `Airplane.v1-...yolo26-roboflow-rgb`,
  `Helicopter-kaggle-dataset`, raihanrsd. **Merged-root path TBC** (not in `eval/config.yaml`; doc lives at
  `rgb_confusers_merged/dataset_documentation.md`). → resolve during G: audit.
- **selcom_cctv** — fine-tune variants at `G:/drone/_finetune_selcom_{ft1,mixed_ft1,mixed_ft2,mixed_ft3}`.
  Production weights: `Yolo26n_selcom_mixed_ft2_1280`. Raw source clip (~1:30 CCTV) path TBC.
- **Roboflow OOD** — candidate dirs on G: include `academy_roboflow_second_dataset_*`, `academy_roboflow_v4i`,
  `Drone Detection in Various Envir.v1i`, `Drone.v1i.yolo26-*`, `medium_large_drones_UAVs finder.v2i`,
  `Infrared_bird_drone_airplane_CBAM_TF-Net.v1i`, `bird.v1i.yolo26-*`, `final helicopter.v1i.*`. Exact
  9-set → dir mapping TBC; results cache `eval/results/roboflow_ood/summary.csv`.
- **antiuav_rgb_gray** (grayscale-deploy eval) — `G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB`, RGB fed
  grayscale to the IR model (`eval_model.py --grayscale`).
- **cutpaste (synthetic)** — `G:/drone/cutpaste_drone_v4` (also `C:/Users/User/Desktop/cutpaste_drone_v4`),
  `cutpaste_confusers_v4`, `cutpaste_drone2`, `cutpaste_drone_smoke`, `cutpaste_eval{,_v2,_v3}`.

---

## 3. Secondary / derived / training-only sets (not top-level corpora, but thesis-referenced)

| Tag | Modality | Used for | Path / source |
|---|---|---|---|
| `retrain_dataset` (183,751) | RGB | retrained_v2 training (rgb_dataset + 11,729 mined hard-negs) | derived from rgb_dataset + rgb_confusers_merged |
| `distill_train_3351` | RGB feats | V5 MLP verifier distillation (664 drone TP / 2,687 confuser FP; 261-D) | `eval/results/_v5_*` |
| `3way_300frame` | RGB | 3-way drone/confuser/bg classifier eval (100 each antiuav/svan/video) | derived |
| `cbam_thermal_valid` (180) | IR | held-out thermal bird/drone/plane (now in V5-IR mining → optimistic) | `G:/drone/Infrared_..._CBAM_...` |
| `ir_video_test_confuser`, `cutpaste_confusers_v4_ir` | IR | held-out thermal confuser halluc surfaces | repo + G: |
| IR predecessors `IR_dsetV5/V6/V9/V9b1`, `CST_selected_{10k,50k}`, `IR_dset_gold*` | IR | HITL V2–V6 lineage (do NOT delete — repro) | G: |

---

## 4. Gaps & reconciliations to resolve

1. **`tab:datasets` clip count vs appendix:** top table says "≈2,484 sampled / 9 drone clips"; appendix lists
   **10 drone + 10 confuser clips**. Already explained (appendix L2176): the "9 clips / 1,234 GT" figure
   excludes `flock_of_birds_attack_drone` from the corrected-scoring re-run. Line 374 "19-clip" =
   9 (re-run drone) + 10 confuser. **Not an error — but the registry should state the canonical count once.**
2. **Fig 5.1 dataset montage** (`fig:dataset_montage`, thesis L396) is a **`[PLACEHOLDER]`** — needs the 4×4
   representative-frame grid generated. (Independent of which classifier wins.)
3. **Merged/selcom/roboflow physical roots TBC** (§2) — resolve in the G: audit.
4. **No machine-readable registry yet** — this doc is prose+tables; see §5.

---

## 5. Recommendation — where this should live

**Now (this doc):** the reviewable v1 registry. Cross-checked against thesis + config.yaml + eval_configs.

**Next (on your green light):** formalise as a knowledge-system table **`knowledge/datasets.csv`** (one row per
corpus; `eval_configs.csv.dataset` becomes a foreign key into it). This is a **schema addition** → needs a
`knowledge/DECISIONS.md` entry + a small `kb.py` extension, per CLAUDE.md rule 5. Benefits: the thesis
`app:datasets` appendix + `tab:datasets` can then be **generated** from it (no drift), and it becomes the
canonical index for the cleanup.

**Later (backlog item "organized datasets/ folder"):** physical bundle — copy/symlink the 7 canonical corpora
(minus the proprietary SelCom raw frames) into one tree as a release artifact, driven by this registry.

---

## Delivered
- `docs/analysis/2026-06-05_datasets_registry.md` (this file) — v1 datasets registry covering 7 canonical
  corpora + Roboflow + YouTube, with physical paths, sizes, splits, scoring, thesis refs, and the G: scatter audit.

### Open follow-ups (not done here)
- Resolve §2 TBC paths via a full `G:/drone` audit (canonical/derived/superseded/unused tagging).
- Formalise as `knowledge/datasets.csv` (kb.py + DECISIONS.md) — **awaiting green light**.
- Generate Fig 5.1 dataset montage (placeholder).
