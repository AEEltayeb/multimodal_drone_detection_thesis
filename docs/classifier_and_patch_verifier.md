# Trust Classifier and Patch Verifier

This document describes the two post-detection learned components in the
fusion pipeline:

1. **Trust Classifier** — frame-level XGBoost over scene + best-box
   features that decides which modality (or both) to trust on each
   frame.
2. **Patch Verifier (a.k.a. Confuser Filter)** — per-detection 4-class
   CNN that vetoes individual boxes the YOLO detectors fire on
   airplanes / helicopters / birds.

They run sequentially: detectors → trust classifier → patch verifier →
temporal alert gate. Both are *post-hoc safety nets* over the YOLO
detectors; neither is itself a detector.

Production weights as referenced by `ir_gui/fusion_settings.json`:

| Component | Path |
|---|---|
| Trust classifier | `classifier/fusion_models/scene_aware_v3more_32feat/model.joblib` |
| RGB patch verifier | `classifier/runs/patches/confuser_filter4_rgb.pt` |
| IR patch verifier | `classifier/runs/patches/confuser_filter4_ir.pt` |

---

## 1. Trust Classifier

### What it does

Given the per-frame outputs of both YOLO detectors plus a set of scene
and best-box features, predict one of four trust labels:

| label | meaning |
|---|---|
| 0 | reject_both — neither modality is trustworthy |
| 1 | trust_rgb — believe RGB only |
| 2 | trust_ir  — believe IR only |
| 3 | trust_both — both modalities agree |

The selected label drives `_select_trusted` in `ir_gui/fusion/engine.py`,
which forwards only the trusted modality's detections downstream.

### Production model: `scene_aware_v3more_32feat`

- **Path**: `classifier/fusion_models/scene_aware_v3more_32feat/model.joblib`
- **Training tag**: `fusion_no_fn_v3more_no_det_signals`
- **Type**: XGBoost, 4-class softprob

### Why "scene-aware" / 32 features

This variant is the result of a deliberate ablation: detection-presence
shortcut features (`*_detected`, `*_n_dets`, `both_detect`,
`neither_detect`, `*_only_detect`) were **removed** from the feature
set. Without those flags, the classifier is forced to learn from scene
context and target geometry rather than from "did YOLO fire?". The
result is a model that disagrees with the detectors more often when
their evidence is weak — which is precisely the role we want it to
play.

The 32 features are:

| group | features |
|---|---|
| Detector confidences (4) | `rgb_max_conf`, `rgb_mean_conf`, `ir_max_conf`, `ir_mean_conf` |
| RGB scene stats (7) | `rgb_img_mean`, `rgb_img_std`, `rgb_img_dynamic_range`, `rgb_img_entropy`, `rgb_sky_ground_ratio`, `rgb_edge_density`, `rgb_blurriness` |
| IR scene stats (7)  | `ir_img_mean`, `ir_img_std`, `ir_img_dynamic_range`, `ir_img_entropy`, `ir_sky_ground_ratio`, `ir_edge_density`, `ir_blurriness` |
| RGB best-box geometry (7) | `rgb_best_log_bbox_area`, `rgb_best_aspect_ratio`, `rgb_best_pos_x`, `rgb_best_pos_y`, `rgb_best_dist_to_center`, `rgb_best_local_contrast`, `rgb_best_target_bg_delta` |
| IR best-box geometry (7) | `ir_best_log_bbox_area`, `ir_best_aspect_ratio`, `ir_best_pos_x`, `ir_best_pos_y`, `ir_best_dist_to_center`, `ir_best_local_contrast`, `ir_best_target_bg_delta` |

Notable absences: no `*_detected` flags, no `*_n_dets`, no agreement
flags, no `*_max_fn` (failure-model scores from a separate ablation),
no time-of-day features.

### Architecture / hyperparameters

XGBoost classifier (`classifier/reliability/fusion/train_fusion.py`):
- `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- `objective="multi:softprob"`, `num_class=4`
- `eval_metric="mlogloss"`, `tree_method="hist"`

### Training data

The fusion dataset is built by
`classifier/reliability/fusion/build_fusion_dataset.py` from cached YOLO
inference plus per-frame scene/target features computed on the original
images. Sources merged:

| dataset | role |
|---|---|
| Anti-UAV-RGBT (paired) | majority drone positives, paired RGB+IR |
| Svanström multi-sensor | real-thermal aerial scenes (drone + plane/heli/bird) |

(Earlier `youtube_aerial` rows from the v1.x classifiers are *not* part
of this 32-feature variant's training/test split — see "Per-dataset
metrics" below.)

The split is **sequence-level** (`extract_sequence_id` strips
`_f<digits>` and modality suffixes), enforced via `GroupShuffleSplit`
with `test_size=0.25`, `random_state=42`, and an explicit `assert`
against sequence leakage between train and test.

- **n_train**: 106 963
- **n_test**: 45 088

### Metrics

From `classifier/fusion_models/scene_aware_v3more_32feat/metrics.json`:

| metric | value |
|---|---:|
| accuracy | 0.9792 |
| F1 (macro) | 0.9493 |
| F1 (weighted) | 0.9786 |

Per-class one-vs-rest AUC:

| class | AUC |
|---|---:|
| reject_both | 0.9984 |
| trust_rgb | 0.9792 |
| trust_ir | 0.9921 |
| trust_both | 0.9913 |

Per-dataset on the test split:

| dataset | n | acc | F1 (macro) | F1 (weighted) |
|---|---:|---:|---:|---:|
| antiuav_test | 28 463 | 0.983 | 0.947 | 0.982 |
| antiuav_val | 10 319 | 0.974 | 0.905 | 0.972 |
| svanstrom | 6 306 | 0.973 | 0.792 | 0.968 |

Top features by importance:

| feature | importance |
|---|---:|
| `rgb_best_log_bbox_area` | 0.240 |
| `ir_best_log_bbox_area` | 0.184 |
| `rgb_max_conf` | 0.130 |
| `ir_best_pos_y` | 0.092 |
| `ir_best_aspect_ratio` | 0.075 |
| `rgb_mean_conf` | 0.059 |
| `ir_best_pos_x` | 0.039 |
| `ir_max_conf` | 0.019 |
| `rgb_blurriness` | 0.014 |
| `rgb_best_pos_y` | 0.014 |

The two best-box-area features dominate (combined ~42% of importance) —
the classifier mostly arbitrates by "how big and well-shaped is the
candidate target relative to typical drones in this scene?" rather than
by raw confidence numbers.

Cheap rule-based baselines on the same test split (from `metrics.json`):

| policy | acc | F1 (macro) |
|---|---:|---:|
| `always_ir` | 0.193 | 0.376 |
| `higher_conf` | 0.940 | 0.819 |
| `both_or_ir` | 0.940 | 0.819 |
| **scene_aware_v3more_32feat** | **0.979** | **0.949** |

The learned model gives ~4pp accuracy and ~13pp F1-macro over the best
rule baseline, despite being denied the very flags those baselines lean
on (`*_detected`).

---

## 2. Patch Verifier (Confuser Filter)

### What it does

For every detection box that survives the trust classifier, crop a
context-padded patch around the box and run a small CNN that scores it
across **4 classes**: airplane, helicopter, bird, other. The
`_confuser_probs` method computes P(confuser) per crop:

```
P(confuser) = softmax[argmax]   if argmax ∈ {airplane, helicopter, bird}
            = 0.0               otherwise (argmax = other)
```

Veto rule:

```
P(confuser) ≥ patch_threshold  AND  predicted class ∈ suppressed_classes
```

`patch_threshold` defaults to **0.9** in `ir_gui/fusion_settings.json`
(the web API's config) and to **0.70** in `ir_gui/settings.json` (the
PySide app's config). The `FusionEngine.__init__` code default is
**0.70**. The two config files target different UIs but share the same
backend.

Anything else passes — including novel/OOD inputs (drone variants the
verifier has never seen drift toward "other" by design).

The verifier is per-modality: separate weights for RGB
(`confuser_filter4_rgb.pt`) and IR (`confuser_filter4_ir.pt`). The IR
verifier is only run when the IR feed is real thermal (or when the
operator explicitly enables it on grayscale via
`grayscale_run_ir_filter`).

> **Version note**: the current production weights are the v2 retrain
> (`confuser_filter4_{rgb,ir}.pt`, dated 2026-04-30). The v1 weights
> (dated 2026-04-20) are kept alongside as
> `confuser_filter4_{rgb,ir}_v1_backup.pt` for ablation. The v2 retrain
> uses a substantially expanded manifest — most notably more
> hard-negative confuser crops mined from the production YouTube
> sources and additional drone "other" crops from the in-house IR/RGB
> datasets and Anti-UAV. The metrics in this section all describe the
> shipped v2 weights.

### Architecture

- Backbone: `mobilenet_v3_small` (ImageNet-pretrained), final classifier
  head replaced with `Linear(in_features, 4)`.
- Input: 224×224 BGR→RGB, normalised with ImageNet mean/std
  (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
- Crop: `_crop_with_context` in `patch_verifier.py` expands the YOLO box
  by `pad_frac=0.5` around the centre and pads to a minimum side of 24
  px. The expansion produces a **square** crop centred on the detection
  with side = `max(bw, bh) * (1.0 + 2.0 * pad_frac)`. Context matters
  — a tight crop on a 12-px drone is too sparse for the CNN.
- Optional Mahalanobis OOD head: penultimate-layer features (1024-d) +
  shrinkage-regularised inverse covariance estimated on the training
  set, threshold at the 99th percentile of in-distribution distances.
  At inference, OOD-flagged crops are excluded from the veto vote
  (`_in_dist_max` in `engine.py`) — the verifier defaults to "no opinion"
  rather than triggering a veto on inputs it has never seen.

### Extended `patch_veto` API

The production `patch_veto` method in `fusion/engine.py` supports
additional parameters beyond the basic 5-argument signature:

```python
def patch_veto(self, label, rgb_dets, ir_dets, rgb_bgr, ir_bgr,
               ir_is_real_thermal: bool,
               ir_verifier_enabled: Optional[bool] = None,
               skip_ir_ood_gate: bool = False,
               suppressed_classes: Optional[set] = None):
```

| Parameter | Purpose |
|---|---|
| `ir_verifier_enabled` | Override the `ir_is_real_thermal` gate. When explicitly set, forces the IR verifier on (`True`) or off (`False`) regardless of the thermal flag. Used in grayscale mode to enable the IR verifier on non-thermal input. |
| `skip_ir_ood_gate` | When `True`, disables the Mahalanobis OOD gate for IR crops. Required in grayscale mode because the OOD calibration was fitted on real thermal data — every grayscale crop would be flagged OOD otherwise. |
| `suppressed_classes` | A `set` of class names (e.g. `{"helicopter", "airplane"}`) that should trigger a veto. If `None` (default), all confuser classes trigger. Enables selective per-class suppression from the UI. |

### Per-class suppression

The PySide app exposes three toggles: `suppress_helicopter`,
`suppress_airplane`, `suppress_bird` (all default `True`). These are
collected into a `suppressed_classes` set by
`_build_suppressed_classes()` in `pyside_engine.py` and passed to
`patch_veto()`. This allows the operator to, for example, disable bird
suppression in environments where the verifier over-vetoes.

### Grayscale mode handling

In grayscale mode, the `patch_veto` call uses:
- `ir_is_real_thermal=False` — honest about non-thermal input
- `ir_verifier_enabled=True` via `grayscale_run_ir_filter` setting —
  forces the IR verifier to run despite the non-thermal flag
- `skip_ir_ood_gate=True` — disables OOD gate since the Mahalanobis
  calibration is thermal-only

This ensures the IR verifier can still veto confusers on grayscale input
without every crop being rejected as OOD.

### Training data (current v2 manifest)

Source: `classifier/runs/patches/manifest.csv` (44 600 rows total),
built by `extract_patches.py` and significantly expanded by
`extract_patches_v2.py` for the v2 retrain.

| modality | airplane | helicopter | bird | drone | background | total |
|---|---:|---:|---:|---:|---:|---:|
| RGB | 1 952 | 1 166 | 879 | 11 111 | 8 442 | 23 550 |
| IR  | 1 581 | 1 326 | 1 000 | 11 327 | 5 816 | 21 050 |

`drone` and `background` both map to the "other" class at training time
(`manifest_category_to_class` in `train_confuser_4class.py`). The
manifest pulls from:

- **Drone crops** (other): YOLO-labelled drone boxes from a custom IR
  drone dataset (`G:/drone/IR_dset_final`, IR modality), a custom RGB
  drone dataset (`G:/drone/dataset/dataset`, RGB modality), and
  Anti-UAV-RGBT drone boxes (`G:/drone/Anti-UAV-RGBT_yolo_converted`,
  via `train_confuser_filter.py:extract_antiuav_crops`).
- **Confuser crops**: hard-negative mining — run YOLO on YouTube
  confuser videos (airplane / helicopter / bird footage) and crop each
  detection. IR sources from thermal YouTube clips in
  `ir_gui/demo_outputs/yt_*.mp4`; RGB from a custom YouTube collection
  at `D:/Downloads/youtube_classifier_videos/*.mp4`. **This is the key
  trick**: the verifier is trained specifically on the YOLO false
  positives the production pipeline produces, not on clean web crops.
  The v2 retrain expanded this mining significantly.
- **Background crops**: random crops with no label overlap, also into
  "other".

### Training procedure (`train_confuser_4class.py`)

- **Split**: by video, 80/20 train/val (`sequence_split`). Same video
  never appears in both halves.
  - RGB: `n_train=20 276`, `n_val=3 274`
  - IR:  `n_train=18 020`, `n_val=3 030`
- **Loss**: `CrossEntropyLoss` with **inverse-frequency class weights**
  computed from the training split. With drone+background dominating,
  this lifts the per-batch weight on airplane / helicopter / bird.
- **Optimiser**: AdamW, `lr=3e-4`, `weight_decay=1e-4`.
- **Schedule**: cosine annealing over the training run. Default 12
  epochs, batch size 64.
- **Augmentation (train only)**: resize 224, random horizontal flip,
  ColorJitter(0.2, 0.2, 0.2, 0.05). Eval is resize-only.
- **Selection**: best-val-accuracy checkpoint kept, then re-evaluated to
  produce the final metrics block.

### OOD calibration (`calibrate_confuser_ood.py`)

Run after training. Extracts 1024-d penultimate features for the entire
training manifest, fits a Gaussian (shrinkage inverse covariance), and
saves `{mean, inv_cov, threshold}` to `*_ood.npz`. Threshold is the 99th
percentile of in-distribution Mahalanobis distances:

| modality | threshold | n | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| RGB | 47.70 | 8 191 | 30.10 | 41.83 | 47.70 |
| IR  | 49.02 | 8 736 | 30.27 | 42.24 | 49.02 |

Crops above the threshold are tagged OOD by `predict_boxes_with_ood`;
the engine's `_in_dist_max` excludes them from the veto vote.

### Metrics (current v2 weights)

**RGB verifier** — `classifier/runs/patches/confuser_filter4_rgb_metrics.json`
(best val accuracy = 0.978):

Per-class on val (4-way classification):

| class | n | precision | recall |
|---|---:|---:|---:|
| airplane | 322 | 0.919 | 0.957 |
| helicopter | 256 | 0.988 | 0.996 |
| bird | 130 | 0.779 | 0.838 |
| other | 2 566 | 0.996 | 0.986 |

Reject-rule sweep (veto if `argmax ∈ confusers AND prob ≥ thr`):

| thr | veto-precision | veto-recall | drone pass-acc |
|---:|---:|---:|---:|
| 0.5 | 0.952 | 0.983 | 0.986 |
| 0.6 | 0.956 | 0.977 | 0.987 |
| 0.7 | 0.960 | 0.959 | 0.989 |
| 0.8 | 0.963 | 0.946 | 0.990 |
| **0.9 (ship)** | **0.969** | **0.917** | **0.992** |

**IR verifier** — `classifier/runs/patches/confuser_filter4_ir_metrics.json`
(best val accuracy = 0.938):

Per-class on val:

| class | n | precision | recall |
|---|---:|---:|---:|
| airplane | 368 | 0.884 | 0.706 |
| helicopter | 232 | 0.888 | 0.987 |
| bird | 231 | 0.823 | 0.766 |
| other | 2 199 | 0.962 | 0.989 |

Reject-rule sweep:

| thr | veto-precision | veto-recall | drone pass-acc |
|---:|---:|---:|---:|
| 0.5 | 0.971 | 0.893 | 0.990 |
| 0.6 | 0.976 | 0.868 | 0.992 |
| 0.7 | 0.979 | 0.847 | 0.993 |
| 0.8 | 0.986 | 0.828 | 0.996 |
| **0.9 (ship)** | **0.991** | **0.792** | **0.997** |

### Notes on the operating point

At `patch_threshold = 0.9`, both verifiers prioritise **veto precision
over veto recall**. They will only block a detection when they're very
sure it's a confuser, so genuine drones almost always survive the
verifier (RGB 99.2%, IR 99.7% pass-through on the "other" class). The
trade-off is that ~8% (RGB) / ~21% (IR) of true confusers slip past at
the per-frame level — which is why the temporal alert-gate
(`should_suppress_alert` in `ir_gui/fusion/temporal.py`) layers an
additional window-level rule on top of the per-frame veto.

The IR verifier's lower per-class precision on airplane/bird reflects
the harder thermal problem: small low-contrast blobs at altitude can
look near-identical between drones and birds. The OOD gate is the
release valve for genuinely novel-looking thermal targets.

### Runtime modes

The confuser filter can operate in two modes at runtime, controlled by
the `confuser_filter_history` setting:

#### History mode (`confuser_filter_history = True`)

P(confuser) is computed **every inference frame** and accumulated in a
per-modality sliding window (`add_confuser_prob`). When the temporal
alert window fills and an alert is about to fire, the accumulated
history is checked via `should_suppress_alert()`. This gives the system
multiple frames of evidence before deciding whether to suppress.

The temporal vote has three configurable modes
(`confuser_suppress_mode` in `temporal.py`):

| mode | rule |
|---|---|
| `primary_only` (default) | ≥30% (min 2) of valid frames exceed threshold |
| `primary_and_avg` | primary OR average P(confuser) ≥ threshold × 0.7 |
| `any_above` | any single frame in history exceeds threshold |

#### Alert-gate mode (`confuser_filter_history = False`, default)

The CNN is **not** run every frame. Instead, detections accumulate in
the temporal window without confuser checking. Only when the temporal
window fills and an alert is about to fire does the engine run
`patch_veto` on the current frame's detections as a one-shot gate:

```python
if fe.use_patch_verifier and (rt.alert_active or it.alert_active):
    _, gate_rgp, gate_irp, _ = fe.patch_veto(...)
    if rt.alert_active and gate_rgp and max(gate_rgp) >= patch_thr:
        rt.alert_active = False
        rt.confuser_suppressed = True
```

This is cheaper (CNN runs only at alert edges, not every frame) but
provides only single-frame evidence.

### Cascade order

The `FusionEngine` supports two cascade orders
(`cascade_order` setting):

| order | flow |
|---|---|
| `filter_then_classifier` (default) | YOLO → patch filter (per-box veto) → classifier on surviving dets |
| `classifier_then_filter` | YOLO → classifier on all dets → patch veto (per-modality trust revoke) |

In `filter_then_classifier` mode, the verifier runs via
`_filter_dets_by_patch()` which drops individual boxes before the
classifier sees them. In `classifier_then_filter` mode, the verifier
runs via `patch_veto()` after the classifier and can only revoke an
entire modality's trust.

### Consumers

Two GUIs consume the patch verifier:

| GUI | Backend | Config file |
|---|---|---|
| Web (React + FastAPI) | `ir_gui/api.py` | `ir_gui/fusion_settings.json` |
| PySide6 (desktop) | `ir_gui/pyside_engine.py` | `ir_gui/settings.json` |

The PySide engine is the more feature-complete consumer: it supports
both runtime modes, per-class suppression toggles, configurable
suppress modes, and the extended `patch_veto` API. The web API backend
currently uses only history mode with shared confuser probabilities
(max of RGB and IR fed to both temporal states).

---

## Source Datasets

Below is the full provenance of every dataset used to train or evaluate
the trust classifier and patch verifier.

### Anti-UAV-RGBT
- **Origin**: Anti-UAV challenge benchmark (CVPR / ICCV Anti-UAV
  Workshop series), distributed as paired visible (RGB) + thermal (IR)
  tracking sequences with bounding-box annotations.
- **Local path**: `G:/drone/Anti-UAV-RGBT_yolo_converted/` (re-converted
  from the original tracking annotations into YOLO-format detection
  labels per frame).
- **Used by**:
  - Trust classifier — primary source of paired drone-positive frames
    (antiuav_test / antiuav_val splits in
    `fusion_dataset_v3more.csv`).
  - Patch verifier — drone "other" crops via
    `train_confuser_filter.py:extract_antiuav_crops` (max 1500/split
    /modality, every 30th frame).
  - Inference caches — `antiuav_test_{rgb,ir}.json`,
    `antiuav_val_{rgb,ir}.json`.

### Svanström multi-sensor drone dataset
- **Origin**: F. Svanström, F. Alonso-Fernandez, C. Englund, "A dataset
  for multi-sensor drone detection" (Data in Brief / IEEE; commonly
  cited as the *Svanström dataset*). Paired thermal-IR + visible (RGB)
  + audio captures of drones, airplanes, helicopters and birds, with
  YOLO-style annotations. Real thermal sensor data — not
  grayscale-replicated.
- **Local path**: `G:/drone/svanstrom_paired/{IR,RGB}/` with parallel
  `images/` and `labels/` directories per modality.
- **Used by**:
  - Trust classifier — real-thermal aerial scenes, important source of
    confuser negatives (`svanstrom` rows in `fusion_dataset_v3more.csv`).
  - Patch verifier — confuser crops (airplane / helicopter / bird)
    extracted via `convert_svanstrom_paired.py` and `extract_patches.py`.
  - Inference caches — `svanstrom_{rgb,ir}.json`.

### YouTube aerial confusers (custom-curated)
- **Origin**: manually selected YouTube clips of airplanes, birds and
  helicopters in both visible (RGB) and thermal (IR) modalities. Used
  as a "production-distribution" hard-negative source — videos that
  resemble realistic confuser scenarios the deployed system might face,
  not lab-clean crops. *Not redistributable; collected for research
  use.*
- **Local paths**:
  - RGB: `D:/Downloads/youtube_classifier_videos/*.mp4`
    (frame-feature CSV built by `process_youtube_videos.py`).
  - IR: `ir_gui/demo_outputs/yt_*.mp4` (real-thermal YouTube clips).
- **Categories** (from `youtube_aerial_summary.json`): airplane,
  helicopter, bird in RGB and IR.
- **Used by**:
  - Trust classifier — *not* in the training/test split for the
    32-feature production model; used in evaluation-only sweeps and in
    earlier v1.x classifiers.
  - Patch verifier — primary source of mined confuser crops via
    `extract_patches_v2.py`. The v2 retrain expanded this pool.
  - Threshold/size sweeps — pure-negative test pool in
    `fusion_youtube_rows.csv`.

### Custom IR drone dataset (`IR_dset_final`)
- **Origin**: in-house thermal drone dataset assembled for this thesis
  (custom captures + curated public material), YOLO-format labels.
- **Local path**: `G:/drone/IR_dset_final/{train,test}/`.
- **Used by**:
  - Patch verifier — drone "other" crops (`extract_patches_v2.py`).
  - IR YOLO finetune supervision (outside the scope of this doc — see
    the IR YOLO model card).
  - Inference caches — `ir_dset_final_{val,test}.json`.

### Custom RGB drone dataset (`G:/drone/dataset/dataset`)
- **Origin**: in-house RGB drone dataset used for the RGB YOLO
  finetune, repurposed for confuser-filter drone crops.
- **Local path**: `G:/drone/dataset/dataset/`, YOLO-format labels.
- **Used by**:
  - Patch verifier — drone "other" crops (`extract_patches_v2.py`).
  - Inference caches — `rgb_dataset_{val,test}.json`.

### Auxiliary evaluation-only datasets

These are **not** in any training pipeline — they appear only in
evaluation caches and ablations:

- **VTUAV** — `vtuav_detections.json` /
  `vtuav_detections_flipped.json`. Public Visible-Thermal UAV tracking
  benchmark, used as an out-of-training generalisation check.
- **CST-AntiUAV** — `cst_antiuav_test.json`. A separate Anti-UAV
  variant test split (CST-AntiUAV / "challenging single-target
  thermal"), used for stress-testing IR FP rates. Excluded from the
  primary IR min-size sweep at the user's request because it
  dominates FP counts.

### Quick reference: where each dataset shows up

| Dataset | Trust classifier (train) | Patch verifier (train) | Inference cache |
|---|:---:|:---:|:---:|
| Anti-UAV-RGBT | ✅ pos (paired) | ✅ drone | ✅ |
| Svanström | ✅ neg (aerial) | ✅ confuser | ✅ |
| YouTube confusers (RGB) | — | ✅ confuser | ✅ |
| YouTube confusers (IR) | — | ✅ confuser | ✅ |
| IR_dset_final (custom IR) | — | ✅ drone | ✅ |
| RGB drone (custom) | — | ✅ drone | ✅ |
| VTUAV | — | — | ✅ (eval only) |
| CST-AntiUAV | — | — | ✅ (eval only) |

---

## 3. Runtime Feature Extraction — Changes and Optimisations

This section documents changes made to how the trust classifier's input
features are computed at inference time. The trained model and its
weights are unchanged; only the runtime extraction pipeline was updated.

### Background: the 32-feature vector at inference

The trust classifier expects a fixed 32-dimensional feature vector per
frame:
- **4 confidence features**: `rgb_max_conf`, `rgb_mean_conf`,
  `ir_max_conf`, `ir_mean_conf` — from YOLO outputs, computed every
  frame, cheap.
- **14 scene features** (7 per modality): `img_mean`, `img_std`,
  `img_dynamic_range`, `img_entropy`, `sky_ground_ratio`,
  `edge_density`, `blurriness` — global pixel-intensity statistics
  over the frame, computed from a grayscale conversion.
- **14 best-box geometry features** (7 per modality):
  `log_bbox_area`, `aspect_ratio`, `pos_x`, `pos_y`,
  `dist_to_center`, `local_contrast`, `target_bg_delta` — from the
  highest-confidence detection box, computed every frame, cheap.

All features are modality-agnostic scalar statistics; the classifier
never receives raw pixels or color channels.

### Why scene features are computed on grayscale (not color)

The 7 scene features are pixel-intensity statistics — they require a
single-channel image by definition. The RGB side computes them on a
grayscale conversion of the color frame (`cvtColor(rgb, BGR2GRAY)`),
mirroring the training pipeline exactly (`build_fusion_dataset.py`
does the same). Color information is not discarded — it is already
encoded in the YOLO confidence scores, which are top-2 importance
features. Adding color statistics to the trust classifier would
duplicate information that YOLO has already distilled into its
confidence output.

### Grayscale mode: "fake IR" input

In grayscale mode (single RGB camera, no IR hardware), the IR YOLO
model receives a grayscale-replicated 3-channel frame. The trust
classifier receives the same grayscale image for both the RGB-side
and IR-side scene features — an unavoidable consequence of having one
source image. The per-box geometry and confidence features still
differ between modalities (RGB YOLO sees the color frame; IR YOLO
sees the grayscale version), preserving the most important classifier
signals. Grayscale mode is intended for demo use only.

### `compute_global_features` — new parameters

**File**: `ir_gui/fusion/features.py`

Two new explicit parameters added:

| parameter | default | meaning |
|---|---|---|
| `modality` | `"rgb"` | `"rgb"` or `"ir"` — selects the correct training-set mean values for cold-cache fill. **Must be passed explicitly by all callers.** The previous heuristic (threshold on `img_mean < 91.0`) was removed. |
| `max_h` | `480` | Maximum height (px) before downsampling the grayscale image for feature computation. Was previously hardcoded as `_MAX_H = 480`. Now configurable at runtime. |

All live GUI callers (`engine.py`, `flet_app/engine.py`,
`fusion_app.py`) updated to pass `modality="rgb"` and `modality="ir"`
explicitly.

### `GlobalFeatureCache` — strided scene-feature caching

**File**: `ir_gui/fusion/features.py`

New class added. Scene globals change on the timescale of camera
motion (seconds), not target motion (per-frame). Computing them every
inferred frame is wasteful; mean-filling lies to the model.
`GlobalFeatureCache` caches real, full-quality globals and recomputes
only when needed.

```python
cache = GlobalFeatureCache(stride=5, max_h=480, scene_cut_delta=15.0)
features = cache.get(img_gray, modality="rgb")  # cache-hit = ~0 ms
```

**Recompute triggers** (per modality, tracked independently):
1. **Cold cache** — first call after `reset()`.
2. **Stride boundary** — every Nth call (`frame_idx % stride == 0`).
3. **Scene cut** — if `|current_img_mean − cached_img_mean| > scene_cut_delta`, forced recompute between stride boundaries. This handles camera cuts and rapid scene changes without waiting for the next stride boundary.

**Parameters**:

| parameter | default | meaning |
|---|---|---|
| `stride` | 5 | Recompute every Nth inferred frame. 1 = every frame (original behaviour). Exposed as `feature_stride` in the settings dialog. |
| `max_h` | 480 | Downsample target height; passed through to `compute_global_features`. Exposed as `feature_max_height`. |
| `scene_cut_delta` | 15.0 | `img_mean` delta threshold for forced recompute. |

**Speed (synthetic 1080p grayscale, n=200 reps)**:

| mode | mean ms |
|---:|---:|
| `compute (max_h=720)` | 71 |
| `compute (max_h=480, default)` | 35 |
| `compute (max_h=320)` | 17 |
| `compute (max_h=240)` | 12 |
| **cache hit** | **0.0002** |

Amortised cost at stride=5, max_h=480: ~7 ms/inferred frame.

### Accuracy impact of strided caching

**File**: `classifier/bench_feature_cache.py`

Evaluated by simulating temporal stride on the cached
`fusion_dataset_v3more.csv` (preserving within-sequence temporal
ordering, scene-cut probe active at delta=15). Results on 15 206 rows
(stride-10 subsample of the full 152 k, pooled across AntiUAV-test/val
and Svanström):

| stride | accuracy | Δ accuracy |
|---:|---:|---:|
| 1 (every frame) | 0.9916 | — |
| 2 | 0.9899 | −0.17 pp |
| **5 (default)** | **0.9882** | **−0.34 pp** |
| 10 | 0.9876 | −0.40 pp |
| 20 | 0.9873 | −0.43 pp |

Per-class F1 at stride=5 vs stride=1:

| class | stride 1 | stride 5 | Δ F1 |
|---|---:|---:|---:|
| reject_both | 0.995 | 0.992 | −0.003 |
| trust_rgb | 0.937 | 0.908 | −0.029 |
| trust_ir | 0.978 | 0.969 | −0.009 |
| trust_both | 0.995 | 0.994 | −0.001 |

`trust_rgb` is the most sensitive class (depends most on scene context
to distinguish a single-modality RGB detection from noise). Svanström
test rows see the largest per-source accuracy drop (−2.8 pp at
stride=5) vs AntiUAV (−0.4 pp), because Svanström contains more
diverse scene transitions.

*Note*: absolute numbers are inflated vs the test-only `metrics.json`
figure (0.979) because this simulation pools train + val + test rows.
Relative drops between strides are accurate.

### Why the old feature-mode dropdown was removed

A previous `classifier_features` setting exposed three modes:

- `all` — compute every frame
- `skip_expensive` — compute cheap globals only, mean-fill the rest
- `detections_only` — mean-fill all globals

Mean-filling is strictly worse than real cached values because it
substitutes a global training-distribution average for the actual
current scene. In particular:

- `skip_expensive` mixes real and mean-filled values in the same
  feature vector, breaking learned correlations between features. On
  Svanström it caused −0.87 pp accuracy vs −0.34 pp for stride=5.
- `detections_only` fills 14 of 32 features with constants, causing
  `trust_rgb` precision to drop 0.169 points vs 0.071 at stride=5.

`GlobalFeatureCache` supersedes both modes: it delivers real feature
values at near-zero marginal cost per cached frame, without
distributional shift. The dropdown was removed from the settings
dialog and replaced with `feature_stride` (int) and
`feature_max_height` (dropdown).

### Settings added (`ir_gui/flet_app/settings_dialog.py`)

| key | type | default | meaning |
|---|---|---|---|
| `feature_stride` | int | 5 | Recompute scene globals every N inferred frames. 1 = original per-frame behaviour. |
| `feature_max_height` | int (dropdown) | 480 | Downsample height for scene-feature images. Options: 240 / 320 / **480** / 720 / 1080. 480 matches the training pipeline. |

Both are in the **Detection** section of the settings dialog.
`feature_max_height` is a dropdown (same pattern as `imgsz`) and is
stored as int via `CHOICE_INT_KEYS`.

### `FusionEngine` — cache integration

**File**: `ir_gui/fusion/engine.py`

`FusionEngine.__init__` accepts two new kwargs:

```python
FusionEngine(
    ...
    feature_stride=5,       # passed to GlobalFeatureCache
    feature_max_height=480, # passed to GlobalFeatureCache
)
```

`self.feature_cache = GlobalFeatureCache(stride, max_h)` is created at
init time and shared across both the paired and grayscale inference
paths. `extract_features` now calls `self.feature_cache.get(img_gray,
modality)` instead of `compute_global_features` directly.

On every new video stream (`pyside_engine.py:start()`), the engine
calls `feature_cache.configure(...)` to pick up any settings changes
then `feature_cache.reset()` to clear stale values from the previous
video.
