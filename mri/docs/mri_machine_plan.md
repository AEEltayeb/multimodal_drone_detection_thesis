# Model MRI — plan for a reusable detector "brain statistics" tool

**Date:** 2026-05-30
**Goal:** Turn the one-off V5 distillation/visualization scripts into a single,
methodical tool: *attach a YOLO model + a positive dataset + one or more negative
datasets, and it runs the detector, extracts internal features, computes the
brain statistics (PCA / LDA / ANOVA / separability), generates all plots,
optionally trains an MLP confuser-classifier, and emits a verdict on whether the
model needs a classifier for FP reduction at all.*

This becomes the 4th headline contribution alongside the YOLO models, the label
reviewer, and the PySide GUI. The framing is an **"MRI machine"** for a detector:
it images the model's internal feature space and reports a diagnosis.

---

## 1. What already exists (and what's wrong with it)

| Capability | Current script | Limitation |
|---|---|---|
| FPN feature extraction via Detect-head hook (`p3/p4/p5`), ROI pool, 5 metadata | `eval/distill_v5_p3p5_ft4.py` | Hardcoded `SOURCES` registry, absolute `G:/drone/...` paths, V5-specific quotas |
| MLP/LogReg/RF/XGB wrappers, focal loss, sample weights, 5-fold CV, `mlp_v5.pt` save | `eval/distill_v5_p3p5_ft4.py` | Tied to the 517-D V5 schema and the fixed source list |
| PCA (p3/p5/fused), LDA hist+scatter, ANOVA F-rank, class heatmap, mean signature, top-neuron KDE, per-layer ANOVA box | `scripts/visualize_v5_features.py` | Reads only the one cached `training_data.npz`; titles/paths hardcoded |
| Multi-class LDA (drone/bird/airplane/heli) | `scripts/visualize_v5_lda_multiclass.py` | Re-mines fixed Svanström + RGB-video paths |
| P/R/F1 + IoU/IoP scoring | `eval/metrics.py` (`compute_prf`, `score_detections`) | Fine — **reuse as-is** |

The extraction core (`DetectInputHook`, `roi_pool`, `extract_box_metadata`,
`_extract_detection_features`, `_resolve_labels_dir`, `_iou/_iop/_match_det_to_gt`)
and the classifier core (`FocalLoss`, `MLPWrapper`, the sklearn wrappers,
`cross_val_score_f1`) are already correct and battle-tested. The job is to
**lift them out of the V5 script into a package**, parameterize the parts that are
hardcoded, and add a diagnosis layer + a clean CLI. We are refactoring, not
rewriting.

---

## 2. Directory & module layout

New top-level package `mri/` (model-MRI; short, fits the "MRI machine" framing):

```
mri/
  __init__.py
  __main__.py          # enables `python -m mri ...`
  cli.py               # argparse, orchestration, the run() pipeline
  datasets.py          # DatasetSpec, auto-detect GT/labels, --pos/--neg parsing
  extract.py           # DetectInputHook, roi_pool, metadata, feature schema
                       #   (lifted verbatim from distill_v5, paths removed)
  stats.py             # PCA coords, LDA fit+accuracy, ANOVA F, silhouette,
                       #   AUROC-per-feature, separability summary
  plots.py             # all figure functions (lifted from visualize_v5_*)
  classifier.py        # FocalLoss, MLPWrapper, LogReg/RF/XGB, CV, artifact save
  diagnose.py          # THE verdict: "does this model need a classifier?"
  report.py            # assemble report.md + stats.json + manifest
  configs/
    example_rgb.yaml   # worked example wiring ft4 + svan/confuser dirs
  README.md            # usage, flag reference, interpretation guide
tests/
  test_mri_smoke.py    # tiny synthetic run (see §7)
```

Output for a run goes to `--out` (default `mri/results/<model-stem>_<timestamp>/`):

```
<out>/
  features.npz           # X, y, w, src_id  (cache; --resume skips re-extraction)
  features_meta.json     # schema, per-source counts, imgsz, grids, detector path
  stats.json             # all numeric results (LDA acc, ANOVA tops, sep metrics)
  images/                # every .png the plotters emit
  classifiers.pkl        # if --train; all CV models
  mlp.pt                 # if --train-mlp; callable artifact (mlp_v5.pt schema)
  report.md              # human-readable diagnosis + embedded figure links
  manifest.json          # exact CLI args + git SHA for reproducibility
```

---

## 3. Datasets: "just attach them"

A dataset is **a directory of images**. The tool auto-resolves labels and role.

- **Positive** (`--pos`): drone-bearing. The tool reads YOLO `.txt` labels via the
  existing `_resolve_labels_dir` (handles both `images/labels` siblings and
  `images/<split>` mirrored layouts). A detection matching a GT box → **drone (y=1)**;
  a detection NOT matching → **hard-negative confuser (y=0)** (same dual-mining the
  V5 script does).
- **Negative** (`--neg`): confuser-only, no drones expected. Every detection is a
  **confuser (y=0)**. Labels not required (`kind="image_no_gt"`).
- A positive dir with **no labels found** → warn and treat as negative, or hard-fail
  (controlled by `--on-missing-labels {warn,error}`, default `error` for `--pos`).

Per-dataset overrides without YAML: `--pos PATH` accepts an optional inline spec
`PATH:imgsz=1280,rule=iop,stride=2,max=5000`. A `--config x.yaml` form supports the
same fields for many sources (mirrors the V5 `SourceConfig` dataclass). Defaults
respect project rules from memory:

- **Svanström / Selcom → `imgsz=1280`, `rule=iop`** (auto-applied if the path stem
  matches `svan*`/`selcom*`, else `--imgsz` global default 640, `rule=iou`). This
  encodes `[[project_imgsz_1280_svanstrom]]` and `[[project_svanstrom_iop_scoring]]`
  so the user can't silently squash drones or under-count with IoU.

`DatasetSpec` carries: `name, path, role(pos/neg), imgsz, stride, match_rule,
max_drones, max_confusers, weight_drone, weight_confuser, filter_prefixes`.

---

## 4. CLI — purposeful flags

```
python -m mri --yolo RGB_model/.../best.pt \
              --pos  G:/drone/svanstrom_paired/RGB/images \
              --neg  G:/drone/rgb_confusers_merged/images/test \
              --out  mri/results/ft4_svan \
              --train-mlp
```

| Flag | Purpose |
|---|---|
| `--yolo PATH` *(req)* | Detector weights. Hook auto-attaches to `model.model.model[-1]`. |
| `--pos PATH[:spec] ...` | One or more positive (drone) dirs. |
| `--neg PATH[:spec] ...` | One or more negative (confuser) dirs. |
| `--config FILE.yaml` | Alternative to `--pos/--neg` for many sources. |
| `--out DIR` | Output root (default auto-named). |
| `--imgsz N` | Global YOLO input size (default 640; auto-1280 for svan/selcom). |
| `--conf / --iou / --iop` | Detector + matching thresholds (default 0.25/0.5/0.5). |
| `--match-rule {iou,iop}` | Global GT-match rule (per-dataset override wins). |
| `--layers p3,p5` | Which FPN maps to pool (default `p3,p5`; allows `p3,p4,p5`). |
| `--p3-grid 2x2 --p5-grid 1x1` | ROI pool grids → controls feature dim. |
| `--max-per-source N` | Cap detections mined per dataset (subset testing). |
| `--stride N` | Frame stride (global; per-dataset override wins). |
| `--quick` | Smoke preset: stride×5, `--max-per-source 200` (mirrors V5 `--quick`). |
| `--stats LIST` | Subset of `pca,lda,anova,heatmap,signature,neurons,silhouette` (default all). |
| `--train` | Train + 5-fold CV the full bench (logreg/rf/xgb/mlp × meta/yolo/fused). |
| `--train-mlp` | Train **only** the production MLP-on-fused and save `mlp.pt`. |
| `--feature-set {meta,yolo,fused}` | Which features the trained classifier uses (default fused). |
| `--resume` | Reuse cached `features.npz`; skip extraction (re-plot/re-train fast). |
| `--device cuda|cpu` | Inference device (default cuda). |
| `--seed N` | Reproducibility (default 42). |

`--train-mlp` implies enough to recreate the production classifier's initial
phases; `--train` is the full ablation bench. Either one, plus the always-on stats,
covers "recreate the MLP classifier or at least the initial phases."

---

## 5. The diagnosis — "does this model need a classifier?"

`diagnose.py` is the contribution's punchline. After extraction (and optionally a
quick CV MLP), it computes and prints a verdict grounded in
`[[project_pipeline_useful_when]]` (classifier earns its keep only on
confuser-rich data with raw F1 < ~0.7) and `[[project_v5_distillation_production]]`.

Signals computed:
1. **Raw confuser hallucination rate** = FP per negative image (no classifier).
   This is the existing bare-detector measurement.
2. **Raw P/R/F1 on positives** (IoU or IoP per rule).
3. **LDA train separability** of drone-vs-confuser in fused feature space
   (already implemented — `lda.score`). High accuracy = a classifier *can* split them.
4. **Projected FP reduction & recall cost**: a fast 5-fold CV MLP estimates how many
   confusers it would reject and how much drone recall it sacrifices (held-out folds).
5. **Per-feature AUROC / ANOVA top-F**: confirms the signal lives in the features,
   not just metadata (guards against the meta-only shortcut).

Verdict matrix (printed + written to `report.md`):

| Raw FP rate | LDA separability | Projected recall cost | Verdict |
|---|---|---|---|
| Low (≤ ~5%) | — | — | **No classifier needed** — detector is already clean. |
| High | High (>~0.95) | Low (<~3pp) | **Classifier strongly recommended** — big FP cut, cheap. |
| High | Low (<~0.85) | — | **Classifier won't help** — features don't separate; fix the detector / data. |
| High | High | High (>~10pp) | **Marginal** — FP cut trades real recall; report the curve, let user decide. |

Thresholds are flags (`--fp-rate-thr`, `--sep-thr`, `--recall-cost-thr`) with the
defaults above. Output is a one-paragraph plain-English recommendation plus the
table of evidence — no menu, one call, matching `[[feedback_short_responses]]`.

---

## 6. Reproducibility & evidence wiring

- Every run writes `manifest.json` (CLI args + git SHA + dataset paths + counts).
- `report.md` ends with a **Delivered** section listing absolute artifact paths,
  per `[[project_analysis_directory_convention]]`.
- Any headline number the tool produces (raw FP rate, post-classifier FP rate,
  recall delta, CV F1) gets appended as a row to `docs/EVIDENCE_LEDGER.md` with the
  reproduction command, per `[[project_evidence_ledger]]`.

---

## 7. Validation with a small subset (the "test it" requirement)

Two layers:

1. **`tests/test_mri_smoke.py`** — no GPU, no real data. Generate ~40 synthetic
   feature rows (2 Gaussian blobs), exercise `stats.py` + `plots.py` +
   `classifier.py` + `diagnose.py` end-to-end on the cached path (`--resume` against
   a hand-written `features.npz`). Asserts: plots written, `stats.json` keys present,
   verdict string is one of the four, MLP artifact loads. Fast CI guard against schema
   drift.

2. **Real smoke run** (documented command for the user to run, per
   `[[feedback_eval_workflow]]` — give the command, they run it):
   ```
   python -m mri --yolo "RGB model/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt" \
                 --pos  "G:/drone/svanstrom_paired/RGB/images:imgsz=1280,rule=iop" \
                 --neg  "G:/drone/rgb_confusers_merged/images/test" \
                 --out  mri/results/_smoke --quick --train-mlp
   ```
   Expected (sanity vs known V5 results): LDA separability high, Svan raw F1 low →
   verdict "classifier recommended"; confuser FP rate should fall after the MLP.
   This reproduces the *initial phases* of the V5 MLP on a 5-minute budget.

---

## 8. Build order (incremental, each step runnable)

1. **`extract.py`** — lift hook/ROI/metadata/label-resolve out of `distill_v5`.
   Parameterize `--layers`, `--p*-grid`; compute `INPUT_DIM` from config.
2. **`datasets.py`** — `DatasetSpec`, inline-spec parser, auto imgsz/rule defaults,
   YAML loader.
3. **`cli.py` extraction path** — wire 1+2, produce `features.npz` + meta. Verify on
   `--quick`.
4. **`stats.py` + `plots.py`** — port `visualize_v5_features.py` functions to read
   the generic npz; make titles/labels derive from the run config, not constants.
5. **`classifier.py`** — lift `FocalLoss`/`MLPWrapper`/wrappers/CV; `--train` and
   `--train-mlp`; save `mlp.pt` in the existing `mlp_v5.pt` schema so
   `eval/eval_v4_vs_patch.py --mlp-weights` still loads it.
6. **`diagnose.py` + `report.md`** — verdict matrix, evidence table.
7. **`tests/test_mri_smoke.py`** — synthetic CI smoke.
8. **`README.md` + `configs/example_rgb.yaml`** — usage + interpretation guide.
9. Append results row to `docs/EVIDENCE_LEDGER.md`; add this contribution to the
   thesis contributions list.

Steps 1–6 can each be validated with `--quick`/`--resume` before moving on. Total:
mostly refactor + glue; the heavy logic already works in the two source scripts.

---

## 9. Open decisions for the user

- **Package name**: `mri/` (proposed) vs `statistics/` (clashes with Python stdlib
  `statistics` — avoid) vs `model_mri/` / `diagnostics/`. Recommend `mri/`.
- **Multi-class LDA** (drone/bird/airplane/heli) is currently driven by Svanström GT
  classes + filename prefixes. Keep it as an opt-in `--subclass-from {gt,prefix}` plot
  or leave it out of v1? (Recommend opt-in, off by default.)
- **Verdict thresholds**: defaults proposed in §5; confirm or tune.

---

## Delivered

- `docs/analysis/2026-05-30_mri_machine_plan.md` — this plan.
- **IMPLEMENTED (2026-05-30)** — the `mri/` package now exists and is verified
  end-to-end on `Yolo26n_selcom_confuser_ft4_1280` + Svanström (pos) + rgb_confusers
  (neg). Modules:
  - `mri/extract.py` — model-agnostic Detect-head hook + ROI pool + `FeatureSchema`
    (per-layer channel dims read from the model at runtime).
  - `mri/datasets.py` — `DatasetSpec`, label auto-resolve, inline-spec parser,
    svan/selcom → imgsz=1280+IoP auto-defaults, YAML loader.
  - `mri/scan.py` — detector sweep + dual-mining + bare-detector tallies.
  - `mri/stats.py` — PCA / LDA separability / ANOVA F / per-feature AUROC / silhouette.
  - `mri/plots.py` — pca, lda, per-block ANOVA, class heatmap, top-neuron KDE, FP-reduction.
  - `mri/classifier.py` — FocalLoss MLP + LogReg/RF/XGB + CV (returns OOF probs) +
    `save_mlp_artifact` (mlp_v5.pt schema).
  - `mri/diagnose.py` — 4-signal verdict (not_needed / recommended / wont_help / marginal).
  - `mri/report.py`, `mri/cli.py`, `mri/__main__.py`, `mri/README.md`,
    `mri/configs/example_rgb.yaml`.
  - `tests/test_mri_smoke.py` — synthetic no-GPU CI smoke (passes).
  - `.gitignore` — `mri/results/` excluded.
- **Smoke validation** (quick subset, 556 dets): verdict "Classifier strongly
  recommended" — LDA separability 1.000, projected FP cut 98%, recall cost 1.5%,
  raw drone F1 0.185 on Svanström. Consistent with the known V5 finding. These are
  smoke-run numbers (not a headline eval) so they are NOT logged to the evidence ledger.

*Status: built and working. §9 open decisions (multiclass-LDA opt-in, threshold tuning)
remain as polish.*
