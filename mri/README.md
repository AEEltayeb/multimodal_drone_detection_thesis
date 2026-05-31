# Model MRI

A model-agnostic **methodology** for imaging a detector's internal feature space.

Attach any [ultralytics](https://github.com/ultralytics/ultralytics) detector and
two roles of image folders — a **positive** (target-class) set and one or more
**negative** (distractor) sets — and MRI runs the detector, extracts the FPN
features behind every detection, and reports a diagnosis:

> **Does this detector need a downstream classifier to suppress false positives —
> and if so, will one actually work?**

Feature dimensions are read from the model at runtime, so nothing is tied to a
particular architecture, input size, class set, or domain. Drone-vs-bird
detection is the worked **case study** in this repo, not a requirement — point it
at any detector and any two folders of "things it should fire on" / "things it
shouldn't".

It generalizes two one-off scripts (`eval/distill_v5_p3p5_ft4.py`,
`scripts/visualize_v5_features.py`) into a reusable tool. The 4th project
contribution alongside the YOLO models, the label reviewer, and the PySide GUI.

## Vocabulary (role labels, not domain terms)

The code uses two short labels for the two roles. They are **generic**:

| Code term | Means | `y` |
|---|---|---|
| **target** / "drone" | a detection that matches a ground-truth box in a `--pos` set | `1` |
| **distractor** / "confuser" | a detection with no matching GT (in `--pos`), or any detection in a `--neg` set | `0` |

Read "drone" as *the target class* and "confuser" as *a hard negative the detector
fired on*. The methodology is class-agnostic.

## The only inputs you need

```bash
python -m mri \
  --yolo  path/to/detector/best.pt \
  --pos   DATA/targets/images \
  --neg   DATA/distractors/images \
  --train-mlp
```

- `--pos` — folder(s) containing the target class, with YOLO `.txt` labels
  (auto-resolved). A detection matching a GT box → **target**; one that doesn't →
  **hard-negative distractor**.
- `--neg` — folder(s) of distractors only; every detection is a **distractor**.
  No labels needed.
- `--yolo` — any ultralytics detector. Works for any architecture/width/imgsz.

Output auto-lands in `mri/results/<detector>_<timestamp>/` (override with `--out`).
**All MRI artifacts stay under `mri/`** — per-run outputs in `mri/results/`
(gitignored), curated reports + their figures in `mri/docs/` (committed).

## What it produces

```
<out>/
  features.npz          X, y, w feature corpus   (re-use with --resume)
  features_meta.json    schema + per-dataset scan counts
  stats.json            LDA separability, ANOVA tops, AUROC, diagnosis
  images/               pca / lda / anova / heatmap / neuron-kde / fp-reduction
                        + example_<dataset>.png — per-dataset spatial activation
                          panel (crop + P3 + P5 heatmaps; live --pos/--neg scan only)
  mlp.pt                trained classifier (callable artifact)  [if --train*]
  manifest.json         CLI args + git SHA
  report.md             verdict + evidence tables + figures + Delivered
```

## The verdict

Four signals → one recommendation (thresholds tunable via
`--fp-rate-thr/--sep-thr/--recall-cost-thr`):

| Raw FP rate | Feature separability (LDA) | Recall cost | Verdict |
|---|---|---|---|
| ≤ thr | — | — | **No classifier needed** — detector is already clean |
| high | high | low | **Classifier recommended** — big FP cut, cheap |
| high | low | — | **Won't help** — features don't separate; fix detector/data |
| high | high | high | **Marginal** — FP cut trades real recall; read the curve |

> ⚠️ The default verdict uses in-pool **cross-validation**, which is *optimistic* —
> it cannot see the out-of-distribution recall a verifier sacrifices at
> deployment. For a shippable decision, follow up with `--holdout-eval` (below):
> in this project a CV-F1 of 0.987 still hid a verifier that was *not* deployable.

## Modules

| Module | Role |
|---|---|
| `cli` / `__main__` | entry point, orchestration, `run()` pipeline |
| `datasets` | `DatasetSpec`, label auto-resolve, inline-spec + YAML parsing |
| `extract` | model-agnostic Detect-head hook, ROI pool, `FeatureSchema` |
| `scan` | detector sweep + target/distractor mining + bare-detector tallies |
| `stats` | PCA, LDA separability, ANOVA F, per-feature AUROC, silhouette |
| `plots` | static figures (pca / lda / anova / heatmap / neuron-kde / fp) |
| `classifier` | FocalLoss MLP + LogReg/RF/XGB + CV; saves a callable `.pt` |
| `diagnose` | the four-signal verdict |
| `report` | assembles `report.md` + Delivered manifest |
| `examples` | per-dataset spatial activation panels (crop + P3/P5 heatmaps) |
| `holdout` | **honest gate**: held-out per-surface bare-vs-verifier P/R/F1 |
| `modality_probe` | is a class's signature the same across two input modalities? |
| `modality_align` | rescue cross-modality transfer via per-feature affine alignment |
| `train_aligned` | synthesize a verifier from cheap-modality negatives, aligned |
| `plot_alignment` | the cross-modal transfer figure |
| `_stage_v5_corpus` | stage a legacy feature corpus into a run dir for `--resume` |

### Cross-modal extension (optional, advanced)

`modality_probe → modality_align → train_aligned → plot_alignment` answer a
follow-on question the same feature-space machinery enables: *is a class
represented the same way under two input modalities?* The case study (thermal vs
grayscale-RGB) shows the gap is a per-feature affine offset that per-modality
z-scoring removes — so distractors mined cheaply in one modality can train a
verifier deployable in the other. This is domain-specific in its *data*, but the
*method* (probe identical-neuron firing, align the offset, train in the shared
space) is general to any two-modality detector.

## Useful flags

| Flag | Purpose |
|---|---|
| `--train-mlp` | Train only the production focal-loss MLP; save `mlp.pt`. |
| `--train` | Full bench: logreg / rf / xgb / mlp, 5-fold CV. |
| `--feature-set {meta,yolo,fused}` | Which features the classifier uses. |
| `--layers p3,p5` `--p3-grid 2x2` | Which FPN maps + ROI pool grids → feature dim. |
| `--grayscale-input` | Feed images as gray-3ch before the detector (the grayscale-fallback deploy op). |
| `--holdout-eval MLP.pt` | Honest held-out eval of a trained verifier on the `--pos/--neg` surfaces; runs and exits. |
| `--quick` | Smoke preset (stride ×5, 200 dets/source). |
| `--no-examples` | Skip the per-dataset spatial activation panels. |
| `--max-per-source N` `--stride N` | Subset controls. |
| `--resume` | Reuse cached `features.npz`; re-plot / re-train fast. |
| `--config FILE.yaml` | Many datasets with per-source overrides (see `configs/`). |

Inline per-dataset overrides: `--pos "DATA/x/images:imgsz=1280,rule=iop,max=4000"`.

**Domain presets (optional convenience):** paths whose stem matches `svan*` /
`selcom*` auto-default to `imgsz=1280, rule=iop` — encoding two project findings
about that dataset. These are conveniences for the case study, not part of the
method; pass `--imgsz`/`--match-rule` explicitly for any other data.

## Test it

```bash
python tests/test_mri_smoke.py          # synthetic, no GPU
python -m mri --yolo <model> --pos <target_dir> --neg <distractor_dir> --quick --train-mlp
```
