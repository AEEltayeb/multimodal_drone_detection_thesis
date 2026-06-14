# ES Drone Detection — Thesis Workspace

Dual-modality (RGB + thermal IR) drone detection system with a trust-routing classifier,
feature-space verifiers, an operator GUI, and a human-in-the-loop data engine — plus the
MSc thesis built on it. This repo is the curated, **self-contained** thesis workspace —
all model weights, the thesis source + figures, and the full pipeline code, runnable from a
fresh clone. Raw image/video datasets and regenerable caches are **excluded** (they live on
`G:/drone`); everything needed to run the GUI, build the thesis, and regenerate figures is
committed. (The original heavyweight workspace `ES_Drone_Detection/` is a local archive, not
required by this repo.)

## Where everything is

| Directory | What's in it |
|---|---|
| `docs/` | **The thesis**: `thesis_working.tex` (+ Overleaf split in `thesis_working_distilling_overleaf/`), `figures/`, `references.bib`, `build_thesis.ps1`, dated analyses in `analysis/` |
| `models/` | **All model weights, by role** — see the production stack table below |
| `gui/` | **Operator GUI** (PySide6 "TALOS"): `pyside_app.py` + `fusion/` engine, alert-gate temporal logic, `fusion_settings.json` |
| `label_reviewer/` | **Human-in-the-loop label review tool** (tkinter): `review_labels_gui.py` |
| `mri/` | **Model MRI** — detector feature-space diagnosis (PCA/LDA/ANOVA, verifier training): `py -m mri` |
| `thesis_eval/` | **Tier-1 unified eval harness** — detect-once caches + 60 s zero-GPU replay that produces the thesis numbers |
| `eval/` | Evaluation library + harnesses (metrics, ablations, routing comparison) and `eval/results/` artifacts |
| `classifier/` | Trust-classifier / verifier / patch-CNN training and feature code |
| `training/` | RGB detector fine-tuning scripts + `dataset_preparation/` |
| `scripts/` | Utility scripts: dataset prep, confuser mining (`auto_confuser_ft4.py`), runners (`run_afk_pipeline.py`) |
| `knowledge/` | **The knowledge base** (CSV database of scripts/models/evals/findings + generated views). Read `knowledge/README.md`; write only via `knowledge/_tools/kb.py` |
| `datasets/` | Local video eval sets (confuser videos, drone video tests, demo pairs). Big training corpora live on `G:/drone` |
| `configs/` | Training/eval YAML configs |
| `analytics/`, `notebooks/`, `tests/` | Spec analyses, result notebooks, smoke tests |
| `archive/` | Swept legacy scripts (git-tracked, dated subfolders) — nothing is ever deleted |

## Production stack (models/)

| Role | Weights | Notes |
|---|---|---|
| RGB detector | `models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt` | FT4; imgsz 1280 for Svanström/SelCom, 640 default |
| IR detector | `models/ir/corrective_finetune/finetune_v3b/weights/best.pt` | v3b |
| Trust router | `models/routers/robust8_noreject.joblib` (**robust8-nr**) | **shipped** no-reject 3-class, argmax (no τ); per-frame filter owns FP rejection. `robust8` (τ=0.20) / `sa32` / `robust6` kept for comparison |
| RGB verifier | `models/verifiers/rgb_v5/mlp_v5.pt` | V5 distillation MLP, per-frame, thr 0.15 |
| IR verifier | `models/verifiers/ir_aligned/mlp_aligned.pt` (+ `mlp_aligned_gray.pt`) | thermal + grayscale scalers |
| Patch verifier (fallback) | `models/patches/confuser_filter4_{rgb,ir}_v2_backup.pt` | 5-class MobileNetV3, fail-open, thr 0.9 |
| Comparison RGB detectors | `models/rgb/Yolo26n_*/weights/best.pt` | baseline, retrained_v2, selcom variants… |
| IR lineage | `models/ir/IR_dsetV4/5/6…` | historical versions |
| Pretrained | `models/pretrained/yolo11n.pt`, `yolo26n.pt` | training init |

## Run on a fresh clone

```powershell
git clone https://github.com/AEEltayeb/dorne_thesis.git
cd dorne_thesis
py -m venv .venv ; .venv\Scripts\Activate.ps1        # Windows PowerShell
# install PyTorch first (GPU or CPU — see the requirements.txt header), then:
pip install -r requirements.txt
```

Model weights, the thesis, and all figures are committed, so the **GUI** and **thesis build**
work immediately. The GUI reads model paths **repo-relative** (resolved in
`flet_app/settings_dialog.load_settings`), so it finds the weights wherever you cloned. Raw
datasets and detection caches are not shipped: the zero-GPU eval replay
(`thesis_eval/pipeline_eval_unified.py`) needs `thesis_eval/cache/`, which you regenerate by
running detection over the `G:/drone` corpora (or copy the cache across).

## Quickstart

```powershell
# Operator GUI
py gui/pyside_app.py

# Reproduce the thesis numbers (zero-GPU replay from caches, ~60 s)
py thesis_eval/pipeline_eval_unified.py

# Build the thesis PDF (MiKTeX)
powershell -ExecutionPolicy Bypass -File docs/build_thesis.ps1

# Regenerate thesis figures
py docs/generate_thesis_figures.py   # set PYTHONUTF8=1 on Windows consoles

# Label reviewer (HITL)
py label_reviewer/review_labels_gui.py

# Knowledge base health
py knowledge/_tools/kb.py validate
```

## Method

Every script/model/eval lives as a row in `knowledge/*.csv` (the source of truth).
Before writing a new script, search `knowledge/scripts.csv`; after producing a number,
record it (`/record`). Cleanup goes through lifecycle marks + `/sweep` — files are
archived to `archive/<date>/`, never deleted. See `CLAUDE.md` and `knowledge/README.md`.
