# Codebase restructure — EXECUTABLE PLAN (for a fresh chat)

**Status:** plan, 2026-05-31. NOT started. A new session should read this + `knowledge/scripts.csv`
+ `knowledge/models.csv` and execute it. This is a **knowledge-base-driven refactor**: the CSVs are
the map, and every file move MUST update its row's `path`/`weights_path`.

## Goal
Per-modality top-level layout (flat), with a shared cross-cutting layer. Nothing deleted —
superseded/one-off artifacts go to `archive/`, never `rm`.

## Target layout (flat, per the user)
```
rgb/            RGB detector: training scripts + dataset-prep-rgb + weights/runs co-located
ir/             IR detector: training/finetune scripts + weights/runs co-located
classifier/     XGBoost trust classifiers (cross-modal: consumes RGB+IR) — fusion_models, trainers, fusion_classifier.py, utils.py
filters/        post-detection verifiers, CONSOLIDATED: patch CNNs (classifier/runs/patches, train_patch_verifier, patch_verifier.py) + MLP verifiers (mlp_verifier.py, distill_v5*.py, eval_v4_vs_patch.py, eval/results/_v5*/classifiers)
gui/            cleaned PySide app (from ir_gui/: pyside_app + fusion/ engine; flet/app/fusion_app prototypes -> archive)
eval/           SHARED HARNESS (cross-cutting): metrics.py, datasets.py, det_cache.py, run_manifest.py, reporting.py, dryrun.py, ablate.py + ablations.yaml, ALL eval_*/run_*/diagnose_*/cumulative_halluc/audit_* runners
data/           dataset construction: scripts/dataset_preparation/, builders, scan_datasets, mine_confuser_hardnegs, extract_confuser_datasets, consensus_filter, convert_* 
mri/            model-introspection package (stays as-is)
label_reviewer/ labeling tool (+ scripts/review_labels_gui.py)
docs/           thesis (thesis_chapters.tex, thesis_working.tex), analysis/, figures/, generate_thesis_figures.py
knowledge/      the knowledge system — STAYS, TRACKED
archive/        graveyard (receives superseded/one-off/safe-to-archive + scratch + _check_*/_verify + gui prototypes)
```
Untracked (gitignore, NOT moved): `.claude/`, `check.txt`, local scratch `.md`, weights/`runs/` blobs (already gitignored).
**`knowledge/` stays git-TRACKED** (it's the source of truth). Everything else "local/claude" stays gitignored.

## The mapping authority = scripts.csv + models.csv
For each `scripts.csv` row, route by `purpose`/`path`/`role`:
- detector training (RGB/IR) -> `rgb/` or `ir/`
- `type`-classifier trainers/libs -> `classifier/`; patch/MLP verifier code -> `filters/`
- library hubs (metrics/datasets/det_cache/run_manifest/reporting) + `eval_*`/`run_*` runners -> `eval/`
- dataset builders/converters -> `data/`
- gui -> `gui/`; mri -> `mri/`; label tool -> `label_reviewer/`
- `role=one-off` or `lifecycle in {superseded, safe-to-archive}` -> `archive/` candidate (confirm with user)
For each `models.csv` row: weights move with their modality folder (`rgb/`, `ir/`, `filters/`); `lifecycle=archived` rows already point into `archive/`.

## Hard rules (non-negotiable)
1. **Commit first / work on a branch.** ~356 files move; make it reversible. (Recommend a branch despite the thesis no-branch pref — this is a structural refactor.)
2. **Never `rm`.** Use `git mv` (tracked) / `Move-Item` (gitignored) into the target or `archive/`.
3. **Update the knowledge base on EVERY move.** After moving a file, set its row's new `path`/`weights_path` via `kb.py set <table> <id> path=<new>`. Build a **`kb.py mv` helper** first (step 0) that does git-mv + row-update + view-regen atomically, so moves can't desync the registry.
4. **Fix imports.** Per-modality + `eval/` shared means many `import` paths change. After moves, grep for broken imports and update; add `__init__.py` where needed. A modality importing the shared harness uses `from eval.metrics import ...` (or make `eval/` an installed package).
5. **Validate after each phase:** `py knowledge/_tools/kb.py validate` (all paths resolve) + a smoke import of the GUI + one eval runner + `python -m mri`.

## Phases
0. **Build `kb.py mv`** (file move + path-row update + regen) and commit/branch.
1. **eval/ shared harness** first (everything depends on it) — move libs + runners, fix imports, smoke-test one eval.
2. **data/** (dataset prep) — move builders/converters.
3. **rgb/**, **ir/** — detectors + weights/runs; update models.csv weights_path.
4. **classifier/** + **filters/** — trust classifiers vs verifiers; update weights_path.
5. **gui/** (clean: ship pyside+fusion, archive flet/app prototypes) — smoke-launch.
6. **label_reviewer/**, confirm **mri/**/**docs/** in place.
7. **archive sweep** — move `role=one-off`/`superseded`/`safe-to-archive` + `_check_*`/`_verify`/`scratch` to `archive/` (confirm list with user; reuse `/sweep` for the registered safe-to-archive items).
8. Final `kb.py validate` + full smoke (GUI, an eval, mri, a training dry-run) + commit.

## Acceptance criteria
- `kb.py validate` passes; every `scripts.csv`/`models.csv` `path`/`weights_path` resolves on disk.
- GUI launches; one `eval_*` runner runs; `python -m mri --help` works; an RGB + IR training script imports cleanly.
- Repo root has only the target folders + tracked config; no loose `_check_*`/scratch at root.
- `DECISIONS.md` gets an entry; `PROJECT_STATE.md` Resume-Here updated.

## Notes
- The thesis skill (`docs/thesis_working.tex`) and `knowledge/` are unaffected by moves except path
  references — keep them working.
- `analytics/spec_analysis/` (thesis analysis pipeline): move under `docs/` or keep top-level — decide at step 6.
- Top-level orchestrators (`run_afk_pipeline.py`, `run_video_tests.py`) -> `eval/`; `generate_thesis_figures.py` -> `docs/`.
