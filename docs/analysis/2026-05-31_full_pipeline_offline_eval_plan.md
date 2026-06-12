# Full-Pipeline Offline Eval — Plan / Spec

**Date:** 2026-05-31 · Status: **PLAN (awaiting go)**. Captures all requirements for the end-to-end eval so they don't get lost across iterations.

## Requirements (from user)
1. **Offline-runnable** — re-runs with **zero GPU**, from caches.
2. **Full pipeline**, end to end.
3. **Verifier matrix — both modalities:** RGB {bare, patch_v2, mlp_v5} × IR {bare, patch_ir, mlp_v5_ir_aligned}.
4. **IR filter = `mlp_v5_ir_aligned`** (the new dual-mode aligned verifier; replaces the "ship-none" verdict — ledger `ir-grayscale-harvest-solves-thermal-verifier`). Thermal ckpt `mlp_aligned.pt`, grayscale ckpt `mlp_aligned_gray.pt`.
5. **Trust-classifier swap:** {sa32, robust6 (new ft4), none}.
6. **Sampling: 1000 frames per dataset, STRIDED** (stride = ceil(N/1000)), **not** first 1000.

## Verified production stack (knowledge base, 2026-05-31)
ft4 RGB · v3b IR · mlp_v5 RGB verifier (prod) · **mlp_v5_ir_aligned** IR verifier (de-facto prod; CSV flag still `no` → reconcile) · sa32 trust (→ robust6 candidate).

Weights:
- ft4: `RGB model/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt`
- v3b: `runs/corrective_finetune/finetune_v3b/weights/best.pt`
- mlp_v5 (RGB): `eval/results/_v5_selcom_pure_1x8/classifiers/mlp_v5.pt`
- patch_v2 (rgb+ir): `classifier/runs/patches/confuser_filter4_rgb_v2_backup.pt`
- mlp_v5_ir_aligned: `mri/results/ir_aligned/classifiers/mlp_aligned.pt` (+ `_gray.pt`)
- robust6 trust: `classifier/fusion_models/lean_ft4/trust_ft4_robust6.joblib`
- sa32 trust: `classifier/fusion_models/scene_aware_v3more_32feat/model.joblib`

## Offline architecture (the key design)
**Phase A — cache build (GPU, once per detector×dataset):** for each strided-1000 frame, run the detector and persist a per-detection record:
`{box, conf, gt_iou/iop match, 517-D MLP feature vector, patch_v2 score, (IR: aligned-thermal & aligned-gray scores)}`.
→ one `*.feathercache` / npz per (detector, dataset). This is the only GPU step.

**Phase B — eval replay (offline, no GPU, re-runnable):** load caches, apply each verifier threshold + trust classifier + temporal vote + scoring rule, emit per-surface P/R/F1 + halluc/img and the comparison tables. All ablation knobs live here so re-runs are instant.

> Why: box-only caches can't drive MLP/patch verifiers (they need features/pixels). Caching the 517-D feature + verifier scores once makes every downstream comparison free.

## Surfaces (strided 1000 each)
- **RGB:** svanstrom (IoP@1280), antiuav (IoU@640), selcom_val (IoP@1280), rgb_dataset_test (IoU@640), confuser_test (no-GT halluc), + video drone/confuser.
- **IR:** ir_dset_final (IoU@640), ir_video, antiuav-IR, CBAM (confuser-dense), ir confusers (no-GT).

## Reuse (don't rebuild)
- RGB verifier comparison + strided registry: extend `eval/eval_v4_vs_patch.py` (already does bare/patch/mlp + `[::stride]`).
- IR aligned verifier: `eval/run_aligned.py`, `eval/ir_verifier_eval.py`.
- Caches: `eval/det_cache.py`, `eval/cache_inference.py`, `docs/analysis/full_pipeline_ablations/cache/`.
- Add: a Phase-A feature/score cacher + a Phase-B replay/ablation runner.

## Output
Per-surface tables (bare vs patch vs MLP, per modality; sa32 vs robust6 vs none), + a combined dashboard refresh. Metrics: P/R/F1 + halluc/img. Recorded as `evals` rows + a ledger finding.

## Open decisions (confirm before build)
1. **Scope now:** full Phase-A+B build, or start with the verifier matrix on the surfaces that already have caches (faster first numbers)?
2. **Set `mlp_v5_ir_aligned` production=yes** in knowledge? (You said it's production; CSV says no.)
3. **Dataset list** above — add/drop any?

## Delivered (when built)
- (pending) `eval/pipeline_cache.py` (Phase A), `eval/pipeline_eval_offline.py` (Phase B)
- (pending) caches under `eval/results/_offline_pipeline/`
- (pending) `docs/analysis/2026-05-31_full_pipeline_offline_eval.md` (results)
