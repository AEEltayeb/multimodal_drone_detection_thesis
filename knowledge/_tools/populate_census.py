#!/usr/bin/env python3
"""populate_census.py — one-time, idempotent. Applies the forensic-audit census
(three parallel sub-agents, 2026-05-30) into the knowledge/ tables:
  - canonical + active SCRIPTS (Agent A) and NOTEBOOKS (Agent C)
  - weights_path / provenance corrections + IR MLP v5 (Agent B)

Scope = canonical/library + active only (the load-bearing tools); the long tail of
one-offs is intentionally left out and accretes later via the CLAUDE.md record-on-touch
rule. Idempotent: skips ids that already exist; updates models in place.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402


def S(id, path, purpose, role="canonical", inputs="", outputs="", cmd="",
      produces_models="", produces_evals="", lifecycle="active"):
    return dict(id=id, path=path, purpose=purpose, inputs=inputs, outputs=outputs,
                role=role, lifecycle=lifecycle, reproduce_cmd=cmd,
                produces_models=produces_models, produces_evals=produces_evals,
                last_run="")


SCRIPTS = [
    # --- eval/ libraries ---
    S("eval_metrics", "eval/metrics.py", "Core scoring lib: IoU/IoP matching, P/R/F1, per-size buckets, trust-aware scoring", "library"),
    S("eval_datasets", "eval/datasets.py", "Dataset/config lib: load_config, path resolve, YOLO label reader, image/paired/cached/video datasets", "library"),
    S("eval_det_cache", "eval/det_cache.py", "Detection-cache reader/writer (DetCache, parse_dets_str)", "library"),
    S("eval_run_manifest", "eval/run_manifest.py", "Run provenance: write_manifest, cache_identity_tag, weights_short_hash", "library"),
    S("eval_reporting", "eval/reporting.py", "Shared report/plot/table formatting for eval runs", "library"),
    S("eval_dryrun", "eval/dryrun.py", "Pre-flight gate for ablation/eval commands (path/weight existence)", "library"),
    S("eval_prototype_verifier", "eval/prototype_verifier.py", "Prototype (nearest-centroid) verifier lib", "library"),
    # --- eval/ canonical runners ---
    S("eval_cache_inference", "eval/cache_inference.py", "Pre-cache RGB+IR YOLO detections per frame (slow, run once)", inputs="datasets.yaml, YOLO weights", outputs="detection JSON cache", cmd="python eval/cache_inference.py --dataset antiuav"),
    S("eval_eval_pipeline", "eval/eval_pipeline.py", "Unified pipeline eval (YOLO + fusion classifier + confuser filter) across datasets", cmd="python eval/eval_pipeline.py --dataset both --plot"),
    S("eval_eval_model", "eval/eval_model.py", "Raw YOLO benchmarking: multi-model, IoU/IoP, per-source, conf-sweep, size dist, patch verifier", cmd="python eval/eval_model.py --weights best.pt --dataset PATH"),
    S("eval_eval_detector", "eval/eval_detector.py", "Detector-only eval (P/R/F1, size buckets)"),
    S("eval_ablate", "eval/ablate.py", "Sequential ablation driver: expands ablations.yaml factor x level x dataset, aggregates master.csv", inputs="eval/ablations.yaml", outputs="_ablation/<ts>/master.{csv,md}", cmd="python eval/ablate.py --matrix eval/ablations.yaml", produces_evals="rgb_antiuav_baseline;rgb_antiuav_retrainedv2"),
    S("eval_diagnose_failures", "eval/diagnose_failures.py", "Per-detection FP/FN failure diagnosis on a dataset", produces_evals="rgb_svan_baseline"),
    S("eval_diagnose_failures_all", "eval/diagnose_failures_all.py", "Multi-dataset/multi-model failure diagnosis aggregation", produces_evals="rgb_svan_hardneg;rgb_svan_retrainedv2"),
    S("eval_cumulative_halluc", "eval/cumulative_halluc.py", "Cumulative hallucination accounting across pipeline stages (S1->S2->S3)"),
    S("eval_audit_patch_catch", "eval/audit_patch_catch.py", "Audit which FPs the patch verifier catches/misses"),
    S("eval_eval_video_temporal", "eval/eval_video_temporal.py", "Temporal (N-of-M alert gate) video eval"),
    S("eval_eval_video_tests", "eval/eval_video_tests.py", "RGB video-test suite eval (drone + confuser clips)"),
    S("eval_eval_pipeline_video_tests", "eval/eval_pipeline_video_tests.py", "Full-pipeline eval on video tests"),
    S("eval_eval_confuser_videos", "eval/eval_confuser_videos.py", "Eval RGB models on YouTube confuser videos (FPR)"),
    S("eval_eval_full_pipeline_singlepass", "eval/eval_full_pipeline_singlepass.py", "Single-pass full-pipeline eval w/ trust-aware + per-size"),
    S("eval_eval_full_pipeline_persize", "eval/eval_full_pipeline_persize.py", "Full-pipeline eval broken out per size bucket"),
    S("eval_eval_svanstrom_persize", "eval/eval_svanstrom_persize.py", "Svanstrom per-size eval (IoP, imgsz=1280)"),
    S("eval_run_antiuav_per_model", "eval/run_antiuav_per_model.py", "Anti-UAV eval per model (IoU)"),
    S("eval_run_roboflow_eval", "eval/run_roboflow_eval.py", "Roboflow OOD test-set eval runner"),
    S("eval_run_selcom_val", "eval/run_selcom_val.py", "Selcom validation-split eval runner"),
    S("eval_eval_softveto_sweep", "eval/eval_softveto_sweep.py", "Soft-veto threshold sweep (trust-aware tau) on RGB-only"),
    S("eval_compare_rgb_models", "eval/compare_rgb_models.py", "Side-by-side RGB-model comparison (IoU/IoP, per-size)"),
    S("eval_eval_classifier_3way", "eval/eval_classifier_3way.py", "3-way classifier eval (drone/confuser/bg)"),
    S("eval_eval_per_clip_classifier", "eval/eval_per_clip_classifier.py", "Per-clip classifier eval (TRAIN vs TEST membership flagged)"),
    S("eval_build_dashboard_nb", "eval/build_dashboard_nb.py", "Regenerate eval_1000_results dashboard notebook from ablation CSVs", cmd="python eval/build_dashboard_nb.py"),
    S("eval_ir_confuser_necessity", "eval/ir_confuser_necessity.py", "IR MLP necessity test on thermal confusers (FP-catch rate)"),
    S("eval_ir_verifier_eval", "eval/ir_verifier_eval.py", "Evaluate IR MLP verifier"),
    S("eval_eval_v4_vs_patch", "eval/eval_v4_vs_patch.py", "V5 MLP vs patch-v2 head-to-head across 5 surfaces", produces_evals="v5_svan_mlp;v5_svan_patch;v5_svan_bare"),
    S("eval_distill_v5_p3p5_ft4", "eval/distill_v5_p3p5_ft4.py", "V5 feature-distillation trainer (P3+P5 hook -> MLP); base distiller for shipped mlp_v5"),
    S("eval_distill_v5_swap_selcom", "eval/distill_v5_swap_selcom.py", "V5 selcom-source swap retrain (produced pure_1x8 production candidate)", produces_models="mlp_v5", cmd="python eval/distill_v5_swap_selcom.py --variant pure"),
    S("eval_distill_v5_p3p5_ir", "eval/distill_v5_p3p5_ir.py", "V5 distillation for the IR modality", produces_models="mlp_v5_ir"),
    S("eval_eval_pipeline_v5_quick", "eval/eval_pipeline_v5_quick.py", "Quick V5 pipeline eval (overhead, F1, per-frame vs alert-gated)"),
    S("eval_ir_version_comparison", "eval/ir_version_comparison.py", "IR detector version comparison on IR_dset_final test split", produces_evals="ir_final_final;ir_final_v3b;ir_final_v4", cmd="python eval/ir_version_comparison.py --imgsz 640 --split test"),
    # --- classifier/ libraries + trainers ---
    S("clf_utils", "classifier/utils.py", "Shared lib: parse_yolo_labels, compute_iou, align_detections, extract_features", "library"),
    S("clf_patch_verifier", "classifier/patch_verifier.py", "Inference wrapper for per-modality patch CNN verifier", "library"),
    S("clf_mlp_verifier", "classifier/mlp_verifier.py", "Production-importable V5 MLP verifier (frozen schema)", "library"),
    S("clf_fusion_classifier", "classifier/fusion_classifier.py", "XGBoost trust/fusion classifier wrapper (RGB+IR feature fusion)", "library"),
    S("clf_train_patch_verifier", "classifier/train_patch_verifier.py", "Train per-modality patch CNN verifier", produces_models="patch_v2"),
    S("clf_train_classifier", "classifier/train_classifier.py", "Train the fusion/trust classifier"),
    S("clf_train_lean19", "classifier/train_lean19_classifier.py", "19-feature XGBoost trust classifier (newest lean)", produces_models="lean19"),
    S("clf_train_lean10", "classifier/train_lean10_classifier.py", "10-feature XGBoost trust classifier (no brightness)", produces_models="lean10"),
    S("clf_build_dataset", "classifier/build_dataset.py", "Build classifier feature dataset from dets+GT"),
    S("clf_generate_fusion_data", "classifier/generate_fusion_data.py", "Generate paired RGB+IR fusion training data"),
    S("clf_generate_lean19_data", "classifier/generate_lean19_data.py", "Generate lean19 fusion dataset (selcom_1280 RGB + v3b IR)"),
    S("clf_extract_patches_v2", "classifier/extract_patches_v2.py", "Extract drone/confuser crops for patch verifier (newest)"),
    S("clf_run_full_pipeline", "classifier/run_full_pipeline.py", "End-to-end pipeline runner (detector->classifier->filter)"),
    S("clf_eval_full_pipeline", "classifier/eval_full_pipeline.py", "Evaluate full pipeline incl. patch verifier"),
    S("clf_process_youtube", "classifier/process_youtube_videos.py", "Turn YouTube videos into classifier eval frames/crops"),
    S("clf_generate_all_plots", "classifier/generate_all_plots.py", "Generate all classifier thesis plots"),
    # --- training/ ---
    S("rgb_finetune_selcom", "training/finetune_selcom.py", "Fine-tune baseline on selcom (stage->train->eval->compare)", produces_models="selcom_mixed_ft2_1280", cmd='python "training/finetune_selcom.py"'),
    S("rgb_finetune_run_v2", "training/finetune_run_v2.py", "RGB v2 fine-tune driver (retrained_v2 lineage)", produces_models="retrained_v2"),
    S("rgb_finetune_v3_more", "training/finetune_v3_more.py", "hardneg_v3more fine-tune (birds+planes+helis)", produces_models="hardneg_v3more"),
    S("rgb_compare_selcom_ft", "training/compare_selcom_ft.py", "Compare selcom fine-tune variants", produces_evals="selcom_ft3_1280"),
    S("rgb_build_selcom_confuser_ft4", "training/dataset_preparation/build_selcom_confuser_ft4.py", "Stage selcom+confuser FT4 dataset (newest)"),
    S("rgb_build_selcom_mixed_ft3", "training/dataset_preparation/build_selcom_mixed_ft3.py", "Stage selcom-mixed FT3 dataset (50/50 val)"),
    S("rgb_mix_and_split", "training/dataset_preparation/mix_and_split_yolo_datasets.py", "Generic mix + train/val split of YOLO datasets", "library"),
    S("rgb_consensus_filter", "training/dataset_preparation/consensus_filter.py", "Consensus auto-label filtering of mined data"),
    # --- scripts/ ---
    S("scripts_auto_confuser_ft4", "scripts/auto_confuser_ft4.py", "Automated confuser fine-tune loop with regression gating (FT4 search)", produces_models="ft4"),
    S("scripts_regression_gate_ft4", "scripts/regression_gate_ft4.py", "Multi-surface regression gate vs baseline snapshot"),
    S("scripts_baseline_snapshot_ft3", "scripts/baseline_snapshot_ft3.py", "Snapshot ft3 baseline metrics for the gate"),
    S("scripts_mine_confuser_hardnegs", "scripts/mine_confuser_hardnegs.py", "Mine confuser hard negatives from confuser corpora"),
    S("scripts_extract_confuser_datasets", "scripts/extract_confuser_datasets.py", "Build confuser datasets"),
    S("scripts_scan_datasets", "scripts/scan_datasets.py", "Inventory/scan datasets (counts, classes)"),
    S("scripts_dataset_prep_pipeline", "scripts/dataset_preparation/", "Numbered canonical dataset-build pipeline (00-09: roboflow -> dsetV3..V6 -> youtube -> merge_final)", cmd="python scripts/dataset_preparation/09_merge_final.py"),
    S("scripts_visualize_v5_features", "scripts/visualize_v5_features.py", "V5 feature-space thesis figure (LDA/PCA fused)"),
    S("scripts_visualize_v5_lda_multiclass", "scripts/visualize_v5_lda_multiclass.py", "V5 LDA multiclass (4-class confuser) projection figure"),
    S("scripts_visualize_yolo_features", "scripts/visualize_yolo_features.py", "YOLO internal-feature ('brain') visualization figures"),
    # --- gui/ (production stack) ---
    S("gui_fusion_engine", "gui/fusion/engine.py", "FusionEngine: dual-YOLO + XGBoost trust classifier + patch-verifier veto (single/paired/grayscale)", "library"),
    S("gui_fusion_temporal", "gui/fusion/temporal.py", "Per-modality temporal state + detection/overlay drawing", "library"),
    S("gui_fusion_pipeline", "gui/fusion/pipeline.py", "Fusion pipeline orchestration", "library"),
    S("gui_fusion_features", "gui/fusion/features.py", "Feature extraction for fusion classifier", "library"),
    S("gui_pyside_engine", "gui/pyside_engine.py", "PySide detection engine wrapping fusion/engine.py + alert-gate temporal", "library"),
    S("gui_pyside_app", "gui/pyside_app.py", "TALOS PySide6 GUI (production demo app)", cmd="python -B gui/pyside_app.py"),
    S("gui_api", "gui/api.py", "Programmatic detection API over the engine", "library"),
    # --- mri/ ---
    S("mri_cli", "mri/cli.py", "Model-MRI entry: image a YOLO feature space, brain stats, plots, optional confuser-MLP train, verdict", cmd="python -m mri --yolo best.pt --pos DRONE --neg CONFUSER --train-mlp"),
    S("mri_extract", "mri/extract.py", "FeatureExtractor + FeatureSchema (FPN feature hooks)", "library"),
    S("mri_classifier", "mri/classifier.py", "MLP/LogReg/RF/XGB wrappers + cross_val_score_f1 + save_mlp_artifact", "library"),
    # --- analytics/ ---
    S("analytics_spec_pipeline", "analytics/spec_analysis/", "Ordered thesis analysis pipeline (01-11: geometry, imgsz sweep, per-model failures, case library, metrics inventory, confuser profile)", cmd="python analytics/spec_analysis/07_metrics_inventory.py"),
    S("analytics_cross_eval_all", "analytics/eval/cross_eval_all.py", "Cross-model/cross-dataset eval matrix driver"),
    # --- knowledge tooling (migration provenance) ---
    S("kb_migrate_evidence_ledger", "knowledge/_tools/migrate_evidence_ledger.py", "One-time migration of EVIDENCE_LEDGER headline metrics/findings into knowledge tables", "one-off"),
    S("kb_populate_census", "knowledge/_tools/populate_census.py", "One-time application of the forensic-audit census (scripts/models/notebooks) into knowledge tables", "one-off"),
    S("kb_hook_nag", "knowledge/_tools/hook_nag.py", "SessionStart snapshot + Stop-hook nag for unrecorded scripts", "library"),
]

NOTEBOOKS = [
    S("nb_ir_dset_final_results", "notebooks/ir_dset_final_results.ipynb", "Final IR detector results + dataset overview + leakage audit", "canonical", outputs="ir_final_*.png, model_progression_bar.png"),
    S("nb_thesis_results", "notebooks/thesis_results.ipynb", "Official thesis results workspace: IR label-quality + dataset-expansion ablations, FP annotation-error audit, OOD traffic FPPI", "canonical"),
    S("nb_rgb_ir_fusion_v2", "notebooks/rgb_IR_fusion_v2.ipynb", "Deployed XGBoost fusion trust classifier technical doc (40 feat, 152k frames, 10-approach benchmark)", "canonical"),
    S("nb_eval_1000_results", "docs/analysis/eval_1000_results.ipynb", "Full-pipeline ablation dashboard (thesis single source of truth)", "canonical"),
    S("nb_model_results_comparison_v3", "notebooks/model_results_comparison_v3.ipynb", "IR model-progression tables + cross-eval F1/P/R/mAP heatmap incl. Svanstrom", "one-off", outputs="cross_eval_heatmap.png, svanstrom_performance.png"),
    S("nb_rgb_ir_fusion_v1", "notebooks/rgb_IR_fusion.ipynb", "v1 fusion experiments: Anti-UAV-RGBT extraction, OR/AND/gated sweeps, IR-advantage example generator", "one-off", lifecycle="superseded"),
]

# Agent B: weights_path + provenance corrections to existing model rows
MODEL_FIXES = {
    "hardneg_v3more": {"weights_path": "models/rgb/Yolo26n_hardneg_v3_more/weights/best.pt", "trained_from_script": "training/finetune_v3_more.py"},
    "retrained_v2": {"weights_path": "models/rgb/Yolo26n_retrained_v2/weights/best.pt", "trained_from_script": "training/finetune_run_v2.py"},
    "selcom_mixed_ft2_1280": {"weights_path": "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"},
    "selcom_mixed_ft3_1280": {"weights_path": "models/rgb/Yolo26n_selcom_mixed_ft3_1280/weights/best.pt"},
    "ft4": {"weights_path": "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt", "trained_from_script": "scripts/auto_confuser_ft4.py"},
    "ir_final": {"weights_path": "models/ir/IR_final_cleaned/weights/best.pt"},
    "sa32": {"weights_path": "models/routers/scene_aware_v3more_32feat/model.joblib"},
    "fusion_no_fn_v1.1": {"weights_path": "classifier/runs/reliability/fusion/fusion_no_fn_model_v1.1.joblib"},
    "control40": {"weights_path": "models/routers/control_v3more_40feat/model.joblib"},
    "lean19": {"weights_path": "models/routers/lean19/model.joblib"},
    "lean10": {"weights_path": "models/routers/lean10/model.joblib"},
    "lean13": {"weights_path": "models/routers/lean13/model.joblib"},
    "lean17": {"weights_path": "models/routers/lean17/model.joblib"},
}

NEW_MODELS = [
    dict(id="mlp_v5_ir", name="mlp_v5_ir", type="mlp", purpose_tags="confusion-filter",
         trained_from_script="eval/distill_v5_p3p5_ir.py",
         train_dataset="distilled v3b IR p3+p5 ROI features",
         weights_path="eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt",
         provenance_notes="IR-modality V5 verifier; necessity WARRANTED 2026-05-30 (catches ~92% IR FPs at thr~0.15). Not yet a full ledger row.",
         production="no", lifecycle="active"),
]


def _append(table, rows):
    cur = kb._read(table)
    existing = {r["id"] for r in cur}
    added = 0
    for row in rows:
        if row["id"] in existing:
            continue
        errs = kb._validate_row(table, row)
        if errs:
            print(f"SKIP {table}.{row['id']}: {'; '.join(errs)}")
            continue
        cur.append(row); existing.add(row["id"]); added += 1
    kb._write(table, cur)
    print(f"{table}: +{added} rows ({len(cur)} total)")


def _fix_models():
    cur = kb._read("models")
    by_id = {r["id"]: r for r in cur}
    n = 0
    for mid, upd in MODEL_FIXES.items():
        if mid in by_id:
            for k, v in upd.items():
                if not (by_id[mid].get(k) or "").strip():
                    by_id[mid][k] = v; n += 1
    kb._write("models", cur)
    print(f"models: {n} field corrections applied")


def main():
    _append("scripts", SCRIPTS)
    _append("scripts", NOTEBOOKS)
    _append("models", NEW_MODELS)
    _fix_models()
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
