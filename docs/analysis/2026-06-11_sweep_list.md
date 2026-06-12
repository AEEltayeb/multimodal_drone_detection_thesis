# 2026-06-11 — Script-sprawl sweep list (reorg Step 5)

Everything below moves to archive/2026-06-11/<original path> via git mv (history kept) + kb lifecycle=archived.
NOTHING moves without green light. Scripts imported by kept code are auto-excluded (list at bottom).

## A — recorded absorbed (into mri) (2) — clear dead

- eval/ir_confuser_necessity.py
- eval/ir_verifier_eval.py

## B — recorded one-offs, already-run analyses (29) — numbers already recorded in evals/ledger

- classifier/ablation_feature_sets.py
- classifier/ablation_split_v3.py
- classifier/build_ablation_datasets.py
- classifier/build_train_eval_optimal.py
- classifier/eval_rgb_test.py
- classifier/train_ablations.py
- eval/_routing_replay.py
- eval/_veto_vs_confuser.py
- eval/distill_v5_rebalance_svan.py
- eval/distill_v5_remine_rgb.py
- eval/gray_drone_effect.py
- eval/ir_failopen_viability.py
- eval/overnight_ablation_full.py
- eval/overnight_ir.py
- eval/plot_robust6_stats.py
- eval/plot_sell_figures.py
- eval/run_aligned.py
- eval/run_aligned_full.py
- eval/run_email_recompute.py
- scripts/eval_selcom.py
- scripts/visualize_activation_compilation.py
- scripts/visualize_class_heatmap.py
- scripts/visualize_classes.py
- scripts/visualize_domain_shift.py
- scripts/visualize_p3_features.py
- scripts/visualize_v4_features.py
- scripts/visualize_v5_features_ir.py
- scripts/visualize_v5_production.py
- scripts/visualize_v5_speed.py

## C — unrecorded loose scripts, not imported by anything (88) — never recorded in scripts.csv

- classifier/_check_reextract.py
- classifier/_check_superset.py
- classifier/_check_trust_decisions.py
- classifier/_compare_classifier.py
- classifier/_compare_confuser.py
- classifier/_compare_features.py
- classifier/_debug_labels.py
- classifier/_merge_yt_and_retrain.py
- classifier/analyze_sources.py
- classifier/bench_feature_cache.py
- classifier/calibrate_confuser_ood.py
- classifier/check_gt_alignment.py
- classifier/check_offset.py
- classifier/check_offset_per_seq.py
- classifier/clean_patches_consensus.py
- classifier/compare_confuser_ab.py
- classifier/compare_confuser_youtube_ab.py
- classifier/compare_old_new_rgb.py
- classifier/compare_pipelines_from_perdet.py
- classifier/compare_thesis_format.py
- classifier/compute_classifier_then_filter.py
- classifier/compute_new_scoped.py
- classifier/convert_svanstrom_paired.py
- classifier/debug_alignment.py
- classifier/diagnose_ir_labeling.py
- classifier/diagnose_ir_only.py
- classifier/eval_aerial_negatives.py
- classifier/eval_feature_modes.py
- classifier/eval_rgb_finetune.py
- classifier/eval_rgb_finetune_ablation.py
- classifier/evaluate_fusion.py
- classifier/extract_antiuav_crops.py
- classifier/extract_background_crops.py
- classifier/gen_ablation_plots.py
- classifier/generate_lean13_data.py
- classifier/generate_plots.py
- classifier/generate_youtube_plots.py
- classifier/process_youtube_videos_4class.py
- classifier/recompute_classifier_scoped.py
- classifier/run_ablation.py
- classifier/run_full_eval.py
- classifier/run_inference.py
- classifier/run_inference_svanstrom.py
- classifier/smart_merge_lean17.py
- classifier/step1_augment_grayscale.py
- classifier/step1_build_real_grayscale.py
- classifier/step2_train_gray_clf.py
- classifier/step3_eval_both.py
- classifier/step3_fast_eval.py
- classifier/sweep_all_thresholds.py
- classifier/sweep_confuser_threshold.py
- classifier/sweep_minsize.py
- classifier/sweep_patch_threshold.py
- classifier/sweep_rgb_optimal_conf.py
- classifier/train_classifier_v11.py
- classifier/train_retrained_v2_classifier.py
- classifier/visualize_calibration.py
- classifier/visualize_rgb_only.py
- classifier/visualize_six_configs.py
- eval/_agg_fprate.py
- eval/_cbam_breakdown.py
- eval/_check_mlp_v4.py
- eval/_dataset_stats.py
- eval/_diagnose_lean13_failures.py
- eval/_patch_notebook.py
- eval/_probe_drone_assets.py
- eval/_summarize_pipeline.py
- eval/_test_cache_agreement.py
- eval/_test_drone_stats.py
- eval/cache_rgb1280_svanstrom.py
- eval/compare_classifier_temporal_input.py
- eval/compare_grayscale_clf_modes.py
- eval/compare_lean13_vs_sa32.py
- eval/eval_drone_video_full.py
- eval/eval_drone_video_raw.py
- eval/eval_ir_versions.py
- eval/eval_test_splits_full.py
- eval/eval_thesis_ablation.py
- eval/eval_video_persize.py
- eval/generate_cutpaste_confusers_v4.py
- eval/generate_cutpaste_dataset.py
- eval/generate_cutpaste_v2.py
- eval/generate_cutpaste_v3.py
- eval/generate_drone2_only.py
- eval/merge_results.py
- eval/quick_drone_only_1280.py
- eval/render_example_images.py
- eval/run_alert_gate.py

## D — gui flet/tkinter legacy (superseded by PySide) (6) — no external importers

- gui/flet_app
- gui/flet_theme.py
- gui/run_flet.py
- gui/app.py
- gui/fusion_app.py
- gui/run_app.py

## E — scratch + root debris (2) — inert

- generate_professor_plots.py
- scratch

## Auto-KEPT: imported by other code (27)

- classifier/eval_youtube_ir_filter.py
- classifier/eval_youtube_rgb_filter.py
- classifier/generate_all_plots.py
- classifier/generate_fusion_data.py
- classifier/generate_lean19_data.py
- classifier/generate_retrained_v2_data.py
- classifier/mlp_verifier.py
- classifier/patch_verifier.py
- classifier/rebuild_yolo_cache.py
- classifier/train_confuser_4class.py
- classifier/train_confuser_filter.py
- classifier/utils.py
- eval/compare_routing_pipeline.py
- eval/datasets.py
- eval/det_cache.py
- eval/distill_v5_p3p5_ft4.py
- eval/distill_v5_swap_selcom.py
- eval/eval_detector.py
- eval/eval_full_pipeline_persize.py
- eval/eval_pipeline.py
- eval/eval_v4_vs_patch.py
- eval/generate_cutpaste_v4.py
- eval/metrics.py
- eval/overnight_confuser_distill.py
- eval/prototype_verifier.py
- eval/reporting.py
- eval/run_manifest.py

## Delivered
- docs/analysis/2026-06-11_sweep_list.md (this file)