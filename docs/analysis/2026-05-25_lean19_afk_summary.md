# AFK pipeline summary - 2026-05-25

## Models trained

- **Lean-10 (no yt)**: n_features=10, acc=0.9700, F1m=0.9057, feature_importance top-3: `ir_best_aspect_ratio`=0.247, `ir_best_log_bbox_area`=0.221, `ir_max_conf`=0.149
- **Lean-13 (no yt)**: n_features=13, acc=0.9685, F1m=0.9093, feature_importance top-3: `ir_best_aspect_ratio`=0.218, `ir_best_log_bbox_area`=0.212, `ir_max_conf`=0.159
- **Lean-19 (with yt)**: n_features=19, acc=0.9700, F1m=0.9125, feature_importance top-3: `ir_best_aspect_ratio`=0.336, `ir_max_conf`=0.298, `ir_best_pos_x`=0.065
- **Lean-13_yt**: n_features=13, acc=0.9692, F1m=0.9121, feature_importance top-3: `ir_best_aspect_ratio`=0.380, `ir_max_conf`=0.282, `rgb_max_conf`=0.070
- **Lean-10_yt**: n_features=10, acc=0.9649, F1m=0.8998, feature_importance top-3: `ir_best_aspect_ratio`=0.422, `ir_max_conf`=0.284, `rgb_max_conf`=0.076

## Per-source breakdown (from each model's own held-out split)

See each `metrics.json` `per_dataset` field.


## Headline 5-way eval (300-frame unified test set)

Full report: `docs/analysis/2026-05-25_classifier_3way_eval.md`


## Log

`logs/afk_pipeline_20260525_081546.log`
