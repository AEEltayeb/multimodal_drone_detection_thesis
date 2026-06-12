# Fusion classifiers (v3_more RGB pipeline)

Two trained models, both using the v3_more RGB YOLO + finetune_v3b IR YOLO inference,
trained on `fusion_dataset_v3more.csv` (152,051 paired frames; Anti-UAV val/test + Svanstrom).

## control_v3more_40feat/ — baseline, leaky shortcut
- 40 features: detection summary (8) + image stats (14) + best-box geometry (14) + cross-modal flags (4)
- Top feature `ir_detected` carries 35% importance — model learned "if IR fires, trust IR/both".
- Acc 0.9790, F1-macro 0.9490.

## scene_aware_v3more_32feat/ — deployment pick
- 32 features: 40-feat set minus 8 leaky detection signals
  (`ir_detected`, `rgb_detected`, `both_detect`, `neither_detect`, `rgb_only_detect`,
   `ir_only_detect`, `ir_n_dets`, `rgb_n_dets`).
- Forces classifier to reason from box geometry, confidence, and image content.
- Acc 0.9792, F1-macro 0.9493 — **matches control on every metric** but no leakage shortcut.
- Trained via:
  ```
  python classifier/reliability/fusion/train_fusion.py --no-fn \
    --in-suffix _v3more --out-suffix _v3more_no_det_signals \
    --exclude-features "ir_detected,rgb_detected,both_detect,neither_detect,rgb_only_detect,ir_only_detect,ir_n_dets,rgb_n_dets"
  ```

Each folder has:
- `model.joblib` — bundle: {"model": XGBClassifier, "features": [name, ...]}
- `metrics.json` — full training metrics (per-class, per-dataset, AUC, importance)
- `feature_importance.png` — bar chart
