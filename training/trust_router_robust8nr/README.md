# Trust router: robust8-nr

The shipped trust router, an 8-feature XGBoost classifier with three classes (trust RGB, trust IR, trust
both) and no reject class, so it always routes to a modality.

- Weights: `models/routers/robust8_noreject_drop/model.joblib`
- Trainer: `train_robust8_noreject.py` (in this folder)
- Base / predecessor router (kept under `classifier/`): `classifier/train_routing_robust.py`, which the
  eval harness also imports for its sequence-id helper, so it stays there.

The trainer reads its feature table from `models/routers/optimal_v1/fusion_dataset_full56.csv` and writes
the joblib above. It uses only scikit-learn and XGBoost (no detector imports), so it runs standalone.
