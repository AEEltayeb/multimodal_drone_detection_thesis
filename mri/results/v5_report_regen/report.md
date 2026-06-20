# Model MRI — Yolo26n_selcom_confuser_ft4_1280

**Detector:** `RGB model/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt`  
**Corpus:** resumed feature corpus — 19,334 drone / 13,597 confuser detections (no image stream)

## Verdict

> **Classifier strongly recommended — large FP cut at low recall cost.**

_Evidence: 19334 drones / 13597 confusers; raw hallucination not measured (feature-only input); LDA separability 0.952 (thr 0.90); recall cost 1.1% (thr 10%); projected FP cut 97%_

## Diagnostic signals

| Signal | Value | Meaning |
|---|---|---|
| Raw hallucination rate | 54.4% | FP per confuser image (bare detector; bare FT4 on rgb_confusers @imgsz=1280, n=2633) |
| LDA separability | 0.952 | train-set linear split of drone vs confuser |
| Silhouette | 0.067 | feature-space cluster separation |
| Max ANOVA F | 42346 | strongest single discriminative feature |
| Meta-only max AUROC | 0.811 | best metadata feature alone |
| YOLO-feat max AUROC | 0.844 | best learned feature alone |
| Projected FP cut | 97.4% | confusers the classifier would reject |
| Recall retention | 98.9% | true drones the classifier keeps |
| Projected FP rate | 1.4% | hallucination after classifier |

## Per-dataset scan

| Dataset | Role | Images | Dets | Drones | Confusers | bare TP/FP/FN |
|---|---|---|---|---|---|---|

## Top discriminative features (ANOVA F)

- p5 ch=154
- p3 ch=31 cell=0
- p5 ch=79
- p3 ch=31 cell=1
- p5 ch=113
- meta:conf
- p5 ch=240
- p5 ch=85
- p3 ch=31 cell=2
- p5 ch=43

## Classifier CV (F1)

| Classifier | feature set | CV F1 |
|---|---|---|
| mlp | fused | 0.9857 ± 0.0007 |

## Figures

![pca_fused](images/pca_fused.png)
![lda_fused](images/lda_fused.png)
![class_heatmap](images/class_heatmap.png)
![top_neuron_kde](images/top_neuron_kde.png)
![auroc_by_layer](images/auroc_by_layer.png)
![top_feature_auroc](images/top_feature_auroc.png)
![verifier_roc](images/verifier_roc.png)
![operating_curve](images/operating_curve.png)
![fp_reduction](images/fp_reduction.png)

## Delivered

- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\report.md` — this report
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\features.npz` — extracted feature corpus (X, y, w)
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\stats.json` — all numeric results
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\manifest.json` — CLI args + git SHA
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\mlp.pt` — trained MLP classifier (callable)
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\pca_fused.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\lda_fused.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\class_heatmap.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\top_neuron_kde.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\auroc_by_layer.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\top_feature_auroc.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\verifier_roc.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\operating_curve.png`
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\mri\results\v5_report_regen\images\fp_reduction.png`