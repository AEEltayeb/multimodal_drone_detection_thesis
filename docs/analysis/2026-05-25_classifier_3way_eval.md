# Classifier 3-way eval - 2026-05-25

Eval set: 300 frames (target ~100/source).

Sources: `antiuav`=100, `svanstrom`=100, `video_airplanes`=25, `video_birds`=25, `video_drone`=25, `video_helicopters`=25

Trust labels: `reject_both`=135, `trust_rgb`=18, `trust_ir`=14, `trust_both`=133


## Overall

| classifier | n_features | acc | F1m | F1w | ms/frame |
|---|---:|---:|---:|---:|---:|
| lean13 | 13 | 0.9867 | 0.9787 | 0.9867 | 0.577 |
| 32feat | 32 | 0.9233 | 0.8420 | 0.9175 | 0.754 |
| 40feat | 40 | 0.9533 | 0.9197 | 0.9522 | 0.567 |

## Per-source accuracy

| classifier | antiuav | svanstrom | video_airplanes | video_birds | video_drone | video_helicopters |
|---|---:|---:|---:|---:|---:|---:|
| lean13 | 1.0000 | 0.9800 | 1.0000 | 1.0000 | 0.9200 | 1.0000 |
| 32feat | 0.9900 | 0.9600 | 1.0000 | 1.0000 | 0.2800 | 1.0000 |
| 40feat | 1.0000 | 0.9800 | 0.8400 | 0.9200 | 0.8000 | 0.9600 |

## Per-source F1-macro

| classifier | antiuav | svanstrom | video_airplanes | video_birds | video_drone | video_helicopters |
|---|---:|---:|---:|---:|---:|---:|
| lean13 | 1.0000 | 0.9566 | 1.0000 | 1.0000 | 0.8926 | 1.0000 |
| 32feat | 0.9048 | 0.9012 | 1.0000 | 1.0000 | 0.4432 | 1.0000 |
| 40feat | 1.0000 | 0.9892 | 0.3043 | 0.3194 | 0.7973 | 0.4898 |

## Classification report - lean13
```
              precision    recall  f1-score   support

 reject_both       0.99      0.99      0.99       135
   trust_rgb       1.00      0.94      0.97        18
    trust_ir       0.93      1.00      0.97        14
  trust_both       0.99      0.99      0.99       133

    accuracy                           0.99       300
   macro avg       0.98      0.98      0.98       300
weighted avg       0.99      0.99      0.99       300
```


## Classification report - 32feat
```
              precision    recall  f1-score   support

 reject_both       0.88      0.96      0.92       135
   trust_rgb       0.78      0.39      0.52        18
    trust_ir       0.93      1.00      0.97        14
  trust_both       0.98      0.95      0.96       133

    accuracy                           0.92       300
   macro avg       0.89      0.82      0.84       300
weighted avg       0.92      0.92      0.92       300
```


## Classification report - 40feat
```
              precision    recall  f1-score   support

 reject_both       0.97      0.93      0.95       135
   trust_rgb       0.87      0.72      0.79        18
    trust_ir       0.93      1.00      0.97        14
  trust_both       0.95      1.00      0.97       133

    accuracy                           0.95       300
   macro avg       0.93      0.91      0.92       300
weighted avg       0.95      0.95      0.95       300
```


## Delivered

- CSV: `C:/Users/User/Desktop/UNISA projects/Drone detection/es proj 3 thesis workspace/ES_Drone_Detection/docs/analysis/full_pipeline_ablations/csv/classifier_3way.csv`
- This MD: `C:/Users/User/Desktop/UNISA projects/Drone detection/es proj 3 thesis workspace/ES_Drone_Detection/docs/analysis/2026-05-25_classifier_3way_eval.md`