# Notes-round-1 replay extensions (modality A/B, per-size, per-category)
2026-06-11 23:22 | same Tier-1 cache + shipped stack as tier1_results.json
PART M uses COVERAGE scoring (missing/distrusted modality's GT counts as FN) — deliberately different from the headline trust-aware rule; see module docstring. 95% bootstrap CIs.


## antiuav  (n=4000 of 4000, rule=iou, imgsz rgb=640/ir=640)

**M — modality A/B (coverage scoring: absent/distrusted side's GT = FN)**

| arm | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| rgb_only bare | 3657 | 41 | 4016 | 0.9889 | 0.4766 | 0.6432 [0.6402–0.646] |
| rgb_only +filt | 3655 | 39 | 4018 | 0.9894 | 0.4763 | 0.6431 [0.64–0.6458] |
| ir_only bare | 3775 | 133 | 3898 | 0.966 | 0.492 | 0.6519 [0.648–0.6556] |
| ir_only +filt | 3775 | 132 | 3898 | 0.9662 | 0.492 | 0.652 [0.6481–0.6557] |
| both bare | 7432 | 174 | 241 | 0.9771 | 0.9686 | 0.9728 [0.9697–0.9757] |
| both +filt | 7430 | 171 | 243 | 0.9775 | 0.9683 | 0.9729 [0.9697–0.9758] |
| routed[robust8] bare | 7408 | 141 | 265 | 0.9813 | 0.9655 | 0.9733 [0.9702–0.9762] |
| routed[robust8] +filt | 7406 | 139 | 267 | 0.9816 | 0.9652 | 0.9733 [0.9703–0.9762] |

**SZ — per-size, ft4/rgb (median GT sqrt-area 93.1 px, n_gt=3725, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| 16-32px | 38 | 0.9268 | 1.0 | 0.962 | 0.9268 | 1.0 | 0.962 |
| 32-64px | 747 | 0.9865 | 0.9813 | 0.9839 | 0.9892 | 0.9786 | 0.9838 |
| >=64px | 2940 | 0.9904 | 0.9816 | 0.986 | 0.9904 | 0.9816 | 0.986 |

**SZ — per-size, v3b/ir (median GT sqrt-area 37.0 px, n_gt=3948, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 23 | 0.6923 | 0.7826 | 0.7347 | 0.6923 | 0.7826 | 0.7347 |
| 16-32px | 938 | 0.9545 | 0.9168 | 0.9353 | 0.9545 | 0.9168 | 0.9353 |
| 32-64px | 2911 | 0.9714 | 0.9691 | 0.9702 | 0.9718 | 0.9691 | 0.9704 |
| >=64px | 76 | 0.987 | 1.0 | 0.9935 | 0.987 | 1.0 | 0.9935 |

## gray_confuser  (n=2633 of 2633, rule=iou, imgsz rgb=640/ir=640)


**CAT — per-category fire rates (filter=mlp_v5_ir_aligned (grayscale scaler))**

| category | n | bare fire [CI] | filt_mlp fire [CI] |
|---|---|---|---|
| airplane | 1133 | 0.1774 [0.1562–0.1977] | 0.0009 [0.0–0.0026] |
| bird | 575 | 0.0157 [0.0052–0.0261] | 0.0 [0.0–0.0] |
| helicopter | 486 | 0.7181 [0.679–0.7573] | 0.0082 [0.0021–0.0165] |
| other | 439 | 0.1526 [0.1185–0.1868] | 0.0342 [0.0182–0.0524] |

## ir_confusers  (n=4000 of 5237, rule=iou, imgsz rgb=640/ir=640)


**CAT — per-category fire rates (filter=mlp_v5_ir_aligned (thermal scaler))**

| category | n | bare fire [CI] | clf[robust8] fire [CI] | filt_mlp fire [CI] | clf->filt[robust8] fire [CI] |
|---|---|---|---|---|---|
| airplane | 3043 | 0.352 [0.3352–0.3687] | 0.3283 [0.3122–0.3444] | 0.278 [0.2626–0.2945] | 0.2553 [0.2399–0.2708] |
| bird | 871 | 0.1217 [0.0999–0.1435] | 0.1079 [0.0872–0.1286] | 0.1171 [0.0953–0.1389] | 0.1033 [0.0827–0.124] |
| helicopter | 86 | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] |

## ir_dset_final  (n=4000 of 9612, rule=iou, imgsz rgb=640/ir=640)


**SZ — per-size, v3b/ir (median GT sqrt-area 26.1 px, n_gt=2615, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 777 | 0.8854 | 0.955 | 0.9189 | 0.8854 | 0.955 | 0.9189 |
| 16-32px | 778 | 0.9697 | 0.9884 | 0.979 | 0.9697 | 0.9871 | 0.9783 |
| 32-64px | 837 | 0.9675 | 0.9964 | 0.9818 | 0.9673 | 0.9892 | 0.9781 |
| >=64px | 223 | 0.9605 | 0.9821 | 0.9712 | 0.9763 | 0.9238 | 0.9493 |

## rgb_confuser  (n=2633 of 2633, rule=iou, imgsz rgb=640/ir=640)


**CAT — per-category fire rates (filter=mlp_v5)**

| category | n | bare fire [CI] | clf[robust8] fire [CI] | filt_mlp fire [CI] | clf->filt[robust8] fire [CI] |
|---|---|---|---|---|---|
| airplane | 1133 | 0.2339 [0.2092–0.2586] | 0.0459 [0.0335–0.0583] | 0.0026 [0.0–0.0062] | 0.0 [0.0–0.0] |
| bird | 575 | 0.3896 [0.3513–0.4313] | 0.0017 [0.0–0.0052] | 0.0296 [0.0157–0.0435] | 0.0 [0.0–0.0] |
| helicopter | 486 | 0.5802 [0.537–0.6255] | 0.1379 [0.1111–0.1688] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] |
| other | 439 | 0.0638 [0.0433–0.0888] | 0.0205 [0.0091–0.0342] | 0.0182 [0.0068–0.0319] | 0.0091 [0.0023–0.0182] |

## rgb_dataset_test  (n=4000 of 17209, rule=iou, imgsz rgb=640/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 47.7 px, n_gt=3452, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 726 | 0.9342 | 0.7824 | 0.8516 | 0.9588 | 0.2562 | 0.4043 |
| 16-32px | 607 | 0.9083 | 0.8649 | 0.8861 | 0.961 | 0.4465 | 0.6097 |
| 32-64px | 702 | 0.9605 | 0.9359 | 0.9481 | 0.9732 | 0.8291 | 0.8954 |
| >=64px | 1417 | 0.979 | 0.9555 | 0.9671 | 0.9825 | 0.9506 | 0.9663 |

## selcom_val  (n=311 of 311, rule=iop, imgsz rgb=1280/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 39.6 px, n_gt=295, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| 16-32px | 50 | 0.8611 | 0.62 | 0.7209 | 0.9688 | 0.62 | 0.7561 |
| 32-64px | 165 | 0.7966 | 0.2848 | 0.4196 | 0.9792 | 0.2848 | 0.4413 |
| >=64px | 80 | 0.9167 | 0.6875 | 0.7857 | 0.9167 | 0.6875 | 0.7857 |

## svanstrom  (n=4000 of 4000, rule=iop, imgsz rgb=1280/ir=640)

**M — modality A/B (coverage scoring: absent/distrusted side's GT = FN)**

| arm | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| rgb_only bare | 1532 | 1845 | 1782 | 0.4537 | 0.4623 | 0.4579 [0.4477–0.4678] |
| rgb_only +filt | 1433 | 179 | 1881 | 0.889 | 0.4324 | 0.5818 [0.5731–0.5902] |
| ir_only bare | 1610 | 174 | 1704 | 0.9025 | 0.4858 | 0.6316 [0.6261–0.6368] |
| ir_only +filt | 1610 | 173 | 1704 | 0.903 | 0.4858 | 0.6317 [0.6263–0.6369] |
| both bare | 3142 | 2019 | 172 | 0.6088 | 0.9481 | 0.7415 [0.7295–0.7536] |
| both +filt | 3043 | 352 | 271 | 0.8963 | 0.9182 | 0.9071 [0.8996–0.9146] |
| routed[robust8] bare | 3092 | 354 | 222 | 0.8973 | 0.933 | 0.9148 [0.9075–0.9213] |
| routed[robust8] +filt | 2993 | 195 | 321 | 0.9388 | 0.9031 | 0.9206 [0.9139–0.9268] |

**SZ — per-size, ft4/rgb (median GT sqrt-area 29.8 px, n_gt=1673, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 27 | 0.0161 | 0.6296 | 0.0314 | 0.1882 | 0.5926 | 0.2857 |
| 16-32px | 994 | 0.6958 | 0.8974 | 0.7838 | 0.9317 | 0.8229 | 0.8739 |
| 32-64px | 633 | 0.6813 | 0.9558 | 0.7955 | 0.9401 | 0.9179 | 0.9289 |
| >=64px | 19 | 0.12 | 0.9474 | 0.213 | 0.5806 | 0.9474 | 0.72 |

**SZ — per-size, v3b/ir (median GT sqrt-area 14.8 px, n_gt=1641, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 993 | 0.9286 | 0.9698 | 0.9488 | 0.9286 | 0.9698 | 0.9488 |
| 16-32px | 643 | 0.8687 | 0.9984 | 0.9291 | 0.8687 | 0.9984 | 0.9291 |
| 32-64px | 5 | 0.7143 | 1.0 | 0.8333 | 0.7143 | 1.0 | 0.8333 |
| >=64px | 0 | 0.0 | — | — | 0.0 | — | — |

## svanstrom_gray  (n=4000 of 4000, rule=iop, imgsz rgb=1280/ir=640)


**SZ — per-size, v3b/ir (median GT sqrt-area 29.8 px, n_gt=1673, filter=mlp_v5_ir_aligned (grayscale scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 27 | 0.0341 | 0.4444 | 0.0633 | 0.125 | 0.037 | 0.0571 |
| 16-32px | 994 | 0.76 | 0.5161 | 0.6147 | 0.9583 | 0.0231 | 0.0452 |
| 32-64px | 633 | 0.7238 | 0.7867 | 0.754 | 0.9583 | 0.1453 | 0.2524 |
| >=64px | 19 | 0.0812 | 0.8421 | 0.1481 | 0.6667 | 0.5263 | 0.5882 |

## svanstrom_rawrgb  (n=4000 of 28710, rule=iop, imgsz rgb=1280/ir=640)


**SZ — per-size, v3b/ir (median GT sqrt-area 29.5 px, n_gt=1630, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 26 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 16-32px | 989 | 0.3254 | 0.0829 | 0.1322 | 0.3397 | 0.0718 | 0.1185 |
| 32-64px | 595 | 0.7947 | 0.2017 | 0.3217 | 0.823 | 0.1563 | 0.2627 |
| >=64px | 20 | 0.058 | 0.2 | 0.0899 | 0.125 | 0.15 | 0.1364 |

## video_drone  (n=1359 of 1359, rule=iop, imgsz rgb=640/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 105.7 px, n_gt=1234, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 44 | 0.9 | 0.2045 | 0.3333 | 0.0 | 0.0 | 0.0 |
| 16-32px | 119 | 0.6984 | 0.3697 | 0.4835 | 0.5 | 0.0336 | 0.063 |
| 32-64px | 308 | 0.5954 | 0.5877 | 0.5915 | 0.7251 | 0.4026 | 0.5177 |
| >=64px | 763 | 0.8883 | 0.6147 | 0.7266 | 0.9422 | 0.4915 | 0.646 |

**SZ — per-size, v3b/ir (median GT sqrt-area 105.7 px, n_gt=1234, filter=mlp_v5_ir_aligned (grayscale scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 44 | 0.65 | 0.2955 | 0.4062 | 0.6667 | 0.2273 | 0.339 |
| 16-32px | 119 | 0.4035 | 0.3866 | 0.3948 | 0.1154 | 0.0504 | 0.0702 |
| 32-64px | 308 | 0.7077 | 0.4481 | 0.5487 | 0.7176 | 0.1981 | 0.3104 |
| >=64px | 763 | 0.6933 | 0.4325 | 0.5327 | 0.8114 | 0.2425 | 0.3734 |