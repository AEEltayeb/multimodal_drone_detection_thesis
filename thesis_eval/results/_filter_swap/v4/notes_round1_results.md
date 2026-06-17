# Notes-round-1 replay extensions (modality A/B, per-size, per-category)
2026-06-17 21:03 | same Tier-1 cache + shipped stack as tier1_results.json
PART M uses COVERAGE scoring (missing/distrusted modality's GT counts as FN) — deliberately different from the headline trust-aware rule; see module docstring. 95% bootstrap CIs.


## antiuav  (n=4000 of 4000, rule=iou, imgsz rgb=640/ir=640)

**M — modality A/B (coverage scoring: absent/distrusted side's GT = FN)**

| arm | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| rgb_only bare | 3657 | 41 | 4016 | 0.9889 | 0.4766 | 0.6432 [0.6402–0.646] |
| rgb_only +filt | 3654 | 36 | 4019 | 0.9902 | 0.4762 | 0.6431 [0.6402–0.6458] |
| ir_only bare | 3775 | 133 | 3898 | 0.966 | 0.492 | 0.6519 [0.648–0.6556] |
| ir_only +filt | 3773 | 133 | 3900 | 0.9659 | 0.4917 | 0.6517 [0.6477–0.6554] |
| both bare | 7432 | 174 | 241 | 0.9771 | 0.9686 | 0.9728 [0.9697–0.9757] |
| both +filt | 7427 | 169 | 246 | 0.9778 | 0.9679 | 0.9728 [0.9696–0.9757] |
| routed[robust8] bare | 7408 | 141 | 265 | 0.9813 | 0.9655 | 0.9733 [0.9702–0.9762] |
| routed[robust8] +filt | 7403 | 137 | 270 | 0.9818 | 0.9648 | 0.9732 [0.97–0.9761] |
| routed[robust8-nr] bare | 7430 | 160 | 243 | 0.9789 | 0.9683 | 0.9736 [0.9705–0.9764] |
| routed[robust8-nr] +filt | 7425 | 155 | 248 | 0.9796 | 0.9677 | 0.9736 [0.9705–0.9764] |

**SZ — per-size, ft4/rgb (median GT sqrt-area 93.1 px, n_gt=3725, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| 16-32px | 38 | 0.9268 | 1.0 | 0.962 | 0.95 | 1.0 | 0.9744 |
| 32-64px | 747 | 0.9865 | 0.9813 | 0.9839 | 0.9905 | 0.9772 | 0.9838 |
| >=64px | 2940 | 0.9904 | 0.9816 | 0.986 | 0.9907 | 0.9816 | 0.9862 |

**SZ — per-size, v3b/ir (median GT sqrt-area 37.0 px, n_gt=3948, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 23 | 0.6923 | 0.7826 | 0.7347 | 0.6923 | 0.7826 | 0.7347 |
| 16-32px | 938 | 0.9545 | 0.9168 | 0.9353 | 0.9545 | 0.9168 | 0.9353 |
| 32-64px | 2911 | 0.9714 | 0.9691 | 0.9702 | 0.9714 | 0.9684 | 0.9699 |
| >=64px | 76 | 0.987 | 1.0 | 0.9935 | 0.987 | 1.0 | 0.9935 |

## antiuav_clean  (n=57542 of 57542, rule=iou, imgsz rgb=640/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 92.6 px, n_gt=54823, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| 16-32px | 198 | 0.87 | 0.9798 | 0.9216 | 0.9317 | 0.9646 | 0.9479 |
| 32-64px | 12082 | 0.9854 | 0.9857 | 0.9856 | 0.9867 | 0.9828 | 0.9847 |
| >=64px | 42543 | 0.9911 | 0.9865 | 0.9888 | 0.9912 | 0.9863 | 0.9888 |

**SZ — per-size, v3b/ir (median GT sqrt-area 37.2 px, n_gt=57025, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 351 | 0.7173 | 0.7664 | 0.741 | 0.7212 | 0.7664 | 0.7431 |
| 16-32px | 12731 | 0.9544 | 0.9244 | 0.9391 | 0.9544 | 0.9244 | 0.9391 |
| 32-64px | 43020 | 0.9789 | 0.9702 | 0.9745 | 0.979 | 0.9698 | 0.9743 |
| >=64px | 923 | 0.9914 | 0.9989 | 0.9951 | 0.9914 | 0.9989 | 0.9951 |

## dut_antiuav_640  (n=2200 of 2200, rule=iou, imgsz rgb=640/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 40.9 px, n_gt=2245, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 83 | 0.8571 | 0.5783 | 0.6906 | 0.8571 | 0.5783 | 0.6906 |
| 16-32px | 765 | 0.9545 | 0.7399 | 0.8336 | 0.9585 | 0.7255 | 0.8259 |
| 32-64px | 586 | 0.9498 | 0.8072 | 0.8727 | 0.9521 | 0.7799 | 0.8574 |
| >=64px | 811 | 0.9504 | 0.8261 | 0.8839 | 0.9539 | 0.7904 | 0.8645 |

**SZ — per-size, v3b/ir (median GT sqrt-area 40.9 px, n_gt=2245, filter=mlp_v5_ir_aligned (grayscale scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 83 | 0.3565 | 0.494 | 0.4141 | 0.3441 | 0.3855 | 0.3636 |
| 16-32px | 765 | 0.592 | 0.4458 | 0.5086 | 0.5851 | 0.3686 | 0.4523 |
| 32-64px | 586 | 0.8242 | 0.384 | 0.5239 | 0.8082 | 0.302 | 0.4398 |
| >=64px | 811 | 0.946 | 0.4106 | 0.5727 | 0.9618 | 0.3107 | 0.4697 |

## dut_antiuav_960  (n=2200 of 2200, rule=iou, imgsz rgb=960/ir=960)


**SZ — per-size, ft4/rgb (median GT sqrt-area 40.9 px, n_gt=2245, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 83 | 0.9571 | 0.8072 | 0.8758 | 1.0 | 0.7711 | 0.8707 |
| 16-32px | 765 | 0.9621 | 0.8954 | 0.9276 | 0.972 | 0.8157 | 0.887 |
| 32-64px | 586 | 0.956 | 0.8908 | 0.9223 | 0.9721 | 0.773 | 0.8612 |
| >=64px | 811 | 0.9278 | 0.7928 | 0.8551 | 0.9347 | 0.6535 | 0.7692 |

**SZ — per-size, v3b/ir (median GT sqrt-area 40.9 px, n_gt=2245, filter=mlp_v5_ir_aligned (grayscale scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 83 | 0.3168 | 0.6145 | 0.418 | 0.3043 | 0.506 | 0.3801 |
| 16-32px | 765 | 0.6757 | 0.5556 | 0.6098 | 0.6627 | 0.4418 | 0.5302 |
| 32-64px | 586 | 0.8682 | 0.4386 | 0.5828 | 0.8583 | 0.3618 | 0.509 |
| >=64px | 811 | 0.9453 | 0.4686 | 0.6265 | 0.9647 | 0.3711 | 0.5361 |

## gray_confuser  (n=2633 of 2633, rule=iou, imgsz rgb=640/ir=640)


**CAT — per-category fire rates (filter=mlp_v5_ir_aligned (grayscale scaler))**

| category | n | bare fire [CI] | filt_mlp fire [CI] |
|---|---|---|---|
| airplane | 1133 | 0.1774 [0.1562–0.1977] | 0.0009 [0.0–0.0026] |
| bird | 575 | 0.0157 [0.0052–0.0261] | 0.0 [0.0–0.0] |
| helicopter | 486 | 0.7181 [0.679–0.7573] | 0.0041 [0.0–0.0103] |
| other | 439 | 0.1526 [0.1185–0.1868] | 0.0251 [0.0114–0.041] |

## ir_confusers  (n=4000 of 5237, rule=iou, imgsz rgb=640/ir=640)


**CAT — per-category fire rates (filter=mlp_v5_ir_aligned (thermal scaler))**

| category | n | bare fire [CI] | clf[robust8] fire [CI] | filt_mlp fire [CI] | clf->filt[robust8] fire [CI] |
|---|---|---|---|---|---|
| airplane | 3043 | 0.352 [0.3352–0.3687] | 0.3283 [0.3122–0.3444] | 0.0835 [0.0743–0.0933] | 0.0753 [0.0664–0.0845] |
| bird | 871 | 0.1217 [0.0999–0.1435] | 0.1079 [0.0872–0.1286] | 0.0448 [0.0321–0.0597] | 0.0356 [0.0241–0.0482] |
| helicopter | 86 | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] |

## ir_dset_final  (n=4000 of 9612, rule=iou, imgsz rgb=640/ir=640)


**SZ — per-size, v3b/ir (median GT sqrt-area 26.1 px, n_gt=2615, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 777 | 0.8854 | 0.955 | 0.9189 | 0.883 | 0.9228 | 0.9025 |
| 16-32px | 778 | 0.9697 | 0.9884 | 0.979 | 0.9723 | 0.9473 | 0.9596 |
| 32-64px | 837 | 0.9675 | 0.9964 | 0.9818 | 0.9673 | 0.9534 | 0.9603 |
| >=64px | 223 | 0.9605 | 0.9821 | 0.9712 | 0.9784 | 0.8117 | 0.8873 |

## rgb_confuser  (n=2633 of 2633, rule=iou, imgsz rgb=640/ir=640)


**CAT — per-category fire rates (filter=mlp_v5)**

| category | n | bare fire [CI] | clf[robust8] fire [CI] | filt_mlp fire [CI] | clf->filt[robust8] fire [CI] |
|---|---|---|---|---|---|
| airplane | 1133 | 0.2339 [0.2092–0.2586] | 0.0459 [0.0335–0.0583] | 0.0026 [0.0–0.0062] | 0.0 [0.0–0.0] |
| bird | 575 | 0.3896 [0.3513–0.4313] | 0.0017 [0.0–0.0052] | 0.0417 [0.0261–0.0591] | 0.0 [0.0–0.0] |
| helicopter | 486 | 0.5802 [0.537–0.6255] | 0.1379 [0.1111–0.1688] | 0.0062 [0.0–0.0144] | 0.0021 [0.0–0.0062] |
| other | 439 | 0.0638 [0.0433–0.0888] | 0.0205 [0.0091–0.0342] | 0.0182 [0.0068–0.0319] | 0.0046 [0.0–0.0114] |

## rgb_dataset_test  (n=4000 of 17209, rule=iou, imgsz rgb=640/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 47.7 px, n_gt=3452, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 726 | 0.9342 | 0.7824 | 0.8516 | 0.9361 | 0.7672 | 0.8433 |
| 16-32px | 607 | 0.9083 | 0.8649 | 0.8861 | 0.9259 | 0.8435 | 0.8828 |
| 32-64px | 702 | 0.9605 | 0.9359 | 0.9481 | 0.9671 | 0.9202 | 0.9431 |
| >=64px | 1417 | 0.979 | 0.9555 | 0.9671 | 0.9804 | 0.9513 | 0.9656 |

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
| rgb_only +filt | 1423 | 163 | 1891 | 0.8972 | 0.4294 | 0.5808 [0.5721–0.5896] |
| ir_only bare | 1610 | 174 | 1704 | 0.9025 | 0.4858 | 0.6316 [0.6261–0.6368] |
| ir_only +filt | 1610 | 173 | 1704 | 0.903 | 0.4858 | 0.6317 [0.6263–0.6369] |
| both bare | 3142 | 2019 | 172 | 0.6088 | 0.9481 | 0.7415 [0.7295–0.7536] |
| both +filt | 3033 | 336 | 281 | 0.9003 | 0.9152 | 0.9077 [0.9005–0.9151] |
| routed[robust8] bare | 3092 | 354 | 222 | 0.8973 | 0.933 | 0.9148 [0.9075–0.9213] |
| routed[robust8] +filt | 2983 | 187 | 331 | 0.941 | 0.9001 | 0.9201 [0.9134–0.9265] |
| routed[robust8-nr] bare | 3136 | 1997 | 178 | 0.6109 | 0.9463 | 0.7425 [0.7307–0.7547] |
| routed[robust8-nr] +filt | 3027 | 314 | 287 | 0.906 | 0.9134 | 0.9097 [0.9024–0.9169] |

**SZ — per-size, ft4/rgb (median GT sqrt-area 29.8 px, n_gt=1673, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 27 | 0.0161 | 0.6296 | 0.0314 | 0.1889 | 0.6296 | 0.2906 |
| 16-32px | 994 | 0.6958 | 0.8974 | 0.7838 | 0.9301 | 0.8169 | 0.8698 |
| 32-64px | 633 | 0.6813 | 0.9558 | 0.7955 | 0.9616 | 0.91 | 0.9351 |
| >=64px | 19 | 0.12 | 0.9474 | 0.213 | 0.75 | 0.9474 | 0.8372 |

**SZ — per-size, v3b/ir (median GT sqrt-area 14.8 px, n_gt=1641, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 993 | 0.9286 | 0.9698 | 0.9488 | 0.9286 | 0.9698 | 0.9488 |
| 16-32px | 643 | 0.8687 | 0.9984 | 0.9291 | 0.8687 | 0.9984 | 0.9291 |
| 32-64px | 5 | 0.7143 | 1.0 | 0.8333 | 0.7143 | 1.0 | 0.8333 |
| >=64px | 0 | 0.0 | — | — | 0.0 | — | — |

## svanstrom_clean  (n=5557 of 5557, rule=iop, imgsz rgb=1280/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 29.0 px, n_gt=2074, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 50 | 0.0311 | 0.98 | 0.0603 | 0.2316 | 0.88 | 0.3667 |
| 16-32px | 1232 | 0.737 | 0.9213 | 0.8189 | 0.9612 | 0.8442 | 0.8989 |
| 32-64px | 753 | 0.5345 | 0.9668 | 0.6884 | 0.9705 | 0.9615 | 0.966 |
| >=64px | 39 | 0.1402 | 0.9744 | 0.2452 | 0.6863 | 0.8974 | 0.7778 |

**SZ — per-size, v3b/ir (median GT sqrt-area 14.6 px, n_gt=1961, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 1307 | 0.8507 | 0.8891 | 0.8694 | 0.8507 | 0.8891 | 0.8694 |
| 16-32px | 654 | 0.7631 | 1.0 | 0.8657 | 0.7631 | 1.0 | 0.8657 |
| >=64px | 0 | 0.0 | — | — | 0.0 | — | — |

## svanstrom_gray  (n=4000 of 4000, rule=iop, imgsz rgb=1280/ir=640)


**SZ — per-size, v3b/ir (median GT sqrt-area 29.8 px, n_gt=1673, filter=mlp_v5_ir_aligned (grayscale scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 27 | 0.0341 | 0.4444 | 0.0633 | 0.0 | 0.0 | 0.0 |
| 16-32px | 994 | 0.76 | 0.5161 | 0.6147 | 0.92 | 0.0231 | 0.0451 |
| 32-64px | 633 | 0.7238 | 0.7867 | 0.754 | 0.9519 | 0.1564 | 0.2687 |
| >=64px | 19 | 0.0812 | 0.8421 | 0.1481 | 0.7143 | 0.5263 | 0.6061 |

## svanstrom_rawrgb  (n=4000 of 28710, rule=iop, imgsz rgb=1280/ir=640)


**SZ — per-size, v3b/ir (median GT sqrt-area 29.5 px, n_gt=1630, filter=mlp_v5_ir_aligned (thermal scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 26 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 16-32px | 989 | 0.3254 | 0.0829 | 0.1322 | 0.3122 | 0.0597 | 0.1002 |
| 32-64px | 595 | 0.7947 | 0.2017 | 0.3217 | 0.7708 | 0.1244 | 0.2142 |
| >=64px | 20 | 0.058 | 0.2 | 0.0899 | 0.0968 | 0.15 | 0.1176 |

## video_drone  (n=1359 of 1359, rule=iop, imgsz rgb=640/ir=640)


**SZ — per-size, ft4/rgb (median GT sqrt-area 105.7 px, n_gt=1234, filter=mlp_v5)**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 44 | 0.9 | 0.2045 | 0.3333 | 0.0 | 0.0 | 0.0 |
| 16-32px | 119 | 0.6984 | 0.3697 | 0.4835 | 0.7222 | 0.1092 | 0.1898 |
| 32-64px | 308 | 0.5954 | 0.5877 | 0.5915 | 0.7075 | 0.3377 | 0.4571 |
| >=64px | 763 | 0.8883 | 0.6147 | 0.7266 | 0.9384 | 0.519 | 0.6684 |

**SZ — per-size, v3b/ir (median GT sqrt-area 105.7 px, n_gt=1234, filter=mlp_v5_ir_aligned (grayscale scaler))**

| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |
|---|---|---|---|---|---|---|---|
| <16px | 44 | 0.65 | 0.2955 | 0.4062 | 0.7692 | 0.2273 | 0.3509 |
| 16-32px | 119 | 0.4035 | 0.3866 | 0.3948 | 0.1837 | 0.0756 | 0.1071 |
| 32-64px | 308 | 0.7077 | 0.4481 | 0.5487 | 0.7792 | 0.1948 | 0.3117 |
| >=64px | 763 | 0.6933 | 0.4325 | 0.5327 | 0.8944 | 0.1664 | 0.2807 |