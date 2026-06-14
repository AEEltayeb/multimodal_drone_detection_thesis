# Notes-round-1 replay extensions (modality A/B, per-size, per-category)
2026-06-14 19:03 | same Tier-1 cache + shipped stack as tier1_results.json
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
| routed[robust8-nr] bare | 7430 | 160 | 243 | 0.9789 | 0.9683 | 0.9736 [0.9705–0.9764] |
| routed[robust8-nr] +filt | 7428 | 157 | 245 | 0.9793 | 0.9681 | 0.9737 [0.9706–0.9765] |

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
| routed[robust8-nr] bare | 3136 | 1997 | 178 | 0.6109 | 0.9463 | 0.7425 [0.7307–0.7547] |
| routed[robust8-nr] +filt | 3037 | 330 | 277 | 0.902 | 0.9164 | 0.9091 [0.9016–0.9163] |

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