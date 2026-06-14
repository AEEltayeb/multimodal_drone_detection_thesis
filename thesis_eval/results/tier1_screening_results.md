# Thesis Eval — Tier-1 (FINAL thesis numbers, locked 2026-06-10)
2026-06-11 18:55 | detectors ft4+v3b | patch_thr=0.5 | robust8 tau=0.2 | mlp thr rgb=0.25 / ir=0.05 / gray=0.25
Even-strided ~4k cap per surface (all frames where smaller); n and n_source printed. Per-frame grain; temporal/segment evidence = the separate real-video run. 95% bootstrap CIs (frame resample, 1000 iters) in brackets.


## antiuav  (n=4000 of 4000, kind=paired, rule=iou, imgsz rgb=640/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 3657 | 41 | 68 | 0.9889 | 0.9817 | 0.9853 [0.982–0.9882] | 3725 |
| v3b/ir | 3775 | 133 | 173 | 0.966 | 0.9562 | 0.961 [0.9562–0.9657] | 3948 |

**B — full-pipeline ablation (per-modality scoring, NO union)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 7432 | 174 | 241 | 0.9771 | 0.9686 | 0.9728 [0.9697–0.9757] |
| filt_mlp | 7430 | 171 | 243 | 0.9775 | 0.9683 | 0.9729 [0.9697–0.9758] |
| filt_patch | 7430 | 174 | 243 | 0.9771 | 0.9683 | 0.9727 [0.9695–0.9757] |
| clf[robust8] | 7408 | 141 | 93 | 0.9813 | 0.9876 | 0.9845 [0.9819–0.9868] |
| clf->filt[robust8] | 7406 | 139 | 95 | 0.9816 | 0.9873 | 0.9844 [0.9819–0.9868] |
| clf[robust6] | 7384 | 156 | 170 | 0.9793 | 0.9775 | 0.9784 [0.9754–0.9814] |
| clf->filt[robust6] | 7382 | 154 | 172 | 0.9796 | 0.9772 | 0.9784 [0.9754–0.9813] |
| clf[sa32] | 7426 | 135 | 75 | 0.9821 | 0.99 | 0.9861 [0.9837–0.9882] |
| clf->filt[sa32] | 7424 | 134 | 77 | 0.9823 | 0.9897 | 0.986 [0.9836–0.988] |
| filt->clf[robust8] | 7406 | 139 | 93 | 0.9816 | 0.9876 | 0.9846 [0.982–0.9869] |

## gray_confuser  (n=2633 of 2633, kind=gray, rule=iou, imgsz rgb=640/ir=640, drones=False)


**C — confuser FP-reduction (no GT; every surviving det = FP; verifier=mlp_v5_ir_aligned (grayscale scaler))**

| stage | FP | fire_rate [95% CI] |
|---|---|---|
| bare | 656 | 0.2378 [0.2218–0.2552] |
| filt_mlp | 21 | 0.0076 [0.0042–0.011] |
| filt_patch | 563 | 0.2043 [0.1888–0.2218] |

**GRAY operating-point sweep (aligned_gray threshold; cached probs)**

| thr | FP | fire_rate |
|---|---|---|
| 0.02 | 198 | 0.0729 |
| 0.05 | 99 | 0.0357 |
| 0.1 | 49 | 0.0179 |
| 0.15 | 32 | 0.0114 |
| 0.2 | 25 | 0.0091 |
| 0.25 | 21 | 0.0076 |

## ir_confusers  (n=4000 of 5237, kind=ir, rule=iou, imgsz rgb=640/ir=640, drones=False)


**C — confuser FP-reduction (no GT; every surviving det = FP; verifier=mlp_v5_ir_aligned (thermal scaler))**

| stage | FP | fire_rate [95% CI] |
|---|---|---|
| bare | 1203 | 0.2943 [0.2797–0.3085] |
| filt_mlp | 968 | 0.237 [0.2237–0.2497] |
| filt_patch | 1007 | 0.2462 [0.233–0.2598] |
| clf[robust8] | 1117 | 0.2732 [0.2587–0.2868] |
| clf[robust6] | 905 | 0.2225 [0.2087–0.2352] |
| clf[sa32] | 1069 | 0.2627 [0.2487–0.2768] |
| clf->filt[robust8] | 885 | 0.2167 [0.2037–0.2295] |
| clf->filt[robust6] | 728 | 0.1792 [0.1668–0.1908] |
| clf->filt[sa32] | 840 | 0.207 [0.194–0.2188] |

## ir_dset_final  (n=4000 of 9612, kind=ir, rule=iou, imgsz rgb=640/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| v3b/ir | 2564 | 157 | 51 | 0.9423 | 0.9805 | 0.961 [0.9552–0.967] | 2615 |

**S4 — verifier-only ablation (single modality, verifier=mlp_v5_ir_aligned (thermal scaler))**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 2564 | 157 | 51 | 0.9423 | 0.9805 | 0.961 [0.9552–0.967] |
| filt_mlp | 2544 | 153 | 71 | 0.9433 | 0.9728 | 0.9578 [0.9518–0.964] |
| filt_patch | 2435 | 132 | 180 | 0.9486 | 0.9312 | 0.9398 [0.933–0.9471] |
| clf[robust8] | 2415 | 134 | 200 | 0.9474 | 0.9235 | 0.9353 [0.9281–0.9424] |
| clf->filt[robust8] | 2396 | 131 | 219 | 0.9482 | 0.9163 | 0.9319 [0.9246–0.9391] |
| clf[robust6] | 2140 | 105 | 475 | 0.9532 | 0.8184 | 0.8807 [0.8706–0.8897] |
| clf->filt[robust6] | 2129 | 105 | 486 | 0.953 | 0.8141 | 0.8781 [0.8678–0.8872] |
| clf[sa32] | 2356 | 130 | 259 | 0.9477 | 0.901 | 0.9237 [0.9156–0.9308] |
| clf->filt[sa32] | 2337 | 126 | 278 | 0.9488 | 0.8937 | 0.9204 [0.9121–0.9278] |

## rgb_confuser  (n=2633 of 2633, kind=rgb, rule=iou, imgsz rgb=640/ir=640, drones=False)


**C — confuser FP-reduction (no GT; every surviving det = FP; verifier=mlp_v5)**

| stage | FP | fire_rate [95% CI] |
|---|---|---|
| bare | 835 | 0.3035 [0.2867–0.3209] |
| filt_mlp | 29 | 0.0106 [0.0068–0.0148] |
| filt_patch | 282 | 0.1022 [0.0912–0.1139] |
| clf[robust8] | 134 | 0.049 [0.041–0.0573] |
| clf[robust6] | 95 | 0.0353 [0.0281–0.0425] |
| clf[sa32] | 7 | 0.0027 [0.0008–0.0049] |
| clf->filt[robust8] | 4 | 0.0015 [0.0004–0.003] |
| clf->filt[robust6] | 3 | 0.0011 [0.0–0.0027] |
| clf->filt[sa32] | 2 | 0.0008 [0.0–0.0019] |

## rgb_dataset_test  (n=4000 of 17209, kind=rgb, rule=iou, imgsz rgb=640/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 3104 | 149 | 348 | 0.9542 | 0.8992 | 0.9259 [0.9186–0.9328] | 3452 |

**S4 — verifier-only ablation (single modality, verifier=mlp_v5)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 3104 | 149 | 348 | 0.9542 | 0.8992 | 0.9259 [0.9186–0.9328] |
| filt_mlp | 2386 | 59 | 1066 | 0.9759 | 0.6912 | 0.8092 [0.7965–0.823] |
| filt_patch | 2871 | 130 | 581 | 0.9567 | 0.8317 | 0.8898 [0.8814–0.8982] |
| clf[robust8] | 1322 | 26 | 2130 | 0.9807 | 0.383 | 0.5508 [0.5324–0.5703] |
| clf->filt[robust8] | 1302 | 22 | 2150 | 0.9834 | 0.3772 | 0.5452 [0.5266–0.5642] |
| clf[robust6] | 1419 | 27 | 2033 | 0.9813 | 0.4111 | 0.5794 [0.5598–0.5969] |
| clf->filt[robust6] | 1360 | 17 | 2092 | 0.9877 | 0.394 | 0.5633 [0.5437–0.5816] |
| clf[sa32] | 1476 | 28 | 1976 | 0.9814 | 0.4276 | 0.5956 [0.5773–0.6134] |
| clf->filt[sa32] | 1461 | 20 | 1991 | 0.9865 | 0.4232 | 0.5923 [0.5742–0.6102] |

## selcom_val  (n=311 of 311, kind=rgb, rule=iop, imgsz rgb=1280/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 133 | 22 | 162 | 0.8581 | 0.4508 | 0.5911 [0.5315–0.6393] | 295 |

**S4 — verifier-only ablation (single modality, verifier=mlp_v5)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 133 | 22 | 162 | 0.8581 | 0.4508 | 0.5911 [0.5315–0.6393] |
| filt_mlp | 133 | 7 | 162 | 0.95 | 0.4508 | 0.6115 [0.5511–0.6593] |
| filt_patch | 133 | 22 | 162 | 0.8581 | 0.4508 | 0.5911 [0.5315–0.6393] |
| clf[robust8] | 37 | 5 | 258 | 0.881 | 0.1254 | 0.2196 [0.161–0.2761] |
| clf->filt[robust8] | 37 | 3 | 258 | 0.925 | 0.1254 | 0.2209 [0.1615–0.2776] |
| clf[robust6] | 11 | 0 | 284 | 1.0 | 0.0373 | 0.0719 [0.0332–0.115] |
| clf->filt[robust6] | 11 | 0 | 284 | 1.0 | 0.0373 | 0.0719 [0.0332–0.115] |
| clf[sa32] | 74 | 11 | 221 | 0.8706 | 0.2508 | 0.3895 [0.3259–0.4422] |
| clf->filt[sa32] | 74 | 5 | 221 | 0.9367 | 0.2508 | 0.3957 [0.3305–0.4496] |

## svanstrom  (n=4000 of 4000, kind=paired, rule=iop, imgsz rgb=1280/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 1532 | 1845 | 141 | 0.4537 | 0.9157 | 0.6067 [0.59–0.6228] | 1673 |
| v3b/ir | 1610 | 174 | 31 | 0.9025 | 0.9811 | 0.9401 [0.9313–0.9487] | 1641 |

**B — full-pipeline ablation (per-modality scoring, NO union)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 3142 | 2019 | 172 | 0.6088 | 0.9481 | 0.7415 [0.7295–0.7536] |
| filt_mlp | 3043 | 352 | 271 | 0.8963 | 0.9182 | 0.9071 [0.8996–0.9146] |
| filt_patch | 3004 | 782 | 310 | 0.7934 | 0.9065 | 0.8462 [0.8363–0.8552] |
| clf[robust8] | 3092 | 354 | 31 | 0.8973 | 0.9901 | 0.9414 [0.9352–0.9471] |
| clf->filt[robust8] | 2993 | 195 | 130 | 0.9388 | 0.9584 | 0.9485 [0.9425–0.9541] |
| clf[robust6] | 3132 | 286 | 34 | 0.9163 | 0.9893 | 0.9514 [0.9456–0.9568] |
| clf->filt[robust6] | 3033 | 208 | 133 | 0.9358 | 0.958 | 0.9468 [0.941–0.9522] |
| clf[sa32] | 3137 | 187 | 30 | 0.9437 | 0.9905 | 0.9666 [0.9617–0.9713] |
| clf->filt[sa32] | 3039 | 162 | 128 | 0.9494 | 0.9596 | 0.9545 [0.9492–0.9595] |
| filt->clf[robust8] | 2996 | 196 | 35 | 0.9386 | 0.9885 | 0.9629 [0.9574–0.9676] |

## svanstrom_gray  (n=4000 of 4000, kind=gray, rule=iop, imgsz rgb=1280/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| v3b/ir | 1039 | 873 | 634 | 0.5434 | 0.621 | 0.5796 [0.5606–0.5992] | 1673 |

**S4 — verifier-only ablation (single modality, verifier=mlp_v5_ir_aligned (grayscale scaler))**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 1039 | 873 | 634 | 0.5434 | 0.621 | 0.5796 [0.5606–0.5992] |
| filt_mlp | 126 | 17 | 1547 | 0.8811 | 0.0753 | 0.1388 [0.1182–0.1616] |
| filt_patch | 1035 | 595 | 638 | 0.635 | 0.6186 | 0.6267 [0.6075–0.6463] |

**GRAY operating-point sweep (aligned_gray threshold; cached probs)**

| thr | P | R | F1 |
|---|---|---|---|
| 0.02 | 0.6488 | 0.2672 | 0.3785 |
| 0.05 | 0.7634 | 0.1698 | 0.2778 |
| 0.1 | 0.8362 | 0.116 | 0.2037 |
| 0.15 | 0.8659 | 0.0926 | 0.1674 |
| 0.2 | 0.875 | 0.0837 | 0.1528 |
| 0.25 | 0.8811 | 0.0753 | 0.1388 |

## svanstrom_rawrgb  (n=4000 of 28710, kind=rawrgb, rule=iop, imgsz rgb=1280/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| v3b/ir | 206 | 363 | 1424 | 0.362 | 0.1264 | 0.1874 [0.1649–0.2097] | 1630 |

**S4 — verifier-only ablation (single modality, verifier=mlp_v5_ir_aligned (thermal scaler))**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 206 | 363 | 1424 | 0.362 | 0.1264 | 0.1874 [0.1649–0.2097] |
| filt_mlp | 167 | 261 | 1463 | 0.3902 | 0.1025 | 0.1623 [0.1411–0.184] |
| filt_patch | 205 | 242 | 1425 | 0.4586 | 0.1258 | 0.1974 [0.1739–0.2215] |

## video_confuser  (n=1250 of 1250, kind=grayrgb_paired, rule=iop, imgsz rgb=640/ir=640, drones=False)


## video_drone  (n=1359 of 1359, kind=grayrgb_paired, rule=iop, imgsz rgb=640/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 703 | 202 | 531 | 0.7768 | 0.5697 | 0.6573 [0.6348–0.6788] | 1234 |
| v3b/ir | 527 | 278 | 707 | 0.6547 | 0.4271 | 0.5169 [0.491–0.5417] | 1234 |

**B — full-pipeline ablation (per-modality scoring, NO union)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 1230 | 480 | 1238 | 0.7193 | 0.4984 | 0.5888 [0.5688–0.6085] |
| filt_mlp | 765 | 192 | 1703 | 0.7994 | 0.31 | 0.4467 [0.4255–0.4682] |
| filt_patch | 1032 | 407 | 1436 | 0.7172 | 0.4182 | 0.5283 [0.5078–0.5489] |
| clf[robust8] | 943 | 240 | 1206 | 0.7971 | 0.4388 | 0.566 [0.5394–0.591] |
| clf->filt[robust8] | 604 | 84 | 1545 | 0.8779 | 0.2811 | 0.4258 [0.3999–0.4531] |
| clf[robust6] | 1039 | 232 | 1122 | 0.8175 | 0.4808 | 0.6055 [0.582–0.6307] |
| clf->filt[robust6] | 662 | 89 | 1499 | 0.8815 | 0.3063 | 0.4547 [0.4293–0.4811] |
| clf[sa32] | 1020 | 351 | 1173 | 0.744 | 0.4651 | 0.5724 [0.5484–0.5963] |
| clf->filt[sa32] | 640 | 131 | 1553 | 0.8301 | 0.2918 | 0.4318 [0.4076–0.456] |
| filt->clf[robust8] | 578 | 77 | 1559 | 0.8824 | 0.2705 | 0.414 [0.3847–0.4428] |

## D — GRAYSCALE FINDING (good-only config): IR-on-gray + aligned-gray filter vs RGB

| config | P | R | F1 [95% CI] |
|---|---|---|---|
| RGB (ft4) bare on Svanström | 0.4537 | 0.9157 | 0.6067 [0.59–0.6228] |
| IR on RAW RGB (control) | 0.362 | 0.1264 | 0.1874 [0.1649–0.2097] |
| IR-on-gray (v3b) bare | 0.5434 | 0.621 | 0.5796 [0.5606–0.5992] |
| IR-on-gray + aligned_gray filter (clf bypassed) | 0.8811 | 0.0753 | 0.1388 [0.1182–0.1616] |

## SPEED (from knowledge/ledger — NOT this replay)

| component | sad (ms) | happy (ms) | speedup | source |
|---|---|---|---|---|
| trust classifier | fusion_no_fn 38.3 /frame | robust8 0.095 /frame | ~404× | ledger bench |
| confuser filter | patch 59–112 /det | mlp_v5 1.3–2.1 /det | ~37–72× | ledger bench |
_Pipeline overhead ~1–4%. Verify via eval/bench_speed.py; wire to kb before thesis._