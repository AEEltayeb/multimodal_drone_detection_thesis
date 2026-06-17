# Thesis Eval — Tier-1 (FINAL thesis numbers, locked 2026-06-10)
2026-06-17 23:18 | detectors ft4+v3b | patch_thr=0.5 | robust8 tau=0.2 | mlp thr rgb=0.25 / ir=0.05 / gray=0.25
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
| filt_mlp | 7429 | 168 | 244 | 0.9779 | 0.9682 | 0.973 [0.9698–0.9759] |
| filt_mlp_rgb | 7429 | 169 | 244 | 0.9778 | 0.9682 | 0.973 [0.9697–0.9758] |
| filt_mlp_ir | 7432 | 173 | 241 | 0.9773 | 0.9686 | 0.9729 [0.9697–0.9759] |
| filt_patch | 7430 | 174 | 243 | 0.9771 | 0.9683 | 0.9727 [0.9695–0.9757] |
| clf[robust8] | 7408 | 141 | 93 | 0.9813 | 0.9876 | 0.9845 [0.9819–0.9868] |
| clf->filt[robust8] | 7405 | 136 | 96 | 0.982 | 0.9872 | 0.9846 [0.9819–0.987] |
| clf[robust6] | 7384 | 156 | 170 | 0.9793 | 0.9775 | 0.9784 [0.9754–0.9814] |
| clf->filt[robust6] | 7381 | 151 | 173 | 0.98 | 0.9771 | 0.9785 [0.9755–0.9814] |
| clf[sa32] | 7426 | 135 | 75 | 0.9821 | 0.99 | 0.9861 [0.9837–0.9882] |
| clf->filt[sa32] | 7423 | 133 | 78 | 0.9824 | 0.9896 | 0.986 [0.9837–0.9881] |
| clf[robust8_nr_drop] | 7430 | 160 | 81 | 0.9789 | 0.9892 | 0.984 [0.9814–0.9864] |
| clf->filt[robust8_nr_drop] | 7427 | 154 | 84 | 0.9797 | 0.9888 | 0.9842 [0.9816–0.9866] |
| clf[robust8_nr_both] | 7430 | 161 | 119 | 0.9788 | 0.9842 | 0.9815 [0.9788–0.984] |
| clf->filt[robust8_nr_both] | 7427 | 155 | 122 | 0.9796 | 0.9838 | 0.9817 [0.979–0.9841] |
| clf->filt[robust8,rej>=0.8] | 7405 | 138 | 96 | 0.9817 | 0.9872 | 0.9844 [0.9818–0.9868] |
| filt->clf[robust8] | 7405 | 136 | 91 | 0.982 | 0.9879 | 0.9849 [0.9823–0.9872] |
| filt->clf[robust6] | 7381 | 151 | 168 | 0.98 | 0.9777 | 0.9788 [0.9758–0.9817] |
| filt->clf[robust8_nr_drop] | 7427 | 154 | 79 | 0.9797 | 0.9895 | 0.9846 [0.982–0.9868] |
| filt->clf[robust8_nr_both] | 7427 | 155 | 117 | 0.9796 | 0.9845 | 0.982 [0.9793–0.9844] |

## ir_confusers  (n=4000 of 5237, kind=ir, rule=iou, imgsz rgb=640/ir=640, drones=False)


**C — confuser FP-reduction (no GT; every surviving det = FP; verifier=mlp_v5_ir_aligned (thermal scaler))**

| stage | FP | fire_rate [95% CI] |
|---|---|---|
| bare | 1203 | 0.2943 [0.2797–0.3085] |
| filt_mlp | 113 | 0.0278 [0.0227–0.033] |
| filt_patch | 1007 | 0.2462 [0.233–0.2598] |
| clf[robust8] | 1117 | 0.2732 [0.2587–0.2868] |
| clf[robust6] | 905 | 0.2225 [0.2087–0.2352] |
| clf[sa32] | 1069 | 0.2627 [0.2487–0.2768] |
| clf[robust8_nr_drop] | 1203 | 0.2943 [0.2797–0.3085] |
| clf[robust8_nr_both] | 1203 | 0.2943 [0.2797–0.3085] |
| clf->filt[robust8] | 99 | 0.0243 [0.0197–0.0293] |
| clf->filt[robust6] | 78 | 0.0192 [0.0152–0.0235] |
| clf->filt[sa32] | 90 | 0.0225 [0.018–0.0272] |
| clf->filt[robust8_nr_drop] | 113 | 0.0278 [0.0227–0.033] |
| clf->filt[robust8_nr_both] | 113 | 0.0278 [0.0227–0.033] |
| filt->clf[robust8] | 99 | 0.0243 [0.0197–0.0293] |
| filt->clf[robust6] | 76 | 0.0187 [0.0145–0.023] |
| filt->clf[robust8_nr_drop] | 113 | 0.0278 [0.0227–0.033] |
| filt->clf[robust8_nr_both] | 113 | 0.0278 [0.0227–0.033] |
| clf->filt[robust8,rej>=0.8] | 102 | 0.025 [0.0203–0.0298] |

## ir_dset_final  (n=4000 of 9612, kind=ir, rule=iou, imgsz rgb=640/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| v3b/ir | 2564 | 157 | 51 | 0.9423 | 0.9805 | 0.961 [0.9552–0.967] | 2615 |

**S4 — verifier-only ablation (single modality, verifier=mlp_v5_ir_aligned (thermal scaler))**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 2564 | 157 | 51 | 0.9423 | 0.9805 | 0.961 [0.9552–0.967] |
| filt_mlp | 2456 | 143 | 159 | 0.945 | 0.9392 | 0.9421 [0.9344–0.949] |
| filt_patch | 2435 | 132 | 180 | 0.9486 | 0.9312 | 0.9398 [0.933–0.9471] |
| clf[robust8] | 2415 | 134 | 200 | 0.9474 | 0.9235 | 0.9353 [0.9281–0.9424] |
| clf->filt[robust8] | 2340 | 125 | 275 | 0.9493 | 0.8948 | 0.9213 [0.9128–0.9287] |
| clf[robust6] | 2140 | 105 | 475 | 0.9532 | 0.8184 | 0.8807 [0.8706–0.8897] |
| clf->filt[robust6] | 2094 | 101 | 521 | 0.954 | 0.8008 | 0.8707 [0.8602–0.8804] |
| clf[sa32] | 2356 | 130 | 259 | 0.9477 | 0.901 | 0.9237 [0.9156–0.9308] |
| clf->filt[sa32] | 2291 | 121 | 324 | 0.9498 | 0.8761 | 0.9115 [0.9023–0.9195] |
| clf[robust8_nr_drop] | 2564 | 157 | 51 | 0.9423 | 0.9805 | 0.961 [0.9552–0.967] |
| clf->filt[robust8_nr_drop] | 2456 | 143 | 159 | 0.945 | 0.9392 | 0.9421 [0.9344–0.949] |
| clf[robust8_nr_both] | 2564 | 157 | 51 | 0.9423 | 0.9805 | 0.961 [0.9552–0.967] |
| clf->filt[robust8_nr_both] | 2456 | 143 | 159 | 0.945 | 0.9392 | 0.9421 [0.9344–0.949] |

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
| filt_mlp | 3033 | 336 | 281 | 0.9003 | 0.9152 | 0.9077 [0.9005–0.9151] |
| filt_mlp_rgb | 3033 | 337 | 281 | 0.9 | 0.9152 | 0.9075 [0.9004–0.915] |
| filt_mlp_ir | 3142 | 2018 | 172 | 0.6089 | 0.9481 | 0.7416 [0.7296–0.7537] |
| filt_patch | 3004 | 782 | 310 | 0.7934 | 0.9065 | 0.8462 [0.8363–0.8552] |
| clf[robust8] | 3092 | 354 | 31 | 0.8973 | 0.9901 | 0.9414 [0.9352–0.9471] |
| clf->filt[robust8] | 2983 | 187 | 140 | 0.941 | 0.9552 | 0.948 [0.9423–0.9537] |
| clf[robust6] | 3132 | 286 | 34 | 0.9163 | 0.9893 | 0.9514 [0.9456–0.9568] |
| clf->filt[robust6] | 3023 | 207 | 143 | 0.9359 | 0.9548 | 0.9453 [0.9393–0.951] |
| clf[sa32] | 3137 | 187 | 30 | 0.9437 | 0.9905 | 0.9666 [0.9617–0.9713] |
| clf->filt[sa32] | 3029 | 161 | 138 | 0.9495 | 0.9564 | 0.953 [0.9476–0.9583] |
| clf[robust8_nr_drop] | 3136 | 1997 | 27 | 0.6109 | 0.9915 | 0.756 [0.7445–0.7685] |
| clf->filt[robust8_nr_drop] | 3027 | 314 | 136 | 0.906 | 0.957 | 0.9308 [0.9242–0.9376] |
| clf[robust8_nr_both] | 3136 | 2003 | 31 | 0.6102 | 0.9902 | 0.7551 [0.7435–0.7677] |
| clf->filt[robust8_nr_both] | 3027 | 320 | 140 | 0.9044 | 0.9558 | 0.9294 [0.9227–0.9363] |
| clf->filt[robust8,rej>=0.8] | 2983 | 191 | 138 | 0.9398 | 0.9558 | 0.9477 [0.9419–0.9534] |
| filt->clf[robust8] | 2989 | 191 | 35 | 0.9399 | 0.9884 | 0.9636 [0.9582–0.9683] |
| filt->clf[robust6] | 3020 | 205 | 45 | 0.9364 | 0.9853 | 0.9603 [0.9544–0.9652] |
| filt->clf[robust8_nr_drop] | 3027 | 317 | 29 | 0.9052 | 0.9905 | 0.9459 [0.9395–0.9516] |
| filt->clf[robust8_nr_both] | 3027 | 321 | 33 | 0.9041 | 0.9892 | 0.9448 [0.9384–0.9504] |

## SPEED (from knowledge/ledger — NOT this replay)

| component | sad (ms) | happy (ms) | speedup | source |
|---|---|---|---|---|
| trust classifier | fusion_no_fn 38.3 /frame | robust8 0.095 /frame | ~404× | ledger bench |
| confuser filter | patch 59–112 /det | mlp_v5 1.3–2.1 /det | ~37–72× | ledger bench |
_Pipeline overhead ~1–4%. Verify via eval/bench_speed.py; wire to kb before thesis._