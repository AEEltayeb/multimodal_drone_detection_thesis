# Thesis Eval — Tier-1 (FINAL thesis numbers, locked 2026-06-10)
2026-06-17 20:34 | detectors ft4+v3b | patch_thr=0.5 | robust8 tau=0.2 | mlp thr rgb=0.25 / ir=0.01 / gray=0.25
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
| filt_mlp | 7426 | 169 | 247 | 0.9777 | 0.9678 | 0.9728 [0.9695–0.9756] |
| filt_mlp_rgb | 7428 | 169 | 245 | 0.9778 | 0.9681 | 0.9729 [0.9697–0.9758] |
| filt_mlp_ir | 7430 | 174 | 243 | 0.9771 | 0.9683 | 0.9727 [0.9695–0.9757] |
| filt_patch | 7430 | 174 | 243 | 0.9771 | 0.9683 | 0.9727 [0.9695–0.9757] |
| clf[robust8] | 7408 | 141 | 93 | 0.9813 | 0.9876 | 0.9845 [0.9819–0.9868] |
| clf->filt[robust8] | 7402 | 137 | 99 | 0.9818 | 0.9868 | 0.9843 [0.9817–0.9867] |
| clf[robust6] | 7384 | 156 | 170 | 0.9793 | 0.9775 | 0.9784 [0.9754–0.9814] |
| clf->filt[robust6] | 7378 | 152 | 176 | 0.9798 | 0.9767 | 0.9783 [0.9752–0.9811] |
| clf[sa32] | 7426 | 135 | 75 | 0.9821 | 0.99 | 0.9861 [0.9837–0.9882] |
| clf->filt[sa32] | 7420 | 133 | 81 | 0.9824 | 0.9892 | 0.9858 [0.9834–0.9879] |
| clf[robust8_nr_drop] | 7430 | 160 | 81 | 0.9789 | 0.9892 | 0.984 [0.9814–0.9864] |
| clf->filt[robust8_nr_drop] | 7424 | 155 | 87 | 0.9795 | 0.9884 | 0.984 [0.9814–0.9864] |
| clf[robust8_nr_both] | 7430 | 161 | 119 | 0.9788 | 0.9842 | 0.9815 [0.9788–0.984] |
| clf->filt[robust8_nr_both] | 7424 | 156 | 125 | 0.9794 | 0.9834 | 0.9814 [0.9787–0.9839] |
| clf->filt[robust8,rej>=0.8] | 7402 | 139 | 99 | 0.9816 | 0.9868 | 0.9842 [0.9816–0.9866] |
| filt->clf[robust8] | 7402 | 137 | 92 | 0.9818 | 0.9877 | 0.9848 [0.9822–0.9871] |
| filt->clf[robust6] | 7378 | 152 | 169 | 0.9798 | 0.9776 | 0.9787 [0.9757–0.9816] |
| filt->clf[robust8_nr_drop] | 7424 | 155 | 80 | 0.9795 | 0.9893 | 0.9844 [0.9818–0.9868] |
| filt->clf[robust8_nr_both] | 7424 | 156 | 118 | 0.9794 | 0.9844 | 0.9819 [0.9792–0.9843] |

## SPEED (from knowledge/ledger — NOT this replay)

| component | sad (ms) | happy (ms) | speedup | source |
|---|---|---|---|---|
| trust classifier | fusion_no_fn 38.3 /frame | robust8 0.095 /frame | ~404× | ledger bench |
| confuser filter | patch 59–112 /det | mlp_v5 1.3–2.1 /det | ~37–72× | ledger bench |
_Pipeline overhead ~1–4%. Verify via eval/bench_speed.py; wire to kb before thesis._