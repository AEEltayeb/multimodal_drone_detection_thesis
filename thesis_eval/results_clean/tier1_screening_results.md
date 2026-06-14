# Thesis Eval — Tier-1 (FINAL thesis numbers, locked 2026-06-10)
2026-06-13 00:55 | detectors ft4+v3b | patch_thr=0.5 | robust8 tau=0.2 | mlp thr rgb=0.25 / ir=0.05 / gray=0.25
Even-strided ~4k cap per surface (all frames where smaller); n and n_source printed. Per-frame grain; temporal/segment evidence = the separate real-video run. 95% bootstrap CIs (frame resample, 1000 iters) in brackets.


## antiuav_clean  (n=57542 of 57542, kind=paired, rule=iou, imgsz rgb=640/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 54072 | 582 | 751 | 0.9894 | 0.9863 | 0.9878 [0.9871–0.9885] | 54823 |
| v3b/ir | 54699 | 1577 | 2326 | 0.972 | 0.9592 | 0.9656 [0.9643–0.9668] | 57025 |

**B — full-pipeline ablation (per-modality scoring, NO union)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 108771 | 2159 | 3077 | 0.9805 | 0.9725 | 0.9765 [0.9757–0.9772] |
| filt_mlp | 108729 | 2113 | 3119 | 0.9809 | 0.9721 | 0.9765 [0.9757–0.9772] |
| filt_patch | 108771 | 2159 | 3077 | 0.9805 | 0.9725 | 0.9765 [0.9757–0.9772] |
| clf[robust8] | 108470 | 1828 | 1216 | 0.9834 | 0.9889 | 0.9862 [0.9855–0.9867] |
| clf->filt[robust8] | 108429 | 1784 | 1257 | 0.9838 | 0.9885 | 0.9862 [0.9855–0.9867] |
| clf[robust6] | 108155 | 2000 | 2153 | 0.9818 | 0.9805 | 0.9812 [0.9804–0.9819] |
| clf->filt[robust6] | 108119 | 1956 | 2189 | 0.9822 | 0.9802 | 0.9812 [0.9805–0.9819] |
| clf[sa32] | 108705 | 1752 | 989 | 0.9841 | 0.991 | 0.9875 [0.9869–0.9881] |
| clf->filt[sa32] | 108666 | 1720 | 1028 | 0.9844 | 0.9906 | 0.9875 [0.9869–0.9881] |
| filt->clf[robust8] | 108430 | 1784 | 1220 | 0.9838 | 0.9889 | 0.9863 [0.9857–0.9869] |

## svanstrom_clean  (n=5557 of 5557, kind=paired, rule=iop, imgsz rgb=1280/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 1950 | 2798 | 124 | 0.4107 | 0.9402 | 0.5717 [0.5574–0.5851] | 2074 |
| v3b/ir | 1816 | 410 | 145 | 0.8158 | 0.9261 | 0.8674 [0.8538–0.8798] | 1961 |

**B — full-pipeline ablation (per-modality scoring, NO union)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 3766 | 3208 | 269 | 0.54 | 0.9333 | 0.6842 [0.6722–0.6954] |
| filt_mlp | 3670 | 646 | 365 | 0.8503 | 0.9095 | 0.8789 [0.8707–0.8861] |
| filt_patch | 3766 | 3208 | 269 | 0.54 | 0.9333 | 0.6842 [0.6722–0.6954] |
| clf[robust8] | 3653 | 727 | 83 | 0.834 | 0.9778 | 0.9002 [0.8925–0.9075] |
| clf->filt[robust8] | 3557 | 317 | 179 | 0.9182 | 0.9521 | 0.9348 [0.9283–0.9407] |
| clf[robust6] | 3733 | 611 | 136 | 0.8593 | 0.9648 | 0.909 [0.9015–0.9161] |
| clf->filt[robust6] | 3637 | 450 | 232 | 0.8899 | 0.94 | 0.9143 [0.9069–0.9211] |
| clf[sa32] | 3749 | 301 | 141 | 0.9257 | 0.9638 | 0.9443 [0.9379–0.9507] |
| clf->filt[sa32] | 3655 | 268 | 235 | 0.9317 | 0.9396 | 0.9356 [0.9289–0.9417] |
| filt->clf[robust8] | 3560 | 325 | 91 | 0.9163 | 0.9751 | 0.9448 [0.9384–0.9505] |

## SPEED (from knowledge/ledger — NOT this replay)

| component | sad (ms) | happy (ms) | speedup | source |
|---|---|---|---|---|
| trust classifier | fusion_no_fn 38.3 /frame | robust8 0.095 /frame | ~404× | ledger bench |
| confuser filter | patch 59–112 /det | mlp_v5 1.3–2.1 /det | ~37–72× | ledger bench |
_Pipeline overhead ~1–4%. Verify via eval/bench_speed.py; wire to kb before thesis._