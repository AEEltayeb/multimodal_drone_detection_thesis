# Thesis Eval — Tier-1 (FINAL thesis numbers, locked 2026-06-10)
2026-06-13 23:36 | detectors ft4+v3b | patch_thr=0.5 | robust8 tau=0.2 | mlp thr rgb=0.25 / ir=0.05 / gray=0.25
Even-strided ~4k cap per surface (all frames where smaller); n and n_source printed. Per-frame grain; temporal/segment evidence = the separate real-video run. 95% bootstrap CIs (frame resample, 1000 iters) in brackets.


## dut_antiuav_640  (n=2200 of 2200, kind=grayrgb_paired, rule=iou, imgsz rgb=640/ir=640, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 1757 | 95 | 488 | 0.9487 | 0.7826 | 0.8577 [0.8458–0.8694] | 2245 |
| v3b/ir | 940 | 376 | 1305 | 0.7143 | 0.4187 | 0.5279 [0.5062–0.55] | 2245 |

**B — full-pipeline ablation (per-modality scoring, NO union)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 2697 | 471 | 1793 | 0.8513 | 0.6007 | 0.7044 [0.6913–0.7171] |
| filt_mlp | 2425 | 391 | 2065 | 0.8612 | 0.5401 | 0.6638 [0.6511–0.6775] |
| filt_patch | 2603 | 462 | 1887 | 0.8493 | 0.5797 | 0.6891 [0.676–0.7021] |
| clf[robust8] | 2179 | 278 | 1702 | 0.8869 | 0.5615 | 0.6876 [0.6705–0.7053] |
| clf->filt[robust8] | 1968 | 224 | 1913 | 0.8978 | 0.5071 | 0.6481 [0.6312–0.6662] |
| clf[robust6] | 2088 | 359 | 2122 | 0.8533 | 0.496 | 0.6273 [0.6091–0.6451] |
| clf->filt[robust6] | 1843 | 288 | 2367 | 0.8649 | 0.4378 | 0.5813 [0.5634–0.5993] |
| clf[sa32] | 2342 | 424 | 1742 | 0.8467 | 0.5735 | 0.6838 [0.6671–0.6995] |
| clf->filt[sa32] | 2087 | 349 | 1997 | 0.8567 | 0.511 | 0.6402 [0.6238–0.656] |
| filt->clf[robust8] | 1959 | 221 | 1740 | 0.8986 | 0.5296 | 0.6664 [0.6471–0.6854] |

## dut_antiuav_960  (n=2200 of 2200, kind=grayrgb_paired, rule=iou, imgsz rgb=960/ir=960, drones=True)

**A — bare detector (per modality vs own GT)**

| modality | TP | FP | FN | P | R | F1 [95% CI] | n_gt |
|---|---|---|---|---|---|---|---|
| ft4/rgb | 1917 | 104 | 328 | 0.9485 | 0.8539 | 0.8987 [0.8888–0.9084] | 2245 |
| v3b/ir | 1113 | 375 | 1132 | 0.748 | 0.4958 | 0.5963 [0.5764–0.616] | 2245 |

**B — full-pipeline ablation (per-modality scoring, NO union)**

| cell | TP | FP | FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|---|
| bare | 3030 | 479 | 1460 | 0.8635 | 0.6748 | 0.7576 [0.7467–0.769] |
| filt_mlp | 2586 | 399 | 1904 | 0.8663 | 0.5759 | 0.6919 [0.6799–0.7042] |
| filt_patch | 2931 | 470 | 1559 | 0.8618 | 0.6528 | 0.7429 [0.732–0.754] |
| clf[robust8] | 2586 | 302 | 1301 | 0.8954 | 0.6653 | 0.7634 [0.7492–0.7779] |
| clf->filt[robust8] | 2234 | 253 | 1653 | 0.8983 | 0.5747 | 0.701 [0.6855–0.7172] |
| clf[robust6] | 2415 | 374 | 1808 | 0.8659 | 0.5719 | 0.6888 [0.6741–0.7047] |
| clf->filt[robust6] | 2055 | 311 | 2168 | 0.8686 | 0.4866 | 0.6238 [0.6085–0.6413] |
| clf[sa32] | 2651 | 430 | 1480 | 0.8604 | 0.6417 | 0.7352 [0.7211–0.7495] |
| clf->filt[sa32] | 2265 | 361 | 1866 | 0.8625 | 0.5483 | 0.6704 [0.655–0.6857] |
| filt->clf[robust8] | 2182 | 243 | 1564 | 0.8998 | 0.5825 | 0.7072 [0.6914–0.7246] |

## SPEED (from knowledge/ledger — NOT this replay)

| component | sad (ms) | happy (ms) | speedup | source |
|---|---|---|---|---|
| trust classifier | fusion_no_fn 38.3 /frame | robust8 0.095 /frame | ~404× | ledger bench |
| confuser filter | patch 59–112 /det | mlp_v5 1.3–2.1 /det | ~37–72× | ledger bench |
_Pipeline overhead ~1–4%. Verify via eval/bench_speed.py; wire to kb before thesis._