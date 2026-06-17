# Detector-confidence × filter-threshold global sweep — best F1 per dataset

**Date:** 2026-06-17
**Question (notes 16 + 22):** is there a single (RGB conf, IR conf, filter P(drone))
operating point that gives the best F1 on every dataset in the thesis?
**Method:** zero-GPU replay `thesis_eval/conf_sweep_replay.py --cache-dir thesis_eval/cache_conf005`
(conf=0.05 floored caches, so the full conf∈[0.05,0.50] × filter-threshold grid is
available; trust-aware / own-GT scoring, never unioned; bootstrap CIs; points below a
cache's floor are censored, never extrapolated).
**Coverage:** 8 surfaces have a conf=0.05 cache. **Missing: `svanstrom` (paired
flagship) and `dut_antiuav` — they need a GPU re-cache at 0.05** (command at the end);
`antiuav` here is a 500-frame subsample.

## Answer: NO single global optimum exists, and F1 is the wrong objective for the filter

Two separate facts:

1. **The best (conf, filter) is surface-specific** (table below): low conf helps a
   recall-starved detector (SelCom), default conf is right in-domain, and the filter is
   a *net negative* on drone-only surfaces where it can only delete true positives.
2. **Optimising F1 on drone-only datasets will tell you to turn the filter OFF.** A
   filter only removes detections; a drone-only surface has no false positives for it to
   remove, so filtering can only hold F1 flat or lower it. The filter earns its keep on
   the *confuser* surfaces, measured by frame fire-rate, which F1 cannot see (no GT).
   A pure-F1 global optimum would ship a stack that floods on birds and aircraft.

### Drone surfaces — F1 (best bare = filter off, vs best achievable with the filter)

| surface (modality) | shipped default | best **bare** (filter off) | best **with filter** | filter verdict on F1 |
|---|---|---|---|---|
| `selcom_val` (RGB CCTV) | conf 0.25 → filt 0.612 | 0.626 @conf 0.20 | **0.696** @conf 0.05, thr 0.6 | **helps** at low conf (+7 pp): recall-starved detector, filter recall-transparent |
| `rgb_dataset_test` (RGB) | conf 0.25 | **0.926** @conf 0.25 | 0.837 @conf 0.25, thr 0.1 | **hurts** (−9 pp): the OOD carve-out (Section 4.3.4) |
| `ir_dset_final` (IR) | conf 0.40 | **0.965** @conf 0.35 | 0.959 @conf 0.35, thr 0.1 | neutral (−0.6 pp) |
| `svanstrom_gray` (grayscale) | conf 0.40 | **0.568** @conf 0.25 | 0.185 @conf 0.05, thr 0.1 | **catastrophic**: grayscale filter over-vetoes real drones |
| `antiuav` (paired, n=500) | rgb 0.25 / ir 0.40 | ~0.959 | ~0.960 | saturated, neutral; lowering rgb conf to 0.05 keeps 0.959 |

### Confuser surfaces — frame fire-rate (where the filter actually earns its keep)

| surface | conf | bare fire | + filter fire | reduction |
|---|---|---|---|---|
| `rgb_confuser` | 0.25 | 0.304 | **0.011** | ~28× |
| `rgb_confuser` | 0.05 | 0.422 | 0.016 | ~26× |
| `gray_confuser` | 0.40 | 0.220 | **0.006** | ~36× |
| `ir_confusers` | 0.40 | 0.260 | 0.209 | ~1.25× (airplane gap, Section 4.3.4) |
| `ir_confusers` | 0.05 | 0.417 | 0.343 | weak |

## Reading

- **Global operating point = a deliberate compromise, not an argmax.** The shipped
  defaults (RGB 0.25, IR 0.40, filter on) keep drone F1 within ~1–2 pp of the per-surface
  *bare* optimum on the clean surfaces while buying the ~28–36× confuser suppression on
  RGB/grayscale. There is no setting that simultaneously maximises drone F1 and minimises
  confuser fire, because the filter trades the first for the second.
- **Lowering the detector floor is an operating *mode*, not a new default** — it pays only
  where the detector is recall-starved (SelCom-style CCTV: conf 0.05 + filter → 0.696 vs
  default 0.612). The thesis already states this (Table `tab:lowconf_selcom`, §4.1.7
  "an operating mode, not a setting"). This sweep generalises it across surfaces.
- **The filter is a precision tool, confirmed four ways:** it *helps* F1 only where the
  detector over-produces and confusers are present (SelCom, all confuser surfaces);
  it is *neutral* where the detector is already clean (IR, Anti-UAV); and it *hurts* where
  the surface is drone-only and OOD to the filter (rgb_dataset_test carve-out, grayscale
  Svanström over-veto — the latter ties to the RGB-filter recall investigation handed off
  in check.txt).

## Missing for completeness (GPU, user-gated)

`svanstrom` (paired RGB@1280 + IR — the flagship, where the resolvable floor and the
RGB/IR trust interplay make the conf×filter interaction most interesting) and `dut_antiuav`
have no conf=0.05 cache. To complete the grid:

```
py -u thesis_eval/pipeline_cache_unified.py --conf 0.05 --cache-dir thesis_eval/cache_conf005 --no-patch --target 4000 --only svanstrom,dut_antiuav_960
py -u thesis_eval/conf_sweep_replay.py --cache-dir thesis_eval/cache_conf005
```

## Pooled global operating point (all 10 datasets as ONE micro-averaged set)

Treating every dataset as one pooled set (confuser detections = pure FP in the
objective, which is what makes F1 a valid filter objective) and grid-searching the
per-modality vector `(rgb_conf,rgb_thr) x (ir_conf,ir_thr) x (gray_conf,gray_thr)`
for max global micro-F1. Caches at conf=0.05; svanstrom + dut re-cached at 0.05 for
this run. Script: `thesis_eval/pooled_operating_point.py`.

| | RGB conf/thr | IR conf/thr | GRAY conf/thr | Global F1 | P | R | total FP |
|---|---|---|---|---|---|---|---|
| **Optimal** | 0.20 / 0.05 | 0.40 / 0.10 | 0.50 / 0.05 | **0.7927** | 0.851 | 0.742 | 2161 |
| Shipped | 0.25 / 0.25 | 0.40 / 0.05 | 0.40 / 0.25 | 0.7714 | 0.872 | 0.692 | 1701 |
| Filter off | 0.40 / – | 0.40 / – | 0.25 / – | 0.7559 | 0.709 | 0.810 | 5542 |

Optimal beats shipped by **+2.1 pp** global F1. Per-surface at optimal vs shipped:
wins on `svanstrom_gray` +12.0, `dut/gray` +6.2, `rgb_dataset_test` +5.2, `dut/rgb`
+4.4, `selcom` +2.7; costs on **`svanstrom/rgb` -2.8** (flagship RGB precision) and
confuser FP (rgb 29->144, gray 17->72; total FP +27%).

Three load-bearing caveats:
1. **Micro-average = frame-count weighted**; the optimum reflects the pool's
   composition. (Adding the flagship moved IR thr from a partial-pool 0.60 to 0.10,
   i.e. svanstrom's clean thermal surface corrected an operationally-bad setting.)
2. **It trades the flagship + false-alarm rate for aggregate F1** — spends
   svanstrom-RGB precision and ~460 extra FP to gain on rgb_dataset_test/grayscale/DUT.
3. **Plain F1 under-credits the filter**: filter-off is only 3.7 pp below optimal but
   with 5542 FP vs 2161. That is why the optimum picks permissive filters (0.05). To
   make the "perfect point" reward confuser rejection, use Fbeta (beta<1) or
   "max F1 s.t. FP budget", not plain F1.

DEPENDENCY: these numbers are for the CURRENT `mlp_v5`/`aligned` filters. If the RGB
filter is re-mined/retrained (separate investigation), re-run `pooled_operating_point.py`.

SelCom specifically: under the pooled-optimal RGB point (0.20/0.05) SelCom F1=0.638
(R 0.505), vs its OWN best 0.696 (conf 0.05, thr 0.60, R 0.678) and shipped 0.612 --
a 5.8 pp compromise cost, almost all recall, from sharing one global RGB conf floor.

Results JSON: `thesis_eval/results/conf_sweep/pooled_operating_point.json`.

## Delivered
- Pooled optimizer: `thesis_eval/pooled_operating_point.py` -> `thesis_eval/results/conf_sweep/pooled_operating_point.json`
- Sweep harness (pre-existing): `thesis_eval/conf_sweep_replay.py`
- Results: `thesis_eval/results/conf_sweep/conf_sweep_results.{json,md}`
- Figures: `docs/analysis/images/conf_sweep/`
- This note: `docs/analysis/2026-06-17_conf_filter_global_sweep.md`
