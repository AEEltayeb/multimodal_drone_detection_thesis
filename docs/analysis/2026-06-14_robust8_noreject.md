# robust8 no-reject (3-class) router — classifier-only comparison

_2026-06-14. Compares the shipped 4-class **robust8** {reject/rgb/ir/both} against two **no-reject** 3-class variants that must route to rgb/ir/both (reject removed; the verifier does FP rejection):_ **nr_drop** (reject rows dropped in training) and **nr_both** (reject→both). All three are f8 routers trained on the same `fusion_dataset_full56.csv` / hyperparams / seq-split; evaluated zero-GPU on every unified cache. P/R/F1 are per-modality trust-aware (NO union). NOT in the thesis.

## Held-out training (per-class F1, from the trainer)
| variant | macro-F1 | trust_rgb | trust_ir | both |
|---|---|---|---|---|
| robust8 (4-class, ref) | — | — | — | — |
| nr_drop | 0.941 | 0.854 | 0.980 | 0.990 |
| nr_both | 0.885 | 0.718 | 0.951 | 0.985 |

## Drone surfaces — classifier ONLY (no verifier)  ·  P/R/F1
_bare = trust-both, no routing. The no-reject routers keep recall high but, lacking reject, do not suppress confuser FPs at this stage (precision ≈ bare on confuser-rich surfaces)._

| surface | kind | bare F1 | robust8 | nr_drop | nr_both |
|---|---|---|---|---|---|
| antiuav | paired | 0.973 | 0.981/0.988/0.985 | 0.979/0.989/0.984 | 0.979/0.984/0.982 |
| antiuav_clean | paired | 0.977 | 0.983/0.989/0.986 | 0.982/0.991/0.987 | 0.982/0.986/0.984 |
| dut_antiuav_640 | grayrgb_paired | 0.704 | 0.887/0.561/0.688 | 0.854/0.799/0.825 | 0.852/0.654/0.740 |
| dut_antiuav_960 | grayrgb_paired | 0.758 | 0.895/0.665/0.763 | 0.867/0.853/0.860 | 0.865/0.738/0.797 |
| ir_dset_final | ir | 0.961 | 0.947/0.923/0.935 | 0.942/0.981/0.961 | 0.942/0.981/0.961 |
| rgb_dataset_test | rgb | 0.926 | 0.981/0.383/0.551 | 0.954/0.899/0.926 | 0.954/0.899/0.926 |
| selcom_val | rgb | 0.591 | 0.881/0.125/0.220 | 0.858/0.451/0.591 | 0.858/0.451/0.591 |
| svanstrom | paired | 0.742 | 0.897/0.990/0.941 | 0.611/0.992/0.756 | 0.610/0.990/0.755 |
| svanstrom_clean | paired | 0.684 | 0.834/0.978/0.900 | 0.544/0.968/0.696 | 0.545/0.971/0.698 |
| svanstrom_gray | gray | 0.580 | — | — | — |
| svanstrom_rawrgb | rawrgb | 0.187 | — | — | — |
| video_drone | grayrgb_paired | 0.589 | 0.797/0.439/0.566 | 0.734/0.734/0.734 | 0.736/0.544/0.625 |

_mean clf F1 over drone surfaces — robust8 0.754 · nr_drop 0.832 · nr_both 0.806._

## Drone surfaces — classifier → verifier (clf→filt)  ·  P/R/F1
_the verifier (mlp_v5 / aligned) now does the FP rejection the no-reject router skipped._

| surface | kind | robust8 | nr_drop | nr_both |
|---|---|---|---|---|
| antiuav | paired | 0.982/0.987/0.984 | 0.979/0.989/0.984 | 0.979/0.984/0.982 |
| antiuav_clean | paired | 0.984/0.989/0.986 | 0.983/0.991/0.987 | 0.982/0.986/0.984 |
| dut_antiuav_640 | grayrgb_paired | 0.898/0.507/0.648 | 0.864/0.718/0.784 | 0.862/0.589/0.700 |
| dut_antiuav_960 | grayrgb_paired | 0.898/0.575/0.701 | 0.869/0.728/0.792 | 0.868/0.630/0.730 |
| ir_dset_final | ir | 0.948/0.916/0.932 | 0.943/0.973/0.958 | 0.943/0.973/0.958 |
| rgb_dataset_test | rgb | 0.983/0.377/0.545 | 0.976/0.691/0.809 | 0.976/0.691/0.809 |
| selcom_val | rgb | 0.925/0.125/0.221 | 0.950/0.451/0.612 | 0.950/0.451/0.612 |
| svanstrom | paired | 0.939/0.958/0.949 | 0.902/0.960/0.930 | 0.900/0.959/0.929 |
| svanstrom_clean | paired | 0.918/0.952/0.935 | 0.863/0.943/0.901 | 0.866/0.946/0.904 |
| svanstrom_gray | gray | — | — | — |
| svanstrom_rawrgb | rawrgb | — | — | — |
| video_drone | grayrgb_paired | 0.878/0.281/0.426 | 0.816/0.456/0.585 | 0.815/0.339/0.478 |

_mean clf→filt F1 — robust8 0.733 · nr_drop 0.834 · nr_both 0.809._

## Confuser surfaces — fire rate (lower = better)
**clf only:**

| surface | bare | robust8 | nr_drop | nr_both |
|---|---|---|---|---|
| gray_confuser | 0.238 | — | — | — |
| ir_confusers | 0.294 | 0.273 | 0.294 | 0.294 |
| rgb_confuser | 0.303 | 0.049 | 0.303 | 0.303 |

**clf→filt (with verifier):**

| surface | robust8 | nr_drop | nr_both |
|---|---|---|---|
| gray_confuser | — | — | — |
| ir_confusers | 0.217 | 0.237 | 0.237 |
| rgb_confuser | 0.002 | 0.011 | 0.011 |

## Delivered
- Trainer: `classifier/train_robust8_noreject.py` → `models/routers/robust8_noreject_{drop,both}/model.joblib`
- Results: `thesis_eval/results_noreject/tier1_screening_results.md` + `tier1_results.json`
- Harness: `thesis_eval/pipeline_eval_unified.py` (load_classifiers + batch_labels label_map)
- This doc: `docs/analysis/2026-06-14_robust8_noreject.md`  (NOT in the thesis)