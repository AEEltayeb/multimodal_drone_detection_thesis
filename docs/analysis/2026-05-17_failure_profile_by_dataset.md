# Per-dataset confuser failure profile (2026-05-17)

Mined from existing eval CSVs; no re-inference. See `analytics/spec_analysis/06_confuser_failure_profile.py`.

## Sources

- Roboflow OOD: `eval/results/roboflow_ood/{dataset}/{model}/{split}/{model}_frame_detections.csv`
- Svanström per-detection: `eval/results/_patch_catch_audit/baseline_v2/per_detection.csv` (baseline RGB, imgsz=1280, IoP@0.5, conf=0.25, stride=9)

## 1. Roboflow OOD --- fire rate, FPPI, size distribution by (model, dataset)

`fire rate` = fraction of frames with at least one raw detection. `FPPI raw` = FPs per image on negatives-only datasets (drone datasets included for reference). Size buckets are as recorded in `frame_detections.csv` by the Roboflow eval harness.

| Dataset | Model | Frames | Dets | Fire rate | FPPI raw | % small | % medium | % large |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rgb_bird` | `rgb_baseline` | 174 | 85 | 54.0% | 59.2% | 78.8% | 20.0% | 1.2% |
| `rgb_bird` | `rgb_retrained_v2` | 174 | 75 | 46.0% | 48.9% | 77.3% | 21.3% | 1.3% |
| `rgb_airplane` | `rgb_baseline` | 4140 | 873 | 25.6% | 32.1% | 9.4% | 33.4% | 57.2% |
| `rgb_airplane` | `rgb_retrained_v2` | 4140 | 1225 | 34.2% | 40.2% | 5.0% | 17.9% | 77.1% |
| `rgb_helicopter` | `rgb_baseline` | 1407 | 407 | 30.6% | 32.5% | 17.4% | 39.3% | 43.2% |
| `rgb_helicopter` | `rgb_retrained_v2` | 1407 | 269 | 20.1% | 21.1% | 0.0% | 0.7% | 99.3% |
| `ir_airplane_hors2` | `ir_model` | 1944 | 1108 | 57.5% | 58.0% | 60.8% | 29.1% | 10.1% |
| `ir_airplane_plane` | `ir_model` | 2337 | 372 | 16.2% | 16.5% | 4.6% | 63.7% | 31.7% |
| `ir_bird` | `ir_model` | 1200 | 95 | 7.9% | 7.9% | 23.2% | 75.8% | 1.1% |
| `ir_mixed_cbam` | `ir_model` | 10244 | 2119 | 20.9% | 3.8% | 6.2% | 89.3% | 4.5% |
| `ir_drone_night` | `ir_model` | 2169 | 620 | 39.9% | 26.5% | 45.0% | 53.7% | 1.3% |
| `rgb_drone` | `rgb_baseline` | 2864 | 1532 | 74.0% | 8.4% | 18.7% | 54.8% | 26.6% |
| `rgb_drone` | `rgb_retrained_v2` | 2864 | 1862 | 78.0% | 6.9% | 16.2% | 45.6% | 38.2% |

## 2. Roboflow OOD --- detection confidence distribution by (model, dataset)

Per-detection confidence is parsed from the `dets` column. For negative-only datasets every detection is a false positive; for drone datasets a detection may be TP or FP (the CSV does not separate them, so confidences here are mixed for drone datasets).

| Dataset | Model | N dets | conf p25 | conf median | conf p75 |
|---|---|---:|---:|---:|---:|
| `rgb_bird` | `rgb_baseline` | 85 | 0.563 | 0.682 | 0.754 |
| `rgb_bird` | `rgb_retrained_v2` | 75 | 0.487 | 0.679 | 0.772 |
| `rgb_airplane` | `rgb_baseline` | 873 | 0.471 | 0.725 | 0.848 |
| `rgb_airplane` | `rgb_retrained_v2` | 1225 | 0.447 | 0.664 | 0.814 |
| `rgb_helicopter` | `rgb_baseline` | 407 | 0.676 | 0.825 | 0.890 |
| `rgb_helicopter` | `rgb_retrained_v2` | 269 | 0.447 | 0.620 | 0.774 |
| `ir_airplane_hors2` | `ir_model` | 1108 | 0.751 | 0.824 | 0.865 |
| `ir_airplane_plane` | `ir_model` | 372 | 0.581 | 0.720 | 0.829 |
| `ir_bird` | `ir_model` | 95 | 0.539 | 0.681 | 0.788 |
| `ir_mixed_cbam` | `ir_model` | 2119 | 0.658 | 0.790 | 0.838 |
| `ir_drone_night` | `ir_model` | 620 | 0.615 | 0.754 | 0.855 |
| `rgb_drone` | `rgb_baseline` | 1532 | 0.779 | 0.838 | 0.882 |
| `rgb_drone` | `rgb_retrained_v2` | 1862 | 0.752 | 0.851 | 0.902 |

## 3. Svanström baseline --- per-category detection confidence (imgsz=1280, stride=9)

This is the *only* per-detection dump we have on Svanström. It is for the baseline RGB model with patch verifier v2; we read only the `det_conf` column (YOLO confidence, no patch suppression applied here). DRONE_TP and DRONE_FP are split by IoP match to GT; BIRD/AIRPLANE/HELICOPTER are confuser frames where any detection is, by construction, an FP.

| Category | N dets | conf p25 | conf median | conf p75 | patch p median |
|---|---:|---:|---:|---:|---:|
| DRONE_TP | 1248 | 0.768 | 0.822 | 0.860 | 0.000 |
| DRONE_FP | 79 | 0.405 | 0.558 | 0.711 | 0.000 |
| BIRD | 807 | 0.546 | 0.691 | 0.773 | 0.904 |
| AIRPLANE | 532 | 0.571 | 0.733 | 0.806 | 0.540 |
| HELICOPTER | 464 | 0.729 | 0.832 | 0.869 | 0.987 |

## 4. What the numbers say

Five concrete findings from the tables above.

### 4.1 `retrained_v2` has learned a near-categorical "do not fire on small objects" rule.

On `rgb_helicopter`, `retrained_v2`'s FPs are **99.3% large** (vs 43.2% large for baseline). On `rgb_airplane`, **77.1% large** (vs 57.2% large for baseline). The model has shifted its operating regime from "fires on everything roughly drone-shaped" to "fires only on big things." This single behaviour explains three other observations:

- Why `retrained_v2` drone recall collapses on Svanstr\"om to $R=0.306$ (Svanstr\"om drones are predominantly small).
- Why `retrained_v2` looks like a confuser killer on Svanstr\"om (Svanstr\"om confusers are also small).
- Why `retrained_v2` is *worse* than baseline on `rgb_airplane` (Roboflow airplanes are mostly large, exactly what `retrained_v2` is *willing* to fire on).

The "confuser killer" framing of `retrained_v2` is therefore misleading: it is a "small-object suppressor" with confuser-set-specific side-effects.

### 4.2 `rgb_bird` is the exception --- both models still fire on small birds at similar rates.

Bird FPs are **78.8% small (baseline)** vs **77.3% small (`retrained_v2`)**, with overall fire rate dropping only from 54.0% to 46.0% across the model swap. `retrained_v2`'s "don't fire on small things" rule generalises to airplanes and helicopters but not to small birds against sky --- the bird/small-drone visual overlap is too tight for the size-based rule to help. This is consistent with the Svanstr\"om observation that `hardneg_v3more` cannot move bird fire rate (94.4\% $\to$ 94.2\%) while it does move helicopter fire rate (66.2\% $\to$ 41.9\%; \textsc{Ledger}~\S3.1).

### 4.3 The IR airplane fire-rate gap (58\% vs 16.5\% FPPI) is a size-regime story.

Same IR model on two datasets:

- `ir_airplane_hors2` (58\% FPPI): **60.8\% small**, 29.1\% medium, 10.1\% large.
- `ir_airplane_plane` (16.5\% FPPI): 4.6\% small, **63.7\% medium**, 31.7\% large.

Small detections on `hors2` dominate the FP count; the IR detector fires at high confidence (median 0.824) on those small horizon-line silhouettes. On `plane`, the airplanes are clearly larger objects the IR detector recognises as not-drone (median FP confidence 0.720, fewer FPs overall). The IR detector's airplane "weakness" is therefore not a categorical airplane weakness; it is the same small-object regime that defeats every detector in the project.

### 4.4 The Svanstr\"om baseline cannot discriminate helicopters from drones at the confidence level.

From §3:

- DRONE_TP median conf: **0.822**
- HELICOPTER median conf: **0.832** (higher than drone TPs!)
- AIRPLANE median conf: 0.733
- BIRD median conf: 0.691

The model is genuinely \emph{more} confident on Svanstr\"om helicopter frames than on Svanstr\"om drone frames. A confidence threshold cannot help here. The patch verifier earns its keep on this category: patch\_prob median on HELICOPTER is **0.987**, on BIRD is 0.904. The verifier sees what the detector cannot.

### 4.5 The patch verifier's airplane weakness is at the verifier itself.

Patch probability medians from §3:

- BIRD: 0.904 (verifier fires high $\Rightarrow$ catches the FP)
- HELICOPTER: 0.987
- AIRPLANE: **0.540** (verifier uncertain $\Rightarrow$ misses 48\% of catches per \textsc{Ledger}~\S6.1)

Bird and helicopter FPs ride the verifier's confident-confuser bimodality; airplane FPs sit in the verifier's uncertain middle band. This is the catch-rate audit number (52\% airplane catch in \textsc{Ledger}~\S6.1) made visible in patch-probability space.

### Implications for the thesis narrative

1. The "model fires on X\% of bird frames" framing should be **dataset-conditional, not categorical**. On Roboflow `rgb_bird`, baseline fires on 54\% of frames; on Svanstr\"om bird splits, 94.4\%. Both are bird datasets and both fire rates are real --- they characterise the model's behaviour on \emph{different bird distributions}.
2. The cascade's value should be framed against the **small-object failure mode that survives training-time mining**, since that is the irreducible part. `retrained_v2`'s training-time mining \emph{can} suppress confusers when their size matches the training distribution, but the bird/small-drone case is unreachable that way.
3. The IR detector's "airplane weakness" should be qualified as **small-airplane-on-sky weakness, not a general airplane weakness**. On airplane datasets where the airplanes are clearly framed and medium-sized, the IR detector hallucinates at ~3.5$\times$ lower rates.

## 5. Delivered

- `docs/analysis/2026-05-17_failure_profile_by_dataset.md` --- this report.
- `analytics/spec_analysis/06_confuser_failure_profile.py` --- the script that produced it.
