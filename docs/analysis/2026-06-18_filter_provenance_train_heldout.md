# Confuser-filter provenance — training corpora, splits, and held-out eval (2026-06-18)

Authoritative provenance for the **two rebuilt confuser filters** — the RGB **`v4`** and the IR
**`cbam`** (thermal-only). For each: the detector that produces its features, the exact training
sources *with their dataset split*, and — the part that matters most — **which eval surfaces are
genuinely held-out vs which share images with training** (leaky), with the honest alternative for
every leaky one.

Scoring throughout is **own-GT / trust-aware** (a filter only *removes* detections; it can never
raise a lagging modality's recall). RGB uses IoU except Svanström/Selcom (IoP); IR uses IoU except
Svanström IR (IoP).

---

## 0. The two filters at a glance

| | RGB `v4` | IR `cbam` (thermal-only) |
|---|---|---|
| **weight** | `eval/results/_v5_balanced_v4/classifiers/mlp_v5_balanced_v4.pt` | `mri/results/ir_aligned_cbam_thermalonly/classifiers/mlp_aligned.pt` |
| **detector (feature source)** | FT4 `models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt` | V3B `models/ir/corrective_finetune/finetune_v3b/weights/best.pt` |
| **features** | 517-D (5 meta + p3@2×2·256 + p5@1×1·256) | 517-D, thermal per-modality z-score baked into the deploy scaler |
| **deploy threshold** | P(drone) ≥ **0.25** | P(drone) ≥ **0.05** (0.01 = conservative) |
| **builder** | `eval/build_balanced_v4_birdsplit.py` | `mri/train_aligned.py --thermal-confusers --cbam --no-gray` |
| **gray head?** | n/a (RGB) | **none** — `--no-gray` validation build; production needs the with-gray run |
| **status** | held-out WIN, promote-ready (minor +5 non-bird FP) | held-out CBAM recall recovered; ir_dset −3.7pp tradeoff is a your-call op-point |

---

## 1. RGB filter — `mlp_v5_balanced_v4`

### 1.1 Lineage
shipped `mlp_v5` corpus → `distill_v5_balanced_remine.py` (size×source-balanced drones, confusers
protected) → **v2** `build_balanced_v2_surgical.py` (pure-selcom swap + drone:confuser ratio
restore) → **v4** `build_balanced_v4_birdsplit.py` (+ in-distribution bird.v1i train split).

The root problem v4 fixes: the shipped filter vetoed **22%** of real `rgb_dataset_test` drones —
a size×**source** coverage gap (the parent distill took rgb_dataset drones via a stride-8 8000-quota
that left the small-drone manifold unpopulated). imgsz was refuted (1280 worse). Diagnosis:
`2026-06-17_rgbtest_filter_regression.md`.

### 1.2 TRAIN data
**v2 corpus = 19,334 drone / 13,597 confuser (32,931 total)** — matches the shipped budget but with
rgb_dataset balanced:
- balanced rgb_dataset drones from **`dataset/dataset/images/train` + `…/val`** (size×source cells) +
  Anti-UAV **`val`/RGB** drones, subsampled to the 13,500 weight-1.0 drone budget;
- **PURE selcom**: 833 drones + 149 confusers via `mine_pure_selcom` — **blocklists the 311
  `selcom_val` files**, imgsz 1280, IoP, weights 1.8/1.5;
- parent confusers: `rgb_confusers_merged/images/train`+`val`, rgb_video, Svanström drone-empty.

**+ bird.v1i TRAIN split**: bird.v1i = 1,212 images, split **728 train / 484 test (60/40, seed 0)**.
Fires mined from the **728 train** images are added as confusers (y=0, weight 1.5). The **484 test**
images are never seen → the honest bird eval.

### 1.3 Held-out eval splits
| surface | eval source dir | trained on | verdict |
|---|---|---|---|
| **rgb_dataset_test** | `dataset/images/`**`test`** | `…/train` + `…/val` | ✅ DISJOINT — coverage-gap story is honest |
| **antiuav_rgb** | `Anti-UAV/`**`test`**`/RGB` | `Anti-UAV/`**`val`**`/RGB` | ✅ DISJOINT |
| **selcom_val** | `_finetune_selcom_mixed_ft2/images/val` | selcom (**val 311 blocklisted**) | ✅ DISJOINT (by blocklist) |
| **rgb_confuser** | `rgb_confusers_merged/`**`test`** | `…/train` + `…/val` | ✅ DISJOINT |
| **bird.v1i TEST** | bird.v1i 484 held-out names | bird.v1i 728 train names | ✅ DISJOINT (name split, seed 0) |
| svanstrom (RGB) | `svanstrom_paired/RGB` | `svanstrom_paired/RGB` (same dir, no split) | ⚠️ IN-SAMPLE — equal for shipped & v4 (fair Δ), absolute # is in-sample |
| ~~rgb_bird_confuser~~ (cache) | bird.v1i `train` (FULL 1,212) | bird.v1i 728 train ⊂ this | ❌ **LEAKY** → use **bird.v1i TEST** above |

### 1.4 Held-out results (P(drone) ≥ 0.25)
- **rgb_dataset_test recall 0.694 → 0.874** (bare 0.888 → v4 vetoes only 1.6% vs shipped 22%)
- **selcom_val 0.451** (fixed; the mixed-selcom regression is gone)
- antiuav_rgb 0.982 · svanstrom 0.836 (in-sample) — held
- **bird.v1i TEST (484 imgs, 230 fires): v4 keeps 30/230 (13%) vs shipped 91/230 (40%) vs v2 183/230**
  → v4 **beats shipped on unseen birds** (real transfer; birds separable, AUROC 0.981)
- rgb_confuser FP 16 → 21 (**+5**, the only residual; tunable bird-weight or accept)

**No hidden collapse**: v4 holds recall on every cached RGB drone surface. RGB had a clean win because
birds are genuinely separable from drones.

---

## 2. IR filter — `ir_aligned_cbam_thermalonly` (the `cbam` filter)

### 2.1 What it is
The thermal deploy head from `train_aligned.py --thermal-confusers --cbam --no-gray`. Built to fix the
held-out **CBAM drone-recall collapse** the earlier `balanced` filter caused (CBAM valid 0.967 →
0.717). CBAM drones are separable from CBAM airplanes (AUROC 0.964 → MOVABLE), so adding CBAM-train as
GT-aware thermal data recovers them. The `--no-gray` flag made this a clean A/B (isolates the CBAM fix
from any grayscale effect) — so it has **no gray head**.

### 2.2 TRAIN data (`n_train` = 10,157; `train_meta.json`)
- **thermal drones: 8,112** (cap 9000) — sources: Svanström IR (`IR_DRONE_`, 1280), Anti-UAV
  **`val`/IR**, **`IR_dset_final/train`**, **`IR_video/train`** (`IR_DRONE_`),
  `dataset_v3/train` *(dir missing → skipped, lossless)*, **+ CBAM-train drones** (GT-aware).
- **thermal confusers: 2,045** — balanced by **(category × size), per-cell cap 1000**, from
  **`IR_confusers/images/train`** (airplane 3984 / bird 1140 / heli 113) + Svanström IR (`IR_*_`
  prefixes) + **`IR_video/train`** (`IR_*_` prefixes), **+ CBAM-train confusers** (GT-aware).
- **CBAM-train (GT-aware)**: `…CBAM…/`**`train`**`/images`+`/labels`; a fire matching a class-**D**(=1)
  GT box → **drone**, otherwise → **confuser**.
- **gray groups DROPPED** (`--no-gray`).

### 2.3 Held-out eval splits
| surface | eval source dir | trained on | verdict |
|---|---|---|---|
| **CBAM valid** | `…CBAM…/`**`valid`**`/images` (`cbam.pkl`) | `…CBAM…/`**`train`** | ✅ DISJOINT — the recall-recovery proof |
| **IR_confusers val/test** | `IR_confusers/images/`**`val`**+**`test`** (re-mine) | `IR_confusers/images/`**`train`** | ✅ DISJOINT — honest suppression |
| **ir_dset_final** | `IR_dset_final/`**`test`** | `IR_dset_final/`**`train`** | ✅ DISJOINT — the −3.7pp is honest |
| **antiuav_ir** | `Anti-UAV/`**`test`**`/IR` | `Anti-UAV/`**`val`**`/IR` | ✅ DISJOINT |
| **ir_video** | `IR_video/`**`test`** | `IR_video/`**`train`** | ✅ DISJOINT |
| svanstrom_ir | `svanstrom_paired/IR` | `svanstrom_paired/IR` (same dir) | ⚠️ IN-SAMPLE (drone-recall held flat 0.966) |
| ~~ir_confusers~~ (cache) | `IR_confusers/images/`**`train`** | `IR_confusers/images/`**`train`** | ❌ **LEAKY** → use **val/test re-mine** above |

### 2.4 Held-out results
- **CBAM valid @0.05**: cbam **R 0.967 / FP 6** vs shipped 0.917 / 15 vs balanced 0.600 / 2 (bare
  0.967 / 48) → recall collapse **recovered** (balanced 0.717 → cbam 0.967), with fewer FP than shipped.
- **IR_confusers val/test (388 held-out fires)**: cbam@0.05 keeps **22 (94% removed)** vs shipped 90 (77%).
- **Main thermal @0.05**: antiuav_ir 0.937, svanstrom_ir 0.966, ir_video 0.971 — **held**;
  **ir_dset_final 0.965 → 0.928 (−3.7pp)** — the *only* cost, on genuinely airplane-like (AMBIGUOUS)
  ir_dset drones that no threshold separates (`ir_dset_veto_diagnosis.py`).
- **Pareto**: across all four filters, cbam@0.05 kills **94%** of held-out confusers while holding
  drone recall best — it dominates shipped (more suppression + higher CBAM recall), balanced
  (near-equal suppression, vastly better recall), and native (far more suppression).

### 2.5 Caveat — one shared net, two scalers
`train_aligned` fits **one net** on z-scored thermal (+gray when enabled) and emits two deploy heads
(`mlp_aligned.pt` thermal-scaled, `mlp_aligned_gray.pt` gray-scaled) from the *same* `state_dict`.
So the with-gray production run **retrains the shared net** → the thermal numbers above will shift and
**must be re-validated** (`eval_ir_heldout.py` + `filter_acceptance_eval --mode ir`), and the new gray
head re-checked (`ir_grayscale_sweep.py`). The gray path is inherently weak for *any* recipe (bare
gray_svan recall 0.548; balanced-gray ≈ shipped-gray).

---

## 3. Leak map (one-glance)
Genuinely held-out (use freely): rgb_dataset_test, antiuav_rgb, selcom_val, rgb_confuser, **bird.v1i
TEST**, CBAM valid, **IR_confusers val/test**, ir_dset_final, antiuav_ir, ir_video.
**Leaky — never cite for these filters:** `rgb_bird_confuser` cache (full bird.v1i incl. train) →
substitute bird.v1i TEST; `ir_confusers` cache (= train split) → substitute IR_confusers val/test re-mine.
**In-sample (no split exists):** Svanström RGB & IR — fair for shipped-vs-candidate Δ, but the absolute
recall is in-sample; flag if quoted alone.

## 4. Open items
1. **RGB production** — retrain v4 on **all** bird.v1i (drop the 60/40 split, which existed only to
   prove transfer). The thesis bird number then needs a **new** held-out bird set (training on all
   bird.v1i makes `rgb_bird_confuser` a leak).
2. **IR production** — run the with-gray build (`→ mri/results/ir_aligned_cbam/`, ~67 min, gray is the
   only un-cached group), then re-validate thermal + gray (§2.5). Decide the ir_dset −3.7pp op-point.
3. kb: record `v4` / `cbam` as models with provenance + the held-out evals.

---

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_filter_provenance_train_heldout.md` (this doc)

### Weights described (provenance subjects)
- `…\ES_Drone_Thesis\eval\results\_v5_balanced_v4\classifiers\mlp_v5_balanced_v4.pt` (+ `split.json`, `training_data.npz`)
- `…\ES_Drone_Thesis\eval\results\_v5_balanced_v2\classifiers\mlp_v5_balanced_v2.pt` (+ `training_meta.json`)
- `…\ES_Drone_Thesis\mri\results\ir_aligned_cbam_thermalonly\classifiers\mlp_aligned.pt` (+ `train_meta.json`)
- `…\ES_Drone_Thesis\mri\results\_feat_cache\*.npy` (per-group feature cache for surgical IR re-trains)

### Builders / evals referenced (all resident in ES_Drone_Thesis)
- `eval\build_balanced_v4_birdsplit.py`, `eval\build_balanced_v2_surgical.py`, `eval\distill_v5_balanced_remine.py`
- `mri\train_aligned.py` (`--thermal-confusers --cbam --no-gray|--conf-cell-cap N --out`)
- `eval\eval_ir_heldout.py`, `eval\eval_birdtest_heldout.py`, `eval\filter_acceptance_eval.py`,
  `eval\ir_balanced_threshold_sweep.py` (now `--weight/--label`), `eval\ir_grayscale_sweep.py`,
  `eval\cbam_separability_check.py`, `eval\ir_dset_veto_diagnosis.py`, `eval\pipeline_cache.py`
- Prior write-ups: `2026-06-17_rgbtest_filter_regression(_FIX).md`, `2026-06-17_ir_filter_airplane_balance_FIX.md`,
  `2026-06-17_ir_filter_native_vs_aligned.md`
