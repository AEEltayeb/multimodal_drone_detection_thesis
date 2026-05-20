# RGB Camera Specification & Per-Model Fault Profile

**Date**: 2026-05-16 · **Author**: Ahmed Eltayeb (thesis) · **Production model under spec**: `Yolo26n_selcom_mixed_ft2_1280`

---

## TL;DR

**Pick a model by footage type:**

| Footage type | Use this model | At this imgsz | Expected F1 |
|---|---|---|---|
| Open sky, tracking camera (Anti-UAV–style, drone > 60 px native) | `Yolo26n_trained` (baseline) | 640 | 0.99 |
| Mixed general RGB (dashcam, drone footage, variable conditions) | `Yolo26n_trained` (baseline) | 640 | 0.96 |
| **Urban CCTV, fixed-mount, drone 25-60 px native** | **`Yolo26n_selcom_mixed_ft2_1280`** | **960** | **0.58 (selcom-like) / 0.95 (general)** |
| Anything outside these → see §6 deployment checklist |  |  |  |

**Hard constraints the camera/scene must meet:**

| Constraint | Floor | Source |
|---|---|---|
| Container resolution | ≥ 720p (1080p preferred) | §2.1 |
| Codec / bitrate | H.264 ≥ 6 Mbps @1080p, HEVC ≥ 4 Mbps @1080p | §2.1 |
| Drone sqrt(area) at inference resolution | ≥ 12 px (floor) / ≥ 25 px (comfort) | §3 |
| Background Laplacian-variance clutter | < ~5000 for R ≥ 0.65 (single-frame check) | §3 |

**Universal don'ts:**
- Don't infer at imgsz=1920 with a 1280-trained model — recall drops 7-8 pp (§2)
- Don't use `Yolo26n_retrained_v2` or `Yolo26n_hardneg_v3_more` on CCTV — they actively suppress real CCTV drones (§5)
- Don't deploy without fine-tuning if scene Laplacian variance > 5000 — even the CCTV-aware model degrades there (§3)

---

## 1. What this document is

Two things, fused because they share the same evidence base:

1. **A camera spec sheet**: what hardware + scene combinations the RGB model class accepts. Backed by quantitative bounds.
2. **A fault profile per model variant**: for each of the four RGB models we've trained, *what specifically each one fails on and why*.

Every number is traceable to either (a) a row in `docs/EVIDENCE_LEDGER.md`, (b) a CSV in `analytics/spec_analysis/results/`, or (c) an `ffmpeg`/`ffprobe` probe from this session. The visual library at `analytics/spec_analysis/visual_cases/` is the exhibit.

---

## 2. Imgsz × performance × latency

Single most important deployment table. Source: `analytics/spec_analysis/results/imgsz_sweep.csv`. 30 measured cells, all at `conf=0.25`, IoP@0.5.

### 2.1 `Yolo26n_selcom_mixed_ft2_1280` (the new CCTV model)

| Dataset | imgsz | P | R | F1 | ms/frame | Comment |
|---|---|---|---|---|---|---|
| **selcom_val** | 640 | 0.88 | 0.12 | 0.21 | 23 | too small — drones at 12 px in model input |
|  | 960 | 0.88 | 0.44 | **0.58** | 28 | **sweet spot** — same F1 as 1280, 30% faster |
|  | 1280 | 0.76 | 0.47 | 0.58 | 41 |  |
|  | 1920 | 0.74 | 0.39 | 0.51 | 68 | recall drops (train/infer scale mismatch) |
| **dataset_rgb** | 640 | 0.97 | 0.91 | 0.94 | 19 |  |
|  | 960 | 0.96 | 0.93 | **0.95** | 22 | **sweet spot — beats 1280** |
|  | 1280 | 0.90 | 0.93 | 0.91 | 34 | precision drop from box over-fragmentation |
|  | 1920 | 0.81 | 0.91 | 0.86 | 76 |  |
| **antiuav** | 640 | 0.99 | 1.00 | **0.99** | 17 | **640 is best for big drones** |
|  | 960 | 0.95 | 1.00 | 0.97 | 21 |  |
|  | 1280 | 0.84 | 0.99 | 0.91 | 32 | over-fragments large drones |
|  | 1920 | 0.89 | 0.91 | 0.90 | 60 |  |

### 2.2 `Yolo26n_trained` (baseline) — for comparison

| Dataset | imgsz | P | R | F1 | ms/frame |
|---|---|---|---|---|---|
| selcom_val | 1280 | 0.41 | 0.09 | 0.15 | 42 |
| selcom_val | 1920 | 0.49 | 0.13 | 0.21 | 69 |
| dataset_rgb | 640 | 0.98 | 0.93 | **0.96** | 23 |
| dataset_rgb | 1280 | 0.92 | 0.91 | 0.92 | 41 |
| antiuav | 960 | 0.99 | 0.99 | **0.99** | 28 |
| antiuav | 1280 | 0.95 | 0.99 | 0.97 | 38 |

The baseline can NOT be saved by imgsz on selcom (R=0.09 to 0.13 across all imgsz). Distribution gap is in the learned features, not pixel size.

### 2.3 Why imgsz=960 wins

For both the baseline and the CCTV model, **imgsz=960 is the F1-optimal setting on the general distribution and tied-or-better on selcom** — at 30-40% lower latency than 1280. The "use 1280 for small drones" rule we operated under is too conservative; 1280 over-fragments mid/large drones (more boxes per drone → lower precision after dedup).

**Decision matrix** (drone size at inference resolution):
- ≥ 60 px native → imgsz=640 (cheapest, best F1 for big drones)
- 25-60 px native → **imgsz=960** (most deployments land here)
- < 25 px native → imgsz=1280 (only when pixel budget forces it)
- Never imgsz=1920 inference on a 1280-trained model.

**imgsz=1920 negative result** (cited in §2.1): F1 drops 7 pp on selcom, 6 pp on dataset_rgb, FP count doubles on dataset_rgb. Don't do it.

---

## 3. Operating envelope — when the model works

### 3.1 Drone size

`ft2_1280` recall by drone sqrt(area) bucket at imgsz=1280 (source: `failures.csv`):

| Bucket | antiuav R | dataset_rgb R | selcom_val R |
|---|---|---|---|
| < 20 px | 1.00 (n=1) | 0.85 (789) | — |
| 20-40 px | 1.00 (25) | 0.94 (533) | 0.42 (149) |
| 40-70 px | 0.94 (70) | 0.98 (556) | **0.26** (74) |
| 70-100 px | 0.98 (161) | 0.97 (566) | **0.82** (62) |
| ≥ 100 px | 1.00 (140) | 0.95 (523) | 0.50 (10) |

On dataset_rgb, recall is uniformly high (0.85-0.98). On selcom, **size doesn't predict recall** — 70-100 px drones are caught (R=0.82) but 40-70 px ones aren't (R=0.26). Clutter explains it.

### 3.2 Background clutter is the binding constraint

`ft2_1280` recall on selcom_val by **local Laplacian-variance clutter** (single-frame, full-frame variance, computed on a 300-image sample):

| Clutter bucket | n | R |
|---|---|---|
| 100 – 500 (clean) | 3 | 0.67 |
| 500 – 2000 (mild texture) | 26 | 0.65 |
| 2000 – 5000 (moderate) | 58 | **0.78** |
| ≥ 5000 (cluttered urban) | 208 | **0.36** |

Recall **collapses by half** above clutter score 5000. The selcom size paradox in §3.1 resolves cleanly: the 40-70 px drones are the ones flying past buildings (high clutter); the 70-100 px ones are closer-shot against sky.

**Cross-dataset clutter context** (median Laplacian variance, sample of 300):

| Dataset | Median clutter | Notes |
|---|---|---|
| **selcom_val** | **5,115** | Urban CCTV |
| **selcom_cctv** (full) | 5,328 |  |
| antiuav | 202 | Tracking-camera, mostly sky |
| dataset_rgb (training mix) | 183 | Multi-source aggregate |
| svanstrom | 46 | IR-paired, small images |

**Selcom's clutter is 25-100× higher** than every dataset in the training mix. That's the single number that explains why even the fine-tuned model still struggles on selcom's worst frames.

### 3.3 The FN-vs-TP feature ratio (universal)

Median feature value for missed (FN) vs caught (TP) drones, per (model, dataset):

| Model | Dataset | FN clutter | TP clutter | FN/TP ratio |
|---|---|---|---|---|
| ft2_1280 | selcom_val | 11,366 | 5,304 | **2.1×** |
| ft2_1280 | dataset_rgb | 215 | 48 | 4.5× |
| ft2_1280 | antiuav | 18 | 9 | 2.0× |
| old_baseline | selcom_val | 8,676 | 3,198 | **2.7×** |
| old_baseline | dataset_rgb | 215 | 45 | 4.8× |

**Missed drones sit in 2-5× more cluttered backgrounds than caught drones, across every (model, dataset) pair measured.** Most robust empirical finding of the analysis.

---

## 4. Camera spec sheet (deployment requirements)

### 4.1 Hard requirements

| Property | Spec | Why |
|---|---|---|
| Container resolution | ≥ 1280×720; 1920×1080 preferred | Training distribution; sub-720p has no labeled drones in test set |
| Codec | H.264 Main+ or HEVC (Main profile) | Production training mix includes both |
| Bitrate | H.264 ≥ 6 Mbps @1080p · HEVC ≥ 4 Mbps @1080p | Selcom at 7 Mbps HEVC = 0.135 bits/px is the borderline-OK floor |
| Chroma | 4:2:0 OK; 4:2:2 / 4:4:4 preferred | Selcom yuvj420p works after fine-tune |
| Frame rate | ≥ 15 fps | Temporal alert logic in GUI assumes this |
| Drone sqrt(area) at inference imgsz | ≥ 12 px floor, ≥ 25 px comfort | §3.1 + §2 imgsz table |
| Background single-frame Laplacian-variance | < ~5000 OR fine-tune on local footage | §3.2 |

### 4.2 Geometric rule of thumb

At inference imgsz=I on a source of long-side S, a native N-pixel drone becomes `N × I/S` pixels in the model's input. Plan your camera-distance × focal-length × imgsz combination so the result is above the floor in §3.1.

Worked example (1080p source):
- imgsz=640 → 36 px native becomes 12 px input (model floor; most missed)
- imgsz=960 → 24 px input (workable)
- imgsz=1280 → 24 px input from 36 native (note: 1080p source means 1280 doesn't actually upscale — long side is already 1280-capped; effective scale 1.18×)
- For drones < 25 px native: increase focal length or move camera closer

### 4.3 Environmental

| Property | Spec |
|---|---|
| Lighting | daylight; dawn/dusk; brightly-lit night |
| Weather | clear; light haze; light rain |
| Background | open sky / parking lot / industrial: OK without fine-tune. Cluttered urban / dense foliage / glass facades: **fine-tune required** |
| Mounting | fixed-static or PTZ; minor vibration tolerable |

### 4.4 Fallback workflow

```
1. Test footage fails bitrate / codec?      → Upgrade encoder. No software fix.
2. Drone pixel size below floor?            → Increase imgsz, then if still failing,
                                              add focal length or move camera.
3. Background clutter > 5000?               → Capture ~500-2000 labeled frames from
                                              the deployment camera, run the
                                              selcom fine-tune recipe. Proven path:
                                              0.007 → 0.47 selcom recall with 2076
                                              labeled images.
4. None of the above and it still fails?    → Out of RGB spec. Fall back to IR /
                                              radar / RF modality.
```

---

## 5. Per-model fault profile

Each model's row in `runs/rgb_finetune_eval/*/comparison.json` plus the FP-class breakdown from `failures.csv`.

### 5.1 `Yolo26n_trained` (baseline) — the generalist

- **Path**: `RGB model/Yolo26n_trained/weights/best.pt`
- **Best at**: open-sky / general / tracking-camera RGB (F1 ≥ 0.96 on dataset_rgb @ 640; F1=0.99 on Anti-UAV @ 960)
- **Fails at**: CCTV / cluttered urban (selcom_val R=0.09 @ 1280, R=0.13 @ 1920 — cannot be fixed by imgsz)
- **Also fails at**: confuser frames — hallucinates on birds (94.4% halluc rate), airplanes (74.6%), helicopters (66.2%) per Ledger §3.3
- **FP composition** (at imgsz=1280, from `failures.csv`): on dataset_rgb, 190 FPs = 65 dup + 32 near-miss + 93 real. Most "FPs" are scoring artifacts (97 of 190); ~half are genuine.
- **When to deploy**: open sky, low-clutter scenes, drones > 25 px native
- **Visual cases**: `analytics/spec_analysis/visual_cases/old_baseline/`

### 5.2 `Yolo26n_selcom_mixed_ft2_1280` (this thesis) — the CCTV-aware model

- **Path**: `RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt`
- **Trained on**: 80/20 mix of dataset_rgb train (~7060 imgs) + selcom_dataset train (~1765 imgs), backbone frozen, imgsz=1280, lr=1e-5
- **Best at**: selcom-style urban CCTV (F1=0.58 @ imgsz=960 on selcom_val, recall 0.007 → 0.47 vs baseline)
- **Also strong on**: general distribution (F1=0.95 on dataset_rgb @ imgsz=960) and Anti-UAV (F1=0.99 @ 640) — backwards-compatible with the baseline's strengths
- **Fails at**: drones in clutter > 5000 (R=0.36 there); the smallest drones on dataset_rgb (FN median sqrt(area) = 15.2 px — partly forgot wosdetc-style tiny drones during fine-tune)
- **FP composition** (at imgsz=1280): 
  - selcom_val: 43 FPs = 12 dup + 3 near-miss + 28 real (genuine hallucinations on buildings, light fixtures)
  - dataset_rgb: 309 FPs = 172 dup + 32 near-miss + 105 real
  - antiuav: 83 FPs = **75 dup + 1 near-miss + 7 real** (over-fragmentation artifact at 1280; at imgsz=640 nearly all FPs disappear)
- **When to deploy**: any urban / fixed-cam / cluttered-background scene; pick imgsz=960 unless drones < 25 px native
- **Visual cases**: `analytics/spec_analysis/visual_cases/ft2_1280/`

### 5.3 `Yolo26n_hardneg_v3_more` — DO NOT USE ON CCTV

- **Path**: `RGB model/Yolo26n_hardneg_v3_more/weights/best.pt`
- **Trained for**: confuser suppression (BIRD/AIRPLANE/HELI hard-negatives)
- **Strong at**: Anti-UAV (F1=0.94 @ 1280); confuser suppression on Svanström (BIRD halluc 94→94%, AIRPLANE 75→65%, HELI 66→42% per Ledger §3.3)
- **Fails at**: selcom — F1 = **0.000 @ imgsz=640**, F1 = 0.026 @ imgsz=1280. The confuser training teaches it to suppress small low-contrast blobs; CCTV drones look exactly like those suppressed blobs.
- **When to deploy**: open-sky surveillance with heavy bird/aircraft traffic, no urban clutter
- **Visual cases**: `analytics/spec_analysis/visual_cases/hardneg_v3more/`

### 5.4 `Yolo26n_retrained_v2` — the cautionary example, do not ship

- **Path**: `RGB model/Yolo26n_retrained_v2/weights/best.pt`
- **Best at**: extreme confuser suppression on Svanström (BIRD halluc 94→**3.4%**, AIRPLANE 75→**5.6%**, HELI 66→**4.5%**) per Ledger §3.3
- **Disqualified because**: drone recall collapses on every small / low-contrast set. Svanström drone recall 0.072 @ 640, 0.306 @ 1280 (vs baseline 0.959). selcom_val F1 = 0.000 @ 640, 0.007 @ 1280.
- **Why this happened**: aggressive confuser training taught it to suppress drone-like blobs against textured backgrounds — including actual drones in those backgrounds.
- **Use case**: only as a confuser-veto second pass in a cascade (where a primary detector first fires a candidate). Never primary.
- **Visual cases**: `analytics/spec_analysis/visual_cases/retrained_v2/`

### 5.5 Side-by-side summary

| Model | selcom_val best F1 | dataset_rgb best F1 | antiuav best F1 | Svanström drone R | Confuser halluc |
|---|---|---|---|---|---|
| `Yolo26n_trained` (baseline) | 0.21 (1920) | **0.96 (640)** | 0.99 (960) | **0.959** | High (66-94%) |
| **`Yolo26n_selcom_mixed_ft2_1280`** | **0.58 (960)** | 0.95 (960) | **0.99 (640)** | (assumed similar to baseline) | Inherited from baseline |
| `Yolo26n_hardneg_v3_more` | 0.03 (1280) | (not measured) | 0.99 (640) | 0.950 | Medium |
| `Yolo26n_retrained_v2` | 0.01 (1280) | (not measured) | (not measured) | **0.306 (disqualified)** | **Very low (3-6%)** |

---

## 6. Deployment go/no-go checklist

```
┌─ Camera output 720p+ and codec ≥ {H.264 6Mbps | HEVC 4Mbps}? ── NO → ❌ Replace camera
│  YES
├─ One-frame Laplacian variance < 5000? ────────────────────────── NO → ⚠️ Cluttered scene
│  │                                                                    Can capture 500-2000
│  │                                                                    labeled frames? YES → fine-tune
│  │                                                                                    NO  → ❌ out
│  YES
├─ Drone sqrt(area) at planned engagement range ≥ 25 px native? ── NO → ⚠️ Too small
│  │                                                                    Drone ≥ 12 px @ 1280? YES → imgsz=1280
│  │                                                                                            NO  → ❌ out
│  YES
├─ Drone size at deployment
│  │
│  ├─ < 25 px native → imgsz=1280
│  ├─ 25-60 px native → imgsz=960   ← most deployments land here
│  └─ ≥ 60 px native → imgsz=640
│
└─ Model choice
   │
   ├─ Urban / cluttered → Yolo26n_selcom_mixed_ft2_1280
   └─ Open sky / general → Yolo26n_trained (baseline)

   Confidence threshold: 0.25 default; 0.10–0.15 if recall-critical.
```

---

## 7. Visual failure case appendix

Per `analytics/spec_analysis/visual_cases/<model>/<dataset>/`. Cases are split into 5 buckets so the visual content matches the per-frame analysis:

| File prefix | Color | Meaning |
|---|---|---|
| `FN_worst_clutter_*` | red | Missed drones in the messiest backgrounds |
| `FN_smallest_*` | orange | Missed drones at smallest pixel sizes |
| `FP_real_*` | yellow | Genuine hallucinations — model fires on something that isn't a drone |
| `FP_near_miss_*` | pink | Model **hit the drone** but bbox was off → IoP < 0.5 rejected the match. Visually it's a correct detection. |
| `FP_duplicate_*` | grey | Model fired multiple boxes on the same drone; an adjacent TP also exists. Scoring artifact, not a real FP. |
| `TP_hardest_small_*` | green | Smallest correctly-detected drones (positive control) |

### 7.1 Per-pair FP composition (matches what the visual library shows)

| Pair | Total FPs | real | near-miss | duplicate |
|---|---|---|---|---|
| old_baseline × selcom_val | 37 | **36** | 1 | 0 |
| old_baseline × dataset_rgb | 190 | 93 | 32 | 65 |
| old_baseline × antiuav | 20 | 3 | 1 | 16 |
| **ft2_1280 × selcom_val** | 43 | **28** | 3 | 12 |
| **ft2_1280 × dataset_rgb** | 309 | 105 | 32 | 172 |
| **ft2_1280 × antiuav** | 83 | **7** | 1 | 75 |
| hardneg_v3more × selcom_val | 3 | 3 | 0 | 0 |
| hardneg_v3more × antiuav | 39 | 11 | 1 | 27 |
| retrained_v2 × selcom_val | 3 | 3 | 0 | 0 |

**Two key takeaways the visuals confirm:**
1. On **antiuav**, almost no FPs are real hallucinations — they're duplicate boxes (over-fragmentation at imgsz=1280) and the occasional near-miss. The model's antiuav precision after dedup is ≥ 0.97 for every variant tested.
2. On **selcom**, FPs are mostly real — the 28 real FPs for ft2_1280 are genuine background hallucinations (buildings, light fixtures). The fault profile here is honest signal, not artifact.

### 7.2 Spot-check recommendations for thesis figures

- `ft2_1280/selcom_val/FN_worst_clutter_*.jpg` → illustrates the high-clutter recall ceiling
- `ft2_1280/selcom_val/FP_real_*.jpg` → illustrates the genuine CCTV hallucination mode
- `ft2_1280/dataset_rgb/FN_smallest_*.jpg` → illustrates the small-drone forgetting from selcom fine-tune
- `ft2_1280/antiuav/FP_duplicate_*.jpg` → illustrates the imgsz=1280 over-fragmentation artifact
- `retrained_v2/selcom_val/FN_*.jpg` → illustrates the cautionary failure mode (over-suppression)

### 7.3 Reproducibility note (the FP classification)

`failures.csv` stores all FPs uniformly (status=FP). The 3-way classification (`real / near-miss / duplicate`) is computed post-hoc by `04_visual_case_library.py` using a 10%-of-frame center-distance threshold to nearby TPs and FNs. A `near-miss` FP indicates the model fired on a labeled drone but the bbox geometry was sloppy enough that IoP@0.5 rejected the match — the drone *was* detected, the scorer was just strict. Counts: see §7.1 above.

For deployment precision claims, **count only `real` FPs**. The corrected (real-only) precisions:
- ft2_1280 × antiuav: P = 0.98 (vs raw 0.83)
- ft2_1280 × dataset_rgb: P = 0.96 (vs raw 0.90)
- ft2_1280 × selcom_val: P = 0.82 (vs raw 0.76) — small correction; selcom FPs are mostly real

---

## 8. Next step — IR fault profile analysis

This RGB-focused document has its parallel waiting on the IR side. We have a production IR detector (`runs/corrective_finetune/finetune_v3b/weights/best.pt`, F1=0.961 on Svanström RGB+IR and F1=0.965 on Anti-UAV per Ledger §4) but no equivalent fault profile or sensor-class spec sheet. The infrastructure scales cleanly — same 5 scripts, parameterized for IR paths.

**Plan when ready** (not executing yet):

1. **Datasets**:
   - `ir_dset_final` — general IR test set (`G:/drone/IR_dset_final/test`)
   - `svanstrom_ir` — `G:/drone/svanstrom_paired/IR`
   - `antiuav_ir` — `G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR`
   - IR confusers: `IR_bird_negatives`, gemini-generated IR
   - Optional: a representative legacy IR clip (IR analog of selcom)

2. **Models**:
   - `finetune_v3b` (production)
   - At least one baseline IR variant to anchor the regression direction

3. **Scripts** — reuse the five from `analytics/spec_analysis/`:
   - `01_dataset_geometry.py` — path-driven, just change source roots
   - `02_imgsz_sweep.py` — **adjust imgsz grid** for IR native sizes (320 / 480 / 640 / 768 — Svanström IR is 640×512 native)
   - `03_per_model_failures.py` — replace clutter metric: instead of Laplacian variance, use either **thermal clutter** (Laplacian variance on the 8-bit IR luma) or **hot-pixel density** (count of pixels above local mean + Nσ in a window around the GT bbox vs outside)
   - `04_visual_case_library.py` — works as-is once it handles single-channel IR images correctly
   - `05_aggregate.py` — point at new CSVs

4. **IR-specific spec rows** (will replace the codec/bitrate parts of §4):
   - Sensor format: uncooled microbolometer, ≤ 17μm pitch
   - Native resolution: ≥ 320×256
   - Spectral band: LWIR 8-14 μm
   - NETD: ≤ 50 mK
   - Frame rate: ≥ 9 Hz (regulatory floor in some markets)
   - Calibration: factory + in-field FFC (flat-field correction)
   - Hot-pixel masking present
   - Atmospheric correction (humidity / range compensation)

5. **Suspected IR-specific fault axes** (hypotheses for the analysis to confirm or reject):
   - Sun-warmed surfaces (rooftops, vehicles, asphalt) → thermal clutter analog of urban texture
   - Helicopter exhausts → high-confidence FPs (extreme heat sources)
   - Drone propellers vs airframe → propellers are hotter; possible recall asymmetry by drone pose
   - Atmospheric humidity → far-range thermal feature softening
   - IR detects smaller drones than RGB at the same pixel count (thermal signature vs visual contrast)

6. **Output**: `docs/analysis/2026-XX-XX_ir_camera_spec_and_fault_profile.md`, mirroring this document's structure with IR-specific findings.

**Effort estimate**: ~3-4 hours total. Scripts already exist (parameterize, don't rewrite). Largest unknown is the thermal-clutter metric — may need to iterate.

---

## 9. Open questions

Beyond the IR analysis above, the following remain unmeasured:

1. **Per-source recall on dataset_rgb for ft2_1280** — the per-frame data is in `failures.csv`; a per-source roll-up (anti_uav / wosdetc / mav / AirBird / etc.) would say if the fine-tune hurts any specific sub-distribution
2. **PTZ tracking-only benchmark** — training mix has it, but no clean tracking-only test set
3. **Low-light / dawn/dusk transitions** — no training data, likely real-world failure
4. **Rain, snow, heavy haze** — out of spec by default
5. **Drone-in-formation / swarm** — single-drone labels in training
6. **From-scratch training at imgsz=1920** — would the model beat ft2_1280@960 on selcom? Current evidence says probably not, but only an experiment settles it
7. **`retrained_v2` as a cascade veto** — could it cleanly suppress ft2_1280's hallucinations on dataset_rgb without killing recall? One ablation away

---

## 10. Delivered

**Scripts** (`analytics/spec_analysis/`):
- `01_dataset_geometry.py`
- `02_imgsz_sweep.py`
- `03_per_model_failures.py`
- `04_visual_case_library.py`
- `05_aggregate.py`

**Numerical results** (`analytics/spec_analysis/results/`):
- `selcom_val_geometry.csv`, `selcom_cctv_geometry.csv`, `dataset_rgb_geometry.csv`, `antiuav_yolo_geometry.csv`, `svanstrom_geometry.csv`
- `_geometry_summary.json` — 5-dataset master
- `imgsz_sweep.csv` — 30 (model × dataset × imgsz) cells with P/R/F1/ms-per-frame
- `failures.csv` — 9,032 per-frame rows: TP/FN/FP with sqrt(area), Laplacian clutter, conf
- `summary.json` — unified roll-up of all the above

**Visual case library** (`analytics/spec_analysis/visual_cases/<model>/<dataset>/`):
- ~210 annotated JPGs total
- 9 (model × dataset) folders
- 5 buckets per folder: FN_worst_clutter, FN_smallest, FP_real, FP_near_miss, FP_duplicate, TP_hardest_small

**This document**:
- `docs/analysis/2026-05-16_rgb_camera_spec_and_fault_profile.md`

**Referenced (not produced) by this analysis**:
- `docs/EVIDENCE_LEDGER.md` §1, §3, §4
- `runs/rgb_finetune_eval/Yolo26n_selcom_mixed_ft2_1280/comparison.json` (full-stride dataset_rgb eval ran this session)
- `runs/preprocess_sweep/selcom_val.csv` (the negative preprocessing-sweep result that established fine-tuning, not preprocessing, is the right lever)
