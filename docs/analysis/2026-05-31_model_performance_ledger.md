# Model Specifications & Performance Ledger

**Date:** 2026-05-31 · **Source of truth:** `knowledge/models.csv` + `knowledge/evals.csv` + `knowledge/eval_configs.csv` (as of this date).

Every number below is copied from a recorded `evals` row and tagged with its **surface + scoring config**, because F1 is only comparable *within* a surface (Svanström=IoP@1280, Anti-UAV=IoU, IR=IoU@640, etc.). Do not cross-compare F1 across surfaces. `clf_own_holdout` rows are each classifier's *own* train-time split — in-domain, not cross-comparable; flagged where used.

> ⚠️ **No FPS / latency exists in the knowledge base.** `latency_ms` is empty in all 120 eval rows. Only *relative* speed claims are on record (V5 MLP ≈50× faster/detection than patch_v2, 1–4 % pipeline overhead). Absolute FPS per architecture is a measurement gap → see [To-Do](#to-do-added-to-the-pipeline).

---

## 1. RGB drone detectors — ranked by surface

Headline production surfaces. Bold = production.

### Svanström (CCTV-style, IoP@0.5, imgsz=1280)
| Rank | Model | P | R | F1 | Note |
|---|---|---|---|---|---|
| 1 | **baseline** (Yolo26n_trained) | 0.940 | 0.961 | **0.950** | best balanced; production open-sky |
| 2 | hardneg_v3more | 0.941 | 0.950 | 0.946 | near-baseline + lower confuser fire |
| 3 | ft4 (+nothing, bare) | 0.443 | 0.914 | 0.596 | high-R/low-P bare; needs verifier |
| 4 | retrained_v2 | 0.943 | 0.306 | 0.462 | **recall collapse — disqualified** |

### Anti-UAV (IoU, imgsz=1280/640) — *saturated*
| Model | P | R | F1 |
|---|---|---|---|
| baseline | 0.992 | 0.995 | **0.994** |
| retrained_v2 | 0.992 | 0.995 | 0.994 (identical — dataset saturated) |
| ft4 (bare) | 0.988 | 0.984 | 0.986 |
| selcom_1280 | — | — | 0.902 (849 FP, bleeds at high R) |

### selcom CCTV held-out val (IoP@0.5, imgsz=1280)
| Rank | Model | P | R | F1 | Note |
|---|---|---|---|---|---|
| 1 | selcom_mixed_ft3_1280 | 0.847 | 0.488 | **0.619** | candidate CCTV weights |
| 2 | ft4 + V5 MLP | 0.956 | 0.444 | 0.607 | verifier-cleaned |
| 3 | ft4 (bare) | 0.858 | 0.451 | 0.591 | |
| 4 | **selcom_mixed_ft2_1280** | 0.762 | 0.468 | 0.580 | current CCTV prod weights |
| — | selcom_960 (per-size holdout) | 0.88 | 0.44 | 0.585 | **#1 cross-surface generalist** |
| — | baseline (pre-FT) | 0.413 | 0.088 | 0.145 | why selcom FT exists |

### OOD / real-video (relative ordering only)
| Surface | baseline | retrained_v2 | selcom_1280 | selcom_640 |
|---|---|---|---|---|
| Roboflow OOD drone (IoU@640, raw) | **0.820** | 0.813 | — | 0.84 (+patch, #1) |
| Drone video (IoP) F1 | 0.760 | 0.605 | 0.721 | 0.730 |
| Confuser video FIRE rate ↓ (no GT) | 0.475 | **0.264** | 0.593 (worst) | 0.423 |

**RGB continuum:** retrained_v2 = high-P/low-R (collapses on recall), baseline = balanced generalist + best OOD drone, selcom = high-R/low-P (best on bird-mixed CCTV but worst OOD confuser robustness).

---

## 2. IR drone detectors — ranked (IR_dset_final test, IoU, imgsz=640)
| Rank | Model | P | R | F1 | Status |
|---|---|---|---|---|---|
| 1 | **ir_v3b** (finetune_v3b) | 0.957 | 0.977 | **0.967** | **production** |
| 1 | ir_final | 0.955 | 0.980 | 0.967 | base for v3b |
| 3 | ir_v6 | — | — | 0.931 | superseded |
| 4 | ir_v4 | 0.895 | 0.669 | 0.765 | superseded |
| 5 | ir_v5 | — | — | 0.737 | regression (noisy data) |
| 6 | ir_v3 | — | — | 0.611 | superseded |
| 7 | ir_v2 | 0.458 | 0.406 | 0.430 | archived |

**OOD IR (Roboflow, sensor-augmented worst-case):** ir_final night-raw F1 0.344; on `ir_mixed_cbam` raw 0.636. IR detector is precise in-domain (raw-F1 0.95–0.96) but recall-brittle on heavily sensor-shifted OOD.

---

## 3. IR-on-grayscale (cross-modal RGB fallback) — drone video (IoP)
| Mode | P | R | F1 | Note |
|---|---|---|---|---|
| **IR_final on grayscale-RGB** | 0.743 | 0.557 | **0.636** | validated cross-modal fallback |
| IR_final on raw-RGB | 0.647 | 0.191 | 0.295 | unusable — must convert to gray first |

Confuser-video fire rate: IR-gray **0.256 (lowest among usable detectors)**. Empirically the grayscale fallback path works on drone-positive data; raw-RGB-into-IR does not.

---

## 4. Confusion filters — RGB verifiers (rank by hallucination ↓)

### Per-detection verifiers (CNN/MLP), Svanström head-to-head (IoP@0.5, 1280, stride-9)
| Rank | Verifier | Svan F1 | Confuser halluc ↓ | selcom F1 | rgb_dataset F1 | Status |
|---|---|---|---|---|---|---|
| 1 | **mlp_v5** (pure_1x8) | **0.869** | **0.008** (FP 21) | 0.607 | 0.792 ⚠ | **PRODUCTION (per-frame)** |
| 2 | patch_v2 | 0.768 | 0.107 (FP 282) | 0.591 | 0.904 | fallback (photo-style RGB) |
| — | ft4 bare (no verifier) | 0.596 | 0.317 (FP 835) | 0.591 | 0.929 | baseline |

⚠ V5 carve-out: rgb_dataset_test regresses −11 pp F1 (recall ceiling, structural — see `v5-rgbds-ceiling`); remine_rgb attempt 0.763 confirmed dead-end.

### Patch-CNN version sweep (Svan IoP@640, May-10 config)
patch_v4 0.9331 ≈ **patch_v2 0.9311** > patch_v1 0.9241 > patch_v3 0.8781 (over-aggressive). v2 is the shipped pick; v4≈v2.

---

## 5. Confusion filters — IR verifiers — **SHIP NONE per-frame** (verdict 2026-05-30)
| Model | Surface | Result | Verdict |
|---|---|---|---|
| mlp_v5_ir | ir_dset_final | lost 71 TP for 4 FP @thr0.05 | **net-negative on normal IR** |
| mlp_v5_ir | CBAM (in-domain) | 0.699→0.885 F1 | only helps confuser-saturated in-domain |
| mlp_v5_ir_dronediv | held-out thermal | F1 0.962 (dR −0.007) | recall-safe insurance only; FP headroom ≈0 |
| mlp_v5_gray | grayscale fallback | confuser FP 325→13 (−96 %) | for grayscale path ONLY, not thermal |

IR detector is already precise (raw-F1 0.95–0.96) → no FP headroom for a verifier to recover. Verifiers pay off where the detector hallucinates (RGB), not IR.

---

## 6. Trust / scene classifiers — 3-way (drone/confuser/bg), macro-F1
| Rank | Model | F1m | acc | Note |
|---|---|---|---|---|
| 1 | lean13 | 0.979 | 0.987 | ⚠ scene-fingerprint overfit |
| 2 | lean19 | 0.978 | 0.990 | smallest train-test gap |
| 3 | lean10 | 0.963 | 0.980 | OOD-robust, ~½ compute |
| 4 | lean17 | 0.958 | 0.977 | ⚠ pos_x fingerprint |
| 5 | control40 | 0.920 | 0.953 | best raw / worst confuser reject |
| 6 | retrained_v2_32feat | 0.842 | 0.923 | collapses on OOD video (0.280) |

### OOD confuser-zoo FIRE rate ↓ (what actually matters for a filter)
| Model | S2 fire | S3 fire | Note |
|---|---|---|---|
| fusion_no_fn_v1.1 | **0.016** | 0.008 | safest open-world fallback |
| sa32 (**production**) | 0.205 | 0.103 | fires 13× fnfn but best on Svan-distribution |
| control40 | 0.212 | 0.094 | deprecated |

**Production trust classifier = sa32.** dual_classifier_v3 strictly dominates sa32 on RGB-fallback surfaces (BIRD 0.247→1.000, rgb_test 0.690→0.960) — promotion candidate.

---

## 7. Overall production stack (current picks)
| Slot | Model | Why |
|---|---|---|
| RGB detector (open-sky) | **baseline** | best balanced + OOD drone |
| RGB detector (CCTV) | selcom_mixed_ft2_1280 | CCTV recall; ft3 candidate |
| RGB detector (V5 stack) | ft4 | feeds V5 verifier |
| IR detector | **ir_v3b** | F1 0.967 |
| RGB→gray fallback | IR_final on grayscale | F1 0.636 cross-modal |
| RGB verifier | **mlp_v5** (per-frame) | F1 +10pp, halluc 0.008, 50× faster |
| IR verifier | **none** | net-negative on normal thermal |
| Trust classifier | sa32 | best Svan-distribution |
| Cascade | alert_gate_only | per-frame clf is a recall tax off-Svan |

See `memory/project_production_stack.md` for the full rationale.

---

## To-Do (added to the pipeline)
1. **FPS / latency benchmark** — measure & record absolute ms/frame (and FPS) per architecture on deployment hardware into `evals` rows under config `e2e_latency`. Currently zero absolute timing data exists.
2. **Classifier is outdated** — sa32/lean line predates the current detector+V5-verifier stack. Re-eval the classifier *in the current pipeline*, or retrain on current-detector fusion features if it underperforms.
3. **Full-pipeline re-eval** — re-run the entire end-to-end pipeline (detector → IR-gray → classifier → temporal → verifier) with current production weights; the dashboard ablation predates V5-as-production.

---

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\docs\analysis\2026-05-31_model_performance_ledger.md` (this file)

Sources (read-only): `knowledge/models.csv`, `knowledge/evals.csv`, `knowledge/eval_configs.csv`, `knowledge/views/rankings.md`.
