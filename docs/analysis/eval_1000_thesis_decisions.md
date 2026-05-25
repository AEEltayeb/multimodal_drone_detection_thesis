# Eval-1000 results — thesis decisions

Pulled from the notebook (`eval_1000_results.ipynb`) and supporting CSVs.
Five datasets, ~4,845 frames total: antiuav (1000), svanstrom (1000),
drone_video (1359 drone + ~1250 confuser), rgb_test (796), ir_test (690).

## TL;DR

- **Ship the results as the thesis.** Coverage is strong; nothing missing for a defendable evaluation.
- **One element clearly needs retraining: the SA32 trust classifier**, specifically its IR-side features when IR is grayscale-RGB. Everything else is good-enough or has a known operating-mode workaround.
- **One element should be removed from the stack: selcom-on-IR cross-modal** (running RGB selcom on IR images as a 2nd modality for ir_test). It's been disproven multiple times now.
- **Soft-veto on RGB-only / grayscale modes is non-negotiable.** Argmax destroys recall there.

## What's actually broken (in priority order)

### 1. SA32 classifier is OOD on grayscale-RGB IR
- **Evidence**: drone_video classifier argmax F1=0.54 vs RGB-only F1=0.76. Argmax over-rejects because IR-side features come from grayscale-RGB (OOD for the model). Soft-veto recovers to F1=0.73 by fail-opening on RGB.
- **Cost of the workaround**: soft-veto loses the modality-arbitration benefit; it's basically "use RGB whenever RGB fires".
- **Fix**: retrain SA32 with grayscale-RGB IR-side features in the training set. Mix in drone_video / rgb_test frames as additional training data. **This is the single highest-leverage retraining.**
- **Effort**: medium — features are already computed at eval time; training pipeline exists.

### 2. selcom_1280 has high confuser FPR on real videos
- **Evidence** (from memory + Step 7): selcom_1280 41% confuser FR on YouTube confuser videos — WORSE than baseline (37%) and far worse than retrained_v2 (17%).
- **Mitigation in place**: alert gate (patch verifier) recovers precision: 0.74→0.81 on svanstrom, suppresses 50–60% of drone_video confuser FPs.
- **Decision**: don't retrain selcom yet. The alert-gate compensates for downstream FPR. Selcom's strong-recall trade-off is needed on Svanström.
- **If you do retrain**: do an OOD-balanced run (svanstrom + drone_video confuser frames as hard negatives) — but this is a "nice-to-have", not blocking.

### 3. svanstrom RGB raw precision is genuinely poor (0.45)
- Not a bug — Svanström has huge GT boxes and selcom fires on cloud/horizon FPs. IoP scoring is already correct (memory).
- The pipeline fixes this end-to-end: alert_gate gets P→0.81, classifier gets F1→0.98. **Ships as-is.**

### 4. retrained_v2 recall collapse (existing memory)
- Already documented: retrained_v2 has lowest recall across the board. Memory says "production stack: don't ship retrained_v2 classifier; alert_gate_only cascade is correct".
- **Decision**: keep retrained_v2 OUT of the production stack. It's there only as an ablation row.

## Production stack — what ships

| Element | Production? | Why |
|---|---|---|
| `selcom_1280@imgsz=960/1280` (RGB) | ✅ | Best recall RGB. imgsz=1280 for Svanström, 960 elsewhere. |
| `ir_v3b` (IR) | ✅ | F1=0.95 native; F1=0.66 cross-modal on grayscale-RGB — empirically validated |
| `patch_verifier` (rgb_filter / ir_filter v2) | ✅ | Alert-gate boundary only. v3 stays out (memory: bad) |
| `sa32` trust classifier | ✅ with caveat | Production. **Argmax** on paired/IR-primary; **soft-veto τ=0.95** on RGB-only/grayscale |
| Temporal 2/3 voting | ✅ | Universal small win |
| Alert gate (patch at temporal boundary) | ✅ | Neutral on clean data, big win on confuser-heavy / low-P regimes |
| `retrained_v2` (RGB detector) | ❌ | Recall collapse on real video |
| `selcom-on-IR` synthetic-RGB fallback | ❌ | Drops ir_test P from 0.98 → 0.57 (Step 9). Memory confirms this fails. |
| `fnfn_v3` patch verifier | ❌ | Memory: bad |
| `argmax` mode on RGB-only / grayscale | ❌ | Over-rejects, F1 collapses (drone_video 0.76 → 0.54) |

## Deployment recipe (the one diagram)

```
Paired or IR-primary input:
  YOLO(selcom@960) ─┐
                    ├─► sa32 classifier (argmax) ─► temporal 2/3 ─► alert_gate ─► output
  YOLO(ir_v3b)    ─┘                                                    │
                                                                  patch_verifier
RGB-only / no IR available:
  YOLO(selcom@1280) ─┐
                     ├─► sa32 classifier (soft-veto τ=0.95) ─► temporal 2/3 ─► alert_gate ─► output
  YOLO(ir_v3b on    ─┘
       grayscale-RGB)
```

## Headline numbers to put in the thesis

| Dataset | Production F1 | Pipeline contribution |
|---|---|---|
| antiuav | **0.981** | Classifier argmax pushes 0.969 → 0.981 (+0.012). Saturated. |
| svanstrom | **0.976** | Stack rescues catastrophic raw RGB (P=0.45 → 0.97 final). The clearest pipeline win in the thesis. |
| drone_video drone | **0.733** | Soft-veto retains RGB recall on bird-attack scenes. Argmax would have dropped to 0.54. |
| rgb_test | **0.793** | Soft-veto + alert_gate. Raw RGB was 0.93 — the pipeline costs recall here (cleaner data is hurt by the cascade — defensible: optimised for confuser-heavy). |
| ir_test | **0.953** | IR native + argmax. selcom-on-IR mixed in drops to 0.70. Negative result documented. |

## Conclusions to draw in the prose

1. **The pipeline is value-added on confuser-heavy / low-base-precision data, neutral-to-slightly-negative on clean RGB.** This is the central trade-off and worth a section.
2. **Soft-veto is a deployment-time rule, not a model.** It's the empirical answer to "argmax is over-confident when one modality is OOD". Cheap, no retraining.
3. **Cross-modal grayscale-IR fallback is empirically validated.** F1=0.66 on drone_video, beats every RGB on flock_of_seagulls (F1=0.90). Counter-intuitive but real.
4. **Selcom_1280 isn't strictly "the best detector"** — it's the best-recall detector with the worst confuser FPR. The cascade is what makes it shippable.
5. **The trust classifier earns its keep only when raw-F1 < 0.7 AND confusers are in scope.** On saturated benchmarks like Anti-UAV it's neutral; on Svanström-like clutter it's transformative.

## Limitations to declare (so the thesis isn't fragile)

- rgb_test confuser benchmark is degenerate (118 bird-filename frames, base detector fires on 1) — not a useful stress test, kept for completeness.
- drone_video has no per-size n_gt for "small" on some confuser clips → R=0 in those buckets is meaningless (already noted in memory).
- No re-evaluation of the SA32 classifier after the proposed grayscale retraining — flagged as future work.

## What to do BEFORE the thesis is sent out

- [ ] Retrain SA32 with grayscale-RGB IR-side samples (the only high-leverage retraining).
- [ ] Run a final eval on drone_video + rgb_test post-retrain to see if argmax becomes viable. Likely yes for drone_video.
- [ ] Decide whether to include the post-retrain numbers as a "v2 classifier" row or as the production result. Recommendation: include both — shows the original failure mode AND the fix.

Everything else can ship.
