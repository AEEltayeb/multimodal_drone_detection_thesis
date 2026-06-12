# IR Eval Audit — "Is the detector really that good, or did we measure badly?" (2026-05-31)

> Self-audit of the IR verifier conclusion at max scrutiny. Verdict: **the measurement was
> the weak link, not the model.** The IR detector *does* hallucinate; we tested it where it
> can't show. A clean held-out IR confuser benchmark was never built (the RGB side had one),
> and the one good thermal-confuser set (CBAM) was contaminated by training the MLP on it.

---

## 1. The contradiction that should bother us

- We concluded "IR detector is already precise (raw-F1 0.95–0.96) → no verifier needed."
- Yet on **CBAM** (the one held-out *thermal aerial confuser* set) v3b scores **P=0.547** and fires
  on ~**40% of bird/plane images** (48 FP / 120 imgs). That is *not* a precise model — it's a
  model that hallucinates badly on confusers we rarely showed it.

Both are true because they're measured on different surfaces. The "precise" number is an
artifact of **confuser-poor** eval surfaces.

## 2. Why "precise" was an illusion (measurement failures)

| # | Flaw | Consequence |
|---|---|---|
| 1 | **Standard IR surfaces are confuser-poor.** Svanström-IR / Anti-UAV-IR / IR_dset are drone-tracking/benchmark sets with ~no aerial confusers. | P=0.95–0.99 there says nothing about hallucination. |
| 2 | **`ir_video` held-out confusers = 0 FP — but it's in-domain.** ir_video is from the *Drone-detection-dataset* family; v3b's corrective set was built on that same family (svan_IR_BIRD, flir, etc.). | "0 FP" = memorized its own training confusers, **not** robustness. |
| 3 | **We never built a clean held-out IR confuser benchmark.** RGB had `rgb_confusers_merged` (21k held-out). IR had no analog — we leaned on CBAM. | The whole verifier-necessity question was judged without the right instrument. |
| 4 | **CBAM was contaminated.** We put CBAM-*train* into the MLP corpus, then eval'd on CBAM-*valid*. | The verifier's only "confuser win" (48→4) is **in-domain** — can't claim it generalizes to novel confusers. |
| 5 | **Verifier judged mostly on drone-rich surfaces.** ir_dset/ir_video have almost no FP, so *any* verifier there can only cost recall. | We measured the verifier's **cost** thoroughly and its **value** barely. |
| 6 | **MRI's headline verdict is in-pool CV (optimistic)** and doesn't surface the holdout. | The v3b report says "classifier strongly recommended" — contradicting deployment reality. Contributing, not root, cause. |

**Root cause (our approach):** we ported the RGB V5 recipe to IR **without first building the IR
confuser benchmark**. On RGB, `rgb_confusers_merged` made the hallucination visible and the
verifier's value measurable. On IR we skipped that step, so every downstream verdict rests on
confuser-poor or contaminated data.

## 3. The good news — the instrument is buildable (clean thermal confusers exist, unused)

Disk audit of real-thermal (grayscale), non-drone datasets **not** in the V5-IR confuser pool:

| Dataset | mod | n | content | clean for…? |
|---|---|---|---|---|
| `Infrared_bird_drone_airplane_CBAM` (valid) | gray | 180 | **bird/plane** (aerial!) | detector ✔ (v3b never trained on it); MLP ✖ (train split mined) |
| `roboflow_infrared_sea_ships_dataset.ir` | gray | 8,398 | ships (hot blobs) | detector + MLP ✔ (verify vs corrective `sea_`) |
| `road_dog_person_truck_Thermal` | gray | 6,360 | person/vehicle | detector + MLP ✔ |
| `road_thermals…FLIR` | gray | 264 | road/people | detector ✖ (corrective used `flir_video-`) |
| `overhead_aerial_thermal…` | gray | 2,866 | person (overhead) | detector ✖? (corrective `ovh`) — verify |
| `ir-small-target` | gray | 192 | small targets | verify content |
| `Helicopter-kaggle` | **color** | 1,750 | helicopters | ✖ wrong modality (RGB) |

So a **clean held-out IR confuser benchmark** = CBAM-valid (aerial) + sea_ships + thermal_road2
(ground hot-blobs). ~15k held-out real-thermal confuser images we never ran v3b against.

> Caveat: ships/people/vehicles are *ground* confusers, less deployment-relevant for sky
> surveillance than birds/planes. CBAM carries the aerial signal; the others test general
> "fires on any hot blob" robustness. Verify none overlap the corrective `sea_/flir/ovh/cst/dv`
> prefixes before trusting them as held-out for the **detector**.

## 4. Is it MRI? Our approach? — direct answers

- **Not MRI's fault** (it's a fine tool); its in-pool verdict optimism is known and is exactly
  why we built `--holdout-eval`. But the report shouldn't print a verdict that contradicts the
  holdout — fix (A) from the report discussion still stands.
- **It's our approach + data:** no clean IR confuser benchmark + CBAM contamination + judging the
  verifier on confuser-poor surfaces. Fixable.
- **The detector is not perfect** — CBAM P=0.55 proves it hallucinates; we just hadn't shown it
  hard confusers.

## 5. The Plan

### Phase 0 — Build the missing instrument (clean held-out IR confuser benchmark)
- Confirm modality + **provenance** (no overlap with v3b corrective `dataset_v3` prefixes) for
  sea_ships, thermal_road2, ir-small-target. Keep CBAM-valid for the *detector* test only.
- Register as `eval_configs` (e.g. `ir_confuser_bench_640`). This is the deliverable that makes
  everything else trustworthy.

### Phase 1 — Re-measure the DETECTOR's true hallucination (re-base "is it good?")
- `py -m mri --yolo v3b --pos <ir drone surfaces> --neg <clean confuser bench> --conf 0.40`
  → MRI's *raw hallucination rate* + brain viz on a surface that can actually reveal it.
- Hypothesis: significant halluc on OOD thermal confusers (CBAM already says P=0.55). If so,
  the "precise IR detector" claim is retired and replaced with "precise on familiar domains,
  hallucinates on novel thermal confusers."

### Phase 2 — Measure the VERIFIER's value as GENERALIZATION (not in-domain)
- `mri --holdout-eval mlp_v5_ir.pt` on sea_ships + thermal_road2 (**definitely OOD** from the MLP
  training, which was aerial/background). Does it reject novel thermal confusers, or only the
  CBAM-type it trained on? This is the real deployment question CBAM couldn't answer.

### Phase 3 — The untried recall fix (the one live lever)
- drop-`conf` is **dead** (yolo-only CV 0.986 ≈ fused 0.987 — conf isn't load-bearing).
- **Drone-diversity re-mine:** add `corrective_finetune/dataset_v3`'s ~30k thermal drone frames to
  the positive class, retrain **fused 517-D** MLP (deployment-compatible), holdout-eval. Does the
  ~5% recall loss on ir_dset/ir_video shrink to ~0 while confuser-catch holds? If yes → the
  verifier becomes shippable for confuser-rich IR.

### Phase 4 — Decide: verifier-side vs detector-side improvement
- If Phase 2/3 show a recall-safe, *generalizing* verifier → ship it for confuser-dense IR.
- If not → the honest fix is **detector-side**: a `v3c` corrective finetune adding the clean
  thermal confusers (sea_ships / thermal_road2 / CBAM-train) as hard negatives. This improves the
  model the user actually cares about, and is the symmetric story to RGB's retrained_v2 attempt.
- Either way the thermal **"YOLO brain"** viz (Phase 1) is the thesis figure.

---

## 6. Immediate next step
Phase 0 + Phase 1 in one MRI run: point v3b at drones (pos) + the clean confuser bench (neg) and
read the **raw hallucination rate**. That single number settles "is the detector that good?" with
a clean instrument — and it's the same command that produces the brain viz.

## Delivered
- `docs/analysis/2026-05-31_ir_eval_audit_and_plan.md` (this file)
