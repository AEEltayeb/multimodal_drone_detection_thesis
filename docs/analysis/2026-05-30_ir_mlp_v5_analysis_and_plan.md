# IR MLP V5 — Run Analysis, Red Flags, and Next-Step Plan (2026-05-30)

> Reading of the V5-IR distillation run (`eval/results/_v5_ir_p3p5_v3b`), the PCA/LDA
> figures, and a decision plan. Detector under test: IR `finetune_v3b`. Goal stated by
> the user: **eliminate residual IR false positives while preserving P and R.**

---

## 1. What just ran

`py eval/distill_v5_p3p5_ir.py --phase 1` → `visualize_v5_features_ir.py` → `--phase 2`.

- Detector: `runs/corrective_finetune/finetune_v3b/weights/best.pt`
- Mining conf threshold: **0.10** (`distill_v5_p3p5_ir.py:66` overrides `CONF_THR`)
- Pool: **27,931 samples = 20,300 drones / 7,631 confusers**, 517-D (5 meta + 256 p3 + 256 p5)
- CV F1 (5-fold, sample-weighted): **0.9871–0.9873** (two runs)
- LDA train-set accuracy on fused features: **0.9533**
- Artifact: `eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt`

---

## 2. Did we use the correct datasets?

**Drone side — yes, comprehensive.** Svanström IR (4000), Anti-UAV IR (4000), IR_dset_final
train/val (6000+1500), IR_video drone clips (4000+800). Good domain coverage.

**Confuser side — right intent, starved yield.** The intent (airplane/bird IR + IR video
confuser clips) is correct, but `v3b` barely hallucinates on them, so the targets were
massively undershot:

| Source | confusers mined | target | note |
|---|---:|---:|---|
| svanstrom_ir | 4,965 | 5,000 | thermal background / clutter FPs |
| ir_dset_train | 1,630 | 4,000 | background |
| antiuav_val_ir | 263 | 1,000 | background |
| airplane_ir | 553 | 2,000 | **aerial confuser** |
| bird_ir | 155 | 1,500 | **aerial confuser** |
| ir_video_train_conf | **29** | 3,500 | **aerial confuser — collapsed** |
| ir_video_val_conf | **36** | 500 | **aerial confuser — collapsed** |

**Composition of the confuser class:** ~**90% thermal background/clutter** (6,858) vs
~**10% "aerial confusers"** (773 = airplane+bird+video). **No helicopter IR source at all.**

### 2.1 Modality audit (2026-05-30) — the aerial confusers are the WRONG MODALITY

Pixel check of the "dedicated IR confuser" sources actually used:

| Source used | path tag | sampled pixels | reality |
|---|---|---|---|
| `airplane_ir` (553) | `…-ir-grayscale` | 6/6 **GRAY** | grayscale-converted RGB, not thermal |
| `bird_ir` (155) | `bird.v1i…` (no "ir") | 6/6 **COLOR** | **full RGB birds fed to a thermal detector** |
| `ir_video_*_conf` (65) | `IR_*` | thermal | real thermal, but ~0 yield |

So of the 773 "aerial confusers", ~708 are **non-thermal** (grayscale/color RGB) and OOD from
`v3b`; only ~65 are real thermal and the detector barely fired on them. The IR MLP's
aerial-confuser knowledge is effectively built on visible-light images.

**A real thermal aerial-confuser dataset exists and was NOT used:**
`G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net…` (classes `B`/`D`/`P` = bird/drone/plane,
confirmed grayscale-thermal, has train/valid/test). This is the dataset the V5-IR mining should
have drawn aerial confusers from.

So the MLP's "confuser" knowledge is dominated by *drone-vs-thermal-background* (easy) plus a
*wrong-modality* aerial slice — not *thermal-drone-vs-thermal-bird/plane*, which is the job.

---

## 3. Red flags

1. **Low headroom — the strategic flag.** Per EVIDENCE_LEDGER, `v3b` is already high-precision:
   Svanström P=0.950 R=0.973 F1=0.961 (only 117 FP); Anti-UAV P=0.987 R=0.945 (213 FP);
   IR_dset_final test P=0.972 R=0.977 F1=0.967. Contrast RGB FT4 on Svanström: **P=0.443,
   1,499 FP** — that's why the RGB MLP was worth it. The IR detector has almost nothing to
   suppress and a lot of recall to risk. The risk/reward is the inverse of RGB.
2. **Never evaluated out-of-sample.** The only number is in-sample CV F1=0.987.
   `eval/eval_v4_vs_patch.py` has **no `--modality ir`** path (the "next step" command in the
   console output does not yet exist). No bare-vs-patch-vs-MLP comparison on held-out IR data.
3. **Confuser starvation/skew** (§2): aerial-confuser rejection is undertrained; the clean
   metrics mostly reflect the easy background split.
4. **conf train/deploy mismatch.** Mined at conf=**0.10**; the GUI runs IR at **0.40**
   (real) / 0.05 (gray). `conf` is the single most discriminative feature — a shifted conf
   distribution at deploy can move the operating point (same family of bug as the RGB imgsz issue).
5. **imgsz train==deploy** must be verified (the fresh RGB lesson — confusers mined at one
   imgsz, deployed at another, broke veto behavior).
6. **Class imbalance** 73/27, and the minority (confuser) class is itself imbalanced toward
   background.

---

## 4. What the plots say — and is IR more promising than RGB?

| | IR (`v3b`) | RGB (`FT4`) |
|---|---|---|
| LDA train accuracy (fused) | **0.9533** | 0.9492 |
| LDA confuser-cluster center | ~−4 (further) | ~−2.5 |
| Top ANOVA F | 14,160 | ~15,000 (conf) |
| PCA fused | confusers embed in left edge of drone cloud; overlap | same pattern |

On paper IR looks **marginally cleaner** (0.9533 vs 0.9492, confuser peak further from drones).
**But this is deceptive, not "more promising":**

- IR's clean separation is largely **drone vs thermal background** (90% of its confusers),
  which is easy. RGB's 0.9492 was earned on a genuinely harder, aerial-confuser-rich set.
- The PCA overlap is the same story as RGB: unsupervised projection mixes the classes →
  a nonlinear classifier is needed; that part is legitimate for both.

**Verdict:** cleaner-looking LDA ≠ more useful. For *deployment value* IR is **less**
promising than RGB, because (a) the detector has little FP headroom and (b) the aerial-confuser
class — the only place an IR verifier would earn its keep — is starved.

---

## 5. Plan

The whole question is gated on one missing measurement. **Eval first, wire later.**

### Phase 0 — Build the IR head-to-head eval (required; ~1 file)
Add `--modality ir` to `eval/eval_v4_vs_patch.py`:
- Detector → `finetune_v3b`; hook already works (it produced 517-D IR features).
- IR dataset registry: `svanstrom_ir` (IoP@0.5, **imgsz=1280**), `antiuav_ir` (IoU@0.5, 640),
  `ir_dset_final` test (IoU@0.5, 640), `ir_video` test (IoU@0.5, 640) — all held out from mining.
- Verifier branches: `bare_v3b`, `ir_patch_v2` (`confuser_filter4_ir_v2_backup.pt`),
  `mlp_ir_thr_{0.25,0.5,0.7}`.
- **Run at the deploy conf (0.40), not the mining conf (0.10)** — and additionally at 0.10
  to quantify the conf-mismatch effect.

### Phase 1 — Run it; apply the decision gate
Because headroom is tiny, the gate is **recall-first**:
- **Ship only if:** recall delta ≈ 0 on the saturated high-recall surfaces (Anti-UAV, IR_dset)
  **and** FP measurably drops on Svanström/IR_dset. 
- **Keep bare/patch if:** the MLP drops any meaningful recall to buy back a handful of FPs.
  Given P≈0.95–0.99 already, this is the likely outcome — be ready to conclude "IR MLP is a
  marginal precision-polish, not a rescue."

### Phase 2 — Confuser-starvation fix (only if we want real aerial rejection)
The detector won't emit aerial-confuser FPs at conf 0.10. Options, in order:
- (a) **Accept scope = background-FP polish** (cheapest; matches the data we have).
- (b) Mine aerial hard-negs harder: lower conf further on airplane/bird/video-confuser sources
  only, and/or add dedicated IR aerial-confuser image sets; add **helicopter IR** if any exists.
- (c) Drop the idea for IR and keep the IR patch verifier (or nothing).

### Phase 3 — GUI wiring (only if Phase 1 passes)
Mirror the RGB wiring on the IR branch: register a **second** `DetectInputHook` on
`fe.ir_model`, score IR (real-thermal / grayscale) detections, filter per-frame. Separate
hook instance per model (no threading clash — RGB and IR hooks are independent objects).
Add `use_mlp_verifier_ir` / `mlp_verifier_ir_weights` settings parallel to the RGB ones.

### Phase 4 — Documentation
Fill `docs/analysis/mlp_v5_report_ir.md` §4–6 with real yields + head-to-head numbers; add the
IR head-to-head row to `docs/EVIDENCE_LEDGER.md` with reproduction command.

---

## 5b. Is the MLP even necessary? (evidence so far)

Pointing toward **NO** for IR:
- `v3b` standalone precision: Svan P=0.95 (117 FP), Anti-UAV 0.987 (213 FP), IR_dset 0.972.
- Real **thermal** aerial confusers: `IR_video` confuser frames yielded ~65 FP from ~6,000
  frames **at conf=0.10** (~0.01 FP/frame) — the detector essentially does not fire on thermal
  birds/planes/helis.
- The only big IR-confuser FP counts in the repo (`shootout/ir_on_confuser_gray` 656 FP,
  `_rgb` 234 FP) are the **cross-modal** grayscale/RGB fallback path, not thermal deployment.

Missing measurement: `v3b` has never been scored on the **CBAM thermal** confuser test split.
That single run decides necessity.

## 5c. Does the PCA overlap mean IR has the same problem as RGB?

**Same feature entanglement, NOT the same deployment problem.** PCA overlap means: *if* the
detector fires on a confuser, the drone/confuser features are not linearly separable → a
trained classifier is needed. That's true for both modalities. But RGB's actual problem was a
**high FP rate** (FT4 fires constantly, P=0.44, 1,499 FP) — there's a lot to clean. IR's FP
rate is low (P≈0.95–0.99) — there's little to clean. Also, the IR PCA overlap is partly an
artifact of the contaminated confuser set (background + wrong-modality aerial), so it overstates
the entanglement. Same picture, very different stakes.

## 6. Recommendation (revised after the modality audit)

1. **Decide necessity first (one run):** score `v3b` on the **CBAM thermal** confuser test
   split + thermal `IR_video` confusers at the deploy conf (0.40). If thermal-confuser FP is
   negligible (expected), **the IR MLP is not needed** — keep the IR patch verifier or nothing.
2. **The "YOLO brain" showcase is worth doing regardless** — but **re-mine the V5-IR features
   with real thermal confusers** (CBAM + thermal video + thermal background) and drop the
   grayscale-airplane and **RGB-bird** sources before regenerating the PCA/LDA/neuron figures.
   The current figures are scientifically fine as "the IR backbone separates drone vs
   background," but a thesis reviewer who checks the sources will find RGB birds in an "IR"
   confuser pool. Clean thermal mining makes the brain story defensible.
3. **Only build the IR head-to-head eval + GUI wiring if step 1 shows real thermal FP headroom**
   AND the re-mined MLP removes it at ~0 recall cost.

Bottom line: the IR MLP is probably **not necessary** (the IR detector doesn't have RGB's
hallucination disease), and the current IR training data is partly wrong-modality. The
high-value deliverable is the **thermal "YOLO brain" visualization**, re-mined on real thermal
confusers — ship the verifier only if a thermal-confuser FP measurement says there's something
to fix.

---

## Delivered

- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\docs\analysis\2026-05-30_ir_mlp_v5_analysis_and_plan.md` (this file)

### Inputs read
- `eval/results/_v5_ir_p3p5_v3b/training_meta.json`, `check.txt`
- `docs/analysis/images/v5_ir_lda_fused.png`, `v5_ir_pca_fused.png`, `v5_lda_fused.png`
- `docs/analysis/mlp_v5_report_ir.md`, `docs/EVIDENCE_LEDGER.md` (IR rows)
- `eval/distill_v5_p3p5_ir.py`, `eval/eval_v4_vs_patch.py`
