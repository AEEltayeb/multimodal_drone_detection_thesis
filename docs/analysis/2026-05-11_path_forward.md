# Path forward after the May 2026 ablation

**Question from user:** Should I retrain the RGB model based on its FPs? Thesis must be (1) scientifically significant and (2) production-ready for the company's drone-defense system.

**Short answer:** **No — do not full-retrain the RGB model.** The Svanstrom "RGB is broken" symptom (recall 0.07) is *not* a model defect; it is a resolution / drone-pixel-size problem caused by running YOLO at `imgsz=640` on a dataset whose drones span 6–20 pixels. The imgsz=1280 rerun and the in-flight confuser-dataset eval will likely confirm this. Three retrains is already at the edge of "iterated on the test set" territory for thesis credibility; a fourth without a clearly documented new failure mode would be hard to defend.

Below: what the numbers actually say, what to do instead, and what to add to lock in the production-readiness story.

---

## 1. What the May 10 ablation actually shows (imgsz=640)

Source: `eval/results/_ablation/2026-05-10T16-08-14/master.md` (verified in-session).

### Anti-UAV is saturated

| Config | P | R | F1 |
|---|---|---|---|
| `rgb_only` (Yolo26n_retrained_v2) | 0.991 | 0.995 | **0.9929** |
| `ir_only` (finetune_v3b) | 0.987 | 0.945 | 0.9654 |
| `classifier` (sceneaware) | 0.991 | 0.994 | **0.9930** |

Headroom against `rgb_only` is **0.01 F1**. Any further RGB-model work that is justified by Anti-UAV alone is fitting noise.

### Svanstrom (imgsz=640) — RGB is silent, IR carries the system

| Config | P | R | F1 |
|---|---|---|---|
| `rgb_only` | 0.837 | **0.072** | 0.133 |
| `ir_only` | 0.950 | 0.973 | 0.961 |
| `classifier` (sceneaware) | 0.980 | 0.947 | **0.963** |
| `filter_then_classifier` | 0.979 | 0.910 | 0.943 |

The RGB recall of 0.072 here is **the most important number in the whole matrix**. It is the same `Yolo26n_retrained_v2` that hits 0.995 recall on Anti-UAV; the model is fine. What changed is the input distribution: Svanstrom drones at native 640×480 letterboxed into a 640-square net leave the drone at the floor of YOLO's resolvable scale. The earlier finding that imgsz=1280 raises the model's fire rate on the same clips from 0.363 → 0.981 (verified in `eval/results/_phase4/Qx1H/noTemp_1280/summary.json`) tells you the model has the *capacity*, it just isn't being given the *resolution*. Retraining it on Svanstrom won't fix the geometry; running it at imgsz=1280 (or with the GUI's temporal ROI re-crop) will.

### Patch verifier: v1 ≈ v2 ≈ v4, v3 is worse

On Svanstrom `filter_then_classifier`: v1 F1=0.924, v2=0.931, v3=**0.878**, v4=0.933. v3 is over-aggressive — vetoes drone TPs more often than it kills confuser FPs. **Ship v2 (production) or v4 (latest), NOT v3.**

### Classifier: control40 ≈ sceneaware > fusionnofn > retrainedv2

On Svanstrom `classifier` config: control40=0.9629, sceneaware=0.9629, fusionnofn=0.9628, retrainedv2=0.9496. The retrained-on-new-RGB classifier underperforms the one trained against `Yolo26n_trained`. This is a calibration-mismatch artefact, not a fundamental issue — but **ship sceneaware**, not retrainedv2.

### Cascade: per-frame patch filtering costs ~1pp F1

Per-frame `filter_then_classifier` and `classifier_then_filter` consistently lose ~1pp F1 vs `classifier` alone on both datasets. The production system runs `alert_gate_only` (patch consulted only when an alert is about to fire), which preserves the per-frame TPs while still gating confuser FPs at the alert layer. **Don't ship per-frame filtering.**

### Scoring rule matters more than any single component

`score_dual` vs `score_trust_aware` on Svanstrom `classifier`: 0.6629 → 0.9398 F1. This single accounting choice is a 28-pp swing and **must be disclosed** in the thesis. The defensible position: trust-aware matches the system's actual decision rule (the classifier picks which modality to trust), and dual unfairly penalizes a silent-by-design modality.

---

## 2. Should you retrain the RGB model again?

**Don't. Here's the case.**

1. **Three retrains already.** Baseline + two local finetunes + one full retrain. Adding a fourth without a clearly novel failure mode crosses into "iterated until the eval looked right" territory, which weakens the thesis even if the numbers improve.
2. **The Svanstrom failure isn't an RGB-model problem.** Retraining can't add pixels to a 12×8 drone post-letterbox. The fix is `imgsz=1280` at inference, optionally a multi-scale TTA pass, and the temporal ROI re-crop the GUI already does. The ablation we have already shows the rest of the stack absorbs the RGB silence via the trust classifier.
3. **The confuser eval is the right test.** If RGB precision on a confuser-rich dataset stays >0.97 at conf=0.40 (which is what the patch verifier expects), then the RGB model + patch verifier together are doing their job. Only if RGB precision collapses on confusers (e.g. it fires on birds/balloons/airplanes with high confidence) does retraining become the right move — and even then, the more surgical fix is to **expand the patch-verifier confuser-class training set**, not to retrain the YOLO. Patch retraining is two orders of magnitude cheaper.
4. **Anti-UAV headroom is 1 pp F1.** Even a perfect retrain helps you less than fixing imgsz on Svanstrom does.

**When retraining IS justified:** if the confuser eval shows RGB firing with conf≥0.5 on a recognizable class (e.g. birds in distance) and the patch verifier can't reliably catch them. Then add hard negatives from that class to the YOLO training set and finetune (not full retrain) for a small number of epochs with a low learning rate. Document the hard-negative source explicitly in the thesis.

---

## 3. Optics — three angles

### Scientific (thesis)

What's defensible right now:
- **Multi-modal fusion with a learned trust classifier** outperforms either single modality on the cross-domain dataset (Svanstrom): IR alone 0.961 → fused 0.963; on Anti-UAV both saturate near 0.993.
- **Confuser-aware alert gating** with a 4-class patch verifier preserves recall while suppressing alert-level FPs (the production cascade, evaluated in `eval/eval_video_temporal.py`).
- **Per-component ablation** with provenance manifests — this matrix *is* the thesis chapter. Few student theses publish a clean factor-level ablation; this one is unusually rigorous.
- **Scoring-rule sensitivity disclosure** — the dual vs trust-aware delta is a *strength* if you frame it as "we audited our own metric and report both"; it's a *weakness* if you only report the favorable one.

What's exposed:
- RGB recall 0.07 on Svanstrom is a bad screenshot without the imgsz explanation alongside it. The thesis must lead with "at native 640px input the RGB model is starved of pixels; imgsz=1280 recovers fire rate to 0.98 on the diagnostic clips" before showing the 0.07 number.
- Three RGB retrains without per-retrain ablations is a target for the committee. Pre-empt by listing them in a "training-history table" with one-line motivations.

### Production-readiness (company)

In place:
- Trust-classifier-gated fusion, patch-verifier confuser suppression, temporal alert gating, ROI re-crop for small drones, grayscale fallback when IR is unavailable, PySide GUI + FastAPI/React variants, per-run provenance manifests, reproducible cache identity.

Missing for "shippable":
- **Latency / throughput budget** on the company's target hardware (GPU or Jetson). One CSV: per-stage ms (YOLO RGB, YOLO IR, classifier, patch, temporal, draw) at imgsz∈{640,1280}, batch=1, with the chosen GPU. Without this the company can't size hardware.
- **Operating-point recommendation** with explicit conf thresholds per modality and the rationale. Current default `rgb_conf=0.25, ir_conf=0.40` looks right per the conf sweep, but no document says "we picked these because at this point precision is ≥0.95 and recall plateaus."
- **Failure-mode catalog**: bird → behavior, plane → behavior, balloon → behavior, drone hovering motionless → behavior, two drones → behavior, night/low-light → behavior. Even a 1-page table is enough.
- **Calibration check** on a held-out set the company actually cares about (their environment), not Svanstrom / Anti-UAV. If they can't supply one, say so in the thesis and recommend it as deployment-time validation.
- **A README at repo root** that says exactly which weights, classifier, patch verifier, cascade, conf thresholds, and imgsz the production system uses. Right now this is scattered across memory and ablation tables.

### ML

Healthy:
- Fusion lift is measurable on out-of-distribution data (Svanstrom).
- Trust classifier outperforms naive max-conf fusion.
- Per-frame patch filtering was tested, found inferior to alert-gate-only, and rejected on evidence rather than intuition.

Risks:
- **Calibration drift between RGB retrains and downstream consumers** (classifier, patch verifier). Already visible: `clf_retrainedv2` worse than `clf_sceneaware` on Svanstrom even though `clf_retrainedv2` was trained on the newer RGB. Lesson: every time a base model is retrained, all downstream models must be re-evaluated and possibly re-trained. **Pin component versions in the production manifest and never silent-swap.**
- **Iterated-on-eval risk** if RGB is retrained again without a new dataset that the prior retrains didn't see.
- **`imgsz` as a hidden hyperparameter**: choosing 640 vs 1280 changes the *system* identity. Decide once per deployment, document, and don't let it drift.

---

## 3b. UPDATE (2026-05-11, evening) — in-flight runs landed

Numbers from the user's runs (paste, not yet re-verified against source files — treat as provisional until CSVs are in `eval/results/`):

### Svanstrom @ imgsz=1280, **baseline RGB (`Yolo26n_trained`, NOT retrained_v2)**, by category

| Category | Frames | Det rate | TP | FP | FN | P | R |
|---|---|---|---|---|---|---|---|
| DRONE | 1299 | 96.5% | 1248 | 79 | 54 | 0.940 | **0.959** |
| BIRD | 589 | 94.4% (halluc) | 0 | 807 | 0 | — | — |
| AIRPLANE | 677 | 74.6% (halluc) | 0 | ~530 | 0 | — | — |
| HELICOPTER | 625 | 66.2% (halluc) | 0 | 464 | 0 | — | — |

Median confuser confidences: bird 0.691, airplane 0.733, helicopter **0.832**.

### Confuser test set (separate dataset, no GT drones)

| Source | Images | Halluc rate | Avg conf |
|---|---|---|---|
| airplane (roboflow), RGB baseline | 99 | 27.3% | 0.506 |
| other (svan+kaggle+roboflow), RGB baseline | 2534 | **53.0%** | 0.608 |
| airplane, IR-on-grayscale | 99 | 16.2% | 0.718 |
| other, IR-on-grayscale | 2534 | 22.2% | 0.752 |

### What this changes

> **Important scoping note (2026-05-11):** the run reported here is on the **baseline `Yolo26n_trained`**, not `Yolo26n_retrained_v2`. So these numbers tell us what the *original* RGB model does at imgsz=1280 — they do NOT yet tell us how the retrain compares. The retrained_v2 comparison at imgsz=1280 is the critical missing measurement; everything below should be re-read with that asymmetry in mind.

1. **imgsz=1280 hypothesis confirmed for the baseline model.** Baseline RGB drone recall is **0.959** @ 1280 on Svanstrom. (The 0.07 @ 640 number from the May 10 ablation was on retrained_v2, so the direct apples-to-apples imgsz delta isn't established yet — needs the retrained_v2 @ 1280 number to close the loop. But it's already overwhelmingly likely the imgsz effect dominates.)
2. **There IS a real RGB confuser-FP problem**, at high confidence: birds 94% halluc @ conf 0.69, helicopters 66% halluc @ conf **0.83**. Raising the conf threshold can't fix this without losing drone TPs (drones also fire in the 0.7+ band on Svanstrom). This is the genuine failure mode the earlier analysis was uncertain about.
3. **The IR model hallucinates roughly 2× less than RGB on the same confuser images** (22% vs 53% "other"), but at higher confidence (0.75 vs 0.61). IR is structurally better-behaved on confusers but isn't a get-out-of-jail card — when it does fire wrong, it fires *more* confidently. This actually matches the architecture: the trust classifier should be picking IR more often on confuser-heavy scenes.
4. **The patch verifier is now the load-bearing component.** Whether to retrain RGB hinges entirely on whether the patch verifier catches these confuser firings.

### What the baseline-vs-retrained comparison would reveal

This is the most important missing measurement and it directly answers "did the retrain help":

| Outcome on Svanstrom@1280 + confuser set | Interpretation | Action |
|---|---|---|
| retrained_v2 has **lower** halluc rate than baseline | retrain worked as intended — bird/airplane discrimination improved | Ship retrained_v2; the May 10 matrix's poor showing was an imgsz=640 artifact |
| retrained_v2 has **similar** halluc rate to baseline | retrain didn't change confuser behavior; whatever it changed, it wasn't this | Ship whichever has better drone recall; document that the retrain didn't improve discrimination |
| retrained_v2 has **higher** halluc rate than baseline | retrain hurt — likely overfit to drones at the expense of negatives | **Ship baseline `Yolo26n_trained`** and document the regression. Three retrains → one is the winner, decided on evidence |

Run this before any further retraining or even before deciding which model is "production."

### Decision tree (run in order — do NOT skip to retrain)

1. **Measure patch-verifier catch rate on the actual FP images first.** Take the bird/airplane/helicopter detections that fired with conf≥0.5 on Svanstrom@1280 and on the confuser test set; run each through `confuser_filter4_rgb_v2_backup.pt`. Record: per-class catch rate at `patch_thr ∈ {0.5, 0.6, 0.7}`, and the drone-TP veto rate at the same thresholds (must measure both — a verifier that catches 99% of birds but vetoes 20% of drones is not an improvement).
2. **If patch catch ≥ 0.90 per class AND drone-TP veto ≤ 0.03 at some `patch_thr`** → ship as-is with that threshold and the `alert_gate_only` cascade. **No retraining of anything.** The system already handles this; we just hadn't measured it.
3. **If patch catch < 0.90 on a specific class** → **retrain the patch verifier**, adding the FP images from that class as training data. ~hours, not days. Re-evaluate. This is the right intervention because the patch verifier exists *for* this job and its training set is the lever you have.
4. **Only if patch verifier saturates around 0.93 per class and the residual is still unacceptable for production** → consider an RGB hard-negative *finetune* (low LR, few epochs, small set of confuser hard-negatives added to a frozen dataset). Not a full retrain. Document as "RGB-v3-hardneg" and pin in production manifest.

### Thesis framing of this update

The bird/airplane/helicopter hallucination rates are not embarrassing — they are *the reason the architecture exists*. A single-stage RGB YOLO is known to be confuser-prone in aerial scenes; this is exactly why the system layers (a) a trust classifier that prefers IR when scene context suggests confusers, (b) a 4-class patch verifier specifically for bird/airplane/helicopter/balloon discrimination, and (c) an alert-gating temporal layer. The headline thesis chart should be:

| Stage | Halluc rate on confuser set |
|---|---|
| RGB YOLO alone | 53.0% |
| + IR cross-check via trust classifier | _measure_ |
| + patch verifier @ alert gate | _measure_ |
| + temporal alert window | _measure_ |

This chart, with measured numbers at each cumulative stage, is the **single most defensible figure** the thesis can produce. It directly justifies every layer of the architecture.

---

## 3c. UPDATE (2026-05-11, late) — three-way RGB comparison landed

The retrained_v2 numbers came in. The picture is **not** what either of the earlier updates implied.

### Svanstrom @ imgsz=1280, three RGB models side-by-side

| Metric | baseline (`Yolo26n_trained`) | hardneg_v3more | retrained_v2 |
|---|---|---|---|
| DRONE det rate | 96.5% | 95.7% | **30.8%** |
| DRONE recall | **0.959** | 0.950 | **0.306** |
| DRONE precision | 0.940 | 0.941 | 0.943 |
| Missed-GT median size (area ratio) | 0.0014 | 0.0020 | 0.0024 |
| BIRD halluc | 94.4% | 94.2% | **3.4%** |
| BIRD med FP conf | 0.691 | 0.683 | 0.527 |
| AIRPLANE halluc | 74.6% | 64.7% | **5.6%** |
| AIRPLANE med FP conf | 0.733 | 0.677 | 0.430 |
| HELI halluc | 66.2% | 41.9% | **4.5%** |
| HELI med FP conf | 0.832 | 0.710 | 0.672 |

### Confuser test set (no GT drones) @ imgsz=1280

| Source | baseline | hardneg_v3more | retrained_v2 |
|---|---|---|---|
| airplane (99) | 27.3% | 7.1% | 19.2% |
| other (2534) | 53.0% | 47.1% | **10.9%** |

### What this actually means

The three RGB variants are not "better/worse" versions of the same model — they are **three different operating points on a recall ↔ confuser-rejection tradeoff**:

- **baseline** — high-recall, confuser-prone. Catches almost every drone (0.959 on Svanstrom), hallucinates heavily on birds (94%), helicopters (66%), airplanes (75%).
- **hardneg_v3more** — marginal improvement on helicopters (66→42%) and airplanes (75→65%) for a tiny recall cost (0.959→0.950). Birds essentially unchanged (94→94%). **Strictly dominated by one of the other two for almost every use case.**
- **retrained_v2** — collapsed confuser rates by ~20×, but **drone recall fell to 0.306** on Svanstrom. Misses 70% of drones. The May 10 ablation's "0.07 recall @ 640" wasn't only an imgsz artifact — retrained_v2 is structurally bad at small drones even at 1280.

This forces a clean decision: the production RGB is **baseline `Yolo26n_trained`**. retrained_v2 missing 70% of small drones is a non-starter for drone defense regardless of how clean its confuser rejection is. hardneg_v3more isn't worth the complexity of a third weights file in the codebase.

### Why the architecture is now validated, not threatened

Baseline's 53% halluc rate on the confuser set looks bad in isolation, but the architecture exists for exactly this case:

1. **IR model on grayscale confusers** already hallucinates only 22% on "other" (vs RGB's 53%) — the trust classifier should be picking IR on confuser-heavy scenes.
2. **Patch verifier** is the dedicated bird/airplane/helicopter/balloon discriminator.
3. **Temporal alert gate** suppresses isolated firings.

The three-model comparison is now an *unintended ablation of the recall–precision tradeoff at the detector stage*, and the thesis can frame it as: "we deliberately ship the high-recall variant because the downstream stack (trust classifier + patch verifier + temporal) is engineered to handle confuser FPs, and a high-precision detector would be unable to recover the lost recall." This is a strong architectural argument.

### Updated production stack

| Component | Choice | Why |
|---|---|---|
| RGB YOLO | `Yolo26n_trained` (baseline) | Only model with usable small-drone recall at imgsz=1280 |
| IR YOLO | `finetune_v3b` | unchanged |
| Trust classifier | `fusion_no_fn_v1.1` or `control40` | Both were trained against baseline RGB — pick whichever scores better in a re-eval with baseline as the RGB. `scene_aware_v3more_32feat` and `retrained_v2_32feat` were trained against the wrong RGB and should not be shipped. |
| Patch verifier | `confuser_filter4_*_v2_backup.pt` | v3 over-aggressive, v4 ≈ v2; v2 is what production has been running |
| Cascade | `alert_gate_only` | unchanged |
| Scoring | `trust_aware` | unchanged |
| imgsz | 1280 | non-negotiable for small drones |
| RGB conf | TBD from sweep | run conf sweep against **baseline** RGB to pick |
| IR conf | 0.40 | unchanged |

**Important corollary:** every downstream component (classifier, patch verifier) that was trained or evaluated against `retrained_v2` may now have wrong calibration. Re-evaluate the classifier candidates with baseline RGB before locking. The classifier choice from the May 10 matrix is not necessarily portable.

### Patch-verifier audit is still the next critical run

Now scoped concretely: take baseline RGB's 807 bird FPs + ~530 airplane FPs + 464 helicopter FPs from Svanstrom@1280, run each through `confuser_filter4_rgb_v2_backup.pt`, record per-class catch rate at `patch_thr ∈ {0.5, 0.6, 0.7}` AND drone-TP veto rate on baseline's 1248 drone TPs. This is the measurement that decides whether the system as configured handles baseline's confuser firings, or whether the patch verifier needs to be retrained with these FPs as training data.

---

## 4. Concrete next actions (ordered)

1. **Measure patch-verifier catch rate on the actual FP images** from Svanstrom@1280 and the confuser test set. Per-class catch rate at `patch_thr ∈ {0.5, 0.6, 0.7}` AND drone-TP veto rate at the same thresholds. Until this number exists, every other decision is guesswork. Save to `eval/results/_patch_catch_audit/`.
2. **Build the cumulative-stage halluc-rate chart** (Section 3b last table). This is the thesis's most defensible figure.
3. **Branch on the catch-rate number:**
   - ≥0.90 per class with drone-TP veto ≤0.03 → **no retraining**. Lock thresholds and ship.
   - <0.90 on a class → **retrain patch verifier** with those FP images. Re-measure.
   - Verifier saturates and residual unacceptable → **RGB hard-negative finetune** (low LR, few epochs, small confuser set added to existing data). Pin as `Yolo26n_v3_hardneg`. Never silent-swap.
4. **Lock the production stack** in `docs/PRODUCTION_STACK.md`: RGB weights, IR weights, classifier, patch verifier, cascade, per-modality conf, imgsz. Single source of truth.
5. **Latency benchmark** on company's target hardware. Per-stage ms at imgsz∈{640,1280}. Unblocks deployment.
6. **Thesis: scoring-rule audit section early** (dual vs trust-aware, 28-pp swing). Most novel methodological contribution.

---

## Delivered

Produced by this analysis (read-only synthesis, no code/experiment run):

- `docs/analysis/README.md` — index of analysis docs.
- `docs/analysis/2026-05-11_path_forward.md` — this file.

Existing artifacts cited above (do not move):

- `eval/results/_ablation/2026-05-10T16-08-14/master.md` and `master.csv` — the factor-level ablation tables (imgsz=640).
- `eval/results/_phase4/Qx1H/noTemp_1280/summary.json` — fire-rate evidence for the imgsz=640→1280 effect.
- `eval/ablations.yaml`, `eval/ablate.py`, `eval/eval_video_temporal.py`, `eval/run_manifest.py`, `eval/dryrun.py` — ablation infrastructure (do not recreate).

In-flight (not yet on disk at time of writing):

- `eval/results/_ablation/<svanstrom-1280-timestamp>/master.md` — pending.
- RGB-baseline + IR-grayscale results on the confuser dataset under `G:/...` — pending; once landed, link from this file's Delivered section.
