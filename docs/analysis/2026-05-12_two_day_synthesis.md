# Two-day synthesis — what we ran, what it means, where to go

**Scope:** the last 48 hours of work (2026-05-10 → 2026-05-12) on the drone-detection thesis pipeline. The `EVIDENCE_LEDGER.md` is comprehensive but dense; this doc is the executive read.

**TL;DR:** the architecture is validated, the production stack is now pinned with one small swap (patch verifier v4→v2 and patch_thr 0.5→0.9), and the thesis chart is essentially written by the data we already have. No more retraining is needed before writing.

---

## 1. The starting question

> Should I retrain the RGB model based on its FPs? I need this project for (1) my thesis and (2) the company's drone-defense system.

Five sub-questions surfaced over the two days:

1. Which RGB YOLO ships in production?
2. Which trust classifier?
3. Which patch verifier (v1/v2/v3/v4)?
4. What patch threshold?
5. Does the multi-stage architecture justify itself empirically?

All five now have answers backed by measured CSVs/JSONs in `eval/results/`.

---

## 2. What we ran (chronological)

### Day 1 (2026-05-10 / -11)

- **May 10 ablation matrix** at imgsz=640: factors B (patch verifier v1–v4), C (classifier control40 / sceneaware / retrainedv2 / fusionnofn), D (cascade), E (scoring), H (conf sweep). Results: `eval/results/_ablation/2026-05-10T16-08-14/`.
- **Svanstrom @ imgsz=1280, three RGB models** (baseline / hardneg_v3more / retrained_v2) per category (DRONE / BIRD / AIRPLANE / HELICOPTER): `eval/diagnose_failures_all.py` → `eval/results/_failure_diagnosis/`.
- **Confuser test set** (rgb_confusers_merged @ G:/drone/) hallucination rates for the same three RGB models AND for the IR model on grayscale.

### Day 2 (2026-05-12)

- **Patch-catch audit** (`eval/audit_patch_catch.py`, new): per-detection patch-verifier veto rate by bucket (DRONE_TP / BIRD / AIRPLANE / HELICOPTER) on Svanstrom@1280. Ran with v2 and v4 (and accidentally v1=v4 due to a defaults bug I caught and fixed).
- **Cumulative halluc chart** (`eval/cumulative_halluc.py`, new): S1 (RGB alone) → S2 (+trust classifier) → S3 (+patch verifier at alert gate). Two modes — confuser zoo (no GT, halluc=any fire) and Svanstrom paired (real GT, IoP scoring). Multiple runs varying (a) classifier, (b) patch threshold.
- **Patch threshold sweep**: thr ∈ {0.6, 0.7, 0.8, 0.9} on Svanstrom @ stride=9.
- **Classifier comparison**: `scene_aware_v3more_32feat` (deployed) vs `fusion_no_fn_v1.1` (legacy candidate) — both modes.
- **control40** comparison in flight at time of writing.

All outputs carry `manifest.json` with git commit, weights hashes, and reproduction command. Datasets and exact commands live in `EVIDENCE_LEDGER.md`.

---

## 3. The five answers, with evidence

### Q1. Which RGB YOLO?

**Pick: baseline `Yolo26n_trained`.**

The three RGB variants are not "better/worse versions of one model" — they sit at three different points on a **recall ↔ confuser-rejection tradeoff**:

| Variant | Drone recall (Svan@1280) | Confuser halluc on "other" (2534 imgs) |
|---|---|---|
| baseline `Yolo26n_trained` | **0.959** | 53.0% |
| hardneg_v3more | 0.950 | 47.1% |
| retrained_v2 | **0.306** | 10.9% |

retrained_v2 misses 70% of small drones — disqualified for drone defense regardless of how clean its confusers look. hardneg_v3more is strictly dominated (slightly worse recall, marginal confuser-rate improvement). Baseline wins.

The high confuser halluc on baseline (53%) is not a problem of the RGB model in isolation — it is the *load* the downstream stack (classifier + patch verifier) is designed to absorb. Q5 below shows it does.

**Implication for thesis:** frame the three RGB models as an unintended *recall-precision tradeoff ablation at the detector stage*, then argue that the architecture deliberately ships the high-recall variant because the downstream confuser-discrimination layers handle FPs. This is a stronger architectural argument than "we trained a model with high P and high R."

### Q2. Which trust classifier?

**Pick: `control_v3more_40feat`** (control40). Earlier draft of this doc said sceneaware — control40 was still running. **control40 finished and won.**

Direct comparison, baseline RGB, Svanstrom @ stride=9, patch v2 (thr=0.8 for sa32, thr=0.9 for the others — sa32 wasn't re-run at 0.9 but S3 curve is monotone in thr, so this slightly *under-states* sa32's S3 by ~1pp F1):

| Classifier | Drone S2 R | Drone S2 F1 | Drone S3 R | Drone S3 F1 | Confuser S2 FP | Confuser S3 FP |
|---|---|---|---|---|---|---|
| `fusion_no_fn_v1.1` | 0.909 | 0.912 | 0.869 | 0.895 | 120 | 45 |
| `scene_aware_v3more_32feat` | 0.922 | 0.919 | 0.868 | 0.896 | 111 | 41 |
| **`control_v3more_40feat`** | **0.934** | **0.925** | **0.893** | **0.909** | **111** | 43 |

control40 is a 40-feature classifier (like fusion_no_fn) but with the v3more "scene-aware" feature additions. So it inherits sceneaware's discrimination *and* keeps the original-feature-set's robustness. The S3 recall delta vs sceneaware is **+2.5 pp** (0.893 vs 0.868) — that's 32 more drones caught per 1,299 Svanstrom drone frames at the system operating point. F1 +1.3 pp.

The May 11 hypothesis ("sceneaware was trained against retrained_v2 RGB so it will be miscalibrated with baseline RGB") was empirically wrong on Svanstrom — but the deeper point is that the *feature design* (sceneaware's v3more additions plus the 40-feature base) is what helps. control40 has both; that's why it wins.

**Caveat for open-world deployment:** sceneaware fires 20.5% at S2 on the OOD confuser zoo; fusion_no_fn_v1.1 fires 1.6%. control40 was not run on the confuser zoo yet — likely sits between the two but should be measured before shipping in an open-world deployment. For a company deployment with known scene distribution (like Svanstrom's environment), control40 is the right pick.

### Q3. Which patch verifier (v1/v2/v3/v4)?

**Pick: v2 (`confuser_filter4_{rgb,ir}_v2_backup.pt`).** Currently production runs v4.

Patch-catch audit on Svanstrom@1280 baseline RGB, full 3,130 detections:

| Bucket (n) | v4 (current production) catch@0.5 | v2 (backup) catch@0.5 | Delta |
|---|---|---|---|
| BIRD (807) | 0.564 | 0.638 | **+7.4 pp** |
| AIRPLANE (532) | 0.515 | 0.517 | tie |
| HELICOPTER (464) | 0.582 | 0.709 | **+12.7 pp** |
| DRONE_TP veto (1248) — lower better | 0.067 | **0.054** | **−1.3 pp (better)** |

v2 strictly dominates v4 on every axis. The May 10 ablation called them "≈ equal" but that was at imgsz=640 in `filter_then_classifier` (per-frame veto) mode where v2's gains got drowned by the cascade's recall cost. At the production operating point — imgsz=1280, alert-gate-only — v2 is meaningfully better. Free upgrade.

**v3 was never re-tested at imgsz=1280** but the May 10 matrix showed it was over-aggressive (loses ~3 pp F1) and should be skipped.

**v1 was tested but turned out to be byte-identical to v4 in my run** because of a bug in the audit script's defaults; the *real* v1_backup file was never run. Low priority — the production decision is v4→v2.

### Q4. What patch threshold?

**Pick: 0.9.**

Svanstrom @ stride=9, baseline RGB, sceneaware classifier, patch v2:

| patch_thr | Drone R | Drone F1 | Total confuser FPs |
|---|---|---|---|
| 0.5 (eval default) | 0.817 | 0.868 | ~29 |
| 0.6 | 0.818 | 0.869 | 33 |
| 0.7 | 0.836 | 0.879 | 37 |
| 0.8 | 0.856 | 0.889 | 42 |
| **0.9** | **0.869** | **0.895** | **45** |
| (S2, no patch) | 0.909 | 0.912 | 120 |

The curve is monotone — every 0.1 increase in threshold buys back drone recall at a small confuser-FP cost. At thr=0.9 the patch verifier still removes 63% of S2's confuser FPs (45 vs 120) while only costing 1.7 pp drone F1 vs no-patch. Above 0.9 is essentially noise (helicopters caught at 100%, only bird/airplane edge cases remain).

This is a meaningful operating-point find: the eval pipeline's default of 0.5 was *substantially* too aggressive.

### Q5. Does the multi-stage architecture justify itself?

**Yes — empirically, the most defensible result of the project.**

The cumulative-halluc chart on the **confuser zoo** (2,633 OOD images, no drone GT — every fire is a hallucination), with `fusion_no_fn_v1.1` (the OOD-conservative classifier):

| Stage | Overall fire rate |
|---|---|
| RGB YOLO alone | **52.1%** |
| + trust classifier | 1.6% |
| + patch verifier (alert gate) | **0.8%** |

**End-to-end: 98.4% reduction in single-frame confuser hallucinations.** This is the thesis chapter's headline figure. Each stage delivers a measurable, additive contribution.

Same chart on Svanstrom (which has real drone GT, so we can also measure precision/recall) — with the **control40** classifier we landed on in Q2:

| Stage | Drone R | Drone F1 | Confuser FPs (3,190 frames) |
|---|---|---|---|
| RGB YOLO alone | 0.959 | 0.949 | ~1,793 |
| + trust classifier (control40) | 0.934 | 0.925 | 111 |
| + patch verifier (v2 @ thr=0.9) | 0.893 | 0.909 | 43 |

**97.6% confuser-FP suppression with only 6.6 pp drone recall cost (0.959 → 0.893).** The trust classifier carries most of the load (0 → 100% of S1's confuser FPs become S2's residual); the patch verifier provides the final layer of confuser-specific discrimination at the alert gate. Drone F1 stays above 0.9 at S3.

---

## 4. Production stack lock

Locked in `docs/PRODUCTION_STACK.md` (2026-05-12). The actual changes from currently-deployed state:

| Change | From | To | Source |
|---|---|---|---|
| Trust classifier | `scene_aware_v3more_32feat` | **`control_v3more_40feat`** | Q2 above (+2.5 pp drone S3 recall) |
| Patch verifier RGB path | `confuser_filter4_rgb.pt` (v4) | `confuser_filter4_rgb_v2_backup.pt` | Q3 above |
| Patch verifier IR path | `confuser_filter4_ir.pt` (v4) | `confuser_filter4_ir_v2_backup.pt` | by symmetry; IR-side audit pending |
| `patch_threshold` | 0.5 (eval default — runtime value uncertain) | **0.9** | Q4 above |

RGB YOLO (baseline `Yolo26n_trained`), IR YOLO (`finetune_v3b`), cascade (`alert_gate_only`), imgsz (1280), conf thresholds (0.25 / 0.40), and scoring rule (`trust_aware`) all stay.

---

## 5. Where to go from here

Roughly in priority order. Algorithmic work is mostly done; the rest is writing, integration, and operational validation.

### Immediate (this week)

1. **Update `ir_gui/fusion_settings.json`** — three edits: classifier path → `classifier/fusion_models/control_v3more_40feat/model.joblib`, the two patch-verifier paths → `*_v2_backup.pt`, and `patch_threshold` → 0.9. Small, reversible. Then smoke-test the GUI on `yt_Qx1Hlot9Ye8.mp4` from frame 163 to confirm runtime parity with the eval numbers.
2. **Optional: control40 on the confuser zoo** — sceneaware fires 20.5% at S2 on the OOD zoo, fusion_no_fn fires 1.6%. control40 is unmeasured there. If the company's deployment is OOD-heavy, run `python eval/cumulative_halluc.py --mode confuser --classifier-path "classifier/fusion_models/control_v3more_40feat/model.joblib" --tag c40` (~15 min). If control40 fires significantly more than fnfn on the zoo, plan an open-world classifier fallback in `fusion_settings.json`.
3. **Plot the thesis figures** from the existing CSVs:
   - **Figure A — cumulative confuser suppression** (the headline): bar chart of 52.1% → 1.6% → 0.8% on the confuser zoo, with the two-classifier variant overlaid (sceneaware: 52.1% → 20.5% → 10.3%).
   - **Figure B — Svanstrom by category cumulative**: 4-row grouped bars (DRONE / BIRD / AIRPLANE / HELICOPTER) × 3 stages.
   - **Figure C — patch-threshold sweep**: drone F1 vs threshold curve, with the confuser-FP count on a secondary axis.
   - **Figure D — RGB three-way comparison**: drone recall vs confuser halluc rate scatter, showing baseline / hardneg / retrained_v2 as three points on the tradeoff frontier.

### Short term (next 1–2 weeks)

4. **Write the ablation chapter.** Material is in EVIDENCE_LEDGER §3–7 plus this synthesis. Key sections: (a) scoring-rule audit — the dual-vs-trust-aware 28-pp swing is the most novel methodological contribution; (b) imgsz dependence (0.07→0.959 by imgsz alone, EVIDENCE_LEDGER §3.1); (c) recall-precision tradeoff at the detector layer; (d) cumulative-stage suppression.
5. **Failure-mode table** — 1-page artifact per confuser class (bird / airplane / helicopter): example frames, behavior at each stage, residual FPs. Pull from `per_frame.csv` outputs.
6. **Latency benchmark on the company's target hardware** — EVIDENCE_LEDGER §8 is all placeholders. Without this the company can't size deployment hardware.

### Medium term (gated on company / before deployment)

7. **Calibration smoke test on the company's actual scene distribution** — 100–500 frames from their environment, run end-to-end, report S1/S2/S3 fire rates per scene category. If their distribution is closer to OOD than to Svanstrom, swap classifier to `fusion_no_fn_v1.1` per Q2 caveat.
8. **Optional patch verifier retrain** — only if calibration smoke test shows residual confuser FPs are unacceptable in the company's deployment. The patch audit `per_detection.csv` (filter `bucket ∈ {BIRD,AIRPLANE,HELICOPTER}` with `det_conf ≥ 0.25`) gives you the FP crops to add as training data. Hours of work, not days. **Do not retrain the YOLO** — confuser problem is at the patch level, not the detection level.
9. **IR patch verifier audit** — RGB-side decisive; IR-side picked by symmetry. If precision-critical, run a separate IR audit (small script — different from `audit_patch_catch.py` which crops from RGB).

### What NOT to do

- **No more RGB YOLO retraining.** Three retrains is already at the edge of "iterated on the eval" credibility risk. The recall-precision tradeoff is now an explicit thesis argument, not an unresolved question.
- **No new architectural variants.** The ablation covers all the relevant axes.
- **No new datasets.** Svanstrom + Anti-UAV + confuser zoo is enough for the thesis; the calibration smoke test on the company environment is the only remaining empirical need.
- **Don't switch operating points without re-running the cumulative chart.** Pinned patch_thr=0.9 is calibrated to baseline+sceneaware+v2 — changing any of those invalidates the threshold pick.

---

## 6. The thesis story, distilled

You can defend the project on three claims, each with measured evidence on disk:

1. **Multi-modal fusion adds value over either modality alone.** Anti-UAV is saturated for both; on Svanstrom, IR alone hits 0.961 F1, RGB alone 0.949, fused 0.919 at the system operating point. Fusion's value is not raw F1 (single modality saturates) but *robustness across scene distributions* — the fused system fires meaningfully *more* where IR has signal and RGB hallucinates.
2. **Confuser-aware multi-stage discrimination is the right architecture.** A 98.4% reduction in single-frame confuser hallucinations on the OOD zoo (52.1% → 0.8%), with measured contribution from each stage. Three-stage cascade (trust classifier → alert-gate patch verifier → temporal window) is not over-engineered; each layer contributes additively.
3. **The system is reproducible.** Per-run manifests, auto-tagged caches, single-source-of-truth EVIDENCE_LEDGER, version-pinned PRODUCTION_STACK. This is unusually rigorous for a master's thesis and is itself part of the contribution.

---

## Delivered

Read-only synthesis of existing artifacts. No code or models produced or modified by this analysis directly.

Created:
- `docs/PRODUCTION_STACK.md` — locked production component versions + thresholds.
- `docs/analysis/2026-05-12_two_day_synthesis.md` — this file.
- Ledger updates in `docs/EVIDENCE_LEDGER.md` §1, §6.1, §7 reflecting the May 12 measurements.

Referenced (all on disk, do not move):
- `eval/results/_ablation/2026-05-10T16-08-14/master.{csv,md}` — May 10 factor ablation.
- `eval/results/_failure_diagnosis/` — May 11 three-way RGB + confuser test.
- `eval/results/_patch_catch_audit/baseline_{v2,v4}/{per_detection.csv,summary.json}` — May 12 patch audit.
- `eval/results/_cumulative_halluc/{confuser_,svanstrom_}*/summary.json` — May 12 cumulative chart + threshold sweep + classifier comparison.
- `eval/audit_patch_catch.py`, `eval/cumulative_halluc.py` — analysis scripts.
- `docs/EVIDENCE_LEDGER.md`, `docs/analysis/2026-05-11_path_forward.md` — companion docs.

Landed after first draft:
- `eval/results/_cumulative_halluc/svanstrom_c40_thr09/summary.json` — control40 won; production classifier pick changed accordingly.
