# Thesis narrative reframe — chapter-by-chapter plan (for approval before execution)

**Date:** 2026-06-06. **Status:** PLAN — no `.tex` edited yet (user chose "plan first"). On approval I
execute the framing rewrites in `docs/thesis_working.tex` only.

## Locked decisions (from this session)
- **Spine = Narrative 1, "Closing the benchmark→deployment gap" (system-first).** Confusers = the marquee
  gap, not the whole story.
- **Through-line = Narrative 2, "statistics-before-training"** — the Model MRI + leakage-aware selection +
  honest evaluation. Connective method, not a numbered contribution.
- **Tightened contributions: 5 → 4** (scoring-rule audit demoted to a methodology/threats note; the
  "novel-to-literature" overclaim dropped). MRI **elevated** into the through-line, calibrated ("a reusable
  diagnostic", not "we invented it").
- **Length: needs to shorten** (159 → target ~125-135). **Keep the 7-chapter structure** (light-touch IMRAD,
  not the 9→5 restructure). **SelCom CCTV kept minimal** (anchor "deployable" on public benchmarks + the
  production stack, not the proprietary partner).

---

## 0. The new spine (the thesis statement)

> A YOLO detector **saturates public drone benchmarks** (baseline P0.94 / R0.96 on Svanström) yet **fails in
> real deployment**. This thesis closes the benchmark→deployment gap by building a **dual-modality drone-
> detection system** and the **statistics-first engineering discipline** that makes it work. The gap has three
> concrete detection causes — **confuser hallucination, small-drone resolution, and thermal-not-always-
> available** — and the system closes each with a learned per-frame modality-trust router and a confuser
> false-positive filter cheap enough to run online, co-developed with its data via a human-in-the-loop loop.

**The four contributions (final):**
1. **A deployable dual-modality drone-detection system** — RGB + thermal-IR detectors, learned per-frame
   modality-trust routing, and a confuser false-positive filter. (was "trust-aware cascade")
2. **A feature-reuse confuser verifier** — the detector's own p3+p5 ROI features already separate
   drone-from-confuser (LDA ~95%), so a lightweight MLP filters false positives at **46-72× the speed** of a
   separate CNN, 1-4% overhead.
3. **An emergent cross-modal transfer** — the thermal-trained detector works zero-shot on grayscale RGB,
   matching the RGB baseline on the hardest bird-cluttered clip while firing 3.2× less on confusers.
4. **A human-in-the-loop IR co-development case study** — six revisions on a fixed test split, the V5
   regression reported (not hidden), and the corpus-level failure mode named.

**Method through-line (woven through all four, not numbered):** statistics-before-training — the **Model MRI**
feature-space instrument, the **leakage-aware feature selection** (robust6/robust8), and **honest evaluation**
(the IoP rule, the Svanström usage audit, and the scoring-rule sensitivity check, presented as a caveat).

---

## 1. Framing rewrites (the load-bearing prose — this is where the narrative lives)

| Section (Ch1/Ch7) | Current | New job |
|---|---|---|
| **Background** (L176) | "The problem is out-of-distribution confuser fire rate… this thesis addresses the second." | Broaden to the **benchmark→deployment gap**: benchmarks are saturated; deployment breaks for three reasons (confusers, resolution, modality). Confusers introduced as the *hardest*, not the *only*. |
| **Problem Statement** (L186) | confuser hard-neg diminishing returns | Keep the "can't fix at the detector stage" argument — but generalise: each gap needs a *system* answer, not a detector retrain. |
| **Research Questions** (L196) | RQ1 confuser-suppression · RQ2 cost · RQ3 scoring · RQ4 cross-modal | **LOCKED: 3 RQs.** RQ1 confuser-gap suppression (keep), RQ2 in-distribution cost + surface-dependence (keep), RQ3 cross-modal grayscale transfer (was RQ4). **Drop the scoring RQ** (→ methodology caveat); **resolution + deployability stay findings, not RQs** (avoid repeating the scoring-audit over-bill). |
| **Contributions** (L210) | 5, incl. scoring audit; #1 "the disclosure is the contribution" | The **4 above**; #1 reframed as *the system* (not "disclosure is the contribution"); scoring audit removed from the list. |
| **Conclusion / RQ answers** (L2083) | mirrors old RQs | Mirror the new RQs; lead with "a deployable system that closes the deployment gap," SelCom minimal. |
| **Ch6 headline §** (L1495 "Cumulative Confuser Suppression") | confuser-first headline | Reframe the results chapter to march the **gaps in order** (confuser → resolution → cross-modal → real-video deployment), so confusers is section 1 of several, not "the headline." |

---

## 2. Chapter-by-chapter plan (7 chapters kept)

| Ch | Current job | New job | Action | Δ length |
|---|---|---|---|---|
| **1 Introduction** | confuser framing | the deployment-gap spine + 4 contributions + system naming | **Rewrite** Background/Problem/RQs/Contributions | ≈ 0 |
| **2 Related Work** | detection · **confuser problem** · thermal · fusion · scoring · cross-modal | same coverage, **rebalanced**: confuser as one deployment gap; add a short "real-time/efficient verification" thread | light edit | −0 |
| **3 Methodology** | datasets · metrics · **scoring audit** · Svan audit · recipes · repro · **MRI** | **Demote scoring audit** to a caveat subsection (drop "novel to literature"); **elevate MRI** as the named statistics-first instrument (the through-line's home) | reframe 2 §§ | −5 |
| **4 System Architecture** | overview · fail-open · trust-fusion · cascade · temporal · resolution · RGB · IR · **trust clf (187)** · **patch verifier (240)** | **the system chapter** (good for Narrative 1). **Trim hard:** patch verifier is *superseded by mlp_v5* — compress 240→~70, move version-history + threshold-sweep tables to **app**. Move classifier-zoo comparison tables to **app**, keep the feature-selection story (it's the method). Elevate `resolution` (L767, 14 lines) — it's now RQ2. | **big trim + 1 elevation** | **−180** |
| **5 HITL** | IR co-evolution + V5 regression | keep — clean contribution #4. Light prose tighten. | light | −15 |
| **6 Experimental Results** | **confuser-first** | **gap-ordered:** confuser-suppression → resolution → cross-modal → real-video deployment → SelCom(min) → threats. Move exhaustive per-category confuser breakdowns to **app**. | reorder + trim | **−80** |
| **7 Conclusion** | old RQ answers | new RQ answers + "deployable system" close | rewrite | ≈ 0 |

---

## 3. What moves to the appendix (rebalance + shorten, nothing deleted)
- **Patch-verifier version history (v1-v4) + threshold sweep tables** → `app:ablations` (it's the *superseded*
  verifier; the body keeps only "why mlp_v5 replaced it").
- **Classifier-zoo comparison tables** (sa32/fnfn/control40 full grids) → `app:ablations` (body keeps the
  robust6→robust8 story + the verdict).
- **Per-category confuser FPR breakdowns** (the thin per-clip tables) → `app:ablations`.
- These join the already-generated `app:models` + `app:datasets`. (Net: the *evidence* stays in the thesis,
  just not in the narrative spine — the same "completeness in appendices" pattern.)

## 4. Length budget (estimate)
159 pp → **~125-135 pp**. Sources: patch-verifier compress (~180 lines), classifier-zoo + per-category tables
to app (~140 lines), prose tightening across Ch5/Ch6 (~80 lines). Tables cost more page-space than prose, so
the appendix moves recover the most pages. (Estimate — confirm after a compile.)

## 5. Execution order (on approval)
1. Ch1 framing rewrite (Background/Problem/RQs/Contributions) — sets the spine.
2. Ch3 scoring-audit demotion + MRI elevation.
3. Ch4 patch-verifier compress + table moves to `app:ablations`.
4. Ch6 reorder to gap-order + table moves.
5. Ch7 conclusion rewrite.
6. `app:ablations` assembled from the moved blocks.
7. hygiene + audit + compile; one diff.

Each step is a reviewable diff on `thesis_working.tex`; KB claims/numbers unchanged (this is framing, not
re-measurement).

## Delivered
- `docs/analysis/2026-06-06_thesis_narrative_reframe_plan.md` (this file).

### RQ set — LOCKED (2026-06-06): 3 RQs
- **RQ1** confuser-gap suppression · **RQ2** in-distribution cost + surface-dependence · **RQ3** cross-modal
  grayscale transfer. Scoring → methodology caveat; resolution + deployability → findings (not RQs).
- Resolution lives in §Resolution-Dependency (justifies imgsz=1280) + Threats; deployability/speed lives in the
  verifier contribution (feature reuse, 46-72×) + Limitations (edge latency unmeasured).
