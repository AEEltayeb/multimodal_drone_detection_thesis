# Thesis coherence + structural-architecture scan (2026-05-31)

Full read of `docs/thesis_working.tex` (7 chapters + 2 appendices, ~125 pp). Two questions:
(1) is the thesis internally coherent? (2) how is it organised "architecture-wise," and are there
alternative organisations that still satisfy the `tesi_master.tex` IMRAD template?

---

## 1. Document structure as it stands

| Ch | Title | Role | Notes |
|----|-------|------|-------|
| 1 | Introduction | Intro | RQs (4 + 1 methodological thread), 5 contributions, ethics, outline |
| 2 | Related Work | Background | + architectural & numerical comparison tables |
| 3 | System Architecture and Component Design | **Contribution + inline results** | design rationale, all components, **and** their ablations (classifier comparison, patch sweep, catch-rate audit, mlp_v5 eval) |
| 4 | Dataset Curation and HITL | Contribution (process) | label reviewer, co-evolution loop, IR V2→v3b case study |
| 5 | Methodology | **Methods (defined late)** | datasets, IoP rule, scoring audit, leakage audit, recipes, Model MRI |
| 6 | Experimental Results | Results + Discussion + Threats | cumulative, OOD, real-video, grayscale, full cascade, SelCom, threats, limits |
| 7 | Conclusion | Conclusion | RQ answers, production stack, future work |
| A | Datasets in Detail | Appendix | per-source / per-clip metadata |
| B | Glossary | Appendix | abbreviations |

`tesi_master.tex` target IMRAD (5 ch): Intro · Background · Contribution/System · **Empirical Evaluation**
(Study Design→RQs, Context, Data Analysis · Result · Discussion · Threats) · Conclusion.

### How it looks architecture-wise — verdict

**Content is strong; the *ordering* has two real seams.**

- **Seam A — Methodology is defined after it is used.** Ch3 and Ch4 report IoP-scored Svanström
  numbers, the 28-pp scoring swing, and dataset compositions *before* Ch5 defines the IoP rule,
  the scoring-rule audit, and the corpora. A reader meets `IoP@0.5`, `trust-aware F1`, and
  `28 pp` roughly 800 lines before their definitions. The thesis leans on forward-refs
  (`\S\ref{sec:metrics}`, `\S\ref{sec:scoring_audit}`) to paper over this, but the dependency runs
  backwards.
- **Seam B — Results are split across two chapters.** Component ablations that decide production
  picks (classifier comparison `tab:classifiers`, patch threshold sweep, catch-rate audit, the
  `mlp_v5` table `tab:distill_verifier`) sit in Ch3; the surface-level results (cumulative
  suppression, OOD, real-video, grayscale, SelCom) sit in Ch6. tesi_master would gather all
  empirical results in the Empirical Evaluation chapter. Today "where do I find a number" has two
  answers depending on whether it justifies a design choice or reports an outcome.

Neither seam is fatal — the inline-ablation style (justify each component where it's introduced)
is a legitimate and readable systems-thesis convention. But it is a *deviation* from the
template, and the methodology-after-use inversion is the weaker of the two.

---

## 2. Coherence findings (recorded in `knowledge/coherence.csv`)

### HIGH — Production-verifier identity is ambiguous (row_1)
The deepest issue, and a direct consequence of the `mlp_v5` architecture change.

- Ch3 design (`fig:pipeline`, component list), §distill_verifier, and the Conclusion production
  stack all state: **production verifier = per-frame `mlp_v5`**, patch verifier superseded.
- But the *as-evaluated* sections still describe the **patch verifier alert-gate** as the deployed
  stage: §alert_gate ("production cascade is `alert_gate_only` … in the PySide GUI"),
  §realvideo_cascade ("…→ patch verifier alert gate", `patch_thr=0.7`), and the GUI caption
  `fig:pyside_gui` ("the patch verifier runs silently at the alert gate").

These are not strictly contradictory — `mlp_v5` is a *recommendation* per `ledger=v5-ship-per-frame`,
while every cascade *experiment* (and the current GUI) ran the patch verifier, which predates
`mlp_v5`. But the thesis never says that plainly in one place, so a reader cannot tell whether
`mlp_v5` is wired into the evaluated system or is a forward-looking production choice.
**Fix (needs your input):** one sentence, stated once early (Overview) — e.g. "the cascade
*experiments* in Ch6 use the patch verifier; `mlp_v5` is the recommended production verifier that
supersedes it on confuser-rich surfaces, and is dropped into the same alert-gate slot as a
per-frame stage." Then the GUI/eval descriptions are honestly "what was run," not contradictions.
*Open question I can't resolve from the repo: is `mlp_v5` actually loaded in the PySide GUI today,
or still the patch verifier?* That determines the exact wording.

### MED — Forward dependency, methodology after use (row_2). See Seam A above.
### MED — Results split across Ch3/Ch6 (row_3). See Seam B above.

### LOW / already-consistent (no action)
- RQ1 (`patch verifier, superseded … by mlp_v5`), contribution 5, §patch_verifier_arch intro,
  `tab:related_systems`, fig 3.1, and the Conclusion are now mutually consistent on supersession.
- `patch_thr=0.9` is consistent across abstract, contribution 1, RQ2, §cumulative, sweep table.
- `R=0.072` honest framing (retrained_v2@640; baseline@640 pending) is consistent in all 4 spots.
- Grayscale numbers (`F1=0.636` aggregate, `0.837 vs 0.840` seagull clip) agree across Ch1/Ch3/Ch6/Concl.
- Two parked evidence flags remain (row_5 IR-Svanström@1280; baseline@640 recall) — left flagged per your instruction.

### Fixed this pass
- §design_rationale: added an explicit sentence that fail-open is a property of the detector
  choice + cascade ordering, **not** the patch verifier (your question).
- Overview "three architectural choices": third pillar reworded from "alert-gated patch verifier"
  to "downstream confuser verifier — in production the per-frame `mlp_v5`, superseding the
  alert-gated patch verifier." (Removed the last spot that named the superseded component as a
  current pillar.)

---

## 3. Structural alternatives that still satisfy tesi_master IMRAD

All three keep the 5-chapter IMRAD skeleton OR map cleanly onto it. Ranked by effort.

### Option C — Minimal: signpost, don't move (lowest effort, ~0.5 day)
Keep the 7-chapter layout. Add a short "Methods used ahead of their definition" note at the top of
Ch3 pointing to Ch5, and tighten forward-refs. Fixes nothing structurally; only mitigates Seam A's
surprise. **Pick this if the examiner cares about content, not IMRAD conformance.**

### Option B — Re-order, don't merge (medium effort, ~1–2 days) — RECOMMENDED
Move Methodology *before* Architecture; otherwise keep chapters intact:
1 Intro · 2 Related Work · **3 Methodology** (datasets, IoP, scoring audit, MRI, reproducibility) ·
4 System Architecture & Component Design (inline ablations now have their metrics pre-defined) ·
5 Dataset Curation & HITL · 6 Experimental Results · 7 Conclusion.
Kills Seam A entirely with almost no prose rewriting (mostly a chapter-move + ref check). Seam B
(results in two chapters) remains but is far less jarring once metrics are defined up front.
**Best effort-to-payoff.**

### Option A — Full IMRAD collapse to 5 chapters (high effort, ~3–5 days)
What the `structure` mode originally targeted:
1 Intro · 2 Background · **3 Contribution/System** (architecture + components + HITL as a
methodology subsection, *ablations moved out*) · **4 Empirical Evaluation** (Study Design→RQs +
datasets + IoP/scoring, then ALL results: ablations + cumulative + OOD + real-video + grayscale +
SelCom, then Discussion, then Threats) · 5 Conclusion.
Strict template match; resolves both seams. Cost: large reshuffle, every `\ref` re-pointed, the
inline-ablation narrative ("justify each component as introduced") is lost — components are
described in Ch3 but evaluated in Ch4, so each gains a forward-ref. **Pick only if strict IMRAD
conformance is graded.**

| Option | Fixes Seam A (methods-late) | Fixes Seam B (split results) | Effort | Keeps inline-ablation style |
|--------|:---:|:---:|:---:|:---:|
| C signpost | partial | no | XS | yes |
| **B re-order** | **yes** | partial | M | yes |
| A IMRAD collapse | yes | yes | L | no |

---

## Delivered
- `docs/thesis_working.tex` — 2 edits: §design_rationale fail-open clarification (L415);
  Overview third-pillar reworded to per-frame `mlp_v5` (L410).
- `knowledge/coherence.csv` — rows row_1 (high, verifier identity), row_2 (med, methods-late),
  row_3 (med, split results); views regenerated.
- This doc: `docs/analysis/2026-05-31_thesis_coherence_architecture_scan.md`.
