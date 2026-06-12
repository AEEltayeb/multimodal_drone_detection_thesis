# Thesis completion plan — what we have, the story, the gaps, what runs
**2026-06-10.** Read-only audit of: the 3 thesis copies, `knowledge/evals.csv` (119 rows),
`knowledge/ledger.csv` (73 findings), `thesis_eval/cache/`, the live Overleaf chapters.
Verdict up front: **you are one cache run away from a complete thesis.** Almost everything
else is writing, not evaluating.

---

## 1. The three copies — resolved

| copy | date | what it is | verdict |
|---|---|---|---|
| `docs/thesis_working_distilling_overleaf/` | **Jun 9** (newest, compiles to main.pdf) | the Overleaf split | **THE thesis. Only this gets edited.** |
| `docs/thesis_working.tex` (2420 ln) | Jun 6 | /thesis-skill scratch copy | frozen ancestor — harvest prose only |
| `docs/thesis_chapters.tex` (1675 ln) | May 19 | pre-June monolith | frozen ancestor |

Lineage is linear (chapters → working → overleaf-split), so nothing is lost by freezing the
two monoliths. Recommend renaming them `_frozen_*` on your green light (I won't move files
without it).

---

## 2. THE STORY (the sell, built only from data we hold)

**One sentence:** *A deployable dual-modality drone-detection system — detectors first —
whose components were chosen by statistics rather than benchmarks, and which is kept
correct in production by a human-in-the-loop data engine and a feature-space diagnostic.*

The five beats, each already evidence-backed:

**Beat 1 — Benchmarks pick the wrong detector (the hook).**
Our own data is the proof, and it's brand new (today):
- retrained_v2 is the *best* detector in-domain (F1 0.949 on its own test split — beats
  baseline 0.942 and ft4 0.929) and the *worst* deployed: 0.462 on Svanström (R=0.306),
  **0.007 on CCTV (1 TP in 295)**. `[ledger: rgb-collapse-ood-specific]`
- Anti-UAV is saturated at 0.994 for everything — it cannot rank models. `[antiuav-saturated]`
- And there is **no single best bare detector**: baseline wins Svanström (0.950), ft4 is the
  only one alive on CCTV (0.591 vs 0.145), nobody wins everywhere.
→ So "train a better detector" is not the answer. **A system is.** This motivates everything.

**Beat 2 — The system (C1, the headline).**
ft4 + v3b detectors → robust8 trust router → mlp_v5 / mlp_v5_ir_aligned verifiers →
temporal alert-gate → operator GUI. The claim is NOT that each part is best; it's that the
**composition** is the only deployable configuration:
- Verifier recovers ft4's deliberate precision trade on Svanström: 0.596 → **0.869**
  (P 0.44→0.90, R kept ≥0.84). `[evals: v5_svan_bare/patch/mlp]`
- OOD confuser FPs cut ~13× (RGB), 96% (grayscale), CBAM F1 0.699→0.841 (IR). `[mini + ir-grayscale-harvest..., ir_aligned_gray_heldout]`
- The cascade *tightens variance across detector versions* (15.5pp spread → small) —
  the system makes detector choice less critical. `[cascade-tightens-variance]`

**Beat 3 — Statistics before training (C2, why the system is cheap and right).**
The MRI is the method-contribution: every component was *derived from a measurement*:
- Detector's own ROI features separate drones/confusers: LDA 0.952/0.981, ANOVA F=42,346
  → so the verifier is a 1-2ms MLP on features we already computed, not a second CNN
  (37–72× faster, and *better* OOD). `[v5-lda-separability, mlp-beats-patch-both-modalities]`
- Leakage audit killed scene-fingerprint features → robust8 = 8 honest features,
  404× faster than the 40-feature alternative. `[fusion-feature-leakage, robust6-speed-feature-efficiency]`
- Gray↔thermal gap is an affine offset → z-alignment AUROC 0.500→0.919 → one verifier
  serves thermal AND grayscale. `[gray-thermal-alignable]`

**Beat 4 — The data engine (C3, why it stays correct).**
IR arc V2 0.503 → **V5 0.737 (regression: a positive batch bypassed per-batch HITL review)**
→ v3b 0.967. The one time the discipline was skipped is the one regression. `[ir-version-progression]`

**Beat 5 — The findings (supporting, not the spine).**
Grayscale cross-modal transfer (good-only config); the 28pp scoring-rule swing
`[scoring-rule-swing]`; the speed table. RQ3 closes here.

This **is** the locked A+D spine, and it **is** "drone detection model first" — the models
and pipeline lead; MRI/HITL explain *why they work and stay working*.

---

## 3. Coverage matrix — have / in-flight / gap

| thesis table | status | source |
|---|---|---|
| **Part A** RGB 3-version (4 surfaces) | ✅ **HAVE (final)** — completed today | evals: rgb_rgbds_*, rgb_svan_*, selcom_*, v5_*_bare |
| **Part A** IR 3-version | ✅ HAVE (final) | ir_final_v2/v5/v3b @640 |
| **Part A** verifier patch-vs-mlp (4 surfaces) | ✅ HAVE (final) | v5_svan/selcom/rgbds/antiuav + confuser |
| Speed table (404× / 37–72×) | ✅ HAVE (ledger) | robust6-speed-feature-efficiency, v5 rows |
| MRI numbers (LDA/ANOVA/AUROC) | ✅ HAVE (final, prose exists in thesis) | mri section |
| Grayscale finding (good-only) | ✅ HAVE (final) + 1 cache cell | ir_aligned_gray_heldout etc. |
| **Part B** paired pipeline grid (bare→+C→+F→full) | 🔶 **IN FLIGHT tonight** — antiuav.pkl landed (4000 fr ✓), svanstrom running | thesis_eval cache + replay (written, ready) |
| **Part C** confuser FP grids (rgb/ir/gray) | 🔶 IN FLIGHT (same run; mini n=1000 exists as provisional) | same |
| **Part D** gray pipeline cell | 🔶 IN FLIGHT (svanstrom_gray surface) | same |
| Temporal / segment voting | ❌ GAP **by design** — needs Tier-2 consecutive frames | placeholder + cite video segment findings as motivation |
| GUI + label-reviewer screenshots | ❌ GAP (non-eval) | you supply; boxes exist in tex |
| Edge/desktop latency end-to-end | ❌ GAP (ledger: latency-edge-unmeasured) | optional 10-min bench_speed.py |
| Anti-UAV baseline/rv2 @640 (now @1280) | ⚪ cosmetic — saturated surface | footnote or optional 30-min run |

**Read this table once more: there is NO eval gap that tonight's cache doesn't fill,
except temporal (deliberately Tier-2) and two optional 10–30 min runs.**

## 4. Gap-driven run list (nothing else gets GPU time)

1. **NOW: nothing.** Let the running cache finish (it's alive — antiuav done 01:48,
   ~8 surfaces left, the rest are smaller). Do not restart it; the patched script
   (heartbeat + smallest-first) is for *future* runs.
2. **When cache lands:** `py -u thesis_eval/pipeline_eval_unified.py` (zero-GPU replay)
   → Parts B/C/D at one standard (640/1280 rule, conf=0.25, per-modality scoring).
3. **Optional (10 min):** `eval/bench_speed.py` → kills the ⏱ latency placeholders.
4. **Optional (30 min):** antiuav @640 for baseline/retrained_v2 → table consistency.
5. **LATER, gated on Tier-1:** Tier-2 full-frames on **svanstrom only** (28k, the surface
   where temporal matters) → temporal rows + final digits. Anti-UAV full (85k) only if
   Tier-1 shows it's worth it.

## 5. Writing plan (chapter by chapter, live Overleaf copy)

| file | action |
|---|---|
| `introduction.tex` | contributions **4→3** (MRI=C2, grayscale demoted to finding, GUI into C1); reword RQ2 (drop "real-world video"); KEEP the statistics-before-training paragraph (already good) |
| `related_work.tex` | KEEP as-is |
| `methodology.tex` | mostly KEEP (datasets, IoP+28pp, MRI, architecture, recipes). Demote Roboflow/YouTube dataset subsections to internal mention. ADD: GUI section, ft4 provenance, eval-standard subsection (imgsz/budget/per-modality scoring) |
| `empirical.tex` | **the surgery.** KEEP: HITL/label-reviewer/IR case study (=C3+Part A IR), grayscale section (trim to good-only), selcom, threats. **CUT: Roboflow audit, real-video six-mode, full-cascade-on-video** (→ one-line internal-diagnostic mentions; YouTube/Roboflow stay in appendix as honest diagnostics). **ADD: Part A/B/C/D sections** with tables wired to thesis_eval ids |
| `conclusion.tex` | re-answer RQ1–3 from the new tables |
| `appendices.tex` | keep dataset appendices; label YouTube/Roboflow "internal diagnostics" |

Sequencing: **skeleton edits can start NOW** (cuts, moves, contribution rewrite, table
placeholders tagged with eval-ids) — they don't need the cache. Numbers auto-slot when the
replay runs. Prose polish after you sign off the mini-preview
(`docs/thesis_mini_preview.pdf` — it is this story, compressed).

## 6. Decisions I need from you (only 3)

1. **Green-light the story in §2** (= read the mini-preview; same story).
2. **Freeze + rename the two old copies?** (I propose `_frozen_2026-06-10_*`; needs your OK to move files.)
3. **Temporal:** placeholder-with-citation now + Tier-2 svanstrom later — OK?

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\docs\analysis\2026-06-10_thesis_completion_plan.md` (this file)
