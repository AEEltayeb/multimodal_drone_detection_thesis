# Figure Audit v7 — redundancy + correctness (read-only)

**Scope:** every `\begin{figure}` (incl. subfigures) in the live thesis
`docs/thesis_working_distilling_overleaf/` (main.tex + chapters/*.tex).
`\graphicspath{{figures/}}`; raster + tikz-pdf assets in
`docs/thesis_working_distilling_overleaf/figures/`.
**Method:** every image file opened and viewed (VLM check) except 5 noted below
that hit the conversation image-dimension cap; their captions were instead
cross-checked against body text + `% [source: …]` comments + `knowledge/evals.csv`
+ `docs/generate_thesis_figures.py`.

**Generator note (load-bearing):** `docs/generate_thesis_figures.py` writes to
`docs/figures/` (`OUT_DIR`), **NOT** the live thesis dir. The live
`…/thesis_working_distilling_overleaf/figures/` is a *copy*, so a regenerate does
not propagate unless the PNG/PDF are copied over. This is the root cause of the
one stale-figure finding below (the live PNG is what compiles).

Total figures: **30** (24 raster `\includegraphics`, 2 tikz→PDF, 4 placeholder
`\fbox`). 26 distinct float environments + 4 multi-panel subfigure groups.

---

## TABLE 1 — Figure inventory + checks

Loc = `chapters/<file>:<line>` (figure env line). "Ref'd?" = appears in a
`\ref{}` somewhere in the body.

| Figure (label) | Loc / section | Image file(s) | File OK? | Caption↔image | Caption #s backed? | Ref'd? |
|---|---|---|---|---|---|---|
| `fig:datasets_pie` | methodology:34 §3.1 | `fig_datasets_pie` | ✓ | ✓ (a) train 327,619 / (b) eval 151,695; both pies + legends correct | ✓ matches tab:datasets | **✗ ORPHAN** |
| `fig:dataset_montage` | methodology:42 §3.1 | `fig_dataset_montage` | ✓ | ✓ 8 crops; Svan 19–29px, AntiUAV 36–254px, IR bright blobs | ✓ | **✗ ORPHAN** |
| `fig:confuser_examples` | methodology:116 §3.1.2 | `fig_confuser_examples` | ✓ | ✓ top RGB airplane/bird/heli/other, bottom IR airplane/bird/heli | n/a (qualitative) | ✓ (intro+meth) |
| `fig:confuser_fp_examples` | methodology:124 §3.1.2 | `fig_confuser_fp_examples` | ✓ | ✓ 6 FP crops; det conf 0.82–0.86, P(drone) 0.001–0.077 — exact | ✓ matches caption range | ✓ (intro+meth) |
| `fig:drone_size_hist` | methodology:225 §3.2.3 | `fig_drone_size_hist` | ✓ | ✓ axis="fraction of GT boxes", median 28px Svan / 93px AntiUAV | ⚠ fig median **28** vs body §2.1/§3 "**29.8**" | ✓ |
| `fig:label_reviewer_home` | methodology:409 §3.5.2 | — (`\fbox` PLACEHOLDER) | n/a | PLACEHOLDER text only | n/a | **✗ ORPHAN** |
| `fig:label_reviewer_launch` | methodology:418 §3.5.2 | — (`\fbox` PLACEHOLDER) | n/a | PLACEHOLDER text only | n/a | **✗ ORPHAN** |
| `fig:hitl_loop` | methodology:441 §3.5.3 | `fig_hitl_loop.pdf` | ✓ | (tikz, not VLM-opened) blue disciplined path / red V5 shortcut | n/a | **✗ ORPHAN** |
| `fig:mri_report` | methodology:480 §3.7.2 | — (verbatim text block) | n/a | text excerpt, not an image | LDA 0.952 / F 42,346 backed (stats.json) | ✓ (meth+app) |
| `fig:pipeline` | methodology:521 §3.8.1 | `fig_pipeline.pdf` | ✓ | (tikz, not VLM-opened) det→clf→filter, patch greyed | n/a | **✗ ORPHAN** |
| `fig:confuser_problem` | methodology:531 §3.8.1 | `fig_confuser_panel` | ✓ | ✓ L bird conf 0.46 P=0.00 VETO / R drone 0.85 P=0.96 KEEP — exact | ✓ | **✗ ORPHAN** |
| `fig:pyside_gui` | methodology:577 §3.8.7 | — (`\fbox` PLACEHOLDER) | n/a | PLACEHOLDER text only | n/a | **✗ ORPHAN** |
| `fig:resolution` | methodology:593 §3.8.9 | `fig6_6_resolution` | ✓ | ✓ baseline 0.684/0.964, retr_v2 0.070/0.323; within-model | ✓ +28/+25pp exact | ✓ |
| `fig:fusion_lda` (sub a) | methodology:752 §3.8.13 | `fig8_fusion_lda` | ✓ | (cap-blocked) LDA 98.2% train acc | ✓ (consistent w/ body 98.2%) | ✓ |
| `fig:fusion_auroc` (sub b) | methodology:754 §3.8.13 | `fig8_fusion_auroc` | ✓ | (cap-blocked) per-feature AUROC | ✓ | ✓ |
| `fig:fusion_leakage` (sub c) | methodology:756 §3.8.13 | `fig8_fusion_leakage` | ✓ | (cap-blocked) leakage map | ✓ (matches tab:leakage) | **✗ ORPHAN (subfig c only)** |
| `fig:fusion_stats` (parent) | methodology:750 §3.8.13 | (the 3 above) | ✓ | parent caption (a)/(b)/(c) | ✓ | **✗ parent never `\ref`'d** |
| `fig:pipeline_ablation` | empirical:176 §4.2.2 | `fig_pipeline_ablation` | ✓ | ✓ (a) whiskers visible; (b) RGB 30.3→0.1%, gray 23.8→0.5%, IR 29.4→2.4% | ✓ matches tab:ablation_confusers; whisker def in caption | **✗ ORPHAN** |
| `fig:cascade_segment_fig` | empirical:364 §4.3 | `fig8_cascade_segment` | ✓ | ✓ (a) F1 lifts all 4; (b) FPR −39…−81% (baseline −68%) | ✓ | **✗ ORPHAN** |
| `fig:ir_evolution` | empirical:449 §4.4.2 | `fig4_ir_evolution` | ✓ | ✓ V2 0.661 … V4 0.895 … V5 0.768 (regression marked) … v3b | ✓ matches tab:ir_evolution | **✗ ORPHAN** |
| `fig:robust8_operating` | empirical:498 §4.4.3 | `fig_robust8_operating_point` | ✓ | ✓ robust6 vs +rgb_mean_conf+is_grayscale, argmax dot | ⚠ fig annot "recall 0.12→0.82" vs body "0.577→0.681" (diff sweep/axis) | ✓ |
| `fig:classifier_reversal` | empirical:528 §4.4.3 | `fig8_classifier_reversal` | ✓ | ✓ sa32 ~0.83/0.205, fnfn ~0.22/0.016, control40 | ✓ matches tab:classifiers | **✗ ORPHAN** |
| `fig:patch_catchbar` | empirical:567 §4.4.4 | `fig8_patch_catchbar` | ✓ | ✓ Heli 71%/p0.99, Bird 64%/p0.90, Airplane 52%/p0.54 | ✓ matches tab:patch_audit | **✗ ORPHAN** |
| `fig:distill_verifier_bar` | empirical:599 §4.4.4 | `fig8_distill_verifier` | ✓ | **✗ STALE: rgb_dataset bar = 0.79 (old mlp_v5); caption claims v4 "recovers (0.792→0.916)"** | **✗ plots superseded `v5_rgbds_mlp`=0.7922; v4=0.916 not shown** | **✗ ORPHAN** |
| `fig:failopen_expanded` | empirical:619 §4.4.4 | `fig8_failopen_expanded` | ✓ | ✓ orig(red)/expanded(green) refs, full-veto star, bare square | ✓ (consistent w/ 0.887→0.631) | ✓ |
| `fig:filter_operating` | empirical:656 §4.4.4 | `fig_filter_operating` | ✓ | ✓ RGB shoulder 0.949@1.4%, thermal falls, gray 0.476@0.25; shipped pts marked | ✓ | ✓ |
| `fig:mri_stats` (sub a/b) | empirical:669 §4.5 | `fig8_mri_lda` + `fig8_mri_anova` | ✓ | ✓ (a) LDA 0.9502; (b) ANOVA per-block, p5 outlier ~42k | ⚠ caption "mean 2,006" but fig annotates **median 657** | **✗ ORPHAN (parent never `\ref`'d)** |
| `fig:mri_activation` (sub) | empirical:682 §4.5 | `fig8_mri_act_drone` + `_confuser` | ✓ | (cap-blocked, qualitative) p3/p5 brain-scan drone vs confuser | n/a | **✗ ORPHAN** |
| `fig:ir_gray_align` | empirical:695 §4.5 | `fig9_ir_gray_align` | ✓ | ✓ (A) z-score LDA overlap; (B) bars 0.500/0.707/0.919/0.974 | ✓ matches tab:gray_thermal_auroc | ✓ — **SLATED FOR REMOVAL** |
| `fig:grayscale_qualitative` | empirical:757 §4.6.1 | `fig_grayscale_panel` | ✓ | ✓ L RGB-on-RGB / M IR-gray IoU 0.81 KEEP / R IR-raw IoU 0.00 | ✓ exact | ✓ |

**File existence: all referenced image files resolve. No broken `\includegraphics`.**

---

## TABLE 2 — Redundancy clusters

| Cluster | Figures | Distinct or redundant? | Recommendation | Reason |
|---|---|---|---|---|
| **MRI feature-space** | `fig:mri_stats` (lda+anova), `fig:mri_activation` (act brain-scans), `fig:ir_gray_align`, `fig:fusion_stats` (lda/auroc/leakage) | Mostly **distinct** but **two LDA panels overlap** | **Keep mri_stats + mri_activation; CUT `fig:ir_gray_align` (already slated); keep fusion_stats** | `fig8_mri_lda` (RGB filter LDA 0.95) and `fig8_fusion_lda` (router LDA 0.982) are *different models* → keep both. `fig:ir_gray_align` panel A is a 3rd LDA (IR z-score) and panel B duplicates tab:gray_thermal_auroc exactly → removal is correct. mri_activation (qualitative heatmaps) is unique. |
| **Confuser panels** | `fig:confuser_examples` (raw corpus), `fig:confuser_fp_examples` (FP crops+verdict), `fig:confuser_problem` (2-up bird/drone decision) | **Genuinely distinct** | **Keep all 3** | examples = "what the corpora look like"; fp_examples = "the hallucinations + filter verdict"; confuser_problem = the 2-panel *why-a-filter-works* teaching figure. Different jobs, no overlap. |
| **failopen** | `fig:failopen_expanded` (in thesis) + `fig8_failopen_hist/pca/tradeoff` (on disk, **not included**) | n/a (3 unused on disk) | **Keep failopen_expanded; the 3 unincluded files are dead assets** | Only `fig8_failopen_expanded` is `\includegraphics`'d. `fig8_failopen_{hist,pca,tradeoff}.png` exist in figures/ but are never referenced (archive candidates, not a thesis problem). |
| **Pipeline diagrams** | `fig:pipeline` (tikz architecture) vs `fig:pipeline_ablation` (bar results) | **Distinct** | **Keep both** | One is the architecture schematic, the other the ablation result bars. Not redundant. |
| **Grayscale** | `fig:grayscale_qualitative` (panel, survives) vs `fig:ir_gray_align` (cut) vs `fig_grayscale_panel`(=same as qualitative) | Distinct | **Keep `fig:grayscale_qualitative`** (it IS `fig_grayscale_panel`); cut ir_gray_align | The grayscale **detector** finding stays → its qualitative panel survives. Only the z-score/AUROC alignment figure (filter-side) goes with the grayscale-filter removal. |
| **filter-operating** | `fig:filter_operating` (3-panel dial) | **Distinct / unique** | **Keep** | No overlap; it is the recall/FP operating-point dial. |
| **IR detector dup assets** | `fig9_ir_v3b_anova/heatmap/lda.png` on disk | n/a (unused) | dead assets | Three `fig9_ir_v3b_*` files exist in figures/ but are never `\includegraphics`'d (the IR MRI is reported via tab:ir_mri_sep, not a figure). Archive candidates. |

**Redundancy verdict:** No two *included* figures are true duplicates. The only
genuine cut is the already-slated `fig:ir_gray_align`. Six raster files sit in
`figures/` unreferenced (`fig8_failopen_{hist,pca,tradeoff}`, `fig9_ir_v3b_{anova,heatmap,lda}`,
`fig5_cascade_percategory`, `svan_iop_1280_s9_bar`) — dead assets, not thesis errors.

---

## TABLE 3 — Author notes resolution

| Note | Figure | Current state (caption quote) | Verdict / fix |
|---|---|---|---|
| fig 3.5: "total number of frames? what does 'density' mean?" | `fig:drone_size_hist` (3.5) | y-axis literally **"fraction of GT boxes"**; caption: "$n$ in the legend counts boxes, not frames; each histogram shows the fraction of that dataset's boxes per bin" | **Resolved already** — no "density" anywhere; axis + caption are explicit. **One residual:** figure says median **28px**, body §2.1/§3.2 says **29.8px**. Reconcile to one number. |
| fig 3.6: "looks randomly placed" | (fig 3.6 = `fig:dataset_montage`, the 2nd methodology figure) | montage of GT crops, placed right after the datasets-pie in §3.1 | **Placement is fine** (it belongs in §3.1 Datasets). The real issue is it is an **ORPHAN** (never `\ref`'d), so it floats with no in-text anchor → reads as "random". **Fix: add `Figure~\ref{fig:dataset_montage}`** to the §3.1 drone-size / design-rationale prose. |
| fig 3.12: "unfair, two different models, isolate imgsz" | `fig:resolution` (3.12 = `fig6_6_resolution`) | "Doubling \texttt{imgsz} buys both variants a similar recall increment ($+28$pp baseline, $+25$pp retrained_v2)… isolated **within** each model" | **No confound** — the figure shows *each model at both 640 and 1280*; the comparison is the within-model 640→1280 increment, exactly the imgsz isolation the author wants. **Reassure; no change needed** (optionally add an explicit "within-model Δ" note to pre-empt the question). |
| fig 4.1: "Whiskers are 95% bootstrap CIs — explain?" | `fig:pipeline_ablation` (4.1) | "Whiskers are 95% bootstrap confidence intervals (the central range of the metric over $1{,}000$ resamples of the frames; bars whose whiskers do not overlap differ by more than sampling noise)." | **Resolved** — the whisker definition is already inline in the caption (and again in the chapter intro). No change needed. |
| label-reviewer §3.5.2: wants TWO figures (home + launch) | `fig:label_reviewer_home` + `fig:label_reviewer_launch` | Already **two separate** figure envs: home = "setup window (decide paths/actions)", launch = "review canvas (edit GTs)" | **Resolved structurally** — both exist. **BUT both are `\fbox` PLACEHOLDERS** (no screenshot) and **neither is `\ref`'d**. Fix: drop the two screenshots in, and add in-text refs in §3.5.2. |

---

## Already-slated removals (confirmed)

- **`fig:ir_gray_align` (`fig9_ir_gray_align`)** — confirmed: panel A z-score LDA
  overlap, panel B AUROC bars 0.500/0.707/0.919/0.974. Removal OK.
- **`tab:gray_thermal_auroc`** — its 4 AUROC values are duplicated by panel B above.

**⚠ Orphaned-number risk on removal:** the AUROC progression **0.500 → 0.707 (CORAL)
→ 0.919 (z-score) → 0.974 (ceiling)** is cited in *prose* in the **abstract**
(`0.500 → 0.919`), **introduction** Contribution 2, **§lit_probing** (`0.919 vs 0.707`),
**§ir_xmodal_verifier** and **§mri_findings**. If BOTH the figure and tab:gray_thermal_auroc
go, these inline numbers lose their display anchor. **Keep at least the inline
`(0.500→0.919)` numbers in prose** (they survive as a sentence), or retain a single
one-line statement of the CORAL-vs-zscore result. The grayscale **detector** finding
figure (`fig:grayscale_qualitative`) is untouched and correctly survives.

---

## TOP FIXES (ranked)

1. **`fig:distill_verifier_bar` (fig8_distill_verifier) — STALE, contradicts its own
   caption.** The `rgb_dataset` bar shows **0.79** (it plots `v5_rgbds_mlp`=0.7922,
   the *superseded* pre-v4 filter) while the caption + `tab:distill_verifier` claim
   the **v4 bird-split build recovers rgb_dataset to 0.916**. Title even reads
   "mlp_v5", not v4. **Action:** repoint the generator's `rgb_dataset` mlp id from
   `v5_rgbds_mlp` to the v4 eval row (F1 0.916 / R 0.887), regenerate, **and copy
   the PNG+PDF into `…/thesis_working_distilling_overleaf/figures/`** (generator
   writes to `docs/figures/`, not the live dir). Until copied, the stale bar compiles.

2. **`fig:patch_catchbar` (fig8_patch_catchbar) — raw LaTeX leaking into the raster.**
   The rendered PNG prints literal matplotlib strings: x-label shows
   `Confuser catch rate (patch v2, \texttt{patch\_thr}=0.5)` and the airplane bar
   shows `(drone-TP veto only 5.4\%)` — `\texttt{}`/`\%` are unescaped. Also the
   title overlaps the "0.90 decisiveness bar" annotation. **Action:** fix the
   generator strings (plain text, not LaTeX) and nudge the bar-label position;
   regenerate + copy.

3. **Orphan figures (17) — add in-text `\ref`s or cut.** Defined but never
   `\ref`'d: `datasets_pie, dataset_montage, hitl_loop, pipeline, confuser_problem,
   fusion_stats(+leakage subfig), pipeline_ablation, cascade_segment_fig,
   ir_evolution, classifier_reversal, patch_catchbar, distill_verifier_bar,
   mri_stats, mri_activation`, plus the two label-reviewer placeholders + pyside_gui.
   Several are *discussed* in prose but the `Figure~\ref{…}` was dropped. **Action:**
   add one `\ref` each at the natural sentence (e.g. `fig:pipeline` at "the pipeline
   processes…", `fig:ir_evolution` at tab:ir_evolution, `fig:pipeline_ablation` at
   the four-observations paragraph, `fig:dataset_montage` in §3.1). This is the
   single biggest hygiene gap and is what makes fig 3.6 "look randomly placed".

4. **4 placeholder figures still un-rendered.** `fig:label_reviewer_home`,
   `fig:label_reviewer_launch`, `fig:pyside_gui` are `\fbox` placeholders.
   **Action:** drop in the three screenshots (label-reviewer setup, label-reviewer
   canvas, PySide GUI) before submission; they are referenced as real contributions.

5. **`fig:drone_size_hist` median mismatch.** Figure = "median 28 px", body =
   "median 29.8 px" (both Svanström √area). **Action:** pick one (29.8 is the
   per-frame median cited everywhere else) and reconcile the figure title.

6. **`fig:mri_stats` caption stat mismatch.** Caption says ANOVA "mean 2,006" but
   panel (b) annotates "median F=657". **Action:** either cite median 657 in the
   caption (matches the plotted line) or add both; don't name a statistic the figure
   doesn't show.

7. **`fig:robust8_operating` annotation vs body.** Figure annotation "recall
   0.12→0.82" vs body "Svanström-grayscale recall 0.577→0.681 at τ=0.20". Different
   sweep/axis, but a reader will conflate them. Since this is now *design-history*
   (robust8-nr shipped), **lightly reword** the caption to say the figure's curve is
   the grayscale `trust_rgb` P/R sweep, distinct from the Svanström-grayscale
   detection-recall figure cited in §4.4.3.

8. **Cosmetic:** `fig8_failopen_expanded` title "Svanstrom" (no umlaut);
   `fig_datasets_pie` legend "(39.4%…" slightly clipped at right margin. Low priority.

---

### Delivered
- `docs/analysis/2026-06-18_verify_v7_figures.md` (this file) — absolute path:
  `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_verify_v7_figures.md`
- No thesis/source files modified (read-only audit).
