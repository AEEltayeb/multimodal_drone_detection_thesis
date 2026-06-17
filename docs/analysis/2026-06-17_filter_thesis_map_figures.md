# Filter -> Thesis Map: Figures, Tables, Cited Artifacts & Traceability (2026-06-17)

**Purpose.** Map every FIGURE, TABLE, and CITED ARTIFACT in the LaTeX thesis whose content
depends on the two "confuser filters" (verifiers), so they can be regenerated when the filters
are swapped.

**The two filters.**
- RGB: `mlp_v5` (weight `models/verifiers/rgb_v5/mlp_v5.pt`), P(drone) threshold **0.25**.
- IR: `mlp_v5_ir_aligned` — ONE network, TWO per-modality input scalers:
  - thermal scaler `mlp_aligned.pt` (`models/verifiers/ir_aligned/mlp_aligned.pt`), thr **0.05**, conf 0.40;
  - grayscale scaler `mlp_aligned_gray.pt` (`models/verifiers/ir_aligned/mlp_aligned_gray.pt`), thr **0.25**, conf 0.25.
- Predecessor (appears in many tables as the comparison baseline): MobileNetV3 **patch verifier v2**
  (`confuser_filter4_{rgb,ir}_v2_backup.pt`), `patch_thr` 0.5/0.9.

---

## ⚠️ CRITICAL: which thesis file is live

There are THREE divergent thesis sources in `docs/`. The assigned PRIMARY (`docs/thesis_working.tex`)
is **NOT** the one the audit and the recent (robust8-nr / `filt`-cell) work target.

| File | State re: filters | Notes |
|------|-------------------|-------|
| `docs/thesis_working.tex` (2420 ln) — **assigned primary** | **patch-verifier-era snapshot.** `mlp_v5` / `mlp_v5_ir_aligned` are present and described as the *successors*, but the headline pipeline tables still run **patch v2** + classifiers `sa32`/`fnfn`/`control40`. **No** `robust8-nr`, **no** `filt`/`clf->filt`/`filt->clf` paired cells, **no** `fig:filter_operating`, **no** `fig:robust8_operating`, **no** `tab:ablation_dut`. | This file does NOT match the CLAUDE/MEMORY handover, which describes the live thesis. |
| `docs/thesis_chapters.tex` (1675 ln) — older synced copy | Even older. No `mlp_v5`/`robust8`/`filt` strings at all; only generic "confuser/filter/verifier" prose. | Report separately (below). Effectively superseded. |
| `docs/thesis_working_distilling_overleaf/chapters/*.tex` (methodology/empirical/...) — **the live overleaf thesis** | THE one with `robust8-nr`, `filt`/`clf->filt`/`filt->clf` cells, `fig:filter_operating`, `tab:ir_aligned`, DUT ablation. **The audit `thesis_eval/_audit_headline_numbers.py` pins its CBAM cells directly to `methodology.tex` + `empirical.tex`.** | Per CLAUDE rules this skill must edit `thesis_working.tex`, but the *enforced* numbers live here. **A filter swap must regenerate against THIS file's tables + the Tier-1 harness, then re-sync into `thesis_working.tex`.** |

The sections below catalog `docs/thesis_working.tex` (the assigned primary) exhaustively, and flag the
filter cells that exist *only* in the overleaf chapters as a separate to-do.

---

## (A) Filter-dependent FIGURES — `docs/thesis_working.tex`

"Regen needed?" = YES if a filter swap changes the rendered content.

| Line | \label | Caption (short) | Image path(s) | Generating script | Regen needed? |
|------|--------|-----------------|---------------|--------------------|---------------|
| 1164 | `fig:patch_sweep` | Patch verifier threshold sweep on Svanström drones (elbow @ patch_thr=0.9) | `fig6_3_threshold_sweep` | `docs/generate_thesis_figures.py::fig_threshold_sweep` (L134) | YES (patch-v2; swap = new filter sweep) |
| 1210 | `fig:patch_catchbar` | Patch-v2 per-bucket confuser catch rate vs 0.90 bar | `fig8_patch_catchbar` | `docs/generate_thesis_figures.py::fig_patch_catchbar` (L577) | YES (v2-specific; new filter = new catch bars) |
| 1200 | `fig:patch_failure_modes` | Representative patch-verifier outputs (PLACEHOLDER, no image) | — (fbox placeholder) | none (manual) | YES (qualitative, filter-specific) |
| 1261 | `fig:distill_verifier_bar` | `mlp_v5` vs patch-v2 vs bare FT4, per surface (F1 + halluc) | `fig8_distill_verifier` | `docs/generate_thesis_figures.py::fig_distill_verifier` (L398) — reads `knowledge/evals.csv` | **YES (core filter figure)** |
| 1281 | `fig:mri_activation` | RGB MRI activations: drone vs confuser (incl. `fig8_mri_act_confuser`) | `fig8_mri_act_drone`, `fig8_mri_act_confuser` (subfigs L1283-1285) | MRI tool (mri/), not a resident standalone .py | PARTIAL (qualitative; RGB verifier feature-space) |
| 1314 | `fig:failopen_expanded` | Fail-open recovery, expanded confuser ref | `fig8_failopen_expanded` | MRI/failopen script (`eval/test_failopen_verifier.py` cited; image gen not resident) | YES (mlp_v5 OOD-abstain) |
| 1336 | `fig:failopen` (+ subfigs `fig:failopen_hist/tradeoff/pca`) | Fail-open hist / tradeoff / PCA | `fig8_failopen_hist`, `fig8_failopen_tradeoff`, `fig8_failopen_pca` (L1338-1342) | same as above | YES (mlp_v5 distance-to-confuser) |
| 1037 | `fig:fusion_stats` (+ subfigs `fig:fusion_lda/pca/auroc/leakage`) | Fused-feature LDA/PCA/AUROC/leakage (classifier feature space) | `fig8_fusion_lda/pca/auroc/leakage` (L1039-1045) | MRI/stats (not resident standalone) | NO-ish (drives the trust classifier, not the verifier filters) |
| 880 | `fig:ir_v3b_lda` | IR `v3b` LDA drone-vs-confuser separability | `fig9_ir_v3b_lda` (+ `fig9_ir_v3b_anova`) | MRI (`mri/`, `mri/results/ir_v3b_report/stats.json`) | PARTIAL (IR verifier feature basis) |
| 892 | `fig:ir_v3b_heatmap` | Top-20 IR discriminative features (z-score) | `fig9_ir_v3b_heatmap` | MRI | PARTIAL (IR verifier feature basis) |
| 904 | `fig:ir_mri_activation` | IR MRI activations: drone vs held-out CBAM confuser | `fig8_mri_ir_act_drone`, `fig8_mri_ir_act_confuser` (L908-909) | MRI | YES (IR aligned verifier) |
| 916 | `fig:ir_gray_align` | Cross-modal grayscale->thermal feature alignment | `fig9_ir_gray_align` | MRI (`mri/modality_align.py`) | **YES (IR aligned/grayscale filter)** |
| 1516 | `fig:cumulative_confuser` | Cumulative confuser-zoo fire S1->S2->S3 (fnfn + patch v2) | `fig6_1_cumulative_confuser` | `docs/generate_thesis_figures.py::fig_cumulative_confuser` (L55) reads `eval/results/_cumulative_halluc/.../summary.json` | YES (S3 = patch v2) |
| 1538 | `fig:svanstrom_by_cat` | Per-category Svanström cascade suppression (S3 = patch) | `fig6_2_svanstrom_by_category` | `docs/generate_thesis_figures.py::fig_svanstrom_by_category` (L93) | YES (S3 = patch v2) |
| 1554 | `fig:surface_exchange` | Cascade drone-F1 exchange (Svanström vs real video), patch_thr=0.9 | `fig8_surface_exchange` | `docs/generate_thesis_figures.py::fig_surface_exchange` (L546) | YES (S3 = patch v2) |
| 1838 | `fig:cascade_per_stage` | Per-stage cascade on one sequence (PLACEHOLDER, d = post-patch-verifier) | — (fbox placeholder) | none (manual) | YES (qualitative; final = filter) |
| 1873 | `fig:perframe_segment` | Per-frame vs segment F1 (cascade incl. patch veto) | `fig8_perframe_segment` | `docs/generate_thesis_figures.py::fig_perframe_segment` (L601) | YES (cascade contains filter) |
| 1903 | `fig:cascade_segment_fig` | Segment F1 / confuser FPR, RGB vs full cascade | `fig8_cascade_segment` | `docs/generate_thesis_figures.py::fig_cascade_segment` (L499) | YES (cascade contains filter) |
| 1995 | `fig:classifier_reversal` | Classifier choice reversal on real video (cascade incl. patch) | `fig8_classifier_reversal` | `docs/generate_thesis_figures.py::fig_classifier_reversal` (L623) | PARTIAL (mostly classifier, filter present) |

NOT filter-dependent (for contrast, excluded from regen): `fig:pipeline`, `fig:rgb_threestance`,
`fig:resolution`, `fig:realvideo_pareto` (six raw-detector modes, no filter), `fig:drone_size_hist`,
`fig:dataset_montage`, `fig:ir_evolution`, `fig:v5_regression`, `fig:label_reviewer`, `fig:hitl_loop`,
`fig:confuser_problem` (placeholder), `fig:mlp_pipeline_placeholder`, `fig:mri_stats` (RGB classifier stats).

---

## (B) Filter-dependent TABLES — `docs/thesis_working.tex`

| Line | \label | Caption (short) | Filter-driven cells | Data source |
|------|--------|-----------------|---------------------|-------------|
| 1146 | `tab:patch_sweep` | Patch-thr sweep on Svanström (S3 rows) | All S3 rows (patch_thr 0.5-0.9); R/F1/FP per threshold | Ledger §7; `eval/cumulative_halluc.py` (svan_iop_1280_s9) |
| 1180 | `tab:patch_audit` | Patch-v2 catch/veto by bucket @0.5 | Every catch/veto + median-patch-prob cell | `eval/audit_patch_catch.py` (Ledger §6.1) |
| 1242 | `tab:distill_verifier` | **`mlp_v5` vs patch-v2 vs bare FT4** (F1 + halluc, 5 surfaces) | `+ patch v2` and `+ mlp_v5` columns (both halves) | Ledger v5-beats-patch/v5-rgbds-ceiling; `eval/eval_v4_vs_patch.py` |
| 1769 | `tab:ir_aligned` | **Thermal-deploy `mlp_v5_ir_aligned` (mlp_aligned.pt @0.05) vs bare IR** | `+ aligned MLP` column + ΔR/ΔF1 + the `patch verifier on CBAM` row | Ledger ir-grayscale-harvest-solves-thermal-verifier; `mri_train_aligned`, `eval_run_aligned_full`; `mri/results/...` |
| 1792 | `tab:ir_aligned_gray` | **Grayscale-deploy aligned (mlp_aligned_gray.pt @0.25)** | `aligned-gray` column + dedicated `mlp_v5_gray` column | Ledger ir-grayscale-harvest-solves-thermal-verifier; `eval_run_aligned_full` (rgb_gray_heldout_640) |
| 1501 | `tab:cum_confuser` | Confuser-zoo S1/S2/S3 fire (fnfn + patch v2) | **S3 fire** column (patch verifier) | `eval/results/_cumulative_halluc/confuser_fusion_no_fn_model_v1.1/summary.json` |
| 1523 | `tab:cumulative_svanstrom` | Svanström paired S1/S2/S3 (fnfn + patch v2) | **S3 + patch v2 (thr=0.5)** row | Ledger §7; `eval/cumulative_halluc.py` (svan_iop_1280) |
| 1591 | `tab:ood_rgb_confuser` | Patch-verifier RGB confuser FP suppression (Roboflow OOD) | All supp% + cuts columns (patch run as per-frame filter) | Ledger §8.2; `eval/run_roboflow_eval.py` |
| 1611 | `tab:ood_ir` | IR detector on Roboflow OOD (raw vs + patch v2) | `+ patch v2` rows | Ledger §8.3; `eval/run_roboflow_eval.py` (roboflow_ir_drone_640) |
| 1855 | `tab:cascade_perframe` | Per-frame RGB vs +classifier (cascade) | classifier-merge cols (filter downstream) — indirect | Ledger §9.5; pipe_vid_*_pf |
| 1887 | `tab:cascade_segment` | Segment F1 / confuser FPR, RGB vs full cascade | Cascade F1 / Cascade FPR / FPR-cut cols (incl. patch veto) | Ledger §9.5; pipe_vid_*_seg |
| 1928 | `tab:cascade_percategory` | Segment confuser FPR by category (after temporal+patch veto) | All bird/airplane/heli cells (post-patch) | Ledger §9.5.9; pipe_percat_sa32 |
| 1963 | `tab:cascade_classifier_drone` | Segment drone F1 under 3 classifiers (each w/ patch verifier) | All cells (each col = classifier + same patch filter) | Ledger §9.5.8 |
| 1979 | `tab:cascade_classifier_fpr` | Segment confuser FPR under 3 classifiers (+ patch) | sa32/control40/fnfn cells (post-filter) | Ledger §9.5.8 |
| 982  | `tab:classifiers` | Classifier comparison on OOD zoo (fnfn/sa32/control40) | filter not the driver, but cascade context | `eval/results/_cumulative_halluc/confuser_*/summary.json` |
| 304/316 | `tab:related_systems` | Related systems (prose names `mlp_v5` as the verifier) | textual only (row "This thesis") | — |

NOT filter-dependent: `tab:rgb_comparison`, `tab:selcom`, `tab:ood_rgb_drone`, `tab:realvideo_master`
(six raw modes), `tab:realvideo_seagull`, `tab:ir_grayscale`, `tab:leakage` (classifier features),
`tab:robust6_pipeline` (alert ablation; classifier metric, patch fixed), dataset tables, `tab:models_evaluated`.

---

## (B2) Filter cells that exist ONLY in the overleaf chapters (NOT in `thesis_working.tex`)

These are the `filt` / `clf->filt` / `filt->clf` paired-ablation cells and the new figures the
audit pins. They live in `docs/thesis_working_distilling_overleaf/chapters/{methodology,empirical}.tex`
(and are described in the CLAUDE handover). A swap must regenerate these too:

- **Paired full-pipeline tables** (Svanström / Anti-UAV / DUT) with cells:
  `filt_mlp_rgb`, `filt_mlp_ir`, `filt_patch`, `clf->filt[robust8]`, `filt->clf[robust8]`,
  `clf->filt[robust8_nr_drop]`, `filt->clf[robust8_nr_drop]` — all from
  `thesis_eval/results/tier1_results.json` (+ `results_noreject/`, `results_dut/`, `results_clean/`).
- **Part C confuser tables**: `C_confuser.{bare,filt_mlp,filt_patch,clf->filt[...]}` fire_rate / FP.
- **S4 verifier rows** (`rgb_dataset_test`, `selcom_val`, `ir_dset_final`): `bare`/`filt_mlp`/`filt_patch`.
- **`tab:ir_aligned`** CBAM `48->15` cell — restated in BOTH methodology prose and empirical table
  (audit pins both to canonical 15).
- **Figures** `fig:filter_operating` (3-panel per-filter P(drone) sweep) and `fig:robust8_operating`
  — generators `eval/filter_operating_sweep.py` (-> `eval/results/filter_operating_sweep.json`),
  output `docs/thesis_working_distilling_overleaf/figures/fig_filter_operating.pdf`.

---

## (C) CITED ARTIFACT PATHS (filter-relevant)

### Filter weights (resident, swap targets)
| Path | Where cited | Type |
|------|-------------|------|
| `models/verifiers/rgb_v5/mlp_v5.pt` | harness `pipeline_eval_unified.py` (THESIS_MLP_V5); thesis prose `tab:distill_verifier` | weight (RGB filter) |
| `models/verifiers/ir_aligned/mlp_aligned.pt` | harness (THESIS_ALIGNED); `tab:ir_aligned` | weight (IR thermal scaler) |
| `models/verifiers/ir_aligned/mlp_aligned_gray.pt` | harness (THESIS_ALIGNED_GRAY); `tab:ir_aligned_gray` | weight (IR grayscale scaler) |
| `models/routers/robust8_noreject_drop/model.joblib` | audit CITED_PATHS; overleaf prod router | weight (router, pairs w/ filter) |
| `confuser_filter4_{rgb,ir}_v2_backup.pt` (patch v2) | `tab:distill_verifier`, `tab:ir_aligned`, all cascade tables | weight (predecessor filter) |

### Result JSONs the filter cells read (audit-pinned)
| Path | Where cited | Type |
|------|-------------|------|
| `thesis_eval/results/tier1_results.json` | `_audit_headline_numbers.py` (T) — all `filt`/`clf->filt` cells | json |
| `thesis_eval/results/temporal_results.json` | audit (V) — video filter cells | json |
| `thesis_eval/results_noreject/{tier1,temporal,notes_round1}_results.json` | audit (TN/VN/NN) — robust8-nr filter cells | json |
| `thesis_eval/results_dut/tier1_results.json`, `runs/results_dut/tier1_results.json` | audit (DUT ablation) | json |
| `thesis_eval/results_clean/tier1_results.json`, `runs/clean_split/clean_split_results.json` | audit (CLEAN) | json |
| `eval/results/filter_operating_sweep.json` | audit (FIG rgb/gray recall/fire) | json |
| `eval/results/_cumulative_halluc/confuser_*/summary.json` | `thesis_working.tex` `tab:cum_confuser`, fig6_1/6_2; `docs/generate_thesis_figures.py` | json |
| `mri/results/ir_v3b_report/stats.json` | audit (MRI ir LDA/halluc/fp_cut) | json |
| `mri/results/v5_report_regen/{stats.json,report.md}`, `mri/docs/mlp_v5_report_regen.md` | audit CITED_PATHS | json/md |

### Scripts (generators)
| Path | Role |
|------|------|
| `thesis_eval/pipeline_eval_unified.py` | **Applies both filters** (env-overridable weights+thresholds); writes tier1/temporal results. CORE swap point. |
| `thesis_eval/pipeline_cache_unified.py` | Builds the cached features both filters score. |
| `docs/generate_thesis_figures.py` | fig6_1, fig6_2, fig6_3, fig8_distill_verifier, fig8_patch_catchbar, fig8_cascade_segment, fig8_surface_exchange, fig8_perframe_segment, fig8_classifier_reversal |
| `eval/filter_operating_sweep.py` | fig:filter_operating + its JSON (overleaf) |
| `eval/eval_v4_vs_patch.py` | `tab:distill_verifier` numbers |
| `eval/audit_patch_catch.py` | `tab:patch_audit`, `fig:patch_catchbar` |
| `eval/cumulative_halluc.py` | cumulative S1/S2/S3 tables + figs |
| `eval/run_roboflow_eval.py` | `tab:ood_rgb_confuser`, `tab:ood_ir` |
| `eval/test_failopen_verifier.py` | `tab:failopen`, `fig:failopen*` |
| `mri/modality_align.py`, `mri/classifier.py`, `mri/holdout.py` | IR alignment + held-out CBAM gate (`tab:ir_aligned`, `fig:ir_gray_align`) |
| MRI tooling (`mri/`) | fig8_fusion_*, fig8_failopen_*, fig8_mri_*, fig9_ir_* (no resident standalone generator found — produced via mri/ + manual export) |

### Ledger rows (knowledge/ledger.csv) about the filters
`v5-beats-patch`, `v5-rgbds-ceiling`, `v5-ship-per-frame`, `v5-lda-separability`,
`mlp-v5-recall-drop-is-ood-coverage`, `mlp-beats-patch-both-modalities`, `mri-v5-report-regen`,
`ir-grayscale-harvest-solves-thermal-verifier`, `ir-recall-fixed-by-drone-diversity`,
`patch-catch-below-bar`, `patch-verifier-distribution-bound`, `filter-threshold-sweep`,
`robust8-grayscale-router`, `robust8_nr_drop` (production), `noreject-router-over-reject`.

---

## (D) TRACEABILITY MECHANISM — how a number is tied to its source, and what to add for new filters

**Chain today (3 layers).**
1. **Harness -> JSON.** `thesis_eval/pipeline_eval_unified.py` runs the filters over a cached corpus
   and writes `thesis_eval/results/tier1_results.json` (+ temporal/noreject/dut/clean variants). Filter
   identity is parameterised: weights and thresholds are env vars (`THESIS_MLP_V5`, `THESIS_ALIGNED`,
   `THESIS_ALIGNED_GRAY`, `THESIS_RGB_THR_MLP=0.25`, `THESIS_IR_THR_MLP=0.05`, `THESIS_GRAY_THR_MLP=0.25`),
   defaulting to the resident weight paths in §C.
2. **JSON -> thesis cell pin.** `thesis_eval/_audit_headline_numbers.py` hard-codes the *claimed* value
   of each headline cell next to a direct lookup into the JSON (e.g.
   `("rgbconf mlp fire", 0.0106, T["rgb_confuser"]["C_confuser"]["filt_mlp"]["fire_rate"])`), and fails
   if they differ by >5e-4. It ALSO regexes two prose/table restatements of the CBAM `48->15` cell
   directly out of `docs/thesis_working_distilling_overleaf/chapters/{methodology,empirical}.tex`.
3. **Path existence (`CITED_PATHS`).** Same audit asserts every cited artifact (weights, result JSONs,
   scripts, the `fig_filter_operating.pdf`, the robust8-nr router/script/results) physically exists in
   this repo. A moved/missing file is an audit failure. (Per `[thesis-artifact-residency]` rule.)
4. **Source comments in `.tex`.** Most numbers carry an inline `% [source: ledger=...; eval=...; run=...; cache=...; config=...]` trailer, and `knowledge/ledger.csv` is the canonical finding store. `docs/generate_thesis_figures.py` reads `knowledge/evals.csv` for the distill bar chart.

**What must be added/updated when the filters are swapped.**
1. **Re-run the harness** with new weights: point `THESIS_MLP_V5` / `THESIS_ALIGNED` /
   `THESIS_ALIGNED_GRAY` (and thresholds if they change) at the new files, regenerate
   `tier1_results.json` + `temporal_results.json` + `results_noreject/` + `results_dut/` + `results_clean/`,
   and `eval/results/filter_operating_sweep.json`.
2. **Update audit CLAIMED constants** in `_audit_headline_numbers.py` for EVERY filter cell:
   `rgbconf mlp/patch fire+FP`, `grayconf mlp fire+FP`, `irconf composed`, all `filt_mlp_rgb/ir`,
   `rgbtest/selcom/ir_dset filt_mlp/filt_patch F1`, every `clf->filt[*]` / `filt->clf[*]` row,
   the `FIG rgb/gray recall@*` sweep cells, and the CBAM `_CBAM_CANON_FP` regex pair.
3. **Update CITED_PATHS** to the new weight files (and remove old ones if archived).
4. **Regenerate figures** via `docs/generate_thesis_figures.py` (fig6_1/6_2/6_3, fig8_distill_verifier,
   fig8_patch_catchbar, fig8_cascade_*, fig8_surface_exchange, fig8_perframe_segment) and
   `eval/filter_operating_sweep.py` (fig_filter_operating); regenerate the MRI figures
   (fig8_mri_ir_*, fig9_ir_gray_align, fig:failopen*) via the mri/ tooling.
5. **Record** new `models` rows (provenance) + new `evals` rows + a `ledger` finding via `kb.py`
   (no hand-editing CSVs), then update the `% [source: ...]` trailers in the .tex.
6. **Edit the right .tex.** The audit-enforced tables live in the overleaf chapters
   (`methodology.tex` + `empirical.tex`); `thesis_working.tex` (assigned primary) is a behind-snapshot
   and must be brought in line OR the swap retargeted to the overleaf chapters per the CLAUDE skill rule.

---

## Totals (`docs/thesis_working.tex`, the assigned primary)

- **Filter-dependent FIGURES: 19** (12 plotted via resident scripts, 4 MRI-tool figures, 3 placeholders/qualitative).
- **Filter-dependent TABLES: 14** (2 are textual/contextual).
- **Filter weight files: 5** (3 production + robust8-nr router + patch-v2 predecessor).
- **Audit-pinned result JSONs: ~10 families** (tier1/temporal/noreject/dut/clean/filter-sweep/cumulative/mri).
- **Generating scripts: ~12** (1 core harness, 1 figure pack, plus per-table eval scripts + mri/).
- **Ledger rows: 15.**
- Plus a **separate block of `filt`/`clf->filt`/`filt->clf` cells + `fig:filter_operating` + DUT
  ablation that exist ONLY in `docs/thesis_working_distilling_overleaf/chapters/*.tex`**, which the
  audit actually enforces (see §B2/§D).
- `docs/thesis_chapters.tex`: no `mlp_v5`/`robust8`/`filt` strings — superseded copy, no filter cells to map.

### Delivered
- `docs/analysis/2026-06-17_filter_thesis_map_figures.md` (this file) —
  `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-17_filter_thesis_map_figures.md`
