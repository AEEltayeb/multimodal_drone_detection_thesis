# 2026-06-07 — Thesis Figure Audit (per-figure keep + correctness)

**File audited:** `docs/thesis_working_distilling.tex` (37 figures across 28 floats).
**Method:** 9 parallel cluster-auditor agents. Each read every figure PNG, read the ±40-line
context (caption + the claim it backs), and verified the figure's numbers against
`knowledge/{ledger,evals,models}.csv`. Each figure scored **keep 0–10** (importance to the
argument) and **correct 0–10** (accurately depicts an evidence-backed result).

## Result: 26 keep, 11 delete (4 of the keeps need a small correctness fix)

### DELETE (11) — redundant or low-keep
| figure | keep | correct | why | redundant with |
|---|---|---|---|---|
| fig9_ir_v3b_lda | 5 | 9 | IR-MRI LDA duplicates RGB-distill LDA (same format, same moral) | fig8_mri_lda |
| fig9_ir_v3b_anova | 4 | 8 | IR-MRI ANOVA duplicates RGB-distill ANOVA | fig8_mri_anova |
| fig9_ir_v3b_heatmap | 6 | 8 | IR feature heatmap; minor add over LDA, cluster being cut | fig9_ir_v3b_lda |
| fig8_mri_ir_act_drone | 5 | 9 | IR activation pair duplicates RGB activation pair | fig8_mri_act_drone |
| fig8_mri_ir_act_confuser | 5 | 9 | (same pair) | fig8_mri_act_confuser |
| fig8_fusion_pca | 4 | 8 | "variance≠signal" already in prose; LDA makes it stronger | fig8_fusion_lda |
| fig8_failopen_hist | 3 | 9 | separability already in tab:failopen | fig8_failopen_tradeoff |
| fig8_failopen_tradeoff | 4 | 9 | recovery-vs-leak curve duplicates the 3-row table | tab:failopen |
| fig8_failopen_pca | 2 | 8 | decorative; OOD claim made in text | fig8_failopen_hist |
| fig8_perframe_segment | 6 | 9 | per-frame dip is a narrative step in the cascade timeline | fig8_cascade_segment (merge) |
| fig8_rgb_threestance | 6 | 9 | same data as tab:rgb_comparison already in text | tab:rgb_comparison (merge) |

**Convergent finding:** both MRI clusters say the same thing in the same visual language. Keep ONE
canonical MRI demonstration (the RGB-distill verifier: LDA + ANOVA + activation pair), delete the
entire IR-MRI mirror (5 figures), and carry the IR-separability point in one sentence + the
load-bearing `fig9_ir_gray_align` (cross-modal alignment) which is NOT redundant.

### FIX (keep, but correct an error caught in the audit)
| figure | issue | fix |
|---|---|---|
| fig_drone_size_hist | legend shows raw LaTeX `28\,px` literal | fix string in `gen_drone_size_hist.py`, re-render |
| fig6_6_resolution | bar label R=0.959 vs ledger/text R=0.961 | correct value in generator, re-render |
| fig4_ir_evolution | V3 precision 0.636 in figure vs 0.648 in companion table | reconcile against cache CSV, fix whichever is wrong |
| fig6_3_threshold_sweep | curve omits τ=0.5 (the worst-recall / max-FP-cut point) | add τ=0.5 datum or note the omission in caption |

### KEEP (22, clean) — load-bearing, correctness verified
fig_dataset_montage, fig_confuser_panel, fig5_realvideo_pareto, fig9_ir_gray_align (10/10 — all 4
AUROC match ledger), fig_grayscale_panel, fig7_1_ood_classifier (10/10), fig8_fusion_lda,
fig8_fusion_auroc, fig8_fusion_leakage (the novel feature-leakage statistic), fig_robust8_operating_point,
fig8_patch_catchbar, fig8_distill_verifier (10/9 — the production verifier), fig8_mri_lda, fig8_mri_anova,
fig8_mri_act_drone, fig8_mri_act_confuser, fig8_classifier_reversal, fig8_failopen_expanded,
fig6_1_cumulative_confuser, fig6_2_svanstrom_by_category, fig8_surface_exchange, fig8_cascade_segment.

Minor caption tweaks (not blocking): fig_robust8_operating_point (clarify the shipped model is the
green augmented curve), fig6_2 (note fnfn, not sa32, is the production classifier).

## Page-budget reality (159 → 120 = −39 pages)
- **Figure deletions ≈ −3 pages.** Most deletions are subfigure floats; removing 11 figures (several
  whole floats) recovers ~2.5–3 pages — real, but not the main lever.
- **Relocating tables to an appendix does NOT cut pages** — appendix pages still count. The saving
  comes from **deleting the surrounding prose discussion**, not moving the table.
- **The −39 must come from content compression**, in priority order:
  1. Superseded **patch-verifier** section → compress to ~½ page + dense appendix table.
  2. **MRI prose** (author flagged over-indexing) → tighten to the load-bearing claims.
  3. Secondary **ablation tables + their prose** → one dense appendix (`app:ablations`), drop per-row narration.
  4. **9→5 IMRAD consolidation** → removes chapter-boundary overhead and redundant transitions.
  5. Line-level prose tightening throughout.

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\docs\analysis\2026-06-07_figure_audit.md` (this file)
- Pending: edits to `docs\thesis_working_distilling.tex` (11 deletions + 4 fixes) and the compression passes above.
