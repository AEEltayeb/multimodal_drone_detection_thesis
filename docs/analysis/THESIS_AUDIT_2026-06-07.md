# Thesis audit — thesis_working_distilling.tex (2026-06-07)

_Deterministic checks (numbers + figures). Qualitative claim verdicts are produced by the `thesis` skill's agent pass, recorded in `knowledge/claims.csv`._

## Numbers vs knowledge/evals.csv
- result-like numbers scanned: **1093**
- matched to an eval value: **1090**
- **not found (3)** — verify (stale/copied, OR derived/cited/aggregate):

| value | line | context |
|---|---|---|
| 27.5\% | 465 | 129{,}130 thermal frames split 83.5 / 9.1 / 7.4 (107{,}809 train, 11{,}709 val, 9{,}612 te |
| 27.5\% | 487 | \textbf{Total}  & \textbf{129{,}130} & 83.5 / 9.1 / 7.4 split; 72.5\% positive / 27.5\% ne |
| 37.2\% | 1689 | The cross-modal equivalence established in \S\ref{sec:ir_xmodal_verifier} has a deployment |

## Figures (\includegraphics)

| tex_path | exists | generated_by | status |
|---|---|---|---|
| figures/unisa-logo.png | False | (institutional) | verified |
| fig_dataset_montage | True | (unknown) | orphan |
| fig_drone_size_hist | True | (unknown) | orphan |
| fig_confuser_panel | True | (unknown) | orphan |
| fig6_6_resolution | True | thesis_figures_gen | verified |
| fig9_ir_gray_align | True | (unknown) | orphan |
| fig7_1_ood_classifier | True | thesis_figures_gen | verified |
| fig8_fusion_lda | True | (unknown) | orphan |
| fig8_fusion_auroc | True | (unknown) | orphan |
| fig8_fusion_leakage | True | (unknown) | orphan |
| fig_robust8_operating_point | True | (unknown) | orphan |
| fig6_3_threshold_sweep | True | thesis_figures_gen | verified |
| fig8_patch_catchbar | True | (unknown) | orphan |
| fig8_mri_lda | True | (unknown) | orphan |
| fig8_mri_anova | True | (unknown) | orphan |
| fig8_distill_verifier | True | (unknown) | orphan |
| fig8_mri_act_drone | True | (unknown) | orphan |
| fig8_mri_act_confuser | True | (unknown) | orphan |
| fig8_failopen_expanded | True | (unknown) | orphan |
| fig4_ir_evolution | True | thesis_figures_gen | verified |
| fig6_1_cumulative_confuser | True | thesis_figures_gen | verified |
| fig6_2_svanstrom_by_category | True | thesis_figures_gen | verified |
| fig8_surface_exchange | True | (unknown) | orphan |
| fig5_realvideo_pareto | True | thesis_figures_gen | verified |
| fig_grayscale_panel | True | (unknown) | orphan |
| fig8_cascade_segment | True | (unknown) | orphan |
| fig8_classifier_reversal | True | (unknown) | orphan |

_figures.csv updated (27 rows)._
