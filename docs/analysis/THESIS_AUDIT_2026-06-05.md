# Thesis audit — thesis_working.tex (2026-06-05)

_Deterministic checks (numbers + figures). Qualitative claim verdicts are produced by the `thesis` skill's agent pass, recorded in `knowledge/claims.csv`._

## Numbers vs knowledge/evals.csv
- result-like numbers scanned: **1183**
- matched to an eval value: **1180**
- **not found (3)** — verify (stale/copied, OR derived/cited/aggregate):

| value | line | context |
|---|---|---|
| 27.5\% | 469 | The IR detector's training and in-distribution test corpus. 129{,}130 thermal frames split |
| 27.5\% | 491 | \textbf{Total}  & \textbf{129{,}130} & 83.5 / 9.1 / 7.4 split; 72.5\% positive / 27.5\% ne |
| 37.2\% | 1763 | The same cross-modal equivalence that the feature-space analysis established (\S\ref{sec:i |

## Figures (\includegraphics)

| tex_path | exists | generated_by | status |
|---|---|---|---|
| figures/unisa-logo.png | False | (institutional) | verified |
| fig6_6_resolution | True | thesis_figures_gen | verified |
| fig8_rgb_threestance | True | (unknown) | orphan |
| fig9_ir_v3b_lda | True | (unknown) | orphan |
| fig9_ir_v3b_anova | True | (unknown) | orphan |
| fig9_ir_v3b_heatmap | True | (unknown) | orphan |
| fig8_mri_ir_act_drone | True | (unknown) | orphan |
| fig8_mri_ir_act_confuser | True | (unknown) | orphan |
| fig9_ir_gray_align | True | (unknown) | orphan |
| fig7_1_ood_classifier | True | thesis_figures_gen | verified |
| fig8_fusion_lda | True | (unknown) | orphan |
| fig8_fusion_pca | True | (unknown) | orphan |
| fig8_fusion_auroc | True | (unknown) | orphan |
| fig8_fusion_leakage | True | (unknown) | orphan |
| fig6_3_threshold_sweep | True | thesis_figures_gen | verified |
| fig8_patch_catchbar | True | (unknown) | orphan |
| fig8_mri_lda | True | (unknown) | orphan |
| fig8_mri_anova | True | (unknown) | orphan |
| fig8_distill_verifier | True | (unknown) | orphan |
| fig8_mri_act_drone | True | (unknown) | orphan |
| fig8_mri_act_confuser | True | (unknown) | orphan |
| fig8_failopen_expanded | True | (unknown) | orphan |
| fig8_failopen_hist | True | (unknown) | orphan |
| fig8_failopen_tradeoff | True | (unknown) | orphan |
| fig8_failopen_pca | True | (unknown) | orphan |
| fig4_ir_evolution | True | thesis_figures_gen | verified |
| fig6_1_cumulative_confuser | True | thesis_figures_gen | verified |
| fig6_2_svanstrom_by_category | True | thesis_figures_gen | verified |
| fig8_surface_exchange | True | (unknown) | orphan |
| fig5_realvideo_pareto | True | thesis_figures_gen | verified |
| fig8_perframe_segment | True | (unknown) | orphan |
| fig8_cascade_segment | True | (unknown) | orphan |
| fig8_classifier_reversal | True | (unknown) | orphan |

_figures.csv updated (33 rows)._
