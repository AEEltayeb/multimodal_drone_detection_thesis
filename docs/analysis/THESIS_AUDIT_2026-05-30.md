# Thesis audit — thesis_working.tex (2026-05-30)

_Deterministic checks (numbers + figures). Qualitative claim verdicts are produced by the `thesis` skill's agent pass, recorded in `knowledge/claims.csv`._

## Numbers vs knowledge/evals.csv
- result-like numbers scanned: **795**
- matched to an eval value: **792**
- **not found (3)** — verify (stale/copied, OR derived/cited/aggregate):

| value | line | context |
|---|---|---|
| 27.5\% | 884 | The IR detector's training and in-distribution test corpus. 129{,}130 thermal frames split |
| 27.5\% | 905 | \textbf{Total}  & \textbf{129{,}130} & 83.5 / 9.1 / 7.4 split; 72.5\% positive / 27.5\% ne |
| 50.5\% | 1135 | bird       & 50.5\% & 51.8\% & 52 & verifier transfers well to OOD birds \\ |

## Figures (\includegraphics)

| tex_path | exists | generated_by | status |
|---|---|---|---|
| figures/unisa-logo.png | False | (unknown) | stale |
| fig6_6_resolution | True | (unknown) | orphan |
| fig7_1_ood_classifier | True | (unknown) | orphan |
| fig6_3_threshold_sweep | True | (unknown) | orphan |
| fig4_ir_evolution | True | (unknown) | orphan |
| fig6_1_cumulative_confuser | True | (unknown) | orphan |
| fig6_2_svanstrom_by_category | True | (unknown) | orphan |
| fig5_realvideo_pareto | True | (unknown) | orphan |
| fig5_cascade_percategory | True | (unknown) | orphan |

_figures.csv updated (9 rows)._
