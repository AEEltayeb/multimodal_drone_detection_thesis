# Thesis script manifest — the scripts to include in the reproducibility appendix

**Date:** 2026-06-05. **Purpose:** the canonical list of scripts the thesis depends on, extracted from
every `% [source: run=…]` provenance comment in `docs/thesis_working.tex` and cross-checked against
`knowledge/scripts.csv`. **All 31 are now recorded** (3 were missing — `_veto_vs_confuser`,
`overnight_confuser_distill`, `visualize_ir_activation` — now added). This is the set to ship in the
thesis `app:reproducibility` appendix (parallel to `app:datasets`).

> Source of truth is `knowledge/scripts.csv` (id ↔ path ↔ purpose ↔ produces_evals). This doc is the
> thesis-facing grouping; regenerate the cited set with the extractor in the Delivered section.

---

## Detector training (3)
| script | kb id | role |
|---|---|---|
| `RGB model/finetune_selcom.py` | `rgb_finetune_selcom` | canonical — SelCom+confuser fine-tunes (→ ft4) |
| `scripts/auto_confuser_ft4.py` | `scripts_auto_confuser_ft4` | canonical — confuser hard-neg mining for ft4 |
| `eval/retrain_v5_targeted.py` | `retrain_v5_targeted` | canonical — targeted V5 retrain |

## Trust classifier — feature selection + routing (6)
| `classifier/fusion_feature_stats.py` | `fusion_feature_stats` | canonical — leakage stat + LDA/PCA/AUROC (fig8_fusion_*) |
| `classifier/train_lean_ft4.py` | `train_lean_ft4` | canonical — robust6 lean-feature trainer |
| `classifier/forward_select_routing.py` | `forward_select_routing` | canonical — statistics-gated forward selection (→ robust8) |
| `classifier/verifier_as_feature_stats.py` | `verifier_as_feature_stats` | canonical — verifier-P(drone)-as-feature AUROC |
| `classifier/vet_blurriness_stats.py` | `vet_blurriness_stats` | canonical — within-source AUROC artifact gate |
| `eval/compare_routing_pipeline.py` | `compare_routing_pipeline` | canonical — full-pipeline robust8 vs robust6 vs sa32 |

## Confuser verifier — distillation (4)
| `eval/overnight_confuser_distill.py` | `overnight_confuser_distill` | canonical — YOLO-neck embedding distillation (mlp_v5) |
| `eval/eval_v4_vs_patch.py` | `eval_eval_v4_vs_patch` | canonical — mlp_v5 vs patch verifier |
| `eval/diagnose_mlp_recall_drop.py` | `diagnose_mlp_recall_drop` | canonical — rgb_dataset recall-drop diagnosis |
| `eval/_veto_vs_confuser.py` | `veto_vs_confuser` | one-off — verifier veto behaviour on confusers |

## Evaluation harness + ablations (9)
| `eval/ablate.py` | `eval_ablate` | canonical — the May-10 ablation matrix (B/C/D/E/H) |
| `eval/overnight_ablation.py` | `overnight_ablation` | canonical — full-pipeline alert ablation (tab:robust6_pipeline) |
| `eval/cumulative_halluc.py` | `eval_cumulative_halluc` | canonical — cumulative confuser suppression (Ch6 headline) |
| `eval/eval_classifier_3way.py` | `eval_eval_classifier_3way` | canonical — 3-way drone/confuser/bg eval |
| `eval/temporal_ablation.py` | `temporal_ablation` | canonical — N-of-M temporal smoother ablation |
| `eval/bench_speed.py` | `bench_speed` | canonical — per-frame/per-det latency (robust8 ~400×, mlp 46–72×) |
| `eval/run_roboflow_eval.py` | `eval_run_roboflow_eval` | canonical — 9-set Roboflow OOD audit |
| `eval/eval_video_tests.py` | `eval_eval_video_tests` | canonical — real-video diagnostic + grayscale transfer |
| `eval/ir_version_comparison.py` | `eval_ir_version_comparison` | canonical — IR HITL V2→Final evolution table |

## Fail-open verifier analysis (3)
| `eval/eval_failopen_prepost.py` · `eval/failopen_expanded_ref.py` · `eval/test_failopen_verifier.py` | `eval_failopen_prepost`, `failopen_expanded_ref`, `test_failopen_verifier` | canonical — fail-open viability (fig8_failopen_*) |

## Failure diagnosis (2)
| `eval/diagnose_failures.py` · `eval/diagnose_failures_all.py` | `eval_diagnose_failures`, `eval_diagnose_failures_all` | canonical — per-category Svanström failure breakdown |

## Model MRI + activation visualisation (4)
| `mri/cli.py` | `mri_cli` | canonical — Model MRI entrypoint (LDA/ANOVA/AUROC/CORAL) |
| `mri/modality_align.py` · `mri/modality_probe.py` | `mri_modality_align`, `mri_modality_probe` | recorded — cross-modal alignment probes |
| `scripts/visualize_ir_activation.py` | `visualize_ir_activation` | canonical — IR P3/P5 activation panels (fig8_mri_ir_act_*) |

---

## Notes for the appendix
- **Figure generators among the above** (`fusion_feature_stats`, `visualize_ir_activation`, the `failopen_*`
  scripts, `mri/*`) produce the **18 currently-orphan figures** — they save under non-`fig8_*` names, which is
  the open figure-rename-map item (backlog task 7). The appendix should pair each figure with its generator.
- **`docs/generate_thesis_figures.py`** (`thesis_figures_gen`) produces the other 7 figures and is itself
  worth listing as the figure-assembly entrypoint.
- The trust-classifier line is the **freshest** (robust8, June 2026); the eval harness (`ablate`,
  `overnight_ablation`, `cumulative_halluc`) is the **load-bearing reproducibility core**.

## Delivered
- `docs/analysis/2026-06-05_thesis_script_manifest.md` (this file) — 31 thesis-cited scripts grouped by role,
  all now recorded in `scripts.csv`.
- Recorded 3 previously-missing scripts: `veto_vs_confuser`, `overnight_confuser_distill`, `visualize_ir_activation`.
- **Re-extract command:** `grep -o 'run=[^;]*\.py' docs/thesis_working.tex` → dedupe → cross-ref `scripts.csv`.

### Follow-up
- Build the actual `app:reproducibility` appendix section in `thesis_working.tex` from this list (pairs with `app:datasets`).
- Pair each orphan figure with its generator here once the rename-map (task 7) is done.
