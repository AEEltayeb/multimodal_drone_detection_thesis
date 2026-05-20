# Word Document Audit — What's Worth Adding

**Date**: 2026-05-17  
**Auditor**: automated cross-reference against `EVIDENCE_LEDGER.md`, `thesis_chapters.tex`, and existing `docs/analysis/` files.  
**Sources scanned**: 9 `.docx` files on Desktop (extracted to `docs/analysis/_docx_extracts/`).

---

## Executive Summary

The Word documents are **conversation transcripts and AI-generated analysis reports** from the iterative development of the drone detection pipeline (Mar–May 2026). They capture design rationale, ablation results, and intermediate metrics at various pipeline stages. Most of the _quantitative claims_ are already superseded by the current evidence ledger or have been folded into thesis_chapters.tex — but several contain **unique provenance-backed data, design reasoning, and historical metrics** that are missing from both.

### Verdict at a Glance

| Document | New for Ledger? | New for Thesis? | Provenance quality | Action |
|---|---|---|---|---|
| `architecture analysis` | ❌ No | ⚠️ Partial | Low — no source files, no commands | Use for **narrative** only (Ch3 design rationale prose) |
| `classifier and confuser filter` | ✅ **Yes** | ✅ **Yes** | **High** — paths, metrics.json, sweep tables | Add to ledger + thesis Ch4 |
| `drone classifier no tautological features (c++)` | ⚠️ Partial | ✅ **Yes** | **High** — importance table, ablation logic | Add to thesis Ch4 (tautology ablation) |
| `drone classifier` | ❌ No | ⚠️ Partial | Medium — numbers exist but superseded by IoP fix | Keep as historical reference |
| `Drone detection classifier gpt` | ❌ No | ❌ No | N/A — design advice, no empirical data | Skip |
| `pipeline overview, old vs new` | ✅ **Yes** | ✅ **Yes** | **High** — CSV paths, per-table sources | Add OLD→NEW delta to ledger |
| `presentation` | ❌ No | ❌ No | Low — fragment of IR DsetV4 size breakdown | Skip (one tiny table, no context) |
| `rgb v3_more conversation` | ⚠️ Partial | ⚠️ Partial | Medium — sweep results exist in conf_sweep.json | Useful for F1-optimal conf narrative |
| `six config eval` | ❌ No | ⚠️ Partial | Medium — superseded by `six eval config + youtube OOD` | Skip (duplicate) |
| `six eval config + youtube OOD` | ✅ **Yes** | ✅ **Yes** | **High** — per-video IR OOD table, scoped GT | Add YouTube IR OOD to ledger + thesis |

---

## Document-by-Document Analysis

### 1. `architecture analysis.docx` (Apr 24, 5 KB)

**Content**: An AI-generated architecture recommendation memo. Walks through a decision tree: single modality enough? → No. Fusion helps? → Yes. Filter helps? → Yes (but only OOD). Order? → classifier→filter wins. Includes an ASCII pipeline diagram.

**Already in thesis?** Yes — this reasoning is faithfully encoded in Ch3 §3.1 (Overview), §3.2 (Design Rationale: Fail-Open), and §3.4 (Alert-Gate Cascade). The specific numbers cited (F1=0.9916, F1=0.9937, heli 40.5%→7.6%) are from the OLD pipeline (pre-v3more), now superseded by current production numbers.

**Provenance**: ❌ No source files, no commands, no CSVs. This is analysis prose, not data.

**Verdict**: **Skip for ledger**. Already absorbed into thesis architecture chapter. Could mine one good quote: "You already built the right system… Don't change the architecture; sharpen the narrative."

---

### 2. `classifier and confuser filter.docx` (May 3, 44 KB) ⭐

**Content**: The most valuable document. A comprehensive technical report covering:
- Trust Classifier architecture, hyperparameters, training data composition, metrics
- 32-feature list with exact importance values (full table)
- Per-class AUC: reject_both=0.9984, trust_rgb=0.9792, trust_ir=0.9921, trust_both=0.9913
- Per-dataset accuracy: antiuav_test 0.983, antiuav_val 0.974, svanstrom 0.973
- Rule-based baselines comparison (always_ir 0.193 acc, higher_conf 0.940, scene_aware 0.979)
- Training data sizes: n_train=106,963, n_test=45,088
- Top features by importance: rgb_best_log_bbox_area 0.240, ir_best_log_bbox_area 0.184, rgb_max_conf 0.130
- Patch Verifier v2 architecture, training manifest (RGB: 23,550 crops, IR: 21,050 crops)
- Per-class val metrics: RGB verifier best_val_acc=0.978; IR verifier best_val_acc=0.938
- Reject-rule sweep tables for both RGB and IR verifiers
- GlobalFeatureCache strided caching benchmarks (stride=5 costs -0.34pp accuracy)
- Mahalanobis OOD gate thresholds (RGB 47.70, IR 49.02 at p99)
- Complete source dataset provenance table

**Already in ledger?** Partially. §5 has trust classifier pipeline-level F1s but NOT the per-class AUCs, NOT the feature importance table, NOT the training data sizes, NOT the rule-based baselines, NOT the verifier val-set metrics. §6 has the patch catch audit but NOT the verifier training metrics.

**Already in thesis?** Partially. Ch4 §4.3 (Trust Classifier) mentions 40/32 features and the XGBoost architecture but does NOT include:
- The full 32-feature importance table (only mentioned in passing)
- Per-class AUC numbers
- Rule-based baseline comparison
- Training data sizes (106K/45K)
- Patch verifier training metrics (val accuracy, per-class precision/recall on val)
- Feature caching benchmarks

**Provenance**: ✅ High quality.
- `classifier/fusion_models/scene_aware_v3more_32feat/metrics.json` — verifiable
- `classifier/runs/patches/confuser_filter4_rgb_metrics.json` / `confuser_filter4_ir_metrics.json` — verifiable
- `classifier/runs/patches/manifest.csv` — verifiable
- `classifier/bench_feature_cache.py` — script exists, reproducible

**Recommended additions**:

#### → EVIDENCE_LEDGER.md

Add a new **§5.1 Trust classifier internal metrics** section:

| Metric | Value | Source |
|---|---|---|
| accuracy | 0.9792 | `classifier/fusion_models/scene_aware_v3more_32feat/metrics.json` |
| F1 (macro) | 0.9493 | same |
| F1 (weighted) | 0.9786 | same |
| AUC reject_both | 0.9984 | same |
| AUC trust_rgb | 0.9792 | same |
| AUC trust_ir | 0.9921 | same |
| AUC trust_both | 0.9913 | same |
| n_train | 106,963 | same |
| n_test | 45,088 | same |

How to reproduce: `python classifier/reliability/fusion/train_fusion.py` (outputs metrics.json).

Add a new **§6.2 Patch verifier training metrics** section:

| Verifier | Val accuracy | Source |
|---|---|---|
| RGB v2 | 0.978 | `classifier/runs/patches/confuser_filter4_rgb_metrics.json` |
| IR v2 | 0.938 | `classifier/runs/patches/confuser_filter4_ir_metrics.json` |

Plus the per-class precision/recall and reject-rule sweep tables from the document.

#### → thesis_chapters.tex

- Add Table: rule-based baselines vs learned classifier (always_ir 0.193, higher_conf 0.940, scene_aware 0.979) — this is a killer comparison for the thesis
- Add feature importance group breakdown (box geometry 51%, confidence 22%, position 16%, image stats 8%, local contrast 3%) — currently only mentioned in prose
- Add patch verifier val accuracy (RGB 0.978, IR 0.938) and mention per-class performance
- Add strided-caching accuracy impact table (stride 5 → -0.34pp) to Ch4 or Ch5

---

### 3. `drone classifier no tautological features (c++).docx` (May 1, 18 KB)

**Content**: Explains the 32-vs-40 feature ablation. Key insight: the 8 detection-presence features (ir_detected, rgb_detected, etc.) are **near-tautological with the label** because the label is derived from per-modality TPs. XGBoost discovers this and builds a 35% importance shortcut on ir_detected. Removing these 8 features → same accuracy, but the classifier now reasons from visual features (box geometry + confidence = 73% of importance).

**Already in ledger?** No — the tautology ablation rationale is not in the ledger.

**Already in thesis?** Partially — §4.3.2 (Classifier Variants) mentions the scene_aware_v3more_32feat variant drops detection flags but does NOT explain the tautology argument.

**Provenance**: ✅ High — the importance table is verifiable from `metrics.json`, the feature list matches the 32 features in the document.

**Recommended additions**:

#### → thesis_chapters.tex

Add ~2 paragraphs to §4.3.2 explaining the tautological-feature discovery. This is thesis gold:
- "8 features are near-tautological with the label by construction"
- "XGBoost discovered this and built a 35%-importance shortcut on ir_detected"
- "The 32-feature model achieves the same accuracy because box geometry + confidence carry the real signal"
- "This is the case that justifies the 32-feature variant for defensible architecture"

Include the aggregated importance breakdown table:
| Group | Features | Importance |
|---|---|---|
| box geometry | 4 | 51% |
| confidence | 4 | 22% |
| position | 6 | 16% |
| image stats | 14 | 8% |
| local contrast | 4 | 3% |

---

### 4. `drone classifier.docx` (Apr 17, 37 KB)

**Content**: Two walkthroughs of the 10-approach fusion evaluation. Walkthrough 1 uses IoU only; Walkthrough 2 uses IoU+IoP. Contains the full ranking tables for all 10 fusion architectures.

**Already in ledger?** The per-approach ranking is NOT in the ledger. However, the top approaches (#04, #08, #10) are already the ones selected as production candidates. The 152K paired-frame number is historical context.

**Already in thesis?** Not directly — the thesis mentions the classifier's value but does not present the full 10-approach comparison. The "10 fusion architectures" claim appears in the thesis abstract/conclusion but the supporting table is missing.

**Provenance**: ⚠️ Medium — the document cites `fusion_no_fn_model.joblib` and `fusion_dataset.csv` but these are the OLD (pre-v3more) artifacts. The current production uses `scene_aware_v3more_32feat` or `control_v3more_40feat`. The IoU-only numbers in Walkthrough 1 are superseded by the IoP-corrected numbers in Walkthrough 2.

**Verdict**: **Do not add raw numbers to ledger** (they're against the OLD RGB model with the 40-feature classifier, pre-IoP fix, and the current production RGB is `Yolo26n_trained` baseline which was not the RGB used in this eval). **But**: the thesis could reference the 10-architecture survey as motivation for the final choice. The key data points worth citing:

- 10 architectures compared on 45K test frames
- Top-3 all achieve 99.7%+ detection precision
- FN models add zero value (unchanged conclusion across both walkthroughs)
- Consensus+Solo ML (#08) is competitive with full classifier (#10) — strong alternative for interpretability

These are historical/progress claims, not production claims.

---

### 5. `Drone detection classifier gpt.docx` (Mar 25, 20 KB)

**Content**: An early ChatGPT conversation about how to build a fusion classifier. Covers theory: late fusion, box matching, WBF, logistic regression → XGBoost progression, feature engineering advice.

**Already in?** The pipeline was built following this advice. No empirical data in this document.

**Provenance**: N/A — this is design advice, not results.

**Verdict**: **Skip entirely**. No numbers, no provenance, purely advisory. The advice has been implemented and validated by the actual experiments.

---

### 6. `pipeline overview, old vs new.docx` (May 2, 32 KB) ⭐

**Content**: The OLD→NEW pipeline ablation. Compares:
- OLD: Yolo26n_trained (baseline) + confuser_filter4_*_v1_backup + fusion_no_fn_model (40 feat)
- NEW: Yolo26n_hardneg_v3_more + confuser_filter4_* (v2) + scene_aware_v3more_32feat (32 feat)

Across Anti-UAV (IoU@0.5), Svanström (IoP@0.5), YouTube OOD IR (14 videos), YouTube OOD RGB (4,265 frames).

**Key results**:
- Anti-UAV classifier F1: OLD 0.9868 → NEW 0.9908 (+0.40pt)
- Svanström classifier→filter F1: OLD 0.9726 → NEW 0.9823 (+0.97pt)
- YouTube IR confuser suppression: OLD 53.0% → NEW 75.2% (+22.2pt)
- YouTube IR helicopter: 81.0% → 100.0%
- YouTube RGB confuser any-det rate: 56.5% → 12.9% (−43.5pt)
- DRONE_CLEAN regression: 7.5% → 23.8% suppression (acknowledged, mitigated by temporal smoothing)

**Already in ledger?** ❌ **Not in the ledger**. The ledger has the current production stack metrics but does NOT have the OLD→NEW delta. This is a progression/history claim.

**Already in thesis?** Not directly. The thesis mentions the v3more variant in §4.1.2 (Three Training Stances) but does NOT present the full OLD→NEW pipeline ablation.

**Provenance**: ✅ High — CSV paths listed (e.g., `ablation_old_vs_new/{old,new}/antiuav/metrics_scoped_iou.csv`), script named (`classifier/run_ablation.py`).

**Recommended additions**:

#### → EVIDENCE_LEDGER.md

Add a new **§3.5 OLD→NEW pipeline ablation (2026-05-02)** section with the headline delta table:

| Metric | OLD | NEW | Δ | Source |
|---|---|---|---|---|
| Anti-UAV classifier F1 (IoU) | 0.9868 | 0.9908 | +0.40pt | `ablation_old_vs_new/{old,new}/antiuav/metrics_scoped_iou.csv` |
| Svanström classifier→filter F1 (IoP) | 0.9726 | 0.9823 | +0.97pt | `ablation_old_vs_new/{old,new}/svanstrom/metrics_scoped_iop.csv` |
| YouTube IR confuser suppression | 53.0% | 75.2% | +22.2pt | `ablation_old_vs_new/{old,new}/youtube_ir/category_summary.csv` |
| YouTube RGB any-det rate | 56.5% | 12.9% | −43.5pt | `eval_youtube_rgb/summary.json` |

How to reproduce: `python classifier/run_ablation.py` with the two component sets.

Status: `superseded` (neither OLD nor NEW is the current production stack — baseline RGB replaced v3_more, and the classifier may be control_v3more_40feat).

**Note**: This is **historical/progress** data showing improvement trajectory. The current production stack is different from both OLD and NEW.

#### → thesis_chapters.tex

This could support a "Pipeline Evolution" subsection or paragraph in Ch5, showing that the transition from v1 to v2 components yielded measurable improvement across all axes. But since neither OLD nor NEW matches the current production stack, it should be presented as historical evidence of the HITL-driven improvement cycle, not as production claims.

---

### 7. `presentation.docx` (Mar 17, 774 KB — mostly images)

**Content**: A fragment — just a tiny IR DsetV4 size breakdown table (P/R by tiny/medium/large at T=0.17).

**Provenance**: ❌ No source, no commands. The numbers are from an early IR dataset version that is now superseded by IR_dset_final.

**Verdict**: **Skip**. The V4 numbers are already captured (superseded) in ledger §4.1 with the proper test-split re-evaluation.

---

### 8. `rgb v3_more conversation.docx` (May 1, 25 KB)

**Content**: A conversation transcript about the v3_more RGB YOLO improvement over the OLD (baseline) RGB YOLO. Contains:
- Per-dataset comparison tables (Anti-UAV, dataset_rgb, Svanström) at fixed conf=0.25 and at F1-optimal conf
- F1-optimal conf sweep results (v3_more optimal is 0.40–0.45 vs OLD's 0.30–0.35)
- Headline: v3_more trades ~1pt drone F1 for 15pt confuser reduction and ~5000 fewer FPs on Svanström

**Already in ledger?** ⚠️ Partially — the Svanström@1280 per-category breakdown is in §3.1 for three variants (baseline, hardneg_v3more, retrained_v2). But the **F1-optimal conf sweep** is NOT in the ledger.

**Already in thesis?** The three-variant comparison is in §4.1.2 (Table 4.1). The F1-optimal conf sweep is NOT in the thesis.

**Provenance**: ✅ Medium — references `classifier/runs/rgb_finetune_eval/comparison.json` and `classifier/runs/rgb_finetune_eval/conf_sweep.json`. The sweep was generated by `classifier/sweep_rgb_optimal_conf.py`.

**Recommended additions**:

#### → EVIDENCE_LEDGER.md

The ledger §1 (Production stack) has "RGB conf: TBD via conf sweep with baseline RGB". This document has sweep data but for the **v3_more** model, not the baseline. Still, the optimal conf finding (v3_more optimal = 0.40–0.45) is useful context. Could add a note:

> **Note (conf sweep history):** v3_more's F1-optimal conf on Anti-UAV is 0.40–0.45 (IoU) vs baseline's 0.30–0.35. At F1-optimal conf, the drone F1 gap between v3_more and baseline shrinks by 25–37% vs the gap at fixed conf=0.25. Source: `classifier/runs/rgb_finetune_eval/conf_sweep.json`. Status: `superseded` (v3_more is no longer the production RGB model).

#### → thesis_chapters.tex

The confidence-threshold discussion could strengthen §4.1.2 (Three Training Stances) — but since the current production RGB is the baseline (not v3_more), this is historical context at best. A footnote or a sentence: "A confidence sweep confirmed that the drone-F1 gap between variants shrinks by 25–37% when each is evaluated at its own F1-optimal threshold."

---

### 9. `six eval config + youtube OOD.docx` (Apr 24, 40 KB) ⭐

**Content**: The most complete evaluation document. Extends the six-config eval with:
- Anti-UAV IoU and IoP tables (7 configs including filter→classifier and classifier→filter)
- Svanström IoU and IoP tables (7 configs)
- Per-category FP breakdowns
- YouTube OOD IR filter test (14 thermal videos, per-video table)
- Filter classification accuracy analysis (within-confuser misclassification doesn't affect veto)
- **Scoped GT explanation** — the correct methodology for scoring classifier configs

**Key unique data**:
- YouTube IR per-video results (Table 10): yt_gg0Da0AtWJk airplane 89.1%→28.7% (67.8% suppression), yt_EdOX8tJZDzw helicopter 40.5%→7.6% (81.3% suppression), yt_zFu7hAi5mIc drone clean 89.0%→80.4% (9.6% loss)
- **Classifier with scoped GT**: F1=0.9937 on Svanström IoP (not the unscoped 0.8948)
- **classifier→filter ordering**: Svanström IoP F1=0.9747 with only 190 FPs — the correct ordering justified
- 7-way config comparison including both filter orderings

**Already in ledger?** ⚠️ Partially — the ledger has the May 10 ablation numbers but uses a different pipeline (post-v3more). The OLD pipeline's six-config eval is NOT in the ledger as a historical entry. The YouTube OOD IR data (per-video) is mentioned in the architecture analysis doc but NOT in the ledger with per-video provenance.

**Already in thesis?** The scoped GT methodology is mentioned in §5.5 (Svanström Usage Audit). The YouTube OOD results are partially captured in the §5.6 (Cumulative Confuser Suppression) and the architecture analysis doc's reasoning. BUT:
- The per-video YouTube IR table is NOT in the thesis
- The 7-way config comparison (with both filter orderings) is NOT presented
- The scoped GT's impact (0.8948→0.9937 on classifier) is only implicitly covered

**Provenance**: ✅ High — `eval_six_configs{,_v3more_32feat}/{antiuav,svanstrom}/per_det.jsonl` paths, script implied.

**Recommended additions**:

#### → EVIDENCE_LEDGER.md

Add a **§7.1 YouTube OOD IR filter test (OLD pipeline, 2026-04-24)** section:

| Video | Category | Frames | ir_only det% | ir_filter det% | Suppression | Source |
|---|---|---|---|---|---|---|
| yt_EdOX8tJZDzw | HELICOPTER | 422 | 40.5% | 7.6% | 81.3% | `eval_six_configs/youtube_ir/youtube_per_video.csv` |
| yt_zFu7hAi5mIc | DRONE (CLEAN) | 828 | 89.0% | 80.4% | 9.6% | same |
| ALL CONFUSERS | — | 4,993 | 10.2% | 4.6% | 54.6% | same |

Status: `superseded` (v1 verifier, OLD pipeline). The per-video data still demonstrates the OOD failure mode that motivates the filter.

Reproduce: `python eval/eval_six_configs.py --youtube-ir` (with OLD component set).

#### → thesis_chapters.tex

The per-video YouTube IR data would strengthen Ch5's OOD sections. Currently the thesis references YouTube OOD only via the architecture analysis reasoning and the Roboflow audit. A table or figure showing the per-video helicopter 40.5%→7.6% suppression and drone 89.0%→80.4% preservation would be a compelling visual. However, note these are OLD pipeline numbers — either use them as historical context or re-run with the current pipeline.

---

## Summary of Recommended Actions

### For EVIDENCE_LEDGER.md

| New Section | Source Document | Data Type | Status to assign |
|---|---|---|---|
| §5.1 Trust classifier internal metrics | `classifier and confuser filter.docx` | per-class AUC, baselines, training sizes | `current` (same model) |
| §6.2 Patch verifier val metrics | `classifier and confuser filter.docx` | per-class P/R, reject-rule sweep | `current` (v2 weights) |
| §3.5 OLD→NEW pipeline ablation | `pipeline overview, old vs new.docx` | anti-UAV/svanström/YouTube deltas | `superseded` (historical) |
| §7.1 YouTube OOD IR per-video | `six eval config + youtube OOD.docx` | per-video det/filter rates | `superseded` (historical) |
| Note on conf sweep | `rgb v3_more conversation.docx` | v3_more F1-optimal conf 0.40–0.45 | `superseded` (historical) |

### For thesis_chapters.tex

| Addition | Location | Source Document | Priority |
|---|---|---|---|
| Rule-based baselines vs learned classifier table | Ch4 §4.3.2 | `classifier and confuser filter.docx` | **HIGH** — this is a thesis argument |
| Feature importance by group table | Ch4 §4.3.1 | `drone classifier no tautological features (c++).docx` | **HIGH** — explains 32-feature choice |
| Tautological feature ablation (2 paragraphs) | Ch4 §4.3.2 | `drone classifier no tautological features (c++).docx` | **HIGH** — thesis defense material |
| Patch verifier val accuracy (RGB 0.978, IR 0.938) | Ch4 §4.4.1 | `classifier and confuser filter.docx` | MEDIUM |
| Feature caching accuracy impact | Ch4 §4.3 or Ch5 | `classifier and confuser filter.docx` | LOW — engineering detail |
| 10-architecture survey as motivation | Ch3 or Ch5 | `drone classifier.docx` | MEDIUM — historical context |
| OLD→NEW pipeline improvement trajectory | Ch5 future-work/history | `pipeline overview, old vs new.docx` | LOW — superseded |

### NOT recommended for addition

| Document | Reason |
|---|---|
| `Drone detection classifier gpt.docx` | Design advice, no empirical data |
| `presentation.docx` | Fragment, superseded by IR version comparison |
| `six config eval.docx` | Duplicate of `six eval config + youtube OOD.docx` (lacks YouTube section) |

---

## Provenance Gaps to Close

Before adding anything from these documents to the ledger, verify these files still exist:

1. `classifier/fusion_models/scene_aware_v3more_32feat/metrics.json` — trust classifier metrics
2. `classifier/runs/patches/confuser_filter4_rgb_metrics.json` — RGB verifier metrics
3. `classifier/runs/patches/confuser_filter4_ir_metrics.json` — IR verifier metrics
4. `classifier/runs/patches/manifest.csv` — patch verifier training manifest
5. `classifier/bench_feature_cache.py` — caching benchmark script
6. `classifier/runs/ablation_old_vs_new/` — OLD→NEW ablation CSVs
7. `classifier/runs/rgb_finetune_eval/conf_sweep.json` — conf sweep results
8. `eval_six_configs/youtube_ir/youtube_per_video.csv` — per-video YouTube results

These files are the "proof" that backs the claims. If any are missing, the corresponding claim cannot be added to the ledger.
