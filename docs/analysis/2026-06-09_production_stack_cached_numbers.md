# Production-Stack Numbers — already cached, ready for the thesis (2026-06-09)

**Headline: the full production-stack pipeline eval does NOT need re-running.** It is already computed
and cached across three result dirs. This consolidates them into the numbers needed to retire
`fig:mlp_pipeline_placeholder` and answer RQ1/RQ2 with the *shipped* stack (ft4 + v3b + robust8 +
mlp_v5 / mlp_v5_ir_aligned), instead of the comparison config (baseline + sa32 + patch) currently in the text.

Sources (all under `eval/results/`):
- `_routing_pipeline_cmp/comparison.md` — robust8 routing, full pipeline, 7 surfaces incl. video (2026-06-05). **The main one.**
- `_offline_pipeline/offline_eval_results.md` — verifier matrix, 12 surfaces, bare/patch/mlp (2026-06-02).
- `_email_recompute/comparison_{svanstrom_iop,antiuav_iou}.md` — old-email vs new-stack, paired (2026-06-04).

Scoring: trust-aware; Svanström IoP@0.5, Anti-UAV IoU@0.5. "fire" = confuser/false-alert rate.

---

## 1. Production cascade scorecard (clf→filter cell) — `_routing_pipeline_cmp`

| Router | thermal-drone F1 | grayscale-drone recall | confuser fire-rate |
|---|---|---|---|
| sa32 (comparison) | 0.983 | 0.537 | 0.051 |
| robust6 | 0.980 | 0.553 | 0.013 |
| **robust8@0.20 (shipped)** | **0.979** | **0.575** | **0.030** |

robust8 buys grayscale-drone recall (0.537→0.575 vs sa32) at a small thermal-F1 cost and a higher confuser
fire than robust6 — the documented robust8 trade. This table is the production answer the thesis currently
lacks.

## 2. Per-surface production pipeline — robust8@0.20, key cells (`_routing_pipeline_cmp`)

| Surface (n) | bare F1 | filter_only F1 | clf→filter F1 | filter→clf F1 | bare fire | clf→filter fire |
|---|---|---|---|---|---|---|
| svanstrom (4000, IoP) | 0.673 | 0.957 | 0.9875 | 0.9893 | 0.695 | 0.0133 |
| svanstrom_gray (4000) | 0.624 | 0.874 | 0.780 | 0.693 | 0.758 | 0.0472 |
| antiuav (4000, IoU) | 0.970 | — | 0.9699 | — | 0.796 | — |
| video_drone (1359, IoP, frame-level) | 0.891 | 0.730 | 0.631 | 0.586 | 0.460 | — |
| confuser surfaces | — | — | — | — | — | — |

Confuser-only (FP fire-rate), robust8@0.20:
| Surface (n) | bare | clf_only | clf→filter | filter→clf |
|---|---|---|---|---|
| rgb_confuser (2633) | 0.379 | 0.140 | 0.0129 | **0.0042** |
| video_confuser (1250) | 0.415 | 0.138 | 0.0464 | 0.0392 |

## 3. Verifier matrix (mlp_v5 / aligned vs patch vs bare) — `_offline_pipeline`

| Surface | bare F1 | patch F1 | **mlp/aligned F1** | bare→mlp halluc or FP |
|---|---|---|---|---|
| svanstrom RGB (1000, IoP) | 0.613 | 0.784 | **0.865** | halluc 0.449→0.045 |
| rgb_confuser (1000) | — | — | — | FP 216→104→**16** |
| selcom_val (311, IoP) | 0.591 | 0.591 | **0.612** | P 0.858→**0.950** |
| rgb_dataset_test (1000) | 0.922 | 0.894 | 0.812 | **carve-out**: R 0.888→0.694 |
| cbam IR (180) | 0.699 | 0.688 | **0.846** | FP 48→**15** |
| gray_confuser (1000) | — | — | — | FP 143→**20** (−86%) |
| gray_svan (1000, IoP) | 0.548 | 0.591 | 0.273 | **over-veto carve-out**: R 0.548→0.164 |
| antiuav RGB (1000) | 0.9866 | 0.9866 | 0.9866 | neutral (saturated) |
| ir_dset / ir_video / svanstrom_ir / antiuav_ir | ~0.95–0.98 | — | recall-safe (≈neutral) | aligned does not erode IR recall |

## 4. New-stack vs old-email, paired (robust6 + V5 MLP) — `_email_recompute`

Svanström (IoP), NEW = ft4+v3b+robust6+V5: `rgb_filter` F1 **0.699→0.869 (+17pp**, both P+R up);
`ir_filter` +1.4pp; `classifier→filter` F1 0.9496; `classifier` paired 0.9594 (−3.4pp vs old fusion_no_fn
0.9937, but that gap is on the *saturated/in-domain* benchmark — robust6's value is OOD-confuser FP, not
visible here). Anti-UAV: essentially neutral (saturated), NEW classifier 0.9818 vs old 0.9916.

---

## 5. What this retires / answers in the thesis (maps to the audit)

- **RQ1 (production):** §1+§2 give the shipped-stack confuser suppression (rgb_confuser 0.379→**0.004**;
  video_confuser 0.415→**0.039**) — replaces the comparison-config-only answer.
- **RQ2 (production cost):** §1 scorecard + §2 give the in-distribution cost with robust8/mlp, not sa32/patch.
- **`fig:mlp_pipeline_placeholder`:** §1 scorecard is a drop-in replacement figure/table.
- **CBAM, grayscale carve-out, rgb_dataset carve-out:** §3 confirms the canonical numbers
  (CBAM 0.699→0.846 FP48→15; gray over-veto; rgb_dataset recall carve-out).

## 6. The ONE genuine remaining gap (needs work, not just cache-read)

The video numbers in §2 are **frame-level** (IoP per frame). The thesis's real-video story
(`tab:cascade_segment`) is **segment-level with the temporal smoother** — and at frame level the
production cascade *drops* drone F1 (video_drone 0.891→0.631) because the verifier vetoes real drones
per-frame; the temporal smoother is what recovers this at segment grain. The segment-level
temporal-smoothed cascade with the **production** stack is NOT in any cache.

Two ways to close it (decide in the morning):
- **(a) Reuse cache + logic [no GPU]:** apply the temporal-smoothing + segment-aggregation logic (as in
  `eval/eval_video_temporal.py` / the segment aggregator) to the cached per-frame video results in
  `_routing_pipeline_cmp/cache/{video_drone,video_confuser}.pkl`. If the cache stores per-frame
  per-detection decisions, this is a pure re-aggregation — no inference. **Needs a code check that the
  cache has what the temporal logic needs.**
- **(b) Fresh run [GPU, ~hours]:** wire `MLPVerifier` + robust8 into a video harness and run the 19 clips.
  Only needed if the cache lacks per-frame decisions.

## 7. Tiny separate gaps (not overnight-worthy)
- baseline RGB @ imgsz=640 on Svanström (one detector run, minutes) for the resolution sweep figure.
- bootstrap CI on the RQ3 grayscale aggregate — computable from cached per-frame results, no GPU.

## 8. Recording note
The robust8 routing numbers (§1, §2; `_routing_pipeline_cmp`, 2026-06-05) should be recorded into
`knowledge/evals.csv` + a `ledger` row via `kb.py` so the thesis can cite them with provenance (currently
only `robust6`-stage rows like `svan_classifier_robust6_ta` are in evals.csv). Not yet done.

## Delivered
- This consolidation: `docs/analysis/2026-06-09_production_stack_cached_numbers.md`
- Sources: `eval/results/{_routing_pipeline_cmp,_offline_pipeline,_email_recompute}/`
- No eval run executed (all numbers read from existing caches). No thesis edits.
