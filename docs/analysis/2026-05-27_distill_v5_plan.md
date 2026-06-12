# Distill V5 plan — fixing V4's sampling-balance and capacity ceilings

## Context

V4 smoke test (1093 samples on FT4 R3 features, p3+p5 fused 325-D) showed:

| Metric | Patch v2 @ 0.5 | MLP V4 @ 0.5 | Δ |
|---|---|---|---|
| Svanström drone R | 0.866 | 0.143 | **−72 pp** |
| Svanström F1 | 0.764 | 0.245 | −52 pp |
| Confuser halluc/img | 0.207 | 0.0015 | **−136×** |
| Train CV F1 (mlp meta+yolo) | n/a | **0.8804** | — |
| LDA train accuracy (fused 320-D) | n/a | **0.9844** | — |

**Read:** The features clearly separate drones from confusers (LDA 0.98). The MLP fits the training distribution well (CV 0.88). But Svanström drone recall collapses at deploy time. This is *not* a feature-ceiling problem — it's a **sampling-distribution problem** plus a **model-capacity problem**, plus a **loss-shape problem**.

V2 (baseline features, p5 only) had the same recall-collapse pathology with the same training corpus. The cure must address the corpus, not the architecture alone.

## Diagnosis: why V4 recall collapsed

Looking at the smoke-test training distribution (1093 samples):

| Source | Count | % of pool | Notes |
|---|---|---|---|
| Anti-UAV val drones | 166 | 15.2% | Easy drones; bulk of drone pool |
| Svanström drones | 37 | 3.4% | **Critical — small/distant, the thesis surface** |
| Selcom val drones | 0 | 0% | None detected at stride=30 |
| Svanström confusers | 500 | 45.7% | **Svanström confusers DOMINATE negatives** |
| Confuser train (web) | 250 | 22.9% | birds + airplanes + helicopters |
| Confuser val (web) | 113 | 10.3% | same |
| Selcom val confusers | 27 | 2.5% | CCTV background FPs |

Three structural failures stack:

1. **Anti-UAV : Svanström drone ratio is 4.5 : 1.** MLP learns "drones look like Anti-UAV drones" (high-conf, mid-size, paired backgrounds). Svanström small drones at deploy time look more like confusers in feature space than like Anti-UAV TPs.
2. **Svanström-confuser : Svanström-drone ratio is 13.5 : 1 *within Svanström*.** The MLP sees thirteen "this is a Svanström confuser → confuser" examples for every "this is a Svanström drone → drone" example, then learns the heuristic "Svanström-domain features → veto". At deploy time, real Svanström drones get vetoed.
3. **No real-video drones in training pool.** The deployment surface (pipeline_video_tests) is unseen.

The MLP arch is fine. The patch verifier wins because it trained on **45,917 patches with sequence-based 80/20 split + class-weighted CE + 12 epochs on a 2.5M-param MobileNet-V3**. V4 has ~50k params and 30 % the data of the patch verifier even at full Phase 1. There's headroom on both fronts.

## V5 plan — five lever pulls, ranked by expected impact

### Lever 1: Per-source quotas (cheapest, biggest win)

Replace the current "split max_per_source equally across N sources" with **target-percentage quotas** that match the *deployment* distribution, not the *available* distribution.

**Drone pool target (12k total):**

| Source | Quota | Reasoning |
|---|---|---|
| Anti-UAV val | 30 % (3,600) | Baseline drone distribution, plenty of supply |
| Svanström drone-positive frames | 40 % (4,800) | Thesis-critical surface; small drones |
| Selcom mixed ft2 train + val | 15 % (1,800) | OOD CCTV distribution |
| Real-video drone-positive frames (pipeline_video_tests, 9 videos) | 15 % (1,800) | Deployment-distribution drones |

**Confuser pool target (18k total):**

| Source | Quota | Reasoning |
|---|---|---|
| Web confusers (rgb_confusers_merged train + val) | 40 % (7,200) | Largest pool; bird/airplane/helicopter web variety |
| Svanström BIRD/AIRPLANE/HELICOPTER categorized frames | 25 % (4,500) | Domain-specific confusers — reduced from 56 % smoke share |
| Anti-UAV background false alarms (FPs from running FT4 on Anti-UAV val) | 15 % (2,700) | Hard negatives in same domain as positive Anti-UAV drones |
| Real-video confuser-only frames (pipeline_video_tests, 10 videos) | 20 % (3,600) | Deployment-distribution confusers |

**Implementation:** rewrite `collect_predictions` in `eval/distill_v4_p3p5_ft4.py` to accept per-source `target_n_drones` and `target_n_confusers` instead of a shared `max_samples`. Track running counts and skip a source when its quota is met.

### Lever 2: Focal loss + sample weights

The current `BCEWithLogitsLoss(pos_weight=...)` is class-balanced cross-entropy. That treats every sample equally within its class. We want to:

- **Up-weight hard examples** (small drones, confusers near decision boundary). Focal loss with γ=2 does this automatically by down-weighting easy correctly-classified examples.
- **Up-weight thesis-critical surfaces.** Svanström drones get sample weight 2.5×, real-video drones 2.0×, Anti-UAV drones 1.0×. Same for confusers but smaller multipliers (1.5× / 1.2× / 1.0×).

**Implementation:** add `FocalLoss` class to the MLPWrapper, replace BCE with it. Pass a `sample_weights` vector through `fit()`. Per-source weights computed at sample-collection time and stored in `training_data.npz` alongside `X, y`.

### Lever 3: Multi-scale ROI pooling

Current: pool p3 and p5 to a single 1×1 grid → 64 + 256 = 320 features per detection. Lossy — collapses all spatial structure.

V5: pool p3 to a 2×2 grid (4 × 64 = 256-D) and p5 to a 1×1 grid (256-D). Total YOLO features go from 320 to **512**.

This gives the MLP **spatial structure of the high-resolution p3 layer** — the part that doc claimed matters for small drones. With ~30k training samples a 512-D input is well below the rule-of-thumb "10× samples per feature" floor.

**Implementation:** edit `roi_pool` to accept `(out_h, out_w)` (already supports it via `nn.functional.adaptive_avg_pool2d`). Edit `_extract_detection_features` to use `out_h=2, out_w=2` for p3 and flatten to 256-D.

### Lever 4: Bigger MLP + better regularization

Current arch: `325 → 128 → 64 → 1` (≈50k params, dropout=0.2, no batchnorm).

V5 arch: `517 → 512 → 256 → 128 → 64 → 1` (≈300k params), with:

- **BatchNorm1d** after each Linear (helps with the standardized input distribution and large depth)
- **Dropout=0.3** between layers
- **Label smoothing 0.1** in the focal loss (prevents overconfident veto)
- **Cosine annealing LR schedule** from 1e-3 → 1e-5 over 150 epochs

300k params is still 8× smaller than MobileNet-V3-Small (2.5M), so we're not in over-parameterized territory.

### Lever 5: Two-stage training (Svanström finetune)

Only if Levers 1–4 leave Svan F1 < patch v2.

- **Stage A:** train on full V5 pool (30k samples, all sources, balanced).
- **Stage B:** load Stage A weights, finetune *only* on Svanström pool (4,800 drones + 4,500 Svan confusers) at LR 1e-4 for 30 epochs. Late layers only (freeze first 2 linear layers).

This is the deep-learning analog of FT4 R3's hard-negative finetune step — first pretrain on general data, then surgically adapt to the deployment surface. Two checkpoints saved: `mlp_v5_stageA.pt` (general) and `mlp_v5_stageB.pt` (Svan-specialized). Head-to-head will report both.

## Risk and contingency

If Lever 1+2 alone don't lift Svan F1 above patch v2:

- Lever 3 (multi-scale) is the next-cheapest add (no extra training data needed).
- Lever 4 (bigger MLP) adds capacity but risks overfitting if the corpus is somehow still too small. K=5 CV will catch this.
- Lever 5 (two-stage) is the nuclear option — likely shifts Svan F1 by 5–10 pp at the cost of slight drop on other surfaces.

If after **all five levers** Svan F1 is still <patch v2: the patch verifier wins genuinely, and the V5 chapter pivots to "what raw image crops capture that YOLO embeddings do not." That's still a thesis-grade finding because the LDA-acc-0.98 vs Svan-F1-collapse paradox is a clean negative result.

## Execution sequence

1. **Implement Lever 1 + 2 + 3 in `eval/distill_v5_p3p5_ft4.py`** (clone V4, apply diffs). One file.
2. Run Phase 1 (~45 min) — mine the 30k-sample balanced corpus.
3. Run Phase 2 (~30 min) — train V5 MLP with focal + sample weights + multi-scale.
4. Run head-to-head harness (already exists; pass `--mlp-weights` pointing at `mlp_v5.pt`).
5. If Svan F1 ≥ patch v2 *or* within −1 pp → ship V5, update EVIDENCE_LEDGER, write decision artifact.
6. If still trailing → add Lever 4 (bigger MLP). Re-train (~45 min), re-eval.
7. If still trailing → add Lever 5 (two-stage). Re-train (~1 h), re-eval.

**Total worst-case budget:** ~3 h experiment time + ~3 h eval time = ~6 h wall clock. Well-bounded.

## Files to create / modify

**Create:**

- `eval/distill_v5_p3p5_ft4.py` — clone of `eval/distill_v4_p3p5_ft4.py` with Levers 1+2+3 (and 4+5 gated by flags).
- `eval/sampling_quotas.py` — small module: per-source quota tracker, sample-weight assignment.
- `docs/analysis/2026-05-28_distill_v5_results.md` — decision artifact (written after head-to-head completes).

**Modify (in V5 only — V4 stays as the cache reference):**

- Replace `BCEWithLogitsLoss` → `FocalLoss(alpha=0.75, gamma=2.0, label_smoothing=0.1)`.
- Replace `roi_pool(..., 1, 1)` → multi-scale: p3 at 2×2, p5 at 1×1.
- Replace `MLPWrapper(input_dim=320+5)` → `MLPWrapper(input_dim=256*2*2 + 256 + 5 = 517)` and bigger hidden dims.
- Add `target_n` quota arg to `collect_predictions`.
- Add `sample_weights` field to the saved `training_data.npz`.

**Not modified:**

- `eval/eval_v4_vs_patch.py` — already accepts `--mlp-weights` pointing at any artifact with the same checkpoint schema. V5 will save in identical format.
- `RGB model/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt` — frozen detector.
- `classifier/runs/patches/confuser_filter4_rgb_v2_backup.pt` — production patch verifier (stays as baseline).

## Decision after V5

Same gate as before:

- V5 *strictly* beats patch v2 on ≥3 of {Svan R, Svan F1, real-video F1, confuser halluc} with ≤1 pp loss elsewhere → **swap.**
- V5 within ±1 pp across the board → **swap** (latency win).
- V5 loses by >1 pp on any surface → **keep patch v2;** V5 becomes the thesis-chapter closing negative-result row.

## Why this should work (signal vs noise)

The strongest pieces of evidence we already have that this approach is sound:

1. **LDA train accuracy = 0.9844 on the smoke-test 1093 samples.** With balanced sampling and 25× more data, train acc → ~0.99, and *test-set* acc should be ≥0.95.
2. **CV F1 = 0.8804** on smoke. With proper sampling this should land at 0.93–0.95.
3. **Confuser halluc on web set is already 0.0015–0.207 (50–270× better than patch v2).** Confuser-side of the problem is solved; only the recall side needs lifting.
4. **The drone-pool recall collapse pattern matches V2's pathology** — V2 also failed Svanström at 0.43, V4 at 0.14. Both failed in the same way, both with the same root cause (tiny / unbalanced training pool). V2 was 3.3k samples; V4 smoke was 1.1k. V5 will be 30k with quotas — order-of-magnitude more data, structurally different distribution.

## Delivered (after execution)

(Will be filled in after V5 head-to-head completes:)

- `eval/distill_v5_p3p5_ft4.py`
- `eval/sampling_quotas.py`
- `eval/results/_v5_p3p5_ft4_distill/training_data.npz`
- `eval/results/_v5_p3p5_ft4_distill/classifiers.pkl`
- `eval/results/_v5_p3p5_ft4_distill/classifiers/mlp_v5.pt` (and optionally `mlp_v5_stageB.pt`)
- `eval/results/_v4_head_to_head/v5_vs_patch_summary.json` (head-to-head with V5 weights)
- `docs/analysis/2026-05-28_distill_v5_results.md`
- `docs/EVIDENCE_LEDGER.md` row appended
