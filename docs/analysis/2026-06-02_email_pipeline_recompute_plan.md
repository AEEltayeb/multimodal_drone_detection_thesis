# Recompute the supervisor-email pipeline tables with the NEW stack (plan)

**Date:** 2026-06-02 · **Goal:** Reproduce the *exact* 3-domain tables from the bottom email
(`check.txt`) with the **new classifier (robust6)** and the **new filter (MLP V5)** swapped in, so we
can answer Pietro with a clean **2-column comparison** (old email stack vs new stack) — same datasets,
same frames, same scoring, same table layout.

**Decisions locked (2026-06-02):**
- New classifier = **robust6** (`classifier/fusion_models/lean_ft4/trust_ft4_robust6.joblib`).
- New filter = **MLP V5** — RGB `mlp_v5` (`eval/results/_v5_selcom_pure_1x8/classifiers/mlp_v5.pt`),
  IR `mlp_v5_ir_aligned` thermal ckpt (`mri/results/ir_aligned/classifiers/mlp_aligned.pt`).
- Frame scope = **full sets** (85,374 Anti-UAV / 28,710 Svanström / all 14 YouTube clips).
- Table = **2 columns**: OLD (email, transcribed) vs NEW (re-eval). No bridge column.

---

## 1. The two stacks being compared

| Component | OLD (email column — already computed, just transcribe) | NEW (this re-eval) |
|---|---|---|
| RGB detector | `Yolo26n_trained` (Jan, confuser-naive high-recall → the 10,777 Svan FP) | **`ft4`** `Yolo26n_selcom_confuser_ft4_1280` |
| IR detector | `v3b` (already) | **`v3b`** (unchanged) |
| Trust classifier | `fusion_no_fn` (40-feat) | **`robust6`** (6-feat XGBoost router) |
| Confuser filter | `confuser_filter4_{rgb,ir}` patch CNN | **MLP V5** (`mlp_v5` RGB / `mlp_aligned` IR) |

**Unavoidable caveat (must state to Pietro):** the MLP filter is *distilled from FT4 features*, so adopting
it forces the RGB detector to FT4. The email's RGB detector was the old high-recall model. Therefore the NEW
column differs from OLD by **detector + classifier + filter together** — it is a whole-stack comparison, not a
filter-only swap. (IR is `v3b` in both, so the IR side *is* a clean filter-only swap.)

---

## 2. Reuse vs re-eval — verdict: **re-eval the new column**

- **OLD column = free.** It's the email's own numbers — transcribed verbatim into the left column.
- **Existing box-caches are NOT reusable** for the new column: `raw_detections.json` /
  `svanstrom_detections.json` store **boxes only (no P3/P5 features)** and were generated with the **old RGB
  detector**. The MLP needs YOLO's internal features from **FT4**, so the detections themselves differ.
- **`eval/pipeline_cache.py` (parallel agent's Phase A) is reused for machinery, not as-is**: it already runs
  FT4/v3b with the `DetectInputHook` and caches the 517-D MLP features + patch probs. But it caches each
  modality **independently and strided to ~1k**, with **no RGB-IR pairing** and **no trust router** — so it
  cannot produce the email's paired `classifier` / `filter→classifier` / `classifier→filter` rows. We reuse its
  hook + extractor imports and write a **paired, full-set** variant.

---

## 3. Architecture — 2 phases (offline, GPU-gated, resumable)

Reuses proven pieces; records new scripts per `knowledge/` rules. **No duplication of purpose:** Phase A
imports the hook/extractor from `distill_v5_p3p5_ft4.py` (same as `pipeline_cache.py`); Phase B imports the
scoring helpers (`iou_iop`, `score_dets`, GT-scope logic) from `classifier/eval_six_configs.py`.

### Phase A — `eval/pipeline_cache_paired.py` (NEW; GPU; ~the only expensive step)
For each **paired stem** in Anti-UAV and Svanström (match RGB/images ↔ IR/images by filename):
1. FT4 forward on the RGB image (hook on) → boxes, confs, **517-D feats per box**, RGB GT boxes.
2. v3b forward on the IR image (hook on) → boxes, confs, 517-D feats per box, IR GT boxes.
3. Also store patch P(confuser) per box (cheap; for sanity only — not used in the 2-col output).
4. Write one paired record per stem → `eval/results/_email_recompute/cache/{antiuav,svanstrom}.pkl`.
- Resumable (skip stems already cached), per-frame try/except, internet-off, `nvidia-smi` GPU-gate before start.

### Phase B — `eval/pipeline_eval_paired.py` (NEW; CPU only; instant; re-runnable for threshold tweaks)
Loads the cache, applies **robust6 + MLP V5**, computes the **7 configs × {IoU, IoP} × per-category FP**,
writes `metrics_{iou,iop}.csv` + the side-by-side comparison `.md`. No GPU → re-run freely to tune thresholds.

### Domain 3 — extend `classifier/eval_youtube_ir_filter.py` (NEW flag `--mlp`)
Register `DetectInputHook` on v3b, replace `PatchVerifier` with `MLPVerifier(mlp_aligned.pt)`; a detection
**survives if P(drone) ≥ thr** (MLP) instead of P(confuser) < thr (patch). Keeps the identical det%/suppression
metric and the same 14 clips, so the table structure is byte-for-byte the email's domain-3.

---

## 4. The 7 configs — exact recipes (filter = MLP V5, classifier = robust6)

Per frame: `rgb_raw` = FT4 dets @conf≥0.25; `ir_raw` = v3b dets @conf≥0.40. A det **survives the MLP** when its
`P(drone) ≥ thr` → gives `rgb_flt`, `ir_flt`. robust6 label ∈ {0 reject, 1 trust_rgb, 2 trust_ir, 3 trust_both}.

| # | Config | robust6 input | Kept RGB | Kept IR | GT scope |
|---|---|---|---|---|---|
| 1 | `ir_only` | — | — | `ir_raw` | IR |
| 2 | `rgb_only` | — | `rgb_raw` | — | RGB |
| 3 | `classifier` | **raw** feats | `rgb_raw` if label∈{1,3} | `ir_raw` if∈{2,3} | both |
| 4 | `ir_filter` | — | — | `ir_flt` | IR |
| 5 | `rgb_filter` | — | `rgb_flt` | — | RGB |
| 6 | `filter→classifier` | **filtered** feats | `rgb_flt` if label∈{1,3} | `ir_flt` if∈{2,3} | both |
| 7 | `classifier→filter` | **raw** feats | `rgb_flt` if label∈{1,3} | `ir_flt` if∈{2,3} | both |

Difference #6 vs #7 (matches email's note exactly): **#6** filters *both* modalities first, *then* robust6
decides on the filtered detections; **#7** robust6 decides on *raw* detections, then the MLP is applied *only to
the trusted modality*. Scoring/GT-scope is identical to the email: single-modality configs score against their
own GT; classifier configs see both modalities' GT (so "trust_both" frames count TPs from both — the higher
absolute counts the email footnote explains).

---

## 5. Per-domain settings (pin these — they are correctness-critical)

| Domain | Dataset | N (full) | Scoring | RGB imgsz | IR imgsz | RGB conf | IR conf |
|---|---|---|---|---|---|---|---|
| 1 Anti-UAV | `Anti-UAV-RGBT_yolo_converted/test/{RGB,IR}` | 85,374 | **IoU@0.5** | 1280 | 640 | 0.25 | 0.40 |
| 2 Svanström | `svanstrom_paired/{RGB,IR}` | 28,710 | **IoP@0.5** | **1280** | 640 | 0.25 | 0.40 |
| 3 YouTube IR | `ir_gui/demo_outputs/yt_*.mp4` (14 clips) | det%/suppr | n/a | n/a | 640 | — | 0.40 |

- **Svanström = IoP** and **RGB imgsz=1280** are hard rules (memory: Svan native 640×480; IoU under-counts its
  oversized GT). Anti-UAV stays IoU.
- **RGB imgsz=1280 on Anti-UAV too** (not the email/`pipeline_cache.py` 640): `mlp_v5` was distilled from FT4 at
  1280 — feature distribution must match or the MLP mis-scores. Costs ~2-3× GPU time on Anti-UAV; acceptable.
- **MLP thresholds:** start at each checkpoint's stored `threshold`; IR aligned was validated at thr=0.05
  (recall-safe). Thresholds are a Phase-B knob (CPU replay) → sweep cheaply if a config over/under-vetoes.

---

## 6. Correctness risks (each has a mitigation — do these or the numbers lie)

1. **robust6 feature parity (highest risk).** robust6's 6 features must be computed *exactly* as its training
   CSV (`fusion_dataset_lean19.csv`) built them — esp. `best_log_bbox_area` (natural log? pixel vs normalized
   area?) and `best_aspect_ratio` (w/h?). **Mitigation:** before Phase B, read the lean_ft4 dataset builder,
   replicate the formula, and spot-check that features recomputed from a cached frame match the CSV row for the
   same stem. Wrong scaling → garbage routing.
2. **robust6 recall hole.** robust6 over-rejects on grayscale/hard-drone surfaces and costs ~1.2pp recall on
   Anti-UAV (55 FN) vs sa32 (per `2026-06-01_robust6_state_and_improvement_plan.md`). Svanström RGB is *real*
   RGB (not gray fallback) so it ties sa32 there. **Expectation to set with Pietro:** the new `classifier` rows
   may show slightly lower recall than the old fusion_no_fn on Anti-UAV — that is robust6's known FP-vs-recall
   trade, not a bug.
3. **MLP P(drone) vs patch P(confuser) inversion.** Survive-if `pdrone≥thr` (MLP) is the logical inverse of
   survive-if `pconf<thr` (patch). Easy to flip a sign. **Mitigation:** unit-assert on a known confuser frame
   (FP count must drop, not rise).
4. **Bundle format.** Confirm `trust_ft4_robust6.joblib` unpacks to `{"model","features",...}` and that
   `features` == the 6 ROBUST6 columns (there's also a newer `routing_robust/trust_routing_best.joblib` — the
   doc names the `lean_ft4` one as robust6; use that, verify `tag`).

---

## 7. Runtime, execution, recording

- **Phase A (GPU):** ~170k forward passes (85k×2 + 28k×2). At FT4@1280 + v3b@640 ≈ 25-40 fps → **~2-4 h**.
  Run as one offline, GPU-gated, resumable script (internet off). Phase B + Domain 3 are minutes.
- **Knowledge system (mandatory):** searched `scripts.csv` first — canonical pieces are `eval_six_configs.py`,
  `eval_youtube_ir_filter.py`, `pipeline_cache.py` (reuse, don't dup). After the run: `kb.py record` the 2 new
  scripts; `kb.py check-eval` before recording metrics; record an `evals` row per (domain × stack) and a
  `ledger` finding ("new stack vs old ensemble on the email's 3 domains"). Do **not** hand-edit CSVs.

## 8. Output — the deliverable for Pietro

One `.md` per domain with the email's exact columns, OLD vs NEW side by side, e.g. Anti-UAV/Svanström:

```
Config              | OLD F1 (email) | NEW F1 (ft4+robust6+MLP) | Δ
ir_only / rgb_only / classifier / ir_filter / rgb_filter / filter→classifier / classifier→filter
(TP/FP/FN/P/R/F1 for both columns)
```
plus the Domain-3 category table (CONFUSERS-wtd / AIRPLANE / BIRD / HELICOPTER / DRONE_CLEAN / DRONE_LABELED)
with OLD vs NEW det% + suppression. A 3-4 line plain-English readout per domain for the reply.

---

## Delivered (this session)
- `docs/analysis/2026-06-02_email_pipeline_recompute_plan.md` (this plan)
- No runs executed yet — Phases A/B + Domain-3 are the runnable units, pending green light to build.
- Verified inputs: robust6 + FT4 + v3b + MLP RGB/IR ckpts exist; 14 email clips present in `ir_gui/demo_outputs/`;
  Anti-UAV + Svanström paired dirs present.
