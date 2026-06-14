# Trust routers / fusion classifiers — what each model is and how it was trained

All routers are 4-class XGBoost frame-level classifiers: `0=reject_both, 1=trust_rgb,
2=trust_ir, 3=trust_both`, consumed by the GUI (`gui/fusion/engine.py classify()`) via a joblib
bundle `{"model", "features", ["tau"]}`. When a bundle carries `tau`, the GUI applies
`trust_rgb iff P(trust_rgb) >= tau, else argmax` (rare-class fix). The `router_conf` GUI setting
controls which detections feed the router's FEATURES: **0.25 for every model below except the
`robust_mf_*` candidates (use 0)** — they were trained multifloor.

Source of truth for numbers: `knowledge/{models,evals,ledger}.csv` in the working repo.

---

## PRODUCTION

### `robust8_noreject.joblib` (**robust8-nr**) — the SHIPPED trust router (no-reject, 2026-06-14)
- **What:** robust8's exact 8 features + training recipe, but the `reject` class is removed
  (3-class: `1=trust_rgb, 2=trust_ir, 3=both`). **argmax, no τ** — the router always routes; the
  per-frame MLP/patch filter owns all FP rejection. Stored with `label_map {0:1,1:2,2:3}`; the GUI
  maps the model's 0-indexed classes back to trust labels in `gui/fusion/engine.py classify()`.
- **Why shipped:** best mean drone-surface F1 (clf→filt 0.834 vs robust8 0.733); removes robust8's
  over-rejection of hard/grayscale drones. Deployed composition is **`filt→clf`** (MLP filter first,
  then route the survivors). Flat path `models/routers/robust8_noreject.joblib`; trained by
  `classifier/train_robust8_noreject.py`. (Never emits class 0, so the 4-class header above still
  describes the bundle label space.)

### `robust8.joblib` — reject-class ablation / paired-stream comparison (τ=0.20)
- **What:** 8 free features — `rgb_max_conf, ir_max_conf, rgb/ir_best_log_bbox_area,
  rgb/ir_best_aspect_ratio` (= robust6) **+ `rgb_mean_conf` + `is_grayscale`** — with the
  τ=0.20 trust_rgb rule. Closes the grayscale trust_rgb hole (argmax recall 0.12 → 0.91).
- **Trained:** `eval/_routing_replay.py` on `optimal_v1/fusion_dataset_full56.csv`
  (65,192 rows; ft4+v3b detections at conf 0.25; trust labels = has-TP per modality vs own GT;
  regime flag derived from source: antiuav/svanstrom=thermal, everything else=grayscale).
  Group-split by sequence; τ picked by held-out grayscale trust_rgb F1 sweep.
- **Headline:** svan-gray pipeline 0.658→0.750 vs sa32 at ~½pp cost on thermal. Picked 2026-06-05.
- **Known limits (audited 2026-06-12, ledger `router-prior-audit`):** decisions are ~95%
  reproducible from size+regime priors; over-rejects medium "bird-band" both-fire detections on
  OOD video (0.566 vs robust6 0.606); features go out-of-regime if the detector runs below
  conf 0.25 unless `router_conf=0.25` pins them (GUI fix 2026-06-12).

### `robust8_tau0.10.joblib` / `new_router.joblib` (byte-identical twins)
Earlier τ=0.10 build of the same robust8 retrain (2026-06-05..09). Superseded by τ=0.20.

---

## CANDIDATE (2026-06-12, not signed off)

### `robust_mf_f8_bal_caches+full56.joblib` — de-biased multifloor router
- **What:** same 8 features as robust8; bundle τ=0.10 (auto-read by the GUI). **Use with
  `router_conf: 0`** — trained across detector conf floors 0.05–0.90 (asymmetric rgb×ir pairs),
  so it consumes the live operating point.
- **Trained:** `thesis_eval/train_router_multifloor.py` (working repo) on 1.5M rows =
  conf-floor-augmented low-conf caches (11 surfaces incl. video drone/confuser → fills the
  628-row grayscale coverage hole) + full56, **cell-balanced** sample weights
  (1/count per size-band×regime×label, clip 10×) to remove the "medium both-fire = bird" prior.
- **Why it exists:** the 2026-06-12 audit showed the shipped routers re-encode dataset label
  priors; this is the balanced retrain. **+9pp OOD video (0.697 vs robust6 0.606), antiuav 0.989
  (best recorded), bird-band probe PASS.** Costs: Svanström-night −1.3pp vs robust8 (needs the
  MLP filter), rgb_confuser 24 FP vs robust6's 3 (post-filter, <1% fire).
- Siblings in the working repo (`thesis_eval/results/router_multifloor/`): `f10_raw` (+ verifier
  P(drone) features — ir_confusers 199 FP, 5× better than anything; needs GUI pdrone wiring) and
  `f10_bal` (best video 0.699). Recorded as `robust_mf_*` in knowledge/models.csv.

---

## COMPARISON / SUPERSEDED

### `lean_ft4/` — **robust6** (`trust_ft4_robust6.joblib`)
6 free features (confs + best-box geometry only), statistically selected via ANOVA/AUROC +
leakage screen (`leakage_ratio = F_domain_inclass/F_class` kills scene fingerprints). Trained
2026-05-31 on ft4+v3b lean data (`classifier/train_lean_ft4.py`). The OOD-robust comparison
model: −30% false alerts on real video vs sa32; argmax (no τ) → has the grayscale trust_rgb hole.

### `scene_aware_v3more_32feat/` — **sa32** (former deployment pick)
32 features = 40-feat set minus 8 detection-presence leaks. Trained on
`fusion_dataset_v3more.csv` (152,051 paired frames, Anti-UAV + Svanström) with the v3_more-era
detectors. Strongest in-domain (its scene statistics memorize the benchmarks) — that strength is
the leakage the lean line removed. Superseded 2026-06-05.

### `control_v3more_40feat/` — leaky 40-feature control for sa32 (keeps `ir_detected` etc.;
top feature carries 35% importance — the shortcut demonstration).

### `retrained_v2_32feat/` — sa32 recipe re-trained on the retrained_v2 RGB detector stack.
Superseded with that detector (OOD recall collapse).

---

## ABLATION ARTIFACTS (kept for the thesis paper trail)

- `lean10/ lean13/ lean13_smoketest/ lean13_yt/ lean13_yt_only/ lean17/ lean19/ lean19_v2_{A,B,C,BC,ABC}/ lean10_yt/`
  — May-25/26 lean-feature-set ablations (which feature subsets transfer, YouTube-video-trained
  variants) that led to lean19 → robust6. One-offs; see `knowledge/scripts.csv` (ablation_split*).
- `split_v1/ split_v2/ split_v3/` — routing-dataset split experiments (May 26).
- `feature_selection_pilot/ feature_selection_pilot_v2/` — pilot statistical feature-selection
  sweeps (scripts archived 2026-06-05).
- `optimal_v1/` — home of **`fusion_dataset_full56.csv`**, the routing training corpus
  (ft4+v3b dets @0.25, has-TP trust labels) used by robust8 and merged into the `robust_mf_*`
  retrains; plus the early "optimal" router experiment.
- `routing_robust/` — outputs of `classifier/train_routing_robust.py` (per-class × per-regime
  routing reports; the script whose dormant "Phase-1b" verifier-pdrone hooks the f10 candidates
  finally fed).
