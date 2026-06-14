# runs/ — canonical results behind every number in the thesis

This directory is the **frozen evidence snapshot** for the thesis (`docs/thesis_working_distilling_overleaf/`).
Every headline number in the thesis maps to a cell in one of these files; the table below gives the map.
Live re-runs regenerate into `thesis_eval/results/` — the files here are the copies the thesis text was
audited against. The automated audit (`py thesis_eval\_audit_headline_numbers.py`, run from the repo root)
re-checks every headline cell of the thesis against the JSONs and must pass before any submission build.

All replay commands are **zero-GPU**: they re-score the unified detection cache
(`thesis_eval/cache/*.pkl`, built once by `pipeline_cache_unified.py` with the production detectors)
on CPU in about a minute. Caches are tagged with detector weight identity; `knowledge/` (the project's
relational evidence store) holds the same numbers as queryable rows (`evals.csv`, `ledger.csv`).

## Number → file → command

| Thesis number (where) | Value | File (cell) | Reproduce |
|---|---|---|---|
| Svanström composed F1 (abstract, §1.4, §4.1) | 0.742 → 0.949 (R 0.948→0.958, P 0.609→0.939) | `tier1_results.json` → `svanstrom.B_pipeline.{bare, clf->filt[robust8]}` | `py -u thesis_eval/pipeline_eval_unified.py --only svanstrom` |
| OOD confuser fire (abstract, §1.4, §4.1) | 30.4% → 4.9% (router) → 1.1% (filter) → 0.15% (composed) | `tier1_results.json` → `rgb_confuser.C_confuser` | `py -u thesis_eval/pipeline_eval_unified.py --only rgb_confuser` |
| Anti-UAV no-harm (abstract, §4.1) | 0.973 → 0.984 | `tier1_results.json` → `antiuav.B_pipeline` | `py -u thesis_eval/pipeline_eval_unified.py --only antiuav` |
| Anti-UAV bare detector (§1.1) | P 0.989 / R 0.982, 41 FP / 4,000 | `tier1_results.json` → `antiuav.A_bare.ft4/rgb` | same as above |
| Grayscale 3-way (§4.5) | RGB 0.607 / IR-on-rawRGB 0.187 / IR-on-gray 0.580 | `tier1_results.json` → D table (svanstrom, svanstrom_rawrgb, svanstrom_gray) | `py -u thesis_eval/pipeline_eval_unified.py` |
| Gray-confuser filter cut (§4.5) | 656 → 21 FP (grayscale scaler) | `tier1_results.json` → `gray_confuser.C_confuser` | `--only gray_confuser` |
| Temporal 2-of-3 windows (§4.2) | bare 0.843 @ 0.350 fire; router/composed arms | `temporal_results.json` | `py -u thesis_eval/temporal_replay.py` |
| Filter-threshold sweep on video (§4.2) | drone probs smeared [0.01, 0.25) | `video_thr_sweep.json` | `py -u thesis_eval/video_thr_sweep.py` |
| Modality A/B, coverage-scored (§4.1, tab:modality_ab) | RGB-only 0.458 / IR-only 0.632 / routed+filt 0.921 | `notes_round1_results.json` → `svanstrom.M_modality_ab` | `py -u thesis_eval/notes_round1_replays.py --only svanstrom,antiuav` |
| Per-size buckets (§4.1, tab:per_size) | rgb_test carve-out small-drone concentrated; svan medians 29.8/14.8 px | `notes_round1_results.json` → `*.SZ_per_size` | `py -u thesis_eval/notes_round1_replays.py` |
| Per-category confuser fire (§1.2, §2.2) | birds 39.0% / heli 58.0% / airplane 23.4% @640 | `notes_round1_results.json` → `rgb_confuser.CAT_confuser` | same |
| Background failure profile (§4.1, tab:failure_profile) | confuser-seq fire 65–74% background-invariant | `failure_profile_results.json` | `py -u thesis_eval/failure_profile_aggregate.py` |
| Low-conf operating mode (§4.1) | SelCom 0.591 → 0.692 @ floor 0.05 | `conf_sweep/conf_sweep_results.json` | `py -u thesis_eval/conf_sweep_replay.py` |
| IR HITL trajectory (abstract, §1.4, §4.3) | F1 0.503 → 0.967, one fixed split (n=9,612 @640) | `knowledge/evals.csv` rows `ir_final_*` (config `ir_final_640`); source CSV `eval/results/ir_version_comparison/` | `python eval/ir_version_comparison.py --imgsz 640 --split test` (GPU) |
| Stage latencies (§4.1, tab:speed) | router 0.095 ms/frame (404×); filter 1.3–2.1 ms/det (37–72×) | `knowledge/ledger.csv` → `robust6-speed-feature-efficiency`, `latency-edge-unmeasured` | `python eval/bench_speed.py` (GPU for filters) |
| Training–eval overlap audit (§3.3, tab:svanstrom_audit) | svan 17,314 train (+1,325 czoom), 37.3% exact frames; auv 22,603 train, 30/90 segments, 6.3% exact; router trained on 214 svan seqs / 61 auv segments; RGB corpus touches all 91 auv segments (16.0% exact) | `leakage_controlled.json` → `derived`; `clean_split/README.md` | `py -u thesis_eval/leakage_controlled_replay.py` |
| **Held-out clean split (§3.3, tab:clean_split)** | svan (54 seqs, n=5,557, both-detector-clean): v3b solo 0.867 vs 0.940, pipeline 0.935 vs 0.949; auv (61 segs, n=57,542, IR-clean): bare 0.977 vs 0.973, pipeline 0.986 vs 0.984 | `clean_split/clean_split_results.json` (+ sequence lists + manifest in `clean_split/`) | GPU: `pipeline_cache_unified.py --only svanstrom_clean,antiuav_clean --full --no-patch`; replay: `pipeline_eval_unified.py --out thesis_eval/results_clean` |
| Cached-subset leakage replays (superseded by the full clean split; kept as the router-residual control) | cascade-clean auv 0.964→0.975 (n=833), svan 0.952 (n=136) | `leakage_controlled.json` → `cascade_clean` | `py -u thesis_eval/leakage_controlled_replay.py` |
| Scoring-rule swing, current stack (§3.2.2) | 2.8 pp: dual 0.921 vs trust-aware 0.949 at identical P 0.939 | `notes_round1_results.json` → `svanstrom.M_modality_ab."routed[robust8] +filt"` vs `tier1_results.json` → `svanstrom.B_pipeline."clf->filt[robust8]"` | `py -u thesis_eval/notes_round1_replays.py --only svanstrom` |

## Also in this directory

- `corrective_finetune/` — the production IR detector's (v3b) actual Ultralytics training run:
  args, curves, confusion matrices, acceptance reports for finetune_v1…v3b. This is the training-side
  provenance for the detector whose evaluation numbers appear above.

## Verification chain

1. `thesis_eval/pipeline_cache_unified.py` — detect once (GPU), store everything per detection.
2. `thesis_eval/pipeline_eval_unified.py` + sibling replays — deterministic CPU re-scores (files above).
3. `thesis_eval/_audit_headline_numbers.py` — asserts the thesis's headline cells equal these files.
4. `knowledge/` — every eval/finding recorded as a row with source path + reproduce command.
