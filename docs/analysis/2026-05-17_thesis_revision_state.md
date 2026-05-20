# Thesis revision — active state (2026-05-17, last updated 2026-05-18)

## 2026-05-18 STATUS AT TOP (read this first)

**Status as of 2026-05-18 end-of-day**: Full pipeline data integrated, classifier comparison done, production-pick flipped to sa32, chapter-by-chapter review pass complete (Ch1–Ch6 all updated for sa32 framing + surface-dependent cascade story). Ledger §9.4, §9.5, §9.5.7–9.5.9 added; §1 production stack rewritten with control40 deprecated. Open items remaining: end-to-end latency, real-video patch_thr sweep (low value — Svanström sweep monotone), `flock_of_birds_attack_drone` cascade run (visceral case only, optional). Three skipped runs are documented as future work in §5.limits and Ch6 conclusion. Next pending task: stylistic/prose review pass requested by user 2026-05-18 (chapter-by-chapter what's gold / what's AI slop / what to cut). `references.bib` still empty.

---



**Headline shifts:**
- IR-grayscale aggregate drone F1: $0.664 \to 0.636$.
- `flock_of_seagulls_attack_drone_beach` IR-grayscale F1: $0.901 \to 0.837$ (now **ties** baseline RGB at $0.840$, no longer dominant).
- New Pareto frontier (drone F1 vs all-confuser FPPI): **{IR-grayscale, selcom\_640, baseline}**. `retrained_v2` is now Pareto-dominated by IR-grayscale. `selcom_1280` is Pareto-dominated by baseline.
- `selcom_640` is the new third Pareto point ($F1=0.730$, FPPI=$0.260$); added to all real-video tables.
- `flock_of_birds_attack_drone` is NOT in the re-run; user chose to skip re-staging. The §9.2 row remains the only source for that case (preserved in memory and ledger).

**Thesis sections updated** (all done):
- Abstract — IR-gray $F1$ line.
- Ch1 §1.4 contribution bullet 4.
- Ch2 §2.thermal paragraph (line ~237).
- Ch3 §3.fusion paragraph (line ~319; "Pareto-dominates" → "sits on the Pareto frontier").
- Ch4 §4.2.3 IR Grayscale-RGB Operating Mode (line ~463).
- Ch5 §5.realvideo: section title "Five-Stack" → "Six-Mode"; Table~\ref{tab:realvideo_master} rebuilt (added \texttt{selcom\_640} row, corrected all numbers); three-paragraph reading rewritten ("Pareto frontier has three points").
- Ch5 §5.realvideo Table~\ref{tab:realvideo_seagull} rebuilt and prose rewritten ("ties baseline").
- Ch5 §5.grayscale Table~\ref{tab:ir_grayscale} corrected; "Why was it unexpected" and "transfer is not perfect" paragraphs corrected (also fixed an inherited factual error: $0.142$ is **not** below \texttt{retrained\_v2}'s $0.139$).
- Ch6 conclusion four-findings paragraph.

### Pinned next steps

1. Resume **chapter-by-chapter pass through Ch3--Ch6** (user originally asked, paused twice for data drops). Wait for user confirmation before starting.
2. Fill `references.bib` (still empty; multiple TODO citation keys exist). User wants explicit go-ahead.
3. Dataset-conditional sweep of remaining categorical "fires on X%" claims.
4. Pipeline video eval is still pending (was the *other* original next step before the scoring bug surfaced). When that lands, it will need to be reconciled against §9.4 numbers (the pipeline script was always correct so the numbers should now agree).

### What was fixed in this session

### What was fixed in this session
`eval/eval_video_tests.py` had a bug where `iop_25` (the headline conf=0.25 metric) was computed by post-filtering a YOLO inference pass run at `base_conf=0.05`. Ultralytics applies the `conf` parameter pre-NMS, so a conf=0.05 pass post-filtered to conf>=0.25 is **not** the same as a direct conf=0.25 pass. This produced FP counts 3–10× higher than the pipeline eval (`eval_pipeline_video_tests.py`) for identical inputs.

Fix applied: added a second YOLO inference pass at `prod_conf=0.25` per frame. `iop_25` now uses that pass. The script signature is `eval_on_dataset(..., prod_conf: float = 0.25)`; the CLI exposes `--prod-conf`. Per-video JSON now includes `total_dets_prod`, `base_conf`, `prod_conf` provenance fields. **Cost: ~1.5–2× runtime.**

`eval_pipeline_video_tests.py` was already correct (uses `score_dets_pipeline` → `score_detections` for proper bipartite matching, YOLO run at `rgb_conf=0.25` directly). No edits needed.

### When the re-run lands

Expected output paths (overwriting):
- `eval/results/video_tests/video_tests_comparison.{csv,json}`
- `eval/results/video_tests/{cat}/{video}/{model}.json`
- `eval/results/pipeline_video_tests/pipeline_comparison.{csv,json}`
- `eval/results/pipeline_video_tests/{cat}/{video}/{model}.json`

Expected changes vs the pre-fix data:
- Aggregate IR-grayscale drone F1 likely drops from **0.664 → ~0.649** (the per-frame JSON aggregate falls in line with the pipeline-side number).
- The `flock_of_seagulls_attack_drone_beach` IR-grayscale single-video F1=**0.901** may shift; check the per-video JSON. If it changes by >5 pp, the thesis citation needs updating.
- `selcom_1280` on the same video: pre-fix per-frame JSON showed TP=166, FP=118; pipeline showed TP=164, FP=25. After fix both should converge to roughly the pipeline numbers (much lower FP count).
- `best_sweep` should no longer produce impossible values (the older code had cases where FN went negative and F1 > 1; the current code already routes through proper bipartite, so this is gone).

### Pinned next step after re-run

1. Re-read `docs/video_test_evaluation.md` if user updates it, or re-derive aggregates from the new CSVs.
2. Update the thesis numbers in:
   - Abstract (IR-grayscale F1 line)
   - Ch1 §1.4 contribution bullet 4 (IR-grayscale F1 numbers)
   - Ch4 §4.2.3 IR Grayscale-RGB Operating Mode
   - Ch5 §5.realvideo master table (Table~\ref{tab:realvideo_master})
   - Ch5 §5.realvideo seagulls table (Table~\ref{tab:realvideo_seagull})
   - Ch5 §5.grayscale (entire section's numbers)
   - Ch6 conclusion four-findings paragraph
3. Add the §9.x section to `EVIDENCE_LEDGER.md` documenting the new run + fix.
4. Resume chapter-by-chapter pass through Ch3–Ch6 (still pinned from before).
5. Verify `flock_of_birds_attack_drone` is in the re-run or update thesis to drop that case.

---



This document captures the live state of the thesis-revision work in progress on `docs/thesis_chapters.tex`. It exists so the next conversation (post-context-compaction) can pick up without re-deriving the narrative decisions or re-checking what is already rewritten. Pair this with the `EVIDENCE_LEDGER.md` (which has all the numbers) and the relevant project memories.

## 1. What is already done

The thesis was inherited from a Gemini-generated first draft full of fabricated numbers and AI-tells. It has been progressively rewritten chapter by chapter. The structure has been reorganised from 9 chapters to 6 (Intro, Related Work, System Architecture, Components + HITL, Experimental Study, Conclusion). Every quantitative claim is now traced to a section of `EVIDENCE_LEDGER.md`. Three narrative-correction passes have been completed:

1. **First pass** (style + structure): em-dash addiction, AI-tells, "this section presents" openers — cleaned. Restructured to mirror `tesi_master.tex` style. Added §5.X Comparison to Prior Work and expanded the IR-on-grayscale section.
2. **Second pass** (narrative): disentangled "drone-class P/R" from "OOD confuser fire rate" (they had been conflated). Reframed the cascade as an OOD safety mechanism trading in-distribution F1 for OOD robustness, not as an unconditional precision boost. Reframed RGB-vs-IR as complementary failure profiles, not "RGB-for-day, IR-for-night."
3. **Third pass** (real-video data integration): incorporated the 2026-05-17 real-video studies (§9.1, §9.2, §9.3 of the ledger). Added the new §5.X "Real-Video Five-Stack Diagnostic" section with master table covering all 5 modes (baseline RGB, retrained_v2, selcom_1280, IR on grayscale-RGB, IR on raw RGB). Major rewrite of §5.11 grayscale section, retitled "An Unexpected Cross-Modal Result", framing the IR-on-grayscale finding as a deployment-necessity discovery rather than a designed feature.

## 2. Narrative decisions to respect (do not regress)

These are load-bearing framing choices the user has explicitly endorsed. Any rewrite must respect them.

- **Detectors are NOT trained without confuser negatives.** Baseline RGB has birds (drone-vs-bird subset). hardneg_v3more and retrained_v2 add airplanes + helicopters. IR `v3b` was trained with confuser negatives via the HITL loop. **Never** frame the cascade as "naïve detector + downstream precision." Frame it as "residual hallucinations that even confuser-aware detector training cannot eliminate." See memory `project_detector_training_corpora.md`.
- **Confuser fire rate is dataset-conditional, not categorical.** "94.4% bird fire rate" is Svanström-only. On Roboflow `rgb_bird` baseline fires on 59% of frames; on real YouTube bird videos 54%. Every "fires on X% of birds/airplanes/helicopters" claim must specify the dataset. The categorical framing is wrong and must be retired throughout. (Partial work done; see pinned items.)
- **The three RGB models occupy a recall-precision continuum.** retrained_v2 = conservative (high P, low R); baseline = middle; selcom_1280 = aggressive (high R, lower P). Empirically supported by §9.1 + §9.2 + §9.3 master table. No single RGB model is dominant; production deployment is scene-conditional.
- **selcom_1280 OOD damage finding.** selcom_1280 is the *worst* RGB model on OOD confuser videos (FPR 41% vs baseline 37%, retrained_v2 17%). The fine-tune trades CCTV drone recall for OOD confuser robustness. Production stack picks should now flag selcom as "CCTV-only, not for non-CCTV deployment." See memory `project_confuser_videos_eval.md`.
- **IR-on-grayscale-RGB is unexpected.** Frame as a deployment-necessity discovery (thermal cameras are scarce), not a designed capability. The result is real: $F1=0.664$ aggregate on drone video; $F1=0.901$ on the hardest bird-cluttered single video, beating every dedicated RGB model. Pareto-dominates the RGB baseline on the joint drone-F1 vs confuser-FPPI axis. See memory `project_ir_grayscale_video_eval.md`. **Do not soften this finding** — the user explicitly wants it pushed as unexpected.
- **The cascade trades in-distribution F1 for OOD robustness.** On Svanström paired drones S1 F1=0.950 falls to S3 F1=0.895 (production op point). On the OOD confuser zoo fire rate falls 52.1% to 0.8%. Both numbers go together in every cascade headline. Do not claim "cascade improves precision" without the in-distribution F1 cost.
- **`retrained_v2` recall collapse generalises beyond Svanström.** On real-video bird-cluttered drone scenes retrained_v2 R = 0.000 to 0.080. The Svanström R = 0.306 is consistent with this pattern; it is not an outlier. The "retrained_v2 is disqualified by Svanström" framing has been retired; the honest version is "retrained_v2 is the conservative end of the continuum, wrong for drone-in-clutter deployment."
- **selcom_1280 is fine-tuned from baseline, not from retrained_v2.** This matters for trust-classifier calibration (the classifier was trained against baseline-family RGB confidence distributions).
- **mAP is supplementary, not primary.** Headline metrics are always P, R, F1. mAP appears only in the IR-evolution table (Table 4.1) as a supplementary column.
- **Trust-aware scoring is the reporting rule** for all cascade-level F1 numbers, never dual scoring. The 28-pp F1 swing audit (§5.3) is its own contribution.

## 3. Pinned action items (in priority order)

These were owed before the §9.3 work and remain owed.

### 3.1 Waiting for: full-pipeline video eval

The user has run RGB-only video eval (§9.1, §9.2) and IR-grayscale/raw video eval (§9.3). The full-pipeline video eval — trust classifier + patch verifier on these same videos — is the next data drop. **Do not start chapter-by-chapter pass on Ch3–Ch6 until this lands.** It will reshape Ch3 / Ch5 again. The user said: "we will wait for the pipeline data, before we do anything though, save important things to your memory."

When the data arrives, expect:
- Per-video cascade trust-aware F1 on drone-positive videos
- Per-video cascade confuser fire rate on confuser-only videos
- Both with and without the patch verifier alert gate
- Possibly grayscale-RGB plumbed through the cascade as a third modality option for the classifier

This is the missing piece that lets the thesis claim end-to-end cascade behaviour on real video, not only on Svanström.

### 3.2 Chapter-by-chapter pass (still owed)

The user originally asked for a chapter-by-chapter critique pass through Ch3–Ch6. We completed Ch1 + Ch2 (with stops to discuss after each chapter). Ch3 onward was paused twice for new data drops. After the full-pipeline eval, resume the pass. The user expects me to stop after each chapter and report thoughts before moving on. **Do not batch all chapters together.**

### 3.3 Dataset-conditional rewrite of categorical fire-rate claims (partial)

Per §06 analysis (`docs/analysis/2026-05-17_failure_profile_by_dataset.md`), the categorical "fires on X% of birds" framing is wrong. Every such claim must be dataset-qualified. Some progress in Ch2 §2.5 and Ch4 §4.1.1; not exhaustively swept across the whole document. After the full-pipeline pass, do a final sweep to catch residual instances.

### 3.4 §9.2 per-video drone story not fully integrated

The new §5.X Real-Video Five-Stack Diagnostic has the aggregate table and the flock_of_seagulls per-video table. But the broader §9.2 story (three RGB models on 10 drone-positive videos, including the load-bearing `flock_of_birds_attack_drone` case where baseline + retrained_v2 score F1=0 and only selcom detects the drone) is not in the thesis. Consider promoting that case as a second per-video table or figure.

### 3.5 Open citation TODOs

`references.bib` is empty. Multiple `\cite{}` placeholders exist:
- `TODO_counterUAS_survey` — Ch1 §1.1 (need a real C-UAS survey)
- `hoffman2016modalityhallucination` — §5.11 (real paper, CVPR 2016, need to verify bibkey + add to bib)
- `TODO_rgb2thermal` — §5.11 (RGB-to-thermal translation citation; CycleGAN-class)
- `TODO_coluccia2021` — §5.13 Comparison section
- `TODO_xmodal_fusion` — §5.13
- Several "real" citations need bib entries: `redmon2016yolo`, `ren2015fasterrcnn`, `ultralytics2024`, `svanstrom2021real`, `jiang2021antiuav`, `shi2018counteruas`, `ramachandram2017fusion`, `shrivastava2016ohem`, `lin2017focal`, `guo2017calibration`, `settles2009active`, `ng2021datacentric`, `chen2016xgboost`, `howard2019mobilenetv3`

The thesis will not compile until `references.bib` is populated. Don't start filling it without the user's go-ahead — they may want to do this themselves with a reference manager.

### 3.6 Three open ledger items (`EVIDENCE_LEDGER.md` §11)

- `control_v3more_40feat`'s confuser-zoo OOD behaviour was not measured at the time of writing. Currently marked "TBD-but-comparable-to-`scene_aware`" in Ch4 §4.3 / Ch5 §5.7.
- The grayscale IR confuser numbers' original CSV source (`confuser_test_hallucination.csv`) was marked UNKNOWN; with the §9.3 data now load-bearing, this is moot, but the §5.11 caveat about "CSV source row marked for re-derivation" may still be in the text — verify and remove if so.
- The patch verifier v3 file path is still unresolved (Ledger §11).

## 4. What survives the compaction (do not waste tool calls re-checking)

- All numerical claims trace to `EVIDENCE_LEDGER.md` sections. Read that file for any specific number rather than re-running anything.
- The thesis itself is on disk at `docs/thesis_chapters.tex`. Read it; do not regenerate.
- Memory files under `~/.claude/projects/.../memory/`:
  - `project_detector_training_corpora.md` — all detectors trained with confuser negatives
  - `project_confuser_videos_eval.md` — §9.1 (RGB on confuser videos; selcom worst)
  - `project_video_drone_eval.md` — §9.2 (RGB on drone+bird videos; three-mode continuum)
  - `project_ir_grayscale_video_eval.md` — §9.3 (IR on grayscale-RGB beats raw-RGB by 2× F1)
  - `project_production_stack.md` — older production stack picks (now superseded by §9.x in places; check ledger §1 for current)
  - `project_pyside_gui_features.md` — GUI non-obvious behaviours (alert gate, ROI re-crop, grayscale path)
- `docs/analysis/2026-05-17_failure_profile_by_dataset.md` — the per-detection size/conf analysis from §06.

## 5. Active TODO list status at compaction time

From the session's task tracker:

| ID | Task | Status |
|---|---|---|
| 1 | Stylistic pass Ch1-2 | completed |
| 2 | Stylistic pass Ch3 (Architecture) | completed |
| 3 | Stylistic pass Ch4 (Components) | completed |
| 4 | Stylistic pass Ch5-6 (Experiments+Conclusion) | completed |
| 5 | Add §5.X Comparison to Prior Work | completed |
| 6 | Promote IR-on-grayscale to dedicated study section | completed |
| 7 | Narrative pass: disentangle drone-P/R from OOD confuser fire rate | completed |
| 8 | Narrative pass: cascade as OOD safety mechanism | completed |
| 9 | Narrative pass: modalities as complementary failure profiles | completed |

All current tasks completed. New tasks to create when full-pipeline data lands:
- Integrate full-pipeline video eval results into Ch5
- Resume chapter-by-chapter pass through Ch3 onward
- Dataset-conditional fire-rate sweep
- Decide on §9.2 per-video promotion

## 6. How the user resumes the conversation

The user should say something like:

> "Continue the thesis revision. Full-pipeline video eval is at `<path>` / see `check.txt`. Use the state document at `docs/analysis/2026-05-17_thesis_revision_state.md` to pick up."

Or, if they just want me to look at new data and tell them what to do:

> "I ran the full-pipeline eval, results in check.txt. Read the state doc first then tell me your thoughts."

I should:
1. Read `docs/analysis/2026-05-17_thesis_revision_state.md` (this file) first
2. Read `docs/EVIDENCE_LEDGER.md` for the data structure
3. Read `check.txt` for the new run results
4. Do the §9.x-style analysis: save to memory, add to ledger, then write up findings honestly
5. Then resume the chapter-by-chapter pass through Ch3–Ch6 with stops after each chapter

## 7. Things I should not do after compaction

- Do **not** regress to the "YOLO is recall-catcher, downstream is precision" framing.
- Do **not** present "94% bird fire rate" as a categorical/global claim — always with the Svanström qualifier.
- Do **not** call retrained_v2 a "confuser killer" without the dataset/regime caveat.
- Do **not** claim the cascade is an unconditional precision boost without the in-distribution F1 cost in the same sentence.
- Do **not** soften the IR-on-grayscale finding. The user wants it pushed as unexpected.
- Do **not** start the chapter-by-chapter pass before the full-pipeline data is integrated.
- Do **not** start filling `references.bib` without the user's explicit go-ahead.
- Do **not** offer `/schedule` for any of this — these are active work items, not future-dated artefacts with concrete cleanup conditions.

## 8. Delivered

- `docs/analysis/2026-05-17_thesis_revision_state.md` — this document.
