# Production-pipeline reframe — consolidated edit package (2026-06-02)

Goal (user): the thesis must present the PRODUCTION pipeline as **robust6 classifier · ft4 RGB · v3b IR ·
mlp_v5 (RGB verifier) · mlp_v5_ir_aligned (IR verifier)**; the per-frame MLP filter replaces the
MobileNetV3 patch verifier; **everything else = comparison / ablation / trial**. Define each on first
mention; minimal changed-decision history. Produced by 3 read-only sub-agents (Ch1-2, Ch3-4, Ch5-7) +
the earlier ledger-scan agent; reconciled here.

## ⚠️ Concurrent-edit warning
`thesis_working.tex` is being edited by another agent. Confirmed: `\label{sec:feature_selection}` +
`\paragraph{The robust6 classifier.}` + `tab:robust6_pipeline` + "Full-pipeline verdict" already exist
(lines ~1006–1091) and the production-stack paragraph was rewritten under me. **Do not apply this package
until the other agent is paused** (two writers on one .tex = corruption / clobbered work).

## ALREADY DONE by the other agent — SKIP (do not duplicate)
- robust6 DEFINITION + feature set + `tab:robust6_pipeline` + production verdict → `sec:feature_selection`. ✓
- (Ledger-scan Item 1 "ADD robust6 paragraph" and Agent2 Change 2 "slot marker" are now redundant — SKIP.)

## STILL NEEDED — framing flips (sa32→robust6 "production"; baseline/patch→ablation). Apply when file stable.

### Ch4 Architecture — classifier sections (Agent abecdf package)
1. `sec:classifier_variants` header — replace "The production pick … is sa32 …" with "comparison set that
   motivated the production classifier; the shipped classifier is robust6 (§feature_selection)." (Agent2 Change 1)
2. `sec:classifier_variants` sa32 bullet — "(production)" → "(sa32; strongest hand-engineered variant,
   superseded by robust6)". (Agent2 Change 1b)
3. `sec:trust_fusion` — "the production stack uses 32 features in sa32" → "uses the six features of robust6
   (§feature_selection)…; sa32 was the working pick through the real-video comparison." (Agent2 Change 4)
4. `sec:trust_classifier` Feature Set — "the production classifier sa32 uses 32 features" → "production
   classifier is the six-feature robust6; the 32/40-feature variants are the comparison set." (Agent2 Change 5)
5. `sec:classifier_compare` Deployment-implication — "sa32 is the production pick" → "strongest *hand-engineered*
   variant; robust6 is the shipped classifier (§feature_selection)." (Agent2 Change 6) — NOTE: supersedes
   ledger-scan/Agent3 Change 7 (same anchor); apply Agent2 Change 6 only.
6. `fig:pipeline` classifier node — `(4-way, 32-feat)` → `(4-way, robust6)`. (Agent2 Change 7)
7. ADD `\paragraph{Two-verifier fusion (trust-first).}` after the Overview "A note on the verifier…"
   paragraph — classifier routes, trusted modality's MLP filters; IR always-on (recall-safe). (Agent2 Change 8)
8. ADD `\paragraph{A grayscale-fallback recall hole in robust6.}` after the dual-classifier paragraph —
   IR feats→chance on grayscale, trust_rgb recall R≈0.19; architectural fix. (Agent2 Change 3)
   [ledger robust6-grayscale-ir-features-dead, routing-failure-is-trust-rgb-recall]

### Ch1-2 Intro/Related (Agent ae831 package)
9. RQ1 — patch verifier parenthetical → per-frame distilled MLP (mlp_v5 RGB + mlp_v5_ir_aligned IR),
   superseding the patch verifier. (Agent1 edit 1)
10. Contribution 1 — "a MobileNetV3 patch verifier acts as a confuser-aware alert gate" → trusted modality's
    per-frame MLP verifier filters; patch superseded. Keep numbers. (Agent1 edit 2; + 2b dedupe source comment)
11. Contribution 5 — reframe mlp_v5 from "candidate" to production filter both modalities; add IR aligned
    clause. **USE CANONICAL IR NUMBERS: CBAM F1 0.699→0.846, FP 48→15** (not 0.841/13). (Agent1 edit 3, corrected)
12. Background — label baseline `P=0.940,R=0.959` as the Stage-1 reference vs production ft4. (Agent1 edit 4)
13. Abstract — name both production verifiers; patch = superseded predecessor. (Agent1 edit 5)
14. `tab:related_systems` caption — "patch verifier" → "per-frame distilled confuser verifier". (Agent1 edit 6)
15. `sec:lit_objdet`, `sec:lit_drone`, `sec:comparison` — residual "patch verifier" string → "per-frame
    distilled verifier". (Agent1 edits 7,8,10)

### Ch6/7 Results/Conclusion (Agent a732c package)
16. `sec:production_stack` — full reframe to ft4/v3b/robust6/mlp_v5/mlp_v5_ir_aligned + trust-first fusion;
    baseline/selcom/sa32/fnfn/patch → explored alternatives; keep 3 honest carve-outs. (Agent3 Change 1)
    — RE-VERIFY anchor; the other agent already rewrote this paragraph.
17. `sec:production_stack` future-work — sa32-default selector → robust6 + grayscale routing gate. (Agent3 Change 2)
18. RQ1 answer — tag 52.1%→10.3%/0.8% as the comparison config; add robust6 production reproduction. (Agent3 Change 3)
19. RQ2 answer — tag Svanström/real-video numbers as comparison config; defer production to placeholder. (Agent3 Change 4)
20. `sec:cumulative` config note — "this is the COMPARISON config, not production." (Agent3 Change 5)
21. `sec:realvideo_classifier` — "sa32 as the default … production-pick flip" → sa32 = comparison winner;
    robust6 shipped. (Agent3 Change 6)
22. `ch:conclusion` intro — production stack = integration of component winners. (Agent3 Change 8)
23. `sec:limits` Latency — stage list "MobileNetV3" → two small MLP verifiers (cheaper). (Agent3 FLAG-B, optional)

### ADD — mlp_v5 recall-drop MRI (user's explicit task + ledger-scan Item 6)
24. `sec:distill_verifier` — after "the coverage boost was net-negative, −3 pp recall) … train/test distribution
    mismatch", append the MRI diagnosis: verifier vetoes OOD small *real drones* (deep-feature signature far
    from confusers; conf gap kept-vs-vetoed ≈0; vetoed-vs-confuser AUROC 0.876 ≥ kept 0.862) → it's an OOD
    drone-coverage gap; prescribed fix = targeted drone-diversity re-mine (RGB analog of the IR fix).
    [ledger mlp-v5-recall-drop-is-ood-coverage; cache docs/analysis/2026-06-01_mlp_v5_recall_drop_mri.md;
    config rgb_dataset_iou_640] — NOTE: no evals.csv row; cite doc as cache. Use figure if generated, else prose.

### ADD — offline verifier matrix + grayscale over-veto (ledger-scan Item 4) — OPTIONAL
25. `sec:grayscale_verifier` — single re-runnable offline harness confirms both production verifiers; exposes
    grayscale over-veto (gray drone recall 0.548→0.164) → gate grayscale, not per-frame.
    [ledger offline-verifier-matrix; cache docs/analysis/2026-06-01_full_pipeline_offline_eval.md]

## Sourcing caveat
robust6 / dual-verifier-fusion-rule / offline-verifier-matrix / mlp-v5-recall-drop have **no `evals.csv` rows** —
numbers live in ledger notes + the cited analysis docs. Cite `eval=none` + `cache=<doc>` (existing thesis
convention) OR promote to evals.csv via `/record` first. Never invent numbers.

## Apply procedure (when file is stable)
1. Confirm other agent paused. 2. `git` snapshot / note current state. 3. Apply 1–25 (skip dupes) via Edit,
   re-reading regions as needed. 4. `powershell -File docs/build_thesis.ps1` → expect ~128–130 pp, exit 0.
5. `py knowledge/_tools/thesis_tools.py hygiene` (0 undefined, 0 rule-#2) + `thesis_audit.py`.
