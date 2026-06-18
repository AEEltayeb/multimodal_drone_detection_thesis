# Methodology Chapter (Ch3) — Final Pre-Humanify Review

**Scope:** `docs/thesis_working_distilling_overleaf/chapters/methodology.tex` (801 lines).
**Mode:** READ-ONLY. Findings only; no thesis/code edits.
**Backbone consulted:** `thesis_eval/_audit_headline_numbers.py` (ran: **180/180 pass, 0 failures**);
frozen JSONs (`thesis_eval/results*/`, `runs/`, `mri/results/`); `knowledge/{evals,ledger,models,figures}.csv`;
`references.bib`; `docs/analysis/2026-06-18_filter_provenance_train_heldout.md`; `docs/analysis/2026-06-18_verify_v7_figures.md`.
**Date:** 2026-06-18.

**Headline verdict:** Chapter 3 is in strong shape. Every headline number that flows into a
table/figure is audit-pinned and reproduces; all dataset arithmetic balances exactly; all
`\cite`/`\ref`/`\label` resolve (no dangling cross-references; no orphaned external section refs).
The defects are small: **one genuine prose number conflict** (`rgb_dataset_test` shipped-filter recall
**0.694** in Ch3 vs **0.664** in Ch4), **two dangling `% [source: ledger=…]` keys** (invisible in PDF
but break the integrity convention), the **median 28 vs 29.8 px** figure/body mismatch (shared with Ch4),
and a cluster of **figure-orphan / placeholder hygiene** items (mostly already catalogued in the v7 fig audit).

Status legend: ✓ backed · ✗ unbacked/wrong · ⚠ partial/needs attention.

---

## A — NUMBERS

Pinned = verified by `_audit_headline_numbers.py` (ran 180/180). "calc" = arithmetic re-derived here.

| Number | Loc | Source | BACKED |
|--------|-----|--------|--------|
| rgb_dataset 172,022 (137,506/17,307/17,209) | 20, 52, 512 | dataset.yaml; per-source table sums to 172,022 | ✓ calc |
| 134,339 pos / 37,683 neg (78.1% / 21.9%); 146,540 boxes | 52 | 134,339+37,683=172,022; 134,339/172,022=78.1% | ✓ calc |
| Per-source RGB table (59,413…3,333) Total 172,022 | tab:ds_rgb_components | sum verified | ✓ calc |
| ir_dset_final 129,130 (107,809/11,709/9,612); 72.5/27.5; 94,142 boxes | 21,135,157 | per-source table sums to 129,130 | ✓ calc (boxes≠frames, OK) |
| Per-source IR table (53,512 dv5 … 302 yt) Total 129,130 | tab:ds_ir_components | sum verified | ✓ calc |
| dv5 = 53,512 | tab:ds_ir_components | IR_dset_final/dataset.yaml prefix | ✓ |
| rgb_confusers_merged 27,024 (21,784/2,607/2,633) | 22,85 | dataset_documentation.md; table sums | ✓ calc |
| confuser test 1,754 svan (1034+311+409) of 2,633; 879 other; 66.6/33.4 | 111 | 1034+311+409=1754; 1754+879=2633 | ✓ calc |
| hard-neg mining 11,729 (43.4%); retrain_dataset 183,751 | 108 | 172,022+11,729=183,751 | ✓ calc |
| IR_confusers 5,938 (4,281/1,200/457); evaluated 5,237 (3,043/871/86) | 23,169,171 | registry; audit "irconf" cells | ✓ pinned |
| CBAM ~1,775 train+val; 180-frame probe | 169 | ledger ir_ood_cbam_raw | ✓ |
| svanstrom paired 28,710 / stride 3 / 279 seq / 640×512; 11,695 drone/6,090 ap/5,298 bird/5,627 heli | 176 | tab:datasets; ds_svanstrom | ✓ (per-class not separately audited; consistent w/ confuser splits) |
| antiuav 85,374; 59,413 RGB also in train corpus | 25,181 | matches RGB table `anti` 59,413 | ✓ calc |
| selcom 2,076 (1,953+123); median √area 36.8px; val 311/295GT | 26,187 | manifest; audit selcom cells | ✓ pinned (n) |
| YouTube 2,609 (drone 1,359/1,234GT; confuser 1,250) | 27,193 | tab:datasets; audit video cells | ✓ pinned |
| **IoP@0.5 formula + rule** | 203–207 | standard def; Svanstrom IoP memory rule | ✓ |
| **scoring swing 2.8pp** (F1 0.921 dual vs 0.949 trust-aware @ P 0.939) | 216 | audit SWING cells (0.9206 / 0.948 / 0.941) | ✓ pinned |
| **historical swing 27.7pp** (0.663 dual vs 0.940) | 216 | ledger scoring-rule-swing (dual 0.6629) | ✓ |
| imgsz=1280 two surfaces; baseline 28pp recall 0.964→0.684 @640 | 223,228 | audit RES baseline@640 0.6838 / @1280 0.9641 | ✓ pinned |
| frame budget 4,000 even-spread | 232, tab:eval_protocol | protocol | ✓ |
| 95% bootstrap CI, 1,000 resamples | 257 | protocol | ✓ |
| unified cache: detectors once @0.25; 517-D p3+p5; float32 | 262 | ledger mlp-feats-need-f32 | ✓ |
| **clean-split** svan 54 seq n=5,557; auv 61/91 n=57,542 | 296, tab:clean_split | audit CLEAN cells (5557, 57542) | ✓ pinned |
| clean svan RGB 0.572/−3.5; IR 0.867/−7.3; bare 0.684; cascade 0.935/−1.4 | tab:clean_split | audit CLEAN svan (0.5717/0.8674/0.6842/0.934) | ✓ pinned |
| clean auv 0.988/0.966/0.977/0.986 | tab:clean_split | audit CLEAN auv (0.9878/0.9656/0.9765/0.9861) | ✓ pinned |
| robust8-nr clean svan 0.944→0.911 (−3.3pp) | 301,324 | ⚠ NOT in audit; see C-3 below | ⚠ |
| overlap: IR 17,314 train (+1,325 zoom); 37.3% exact; auv 22,603/6.3%; router 214/61 | tab:svanstrom_audit | leakage_controlled.json derived (matches source comment) | ✓ (derived; not in audit harness) |
| RGB-side auv 16.0% exact (13,676 of 59,226 over 318 seg, all 91 eval) | tab:svanstrom_audit | source comment (G:/ dataset) | ⚠ off-repo path (raw data) — un-auditable but disclosed |
| baseline train-time P 0.978 R 0.915 mAP50 0.951 | 359 | Drone_detection_report.pdf §3.3.4 | ⚠ external PDF (not in repo) |
| **FT4 R3 300HN freeze15 3ep; 16pp halluc cut; freeze12 regresses** | 363,633 | ledger ft4-backbone-freeze; audit not on grid cells | ✓ ledger |
| FT4 grid R1 0.898(−2.9); R3 0.919(−0.8); A1 0.897(−3.0); A2 fail | tab:ft4_variants | docs/analysis/2026-05-26_classifier_ft4_analysis.md §2.2 | ✓ (analysis doc) |
| training recipe (70ep, bs48, imgsz640, AdamW, lr0.001/lrf0.01, wd5e-4, mosaic off@50) | tab:training_recipes | Drone_detection_report.pdf Table 3.1 (all match) | ✓ (external PDF) |
| **MRI RGB report: 54.4% halluc, LDA 0.952, F=42,346, FP cut 97.4%, recall 98.9%** | fig:mri_report | mri/results/v5_report_regen/report.md (verbatim) | ✓ |
| 19,334 drone / 13,597 confuser = 32,931 corpus | fig:mri_report, 482, 795 | 19,334+13,597=32,931 | ✓ calc + pinned (MRI ir uses separate n) |
| MRI corrected 35,098→32,931; conf F=15,000→ rank6 F=10,696; p5 F=42,346 | 497 | mri/docs/mlp_v5_report_regen.md | ✓ (analysis doc) |
| **IR MRI: LDA 0.981, maxF 5,370(p5), medF 256, halluc 1.8%, FP cut 89%, recall 99.7%; n=14,697/1,386** | tab:ir_mri_sep,656 | audit MRI ir block (all 8 cells pinned) | ✓ pinned |
| CBAM held-out R 0.967 / FP 6 (bare 48) | 656,678,679 | audit CBAM aligned FP (prose==table==canonical 6) | ✓ pinned |
| thermal head 8,112 drones / 2,045 confusers | 678 | models.csv mlp_aligned_thermalonly; provenance §2.2 | ✓ |
| thermal recall held: antiuav 0.937, ir_video 0.971, svan 0.966; ir_dset 0.965→0.928 (−3.7pp) | 678,679 | provenance §2.4; tab:ir_aligned | ✓ |
| pipeline order filt→clf Svan 0.946 vs clf→filt 0.931 | fig:pipeline 523 | empirical tab:ablation_svanstrom (0.946 / 0.931 robust8-nr) | ✓ pinned (NR cells) |
| filter veto thresholds RGB 0.25 / IR-thermal 0.05 | 527,537 | provenance; models.csv | ✓ |
| confuser_fp_examples det conf 0.82–0.86, P(drone) 0.001–0.077, thr 0.25 | fig:confuser_fp_examples | gen_dataset_figures.py; v7 fig audit confirmed exact | ✓ |
| **deferred-suppression sweep**: auv bare 0.959–0.963 (0.05–0.5); rgb_test peak 0.926@0.25; selcom 0.591 bare→0.692 filt@0.05 | 543 | audit SWEEP cells (0.9592/0.9631/0.9259/0.5911/0.6993) | ✓ pinned |
| RGB floor 0.25 / IR floor 0.40 (eval caches both @0.25) | 543 | protocol | ✓ |
| **three stances**: retrained_v2 bird 94.4→3.4%, Svan R 0.961→0.306 | 543,629 | empirical tab:rgb_comparison (exact); ledger | ✓ |
| baseline 0.940/0.961, bird 94.4/airplane 74.6/heli 66.2% | 626 | tab:rgb_comparison (exact match) | ✓ |
| hardneg heli 66.2→41.9, airplane 74.6→64.7, bird 94.2% | 628 | tab:rgb_comparison (exact match) | ✓ |
| robust8-nr feature list (8 free conf+geom + is_grayscale) | 559,771 | models.csv robust8; train_routing_robust.py | ✓ |
| AUROC 0.949 (filter prob) vs 0.842 (raw conf) | 775 | ledger filter-score-as-classifier-feature → **MISSING KEY** (B/C below) | ⚠ unbacked key |
| leakage table values (img_std 0.502/349.6 … conf_sum 0.983/0.002 …) | tab:leakage | feature_stats_ranked.csv (source comment); fig audit OK | ⚠ not in audit harness; cite-file resident |
| LDA 98.2% fusion train acc | 715,734, fig:fusion_lda | fig audit confirms fig=98.2%; consistent | ✓ |
| rgb_blurriness 0.872 pooled → 0.516 within-source | 740 | ledger blurriness-is-corpus-artifact (0.872/0.516) | ✓ |
| robust6 OOD F1 0.262 (with fingerprints) → 0.578; lean13→10 +18–26pp | 706,771 | ledger scene-fingerprint-overfit (+18–26pp); ft4-lean | ✓ |
| 56-feat matrix / 65,192 frames | 715 | ledger fusion-feature-leakage; clf_own_holdout | ✓ |
| 404× cheaper (robust6 vs fusion_no_fn) | (Ch4 echoes) | models.csv robust6 (404x) | ✓ |
| **patch filter 45,917 RGB+IR patches**; acc 0.975; per-class 0.907/0.971, 0.988/0.938, 0.893/0.712 | 783 | models.csv patch_v2 (45,917); metrics.json | ✓ |
| patch v2 audited prod (v3 over-vetoes, v4 ties v2) | 786 | ledger patch-version-ranking (v4≈v2>v1>v3) | ✓ |
| **mlp_v5_v4** parent ≈95% LDA, PCA silhouette 0.067 | 795 | ledger v5-lda-separability (0.949 binary); v7 fig | ✓ (≈95% safe; report shows 0.952) |
| **rgb_dataset_test recall 0.694 → 0.874** | 795 | provenance §1.3/1.4 says 0.694→0.874 | ✗ **CONFLICTS Ch4's 0.664** (C-1) |
| bird.v1i 728 train / 484 test (60/40 seed0); held-out 91/230→30/230; AUROC 0.981 | 795 | provenance §1.2/1.4; empirical §615 | ✓ |
| 22% coverage gap (predecessor over-veto) | 795 | provenance §1.1 ("vetoed 22%") | ✓ |
| 46–72× faster (1.3–2.1 ms vs 59–112 ms); 1–4% overhead | 799 | models.csv mlp_v5 (46–72x); patch_v2 (70–112 ms/det) | ✓ |
| 5-fold CV F1 0.9857±0.0004; reject 97% / retain 98.9% | 799 | ledger embedding-distillation-cv; fig:mri_report (97.4/98.9) | ✓ |
| neg-frame: 1,400 frames, 1.6%, 38 FP | 162,543-area | audit NEG cells (1400, 0.0164, 38) | ✓ pinned |
| IR vs RGB on shared svan confuser-seqs: 0–1.7% vs 65–74% | 162 | audit FP cells (svan conf-sky/horizon/ground 0.667/0.741/0.646; IR 0.0169) | ✓ pinned |
| IR near-domain 29.4% / 35.2% ap / 12.2% bird / 0% heli | 162 | audit irconf CAT (0.352/0.1217/0.0); bare 0.2943 | ✓ pinned |
| RGB 30.4% frame-fire (rgb_confusers_merged) | 656 | empirical tab:ablation_confusers bare 0.304; audit 0.3035 | ✓ pinned |
| training-pie 327,619 / eval-pie 151,695 | fig:datasets_pie | gen_dataset_figures.py; naive component sum 328,176 (≈557 diff) | ⚠ minor (D-note; fig-only, v7 audit passed) |

**A-summary:** of ~70 distinct numbers, **one is wrong by conflict (0.694 vs 0.664)**; two rely on a
missing ledger key (AUROC 0.949/0.842 — claim still true, key just absent); a handful rest on the
external `Drone_detection_report.pdf` (baseline train recipe + P/R/mAP) which is not in-repo (acceptable —
it is the pre-existing detector's own report, clearly attributed). Everything else is audit-pinned or
re-derived here.

---

## B — CLAIMS

| Claim (quoted span) | Loc | Verdict + rewording |
|---------------------|-----|---------------------|
| "Every number in this thesis is produced under one declared protocol… bootstrap confidence intervals on headline cells." | 198 | ✓ Backed by protocol + audit. Sound. |
| "IoP is \emph{required} by Svanstr\"om: its GT boxes are routinely larger than the drone… IoU systematically under-counts" | 207 | ✓ Matches the Svanstrom-IoP standing rule. The added "SelCom does not share that annotation problem; … an IoU@0.5 re-score was not separately run for it" is a good honest caveat. |
| "A scoring choice that can move a result by anywhere from $3$ to $28$ points… is large against most inter-system differences in the literature" | 216 | ✓ Both endpoints evidenced (2.8pp pinned; 27.7pp ledger). Defensible. |
| "the largest source of silent incomparability (re-detection under drifted settings)" | 262 | ✓ Reasonable methodological claim; not a measured number. |
| "$7.3$~pp is an upper bound on the leakage component" (IR clean-split) | 324 | ✓ Well-reasoned (RGB-with-no-Svanstrom drops 3.5pp on same seqs → difficulty vs leakage decomposition). Honest framing. |
| "the shipped no-reject \texttt{robust8-nr} shows the same pattern ($0.944 \to 0.911$, $-3.3$~pp)" | 324 | ⚠ Numbers **not in the audit harness** (the clean-split JSON is the reject-class robust8 only). Plausible and consistent in direction, but unverifiable from the frozen clean-split artifact. **Flag**: either pin 0.944/0.911 to a frozen JSON or soften to "shows the same pattern (a comparable few-pp drop)". See C-3. |
| "freezing 15 layers cleared every drone-recall regression gate, whereas freezing only 12 regressed" / "necessary AND sufficient" | 363 | ✓ Ledger ft4-backbone-freeze states exactly "freeze=15 necessary AND sufficient". Strong claim, fully backed. |
| "No formal quality assessment of incoming sources was performed in this project; the frame-level review loop below is the mitigation" | 391 | ✓ Honest limitation statement; good. |
| "The MRI \emph{diagnoses} separability; it does not itself train the filter… a filter is distilled separately" | 472–474 | ⚠ **DO-NOT-FLAG per brief** — pending revert to "MRI trains the filters". Logged in §E, not counted as a defect. |
| "The instrument audits its own paper trail." | 497 | ✓ Backed by the 35,098→32,931 / F-rank correction (analysis doc). Nice, and evidenced. |
| "filter-first ships because it edges classifier-first on Svanstr\"om ($F1=0.946$ vs $0.931$), and the two are within about a point everywhere else" | 523 (fig:pipeline) | ✓ Matches Ch4 robust8-nr cells exactly. "within about a point everywhere else" is true for Anti-UAV (0.984/0.984) but DUT is 0.835 vs 0.790 (4.5pp). **Minor overstatement**: reword "within about a point on the paired benchmarks" (DUT differs more). |
| "A fail-open filter variant… was evaluated and rejected, because the detections it releases on cluttered surfaces are exactly the false positives the filter exists to remove" | 527 | ✓ Backed (svan precision 0.887→0.631), source comment + Ch4 fig:failopen_expanded. |
| "the trust router… cannot pass its evaluation by memorising surface identity" | 561 | ✓ Backed by leakage-ratio gate (Eq + tab:leakage). Sound. |
| "$\approx$1~pp Svanstr\"om F1" cost of patch per-frame veto | 567 | ✓ Backed by ledger v5-ship-per-frame (alert-gating costs −4.0pp the other direction; the ~1pp marginal-TP veto is consistent with patch audit). |
| "competitive with the dedicated RGB detector on bird-cluttered scenes, a few points behind it overall" (grayscale mode) | 649 | ✓ Conditional correctly hedged ("a few points behind overall"); matches grayscale finding memory. Good calibration. |
| "The confuser-rejection signal already lives inside the IR detector." | 656 | ✓ Backed by tab:ir_mri_sep (LDA 0.981). Appropriately scoped. |
| "so a thermal confuser filter is not strictly required. It earns its place on the hardest OOD thermal-aerial confusers" | 656 | ✓ Honest — concedes near-domain cleanliness (1.8%) then justifies on held-out CBAM. Well-calibrated. |
| "only cost is \texttt{ir\_dset\_final} $0.965 \to 0.928$… on genuinely airplane-like drones that no threshold separates" | 678 | ✓ Backed by provenance §2.4 (ir_dset_veto_diagnosis). |
| "Dropping these features… recovers $18$--$26$~pp drone $F1$ on held-out clips" | 706 | ✓ Ledger scene-fingerprint-overfit (+18–26pp). |
| "the RGB filter's drone probability separates trust-positive from reject at AUROC $0.949$ vs $0.842$ for raw confidence" | 775 | ⚠ Claim plausible but its only `% [source:]` key (`ledger=filter-score-as-classifier-feature`) is **MISSING from ledger.csv** (C-2). The analysis doc `2026-06-02_forward_selection_study.md` is also cited — verify that doc carries 0.949/0.842; if so, the claim stands and only the ledger key needs fixing/removing. |
| "It supersedes a 4-class MobileNetV3 patch filter which… was expensive enough to require alert-gating" | 515 | ✓ Backed (45,917-patch CNN, 59–112 ms). |
| "v2 is the audited production version (v3 over-vetoes and is never shipped; v4 ties v2)" | 786 | ✓ Ledger patch-version-ranking. |

**B-summary:** claims are unusually well-calibrated — the chapter repeatedly concedes the weak side
("not strictly required", "a few points behind overall", "upper bound", "in-sample exception"). Only
two need attention: the **robust8-nr clean-split 0.944→0.911** (unverifiable from frozen artifact) and
the **AUROC 0.949/0.842** (missing ledger key). One micro-overstatement: fig:pipeline "within about a
point everywhere else" (DUT is 4.5pp).

---

## C — CONSISTENCY (cross-artifact)

**C-1 (GENUINE NUMBER CONFLICT — fix). `rgb_dataset_test` shipped-filter recall: 0.694 vs 0.664.**
- Ch3 methodology L795: "lifts \texttt{rgb\_dataset\_test} recall $0.694 \to 0.874$".
- Provenance doc §1.3/1.4: "rgb_dataset_test recall **0.694** → 0.874".
- Ch4 empirical L609/L612 (sec:mlp_recall_drop): "a drone recall collapse (**0.896 → 0.664**)" and
  "the coverage gap closes directly: \texttt{rgb\_dataset\_test} recall recovers **0.664 → 0.874**".
- **FROZEN-JSON GROUND TRUTH** (`thesis_eval/results/_filter_ab/{shipped,candidate}/tier1_results.json`,
  `rgb_dataset_test.S4_verifier`): **bare R = 0.8992**, **shipped mlp_v5 filt_mlp R = 0.6912**,
  **candidate v4 filt_mlp R = 0.8867**. So the real transition is **0.691 → 0.887** (with bare 0.899).
- This means **THREE documents each state a different pair, and NONE matches the frozen JSON exactly**:
  - Ch3 L795: **0.694 → 0.874** (both endpoints slightly off: should be 0.691 → 0.887)
  - Ch4 L612: **0.664 → 0.874** (start wrong by 2.7pp; should be 0.691)
  - provenance §1.3/1.4: **0.694 → 0.874** (bare quoted as 0.888 vs frozen 0.899)
- The **recovery endpoint** is quoted as 0.874 in all three but the frozen replay gives **0.887**
  (the v4 candidate). Ch4 `tab:distill_verifier` separately lists this surface's **F1** as 0.916, which
  is the candidate's F1 (0.9212→ rounds to 0.92, not 0.916 — minor, the table's 0.916 is from a slightly
  different cached run; the F1s are close, the RECALLS are what diverge).
- **Action (#1 substantive defect):** reconcile Ch3 + Ch4 + provenance to the frozen `_filter_ab`
  values — **shipped 0.691 → v4 0.887** (bare 0.899). At minimum the **0.664** in Ch4 is wrong and the
  **0.694**/0.874 pair in Ch3 should become **0.691/0.887**.

**C-2 (DANGLING LEDGER KEYS — fix; PDF-invisible but breaks integrity convention).**
Two `% [source: ledger=…]` keys cited in methodology do **not exist** in `knowledge/ledger.csv`
(grep count 0, and absent everywhere under `knowledge/`):
- L538 `ledger=dual-filter-fusion-rule` (backs the two-filter `trust_both` recall-first resolution rule).
- L776 `ledger=filter-score-as-classifier-feature` (backs AUROC 0.949 vs 0.842 "further free lever").
Both also cite a resident doc/analysis path, so the claims are not orphaned — but the convention is
"every cited ledger key resolves". **Action:** either add the two ledger rows (record), or drop the
`ledger=` token and lean on the already-cited analysis docs. (These are the only two missing keys; all
other ~30 ledger ids cited in the chapter resolve.)

**C-3 (UNVERIFIABLE FROZEN CLAIM). robust8-nr clean-split 0.944→0.911.** L301 (caption) + L324 (prose)
state the shipped no-reject router moves Svanstrom 0.944→0.911 on the clean split. The frozen
`runs/clean_split/clean_split_results.json` and the audit only carry the **reject-class robust8** clean
cascade (0.6842→0.934). The no-reject clean number is not in any audited artifact. See B. **Action:**
pin to a frozen JSON or soften the magnitude.

**C-4 (model-name consistency — PASS).** Names used in Ch3 all map to canonical models.csv rows:
- `mlp_v5_v4` = weight filename of model `mlp_v5_balanced_v4` (`models/verifiers/rgb_v5/mlp_v5_v4.pt`) ✓
- `mlp_aligned_thermalonly` = model id (`…/mlp_aligned_thermalonly.pt`) ✓ — note Ch3 also uses the
  longer descriptive provenance once; consistent.
- `robust8-nr` ↔ models.csv `robust8_nr_drop` (the shipped "MAIN trust router") ✓
- `robust8`, `robust6`, `sa32`, `ft4` (`Yolo26n_selcom_confuser_ft4_1280`), `v3b` (`finetune_v3b`) — all ✓.
- `mlp_v5_v4` vs Ch4 `mlp_v5` shorthand: Ch4 tables label the production filter `mlp_v5` (column header)
  while captions say `mlp_v5_v4`; this is intentional and explained, but a reader sees "mlp_v5" in
  `tab:ablation_svanstrom` rows and "mlp_v5_v4" in prose. **Low-risk** but worth one harmonising note
  (Ch4-side, not Ch3) — listed here for cross-artifact awareness only.

**C-5 (cross-references — PASS, exhaustive).** All `\label`s in methodology (73) are well-formed.
All `\ref`s from methodology resolve: 28 internal targets + every external target in **other** chapters
exists — `sec:rgb_results, sec:pipeline_ablation, sec:classifier_results, sec:verifier_results,
sec:ir_evolution, sec:failure_profile, sec:pipeline_confusers, sec:mri_findings, sec:temporal_results,
sec:lowconf_mode, sec:grayscale` (empirical.tex) and `app:datasets, app:mri_report` (appendices.tex),
`ch:hitl` (empirical). Tables `tab:rgb_comparison, tab:ir_evolution, tab:ablation_confusers, tab:selcom,
tab:classifiers, tab:patch_audit, tab:distill_verifier, tab:ir_aligned` all exist in empirical.tex.
**No dangling \ref / \label / \cite.** All 6 `\cite` keys (jiang2021antiuav, zhao2022dutantiuav,
svanstrom2021real, svanstrom2022dronedataset, zhao2023antiuav, chen2016xgboost, howard2019mobilenetv3)
are in references.bib.

**C-6 (composition reads filt→clf — PASS).** Ch3 is internally coherent on the shipped order:
- L4 chapter-intro: "alert-gate placement … by the segment-level gain" (historical, OK).
- L509–518, L537 (Two-filter fusion §): "the production composition is \emph{filter-first}"; the
  resolution rules for reject_both/trust_rgb/trust_ir/trust_both are stated and match Ch4 L12.
- **fig:pipeline caption (L523)**: "detectors, then the confuser filter, then the trust classifier… 
  filter-first ships because it edges classifier-first on Svanstr\"om (0.946 vs 0.931)". ✓ filt→clf.
- L518/L567 correctly relabel the alert-gate analysis as "the superseded patch path". Consistent.

**C-7 (grayscale-FILTER fully removed from Ch3 — PASS).** No orphaned grayscale-filter machinery in
methodology: no "two heads", no "gray scaler", no `tab:gray_thermal_auroc`, no `fig:ir_gray_align`, no
"transfer-AUROC". The IR filter is described as **a single thermal-native head** (`mlp_aligned_thermalonly`,
L537 "single thermal-native CBAM net"; L678 "a single 517-D thermal-native head"). The only grayscale
content is the **detector** grayscale-RGB *operating mode* (`sec:ir_grayscale_mode`, `_infer_grayscale`),
which is a kept finding, not the removed filter. ✓ Clean.

**C-8 (eval-integrity line at sec:eval_protocol — PRESENT).** L198: "Evaluation uses held-out test
splits, and the out-of-distribution surfaces (the confuser corpora and the real-video benchmark) are
held out of all training." ✓ The integrity statement is in place at the protocol section.

**C-9 (sec:svanstrom_audit in-sample by author choice — NOT FLAGGED).** Per brief, the in-sample
treatment of the Svanstrom audit is intentional; I did not flag it as needing a leakage caveat. (For the
record, the section is itself the leakage analysis and is internally rigorous.)

**C-10 (renumbering / transition gaps from grayscale-removal — PASS).** The §3.8.x flow reads cleanly:
architecture overview → design rationale → trust fusion → alert-gate (superseded) → temporal → resolution
→ RGB detector → IR detector (+ grayscale mode + thermal filter) → trust classifier → patch (superseded)
→ feature-reuse filter. No dangling "as removed above" / "the grayscale head" sentences. The
`mlp_v5_v4` filter is forward-referenced consistently (L515, L518, L523, L537, L567, L781, L789).

**C-11 (cross-chapter NUMBER agreement — PASS except C-1).** Spot-checked every Ch3 number that also
appears in Ch4 tables: rgb_comparison (94.4/74.6/66.2; 0.961/0.306), distill_verifier (0.792→0.916,
Svan 0.861, SelCom 0.612), ir_aligned (CBAM 0.967/6, ir_dset 0.965→0.928), pipeline composition (0.946
vs 0.931), RQ3 (IR 0.940 / RGB 0.985 / routed 0.944/0.984), neg-frame (1.6%/38), IR near-domain
(29.4/35.2/12.2/0). **All agree** — the sole conflict is C-1 (0.694 vs 0.664).

---

## D — FIGURES (+ PROPOSED)

Ch3 owns 11 figure environments (see v7 fig audit `2026-06-18_verify_v7_figures.md` for the full
cross-check; I concur with it and add Ch3-specific judgements). `fig:fusion_stats` (a/b/c) is the 12th
(subfig group). Verdicts below are **per the brief's NEEDED/REDUNDANT/BETTER** lens.

| Figure | Loc | Needed? | Verdict |
|--------|-----|---------|---------|
| `fig:datasets_pie` | 34 | NEEDED | Good global-composition orienting figure. **ORPHAN** (never `\ref`'d). Add a ref in §3.1 intro. Minor legend clip noted in v7. |
| `fig:dataset_montage` | 42 | NEEDED | Earns its place — it is the visual basis for "19–29px silhouettes vs order-of-magnitude-larger Anti-UAV" that the whole cascade rationale rests on. **ORPHAN** → this is *why* the author note "looks randomly placed" (no in-text anchor). **Add `Figure~\ref{fig:dataset_montage}`** at L45/L79 (design-rationale / drone-size prose). |
| `fig:confuser_examples` | 116 | NEEDED | Distinct (raw corpus look). Ref'd. ✓ |
| `fig:confuser_fp_examples` | 124 | NEEDED, STRONG | This is the chapter's best teaching figure — high-conf detector (0.82–0.86) vs filter P(drone) 0.001–0.077, all suppressed @0.25. Exactly demonstrates "per-detection discrimination the decision head cannot express". Ref'd. ✓ |
| `fig:drone_size_hist` | 225 | NEEDED | Carries the imgsz=1280 justification. Ref'd. ⚠ **median 28px (fig) vs 29.8px (body L228 + audit pin N["svanstrom"]…=29.8)** — reconcile to 29.8 (the audited value). Author note "what does density mean" already fixed (axis now "fraction of GT boxes"). |
| `fig:label_reviewer_home` | 409 | NEEDED (HITL is a contribution) | **PLACEHOLDER `\fbox` + ORPHAN.** Drop the real screenshot + add a `\ref` in §3.5.2. |
| `fig:label_reviewer_launch` | 418 | NEEDED | **PLACEHOLDER `\fbox` + ORPHAN.** Same. The two-window split (setup vs canvas) correctly answers the author's "wants two figures". |
| `fig:hitl_loop` | 441 | NEEDED | tikz loop with the blue disciplined / red V5-shortcut path — a genuinely useful method diagram. **ORPHAN** → add a `\ref` at L444 ("Each revision follows the same loop") or L448. |
| `fig:mri_report` | 479 | NEEDED | Verbatim MRI verdict block; numbers backed (LDA 0.952, F=42,346). Ref'd (meth+app). ✓ |
| `fig:pipeline` | 520 | NEEDED, CENTRAL | The architecture schematic; caption is the composition-order anchor (0.946 vs 0.931). **ORPHAN** — astonishingly the central pipeline figure is never `\ref`'d. **Add `Figure~\ref{fig:pipeline}`** at L509 ("The pipeline processes paired RGB and thermal-IR streams…"). High priority. |
| `fig:confuser_problem` | 530 | NEEDED | The 2-up bird(0.46→P0.00 VETO)/drone(0.85→P0.96 KEEP) decision figure. Ref'd in prose (L527). ✓ Distinct from confuser_fp_examples (different job). |
| `fig:pyside_gui` | 576 | NEEDED (deployment contribution) | **PLACEHOLDER `\fbox` + ORPHAN.** Drop the GUI screenshot + add a `\ref` in §3.8.12. |
| `fig:resolution` | 592 | NEEDED | Within-model 640↔1280 (baseline 0.684/0.964, retrained_v2 0.070/0.323). Audit-pinned. Ref'd. ✓ The author's "unfair, two models" concern is already answered (within-model Δ); v7 suggests an explicit "within-model Δ" note — optional. |
| `fig:fusion_stats` (a/b/c) | 727 | NEEDED | LDA / AUROC / leakage map for the 56-feat space. Subfigs (a)(b) are `\ref`'d; **parent `fig:fusion_stats` and subfig (c) `fig:fusion_leakage` are never `\ref`'d.** Add a ref to (c) at L740 ("The leakage map separates two groups cleanly") and optionally the parent. |

**Redundancy:** No two Ch3 figures duplicate. The three confuser figures (examples / fp_examples /
problem) do genuinely different jobs (v7 audit Table 2 agrees). The two MRI-adjacent items in Ch3
(mri_report text-block, fusion_stats) are distinct from each other and from Ch4's mri_stats/mri_activation.

**PROPOSED (missing) figures — none strictly required.** Ch3 figure coverage is complete for its
claims. Two *optional* additions if the author wants them (NOT defects):
1. A tiny **filt→clf vs clf→filt** side-by-side bar (Svan 0.946/0.931, DUT 0.835/0.790) would make the
   shipped-order decision visual rather than parenthetical in fig:pipeline's caption — but Ch4
   `tab:ablation_svanstrom` already carries it; skip unless space allows.
2. The **leakage scatter** (AUROC-alone vs leakage-ratio, scene-fingerprints upper-left / robust core
   lower-right) is *described* as fig:fusion_leakage panel (c); it exists. No new figure needed.

**Orphan hygiene (Ch3):** `datasets_pie, dataset_montage, label_reviewer_home, label_reviewer_launch,
hitl_loop, pipeline, fusion_stats(+leakage subfig), pyside_gui` are defined-but-never-`\ref`'d. The two
biggest reader-facing ones are **fig:pipeline** (central schematic) and **fig:dataset_montage** (the one
the author flagged as "randomly placed"). This matches v7 fig audit TOP FIX #3.

---

## E — MRI↔FILTER WORDING LOCATIONS (do NOT flag as defect; pending revert to "MRI trains the filters")

Per the brief, the current "MRI diagnoses but does not train; the filter is distilled separately"
framing is being reverted to "the MRI trains the filters (pos = drones, neg = confusers)". Every Ch3
location that will need the revert (so the editor can sweep them in one pass):

1. **`sec:model_mri` intro, L459** — "design decisions are settled by measuring what a detector's
   features encode **before any training run**." (statistics-before-training framing).
2. **`sec:model_mri` "What the Instrument Does", L463** — "then emits separability statistics… and a
   \texttt{stats.json}" (describes MRI as analysis-only).
3. **Subsection heading "From Statistics to a Trained Filter", L472** — the heading itself.
4. **L472–474 body** — "**The MRI \emph{diagnoses} separability; it does not itself train the filter.**
   When the measured signal is strong enough… **a filter is distilled separately** on the same 517-D
   per-detection embeddings… the embeddings the MRI extracts for analysis **also serve as** the training
   matrix". ← the core sentence to rewrite to "the MRI trains the filters".
5. **L474 (continuation)** — "The production RGB filter (\texttt{mlp\_v5\_v4}) and the thermal-native IR
   filter… follow this path: measure first, train only when the measurement says the signal is there".
6. **fig:mri_report caption, L493** — "states a verdict, the evidence for it, and the projected operating
   point of **the filter it then trains**." (already half-says "it then trains" — closest to the target
   wording; keep/align).
7. **`sec:ir_xmodal_verifier`, L654** — "A Model MRI measurement… establishes that the signal it needs
   is already present in the detector" (diagnose framing; "the IR feature space is… readable by the same
   kind of lightweight MLP").
8. **`sec:distill_verifier`, L795** — "The Model MRI characterised the parent feature space… the signal
   is real but lives in a supervised subspace, so it needs a trained classifier rather than a distance
   threshold." (diagnose-then-train-separately framing).
9. **Provenance/source comment L475** — `% [source: filter train/held-out provenance = …; trainer =
   mri/classifier.py; held-out gate = mri/holdout.py]` — already names `mri/classifier.py` as the
   trainer, which **supports** the revert direction (MRI module trains).

(Also one Ch4 location for the editor's awareness: empirical `sec:grayscale_verifier`/`sec:mri_findings`
echo "trained directly on thermal confuser crops" — already closer to the target wording.)

---

## F — TOP ISSUES (ranked)

1. **[NUMBER CONFLICT] `rgb_dataset_test` recall recovery — three docs, three pairs, none matches the
   frozen replay.** Ch3 L795 **0.694→0.874**, Ch4 L612 **0.664→0.874**, provenance **0.694→0.874**.
   Frozen `_filter_ab/{shipped,candidate}/tier1_results.json` says **shipped 0.691 → v4 0.887** (bare
   0.899). The **0.664** (Ch4) is the clearest error; reconcile all three to **0.691→0.887**.
   **Substantive — #1.** (C-1)

2. **[DANGLING LEDGER KEYS] two `% [source: ledger=…]` keys missing from `knowledge/ledger.csv`:**
   `dual-filter-fusion-rule` (L538) and `filter-score-as-classifier-feature` (L776). PDF-invisible but
   violates the integrity convention; the underlying claims are independently doc-cited. Add the rows or
   drop the `ledger=` token. (C-2)

3. **[UNVERIFIABLE CLAIM] robust8-nr clean-split "0.944→0.911 (−3.3pp)" (L301 caption + L324)** is not in
   any frozen/audited artifact (clean-split JSON holds reject-class robust8 only). Pin to JSON or soften
   to "a comparable few-pp drop". (C-3 / B)

4. **[FIGURE/BODY MISMATCH] Svanstrom drone median 28px (fig:drone_size_hist) vs 29.8px (body L228 +
   audit pin).** Reconcile the figure title to 29.8 (the audited per-frame median). Shared with Ch4 /
   v7 fig audit fix #5.

5. **[ORPHAN — central figure] fig:pipeline (the architecture schematic) is never `\ref`'d.** Add
   `Figure~\ref{fig:pipeline}` at L509. Same gap makes fig:dataset_montage read as "randomly placed"
   (author's own note) — add its ref at L45/L79. (D / v7 fix #3)

6. **[PLACEHOLDERS] three `\fbox` figures un-rendered before submission:** fig:label_reviewer_home,
   fig:label_reviewer_launch, fig:pyside_gui — all referenced as real contributions (HITL tool, GUI).
   Drop the screenshots + add in-text refs. (D / v7 fix #4)

7. **[MICRO-OVERSTATEMENT] fig:pipeline caption "the two are within about a point everywhere else"** — true
   for Anti-UAV but DUT is 0.835 vs 0.790 (4.5pp). Reword to "within about a point on the paired
   benchmarks". (B)

8. **[ORPHAN hygiene, lower priority]** add one `\ref` each for `fig:datasets_pie, fig:hitl_loop,
   fig:fusion_stats`(+ subfig `fig:fusion_leakage`). (D)

9. **[MINOR / fig-only] training-pie 327,619 vs naive component sum 328,176 (≈557 diff).** Figure-caption
   number; v7 fig audit passed it (counts "from Tables + Appendix"). Confirm the pie's de-dup/exclusion
   so the caption's "327,619" is defensible if an examiner sums the tables. Not a body-text defect.

**Not defects (verified clean):** all dataset arithmetic; all `\cite`/`\ref`/`\label`; composition reads
filt→clf throughout (incl. fig:pipeline); grayscale-FILTER fully removed (single thermal-native head, no
gray scaler / two-heads / tab:gray_thermal_auroc / fig:ir_gray_align / transfer-AUROC); eval-integrity
line present at sec:eval_protocol; model names consistent with models.csv; audit 180/180. The MRI↔filter
wording (§E) is **pending revert, not flagged**.

---

### Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_review_methodology.md` (this file).
- Read-only review; no thesis/code files modified.
