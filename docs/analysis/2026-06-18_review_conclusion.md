# Final Review — conclusion.tex + appendices.tex (pre-humanify)

Reviewer: read-only thesis review agent (max effort). Date: 2026-06-18.
Slice: `docs/thesis_working_distilling_overleaf/chapters/conclusion.tex` + `chapters/appendices.tex`.
Backbone: `thesis_eval/_audit_headline_numbers.py` (live), frozen JSONs, `runs/README.md`, `knowledge/{evals,ledger,models,figures}.csv`, `references.bib`, model cards.

Status: **COMPLETE**. Audit re-run live = 180/180. Slice is clean on the shipped state; 2 in-slice low/med fixes + 2 cross-chapter reconciles (conclusion is on the correct side of both). See Section F.

---

## A. NUMBERS

Backbone: `thesis_eval/_audit_headline_numbers.py` re-run live = **180/180 pass** (140 headline cells + 40 cited paths). All audit-pinned cells below inherit that ✓.

### conclusion.tex

| number | loc | source | BACKED |
|---|---|---|---|
| Svan F1 0.742→0.946 | RQ1 (L10) | `results_noreject/tier1.svanstrom.B_pipeline.{bare→filt->clf[robust8_nr_drop]}` = 0.7415→0.9459; audit NR svan filt->clf F1=0.946, bare 0.7415 | ✓ |
| Svan recall 0.948→0.991 | RQ1 (L10) | same cell: 0.9481→0.9905; audit `svan bare R`/`NR svan composed R` | ✓ |
| Svan precision 0.609→0.905 | RQ1 (L10) | same cell: 0.6088→0.9052; audit `NR svan composed P` | ✓ |
| confuser bare fire 30.4% | RQ1 (L10) | `tier1.rgb_confuser.C_confuser.bare.fire_rate`=0.3035; audit `rgbconf bare fire` | ✓ |
| filter fire 1.4% | RQ1 (L10) | `filt_mlp.fire_rate`=0.0144 (= NR filt->clf 0.0144); audit `NR rgb_conf fire` | ✓ |
| robust8 ablation 0.11% (3 frames in 2,633) | RQ1 (L10) | `clf->filt[robust8].fire_rate`=0.0011, FP=3; `rgb_confuser.meta.n`=2633 (verified); 3/2633=0.114%→0.11% | ✓ |
| thermal 29.4%→2.8% | RQ1 (L10) | `ir_confusers.C_confuser.{bare→filt->clf[robust8_nr_drop]}.fire_rate`=0.2943→0.0278; audit `irconf bare fire`/`NR ir_conf fire` | ✓ |
| IR_confusers val/test 94% removed | RQ1 (L10) | sec:ir_xmodal_verifier; eval/results/ir_heldout_results.json (90→22, audit CBAM block) | ✓ (cross-ref) |
| Anti-UAV 0.973→0.984 | RQ1 (L10) | `tier1.antiuav.B_pipeline.{bare→filt->clf[robust8_nr_drop]}`=0.9728→0.9846; audit `antiuav bare F1`/`NR antiuav composed F1` | ✓ |
| 41 FP across 4,000 Anti-UAV frames | RQ1 (L10) | `antiuav.A_bare.ft4/rgb.FP`=41; audit `antiuav ft4 FP` | ✓ |
| Svan filter 2,019→337 (83%) | RQ2 (L14) | bare FP=2019 (`svan bare FP`), `filt_mlp_rgb.FP`=337 (audit `svan filt_mlp_rgb F1` cell exists); 2019→337=83.3% | ✓ |
| confuser 835→39 FP; patch 835→282 (~7×) | RQ2 (L14) | `rgb_confuser`: bare FP=835, filt_mlp FP=39, filt_patch FP=282; audit `rgbconf mlp FP`/`rgbconf patch fire`; 282/39=7.2× | ✓ |
| router removes almost none (nr) | RQ2 (L14) | tab:temporal_production `clf only [robust8-nr]` −0% cut; tier1 nr router = passes all → filter owns it | ✓ |
| robust8 reject 82% of bare FPs at router | RQ2 (L14) | RESOLVED: `svanstrom.B_pipeline.clf[robust8].FP`=354, bare FP 2019 → removed 1665 = **82.5%** | ✓ |
| composed filt→clf R=0.991 F1=0.946; clf→filt F1=0.931; within 1.5pp | RQ2 (L14) | NR filt->clf R 0.9905/F1 0.9459; NR clf->filt F1 0.9308; Δ=1.51pp | ✓ |
| temporal cut 71–77% at 10–17 pp F1 | RQ2 (L14) | tab:temporal_production: robust8 −71%/F1 0.676 (−16.7pp), robust6 −77%/F1 0.737 (−10.6pp) vs bare 0.843 | ✓ |
| router 0.095 ms/frame; filter 1.3–2.1 ms/det; 37–404× ; 1–4% | RQ2 (L14) | ledger `robust6-speed-feature-efficiency` (0.095, 404×), `latency-edge-unmeasured` (1.3–2.1, 37–72×, 1–4%); tab:speed | ✓ (see B: "37–404×" phrasing) |
| RQ3 Svan thermal 0.940 vs visible 0.607 | RQ3 (L18) | `tier1.svanstrom.A_bare`: v3b/ir 0.9401, ft4/rgb 0.6067; audit `svan v3b IR-only F1`/`3way RGB F1` | ✓ |
| RQ3 Anti-UAV visible 0.985 vs 0.961 | RQ3 (L18) | `tier1.antiuav.A_bare`: ft4/rgb 0.9853, v3b/ir 0.961; audit `antiuav ft4 bare F1`/`antiuav v3b F1` | ✓ |
| MRI linear sep ≈95% RGB / 0.981 IR | methodological (L21) | RGB LDA 0.952 (app:mri_report), IR 0.981 (audit `MRI ir LDA`) — "≈95%" rounds 0.952 | ✓ |
| IR HITL 0.503→0.967 across six revisions | methodological (L21) | ledger `ir-version-progression`; runs/README; tab:ir_evolution | ✓ |
| V5 cost 12.7 points precision | methodological (L21) | RESOLVED: tab:ir_evolution V4 P=0.895 → V5 P=0.768 = **12.7 pp** (also fig:ir_evolution caption) | ✓ |
| grayscale 0.580 vs 0.607 (2.7 pp) | findings (L25) | `svanstrom_gray` v3b 0.5796 vs `svanstrom` ft4 0.6067; Δ=2.71pp; audit `3way gray F1`/`3way RGB F1` | ✓ |
| raw-RGB 0.187 (Svan) / 0.295 (video) | findings (L25) | Svan rawrgb 0.1874 (audit `3way rawrgb F1`) ✓. Video 0.295 = empirical L739 tab:realvideo_master "IR on raw RGB ... 0.295" (table-evidenced, not audited JSON) ✓ | ✓ |
| low-conf +10 pp F1 on SelCom | Production Stack (L30) | sec:lowconf_mode; conf_sweep: selcom bare@0.25 0.5911 → filt@0.05 0.6993 (audit SWEEP rows) = +10.8pp | ✓ |
| mlp_v5 v4: rgb_dataset_test 0.809→0.922, recall 0.691→0.887 | carve-out (i) (L32) | FINAL values 0.922/0.887 = audited tier1 `rgb_dataset_test.filt_mlp` (F1 0.9222 / R 0.8873) ✓. Baseline 0.809/0.691 matches empirical **L216** (lead statement) + `2026-06-18_verify_v4_empirical.md`. CROSS-CHAPTER CAVEAT: empirical **L612** (sec:mlp_recall_drop) quotes a DIFFERENT pre-v4 baseline 0.792→0.916 / 0.664→0.874, and the cited provenance doc says recall 0.694→0.874 — three baselines disagree. Conclusion's pair is the audited-final one; the L612 wording is the stale one (empirical defect, see C/F). | ✓ (final) / ⚠ cross-chapter |
| nr trades: rgb-confuser ~1.4%, video ~3× vs robust8 | carve-out (i) (L32) | nr rgb_conf 0.0144 ✓; video nr window-fire 0.213 vs robust8 0.0756 = **2.82×** → "~3×" defensible rounding | ✓ |
| airplane thermal 29.4%→2.8% (cache)/94% (held-out) | Future Work (L36) | same as RQ1 thermal cells | ✓ |

### appendices.tex

| number | loc | source | BACKED |
|---|---|---|---|
| retrained_v2 mined 11,729 (43.4%); 183,751-img retrain_dataset | app:datasets confuser (L28) | VERIFIED in `G:/drone/rgb_confusers_merged/dataset_documentation.md` ("11,729 (43.4%)", "183,751") | ✓ |
| IR svan 21,637 (17,314/2,050/2,273 + 1,325 czoom) | app:datasets IR (L33) | `% [source: ls counts ...; leakage_controlled.json]`; matches runs/README overlap row exactly | ✓ |
| IR_confusers 5,938 (4,281 ap/1,200 bird/457 heli); split 5,237→4,000 (3,043/871/86) | app:datasets thermal (L38) | self-cite; held-out 90→22 = 94% matches CBAM/empirical L649. Counts on G:/ (raw data) — not re-counted here, consistent w/ thermal narrative | ⚠ minor (counts not re-counted; not number-critical) |
| thermal-native held-out recall 0.717→0.967 | app:datasets thermal (L38) | eval/results/ir_heldout_results.json cbam@0.05 R=0.967; audit CBAM FP=6; empirical tab:ir_aligned R 0.967 | ✓ |
| Svan paired 28,710 frames stride 3; drone 11,695/ap 6,090/bird 5,298/heli 5,627 | app:datasets Svan (L42) | self-cite + svanstrom2022dronedataset. Raw-data counts on G:/. Note: methodology uses n=28,710 too (consistent) | ⚠ minor |
| Anti-UAV 85,374 paired frames | app:datasets Anti-UAV (L47) | self-cite; raw-data count. Not number-critical | ⚠ minor |
| DUT 10,000 imgs; 5,200 train; 2,200 test/2,245 GT | app:datasets DUT (L51) | cite zhao2022dutantiuav; eval split results_dut/tier1; tab:ablation_dut. n=2,200 should match results_dut meta — see C check | ⚠ verify meta n |
| SelCom 2,076 frames (1,953+123); median √area 36.8 px | app:datasets SelCom (L56) | self-cite manifest; 36.8 px matches the project's standing imgsz-scaling note | ⚠ minor |
| YouTube drone clips: 7,151 total → 1,633 extracted; evaluated 9-clip/1,359-frame/1,234-GT | app:datasets (L81,88) | extraction_manifest.json; temporal meta n=1,359/1,234 GT matches empirical L328+L726 | ✓ |
| YouTube confuser clips: 36,949 → 1,250 | app:datasets (L110) | extraction_manifest.json; empirical confuser-clip 1,250 (L328) | ✓ |
| AirBird 10,000 imgs; 21.9% bg-negative | app:datasets RGB (L10) | `% [source: rgb_dataset/dataset.yaml]` — **this file does NOT exist in repo and NOT on G:/drone/rgb_dataset/** (glob=0; only G:/drone/IR_dset_final/dataset.yaml exists). Counts not independently confirmable. Source-path is misleading. See A6/F. | ✗ path |
| MRI report verbatim: 19,334 drone/13,597 confuser; LDA 0.952; recall cost 1.1%; FP cut 97.4%; recall ret 98.9%; raw halluc 54.4%; max ANOVA 42,346; AUROC 0.811/0.844; FP rate 1.4% | app:mri_report (L181-198) | VERIFIED faithful vs `mri/results/v5_report_regen/report.md` (every value matches; top-feature list identical). Minor: appendix verdict-block omits report's "raw hallucination not measured (feature-only input)" clause but keeps the table's 54.4% — internally fine | ✓ |
| provenance table cells (Svan 0.742→0.946 robust8-nr filt→clf; confuser 30.4→1.4→0.11; etc.) | app:provenance tab:provenance (L223-235) | mirrors runs/README.md; all cells = audited tier1 cells | ✓ (mirror verified vs runs/README) |

---

## B. CLAIMS

| claim | loc | verdict + rewording |
|---|---|---|
| "raising, not merely maintaining, detection performance" / "lifts drone F1 0.742→0.946, recall rising 0.948→0.991" | RQ1 (L10) | BACKED — recall AND F1 both rise (audited). The strong "raising not maintaining" framing is earned. ✓ |
| "every discrimination they cannot learn without losing recall ... is relocated into cheap, learned, downstream stages" | intro (L4) | BACKED (thesis-position framing, supported by the ablation tables). Acceptable as a thesis claim. ✓ |
| "the per-frame filter is the workhorse: on Svanström it removes 83% of bare FPs ... while the no-reject router ... removes almost none on its own" | RQ2 (L14) | BACKED — filter-only 2019→337 (83%); nr router −0% (tab:temporal_production / tier1 clf[robust8_nr_drop] removes 1.1%). ✓ |
| "Composition order is a recall/precision dial ... within 1.5 pp, so the choice is operational, not structural" | RQ2 (L14) | BACKED — filt→clf 0.946 vs clf→filt 0.931 = 1.51 pp. The "within 1.5 pp" is exactly on the boundary (1.51 rounds to 1.5). Defensible; could soften to "about 1.5 pp" to be safe. ✓ (minor) |
| "a threshold sweep over the cached probabilities shows the cost is structural rather than tunable (true-drone probabilities smear across the operating range)" | RQ2 (L14) | BACKED — matches empirical L357 (video_thr_sweep; probs smeared [0.01,0.25)). ✓ |
| "the predecessor's alert-gated veto on the same surface gained F1" | RQ2 (L14) | BACKED — empirical L361 design-evolution (+6.6 to +16.5 pp). ✓ |
| "37–404× cheaper than their predecessors" | RQ2 (L14) | OVERSTATED-PHRASING (minor): 404× = router speedup, 37× = filter speedup — they are TWO different stages, not one stage's range. tab:speed lists them separately (404× / 37–72×). "37–404×" reads as a single contiguous range. REWORD: "(the router 404×, the filter 37–72× cheaper than their predecessors)" or "between 37× and 404× cheaper depending on stage". |
| "the routed pipeline matches or beats the better single modality on each" | RQ3 (L18) | BACKED — Svan routed 0.946 ≥ IR 0.940; Anti-UAV routed 0.984 ≥ RGB 0.985? **0.984 < 0.985 by 0.1 pp.** "matches or beats" — 0.984 vs 0.985 is a statistical tie (within CI) but literally 0.1 pp BELOW. Strictly "matches" (within CI), not "beats", on Anti-UAV. Acceptable under "matches OR beats" since it matches; flag as borderline. ✓ (borderline — relies on "matches") |
| "each modality is scored against its own ground truth and never unioned; under a union rule the same detections would mis-attribute the contribution by double-digit F1" | RQ3 (L18) | BACKED by sec:scoring_audit (2.8 pp swing is the audited number; "double-digit F1" is the coverage-vs-trust framing). Mild: the audited swing is 2.8 pp (not double-digit) for the SHIPPED comparison; "double-digit" applies to per-modality union mis-attribution specifically. Acceptable w/ the section cross-ref. ✓ |
| "linear separability ≈95% RGB / 0.981 IR" | methodological (L21) | BACKED — RGB LDA 0.952 (app:mri_report), IR 0.981 (audit). "≈95%" for 0.952 is fair. ✓ |
| "Re-running it on its own shipped corpus corrected two figures in an earlier report" | methodological (L21) | BACKED — ledger `mri-v5-report-regen`; sec:mri_findings. ✓ |
| "the one revision that bypassed the review loop (V5) cost 12.7 points of precision and is reported as the protocol's clearest negative result" | methodological (L21) | BACKED — V4→V5 P 0.895→0.768 = 12.7 pp; narrative matches empirical L457. ✓ |
| grayscale "still detects drones when the visible channel fails ... within 2.7 pp F1 ... tying it on the hardest bird-cluttered clip" | findings (L25) | BACKED for 2.7 pp (0.607 vs 0.580). "tying on hardest bird-cluttered clip" = per-clip claim (flock_of_seagulls) — supported by the memory/IR-grayscale-video finding + tab:realvideo_master. ✓ |
| "Its raw-RGB control collapses on both surfaces (F1=0.187 Svan, 0.295 video), isolating the single-channel conversion as the load-bearing step" | findings (L25) | BACKED — 0.1874 (audited) + 0.295 (tab:realvideo_master). "collapses" fair vs 0.580/0.607. ✓ |
| "ft4 ... the only confuser-injection recipe that passed every drone-recall regression gate" | Production Stack (L30) | BACKED — repeated/consistent w/ sec:training_recipes + glossary; comparison detectors (retrained_v2 R=0.306) fail the gate. ✓ |
| "960 is the measured deployment optimum for the SelCom camera" | Production Stack (L30) | PARTIALLY EVIDENCED — appendix DUT row evals @960; the SelCom-specific 960-optimum is a project finding (grayscale-gap memo) but no inline source on this line. Acceptable (cross-ref'd elsewhere); could add a source comment. ✓ (minor) |
| "+10 pp F1 on SelCom at unchanged confuser safety" | Production Stack (L30) | BACKED — selcom 0.591→0.699 @floor 0.05 = +10.8 pp (audit SWEEP rows); "unchanged confuser safety" = the filter owns precision (sec:lowconf_mode). ✓ |
| carve-out (i): "shipped no-reject robust8-nr trades confuser suppression for recall" | carve-out (L32) | BACKED — the central honest trade; consistent with abstract/empirical. ✓ |
| carve-out (ii): "thermal-native IR filter ... operator-GUI wiring is an open engineering item ... as is Jetson-class edge latency" | carve-out (L32) | BACKED — matches empirical L787 (edge latency unmeasured) + ledger `latency-edge-unmeasured`; GUI-wiring is the standing open item (memory). Honest. ✓ |
| "Two carve-outs accompany it" | carve-out (L32) | CONSISTENCY-CRITICAL — verify count. The text lists: closed rgb_dataset gap (now CLOSED, not a carve-out) + (i) nr trade + (ii) GUI/latency. So the live carve-outs = (i) and (ii) = TWO. The rgb_dataset gap is explicitly "now closed". "Two carve-outs" is CORRECT (was three pre-v4). ✓ See C. |
| Future Work track-classifier "the most promising single direction ... aimed at the airplane class" | Future Work (L36) | BACKED as forward-looking (no number claimed beyond the 29.4%→2.8% already verified). ✓ |
| "open-world fusion_no_fn_v1.1 router remains selectable where missed drones are operationally cheaper than false alarms" | Future Work (L36) | BACKED — fnfn is the conservative/open-world fallback (glossary + app:models); consistent. ✓ |

---

## C. CONSISTENCY

Cross-checks the brief specifically called out. Shipped state = **robust8-nr router, filt→clf order, mlp_v5_v4 RGB filter, single thermal-native IR filter (mlp_aligned_thermalonly), NO grayscale/two-head filter**.

### C1. "Two carve-outs" count (was three) — CORRECT ✓
conclusion L32 opens "Two carve-outs accompany it". It then:
- declares the **rgb_dataset coverage gap CLOSED** by the v4 bird-split build (so it is no longer a carve-out);
- lists **(i)** the no-reject confuser/recall trade and **(ii)** the GUI-wiring + edge-latency engineering gap.
Live carve-outs = (i) + (ii) = **two**. Matches the shipped state (the third, rgb_dataset, was retired by v4). No defect.

### C2. Production-stack line = single thermal-native IR filter — CORRECT ✓
conclusion L30 (Production Stack) names "the thermal-native IR filter (`mlp_aligned_thermalonly`, CBAM-trained, @0.05)" — a SINGLE head. No "two heads", no separate grayscale filter. Consistent with empirical sec:grayscale_verifier ("The production IR filter is a single thermal-native head") and the glossary "IR filter ... a single head". Carve-out (ii) and Future Work also say "the aligned filter" (singular). ✓ No "two heads" residue anywhere in my slice.

### C3. Router = robust8-nr, composition = filt→clf — CORRECT ✓
- conclusion L30: "trust router `robust8-nr` ... composed in the filter-then-router order (`filt→clf`, which keeps recall at the router's level)". ✓
- Headline 0.946 = `filt->clf[robust8_nr_drop]` (audited). ✓
- "`robust8` with reject is retained as an ablation and is the recommended choice on confuser-rich paired streams" — matches empirical sec:classifier_results + app:models + glossary. ✓
- tab:speed (L315) router row already says "`robust8-nr` 0.095 ms/frame". ✓

### C4. mlp_v5 = v4 bird-split build — CORRECT ✓
conclusion L30 "`mlp_v5` (RGB, the `v4` bird-split build, @0.25)"; carve-out (i) cites the v4 closure. app:models "`mlp_v5` (`v4` build)" production. Glossary "production is the `v4` bird-split build". Consistent. ✓

### C5. Headline numbers vs Ch4 — CONSISTENT ✓ (one cross-chapter caveat)
- Svan 0.946, R 0.991, P 0.905 → empirical tab:ablation_svanstrom + abstract. ✓
- thermal 29.4%→2.8% → empirical tab:ablation_confusers / sec:ir_xmodal_verifier. ✓ (the brief's "0.946 / 29.4%→2.8%" both confirmed)
- Anti-UAV 0.973→0.984 ✓; RQ values (0.940/0.607, 0.985/0.961) ✓.
- **CAVEAT (cross-chapter, not in my slice but affects a conclusion number's pedigree):** the v4 rgb_dataset_test baseline. Conclusion + empirical L216 say **0.809→0.922 (R 0.691→0.887)**; empirical L612 (sec:mlp_recall_drop) says **0.792→0.916 (R 0.664→0.874)**; the cited provenance doc says **R 0.694→0.874**. The conclusion's FINAL pair (0.922/0.887) is the audited-canonical `filt_mlp` value — so the conclusion is on the correct side. The defect is internal to empirical.tex (L216 vs L612 disagree). Surfaced here because the conclusion inherits the L216 framing. → list in F as a cross-chapter fix for the empirical reviewer.

### C6. app:provenance mirrors runs/README.md — CORRECT ✓
tab:provenance (L223-235) rows checked against runs/README.md "Number→file→command" table:
- Svan 0.742→0.946 (`robust8_nr_drop`, filt→clf), file `results_noreject/tier1_results.json` (svanstrom.B_pipeline) — **matches runs/README L18.** ✓
- Confuser 30.4%→1.4%→0.11%, `results_noreject/...rgb_confuser.C_confuser` — matches README L19. ✓
- Anti-UAV 0.973→0.984; grayscale 0.607/0.187/0.580; temporal; per-size; background; low-conf; IR HITL 0.503→0.967; latencies; scoring 2.8 pp; overlap audit; clean-split — all rows present in both, same files/commands. ✓
- The appendix correctly notes "frozen copies of `thesis_eval/results/`" and marks the two `(GPU)` rows (IR trajectory + latencies). Matches README's GPU annotations. ✓
- Provenance prose says the audit "asserts ... that the numbers printed in this document equal the values in the canonical results files; the submission build ... passes that audit." → **VERIFIED: audit runs 180/180.** ✓

### C7. app:models reflects the model cards — CORRECT ✓
app:models longtable production picks: ft4 (RGB), ir_v3b (IR), robust8-nr (router), mlp_v5 v4 + mlp_aligned_thermalonly (filters). robust8 = ablation/"recommended on confuser-rich paired streams". patch_v2 = superseded comparison "the v4 RGB filter removes its last fallback role". All consistent with production-stack line + glossary + empirical. v3b note "F1=0.967 ... two-epoch corrective fine-tune of Final" matches tab:ir_evolution. ✓

### C8. Stale pre-v4 values in tab:temporal_production / tab:distill_verifier / tab:ablation_dut — does conclusion/appendix quote them? — NO ✗-quote (clean)
The brief warned these three Ch4 tables may carry stale pre-v4 values.
- **tab:temporal_production:** I re-verified its robust8-nr rows against `results_noreject/temporal_results.json` — they MATCH (clf[robust8-nr]=bare=0.843; clf→filt[robust8-nr]=filt-only 0.665, fire 0.236). The conclusion's temporal claims (71–77%, 10–17 pp, "router removes almost none") all map to CURRENT cells. No stale temporal value is quoted in the conclusion.
- **tab:distill_verifier halluc:** conclusion does NOT quote any distill_verifier halluc value. (carve-out (i) uses the verifier-results F1/recall pair, audited-correct.) Appendix app:mri_report quotes the MRI projection (97.4% cut / 1.4% rate) which is the pre-training estimate, explicitly labelled as such (L201). No stale shipped-measurement is presented as current.
- **tab:ablation_dut:** appendix app:datasets DUT references tab:ablation_dut for the eval-split size (2,200/2,245) only — NOT for any pipeline metric. Conclusion does not cite DUT numbers at all. So even if tab:ablation_dut carries stale cells, neither the conclusion nor the appendix imports them. ✓
→ Net: my slice does NOT propagate the flagged stale Ch4 values. (The tables themselves are the empirical reviewer's problem.)

### C9. DUT eval n — CONSISTENT ✓
appendix "2,200 frames / 2,245 GT drones" → `runs/results_dut/tier1_results.json` meta `n=2200`. ✓ (GT 2,245 is the drone-instance count; plausible, ~1.02 drones/frame.)

### C11. Svanström production F1: 0.946 (conclusion) vs 0.944 (tab:rq3) — CROSS-CHAPTER MISMATCH on the SAME cell ⚠
The shipped cell `svanstrom.B_pipeline.filt->clf[robust8_nr_drop]` has F1 = **0.9459** (verified) → rounds to **0.946**, and the audit pins it to **0.946** (`NR svan filt->clf F1`).
- conclusion RQ1 (L10) + abstract: **0.946** ✓ (audited-correct).
- empirical **tab:rq3 (L99)**: routed production = **0.944** [0.938–0.950] — this is the SAME cell but printed as 0.944.
The conclusion RQ3 paragraph (L18) sends the reader to tab:rq3, so a reader comparing the conclusion's 0.946 against tab:rq3's 0.944 sees two headline numbers for the production system. **tab:rq3 is the SOLE outlier**: 0.946 appears correctly in conclusion, empirical **tab:ablation_svanstrom (L41, same CI [0.938–0.950])**, appendix tab:provenance (L223), and the audit. tab:rq3 (L99) prints 0.944 with the *identical* CI [0.938–0.950] — proving it is the same cell, just mis-rounded/stale. **My slice is correct; the fix belongs in empirical tab:rq3 (0.944→0.946).** The "matches or beats" claim holds under either value (both > IR-only 0.940 on Svan).

### C10. Internal denominators introduced by the conclusion — OK ✓
"three frames in 2,633" (RQ1) introduces 2,633 = `rgb_confuser.meta.n` (verified). It is the correct denominator and does not contradict the appendix (the appendix's 4,000 is the IR_confusers sample, a different surface). No conflict.

---

## D. FIGURES / TABLES (+ PROPOSED)

My slice contains NO `\includegraphics` figures. It references one figure (`fig:mri_report`) and owns four float tables. Cross-checked against `docs/analysis/2026-06-18_verify_v7_figures.md` + the audit.

### Figures referenced from my slice
- **`fig:mri_report`** (appendix L176 "the excerpt ... is its verdict block"). Defined in methodology §3.7.2 as a verbatim text-block (not an image), source `mri/results/v5_report_regen/report.md`. v7 audit confirms LDA 0.952 / F 42,346 backed, ref'd from meth+app. **OK** — the appendix's reproduction (app:mri_report) is the FULL report; fig:mri_report is the verdict-block excerpt of the same file. Slight redundancy (the verdict block appears both in fig:mri_report in Ch3 AND inside the full app:mri_report quote) but it is deliberate (figure = teaser, appendix = full reproduction). Not a defect.

### Tables owned by my slice
| Table | Loc | Needed? | Verdict |
|---|---|---|---|
| `tab:youtube_drone_clips` | app:datasets L61-85 | YES | Per-clip provenance (FPS/stride/extracted/YouTube ID). Totals 7,151→1,633 self-consistent; evaluated subset (1,359/1,234) reconciled in note L88. Useful, non-redundant. ✓ |
| `tab:youtube_confuser_clips` | app:datasets L90-114 | YES | Confuser-clip provenance; 36,949→1,250. Two clips "not preserved" honestly flagged. ✓ |
| `tab:models_evaluated` (longtable, app:models) | L133-168 | YES | The model registry (production + comparison/ablation). Matches production-stack + glossary. Generated by `docs/gen_models_appendix.py` (L126 marker). ✓ |
| `tab:provenance` (app:provenance) | L214-238 | YES | Headline-number→file→command map; mirrors runs/README.md (C6 verified). Load-bearing for reproducibility claim. ✓ |

### Redundancy / orphan notes
- The four tables are each referenced (`tab:provenance`, `tab:models_evaluated`, the two YouTube tables sit inside app:datasets prose). No orphan tables in my slice.
- `tab:youtube_drone_clips` row `flock_of_birds_attack_drone` is listed but the note (L88) says it is EXCLUDED from the evaluated set ("9-clip ... flock_of_birds_attack_drone excluded"). The table has 10 drone clips; evaluated = 9. Internally consistent (the table is the full extraction inventory; the note carves out the evaluated subset). ✓

### PROPOSED (missing) — minor, optional
1. **app:datasets RGB-corpus source pin.** The `% [source: rgb_dataset/dataset.yaml]` (L11) points to a file that exists neither in the repo nor on G:/ (verified). Either correct the path to the real manifest or add the per-source table reference (`tab:ds_rgb_components` in methodology already holds the breakdown). The 10,000 AirBird / 21.9% bg-negative numbers currently have no reachable source. (See A6 / F.)
2. **(cross-slice, not mine) `fig:distill_verifier_bar` is stale** (v7 top-fix #1): plots old mlp_v5 0.792, not v4. Relevant here only because it is the same baseline that the conclusion's carve-out (i) and empirical L612 disagree on — fixing the figure + reconciling L612 to 0.809→0.922 would make the whole v4 story consistent end-to-end.

---

## E. MRI↔filter wording locations (DO NOT FLAG — pending change; list only)

Locations in my slice where the Model MRI is tied to / credited for the filter family (the wording slated to change). Listed for the editor, NOT flagged as defects.

1. **conclusion.tex L21** (`\paragraph{The methodological thread.}`):
   > "The \emph{Model MRI}'s statistics-before-training discipline **justified the filter family by diagnosing drone/confuser separability before training** (linear separability ≈95% RGB / 0.981 IR), after which **the filters were distilled separately**, and **selected the router's eight features by leakage analysis**. Re-running it on its own shipped corpus corrected two figures in an earlier report (Section~\ref{sec:mri_findings})."
   — Three MRI↔filter linkages in one sentence: (a) MRI justifies the filter family, (b) filters distilled after the MRI diagnosis, (c) MRI leakage-analysis selected the router features.

2. **conclusion.tex L25** (`\paragraph{The findings.}` — grayscale): no MRI↔filter wording (grayscale detector finding only). [no action]

3. **appendices.tex L173-201** (`\chapter{Model MRI Sample Report}`, app:mri_report):
   - L176: "the report for the **production RGB filter's corpus**" + "the excerpt in Figure~\ref{fig:mri_report} is its verdict block."
   - L182: verdict "**Classifier strongly recommended** — large FP cut at low recall cost."
   - L201: "The report's projection (97.4% FP cut at 98.9% recall retention) is the pre-training estimate; **the shipped filter's measured behaviour** ... The agreement between projection and measurement is itself part of the instrument's validation."
   — The entire appendix chapter frames the MRI report as the filter's design/validation instrument. If the MRI↔filter framing changes, this chapter's intro + closing sentence are the touch-points (the verbatim report block itself is data, not framing).

4. **conclusion.tex L36** (Future Work): "the thermal-native filter already cut ..." — no MRI linkage. [no action]

No other MRI↔filter wording in my slice. (Glossary `app:glossary` defines MRI-adjacent terms but does not link MRI→filter.)

---

## F. TOP ISSUES (ranked)

**Headline verdict:** my slice (conclusion.tex + appendices.tex) is in strong shape. The audit passes **180/180**; all 5 \cite keys resolve; all 40 \ref targets resolve (no dangling refs/labels); NO stray `0.949`/"three carve-outs"/"two heads"/grayscale-filter residue; the production-stack, RQ answers, and provenance table all match the shipped state (robust8-nr, filt→clf, mlp_v5_v4, single thermal-native IR filter). The issues below are mostly **cross-chapter reconciles** (the conclusion is on the correct side) plus two genuine in-slice items.

### IN-SLICE (fix in my files)
1. **[in-slice, LOW-MED] app:datasets RGB-corpus source pin is unreachable (appendices.tex L11).** `% [source: rgb_dataset/dataset.yaml]` — that file exists neither in the repo nor on `G:/drone/` (both verified empty for an rgb_dataset yaml; only `G:/drone/IR_dset_final/dataset.yaml` exists). The 10,000-AirBird / 21.9%-background-negative numbers therefore have no reachable source. Fix: repoint to the real RGB-corpus manifest, or cross-reference `tab:ds_rgb_components` (methodology, which holds the verified breakdown). The confuser doc (L29) and IR yaml (L34) DO resolve and verify — only the base RGB-corpus source is dangling.

2. **[in-slice, LOW] "37–404× cheaper than their predecessors" (conclusion L14) reads as one range but is two stages.** 404× = router, 37–72× = filter (tab:speed lists them separately). Reword to "(the router 404×, the filter 37–72×)" so it doesn't imply a single contiguous speedup range.

3. **[in-slice, LOW/optional] RQ3 "matches or beats ... on each" is borderline on Anti-UAV** (routed 0.984 vs RGB-only 0.985, i.e. 0.1 pp *below*, within CI). Strictly it *matches* (CI-tie), does not *beat*, on Anti-UAV. The phrase "matches OR beats" already covers it, but if a stricter reading is wanted, "matches the better single modality (and beats it where they diverge, Svanström)" is safer.

4. **[in-slice, LOW/optional] "within 1.5 pp" (conclusion L14)** is exactly on the boundary (filt→clf 0.946 vs clf→filt 0.931 = 1.51 pp). Soften to "about 1.5 pp" to avoid an examiner recomputing 1.51 and calling it >1.5.

### CROSS-CHAPTER (conclusion is correct; fix lands in empirical.tex — surfaced because the conclusion points readers there)
5. **[cross-chapter, MEDIUM] Svanström production F1: tab:rq3 prints 0.944, everything else (incl. this conclusion + abstract + audit + tab:ablation_svanstrom) says 0.946.** Same cell, identical CI [0.938–0.950] → tab:rq3's 0.944 is the mis-rounded/stale outlier (canonical = 0.9459 → 0.946). The conclusion RQ3 sentence sends the reader to tab:rq3, so the two numbers collide for a reader. **Fix in empirical tab:rq3: 0.944 → 0.946.** (Not my file, but it's the most visible inconsistency the conclusion is exposed to.)

6. **[cross-chapter, MEDIUM] v4 rgb_dataset_test baseline disagrees across the thesis.** Conclusion carve-out (i) = **0.809→0.922 (R 0.691→0.887)**, matching empirical **L216** + the audited FINAL value (0.9222/0.8873). But empirical **L612 (sec:mlp_recall_drop)** says **0.792→0.916 (R 0.664→0.874)**, the cited provenance doc says **R 0.694→0.874**, and **fig:distill_verifier_bar still plots 0.792** (v7 figures top-fix #1). The conclusion's FINAL pair is audited-correct; the pre-v4 baseline and the figure are the stale ones. **Fix: reconcile empirical L612 + fig:distill_verifier_bar to 0.809→0.922 / 0.691→0.887** so the v4 story is consistent end-to-end. (Conclusion needs no change.)

### NON-ISSUES (verified clean — recorded so they aren't re-litigated)
- Two-carve-outs count (was three): CORRECT — rgb_dataset gap explicitly declared closed; live carve-outs (i)+(ii). (C1)
- Production-stack = single thermal-native IR filter, no "two heads"/grayscale filter: CORRECT. (C2)
- Router robust8-nr + filt→clf composition + retained-robust8-ablation framing: CORRECT throughout. (C3)
- mlp_v5 = v4 bird-split build everywhere: CORRECT. (C4)
- app:provenance mirrors runs/README.md (no-reject numbers), all rows + GPU annotations match; audit-pass claim verified live: CORRECT. (C6)
- app:models longtable matches production-stack + glossary + tab:ir_evolution: CORRECT. (C7)
- Conclusion/appendix do NOT import the flagged stale Ch4 cells (tab:temporal_production verified current; tab:distill_verifier halluc not quoted; tab:ablation_dut only its frame-count cited, =2,200 verified): CLEAN. (C8, C9)
- app:mri_report verbatim block faithful to report.md (every value): CORRECT. (A7)
- Headline numbers (Svan 0.946/R0.991/P0.905, confuser 30.4→1.4→0.11%, thermal 29.4→2.8%, Anti-UAV 0.973→0.984, 41 FP, RQ3 0.940/0.607 + 0.985/0.961, V5 −12.7pp, grayscale 0.580/0.607/0.187/0.295): ALL BACKED.

### Status: COMPLETE.

---

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_review_conclusion.md` (this findings doc) — read-only review of `chapters/conclusion.tex` + `chapters/appendices.tex`.
- No thesis/code files modified (read-only review agent).
- Verification backbone exercised: `thesis_eval/_audit_headline_numbers.py` (180/180 live), `thesis_eval/results_noreject/tier1_results.json`, `thesis_eval/results/{tier1,temporal}_results.json`, `runs/{README.md,results_dut/tier1_results.json}`, `mri/results/v5_report_regen/report.md`, `references.bib`, `knowledge/ledger.csv`, `G:/drone/rgb_confusers_merged/dataset_documentation.md`, and cross-refs into `chapters/{empirical,methodology}.tex` + `docs/analysis/{2026-06-18_verify_v7_figures.md, 2026-06-18_verify_v4_empirical.md, 2026-06-18_filter_provenance_train_heldout.md}`.
