# Final Review — Chapter 4 (empirical.tex) — 2026-06-18

READ-ONLY review of `docs/thesis_working_distilling_overleaf/chapters/empirical.tex` (Chapter 4,
results, ~799 lines). Verification backbone: `thesis_eval/_audit_headline_numbers.py`, frozen
JSONs, `thesis_eval/_filter_swap/final/offline_matrix_v4.txt`, `knowledge/{evals,ledger,figures}.csv`.

Status: COMPLETE. Read-only; no thesis/code edits. Headline audit: 180/180 pass (but does NOT cover the 3 stale display surfaces found below).

**Headline finding:** the v4 filter-swap regenerated the canonical JSONs but 3 display surfaces still carry pre-v4 numbers — (1) `tab:distill_verifier` halluc column + its figure, (2) `tab:temporal_production` mlp-filter rows + prose, (3) `tab:ablation_dut` robust8-nr TP/FN. Plus `fig8_patch_catchbar` raw-LaTeX raster leak. The grayscale-FILTER removal (issue b) is fully coherent and both figures were correctly regenerated to 2-panel.

---

## A. NUMBERS

Verified against: `_audit_headline_numbers.py` (180/180 pass), `tier1_results.json` (+ `results_noreject`, `results_dut`), `offline_matrix_v4.txt`, `runs/clean_split`, the per-category/version CSVs, `ir_heldout_results.json`, `mri` stats, `knowledge/evals.csv`. ✓ = matches source; ✗ = mismatch; ⚠ = minor/rounding or source-not-locatable-but-internally-consistent.

**The audit passing 180/180 does NOT cover three things this review found wrong:** (i) the `tab:distill_verifier` halluc column, (ii) several `tab:temporal_production` mlp-filter cells, (iii) the `tab:ablation_dut` robust8-nr TP/FN integer columns. The audit pins JSON *F1* cells, not these.

| number | location | source | BACKED |
|---|---|---|---|
| **distill halluc Svan 0.037** | tab:distill_verifier L589 | offline_matrix_v4 L54 = **0.044** | **✗ STALE-v5** |
| **distill halluc rgbds 0.010** | L592 | matrix L46 = **0.029** | **✗ STALE-v5** |
| **distill halluc confuser 0.008** | L593 | matrix L42 = **0.021** | **✗ STALE-v5** |
| distill halluc Anti-UAV 0.010 | L590 | matrix L10 = 0.012 | ⚠ also stale |
| distill halluc SelCom 0.019 | L591 | matrix L50 = 0.0225 | ⚠ also stale |
| distill F1 Svan 0.861 / rgbds 0.916 / AntiUAV 0.984 / SelCom 0.612 | L589-592 | matrix L54/L46/L10/L50 | ✓ (F1 col correct v4) |
| **temporal filt(mlp) R0.513/F10.665/fire0.236** | tab:temporal_production L339 | temporal_results.json filt_mlp = **0.489/0.6465/0.213** | **✗ STALE** |
| **temporal clf→filt[robust8-nr] 0.513/0.665/0.236** | L345 | JSON clf->filt[robust8_nr_drop] = **0.489/0.6465/0.213** (= audit `NR video composed F1`0.646/`fire`0.213) | **✗ STALE** |
| **temporal clf→filt[robust8] F1 0.561 / fire 0.098** | L346 | JSON = **0.5436 / 0.0756** (= audit `video composed r8 F1`0.5436) | **✗ STALE** |
| **temporal clf→filt[robust6] F1 0.593/R0.424/fire0.075** | L347 | JSON = **0.5833/0.4121/0.0585** | **✗ STALE** |
| temporal clf→filt patch[sa32] 0.535/0.689/0.080 | L348 | JSON 0.5348/0.6891/0.0797 | ✓ |
| temporal bare/clf[robust8/6/sa32] | L338,342-344 | JSON | ✓ all match |
| **DUT filt→clf[robust8-nr] TP2580/FN614** | tab:ablation_dut L130 | JSON TP=**2558**/FN=**638** (P/R/F1 0.873/0.800/0.835 ✓) | **✗ TP/FN stale; derived metrics current** |
| **DUT clf→filt[robust8-nr] TP2580/FN966** | L127 | JSON TP=**2558**/FN=**988** (P/R/F1 ✓) | **✗ TP/FN stale** |
| DUT ft4/v3b-gray/fused/clf[nr] P,R,F1,TP,FP,FN | L116,117,119,123 | runs/results_dut + results_noreject | ✓ all match |
| Svan paired full table (all cells) | tab:ablation_svanstrom L27-42 | audit-pinned (NR + robust8 cells) | ✓ |
| Anti-UAV paired full table | tab:ablation_antiuav L56-71 | audit-pinned | ✓ |
| RQ3 0.607/0.940/0.944 (Svan), 0.985/0.961/0.984 (AUV) | tab:rq3 L97-99 | audit-pinned | ✓ |
| confuser table fire rates (all) | tab:ablation_confusers L155-164 | audit-pinned | ✓ |
| solo table (IR/RGB/SelCom × cells) | tab:ablation_solo L196-210 | audit-pinned (`NR` + robust8) | ✓ |
| per-size recall (rgbtest+svan buckets) | tab:per_size L232-242 | audit-pinned (`SZ *`) | ✓ |
| Svan median 29.8px / IR 14.8px | L248 | audit `SZ svan rgb/ir median px` | ✓ |
| background profile (all cells) | tab:failure_profile L263-269 | audit-pinned (`FP *`) | ✓ |
| lowconf SelCom 0.451→0.678, filt 0.612/0.692 | tab:lowconf_selcom L293-295 + prose L282 | audit `SWEEP selcom *` | ✓ |
| runtime 38.3ms→0.095ms (404×), 59-112→1.3-2.1ms (37-72×) | tab:speed L315-316 | ledger robust6-speed / v5-ship | ✓ (ledger-cited) |
| temporal predecessor +6.6..+16.5pp, baseline 0.760→0.826, FPR −39..−81% | L361 | ledger=cascade-segment-recovers | ✓ (ledger-cited, design-history) |
| RGB comparison baseline R **0.961** | tab:rgb_comparison L389 | CSV recall=**0.959** (1248/1302) | ⚠ 0.961 vs 0.959 (used as headline elsewhere too) |
| RGB comparison all halluc + P/R | L389-391 | svanstrom_1280_by_category.csv | ✓ (baseline 0.940/94.4/74.6/66.2; hardneg; retrained_v2 0.943/0.306/3.4/5.6/4.5) |
| retrained_v2 1280 R=0.323 (vs 0.306 here) | L397 | audit `RES retrained_v2@1280 R`=0.3234 | ✓ (sampling diff acknowledged) |
| in-domain rgbds F1 0.949 (retr_v2), SelCom 0.007 (1 TP/295) | L398 | ledger=retrainedv2-recall-collapse | ✓ (ledger; cross-checks tab:selcom baseline@640 R0.007) |
| SelCom table (baseline/ft2 × 640/1280) | tab:selcom L409-412 | ledger=selcom-imgsz-win; comparison.json | ✓ (ledger-cited) |
| IR evolution V2..v3b | tab:ir_evolution L438-444 | ir_v2_eval_test_640.csv (V2) + ir_comparison CSV (V3..v3b) | ✓ all match (V2 from a 2nd CSV not named in `% [source:]`) |
| IR MRI LDA 0.981 / maxF 5370 / 14697 drone / 1386 conf | L679 | audit `MRI ir *` | ✓ |
| RGB MRI LDA ~95% / maxF 42,346 / silhouette 0.067 | L673,679 | mri stats.json (verify_v7 ✓) | ✓ value; **caption "mean 2,006" mismatches fig "median 657"** (see D) |
| MRI 32,931-detection corpus | L673 | (corpus size) | ✓ (consistent w/ methodology) |
| carve-out: centroid 16.5 vs 11.1, 15.4; recall 0.664→0.874, F1 0.792→0.916 | L609,612 | ledger=mlp-v5-recall-drop-is-ood-coverage; matrix L44/46 (0.922→0.916) | ✓ (0.916 = matrix; 0.874 recall = matrix R) |
| bird-test held-out 30/230 (13%) vs 91/230 (40%), AUROC 0.981 | L616 | provenance doc + eval_birdtest_heldout.py | ⚠ cited to provenance .md (not re-verified; resident script exists) |
| CBAM held-out 0.905/0.967/0.935 FP6 (Δ+0.236); patch FP41 cut-7 | tab:ir_aligned L638,643 | ir_heldout_results.json cbam@0.05 (R0.967/FP6) + matrix L12-14 | ✓ (P0.905 vs matrix 0.906 rounding) |
| ir_aligned ir_dset_final 0.969/0.928/0.948 (98) @n4806 | L639 | evals.csv:175 (R0.928, F1 0.957@n1000); internally self-consistent @4806 | ⚠ recall pinned ✓; F1 differs from 1000-img matrix by sampling; **frozen n=4806 JSON not locatable** |
| ir_aligned ir_video 0.942(80)/antiuav 0.962(68)/ no-change | L640-641 | matrix L31-34 ir_video 0.975?, antiuav | ⚠ ir_video table says 0.942 but matrix `ir_video aligned`=0.975 — **n differs (831 vs 1000); recheck** |
| held-out IR confuser 90→22 (94% vs 77%), 388 fires | L649 | provenance doc + eval_ir_heldout.py | ⚠ cited to provenance (not re-verified) |
| filter_operating RGB recall 0.956 @ ~1% (0.011) | L653,659 | audit `FIG rgb recall@0.25`0.956/`fire@0.25`0.011 | ✓ (but live-cache; tension w/ table 0.014 — C4) |
| gray three-way 0.454/0.916/0.607; 0.362/0.126/0.187; 0.543/0.621/0.580 | tab:gray_threeway L713-715 | tier1 (audit `3way *`) | ✓ all match |
| gray control 0.187→0.580 tripling; video 0.295→0.636; FPPI 0.261→0.142; clip 0.837 vs 0.840 | L721,754,761 | audit `3way *` + ledger=ir-grayscale-fallback | ✓ (F1s pinned; video/FPPI ledger-cited) |
| realvideo six-mode table (all rows) | tab:realvideo_master L734-739 | ledger=ir-grayscale-fallback; eval/eval_video_tests.py | ⚠ ledger-cited, not individually re-verified (per-clip indicative) |
| gray bird-FPPI 0.142 vs 0.139 (hard-neg-mined) | L754 | tab:realvideo_master (0.142) + L735 retrained_v2 bird 0.139 | ✓ internally consistent |
| §threats: AntiUAV 0.973→0.977 / 0.984→0.986, n=57,542 | L775 | clean_split (audit `CLEAN auv *`) | ✓ |
| §threats: Svan IR ≤7.3pp; robust8 −1.4pp (0.948→0.934); robust8-nr −3.5pp (0.946→0.911) | L775 | clean_split: r8 0.934 ✓, r8-nr filt→clf 0.9126≈0.911 ✓ | ✓ |
| §limits airplane 29.4%→2.8% on-cache, 94% held-out, −3.7pp | L791 | tab:ablation_confusers IR 0.028 ✓; ledger | ✓ |
| §limits IR confuser per-cat 35.2/12.2/0% airplane/bird/heli | L797 | audit `CAT irconf * bare` | ✓ |

---

## B. CLAIMS

Claims judged on whether evidence (ledger/evals/pinned JSON/`% [source:]`) supports the strength of wording. Most Ch4 claims are well-hedged. Items below are either fully backed (listed briefly) or need a tweak.

| claim | location | verdict + rewording |
|---|---|---|
| "the pipeline **raises** recall over bare (0.948→0.991) while adding ~29pp precision" | tab:ablation_svanstrom caption L21 + L79 | **BACKED** — pinned cells; the mechanism (per-modality scoring, not new detections) is correctly explained at L79. ✓ No change. |
| "the **filter suppresses the FPs; the reject class is a complementary net we trade away** (RQ2)" | L81 | **BACKED & well-framed.** Numbers (2019→1997 router-only, →337 filter) pinned. ✓ |
| "`filt→clf` is the better side ... we ship `filt→clf` for the recall (Svan 0.946 vs 0.931, DUT 0.835 vs 0.790)" | L83 | **BACKED** — all four numbers pinned. ✓ |
| "averaged across surfaces the no-reject router leads on the **composed** F1 ... 0.850 vs 0.744" | L85, L137, L492 | **BACKED** (consistent 3×), and the "composed not router-only" caveat is stated each time. ✓ |
| "**Dual-modality routing matches the better modality on every surface** (RQ3)" | L87 | **BACKED but read the hedge** — correctly qualified as "matches or edges" (Svan 0.944>0.940, AntiUAV 0.984≈0.985, treated as tie). Honest. ✓ |
| "the production filters are **517-D MLPs**" | L667 | **BACKED & consistent** — methodology.tex defines the 517-feature p3+p5 fused embedding in 4 places; empirical L667 matches. ✓ No change. |
| "drone/confuser **separability established independently by the Model MRI**" | L79 | **PENDING (E)** — MRI↔filter wording flagged for the pending rewording pass, not a defect. |
| "the airplane gap is now **largely closed**" / "no longer 'resist'" | L173, L791 | **BACKED** with correct residual caveat (in-distribution recall trade, not a hole). Evidence: IR 0.294→0.028 pinned + held-out 94% (ledger). ✓ |
| "**To our knowledge, no published study quantifies this specific transfer**" (grayscale) | L757 | **CLAIM of novelty** — appropriately bounded ("to our knowledge"), and the adjacent-literature contrast cites §sec:lit_xmodal_cascade. Acceptable as stated; ensure §lit actually lists the 3 adjacent literatures it points to (cross-chapter). ✓ |
| "thermal-domain confuser training **transfers along with the detection** (bird-FPPI 0.142 within sampling error of 0.139)" | L754 | **BACKED** — both numbers in tab:realvideo_master; "within sampling error" is the right hedge (per-clip indicative). ✓ |
| "the per-frame numbers **validate the GUI's alert-gate placement** for OOD video" | L355 | **BACKED reasoning**, but note the supporting composed-recall numbers (0.513, F1 0.665) are **STALE** (Section A) — the *argument* survives with the corrected 0.489/0.646, but the cited figures must be refreshed or the sentence's "0.513" updated. |
| "robust8-nr ... **degrades gracefully to the bare detector**" on single-modality | L216, L492 | **BACKED** — clf[robust8-nr] rows = bare rows exactly in tab:ablation_solo (pinned). ✓ |
| "**517-D**" / "32,931-detection corpus" / "14,697 drone / 1,386 confuser" | L667,673,679 | corpus + IR counts pinned (`MRI ir n_drone/n_confuser`); 32,931 consistent with the +RGB corpus. ✓ |
| "patch filter ... **bit-for-bit the bare detector**" on SelCom-scale crops | L216, L575 | **BACKED** — tab:ablation_solo SelCom filt(patch) = bare row (pinned); matrix L47-49 patch_v2 = bare on selcom. ✓ |
| design-history predecessor claims (cascade +F1, fnfn 52.1%→1.6%, sa32 surface reversal, 13× gap) | L361,371,469,510,537 | **BACKED** (ledger-cited, labelled design-history/predecessor throughout). ✓ Labelling discipline is good. |
| "**$404\times$ cheaper**" decision (robust8 vs sa32) | L85, L466, tab:speed | **BACKED** — tab:speed 38.3ms→0.095ms = 403×≈404× ✓ (ledger-cited). ✓ |

No claim is *unsupported*; the only claim-level risk is the alert-gate paragraph leaning on the **stale** temporal recall figures (fix the numbers, the claim holds).

---

## C. CONSISTENCY (cross-artifact) — PRIORITY

### C1. **CONFIRMED DEFECT — `tab:distill_verifier` halluc column is stale-v5 (AND the figure now matches the stale table).**
Known issue (a) fully confirmed, with a refinement. Ground truth = `thesis_eval/_filter_swap/final/offline_matrix_v4.txt` (the v4 offline matrix), corroborated by `knowledge/evals.csv:174` (`distill_verifier_v4`).

`tab:distill_verifier` (empirical.tex:589-593), `+ mlp_v5` **Halluc** column:

| Surface | thesis table | TRUE v4 (offline_matrix_v4) | verdict |
|---|---|---|---|
| Svanström | **0.037** | **0.044** (matrix L54: `mlp_v5@0.25 ... halluc=0.044`) | ✗ STALE |
| Anti-UAV | 0.010 | 0.012 (matrix L10: `halluc=0.012`) | ⚠ (table 0.010 vs 0.012 — also slightly off; v5 had 0.010) |
| SelCom | 0.019 | 0.0225 (matrix L50) ≈ 0.019? **table 0.019 vs matrix 0.0225** | ⚠ mismatch |
| rgb\_dataset | **0.010** | **0.029** (matrix L46: `halluc=0.029`) | ✗ STALE |
| confuser-only | **0.008** | **0.021** (matrix L42: rgb\_confuser `halluc=0.021`) | ✗ STALE |

The three flagged cells (Svan 0.037, rgbds 0.010, confuser 0.008) are the **old shipped `mlp_v5`** halluc values, not the v4 build the row's header (`+ mlp_v5` / caption "production `mlp_v5_v4`") claims. The **F1 column IS correct v4** (Svan 0.861 = matrix L54 ✓; rgb\_dataset 0.916 = matrix L46 ✓; Anti-UAV 0.984 ✓; SelCom 0.612 ✓), exactly as the known-issue brief stated.
NOTE the audit script does NOT guard this: `_audit_headline_numbers.py` pins only F1 cells from `tier1_results.json`; the halluc column is sourced from the offline matrix and has **no pin** → silently stale.

### C2. **CONFIRMED — the companion figure now carries the SAME stale halluc (and was partially regenerated).**
`fig8_distill_verifier.png` (fig:distill_verifier_bar, empirical.tex:599). Viewed the live PNG:
- Panel (a) Drone F1: title reads "mlp\_v5\_v4", rgb\_dataset green bar = **0.92** → **F1 panel IS now v4-correct** (this contradicts verify_v7 finding #1, which predates a regeneration: the F1 panel has since been fixed from the stale 0.79).
- Panel (b) Halluc: green `mlp_v5_v4` bars read **Svan 0.037, rgb\_dataset 0.010, confuser 0.008** (plus Anti-UAV 0.010, SelCom 0.019) — i.e. **identical to the stale table**. So table and figure are *mutually consistent* but **both wrong vs v4 ground truth** on the three cells. The fix must touch BOTH the table and the figure-generator's halluc data (repoint to offline_matrix_v4: Svan 0.044 / rgbds 0.029 / confuser 0.021), then re-copy the PNG/PDF into the live `figures/` dir.

### C3. **RESOLVED — grayscale-FILTER removal coherence in tab:ablation_confusers + the two regenerated figures.**
- `tab:ablation_confusers` (empirical.tex:151-166): columns are **RGB confusers / IR confusers only** — grayscale-filter column is gone. ✓
- Observations renumbered to **"Three observations"** (L170), "First/Second/Third" (L170/L173). No orphaned "Third observation"-vs-fourth, no "656→15", no grayscale-curve mention in this subsection. ✓ (`656`/`15` appear nowhere in the chapter.)
- `fig_pipeline_ablation.png`: viewed — now **2-panel, panel (b) has only "RGB confusers" + "IR confusers" groups** (no grayscale group). Caption (L179) matches: "RGB (30.4%→0.11%) ... thermal-confuser (29.4%→2.4%) ... shipped no-reject ... 1.4% on RGB." Fig prints RGB 30.3/10.2/1.4/4.9/0.1, IR 29.4/24.6/2.8/27.3/2.4 — matches table + caption. ✓ (verify_v7 L47 described the OLD 3-group image; it has been regenerated.)
- `fig_filter_operating.png`: viewed — now **2-panel (RGB mlp\_v5 + IR-thermal aligned)**, no grayscale panel. Caption (L659) matches ("RGB sits on a flat high-recall shoulder; the thermal-native head ... suppresses thermal confusers"). ✓ (verify_v7 L55 and `knowledge/figures.csv:29` both still describe a stale "3-panel ... gray 0.25" — that is stale *metadata*, not a thesis defect. Worth a figures.csv update but out of scope here.)

### C4. **CONFIRMED tension (note, not defect) — fig_filter_operating reads LIVE cache; tab:ablation_confusers reads FROZEN.**
Known issue (c). `fig:filter_operating` + its caption pins (and the audit's `FIG rgb recall@0.25`=0.956 / `FIG rgb fire@0.25`=0.011, lines 236-237) come from `eval/results/filter_operating_sweep.json` (live sweep): caption says "keeps **95.6%** ... at a **~1%** confuser fire-rate." But `tab:ablation_confusers` `filt only (mlp)` RGB = **0.014** (frozen `tier1_results.json`, audit-pinned `rgbconf mlp fire`=0.0144). So the same RGB-filter-at-0.25 confuser rate is quoted as **~1.1%/1.4%** (frozen table) vs **~1.0%** (live-cache figure). The figure even reads the RGB recall as 0.956 while the surrounding prose's coverage is on a different (pooled-drone) base. These are different surfaces/pools (the sweep pools drone surfaces + rgb\_bird\_confuser; the table is the rgb\_confuser test split), so not strictly contradictory, but a reader sees ~1.0% vs 1.4% for "the RGB filter on confusers." Recommend a half-sentence reconciling the two bases, or footnote that the figure's fire-rate pools a different confuser surface than the table.

### C5. Composition-order / no-reject numbers — internally consistent across prose↔tables.
Spot-checked the headline no-reject story end to end; all agree across the four full-pipeline tables, the RQ3 table, §sec:classifier_results prose, the abstract-referenced values, and the audit `NR *` pins:
- Svan production `filt→clf[robust8-nr]` F1 **0.946** (tab:ablation_svanstrom L41 = RQ3 L99 = prose L83/L492 = audit `NR svan filt->clf F1` 0.946). ✓
- Svan `clf→filt[robust8-nr]` 0.931 (L38 = L137 = L492 = audit 0.931). ✓
- DUT `filt→clf[robust8-nr]` 0.835 (L130 = L137 = L492 = audit `NR dut filt->clf F1` 0.835); `clf→filt` 0.790 (audit `NR dut composed F1` 0.79). ✓
- RQ3 routed Svan 0.944 vs IR-only 0.940 (L98/L99 = L87 = audit `svan v3b IR-only F1` 0.940). ✓
- Mean composed F1 0.850 vs 0.744 (L85/L137/L492) — consistent across all 3 statements. ✓
- Confuser fire `filt→clf[robust8-nr]` RGB 0.014 / IR 0.028 (tab:ablation_confusers L162 = audit `NR rgb_conf fire`/`NR ir_conf fire`). ✓
- robust8 (reject-class) confuser RGB 0.0011 composed (L164 = caption L149 = prose L170 "0.11%"). ✓

### C6. Minor cross-artifact rounding / wording nits (low severity)
- **30.4% vs 30.3%**: prose/captions (L170, L179) say RGB bare confuser fire "30.4%"; fig_pipeline_ablation prints "30.3%"; table value 0.3035 (audit `rgbconf bare fire`=0.3035 → 30.35%). Pick one rounding. Trivial.
- **Anti-UAV halluc 0.010 vs 0.012** in tab:distill_verifier (see C1) — the bare/patch Anti-UAV halluc 0.011 matches v4 0.012? matrix L8/L9 say 0.012. Table says bare 0.011, patch 0.011, mlp 0.010. v4 matrix says all 0.012. Minor staleness in the whole Anti-UAV halluc row, same root cause as C1.
- **SelCom halluc**: table mlp 0.019 vs matrix 0.0225; bare/patch 0.071 vs matrix 0.0707 (≈ matches). The mlp 0.019 looks stale-v5 too. Same fix.

---

## D. FIGURES (+ PROPOSED NEW)

All Chapter-4 figure images were opened and viewed (VLM). Verdict per figure: NEEDED / image-vs-caption / defects.

| Figure (label) | Loc | NEEDED? | Image↔caption | Defect / note |
|---|---|---|---|---|
| `fig:pipeline_ablation` (`fig_pipeline_ablation`) | 176 | **YES** — the one at-a-glance system result | ✓ now **2-panel, no grayscale group** (regenerated). RGB 30.3/10.2/1.4/4.9/0.1; IR 29.4/24.6/2.8/27.3/2.4; panel(a) robust8 0.941/0.948 | **CLEAN.** verify_v7 L47 (3-group/gray) is superseded. Minor: prints "30.3%" vs prose "30.4%". |
| `fig:cascade_segment_fig` (`fig8_cascade_segment`) | 364 | YES (design-history) | (not re-opened; verify_v7 ✓) | ORPHAN-ref? It IS `\ref`'d at L367 caption + L361 prose. OK. |
| `fig:ir_evolution` (`fig4_ir_evolution`) | 449 | YES | matches tab:ir_evolution | OK. |
| `fig:robust8_operating` (`fig_robust8_operating_point`) | 498 | YES (design-history) | ✓ curve + argmax dot | **NOTE (low):** green annotation "recall **0.12→0.82**" (trust_rgb class sweep) vs body L495 "**0.577→0.681** at τ=0.20" (downstream detection recall). Different axes; reader may conflate. Caption doesn't repeat 0.577 so no direct caption-contradiction. Optionally reword annotation. |
| `fig:classifier_reversal` (`fig8_classifier_reversal`) | 528 | YES | matches tab:classifiers | OK. |
| `fig:patch_catchbar` (`fig8_patch_catchbar`) | 567 | YES | values ✓ (71/64/52%, p .99/.90/.54 = tab:patch_audit) | **DEFECT (confirmed known-issue):** raw LaTeX leaks into raster — x-label prints literal `\texttt{patch\_thr}=0.5`; airplane bar prints literal `(drone-TP veto only 5.4\%)`; title overlaps the "0.90 decisiveness bar" annotation top-right. Fix generator strings to plain text + nudge label, regenerate, copy PNG/PDF. |
| `fig:distill_verifier_bar` (`fig8_distill_verifier`) | 599 | YES | **(a) F1 panel now v4-CORRECT** (title "mlp_v5_v4", rgbds bar 0.92); **(b) halluc panel STALE** (Svan 0.037 / rgbds 0.010 / confuser 0.008 = old v5, should be 0.044/0.029/0.021) | **DEFECT (known-issue a, refined):** figure was *partially* regenerated — F1 fixed, halluc NOT. Matches the stale table (C1/C2). Fix both together. |
| `fig:failopen_expanded` (`fig8_failopen_expanded`) | 619 | YES (design-history) | ✓ red/green/star/square, full-veto P≈0.88, bare P≈0.46 | Cosmetic: title "Svanstrom" (no umlaut). |
| `fig:filter_operating` (`fig_filter_operating`) | 656 | **YES** | ✓ now **2-panel (RGB + IR-thermal), no gray panel** (regenerated). RGB shoulder, shipped dot recall≈0.956 fire≈0.011; IR-thermal fire falls with thr | **CLEAN** vs caption. But see C4: reads LIVE cache (≈1.0%) vs frozen table 1.4% — live/frozen tension. `knowledge/figures.csv:29` still says "3-panel/gray" = stale metadata. |
| `fig:mri_stats` (`fig8_mri_lda`+`fig8_mri_anova`) | 669 | YES | (a) LDA ~0.95 ✓; (b) ANOVA p5 outlier ~42k ✓ | **DEFECT (caption stat mismatch, confirmed):** anova fig annotates dashed line "**median F=657**"; caption (L673) says "**mean 2,006**". Figure plots a stat the caption doesn't name, names one it doesn't plot. Cite median 657 (or both) in caption. Parent label never `\ref`'d (orphan). |
| `fig:mri_activation` (`fig8_mri_act_drone`+`_confuser`) | 682 | YES (qualitative) | (qualitative brain-scan) | Parent never `\ref`'d (orphan), but discussed in prose L687/L692. |
| `fig:grayscale_qualitative` (`fig_grayscale_panel`) | 745 | YES | ✓ L RGB 0.77 / M IR-gray KEEP IoU 0.81 / R IR-raw collapse IoU 0.00 | OK, exact. |

### Orphan figures (defined but never `Figure~\ref`'d) within Ch4
`fig:pipeline_ablation`, `fig:cascade_segment_fig` (actually ref'd — OK), `fig:ir_evolution`, `fig:classifier_reversal`, `fig:patch_catchbar`, `fig:distill_verifier_bar`, parents `fig:mri_stats`, `fig:mri_activation`. Several ARE discussed in prose but the explicit `\ref` was dropped → they float (the "looks randomly placed" complaint). Add one `\ref` each at the natural sentence (e.g. `fig:pipeline_ablation` at the "Three observations, drawn together in Figure~\ref{...}" — wait, L170 DOES say "drawn together in Figure~\ref{fig:pipeline_ablation}", so that one IS ref'd). Re-verify: actually the chapter refs `fig:robust8_operating` (L495), `fig:filter_operating` (L653), `fig:failopen_expanded` (L612), `fig:grayscale_qualitative` (L721/L754), `fig:ir_evolution` (none in-body — caption only), `fig:pipeline_ablation` (L170 ✓), `fig:classifier_reversal` (none), `fig:patch_catchbar` (none), `fig:distill_verifier_bar` (none), `fig:cascade_segment_fig` (none in-body, mentioned via "below"/Table only). **True Ch4 orphans needing a ref: ir_evolution, classifier_reversal, patch_catchbar, distill_verifier_bar, cascade_segment_fig, mri_stats(parent), mri_activation.** This is hygiene (verify_v7 fix #3), not an integrity defect.

### PROPOSED NEW figures
1. **A "no-reject vs reject-class" recall/confuser-fire scatter across the 6 surfaces.** The single biggest *new* story (robust8-nr shipped) is currently told only in dense tables + the design-history `fig:robust8_operating`. A small 2-axis figure (x = confuser fire, y = drone recall; one point per surface, paired robust8 vs robust8-nr arrows) would make "trade a little fire for a lot of recall, everywhere except the confuser flagship" visible at a glance and would directly serve RQ2. **Highest-value addition.**
2. **Composition-order dial (`filt→clf` vs `clf→filt`) on the recall axis.** Currently a prose claim ("filt→clf is the better side") + scattered cells. A tiny grouped bar (Svan/DUT × two orders, recall) would anchor §sec:pipeline_paired reading 3.
3. **Per-size recall figure** for `tab:per_size`: the "carve-out closed" story (sub-32px bare vs +filter, RGB-test + Svanström) reads better as 2 small grouped-bar panels than a 3-block table; reuses pinned numbers. Optional.
(No NEW figure is strictly required for correctness; #1 is the one that would most improve the chapter's strongest contribution.)

---

## E. MRI↔filter wording locations (DO NOT FLAG — pending change)

Per instructions, NOT treated as defects — listed for the pending MRI↔filter rewording pass. All Ch4 spans where the "Model MRI" instrument is credited with diagnosing/establishing a filter property:

- **L4** (chapter intro): "The **Model MRI findings** (Section~\ref{sec:mri_findings}) ... close the results".
- **L79** (pipeline reading 1): "...removed by the `mlp_v5` filter, **whose drone/confuser separability is established independently by the Model MRI** (Section~\ref{sec:mri_findings})."
- **L609** (carve-out diagnosis): "**The Model MRI localised why**: the falsely-vetoed drones were not low-confidence ... farther from the confuser distribution ... centroid distance 16.5 vs 11.1 ..."
- **L664** §heading: `\section{Model MRI Findings}` (`sec:mri_findings`).
- **L667** (section opener): "The instrument is described in Section~\ref{sec:model_mri}; this section reports what it found... why the production filters are 517-D MLPs rather than CNNs..."
- **L671-675** `fig:mri_stats` caption: "**Model MRI** of the FT4 detector's fused p3+p5 ROI features..."
- **L678-680** ("signal is linear, supervised, deep"): LDA/PCA/ANOVA framing; "a small trained MLP is the right reader."
- **L687** `fig:mri_activation` caption: "**Model MRI** spatial 'brain scan'... This activation-signature difference is what the feature-reuse filter reads."
- **L691-693** ("instrument audits its own paper trail"): "Re-running the **MRI** ... corrected two stale figures and one headline claim ... The same diagnostic pattern (localise the failure in feature space, prescribe the recall-safe fix) recovered both the RGB carve-out diagnosis (Section~\ref{sec:verifier_results}) and the IR filter (Section~\ref{sec:ir_xmodal_verifier})."

(Cross-chapter the same instrument is also cited in methodology §3.7 and the §threats/abstract; this list is Ch4-only per the slice.)

---

## E. MRI↔filter wording locations (DO NOT FLAG — pending change)

_TBD_

---

## F. TOP ISSUES (ranked)

**Root cause of the top 3: the v4 filter-swap (mlp_v5 → mlp_v5_v4 / thermal-native) regenerated the canonical JSONs (mtime 2026-06-18 01:39) but several display surfaces still carry pre-v4 numbers.** The headline-number audit passes 180/180 because it pins the *current* JSON cells, not these stale display surfaces.

1. **[CONSISTENCY — must fix] `tab:distill_verifier` halluc column is stale-v5, AND `fig8_distill_verifier` panel (b) carries the identical stale values.** Three cells wrong vs `offline_matrix_v4`: Svan 0.037→**0.044**, rgb_dataset 0.010→**0.029**, confuser 0.008→**0.021** (Anti-UAV 0.010→0.012 and SelCom 0.019→0.0225 also drifted). The F1 columns of BOTH table and figure are already correct v4 (Svan 0.861, rgbds 0.916). The figure was *partially* regenerated (F1 fixed, halluc not). **Fix:** update the 5 halluc cells in the table to v4 values; repoint the figure generator's halluc data to `offline_matrix_v4`; regenerate + **copy PNG/PDF into the live `figures/` dir** (generator writes to `docs/figures/`, not the live thesis dir). Unguarded by the audit.

2. **[CONSISTENCY — must fix] `tab:temporal_production` mlp-filter rows are stale to the pre-v4 filter** (and the §sec:temporal_results prose + alert-gate argument quote the stale numbers). Every cell that involves the mlp filter is wrong: `filt only (mlp)` and `clf→filt[robust8-nr]` print **0.513/0.665/0.236** but the canonical JSON (and the audit pins `NR video composed F1`=0.646 / `fire`=0.213) say **0.489/0.6465/0.213**; `clf→filt[robust8]` prints F1 **0.561**/fire 0.098 vs JSON **0.5436**/0.0756; `clf→filt[robust6]` prints 0.593/0.424/0.075 vs **0.5833/0.4121/0.0585**. Clf-only (router) rows are correct. Prose L355 ("recall 0.513, F1 0.665") and L357 ("R 0.396→0.507", "0.424→0.577") inherit the stale starts. **Fix:** refresh the table + the two prose sentences from `thesis_eval/results{,_noreject}/temporal_results.json`. The alert-gate *argument* survives the correction.

3. **[FIGURE — must fix] `fig8_patch_catchbar` has raw LaTeX leaking into the raster.** X-axis prints literal `\texttt{patch\_thr}=0.5`; airplane bar prints literal `(drone-TP veto only 5.4\%)`; title overlaps the 0.90-bar annotation. Bar values are correct. **Fix:** plain-text the matplotlib strings, nudge the label, regenerate + copy.

4. **[NUMBER — should fix] `tab:ablation_dut` robust8-nr TP/FN integer columns are stale** (P/R/F1 are current). Both nr rows print TP **2580** / FN **614** (filt→clf) and **966** (clf→filt); JSON says TP **2558** / FN **638** / **988**. 2580/614 doesn't reproduce the printed R=0.800 (2558/638 does). **Fix:** repoint the two TP/FN pairs to the current cache.

5. **[CONSISTENCY — note/half-sentence] `fig:filter_operating` reads the LIVE sweep cache (RGB ≈1.0% fire, recall 0.956) while `tab:ablation_confusers` reads the FROZEN cache (RGB filt_mlp fire 0.014/1.4%).** Different confuser pools, but a reader sees ~1.0% vs 1.4% for "the RGB filter on confusers." **Fix:** one clause noting the figure pools a different/larger confuser surface than the table, or reconcile.

6. **[FIGURE caption — should fix] `fig:mri_stats` (anova panel) caption/figure stat mismatch.** Caption says ANOVA "mean 2,006"; the figure annotates "median F=657". Cite the plotted statistic (median 657), or add both.

7. **[FIGURE — note] `fig:robust8_operating` annotation "recall 0.12→0.82" vs body "0.577→0.681 at τ=0.20".** Different axes (trust_rgb class sweep vs downstream detection recall). Now design-history, caption doesn't repeat 0.577 so no caption-contradiction; optionally reword the in-raster annotation to avoid reader conflation.

8. **[HYGIENE] Ch4 orphan figures (no in-body `\ref`):** `ir_evolution, classifier_reversal, patch_catchbar, distill_verifier_bar, cascade_segment_fig, mri_stats(parent), mri_activation`. Several are discussed in prose with the `\ref` dropped. Add one `\ref` each. (Not an integrity defect.)

9. **[METADATA — out of scope but worth a follow-up] stale `knowledge/figures.csv:29`** still describes `fig_filter_operating` as "3-panel ... gray 0.25"; the live image is 2-panel. Same for the verify_v7 doc's pre-regeneration descriptions of `fig_pipeline_ablation`/`fig_filter_operating`/`fig8_distill_verifier`. Update knowledge metadata after the figure fixes land.

### Low-severity / acceptable-as-is (not blocking)
- `tab:rgb_comparison` baseline Drone R **0.961** vs CSV-derived **0.959** (1248/1302) — used as a headline elsewhere; reconcile to 0.959 or confirm the rounding source.
- `tab:ir_aligned` ir_dset_final (F1 0.948 @n4806) and ir_video (0.942 @n831) differ in *absolute* F1 from the 1000-img `offline_matrix_v4` (0.957 / 0.975) — **legitimate n/surface difference; the load-bearing Δ-claims (R0.928 pinned; ir_video Δ=0) hold.** Frozen n=4806 JSON not locatable in-repo (provenance via the cited script + 2026-06-18 provenance doc).
- bird-test held-out (30/230, AUROC 0.981) and held-out IR-confuser (90→22) cited only to `docs/analysis/2026-06-18_filter_provenance_train_heldout.md` + resident scripts — not independently re-verified here (scripts exist; consistent with surrounding pinned numbers).
- `tab:ir_evolution` V2 row sourced from a 2nd CSV (`ir_v2_eval_test_640.csv`) not named in the `% [source:]` comment — values verified ✓, just add the path.
- "30.4%" (prose/caption) vs "30.3%" (fig_pipeline_ablation) RGB bare confuser fire — pick one rounding.
- `fig8_failopen_expanded` title "Svanstrom" (no umlaut) — cosmetic.

---

### Delivered
- This findings doc — absolute path:
  `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_review_empirical.md`
- No thesis/source/code files modified (read-only review). Verification backbone used: `thesis_eval/_audit_headline_numbers.py` (180/180), `thesis_eval/_filter_swap/final/offline_matrix_v4.txt`, `thesis_eval/results{,_noreject,_dut→runs/results_dut}/tier1_results.json` + `temporal_results.json`, `runs/clean_split/clean_split_results.json`, `eval/results/ir_heldout_results.json`, per-category/version CSVs, `mri/results/*/stats.json`, `knowledge/{evals,figures}.csv`, and VLM inspection of every Ch4 figure PNG.
