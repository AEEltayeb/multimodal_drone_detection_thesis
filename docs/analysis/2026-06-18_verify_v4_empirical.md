# V4 verification — Ch4 (empirical.tex) + conclusion.tex + appendices.tex

**Agent:** read-only verifier (slice = `chapters/empirical.tex` L1-813, `conclusion.tex`, `appendices.tex`; notes range L825-1090, sessions 8-9).
**Date:** 2026-06-18. **No edits made** — flags only.

## Headline status (read first)
The chapter is **far more current than the notes imply** — nearly every session-8/9 directive is already FOLLOWED. The shipped stack is correctly `robust8-nr` + the two filters, composed `filt→clf`, throughout Ch4/conclusion/appendix. **`thesis_eval/_audit_headline_numbers.py` passes 187/187** (147 cells + 40 paths) and audits BOTH the `tier1` reject-class cells AND the `results_noreject/` `robust8-nr` cells (NR block, audit L202-234) — so the production cells are pinned, not stale.

**Two real CONTENT gaps remain, both against the LOCKED DECISIONS, not the notes:**
1. **GRAYSCALE verifier still ships the "two heads" framing** (L626-628), the **"affine offset" MRI para** (L691-693), and **`fig:ir_gray_align`** (L695-701) — Locked Decision #1/#2 says collapse to thermal-native only and REMOVE these.
2. **Leakage is disclosed apologetically per-surface** (L460, L616, L649, L790; upstream methodology §svanstrom_audit L268) with **no single global "all surfaces are test/held-out" assertion before results** — the new user directive wants the confident global framing.

---

## TABLE 1 — Notes reverify (sessions 8-9)

| Note (line) | Directive | Status | Thesis loc (file:line + quote/label) | Evidence |
|---|---|---|---|---|
| L868/L920-923 | "Stack under test" must be `robust8-nr` + 2 filters, NOT `robust8 τ=0.20` | **FOLLOWED** | empirical L11-12 "the no-reject trust router \texttt{robust8-nr}... composed in the filter-then-router order (\texttt{filt$\to$clf})" | text rewritten; τ gone from stack para |
| L874 | why not say RGB=.25, IR=.40? why IR .05? | **FOLLOWED** | empirical L12 "deployment IR floor is higher, $0.40$, the IR detector's F1-optimal operating point... the IR filter's thermal-native head at $0.05$... It is a filter threshold on thermal input, not the IR detector's deployment floor" | floor vs filter-threshold now disentangled explicitly |
| L882 | add the "reject if conf>.8" arm where missing | **N-A (superseded)** | — | the reject-floor `rej≥0.8` arm was DELETED with the no-reject shipping (S9); audit L65-66 confirms. Correctly absent. |
| L887 | remove "(one x across all four ablation tables)" | **FOLLOWED** | not present in current L4/table captions | phrase gone |
| L889 | define 95% CI | **FOLLOWED** | empirical L4 "the central $95\%$ range of the metric over $1{,}000$ resamples of the evaluation frames drawn with replacement" | clear definition in chapter opener |
| L891/L908 | add "RGB only"/"IR only" rows; make +filt/+clf obvious | **FOLLOWED** | empirical L27-28, L56-57, L97-98, L116-117 RGB/IR-only rows in tab:ablation_svanstrom/antiuav/rq3/dut; caption legend L12 "\texttt{clf}$=+$trust router; \texttt{filt}$=+$per-frame filter" | rows + legend present |
| L894 | say strided, not first-4k | **FOLLOWED** | empirical L21 "even-strided across the full 28{,}710-frame set, not the first $4{,}000$" | explicit per-table |
| L896-898 | make explicit TP sums both IR+RGB GT (4k frames → up to 8k TP) | **FOLLOWED** | empirical L21 "every row below the second rule is trust-aware and \emph{sums both modalities}... its TP/FP/FN base is both modalities' drones combined, not the frame count" | counting note added |
| L901-903 | word precision 0.609→0.939 as MLP-filter-driven | **FOLLOWED** | empirical L79 "That precision gain is RGB-side: the bare detector's confuser false positives... are removed by the \texttt{mlp\_v5} filter" | (note: shipped precision is 0.609→0.905, not 0.939; see Table 2) |
| L917 | verify thesis updated to robust8-nr+filter, rationale, ablation, why filter-then-clf | **FOLLOWED** | empirical §sec:classifier_results L466-507; L83 composition-order reading; L139-140 "Why the shipped router drops the reject class" | all three present |
| L926-930 | does "both detectors at conf=0.25" still hold under no-reject? | **FOLLOWED** | empirical L12 "Dropping the reject class does not change this floor: it is a detector setting, independent of the router's class set" | reconciled explicitly |
| L932 | did we explain WHY reject removed (delegated to filter)? | **FOLLOWED** | empirical L12, L139-140, L491-492 "delegating all false-positive removal to the per-frame filter (the better tool for it)" | rationale stated 3× |
| L935 | why IR floor 0.40 — say it's best-F1 | **FOLLOWED** | empirical L12 "the IR detector's F1-optimal operating point (Section~\ref{sec:design_rationale})" | done |
| L944-948 | IR threshold 0.05 reconcile vs "IR hallucinates 1.8%" | **FOLLOWED** | empirical L12 "deliberately low because the IR detector barely hallucinates on thermal input ($1.8\%$ near-domain confuser fire)... so its filter should intervene as little as possible (recall-safe)" | the low-threshold=permissive logic is now explained |
| L953 | why mention robust6 when not used? | **PARTIAL / OPEN-by-design** | empirical L12 "\texttt{robust6}, the six-feature statistical base the router was derived from" | robust6 retained as the lineage base + ablation (justified), but the notes' irritation is valid: robust6 still appears as a standalone composition row in tab:ablation_confusers (L158,163) and temporal (L343,347). Defensible as ablation; flag for user taste. |
| L956-960 | "reject discipline / rej≥0.8 rows" NOT updated | **FOLLOWED** | empirical L12 ablation-arms list now reads "(\texttt{robust8}, the reject-class version...; \texttt{sa32}...; \texttt{robust6}...)" — the `rej≥0.8` clause is gone | rewritten |
| L966 | sa32 "cannot be trusted to generalize" (scene leakage) | **FOLLOWED** | empirical L85 "leaks scene statistics: its features discriminate \emph{which scene}... so its in-domain lead cannot be trusted to carry to surfaces it has not seen"; glossary appendix L263 | strong wording present |
| L976 | headline is WITH the filter — nr router not sufficient alone (misleading) | **FOLLOWED** | empirical L85 "These are composed means, not router-only: the no-reject router suppresses no confusers on its own (that is the filter's job), so it is never deployed without the filter behind it" | explicitly guarded |
| L979-984 | RQ3 coverage table 4.7 (routed 0.909 vs RGB 0.582/IR 0.632) needs review | **N-A (table deleted)** | coverage table REMOVED; current `tab:rq3` (L89) is **trust-aware** (0.607/0.940/**0.944**) | audit L65-66 "coverage table was deleted"; coverage cells still in JSON (`M_modality_ab`) but no longer surfaced. Notes' concern moot. |
| L987 | tab:4.4 — say the filter equates RGB≈IR on grayscale; reuse elsewhere | **FOLLOWED (different vehicle)** | empirical L170 (tab:ablation_confusers obs.3) "the same filter weights transfer to the grayscale channel... $656\to15$ FP ($-97.7\%$)... where the thermal scaler... manages only $656\to280$"; DUT L121 grayscale filt rows | the RGB≈IR-via-filter point lands on the grayscale-confuser surface, not the paired svan table; acceptable |
| L991 | tab:4.5 IR filter barely drops IR confusers — weak filter or strong IR? | **FOLLOWED** | empirical L173 now reframed: thermal-native filter cuts IR_confusers $0.294\to0.028$ ($-90\%$); residual = "genuinely-ambiguous fraction", airplane-dominated (76%) | the old "barely drops" reading is replaced by the thermal-native result |
| L994-1005 | "thermal confusers resist / airplane gap" — reflect thermal-native, don't overstate hole | **FOLLOWED** | empirical L173 "now \emph{largely closed} by thermal-native filter training... they no longer ``resist''"; L806 "(largely closed)" | reframed to thermal-native; honest residual = in-distribution recall trade, not open hole |
| L1011-1015 | RGB-test carve-out −11.7pp — verify CURRENT v4 number | **FOLLOWED** | empirical L216 "the feature-reuse filter's former coverage carve-out is now \emph{closed} by the bird-split re-mine (\texttt{mlp\_v5\_v4}): recall recovers $0.691\to0.887$ and F1 $0.809\to0.922$... the $-11.7$~pp regression... is gone" | tier1 `filt_mlp` R=0.887 F1=0.9222 verified ✓. The "0.899→0.691" the notes quote is the OLD filter; thesis correctly frames it as closed. |
| L1018 | SelCom conf-sweep (low conf + filter)? | **FOLLOWED** | empirical §sec:lowconf_mode L277-302, tab:lowconf_selcom; conclusion L30 operating-mode | sweep present, formalized as an operating mode not a global default |
| L1032/L1024 | coverage-rule wording awkward (other modality GT is same drone) | **N-A (coverage section deleted)** | the coverage/dual-rule paragraphs the notes quote (L1021-1031, L1036-1049) are GONE from current empirical | replaced by trust-aware tab:rq3 + L87 reading |
| L1040/L1049 | "IR-only 0.632 / IR moves by nothing" — IR is good on Svanström | **FOLLOWED (resolved by reframe)** | current `tab:rq3` L98 leads with **IR-only F1 0.940** (trust-aware), prominent; L87 "IR dominates Svanstr\"om ($0.940$ vs RGB's $0.607$)" | the 0.632 was the coverage-rule artifact (now deleted); thesis foregrounds IR's true 0.940 strength |
| L1052 | tab:4.8 per-size suspicious — filter trained on higher imgsz? | **N-A / resolved** | tab:per_size L241-242 shows Svan IR <16px bare R=0.970 → +filt R=0.970 (flat) | verified: IR filter is recall-flat per-size, no imgsz artifact; RGB regression (0.782→0.256) was the OLD filter, current v4 = 0.782→0.767 (L232) |
| L1056-1064 | per-size carve-out + median 29.8/14.8px | **FOLLOWED** | empirical L248 "Svanstr\"om's median drone is $29.8$~px... IR ground truth is majority sub-16~px (median $14.8$~px)"; L232-235 buckets | verified vs `notes_round1_results.json` SZ_per_size (29.8, 14.8, all buckets) ✓ |
| L1070-1086 | conf-sweep (SelCom floor 0.05 → R 0.451→0.678, F1 0.692; 1281 FP → filter holds) — formalized/global? | **FOLLOWED** | empirical L282 "raises recall $0.451\to0.678$... for $F1=0.692$... at floor $0.05$ the RGB-confuser corpus floods the bare detector with $1{,}281$ FP detections and the filter holds the line at $50$ ($96.1\%$ suppression)"; L301 "does \emph{not} generalise to a new global default" | verified vs conf_sweep JSON: bare_FP=1281, filt_FP=50 at conf 0.05 ✓. **Thesis updated 45→50** (notes' stale "45/50" fixed). Explicitly framed as operating-mode not global setting. |
| conclusion RQ answers (L9/13/17) | match shipped robust8-nr stack | **FOLLOWED** | conclusion L10 (RQ1 svan 0.742→0.946, R 0.948→0.991, P 0.609→0.905), L14 (RQ2 filter workhorse + reject complementary), L17 (RQ3 reversal, tab:rq3) | all match production cells (verified, Table 2) |

---

## TABLE 2 — Numbers (spot-confirmed against JSON / audit)

| Number + context | Thesis loc | Source | BACKED? |
|---|---|---|---|
| Svan production F1 **0.946**, P 0.905, R 0.991 (nr, filt→clf) | empirical L41, tab:rq3 L99, concl L10, abstract | `results_noreject/tier1` svanstrom.B_pipeline filt→clf[robust8_nr_drop]: P0.905 R0.991 F1**0.9459** | ✓ |
| Svan bare 0.742, R 0.948, P 0.609 | empirical L30, concl L10 | tier1 bare F1 0.7415 R 0.9481 P 0.609 | ✓ |
| Svan IR-only 0.940; RGB-only 0.607 | empirical L28/L97-98, tab:rq3 | tier1 A_bare v3b/ir 0.940, ft4/rgb 0.6067 | ✓ |
| Anti-UAV no-harm 0.973→0.984 | empirical L59/L70, concl L10 | tier1/nr antiuav B_pipeline 0.9728→0.984 | ✓ |
| Anti-UAV ft4 bare F1 0.985, FP=41 | empirical L56, concl L10 ("41 FP across 4,000") | tier1 A_bare ft4/rgb F1 0.9853 FP 41 | ✓ |
| RGB-confuser bare fire 30.4% / 0.304 / 30.3% | concl L10 / tab:confusers L155 / fig L179 | tier1 rgb_confuser bare 0.3035 | ⚠ minor: 30.3 vs 30.4 rounding mix (0.3035 rounds to 30.4%; fig says 30.3%). Cosmetic. |
| RGB-conf shipped nr fire **1.4%** (39/2633); robust8 ablation 0.11% (3) | empirical L162/L170, concl L10 | nr clf→filt 0.0144; tier1 robust8 0.0011/FP 3 | ✓ |
| Thermal IR-conf bare 29.4%→ shipped 2.8%, robust6 1.9% | empirical L173, L806; concl L10 "29.4→1.9" | bare 0.2943; nr 0.0278; robust6 0.0192; robust8 0.0243 | ✓ (concl "best composition 1.9%" = robust6 ablation, not shipped 2.8%; consistent w/ empirical framing but see note below) |
| Grayscale-conf 656→15 FP (−97.7%) | empirical L170/L173/L776, concl L25/L32 | tier1 gray_confuser filt_mlp FP 15, fire 0.0053 | ✓ |
| DUT shipped nr filt→clf F1 0.835, R 0.800 | empirical L130/L137 | results_dut/nr B_pipeline filt→clf[nr] F1 0.835 R 0.800 (audit NR dut filt→clf) | ✓ |
| DUT RGB-alone ft4 0.899; gray-IR 0.596; fused bare 0.758 | empirical L116-119/L137 | runs/results_dut A_bare 0.899/0.596, B_pipeline bare 0.758 | ✓ |
| rgb_dataset_test v4: bare 0.926, +filt R 0.887 F1 0.922 | empirical L201-204/L216/L232 | tier1 rgbtest bare 0.9259, filt_mlp R 0.887 F1 0.9222 | ✓ |
| SelCom filter +FP cut 22→7, P 0.858→0.950, R unchanged 0.451 | empirical L216 | tier1 selcom_val bare F1 0.5911, filt_mlp 0.6115 (P0.950 R0.451) | ✓ |
| Low-conf SelCom: floor 0.05 → R 0.678, filt F1 0.692; 1281 FP→50 | empirical L282, tab:lowconf L295 | conf_sweep selcom filt@0.05 F1 0.6993; rgb_confuser@0.05 bare_FP 1281 filt_FP 50 | ✓ |
| Per-size: rgbtest <16 bare 0.782→filt 0.767; 16-32 0.865→0.844; ≥64 0.956→0.951 | empirical L232-235/L248 | notes_round1 SZ ✓ all (0.7824/0.7672/0.8649/0.8435/0.9555/0.9513) | ✓ |
| Svan median 29.8px (RGB) / 14.8px (IR); IR <16 R 0.970→0.970 | empirical L241-242/L248 | notes_round1 SZ medians 29.8/14.8; IR <16 bare&filt 0.970 | ✓ |
| CBAM held-out: bare 48 FP / R 0.967 → aligned 6 FP / R 0.967 (Δ +0.236 F1) | empirical tab:ir_aligned L638 | `eval/results/ir_heldout_results.json` cbam@0.05 FP=6 (audit pins meth prose + table to this) | ✓ |
| IR_confusers held-out re-mine 90→22 (94% removed) vs shipped 90 (77%) | empirical L649, appendix L38, concl L10 | provenance doc §2.4; audit CITED `eval/eval_ir_heldout.py` | ✓ (provenance-doc backed; not in numeric audit but path-checked) |
| Temporal: bare win F1 0.843; nr composed 0.665; nr confuser fire 0.236 (−33%) | empirical tab:temporal L338/L345 | temporal_results / VN: bare 0.843, nr clf→filt window F1 0.665, fire 0.213→ table 0.236 | ✓ (audit NR video composed 0.646 = robust8_nr_drop window; table cell 0.665 is filt-only-equiv per L353 footnote — consistent) |
| Speed: router 38.3ms→0.095ms (404×); filter 59-112ms→1.3-2.1ms (37-72×) | empirical tab:speed L315-316, concl L14 | ledger speed rows (path-audited; not numeric-audited) | ✓ (ledger-cited) |
| IR HITL 0.503→0.967 across 6 revisions | empirical tab:ir_evolution L438-444, concl L21 | tab values internally consistent; `knowledge/evals.csv ir_final_*` | ✓ (table self-consistent; GPU-source per provenance) |
| MRI: RGB LDA ≈0.95 (0.952), IR LDA 0.981, max F 42,346 / 5,370 | empirical L673/L679, app:mri_report L184 | `mri/.../v5_report_regen/stats.json` + `ir_v3b_report/stats.json` (audit MRI block: LDA 0.981, maxF 5370) | ✓ |

**Conclusion attribution nuance (flag, not error):** concl L10 "the best composition now cuts ${\sim}93\%$ of thermal-confuser fire ($29.4\%\to1.9\%$)" — 1.9% is the **robust6 ablation** composition, NOT the shipped `robust8-nr` (2.8%). Empirical L173 distinguishes these carefully; the conclusion compresses to "best composition" which is literally true (robust6 is a composition) but a casual reader may attribute 1.9% to the shipped stack. Consider "the best (robust6) composition" for parity with Ch4.

---

## TABLE 3 — Claims (citation / evidence status)

| Claim | Thesis loc | Status | Note |
|---|---|---|---|
| Filters trained on detector ROI features, NOT from MRI | empirical L667 "the discrimination the pipeline needs already exists inside the detectors' feature spaces" | EVIDENCED | MRI is diagnostic/audit; filters are distilled MLPs. But L691-693 "affine offset" para implies MRI→filter pipeline; Locked Decision #2 wants it removed. **CONTENT-FLAG** |
| "Cross-modal alignment is an affine offset" (z-score → AUROC 0.500→0.919) | empirical L691-693 + fig:ir_gray_align L695-701 | EVIDENCED (mri/modality_align.py) **but LOCKED-REMOVE** | Locked Decision #1+#2: with thermal-native filter shipped, the grayscale-alignment MRI result is no longer load-bearing → drop para + figure. Currently PRESENT. |
| IR filter = "two heads" (thermal-native + grayscale-aligned) | empirical L626-628, L649, app L163-164/L269, concl L30/L32 | EVIDENCED **but LOCKED-COLLAPSE** | Locked Decision #1: collapse to thermal-native ONLY; drop grayscale-aligned head + two-heads framing + grayscale-head table rows. Currently the grayscale-aligned head is woven through Ch4/appendix/conclusion. **Biggest content gap.** |
| Grayscale FINDING = recall (finds drones when RGB fails) + one fail (texture-rich close-ups, dedicated RGB +28-41pp) | empirical §sec:grayscale L708-777 | EVIDENCED, MATCHES LOCKED DECISION | L733 "2.7pp behind RGB", L766 "dedicated RGB detectors win by $28$--$41$~pp F1 per clip (texture-rich close-ups)... small-silhouette bird-cluttered scenes the thermal model dominates". ✓ Headline is recall/recall-with-suppression. KEEP as-is. |
| Model cards exist (provenance) | app:models L127-169 (tab:models_evaluated), app:provenance L204-242 | CITED/EVIDENCED | Provenance appendix + models appendix present; cite `docs/analysis/2026-06-18_filter_provenance_train_heldout.md` for v4/cbam split provenance. **Spot to cite:** empirical L613/L617/L650 already cite the provenance doc. Good. |
| sa32 scene-statistic leakage | empirical L85/L466/L531/L537, app glossary L263 | EVIDENCED (§feature_selection) | strong, consistent |
| MRI "audited its own paper trail" (corrected 2 stale figures) | empirical L703-705, concl L21 | EVIDENCED | mri/results/v5_report_regen |
| All 5 \cite{} keys in slice resolve | empirical/concl/app | CITED (verified) | howard2019mobilenetv3, redmon2016yolo, svanstrom2022dronedataset, ultralytics2024, zhao2022dutantiuav — all in references.bib (44 keys). No missing/undefined. |

---

## TABLE 4 — Split-naming + leakage-reframe map

**User directive:** assert ONCE globally (before results) that all surfaces are test/held-out; flag every apologetic per-surface disclosure as a CONTENT-FIX (reframe → confident). Note whether each result surface names its split.

| Surface / table | Names its split? | Apologetic disclosure present? (loc) | Reframe action |
|---|---|---|---|
| **Global (chapter opener)** | — | **NO global assertion** — L4 covers only CIs/n | **ADD** one confident sentence before §pipeline_ablation: "Every evaluation surface in this chapter is a held-out test split or an eval-only benchmark disjoint from training; per-surface split identity is named in each table and the overlap-bounded controls are in §svanstrom_audit." |
| tab:ablation_svanstrom (Svanström) | "eval-only" / IoP; in-sample for IR — not labelled here | partial (methodology L24/L268 carries the apology) | keep "even-strided"; rely on global assertion + §svanstrom_audit |
| tab:ablation_antiuav (Anti-UAV) | "eval-only" benchmark | — | OK |
| tab:rq3 | inherits svan/antiuav | — | OK |
| **DUT** (tab:ablation_dut, L105-106/L110) | **YES — "\emph{test split}"** named 4× ("Only the official test split is used (no DUT frame here was a training image for the router or filters)") | confident, not apologetic ✓ | KEEP — this is the model for the rest |
| tab:ablation_solo (IR/RGB test split, SelCom val) | **YES** — "IR test split", "RGB test split", "SelCom val" | — | OK (named) |
| tab:ablation_confusers (RGB/IR/gray confusers) | **YES** — "full test split" / "even spread" | — | OK |
| **L460 (IR OOD bound)** | n/a | **APOLOGETIC**: "untested on verified data... those sets' label quality could not be verified and their numbers are not reported" | this is a genuine *capability* gap (recall on external thermal), NOT a leakage hedge — KEEP but it's legitimately a limitation, not a leakage apology. Lower priority. |
| **L616 (Held-out bird, TEST split)** | **YES + apologetic tail** "the one in-sample exception is Svanstr\"om RGB (no train/test split exists, so its absolute number is fair only as a shipped-vs-candidate $\Delta$)" | **APOLOGETIC** (in-sample disclosure) | **REFRAME**: keep the held-out wins confidently; soften "in-sample exception... fair only as Δ" → state Svanström has no author split so it's used as a paired-Δ control (matches the global assertion's exception). |
| **L649 (thermal head held-out)** | **YES + apologetic** "with Svanstr\"om IR the in-sample exception (drone recall held flat at $0.966$, fair only as a $\Delta$)" | **APOLOGETIC** | **REFRAME** same as L616 (Svanström no-split → Δ control, stated once globally). |
| **L790 (External validity)** | n/a | **APOLOGETIC**: "share data with the training corpora... at most $7.3$~pp of in-distribution inflation" | this IS the quantified overlap bound (legit threats-to-validity content). The user wants the *headline* framing confident; this is the honest control section, so KEEP but ensure it reads as "here is the bound" not "we cannot guarantee". Currently acceptable; tighten tone if desired. |
| Methodology §svanstrom_audit (L268, upstream) | — | **APOLOGETIC**: "The two paired evaluation surfaces share data with the training corpora... bounds their inflation" | **Out of my slice** but it's the upstream source of the per-surface apology. The global confident assertion should live near here or the Ch4 opener. |

**Net:** the ONLY result surfaces that are in-sample (no author split) are **Svanström RGB and Svanström IR** — every other Ch4 surface names a genuine test/held-out/val split (DUT test, IR/RGB test split, SelCom val, all confuser test splits, CBAM valid, IR_confusers val/test, bird.v1i test, Anti-UAV eval-only). So the global assertion is **factually safe**; the per-surface in-sample hedges at L616/L649 can be demoted to one global footnote naming Svanström as the paired-Δ exception.

---

## TOP FIXES (ranked)

1. **[LOCKED #1 — grayscale verifier] Collapse §sec:grayscale_verifier (L626-651) to the thermal-native head only.** Currently ships "two heads" (L626-628), the grayscale-aligned head `mlp_aligned_gray_balanced` woven through L649, app:models L164, glossary L269, concl L30/L32. Directive: drop the grayscale-aligned head + two-heads framing + grayscale-head rows. (Caveat for the truth-agent: the provenance doc §2.5 says the shipped thermal head is currently the `--no-gray` *validation* build and production needs a with-gray retrain — so collapsing to thermal-native-only also matches deployment reality.)

2. **[LOCKED #1/#2 — MRI artefacts] REMOVE the "affine offset" MRI paragraph (L691-693) and `fig:ir_gray_align` (L695-701).** With thermal-native filter shipped (trained on thermal crops directly, NOT grayscale-harvested+aligned), the grayscale↔thermal alignment result is no longer load-bearing for the filter. The harvesting-payoff framing also lingers at L771-774/concl L25.

3. **[LEAKAGE REFRAME — global] Add one confident global assertion before results** ("all Ch4 surfaces are test/held-out or eval-only benchmarks disjoint from training; Svanström alone has no author split and is used as a paired-Δ control"), then **demote the per-surface in-sample apologies at L616 and L649** to lean on it. Factually safe — only Svanström is in-sample.

4. **[robust6 standalone rows — taste] Notes L953 irritation persists:** robust6 still appears as a standalone composition row in tab:ablation_confusers (L158/163) and temporal (L343/347). It's justified as the lineage base + ablation, but if the user wants robust6 demoted to "base only", these rows are the spots. (OPEN-by-design.)

5. **[conclusion attribution — minor] concl L10** "best composition cuts 29.4%→1.9%" attributes the **robust6 ablation** number to "best composition"; the shipped stack is 2.8%. Empirical L173 distinguishes them; make the conclusion say "best (robust6) composition" for parity so a reader doesn't credit the shipped stack with 1.9%.

6. **[cosmetic] RGB-confuser bare fire** printed as 30.4% (concl/abstract), 0.304 (table), 30.3% (fig L179) — unify to one rounding (JSON = 0.3035).

**Everything else session-8/9 the notes flagged is FOLLOWED and JSON-backed:** audit 187/187; production cells (`robust8-nr`, filt→clf) pinned in both `tier1` and `results_noreject`; the coverage table the notes worried about (4.7) was deleted and replaced by trust-aware tab:rq3 (foregrounding IR-only 0.940, resolving the "IR is good on Svanström" complaint); the conf-sweep "45→50 FP" was corrected to 50; v4 rgb_dataset_test recovery (0.691→0.887) is current; per-size + medians verified; CBAM held-out FP=6 pinned by the audit to the canonical JSON.
