# Filter edit plan — `methodology.tex` (per-edit, READ-ONLY plan)

Scope: `docs/thesis_working_distilling_overleaf/chapters/methodology.tex` (the canonical live methods chapter).
Production filter stack (FINAL): RGB **`mlp_v5_v4`** @0.25 (birdsplit) · IR thermal-only **`mlp_aligned_thermalonly`** @0.05 ·
IR grayscale **`mlp_aligned_gray_balanced`** @0.25. **The IR verifier is now TWO checkpoints** (thermal-native CBAM +
grayscale-aligned), retiring the "one network, two scalers" claim.

Rules honoured: (1) tables + eval surfaces unchanged; cells take the new value. (2) Held-out added ONLY in mlp-filter
sections, labelled TEST SPLIT, train+test datasets named. (3) Filters FINAL. (4) Claim reframes: retire one-net-two-scalers
→ two heads; thermal-airplane hole largely CLOSED; thresholds RGB 0.25 / IR-th 0.05 / gray 0.25; training corpora named
(v4 birdsplit; thermal-only = 8112 thermal drones + 2045 IR_confusers/train confusers; gray = grayscale-harvested). KEEP
"IR filter inert on Svanström" (in-sample exception). Provenance statements ADDED at every mlp-filter mention.

Verification key: **CHECK** = guarded by `thesis_eval/_audit_headline_numbers.py`; **UN-AUDITED** = registry/source is the only
net (provenance doc `2026-06-18_filter_provenance_train_heldout.md`; swap table `filter_PRODUCTION_swap_tables.md`).

Note: methodology holds *design/recipe/MRI* prose; nearly all numeric *result* cells live in `empirical.tex` (separate plan).
Most methodology edits are therefore **claim-reword / provenance-stmt / recipe-update / threshold**, not table-cell swaps.

---

## SECTION 1 — §sec:ds_ir_confusers (CBAM probe definition, line ~169)

| line | label/section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|
| 169 | sec:ds_ir_confusers | "That probe remains the \emph{held-out} gate for the aligned filter's training (Section~\ref{sec:ir_xmodal_verifier}) since \texttt{IR\_confusers} postdates that training." | Keep, but rename "aligned filter" → "thermal-native filter (\texttt{mlp\_aligned\_thermalonly})" and note CBAM **valid** split is the held-out recall-recovery gate. | claim-reword (two-head rename) | neutral (clarifies which head; CBAM held-out story strengthens) | UN-AUDITED → provenance doc §2.3 (CBAM valid DISJOINT); held-out CBAM **R 0.967 / FP 6 @0.05** |

**SECTION VERDICT:** 1 edit — light rename only (probe definition unchanged; the held-out CBAM number is asserted in §ir_xmodal_verifier, not here).

---

## SECTION 2 — §sec:svanstrom_audit, Table tab:svanstrom_audit (overlap rows, lines 285–286)

| line | label/section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|
| 285 | tab:svanstrom_audit | `RGB filter (\texttt{mlp\_v5}) & Clean (SelCom CCTV features only) & Clean` | `RGB filter (\texttt{mlp\_v5\_v4}) & Clean (no Svanström; trained on rgb\_dataset train+val, rgb\_confusers train+val, pure-SelCom, bird.v1i train) & Clean` | provenance-stmt + cell-update (weight rename) | neutral (rename + honest corpus; Svanström still clean/in-sample for Δ) | UN-AUDITED → provenance doc §1.2–1.3 (svanstrom RGB = IN-SAMPLE; all train surfaces disjoint from their test) |
| 286 | tab:svanstrom_audit | `IR filter (aligned) & In-distribution (Svanström IR crops) & Clean (val split only; val$\cap$test $= 0$)` | `IR filter (thermal \texttt{mlp\_aligned\_thermalonly} / gray \texttt{mlp\_aligned\_gray\_balanced}) & In-distribution (Svanström IR crops in thermal train) & Clean (Anti-UAV val only; val$\cap$test $=0$)` | provenance-stmt + claim-reword (two heads) | neutral (names both heads; same overlap verdict) | UN-AUDITED → provenance doc §2.2–2.3 (thermal train uses Svanström IR `IR_DRONE_`; Anti-UAV val; svanstrom_ir IN-SAMPLE, recall held 0.966) |

**SECTION VERDICT:** 2 edits — weight renames + corpus provenance in the audit rows; overlap verdicts (clean/in-sample) are UNCHANGED, so the leakage narrative holds. Both UN-AUDITED → provenance doc is the net.

---

## SECTION 3 — §ch:architecture overview + two-filter fusion (lines 513, 521, 525, 535)

| line | label/section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|
| 513 | architecture overview (item 4) | "in production, the distilled \texttt{mlp\_v5} feature-space filter" | "in production, the distilled \texttt{mlp\_v5\_v4} RGB filter (and its IR counterparts)" | cell-update (weight rename) | neutral | UN-AUDITED → checklist §3 CITED_PATHS (`models/verifiers/rgb_v5/mlp_v5_v4.pt`) |
| 521 | fig:pipeline caption | "The production confuser filter is the distilled \texttt{mlp\_v5}, which runs \textbf{per frame}…" | "…the distilled \texttt{mlp\_v5\_v4} (RGB) with paired IR heads…" | cell-update (rename) | neutral | UN-AUDITED → checklist §3 |
| 525 | sec:design_rationale lead-in | "a detection survives only if the MLP assigns it $P(\text{drone})$ at or above the production threshold." + "A fail-open filter variant … was evaluated and rejected" | Keep; add per-head production thresholds inline: "(RGB $0.25$, IR-thermal $0.05$, grayscale $0.25$)". Fail-open sentence UNCHANGED. | threshold + recipe-update | neutral (fail-open verdict unchanged; precision-craters-0.887→0.631 source still valid) | thresholds UN-AUDITED → swap manifest `_filter_swap/final/swap_manifest.json`; fail-open CHECK not present (prose) |
| 535 | "Two-filter fusion (trust-first)" | "With both RGB (\texttt{mlp\_v5}) and IR (\texttt{mlp\_v5\_ir\_aligned}) filters running per-frame …" + "**The IR filter is one network with two per-modality input scalers** (a thermal scaler and a grayscale scaler), so a single trained filter serves both the thermal-deploy and grayscale-fallback paths" | "With both RGB (\texttt{mlp\_v5\_v4}) and IR filters running per-frame …" + **retire the one-net sentence** → "The IR filter is **two heads**: a thermal-native CBAM net (\texttt{mlp\_aligned\_thermalonly}, $\geq0.05$) for the thermal-deploy path and a grayscale-aligned net (\texttt{mlp\_aligned\_gray\_balanced}, $\geq0.25$) for the grayscale-fallback path." | claim-reword (FLIP: one-net→two-head) + threshold | neutral-to-better (now matches shipped GUI; honest architecture) | UN-AUDITED → checklist §6 (two-head reframe); kb models `mlp_aligned_thermalonly`, `mlp_aligned_gray_balanced` |

**SECTION VERDICT:** 4 edits — the central **one-network-two-scalers → two-heads FLIP** lands at line 535; the rest are weight renames + inline production thresholds. No table cells (architecture prose). All UN-AUDITED (prose), netted by checklist §6 + kb model rows.

---

## SECTION 4 — §sec:ir_xmodal_verifier (the IR-filter build recipe, lines 649–700) — HEAVIEST

| line | label/section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|
| 649–651 | sec:ir_xmodal_verifier title + lead | "Cross-Modal Feature Alignment: a Thermal Filter from Grayscale Confusers" / "the grayscale mode is what let us build **a single filter that serves both** thermal and grayscale-fallback input" | Title kept (alignment finding still real for the gray head). Lead reworded: "the grayscale mode is what let us build a **grayscale-aligned head**; the thermal-deploy path is a separately trained thermal-native net." | claim-reword (FLIP: single→two heads) | neutral (alignment finding preserved for gray head; thermal now native) | UN-AUDITED → checklist §6; provenance §2.1 (`--no-gray` thermal head is a separate net) |
| 654 | (prose) | "on the held-out CBAM set … it cuts false positives $48 \to 15$ at no recall cost (Section~\ref{sec:verifier_results})." | "on the held-out CBAM **valid** split … the thermal-native head recovers recall to **R 0.967** while cutting false positives to **6 FP** (bare 48), at no recall cost." | cell-update + add-heldout (TEST SPLIT) + provenance-stmt | **better** (48→15 → 48→6 FP; recall recovered 0.917→0.967) | **was CHECK** "CBAM aligned FP (methodology prose)"=15 → checklist §2 RESOLVED: CBAM **R 0.967 / FP 6 @0.05** (kb `cbam_heldout_thermalonly`); audit constant must move 15→6 |
| 654 | (prose, same sentence) | "(the IR mining set) it fires on only $1.8\%$ of images" (raw v3b near-domain halluc — KEEP as detector stat) | UNCHANGED (this is the v3b detector's near-domain MRI rate, not the filter; memory `number_needs_dataset` flags it is NOT the OOD set) | (no edit) | neutral | CHECK "MRI ir halluc"=0.018 (keep) |
| 657–674 | tab:ir_mri_sep | LDA 0.981, max ANOVA F 5370, median 256, halluc 1.8%, FP cut 89%, recall ret 99.7%, n=14697/1386 | **UNCHANGED** — this is the v3b detector-feature MRI (separability evidence), not a filter operating point | (no edit) | neutral | CHECK all (MRI ir LDA/maxF/medianF/halluc/fp_cut/recall_ret/n) — keep as-is |
| 676–694 | tab:gray_thermal_auroc + prose | gray→thermal AUROC raw 0.500 / CORAL 0.707 / z-score 0.919 / ceiling 0.974 | **UNCHANGED** — alignment evidence still underpins the **grayscale head** (`mlp_aligned_gray_balanced`); reframe surrounding prose to attribute it to the gray head only | claim-reword (attribution only) | neutral (finding intact for gray head) | UN-AUDITED → ledger `gray-thermal-alignable` (keep) |
| 696 | (prose, the one-net claim) | "The production checkpoint \texttt{mlp\_v5\_ir\_aligned} therefore combines thermal drone detections … with grayscale-harvested confusers, per-modality z-aligned into **a single $517$-D filter. It is \emph{one network with two per-modality input scalers}** … **This is genuinely \emph{one} shipped network, not two**: a separately trained grayscale-only filter was the rejected alternative, because on grayscale Svanström it over-vetoes drones (recall $0.55 \to 0.16$) where the single aligned network stays recall-safe ($0.55 \to 0.51$)." | **REWRITE (FLIP):** "Production ships **two $517$-D heads**: a **thermal-native** net (\texttt{mlp\_aligned\_thermalonly}, CBAM) trained on $8{,}112$ thermal drones $+$ $2{,}045$ \texttt{IR\_confusers/train} confusers, and a **grayscale-aligned** net (\texttt{mlp\_aligned\_gray\_balanced}) built from grayscale-harvested confusers z-aligned into thermal feature space. The earlier single-net design (one net, two scalers) was superseded because thermal-native training recovers held-out CBAM recall ($0.717 \to 0.967$) that the shared net could not. The grayscale path remains recall-limited (gray Svanström $0.55 \to$ ~$0.08$ @0.25), so grayscale detection runs largely unfiltered — a confuser-suppression tool, not recall-safe." | claim-reword (FLIP) + recipe-update + provenance-stmt + threshold | **better** (matches shipped; thermal recall recovered; honest gray limit) | UN-AUDITED → provenance §2.1–2.4 (thermal train n=8112/2045; CBAM recall 0.717→0.967); checklist §6; swap table gray_confuser filt FP 21→15 |
| 698 | (prose) | "The alignment costs nothing on thermal: versus the bare detector, the aligned filter is recall-neutral on held-out thermal surfaces ($\Delta R = -0.007$ on the IR test split, $0.000$ on IR video, $0.000$ on Anti-UAV) while cutting CBAM confuser false positives from $48$ to $15$." | "The thermal-native head holds drone recall on held-out thermal: antiuav\_ir $0.937$, ir\_video $0.971$, svanstrom\_ir $0.966$; the only cost is **ir\_dset\_final $0.965 \to 0.928$ ($-3.7$~pp)** on genuinely airplane-like drones. It cuts CBAM confuser FP **$48 \to 6$** (held-out CBAM valid)." | cell-update + add-heldout (TEST SPLIT) + provenance-stmt | mixed: **better** on confuser FP (15→6) + honest about new −3.7pp ir_dset cost (was −0.007) | thermal ΔR rows UN-AUDITED → provenance §2.4 (ir_dset −3.7pp, antiuav 0.937, ir_video 0.971); CBAM FP was CHECK=15 → 6 (checklist §2) |
| 698 | (prose, end) | "The CBAM probe (Section~\ref{sec:ds_ir_confusers}) is held out of its training; the held-out results are in Section~\ref{sec:verifier_results}." | Keep; **ADD provenance sentence:** "The thermal head is trained on \texttt{IR\_confusers/train} + Svanström-IR/Anti-UAV-val/IR\_dset\_final-train thermal drones and **evaluated on the held-out CBAM valid split and IR\_confusers val/test** (TEST SPLIT, disjoint from training)." | provenance-stmt + add-heldout | neutral-to-better (names train + held-out test per user rule) | UN-AUDITED → provenance §2.2–2.3 (train sources + DISJOINT held-out: CBAM valid, IR_confusers val/test re-mine, ir_dset_final test) |

**SECTION VERDICT:** 8 edits (2 explicit no-edit holds on the MRI separability tables). This is the **anchor section** for the two-head reframe. Net direction **BETTER**: thermal-native CBAM FP 48→6 with recall 0.717→0.967 dominates the old 48→15; honest −3.7pp ir_dset op-point cost disclosed; one-net→two-head FLIP completed; provenance (train + held-out CBAM valid / IR_confusers val/test) added. Two CBAM cells were AUDITED (15→6 must move in `_audit_headline_numbers.py`); the rest UN-AUDITED, netted by provenance §2.

---

## SECTION 5 — §sec:patch_verifier_arch (predecessor patch filter, lines 798–807)

| line | label/section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|
| 801 | sec:patch_verifier_arch | "the production filter is \texttt{mlp\_v5} (Section~\ref{sec:distill_verifier})" | "the production filter is \texttt{mlp\_v5\_v4} (Section~\ref{sec:distill_verifier})" | cell-update (rename) | neutral | UN-AUDITED → checklist §3 CITED_PATHS |
| 803–806 | patch recipe/numbers | 45,917 patches; acc 0.975; v2 production; "v4 ties v2" | **UNCHANGED** — predecessor design history; not a production filter | (no edit) | neutral | UN-AUDITED (predecessor; keep) |

**SECTION VERDICT:** 1 edit — single production-filter rename in the cross-reference; the patch predecessor recipe is design-history and stays. The phrase "v4 ties v2" here refers to the **patch** v4, NOT the RGB filter v4 — do not conflate (no edit, but flagged to avoid a wrong "swap").

---

## SECTION 6 — §sec:distill_verifier (RGB feature-reuse filter recipe, lines 809–819)

| line | label/section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|
| 809/815 | sec:distill_verifier title + body | "Feature-Reuse Filter (\texttt{mlp\_v5})" / "the shipped \texttt{mlp\_v5} was trained on … $32{,}931$-detection corpus ($19{,}334$ drone / $13{,}597$ confuser)" | Title/body name the production weight \texttt{mlp\_v5\_v4}; corpus line: "the production \texttt{mlp\_v5\_v4} extends this corpus with an **in-distribution bird.v1i train split** (birdsplit) and a rebalanced size×source drone manifold (\texttt{build\_balanced\_v4\_birdsplit.py}); the parent MRI characterisation ($32{,}931$ det, $19{,}334$/$13{,}597$) is retained as the feature-space evidence." | recipe-update + provenance-stmt + cell-update (rename) | **better** (names v4 birdsplit recipe; fixes the 22%-veto coverage gap) | corpus 32,931 UN-AUDITED (in-corpus MRI, keep); v4 provenance → provenance §1.1–1.2 (v2 19,334/13,597 + bird.v1i 728 train) |
| 815 | (prose) | "a single linear discriminant separates drone from confuser at ${\approx}95\%$ … silhouette $0.067$" | **UNCHANGED** — parent-corpus MRI separability (feature-space evidence, not an operating point) | (no edit) | neutral | UN-AUDITED (in-corpus MRI; keep) |
| 815 | (prose, end) | "the filter's measured effect on every surface is in Section~\ref{sec:verifier_results}." | Keep; **ADD provenance + held-out sentence:** "\texttt{mlp\_v5\_v4} is trained on rgb\_dataset train+val, rgb\_confusers train+val, pure-SelCom (val 311 blocklisted), and bird.v1i **728-image train split**, and evaluated on held-out **bird.v1i 484-image TEST split** (TEST SPLIT): it keeps $30/230$ unseen-bird fires vs the shipped filter's $91/230$, and lifts rgb\_dataset\_test recall $0.694 \to 0.874$." | add-heldout (TEST SPLIT) + provenance-stmt | **better** (held-out bird 91→30; rgbtest recall 0.69→0.87 — the headline RGB win) | rgbtest mlp F1 was CHECK 0.8092 → **0.9222** (checklist §2); bird 91→30 UN-AUDITED → provenance §1.4 / checklist §7 held-out re-mine `eval_birdtest_heldout.py` |
| 818 | (prose) | "Its IR counterpart, \texttt{mlp\_v5\_ir\_aligned}, is the cross-modal product of Section~\ref{sec:ir_xmodal_verifier}: **one network, two per-modality input scalers**, serving thermal and grayscale-fed channels alike." | **REWRITE (FLIP):** "Its IR counterparts are **two heads** (Section~\ref{sec:ir_xmodal_verifier}): the thermal-native \texttt{mlp\_aligned\_thermalonly} and the grayscale-aligned \texttt{mlp\_aligned\_gray\_balanced}." | claim-reword (FLIP) | neutral-to-better (consistent with §4) | UN-AUDITED → checklist §6; kb model rows |
| 818 | (prose) | "5-fold CV $F1$ of $0.9857\pm0.0004$, rejecting $97\%$ of confuser detections while retaining $98.9\%$ of true drones." | **UNCHANGED** — parent in-corpus CV (feature-space evidence; not a deployed surface) | (no edit) | neutral | UN-AUDITED (in-corpus CV; keep) |

**SECTION VERDICT:** 5 edits (2 no-edit holds on in-corpus MRI/CV evidence). Net **BETTER**: names the v4 birdsplit recipe, adds the held-out **bird.v1i TEST** provenance (91→30) and the rgb_dataset_test recall recovery (0.69→0.87, the RGB headline), and completes the IR two-head FLIP at 818. One AUDITED cell (rgbtest mlp F1 0.8092→0.9222) lives in empirical but is asserted here via the held-out add — flag for the empirical plan to keep consistent.

---

## CROSS-SECTION HOLDS (explicitly NO edit — listed so a swap does not break them)

| line | label/section | why NO edit | KEEP claim |
|---|---|---|---|
| 32 (empirical-ref note here) tab:ablation_confusers | n/a (lives in empirical) | core confuser numbers are empirical, not methodology | — |
| 127 | fig:confuser_fp_examples caption | "$P(\text{drone})$ $0.001$–$0.077$ … suppressed at $0.25$" is illustrative ft4 examples at the **RGB 0.25 threshold** (unchanged) | RGB threshold stays 0.25 |
| 162 | sec:ds_ir (29.4% / 1.8%) | detector fire-rate stats (not filter); but see reframe note below | IR detector near-domain 1.8% vs OOD 29.4% (memory `number_needs_dataset`) |
| 388 | sec:coevolution | "\texttt{mlp\_v5} filter … distinct from the V5 IR-detector regression" — rename to \texttt{mlp\_v5\_v4} for consistency (minor) | disambiguation kept |
| 541 | sec:design_rationale | "SelCom $0.591 \to 0.692$ with the filter at floor 0.05" — swap table shows selcom UNCHANGED (0.6115) | KEEP unchanged |
| 654/665–674 | tab:ir_mri_sep | v3b detector-feature separability (not a filter op point) | KEEP all CHECK values |
| 679–694 | tab:gray_thermal_auroc | alignment evidence underpins the gray head | KEEP (reattribute prose only) |

**Reframe-watch (claim that softens, optional prose, NOT a number):** line 511 / 162 narrative "on the OOD thermal-confuser corpus its residual fire is **airplane-dominated**" — with thermal-native training the **filter** now closes most of this (IR-confuser fire 23.7%→2.8% held-out 90→22). The *bare-detector* 29.4% stat stays (detector unchanged), but any methodology prose implying the **filter** can't beat airplanes should be softened to match the empirical airplane-hole-closed reframe (checklist §6). Methodology line 511 only says the detector is airplane-dominated → KEEP, flag for empirical.

---

## TOTALS

- **Total edits: 21** across 6 sections (+ 9 explicit NO-EDIT holds catalogued).
  - Section 1: 1 · Section 2: 2 · Section 3: 4 · Section 4: 8 · Section 5: 1 · Section 6: 5.
- **By edit type:** claim-reword (FLIP one-net→two-head) **5** (lines 169, 286, 535, 696, 818) · provenance-stmt **6** · add-heldout (TEST SPLIT) **3** (CBAM valid @654/698, bird.v1i TEST @815) · recipe-update **3** · threshold **3** · cell-update (weight rename) **7** (overlapping types counted once per row where dominant).
- **Direction vs current:** **better 5** (CBAM 48→6 + recall 0.717→0.967 @654/698; rgbtest 0.69→0.87 + bird 91→30 @815; v4 recipe @815/809; one-net→two-head honesty @535/696/818 net-positive), **worse/cost-disclosed 1** (ir_dset_final −3.7pp now disclosed @698, was −0.007), **neutral 15** (renames, attributions, threshold inlining).
- **Verifiability:** **AUDITED (CHECK) 3 distinct constants** — CBAM aligned FP (methodology prose) **15→6** (lines 654, 698; `_audit_headline_numbers.py` must move) and rgbtest mlp F1 **0.8092→0.9222** (asserted via held-out add @815, primary cell in empirical). **UN-AUDITED: the remaining ~18 edits** → netted by `2026-06-18_filter_provenance_train_heldout.md` (§1 RGB v4, §2 IR thermal/gray) + `filter_PRODUCTION_swap_tables.md` + checklist §2/§3/§6 + kb model rows (`mlp_v5_balanced_v4`, `mlp_aligned_thermalonly`, `mlp_aligned_gray_balanced`).

### Atomicity flag
The two AUDITED CBAM occurrences (654, 698) share the `_audit_headline_numbers.py` regex pinned to 15 — they must move to **6** together with the empirical `tab:ir_aligned` cell or the audit goes red (checklist "apply atomically"). The one-net→two-head FLIP touches 535 / 696 / 818 here plus the glossary/conclusion/fig_pipeline outside this chapter — list-tracked but out of this plan's scope.
