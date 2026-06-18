# Final pre-humanify review — Abstract + Introduction

**Scope:** `docs/thesis_working_distilling_overleaf/main.tex` (Abstract, L154–163) +
`docs/thesis_working_distilling_overleaf/chapters/introduction.tex` (full).
**Backbone:** `thesis_eval/_audit_headline_numbers.py` (ran clean: **180/180**, 140 cells + 40 paths),
frozen JSONs `thesis_eval/results*/`, `knowledge/{ledger,evals}.csv`, `references.bib` (44 keys).
**Method:** every number opened against its source file; every \ref/\cite resolved against the live
label set (157 labels across main+chapters). READ-ONLY — no thesis files edited.

Headline verdict: numbers are in excellent shape — **every** quantitative claim in this slice traces
to a verifiable path and matches its source. The only substantive issue is a **framing inconsistency
the task flagged: abstract "One finding" vs intro "Two findings"** (real, see C1). Two minor
rounding/labelling drifts (C2, C3) and a few prose-precision nits round out the list. No dangling refs,
no missing citations, no broken figure paths in this slice.

---

## A. NUMBERS

| # | number + context | loc | source / path | status |
|---|---|---|---|---|
| A1 | 41 false positives / 4,000 Anti-UAV frames | abs L156; intro L8; concl L10 | `tier1_results.json` antiuav A_bare ft4/rgb FP=41; audit pin `antiuav ft4 FP` | BACKED ✓ |
| A2 | hallucinates on 2.8% of composite [RGB] test corpus | abs L156; intro L8 | evals.csv `v5_rgbds_bare` halluc col **0.0276**→2.8%; src comment L10 cites `eval=v5_rgbds` | BACKED ✓ |
| A3 | fires on 30.4% of OOD confuser corpus | abs L156; intro L15; concl L10 | `tier1_results.json` rgb_confuser C_confuser bare fire=0.3035; audit `rgbconf bare fire` | BACKED ✓ |
| A4 | up to 94% of bird-only Svanström frames | abs L156 | cache `svanstrom_1280_by_category.csv` BIRD det_rate=94.4%; audit covers via N CAT | BACKED ✓ |
| A5 | retrain collapses small-drone recall 0.961→0.306 | abs L156; intro L19 | cache CSV retrained_v2 DRONE recall 0.306; baseline 0.959 (R); ledger `retrainedv2-recall-collapse` | BACKED ✓ (see C5 on "0.961") |
| A6 | Svanström per-cat fire 94.4 / 74.6 / 66.2 (bird/airplane/heli) | intro L15 | cache CSV baseline BIRD 94.4 / AIRPLANE 74.6 / HELICOPTER 66.2 | BACKED ✓ |
| A7 | rgb_confuser per-cat 39.0 birds / 58.0 heli / 23.4 airplane | intro L15 | `notes_round1_results.json` CAT; audit `CAT rgbconf bird/heli/airplane bare` (0.3896/0.5802/0.2339) | BACKED ✓ |
| A8 | hardneg_v3more heli 66.2→41.9, airplane →64.7, bird ~94.2 | intro L19; meth L628 | cache CSV hardneg_v3more rows (41.9 / 64.7 / 94.2) | BACKED ✓ |
| A9 | retrained_v2 bird fire →3.4% | intro L19 | cache CSV retrained_v2 BIRD 3.4% | BACKED ✓ |
| A10 | retrained_v2 in-domain F1 0.949 > baseline 0.942 | intro L19,L21 | evals.csv `rgb_rgbds_retrainedv2` 0.9494, `rgb_rgbds_baseline` 0.9419; ledger `rgb-collapse-ood-specific` | BACKED ✓ |
| A11 | Svanström median GT √area 29.8 px (640×512) | intro L23 | `notes_round1` SZ svan ft4/rgb median 29.8; audit `SZ svan rgb median px` | BACKED ✓ (fig says 28 — see fig audit, not this slice) |
| A12 | bare RGB recall 0.90–0.96 (16–64px) → 0.63 (<16px) | intro L23 | audit `SZ svan rgb <16 bare R`=0.6296, `16-32`=0.8974; matches | BACKED ✓ |
| A13 | thermal own-GT F1 0.940 vs visible 0.607 (Svan); 0.985 vs 0.961 (AntiUAV) | intro L23; concl L18 | `tier1` svan A_bare v3b 0.940 / ft4 0.607; antiuav 0.985/0.961; audit `svan v3b IR-only F1` | BACKED ✓ |
| A14 | SelCom median √area ≈37 px (1920×1080) | intro L28 | src comment `SelCom CCTV manifest median 36.8px native`; dataset=selcom_cctv | BACKED ✓ (artifact-cited; not in audit) |
| A15 | ft4 lifted SelCom val F1 0.145→0.591 | intro L28 | evals.csv `v5_selcom_bare` 0.5911; baseline 0.145 (ledger `email-recompute`); audit `selcom bare F1` | BACKED ✓ |
| A16 | composite in-domain 0.942→0.929 (ft4) | intro L28 | evals.csv `rgb_rgbds_baseline` 0.9419, `v5_rgbds_bare` ft4 0.9290 | BACKED ✓ (see C4) |
| A17 | full pipeline Svan F1 0.742→0.946, R 0.948→0.991, P 0.609→0.905 | abs L159; intro L50; concl L10 | JSON `filt->clf[robust8_nr_drop]` F1=**0.9459**→0.946, R 0.9905, P 0.9052; audit `NR svan filt->clf F1`=0.946 | BACKED ✓ |
| A18 | rgb_confuser fire 30.4%→1.4% (filter) | abs L159; intro L50; concl L10 | JSON rgb_confuser filt_mlp fire 0.0144→1.4%; audit `NR rgb_conf fire`=0.0144 | BACKED ✓ |
| A19 | reject-class robust8 reaches 0.11% composed (3 frames / 2,633) | abs L159; intro L50; concl L10 | JSON `clf->filt[robust8]` fire=**0.0011** (FP=3) → 0.11% | BACKED ✓ (matches JSON; **audit pin + ledger lag, see C6**) |
| A20 | Anti-UAV no-harm 0.973→0.984 | abs L159; intro L50; concl L10 | JSON antiuav B bare 0.9728 / `filt->clf[robust8_nr_drop]` 0.984; audit `NR antiuav composed F1` | BACKED ✓ |
| A21 | stages cost 0.095 ms + ~2 ms; 37–404× cheaper | abs L159; intro L50; concl L14 | `tab:speed` (empirical L315–316): robust8-nr 0.095 ms/404×; mlp_v5 1.3–2.1 ms/37–72×; ledger `robust6-speed-feature-efficiency` | BACKED ✓ |
| A22 | LDA separability 0.952 RGB, 0.981 IR | abs L162; intro L53; concl L20 | `mri/results/v5_report_regen/stats.json` lda=0.9517→0.952; `ir_v3b_report/stats.json` 0.9813→0.981; audit `MRI ir LDA` | BACKED ✓ |
| A23 | trust classifier's eight leakage-free features | abs L162; intro L43,L53 | ledger `ft4-lean-trust-classifier` (robust6=6) + `robust8-grayscale-router` (robust8=8 feats); `noreject-router-over-reject` (8 minus reject) | BACKED ✓ |
| A24 | IR detector F1 0.503→0.967, n=9,612, six revisions | abs L162; intro L55; concl L21 | evals.csv `ir_final_v2` 0.503, `ir_final_v3b` 0.967, split ir_final_640 n=9,612; ledger `ir-version-progression`; audit `irtest bare F1`=0.961 (own-GT cell) | BACKED ✓ |
| A25 | V5 bypass cost 12.7 points precision | intro L55; concl L21 | evals.csv V4 P=0.895 / V5 P=0.768 → 12.7pp; ledger `ir-version-progression` | BACKED ✓ |
| A26 | grayscale within 2.7 pp F1 (0.580 vs 0.607) | abs L162; intro L61; concl L25 | JSON svan_gray v3b 0.5796→0.580 vs ft4 0.6067→0.607; Δ=2.7pp; audit `3way gray F1`/`3way RGB F1` | BACKED ✓ |
| A27 | grayscale ties on flock_of_seagulls clip (0.837 vs 0.840) | intro L61 | ledger `ir-grayscale-fallback` (0.837 vs 0.840); src comment L62 | BACKED ✓ |
| A28 | raw-RGB control 0.187 (Svan), 0.295 (video) | concl L25 (intro implies via L61) | JSON svan_rawrgb 0.1874→0.187; empirical L739 video raw 0.295; audit `3way rawrgb F1` | BACKED ✓ |
| A29 | corpus counts: 129,130 thermal / 27,024 confuser / 28,710 paired | intro L57 | meth L21/L135/L157 (129,130), L22/L85 (27,024), L24 (28,710); src comment L58 | BACKED ✓ |
| A30 | 19-clip / 2,609-frame / 1,234 GT-instance video | intro L57 | meth L27/L250 (2,609; 19 clips), empirical L726 (1,234 GT); src comment L58 | BACKED ✓ |
| A31 | 517-D ROI feature vector | intro L26, L53 | ledger `v5-lda-separability` / `ir-features-separable` (517-D = 5 meta + p3@2x2 + p5@1x1); audit path `mri/results/v5_report_regen` | BACKED ✓ |
| A32 | filter beats patch: 7× fewer composed (835→39 vs 835→282) | intro L61 (implied), concl L14 | audit `rgbconf mlp FP`=39, `rgbconf patch fire`; ledger `mlp-filter-beats-cnn-ood` | BACKED ✓ |
| A33 | $P=0.989, R=0.982$ Anti-UAV (production RGB) | intro L8 | `tier1` antiuav A_bare ft4/rgb P 0.989 / R 0.982 (FP=41 row); audit `antiuav ft4 bare F1` | BACKED ✓ |

**No unverifiable numbers in this slice.** Every figure resolves to a JSON/CSV/ledger row or an
artifact-cited manifest, and all values match (modulo the two rounding/lag items C2/C6).

---

## B. CLAIMS

| # | claim | loc | backing | status |
|---|---|---|---|---|
| B1 | RF detection defeated by fiber-optic-guided drones fielded at scale since 2024 | intro L6 | CITED `csis2025dronewar` | EVIDENCED ✓ |
| B2 | radar/RF/acoustic modality limitations | intro L6 | CITED `shi2018counteruas,taha2019drone,samaras2019deep` | EVIDENCED ✓ |
| B3 | YOLO high precision/recall at real-time rates | intro L8 | CITED `redmon2016yolo,ultralytics2024` | EVIDENCED ✓ |
| B4 | Anti-UAV is in-distribution (training-corpus overlap) → "sanity floor not generalisation" | intro L8 | EVIDENCED ledger `antiuav-not-clean-but-survives`; xref sec:lit_drone, sec:svanstrom_audit | EVIDENCED ✓ (honest hedge, good) |
| B5 | "appearance gap too narrow for the detector to exploit at its decision layer without collateral damage" | intro L15 | EVIDENCED by A5/A10 (retrain trades recall); fwd-ref sec:model_mri | EVIDENCED ✓ |
| B6 | separation "demonstrably exists deeper in the detector's feature space" | intro L15 | EVIDENCED A22 (LDA 0.952/0.981); ledger `v5-lda-separability`,`ir-features-separable` | EVIDENCED ✓ |
| B7 | "detector recall, once lost, cannot be recovered downstream" | intro L19, L26 | EVIDENCED (architectural; restated as design premise, echoed empirical L79) | EVIDENCED ✓ (sound) |
| B8 | confuser corpus "two-thirds Svanström-sourced and fully unseen by production detector training" | intro L15 | EVIDENCED meth `sec:ds_confusers` + ledger; src comment present | EVIDENCED ✓ |
| B9 | reject variant "at a recall cost on clean and single-modality surfaces we judged not worth paying" | abs L159; intro L50 | EVIDENCED ledger `noreject-router-over-reject`; tab:ablation_dut/solo | EVIDENCED ✓ |
| B10 | "every reported number must correspond to a configuration the system can actually ship" | intro L28 | EVIDENCED ledger `tier1-standard-frozen`; xref sec:pipeline_speed | EVIDENCED ✓ |
| B11 | Model MRI "does not itself produce the filter; filter distilled separately" | intro L53; abs L162 | EVIDENCED ledger `mri-v5-report-regen`,`v5-beats-patch` (see **Section E** — pending revert) | EVIDENCED ✓ (but wording slated to change) |
| B12 | leakage ratio "applied to select the trust classifier's routing features" | intro L43, L53 | EVIDENCED ledger `fusion-feature-leakage`,`ft4-lean-trust-classifier` | EVIDENCED ✓ |
| B13 | "near-linear separability … makes the 517-D MLP a better confuser filter than the MobileNetV3 patch classifier" | intro L61 | EVIDENCED ledger `mlp-beats-patch-both-modalities`,`mlp-filter-beats-cnn-ood`; CITED `howard2019mobilenetv3`; xref sec:verifier_results | EVIDENCED ✓ |
| B14 | "Every number is traceable: Appendix maps each headline figure to results file + replay command" | intro L50, L64 | EVIDENCED `app:provenance` exists; `runs/README.md` (audit path); audit 180/180 | EVIDENCED ✓ |
| B15 | each model "ships a co-located *.model_card.yaml" | intro L64 | EVIDENCED ledger `thesis-model-card-provenance` (23 cards); `app:models` | EVIDENCED ✓ |
| B16 | dual-use / no-effector ethics scope | intro L66–69 | EVIDENCED (policy statement, self-consistent; reproducibility-audit tie genuine) | EVIDENCED ✓ |
| B17 | "real-video benchmark … 17 of 19 clips carry recoverable source identifiers" | intro L57 | EVIDENCED `app:datasets` (xref); meth sec:ds_youtube | EVIDENCED ✓ (could not independently count the 17/19 in this slice; relies on app — verify in appendix pass) |
| B18 | "clean thermal drone data is scarce … public drone datasets difficult in RGB, far more so in thermal/paired" | intro L55; abs L162 | EVIDENCED `settles2009active`/`ng2021datacentric` context + ledger `ir-version-progression` | EVIDENCED ✓ |

**No unsupported or overstated claims found in this slice.** The intro is unusually disciplined: every
absolute is hedged where the evidence is conditional (B4 "sanity floor", empirical-grade ties noted,
"in-corpus diagnoses that license a training attempt" vs OOD validation). The Anti-UAV
training-overlap caveat (B4) is stated up front, which pre-empts the obvious examiner objection.

---

## C. CONSISTENCY

**C1 — [FLAGGED, real] Abstract "One finding" vs Introduction "Two findings".**
- Abstract (main.tex **L162**): *"**One finding emerged unplanned**: the thermal-trained detector, run
  zero-shot on grayscale-converted RGB, still detects drones…"* — describes **only** the grayscale result.
  In the abstract, the near-linear separability is presented as part of the **designed** Model-MRI
  contribution (¶3: "found drone/confuser structure linearly separable … (LDA 0.952 RGB, 0.981 IR)"),
  i.e. NOT as an unplanned finding.
- Introduction §Contributions (**L47**): *"…four contributions, **and reports two findings** that emerged
  from them."* and (**L61**): *"**Two findings emerged** that were not designed. The first is a cross-modal
  transfer result… The second is the **near-linear separability** of drone-versus-confuser structure…"*
- Conclusion §"The findings" (**L24–25**): describes **only** the grayscale finding → agrees with the
  abstract (1), not the intro (2).
- **Net:** abstract=1, conclusion=1, intro=2. The **intro is the outlier.** Either (a) the abstract +
  conclusion should acknowledge the separability as the second unplanned finding, or (b) the intro should
  demote separability from "finding" to "the feature-geometry result that grounds Contribution 2" (matching
  abstract/conclusion). **Recommendation:** (b) is cleaner — separability is the *premise* the MRI was
  built to test (intro L53 itself calls it an "in-corpus diagnosis that licenses a training attempt"), so
  calling it an *unplanned finding* sits awkwardly next to the MRI contribution that was *designed* to find
  it. Reword L47 → "four contributions, and a cross-modal finding that emerged unplanned"; trim L61 to the
  single grayscale finding (move the separability sentence into Contribution 2's body or §verifier_results).
  Whichever direction is chosen, **the number "two" at L47 and the "Two findings emerged" at L61 must both
  change together with the abstract's "One".**

**C2 — [minor, real] Svanström production F1 prints as 0.946 in 3 places but 0.944 in 2 places —
same configuration (`filt→clf[robust8-nr]`).**
- JSON value = **0.9459**. Rounds to **0.946**.
- **0.946**: abstract L159, L162-context; intro L50; empirical `tab:ablation_svanstrom` L41 (bolded prod
  row); empirical reading L83; conclusion RQ1 L10; conclusion RQ2 L14. ✓ correct.
- **0.944**: empirical `tab:rq3` L99 ("Routed (production, robust8-nr, filt→clf)"); empirical RQ3 reading
  L87 ("Svanström 0.944, just above IR-only's 0.940"); conclusion RQ3 area references tab:rq3.
- The RQ3 table reports the **identical** shipped cell as the ablation table but rounds it to 0.944. 0.9459
  → 0.946, so **0.944 is a rounding/transcription drift** (or it was lifted from a slightly different
  pre-NR value). **Fix:** make `tab:rq3` and the RQ3 reading say **0.944 → 0.946** to match
  `tab:ablation_svanstrom` and the abstract. (Outside my slice's files, but the **abstract's 0.946 is the
  anchor** that must agree with the RQ3 table, so it belongs here.) The audit only pins the filt→clf cell
  to 0.946 (`NR svan filt->clf F1`), so the 0.944 RQ3 cell is **unpinned** and slipped through.
  *Caveat:* if "0.944" was an intentionally distinct rounding the author preferred, it still contradicts
  the bolded table value — pick one.

**C3 — [cosmetic] Chapter label `ch:hitl` points to the "Empirical Evaluation" chapter.**
The intro references `Chapter~\ref{ch:hitl}` at L53, L74 (and elsewhere) for the empirical chapter; the
label resolves correctly (no broken ref) but the **name is legacy** (chapter is no longer titled
"HITL"). Not a defect; flagging only because a future grep on "hitl" could mislead. Leave as-is or rename
`ch:hitl→ch:empirical` repo-wide (low priority).

**C4 — [resolved, not a defect] Two distinct "in-domain RGB test" numbers (0.929 vs 0.926).**
Intro uses **0.929** (= `v5_rgbds_bare` ft4, config `rgb_dataset_iou_640`); empirical filter tables use
**0.9259** (tier1 `rgb_dataset_test` verifier surface). These are **different evals at different
configs** and both are correctly source-commented. No fix needed — noting so a later reviewer doesn't
"reconcile" them into one. (The intro's framing "leaving the composite in-domain test essentially
unchanged, 0.942→0.929" is a 1.3-pp **drop**, fairly called "essentially unchanged".)

**C5 — [micro nit] "small-drone recall (0.961 → 0.306)".**
Abstract L156 and intro L19 cite the collapse as 0.961→0.306. In the cache CSV the retrained_v2 DRONE
**recall** is 0.306 ✓, but **0.961** is the *baseline* drone **recall** on the **full** Svanström drone
set (matches `rgb_svan_baseline` R=0.961); the per-category CSV row shows baseline DRONE R=0.959 on the
1,299-frame confuser-eval subset. So 0.961 vs 0.959 is the full-set vs subset distinction — 0.961 is the
right number to cite (full set). Consistent; no action.

**C6 — [housekeeping, not thesis] Audit pin + ledger lag the live JSON on the 0.11% confuser cell.**
Prose says **0.11%** (3/2,633), which **matches the JSON** (`clf->filt[robust8]` fire=**0.0011**). But the
audit pin (`_audit_headline_numbers.py` L37 `rgbconf composed fire = 0.0015`) and ledger row title
`composed-rgbconf-0p15pct` still encode **0.15%**. The pin passes only because |0.0015−0.0011|=4e-4 < the
5e-4 tolerance. **The thesis prose is correct; the audit pin/ledger are stale.** Recommend updating the pin
to 0.0011 and renaming the ledger row, so the audit actually guards the 0.11% the thesis now prints.
(Not a thesis-text defect — listed for the knowledge-system owner.)

**C7 — [model-name check] All model names consistent in this slice.**
`ft4`, `v3b`, `robust8-nr` (prose) / `robust8_nr_drop` (JSON key), `robust8`, `robust6`, `sa32`,
`mlp_v5`, `mlp_aligned_thermalonly`, `fusion_no_fn_v1.1` — all used consistently with the rest of the
thesis and the ledger. Abstract uses plain-language "MLP confuser filter" / "trust classifier" (no raw
weight names), which is appropriate for an abstract. ✓ The task's "mlp_v5_v4" spelling does not appear in
this slice (intro/abstract refer to it generically as "the 517-D MLP" / "MLP confuser filter"); the
`v4` bird-split detail lives in the conclusion/empirical, which is fine.

---

## D. FIGURES

The Abstract has no floats. The Introduction has **no `\begin{figure}` and references no figure by
`\ref`** — it only forward-references **sections** (and Figures `fig:confuser_examples`,
`fig:confuser_fp_examples` are referenced from the **methodology**, not the intro; the intro mentions
them only in prose at L15 without a `\ref`). This is structurally fine for an IMRAD intro.

**Cross-check against `2026-06-18_verify_v7_figures.md`:**
- The figure audit lists `fig:datasets_pie` and `fig:dataset_montage` as **ORPHANS** (defined in
  methodology §3.1, never `\ref`'d anywhere). The intro's Contribution 4 (L57) enumerates exactly the
  corpora those two figures depict (129,130 / 27,024 / 28,710). **PROPOSED:** the natural anchor for
  `fig:datasets_pie` is **not** the intro (keep the intro lean) but methodology §3.1 — already covered by
  fix #3 in the figure audit. **No new intro figure is warranted**; the "datasets piechart the notes
  wanted" already exists (`fig_datasets_pie`, train 327,619 / eval 151,695) and just needs an in-text
  `\ref` in §3.1, not the intro.
- **No intro figure is promised-but-missing.** The intro makes its case in prose + section refs, which is
  correct for this thesis's style.

**PROPOSED NEW FIGURE (optional, low priority):** none required for the intro. If the committee wants a
visual "system at a glance" early, the existing `fig:pipeline` (tikz architecture, methodology §3.8.1,
currently an orphan) could be `\ref`'d once from the intro's Contribution-1 sentence ("…fused by an
XGBoost trust classifier…", L50) — but that duplicates the methodology anchor and is not necessary.

---

## E. MRI ↔ filter relationship — every location (for the pending revert)

The user is reverting the wording back to: *"the MRI was used to train the MLP filters (pos=drones,
neg=confusers)."* Current text everywhere says the **opposite** (MRI is diagnostic-only; filter distilled
**separately**; "the MRI does not itself produce the filter"). Locations **in my slice**:

1. **Abstract, main.tex L162:** *"A statistics-before-training instrument (the Model MRI) measures what a
   detector's feature space can separate before anything is trained… These diagnoses **justify** a
   lightweight MLP confuser filter, **which is then distilled separately** from the detector's own ROI
   features."*
2. **Introduction L43** (RQ preamble): *"a model-agnostic feature-space instrument (the Model MRI)
   **measured what the detectors' features can separate before any training run**, and a leakage-ratio
   statistic… excluded scene-fingerprint features from the trust classifier."* (MRI framed as
   measurement, not trainer.)
3. **Introduction L53** (Contribution 2): *"Its role is **diagnostic**: it measures whether a downstream
   decision is separable… and a positive diagnosis **justifies training** a lightweight MLP confuser
   filter, **which is then distilled separately from those same features (the MRI does not itself produce
   the filter)**."* ← the single most explicit "MRI ≠ trainer" sentence.
4. **Introduction L53 (cont.):** *"Drone/confuser linear separability measured 0.952 (RGB) and 0.981
   (IR)… these are **in-corpus diagnoses that license a training attempt**, and the filter's
   out-of-distribution evals… are what validate it."*

Related (not strictly MRI↔filter but co-located framing the revert may touch):
- **Introduction L26:** defines the filter as "a small MLP that re-reads the 517-dimensional ROI feature
  vector the detector already computed" — consistent with either framing (does not assert who trained it).
- **Introduction L15, L61:** "exploiting it [separability] there instead of at the detection head is
  precisely the design"; "near-linear separability… is what makes the 517-D MLP… a better filter" —
  these are separability claims, neutral to the revert.

Outside my slice (for the editor's awareness — confirm in the methodology/empirical passes):
`sec:model_mri`, `sec:distill_verifier`, `sec:mri_findings`, and the figure-audit note that the
**0.500→0.919 AUROC** progression is cited in abstract/intro/§lit_probing/§ir_xmodal_verifier/§mri_findings.
The revert will need to keep all five MRI↔filter restatements **mutually consistent** with whatever the
methodology says, or the audit's CBAM-prose-drift guard (and a reader) will catch a split.

**Not flagged as a defect per instructions** — listed only so the revert hits every spot.

---

## F. TOP ISSUES (ranked)

1. **C1 — Abstract "One finding" vs Intro "Two findings" (+ conclusion sides with "one").** The one
   genuine cross-artifact inconsistency. Decide whether separability is a *finding* or the MRI's
   *designed result*, then align the abstract count, intro L47 + L61, and conclusion §"The findings".
   Recommended: keep it as ONE unplanned finding (grayscale) and demote separability to Contribution-2
   premise — matches abstract + conclusion and the intro's own "in-corpus diagnosis" framing.
2. **C2 — Svanström production F1 0.946 vs 0.944 in tab:rq3 / RQ3 reading.** JSON=0.9459→0.946; the
   abstract anchors 0.946, so the RQ3 table + reading should read 0.946 (unpinned cell that drifted).
3. **E (pending revert, NOT a defect)** — four MRI↔filter restatements in this slice (abstract L162, intro
   L43/L53/L53). Listed for the revert; keep all consistent with the methodology after the edit.
4. **C6 (housekeeping)** — update audit pin `rgbconf composed fire` 0.0015→0.0011 and rename ledger row
   `composed-rgbconf-0p15pct`; thesis prose (0.11%) is already correct and matches the JSON.
5. **C3 (cosmetic)** — legacy label `ch:hitl` now points to "Empirical Evaluation"; resolves fine,
   optional rename.
6. **Minor prose-precision nits:** (a) abstract L156 "$2.8\%$ of its composite test corpus" — add "RGB"
   to name the surface (intro L8 already says "composite RGB test corpus"; MEMORY rule
   number-needs-dataset). (b) Otherwise every number names its surface.

**Bottom line:** integrity is intact — 0 unverifiable numbers, 0 unsupported claims, 0 dangling refs in
this slice, audit 180/180. Ship after fixing C1 (the findings count) and C2 (the 0.946/0.944 rounding);
everything else is housekeeping or the planned MRI↔filter revert.

### Delivered
- `docs/analysis/2026-06-18_review_intro.md` (this file) — absolute path:
  `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_review_intro.md`
- Read-only review; no thesis/source/knowledge files modified.
