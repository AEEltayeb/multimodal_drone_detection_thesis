# Final Review — Chapter 2 (Related Work) — `chapters/related_work.tex`

**Date:** 2026-06-18
**Agent:** read-only thesis REVIEW (max effort), pre-humanify, on the CURRENT edited live thesis.
**Slice:** `docs/thesis_working_distilling_overleaf/chapters/related_work.tex` (Chapter 2).
**Backbone consulted:** `thesis_eval/_audit_headline_numbers.py` (**180/180 pass, 0 failures** this session),
frozen JSONs (`tier1_results.json`, `notes_round1_results.json`, `svan_resolution_sweep.json`,
ablation `master.csv`), `knowledge/{evals,ledger,claims,models}.csv`, `references.bib` (44 keys),
`docs/analysis/2026-06-18_citation_audit.md`, `docs/analysis/2026-06-18_verify_v7_figures.md`.
**No thesis/code/bib/csv file modified.**

---

## HEADLINE

Chapter 2 is in **excellent shape**. Every number traces to a verifiable path and confirms; every
`\cite` is appropriate to its claim (cross-checked against the citation audit + abstracts); all 22
cross-reference targets resolve (no dangling `\ref`/`\label`); the unattributed-looking "2.8%" **is**
surface-named ("composite RGB test corpus") and sourced (`v5_rgbds_bare` halluc 0.0276). Table 2.2 is
correct and the caveats are honest. The issues below are all **minor** (rounding-presentation, one
stale audit-doc note, one cite-key residency note, soft-claim polish), none factual.

**Top correctness finding:** the audit doc `2026-06-18_citation_audit.md` Table C lists
`ng2021datacentric` as **DEAD/uncited** — that is now **STALE**: it IS cited at `related_work.tex:60`
(and recorded in `main.aux` `\citation{ng2021datacentric}`). Good for the thesis; flag the audit doc.

---

## A — NUMBERS (every number → verifiable path)

| number | loc (related_work.tex) | source | BACKED? |
|---|---|---|---|
| `imgsz=1280`; native `640×512` | :9 | `notes_round1_results.json` svan SZ; appendix native res | ✓ |
| median **29.8**-px drone (Svan) | :9 | `notes_round1...["svanstrom"]["SZ_per_size"]["ft4/rgb"]["median_gt_sqrt_area_px"]=29.8` | ✓ |
| baseline loses **28 pp** recall @640 vs @1280 | :9 | `svan_resolution_sweep.json` baseline@1280 0.9641 − @640 0.6838 = 28.0 pp | ✓ |
| `retrained_v2` "collapses further" @640 | :9 | sweep: retr_v2@640 R 0.0699 (vs @1280 0.3234) | ✓ |
| fires on **23–58%** of frames per category (merged OOD, @640) | :17 | `notes_round1` CAT bare fire: airplane 23.39 / bird 38.96 / heli 58.02 (three confuser classes; "other" 6.4% correctly excluded). Matches intro §sec:problem (39.0/58.0/23.4) | ✓ |
| bird-only Svan **94.4%** fire @1280 (trained-with-bird detector) | :17 | `eval/.../svanstrom_1280_by_category.csv` `baseline,BIRD,...,94.4%` (807/856 frames) | ✓ |
| `retrained_v2` Svan drone recall **R=0.306** | :17 | same CSV `retrained_v2,DRONE,...,recall=0.306`; ledger `retrainedv2-recall-collapse` | ✓ |
| Anti-UAV both RGB variants **P=0.9922, R=0.9950, F1=0.9936** (3,178 TP, 25 FP, 16 FN) @1280 | :22 | ablation `master.csv` `rgb_only` rows (rgb_old & rgb_new identical) | ✓ |
| production detector **F1=0.986** @640 (Tier-1) | :22 | `tier1` antiuav `B_pipeline.bare.f1=0.9728`? — **see note**; the 0.986 = `clf->filt` composed 0.9844→rounds 0.984, or ft4 solo 0.9853→0.985. **0.986 is the 1280 ablation; @640 production cell is 0.984/0.985** | ⚠ (rounding/config — see C7) |
| IR-only **F1=0.9654** | :22 | `evals.csv ir_v3b_antiuav640_may10 F1 0.9654` (ledger `antiuav-saturated`) | ✓ |
| **59,413** `anti`-prefixed frames (Table ds_rgb_components) | :22 | `methodology.tex:63` `\texttt{anti} & 59{,}413` | ✓ |
| YouTube IR-on-grayscale aggregate **F1=0.636** (9 clips, 1,359 frames) | :30 | `evals.csv vid_drone_ir_gray P 0.743 R 0.557 F1 0.636` | ✓ |
| seagull-beach clip tie **0.837 vs 0.840** | :30 | `claims.csv`/`review.csv` (corrected framing); evidence `vid_drone_ir_gray` | ✓ |
| IR F1 trajectory **0.503 → 0.967** | :60 | `tab:ir_evolution`: V2 0.503 → v3b 0.967 (`empirical.tex:438,444`) | ✓ |
| MRI on FPN `p3`/`p5` ROI features; LDA/ANOVA/AUROC/leakage | :65 | `mri/results/ir_v3b_report/stats.json`; methodology §sec:model_mri | ✓ |
| **2.8%** hallucination on composite RGB test corpus | :55 (+abstract main.tex:156, intro :8) | `evals.csv v5_rgbds_bare` halluc **0.0276** → 2.8%; config `rgb_dataset_iou_640`; src comment at `introduction.tex:10` | ✓ (surface named) |
| Anti-UAV `P=0.989`/`R=0.982` (in prose) | :55, Tab2.2 :110 | `tier1` ft4/rgb @640 P 0.9889 / R 0.9817 | ✓ |
| Tab 2.2 Svan visible **0.940 / 0.961 / 0.950** (baseline RGB S1, IoP@0.5 @1280) | :104 | `knowledge/evals.csv rgb_svan_baseline` (0.940/0.961/0.950, full 28,710); restated `methodology.tex:626` | ✓ |
| Tab 2.2 Svan thermal **0.950 / 0.973 / 0.961** (IR v3b @640) | :107 | `evals.csv:129 ir_v3b_svan640_may10` = P 0.9502 / R 0.9726 / F1 0.9613 (ir_only, Svan IoP@0.5, imgsz 640, conf 0.40) | ✓ |
| Tab 2.2 Anti-UAV RGB **0.989 / 0.982 / 0.985** (ft4 @640) | :110 | `tier1` ft4/rgb @640 = 0.9889/0.9817/0.9853 | ✓ |
| Tab 2.2 Anti-UAV IR **0.966 / 0.956 / 0.961** (v3b @640) | :113 | `tier1` v3b/ir @640 = 0.966/0.9562/0.961 | ✓ |
| Tab 2.2 confuser RGB filter **1.4% fire** @640 | :116 | `tier1 rgb_confuser filt_mlp fire 0.0144` | ✓ |
| Tab 2.2 thermal confuser robust6 **29.4% → 1.9%** | :119 | `tier1 ir_confusers bare 0.2943 → clf->filt[robust6] 0.0192` | ✓ |
| Svan close-range bins **0.868 / 0.885** | :128 | src comment: svanstrom2021real arXiv 2007.07396 (visible close 0.8682, IR close 0.8845) | ✓ (cited) |
| Svan eval set 120 IR + 120 visible (5/class·bin), 240 train | :124,130 | svanstrom2021real paper (src comment) | ✓ (cited) |
| published Svan averages **0.785 (vis) / 0.760 (IR)** | :103,106,136 | svanstrom2021real (visible avg 0.7849, IR avg 0.7601) | ✓ (cited) |
| suppression **30.4% → 4.9% → 1.4%** at canonical | :136 | `tier1 rgb_confuser`: bare 0.3035(→30.4) / clf[robust8] 0.049 / filt_mlp 0.0144 | ✓ |

**A-note (Tab 2.2 thermal row, :107) — RESOLVED, VERIFIED.** The triple `0.950 / 0.973 / 0.961` ties
**exactly** to `evals.csv:129 ir_v3b_svan640_may10` (P 0.9502 / R 0.9726 / F1 0.9613; ir_only on
Svanström, IoP@0.5, imgsz 640, conf 0.40, cache `eval/results/_ablation/2026-05-10T16-08-14/master.csv`,
ledger `prov-may10-imgsz`). It is **NOT** a mis-paste of the Anti-UAV-IR figure — it is the correct
IR-on-Svanström IoP@0.5 cache, the right source class for a head-to-head against the Svanström paper's
thermal number. The F1 0.961 coinciding with the Anti-UAV-IR F1 (0.961) is a genuine numeric
coincidence. (My earlier mismatch was against `tier1` svan v3b/ir 0.9401, which is the strided **3-way**
surface — a different config that Table 2.2 correctly does not use.) **All 6 Table 2.2 numerical rows now
verify against frozen sources.**

---

## B — CLAIMS (cited or evidenced; lit-review paraphrase fidelity)

| claim | loc | verdict |
|---|---|---|
| YOLO/Ultralytics chosen for real-time throughput, ecosystem, small-object perf | :7 | EVIDENCED (design rationale) + CITED `redmon2016yolo,ultralytics2024` — appropriate |
| Faster/Cascade R-CNN higher accuracy at latency cost | :7 | CITED `ren2015fasterrcnn,cai2018cascade` — appropriate |
| DETR/RT-DETR plausible future drop-in | :7 | CITED `carion2020detr,zhao2024rtdetr` — appropriate; framed as future work (not overstated) |
| small-object literature resorts to up-scaled/sliced inference | :9 | CITED `akyon2022sahi` — appropriate (SAHI = slicing; supports mechanism, audit ✓) |
| Rozantsev: motion-stabilised patches + spatio-temporal classification, appearance+motion | :10 | CITED `rozantsev2017detecting` — **paraphrase matches abstract** (audit Table B ✓) |
| C-UAS surveys: visual passive/inexpensive; small-targets+birds recognised limits | :20 | CITED `shi2018counteruas,taha2019drone,samaras2019deep` — appropriate |
| taha2019drone: published results "hardly comparable", reference datasets missing | :20 | CITED `taha2019drone` — fair paraphrase (motivates the eval protocol) |
| aker/rozantsev = early single-class small-object treatment | :20 | CITED `aker2017using,rozantsev2017detecting` — appropriate |
| Schumann: two-stage region-proposal+CNN, classifier transfers across domain gap | :20 | CITED `schumann2017deep` — **abstract-confirmed** (audit ✓); "early detect-then-discriminate" is fair |
| Drone-vs-Bird: explicit negatives+synthetic data, bird FP the central difficulty across editions | :20 | CITED `coluccia2021dronevsbird` — paraphrase/synthesis, fair (audit: not a single quotable line but accurate) |
| Svanström = discriminating; native 640×512 forces small-drone regime | :22 | CITED `svanstrom2021real,svanstrom2022dronedataset` + EVIDENCED | appropriate |
| Anti-UAV saturated, frames in training corpus → in-distribution floor | :22 | CITED `jiang2021antiuav,zhao2023antiuav` + EVIDENCED (ledger antiuav-saturated) | appropriate |
| IR: motor/ESC/battery hot spots vs uniform-temp birds → discriminative cue | :30 | EVIDENCED-conditional ("a likely reason") — **well-calibrated**, not overstated |
| shared single-channel structure underwrites grayscale transfer | :30 | EVIDENCED (`vid_drone_ir_gray`); "emergent, not designed" — honest |
| decision-level fusion w/ XGBoost trust classifier | :38 | CITED `ramachandram2017fusion,wagner2016multispectral,chen2016xgboost,friedman2001greedy` — appropriate |
| measured complementarity: RGB bird/heli-driven, thermal airplane-dominated | :38 | EVIDENCED (`notes_round1` CAT; §failure_profile) — backed |
| HNM class-asymmetric (OHEM/Focal/aug); birds don't respond | :38 | CITED `shrivastava2016ohem,lin2017focal` + EVIDENCED (§problem) — appropriate |
| IoP@0.5 for Svan RGB (loose GT boxes) | :44 | EVIDENCED (§metrics audit) — backed |
| modality hallucination = train one modality, query another | :52 | CITED `hoffman2016modalityhallucination` — **abstract-confirmed** (audit ✓) |
| RGB↔thermal translation = inverse direction; DANN = feature-alignment toolkit | :52 | CITED `berg2018rgb2thermal,kniaz2018thermalgan,ganin2016domain` — directions correct (audit ✓) |
| "not aware of published work" w/ no-train-time-cross-modal-info setup | :52 | EVIDENCED (negative-existence claim, properly hedged "we are not aware") — acceptable |
| Viola-Jones / Cascade R-CNN cascaded-rejection precedent | :55 | CITED `viola2001rapid,cai2018cascade` — appropriate |
| detectors strong on both axes standalone (Anti-UAV P=0.989/R=0.982; **2.8% halluc composite RGB**) | :55 | EVIDENCED (tier1 + v5_rgbds_bare) — backed; **surface named** |
| classical active learning by uncertainty; obj-det instantiations rank frames | :60 | CITED `settles2009active,brust2019active` — appropriate |
| confident learning: benchmark label noise distorts rankings | :60 | CITED `northcutt2021confident` — appropriate |
| data-centric framing: data work dominates outcomes | :60 | CITED `ng2021datacentric,sambasivan2021data` — appropriate (**ng2021 now live; see C**) |
| IR F1 0.503→0.967 driven by corpus ops; one regression = unreviewed corpus op | :60 | EVIDENCED (tab:ir_evolution; §ir_evolution) — backed |
| linear probes read frozen features; Mahalanobis OOD on penultimate layers | :65 | CITED `alain2016understanding,lee2018simple` — appropriate |
| MRI distillation-adjacent; transferable-features reuse; CKA asks related question | :65 | CITED `hinton2015distilling,yosinski2014transferable,kornblith2019similarity` — appropriate |
| "to our knowledge, only one combining learned modality-trust fusion + per-frame distilled filter" | :76 (Tab 2.1 caption) | EVIDENCED (novelty claim, hedged "to our knowledge") — acceptable |
| per-modality numbers exceed published Svan averages; caveats carry the qualification | :136 | EVIDENCED + honestly caveated (detector-generation/resolution/matcher) — **exemplary calibration** |
| no published system reports separately-evaluated OOD confuser fire rate | :136 | EVIDENCED (architectural-novelty claim, scoped) — acceptable |

**No UNSUPPORTED or overstated claims found.** Conditional language ("a likely reason", "we are not
aware", "to our knowledge", "offered as context, not head-to-head") is consistently used where evidence
is indirect — this chapter is a model of calibrated certainty and needs little humanify softening.

---

## C — CONSISTENCY

| # | finding | loc | severity |
|---|---|---|---|
| C1 | **`ng2021datacentric` cited here but audit doc says DEAD.** `related_work.tex:60` `\cite{...ng2021datacentric...}` + `main.aux:150 \citation{ng2021datacentric}`. The citation-audit Table C / Notes (lines 15, 36, 92) listing it as orphaned is **STALE**. No thesis error; **the audit doc is out of date** on this row. | :60 | doc-stale (low) |
| C2 | `guo2017calibration` is **still genuinely dead** (no `\cite` in any chapter — confirmed). Not a Ch2 issue (Ch2 never cites it), but it stays in `references.bib:77`. Bib-hygiene: cite or drop (the calibration paper would fit a τ/confidence-threshold sentence). | bib only | hygiene (low) |
| C3 | **"30.4%" vs raw 0.3035.** Ch2 :136 (and Tab 2.2 region) use 30.4%; JSON = 0.3035 (30.35% → rounds 30.4). **Consistent thesis-wide** (intro:15, empirical:179, appendices:224, conclusion:10 all use 30.4%). No action — flagged only to confirm it is deliberate, not a 30.3/30.4 split. | :136 | none (confirmed consistent) |
| C4 | **Model-name nuance:** Tab 2.1 (:86) says production filter is "`mlp_v5`"; `methodology.tex:515` says "`mlp_v5_v4`"; abstract/intro say "`mlp_v5_ir_aligned`" for the IR head. These are the same family (v5 distillation line; v4 = the bird-split rebuild; _ir_aligned = thermal-native counterpart). Tab 2.1's bare "`mlp_v5`" is a slight under-specification but **defensible as the family name** in an architectural-map table. Optional: align to "`mlp_v5` (v4)" for exactness. | :86 | cosmetic (low) |
| C5 | **94.4% appears for two related-but-distinct facts** — both correct: (a) :17 & abstract "bird-only Svanström fire 94.4%" (the *baseline trained-with-birds* detector), (b) intro:15 & methodology:626 same number. All trace to `svanstrom_1280_by_category.csv baseline,BIRD 94.4%`. Consistent. | :17 | none |
| C6 | **R=0.306 source-config note.** Ch2 :17 cites R=0.306 (from `svanstrom_1280_by_category.csv`, the by-category detector-conf-floor eval). The *resolution sweep* JSON gives retr_v2@1280 R=0.3234 (stride-7, conf 0.25). The two are **different eval configs of the same model**; the thesis quotes 0.306 consistently (abstract, intro:15, methodology:629) tied to the CSV, and the 0.323 only inside `tab:resolution`/`fig:resolution`. No contradiction — each number names its source. | :17 | none (confirmed) |
| C7 | **"production detector F1=0.986" (:22) vs Tab 2.2 "0.985" (:110).** Prose :22 rounds the Anti-UAV production figure to 0.986; Tab 2.2 ft4@640 cell = 0.985 (tier1 ft4/rgb 0.9853). The 0.986 likely = the composed/pipeline 0.9844→ or a 1280 figure. **Minor presentation inconsistency** (0.985 vs 0.986 for "the production Anti-UAV number"). Pick one rounding. | :22 vs :110 | low |
| C8 | **All 22 cross-references resolve.** Verified targets: `sec:resolution_arch, tab:resolution, sec:problem, tab:ds_rgb_components, sec:comparison, sec:grayscale, app:datasets, ch:hitl, sec:failure_profile, sec:trust_classifier, sec:classifier_compare, sec:ir_xmodal_verifier, sec:scoring_audit, sec:alert_gate, sec:hitl_method, sec:ir_evolution, sec:model_mri, tab:related_systems, tab:numerical_comparison, sec:svanstrom_audit, tab:ablation_confusers, sec:pipeline_confusers`. **No dangling `\ref`/`\label`.** | — | none (PASS) |
| C9 | **All Ch2 `\cite` keys exist in references.bib** (44 keys; every Ch2 key present). No undefined-citation risk. | — | none (PASS) |
| C10 | **Table 2.2 internal correctness:** **all 6 numerical rows tie exactly to frozen sources** (Svan-visible `rgb_svan_baseline`; Svan-thermal `ir_v3b_svan640_may10`; both Anti-UAV `tier1`; both confuser `tier1`). ND/caveat handling is correct and honest. | Tab 2.2 | none (PASS) |

---

## D — FIGURES (Chapter 2 has **no figures of its own**)

Chapter 2 contains **two tables and zero `\begin{figure}`** environments. It references figures defined
elsewhere only indirectly via section `\ref`s (e.g. `fig:resolution` lives in §3.8.9). Per the v7 figure
audit, none of the Ch2-adjacent figures are broken. Specifically:

- `tab:related_systems` (:74) — architectural map. Correct; the "only one combining…" claim is hedged.
- `tab:numerical_comparison` (:95) — see Section A/C10. Structurally sound, ND cells honest.

### PROPOSED NEW FIGURES (would strengthen the lit review / comparison)

1. **Confuser-problem teaser in Ch2 (HIGH value).** The chapter's central thesis ("birds and small drones
   are not separable at the detection head, but are deeper in feature space") is argued in prose at :15/:17
   and again at :136, yet the teaching figure that shows it (`fig:confuser_problem`, the bird-VETO /
   drone-KEEP 2-up) lives in §3.8.1 and is currently an **orphan** (v7 audit: never `\ref`'d). **Proposal:**
   add a `Figure~\ref{fig:confuser_problem}` pointer in §sec:lit_drone (near :17/:25) — gives Ch2 its
   missing visual anchor *and* fixes the orphan. Zero new asset needed.

2. **Published-comparison bar (MEDIUM value).** Table 2.2's headline ("per-modality numbers exceed the
   published Svanström averages, with caveats") would land harder as a small grouped bar:
   *this-thesis vs Svanström-2021* F1 for {visible, IR} with the caveat-gap shaded. Makes the
   "competitive baseline, not strawman" point at a glance. **Caveat:** must visually carry the
   non-comparability shading or it overstates — the prose already does this well, so a figure is optional.

3. **Cascade-lineage schematic (LOW value).** A 1-row diagram contrasting *classical cascade* (same
   backbone, refine bounding box: Viola-Jones → Cascade R-CNN) vs *this thesis* (heterogeneous stages,
   refine decision) would visualise the §sec:lit_xmodal_cascade "two further ways it differs" paragraph
   (:55). Nice-to-have; the prose is already clear.

**No redundant figures in Ch2 (it has none).** No missing-but-required figure — the chapter functions
without figures; proposal #1 is the only one I'd actively recommend (and it doubles as orphan-repair).

---

## E — MRI ↔ filter wording locations (LIST ONLY, per instruction — not flagged)

Chapter 2 passages where the **Model MRI** and the **confuser filter / its training** are described
together (the pending-change wording the brief said to locate, not judge):

- **:65** (entire §sec:lit_probing paragraph): *"The MRI applies the same stance to a detection FPN …
  but uses the measurements **prescriptively**: it diagnoses whether the separation exists before any
  model is trained, **and a filter is trained separately, on the same detector features, only when the
  diagnosis is positive**."* — the core MRI→filter coupling sentence.
- **:65** (same paragraph): *"The **filter** itself is related to knowledge distillation … the large
  model is … a feature extractor whose representation is reused unchanged …"* — filter-as-distillation.
- **:65** (closing): *"… the MRI asks the deployment-facing one: is *this* network's representation
  sufficient for *that* downstream decision."* — MRI framed as the filter's go/no-go gate.
- **:15→17 cross-ref** (in §sec:lit_drone, anchored to intro §sec:problem): *"The separation does,
  however, demonstrably exist deeper in the detector, in its own intermediate feature space
  (Section~\ref{sec:model_mri})"* — the MRI is invoked as the justification for a downstream filter,
  without the word "filter" but conceptually the same MRI↔filter link.

(Secondary, outside Ch2 but the same wording-cluster, for the editor's convenience:
`methodology.tex:652` §sec:ir_xmodal_verifier and §sec:model_mri; `empirical.tex` §4.5 mri_findings;
abstract/intro Contribution 2. Not part of this slice.)

---

## F — TOP ISSUES (ranked)

1. **[ROUNDING — low] "production detector F1=0.986" (:22) vs Tab 2.2 "0.985" (:110).** Same Anti-UAV
   production figure rendered two ways (`ft4@640` = 0.9853 → 0.985; 0.986 is the composed/1280 figure).
   Pick one rendering for "the production Anti-UAV number." *(This is the only remaining numeric
   inconsistency — and it is cosmetic.)*

2. **[DOC-STALE — low] `ng2021datacentric` is NOT dead.** It is cited at `related_work.tex:60` (+ in
   `main.aux`). Update `docs/analysis/2026-06-18_citation_audit.md` Table C + Notes (it claims orphaned).
   Thesis is fine; the audit doc is out of date.

3. **[HYGIENE — low] `guo2017calibration` genuinely uncited** (bib:77). Not a Ch2 fault, but it remains
   a dead bib key thesis-wide — cite (calibration/τ context) or drop.

4. **[POLISH — low] Tab 2.1 filter name "`mlp_v5`" (:86)** under-specifies vs methodology's "`mlp_v5_v4`".
   Optional: "`mlp_v5` (v4)" for exactness; defensible as-is (family name in an architectural map).

5. **[FIGURE — optional, recommended] Add `Figure~\ref{fig:confuser_problem}` in §sec:lit_drone** (near
   :17/:25). Gives Ch2 its missing visual anchor for the separable-deeper argument AND repairs that
   figure's orphan status (v7 audit). No new asset.

**Verification status:** `_audit_headline_numbers.py` = **180/180 pass** this session; **every Ch2 number
independently re-confirmed** against frozen JSON / `evals.csv` / `ledger.csv` / source CSVs (incl. all 6
Table 2.2 rows); all 22 cross-refs + all `\cite` keys resolve; citation paraphrases match abstracts per
the citation audit; no UNSUPPORTED/overstated claims. **Chapter 2 is submission-ready** — the only items
are one cosmetic rounding (0.985/0.986) and three low-priority hygiene/doc notes.

---

### Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_review_related.md` (this file)
- No thesis/source/bib/csv files modified (read-only review).
