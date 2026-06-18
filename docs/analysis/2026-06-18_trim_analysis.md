# Thesis Concision / Trim Analysis — 2026-06-18

**Goal:** Trim a ~132pp thesis body substantially by cutting (a) repetition, (b) low-value/verbose
passages, (c) tightening prose — without cutting evidence, numbers, RQ structure, or load-bearing
claims. READ-ONLY analysis; feeds a separate trimming PLAN (no edits made here).

**Live thesis:** `docs/thesis_working_distilling_overleaf/` — `main.tex` + `chapters/{introduction,
related_work, methodology, empirical, conclusion, appendices}.tex`. Stale files
(`docs/thesis_chapters.tex`, `docs/thesis_working.tex`) ignored per brief.

**Exact page spans (from `main.toc`; body ends p132, bibliography p133):**

| Chapter | Pages | ~pp | Source lines | Density verdict |
|---|---|---|---|---|
| 1 Introduction | 1–11 | 11 | 74 | **can-trim** (problem statement + contributions over-explained) |
| 2 Related Work | 12–23 | 12 | 141 | tight-ish; **can-trim** (§2.9 comparison + repeated confuser framing) |
| 3 Methodology + System Design | 24–68 | **45** | 800 | **verbose** — biggest target; superseded-component sections, restated tables |
| 4 Empirical Evaluation | 69–110 | **42** | 798 | **verbose** — biggest target; "five readings" prose restates tables; design-evolution dupes |
| 5 Conclusion | 111–116 | 6 | 36 | **can-trim** (RQ answers re-quote every Ch4 number in full) |
| App A–E | 117–132 | 16 | 273 | mostly keep (provenance/datasets load-bearing); App C verbatim report can shrink |
| **Body total** | **1–132** | **132** | 2122 | — |

**Where the leverage is:** Ch3 (45pp) + Ch4 (42pp) = **87pp / 66%** of the body. Any serious
reduction has to come mostly from these two. They are verbose for three structural reasons:
1. **Superseded-component sections kept in full** (patch filter, alert-gate cascade, fail-open,
   `sa32`, `robust6`, reject-class `robust8`, dual-classifier, `fusion_no_fn`, `control40`,
   scoring-swing history) — each re-explained where it appears AND in design-evolution paragraphs.
2. **Prose that restates the tables it sits next to** — the "Five readings"/"Three observations"/
   "Two readings"/"Three per-surface readings" blocks after almost every table re-narrate numbers
   already in the table.
3. **The same ~8 claims repeated across all 5 chapters** (confuser 30.4%, retrained_v2 collapse,
   scoring swing, imgsz=1280, IR-hallucinates-less, robust8-nr trade, "detectors are good"),
   often verbatim with the same numbers and the same source comments.

---

## STATUS / progress
- [x] Read author notes, all 6 chapter files + main.tex + .toc page map
- [x] Per-chapter density read
- [x] TABLE 1 Repetition
- [x] TABLE 2 Verbose/low-value
- [x] TABLE 3 Compress/merge
- [x] DO NOT CUT list
- [x] TOP-10 prioritized actions

---

## Per-chapter notes (detail)

**Ch1 Introduction (11pp).** Background tight. Problem Statement (p2–5, ~3pp) is dense but several
sentences are the *first* statement of claims later repeated 3–4×; the chapter itself is mostly
fine — the waste is downstream duplication of it. Contributions (p6–9): four enumerated items are
each a long paragraph that **re-states the abstract** and pre-announces Ch4 numbers in full
(Svanström 0.742→0.946, confuser 30.4%→1.4%, speeds 404×/37–72×). The last two paragraphs
("Two findings…" + "Evaluation is held to a single declared standard…") **duplicate the abstract's
para 2–3 almost verbatim** and pre-duplicate §3.2/§App D. Ethics (1pp) fine. Outline (1pp) is a
prose re-list of the ToC — candidate to cut to a short paragraph.

**Ch2 Related Work (12pp).** Genuinely needed for a thesis. But: §2.2 and §2.4 both re-derive the
confuser/recall-collapse story already in §1.2 with the same numbers (94.4% bird, retrained_v2
R=0.306). §2.1 imgsz paragraph duplicates §3.8.6. §2.9 (Comparison to Prior Work, p20–23, ~3.5pp)
is the heaviest: two tables + three caveat paragraphs + "Two claims survive" — the caveats restate
each other and restate §3.2.2/§3.3. The scoring-swing "28 pp" appears here AND §3.2.2 AND §2.5.

**Ch3 Methodology (45pp) — primary target.** Structure is sound but padded:
- §3.1 datasets (p24–34, ~10pp): mostly load-bearing (provenance). Some per-corpus prose repeats
  the table beneath it. `IR_confusers` §3.1.4 re-explains CBAM probe at length, then App A.4 repeats
  it, then §3.8.8/§4.3.4 repeat the held-out CBAM numbers again.
- §3.2 protocol (p34–39): keep; this is the "single standard" apparatus. But the uncertainty/CI
  paragraph is defined here AND re-defined in Ch4 intro AND in every figure caption.
- §3.3 overlap audit (p39–41): load-bearing (leakage control). Keep.
- §3.8 architecture (p49–68, ~19pp) is the bulk and the most cuttable. It contains **four explicit
  "superseded predecessor" subsections** (§3.8.4 Alert-Gate, §3.8.10 Patch Filter, plus fail-open
  rationale, plus the superseded sa32/control40 variants in §3.8.9) that are re-explained again in
  Ch4 design-evolution paragraphs. §3.8.2 (Deferred Suppression) re-states the retrained_v2 collapse
  and the conf-floor sweep that §3.8.6/§4.1.6/§4.3.1 also state.

**Ch4 Empirical (42pp) — primary target.** The evidence tables must stay, but the connective prose
is where the bloat is:
- The "Stack under test" paragraph (p69–71) is ~50 lines explaining thresholds/floors that are then
  re-explained in §4.1.2, §4.1.6, §4.3.3, App D, and the glossary.
- After each table a multi-bullet "readings" block restates the table in sentences (the five
  readings, three observations, two readings, three per-surface readings, three results, etc.).
- Design-evolution paragraphs (predecessor stack) appear in §4.2, §4.3.3, §4.3.4 — each re-tells the
  superseded-stack story already told in Ch3's superseded subsections.
- §4.3.3 Trust Classifier re-narrates the entire robust6→robust8→robust8-nr lineage that §3.8.9
  already built, including re-deriving the grayscale hole and the leakage argument.

**Ch5 Conclusion (6pp).** RQ answers re-quote every headline number from Ch4 *in full sentences*
(0.742→0.946, 0.948→0.991, 30.4%→1.4%, 0.11%, 29.4%→2.8%, speeds, etc.). A conclusion should
assert the answer and point to the table, not re-print the ablation. Production Stack section
re-lists every component with its rationale already given in §3.8 + App B.

**App A–E (16pp).** App A (datasets) + App D (provenance) are load-bearing — keep. App B (models
table) is a useful single reference — keep. App C (MRI sample report, verbatim, ~2pp) duplicates
Figure `fig:mri_report` (the verdict block already shown in §3.7) — the appendix repeats the same
table plus a longer one; can shrink. Glossary (App E) partly re-defines robust6/robust8/sa32/mlp_v5
with the same caveats already in-text 2–3×.

---

## TABLE 1 — Cross-chapter / cross-section REPETITION

> Convention: "keep where" = the one location that should retain the full treatment; everywhere
> else collapses to a one-clause cross-reference (e.g. "the retrained_v2 collapse of §1.2") or is
> deleted. "Lines" = rough source-line savings across the duplicate sites.

| # | item / claim | all locations (file:line) | keep where | cut / cross-ref where | est. lines saved |
|---|---|---|---|---|---|
| 1 | **retrained_v2 bird-vs-drone collapse** (94.4%→3.4% bird fire; R 0.961→0.306; "not separable at the detector") | intro:15,19; related_work:17,20,38; methodology:543,629,706; empirical:391,397; (numbers also conclusion implicitly) | §1.2 (problem statement, full) + Table `tab:rgb_comparison` (§4.3.1) as the data | related_work:17 & 38 → one clause; methodology:543 & 629 → cross-ref §1.2; the "not separable…downstream cascade is the response" sentence is stated ~5× verbatim — keep once | 18–24 |
| 2 | **30.4% confuser fire** (+ per-category 39/58/23%, and 94.4/74.6/66.2% Svanström) | intro:15; related_work:17; methodology:162,656; empirical (Table 4.4),170; conclusion:10; abstract main:156 | §1.2 (full per-category) + Table `tab:ablation_confusers` | related_work:17 → clause; methodology:162 keep (IR contrast) but drop RGB re-quote; conclusion:10 → "(§4.1.2)" | 10–14 |
| 3 | **Scoring-rule swing (dual vs trust-aware, "28 pp" / 27.7 / 2.8 pp)** | related_work:128,136; methodology:216 (full derivation); empirical:18,87 (RQ3); conclusion:18; threats:772 | §3.2.2 (full derivation, `sec:scoring_audit`) | related_work:128/136 → one sentence + ref; conclusion:18 & RQ3 prose → clause; the "Multi-modal numbers not comparable without disclosure" line appears 3× | 12–16 |
| 4 | **imgsz=1280 rationale** (Svanström native 640×512; baseline loses 28 pp 0.964→0.684; resolvable floor) | related_work:9; methodology:223,228(fig),588,595(fig),602(tab),638; empirical:248,301 | §3.8.6 + Table `tab:resolution` (the dedicated sweep) | related_work:9 → one sentence; methodology:223 (canonical configs) keep brief; the "+28 pp" figure is restated in fig caption AND table caption AND body 4× — keep once | 14–18 |
| 5 | **"Detectors are already good; problem is concentrated in confuser scenes"** | intro:8 (full), 26; related_work:55; methodology framing; empirical:79; conclusion:10; abstract:156,159 | §1.1–1.2 (full) | empirical:79 & conclusion:10 → clause; the "41 FP across 4,000 frames / 2.8% halluc" stat is re-quoted ~5× (intro:8, abstract:156, related_work:55, conclusion:10) — keep twice (intro+abstract) | 8–12 |
| 6 | **IR hallucinates far less than RGB** (0–1.7% on shared Svanström confuser seqs vs 65–74% RGB; 1.8% near-domain) | lit_ir:30; methodology:162,656; empirical:173,275(profile),649 | Table `tab:failure_profile` (§4.1.5) as data + one prose statement §3.8.8 | methodology:162 & 656 partly overlap; empirical:173 re-states; keep the failure-profile table once, cross-ref elsewhere | 8–10 |
| 7 | **Per-modality / trust-aware scoring explanation** ("each modality vs own GT, never unioned") | methodology:214,262; empirical:12,21(caption),79,87,1023-style; conclusion:18; threats:772 | §3.2.2 (full) | Repeated in nearly every Ch4 table caption + every readings block. Keep the rule in captions as a 4-word tag ("trust-aware; §3.2.2"); delete the re-explanations in body prose | 10–14 |
| 8 | **robust8-nr trade** (drops reject → recovers recall on clean/solo/degraded; costs confuser fire 0.0011→0.014 RGB, 0.024→0.028 IR, 0.098→0.236 video; mean composed 0.744→0.850) | empirical:85,137,140(full),492(full),conclusion:10,32; methodology:561,687 | §4.3.3 "The shipped router: dropping the reject class" (full, `sec:classifier_results`) | empirical:85,137,140 each re-derive it; §4.1.1 reading #4 + §4.1.1 "Why the shipped router drops the reject class" para (140) are ~the same content as §4.3.3:492 — **merge**; conclusion:32 → clause | 25–35 |
| 9 | **Composition order = recall/precision dial (filt→clf ships; 0.946 vs 0.931)** | methodology:523(caption),537; empirical:83,492(end); conclusion:14 | §4.1.1 reading #3 (the data is in Table 4.1) | methodology:523 caption + 537 para state it pre-emptively; conclusion:14 restates; keep once in Ch4, cross-ref from Ch3 | 8–10 |
| 10 | **mlp_v5 feature-reuse: 517-D, p3+p5 ROI, reuses detector features, 37–72× faster, per-frame** | intro:26,50; related_work:7,38,86; methodology:515,789,799; empirical:578+,667; conclusion:14; glossary:267 | §3.8.11 (`sec:distill_verifier`, full) | The "reuses already-computed ROI features so cheap enough per-frame" sentence appears ~7×. Speeds (37–72×, 404×) re-quoted in intro, abstract, methodology, empirical, conclusion, Table speed. Keep speeds in Table `tab:speed` + one prose mention | 12–16 |
| 11 | **MRI "audits its own paper trail" (corrected 2 figures / 1 claim; p5 not confidence is top feature)** | methodology:497(full),498(comment); empirical:691,692; conclusion:20; (+ App C context) | §3.7.2 / `fig:mri_report` (full, methodology) | empirical:691 is a near-verbatim restatement of methodology:497; conclusion:20 restates again. Keep once, cross-ref | 8–10 |
| 12 | **Grayscale finding headline** (zero-shot thermal on grayscale RGB within 2.7 pp; raw-RGB control collapses 0.187/0.295; ties on flock_of_seagulls clip 0.837/0.840) | intro:61; lit_ir:30; related_work:52; methodology:649; empirical:699,721,754,761; conclusion:25 | §4.5 (`sec:grayscale`, full) + Table `tab:gray_threeway` | intro:61 keep (it's a contribution) but trim; lit_ir:30 → clause; the "0.837 vs 0.840 tie on hardest clip" appears 4×; "raw-RGB control isolates single-channel as load-bearing" appears 3× (empirical:721,754,761) — keep once | 12–16 |
| 13 | **CBAM held-out IR filter result** (48→6 FP; recall 0.717/0.967→0.967; 94% removed; 90→22 / IR_confusers val+test) | methodology:656,678,679; empirical:628,632(tab),649; conclusion:10; App A.4:38; threats:791 | §4.3.4 "thermal-native IR filter" + Table `tab:ir_aligned` (full) | methodology:656 & 678 & 679 state the same numbers 3× within §3.8.8; App A.4 repeats; the "−3.7 pp on airplane-like ir_dset drones no threshold separates" clause appears ~5× | 16–22 |
| 14 | **Bootstrap CI definition** ("central 95% of 1,000 resamples; non-overlap = beyond sampling noise") | methodology:257; empirical:4 (chapter intro), fig captions 179, 1008-style; threats:778 | §3.2.3 (`tab:eval_protocol` uncertainty para, full) | Re-defined verbatim in empirical:4 and again in Fig `fig:pipeline_ablation` caption and Fig 4.1 caption. Keep once; captions say "95% bootstrap CI (§3.2.3)" | 6–8 |
| 15 | **RGB-test carve-out closed by v4 bird-split** (0.809→0.922 F1, recall 0.691→0.887; sub-32px) | methodology:795; empirical:216,248,301,607,612,616; conclusion:32; App B:162 | §4.3.4 "carve-out diagnosed and then closed" + Table `tab:per_size` | Stated in §4.1.3, §4.1.4, §4.1.6, §4.3.4 (twice), conclusion, App B. Keep the diagnosis+fix once (§4.3.4), cross-ref the per-size confirmation | 14–18 |
| 16 | **"Detectors hallucinate on different scenes → router picks least-active failure profile"** | lit_fusion:38; methodology:513,559; empirical:87,275 | §3.8.3 (Trust-Aware Fusion, full) | lit_fusion:38 → clause; methodology:513 (IR detector bullet) overlaps §3.8.3:559; empirical:87 restates | 6–8 |
| 17 | **selcom imgsz win (doubling imgsz doubles recall AND raises precision; "resolution not architecture")** | methodology:223,588,638; empirical:402(tab),418; (App A.8) | Table `tab:selcom` (§4.3.1) | methodology:638 + 588 + 223 each assert it; keep table + one sentence | 6–8 |
| 18 | **"Airplanes are every filter generation's weakest category"** | methodology framing; empirical:173,371,570(fig),649,653,791 | §4.7 "airplane gap" limitation (one place) | Stated ~6× (each filter section + temporal + limitations). Keep once as the limitation, cross-ref | 6–8 |

**Table 1 subtotal (rough): ~200–270 source lines** of pure duplication removable, which at this
thesis's density (~16 source lines/page in the prose-heavy chapters) is on the order of **8–12 pages**
before any prose tightening.

---

## TABLE 2 — VERBOSE / low-value passages

| # | passage (file:line + quote) | issue | recommended action | est. lines saved |
|---|---|---|---|---|
| V1 | empirical:11–12 "Stack under test." (the ~1-screen paragraph: detector floor vs filter threshold, "Two different threshold stages should not be conflated… A *lower* bar keeps more detections…") | Over-explains thresholds that are also in App D, glossary, §3.8.1, §4.1.6. Tutorial tone. | Cut to: stack list + "(threshold semantics in §3.2/App D)". Move the floor-vs-threshold tutorial nowhere — it's already in §3.8.2. | 18–24 |
| V2 | empirical:77–87 "Five readings, each attributable to one element." | Each "reading" re-narrates Table 4.1/4.2 in prose (TP/FP/FN restated as sentences). | Keep readings #1 (RQ1) and #5 (RQ3) as 2–3 lines each; fold #2/#3/#4 into the captions or one short para — the data is in the table. | 22–30 |
| V3 | empirical:139–141 "Why the shipped router drops the reject class." | Near-duplicate of §4.3.3:491–492 "The shipped router: dropping the reject class" (same numbers, same argument). | Delete one; cross-ref. Keep the §4.3.3 version (it's the classifier section); §4.1.1 keeps a 2-line pointer. | 14–20 |
| V4 | methodology:564–567 §3.8.4 Alert-Gate Cascade (superseded patch path) — whole subsection | Documents a superseded design "because part of the historical evidence was measured under it." | Compress to a 3-line note in §3.8.1; the historical numbers it supports live in §4.2 design-evolution para (also trimmable). | 8–10 |
| V5 | methodology:778–787 §3.8.10 Patch Filter (Superseded Predecessor) — whole subsection + Table not needed | Full architecture spec of a superseded component (45,917 patches, per-class P/R, 4 versions). | Compress to ~5 lines: "a 4-class MobileNetV3 patch CNN, superseded by mlp_v5 for cost+coverage (audit in §4.3.4)." Drop per-class training metrics. | 14–18 |
| V6 | empirical:545–576 §4.3.4 "The patch generation" + Table `tab:patch_audit` + Fig `fig:patch_catchbar` | The patch filter is superseded; its threshold sweep, per-bucket catch table, AND a bar-figure of the same table all stay. | Keep ONE artifact (the catch table OR the figure, not both) + 3-line verdict. The bimodal-veto detail is design-history. | 12–16 (+1 figure) |
| V7 | empirical:619–624 Fig `fig:failopen_expanded` + para — superseded fail-open gate, "retained only as the record of why a calibration patch was inferior" | Design-history figure for an approach explicitly abandoned. | Delete figure; keep one sentence in §4.3.4 ("a no-retrain fail-open gate was tried and rejected; the re-mine fixed coverage directly"). | 6–8 (+1 figure) |
| V8 | empirical:498–503 Fig `fig:robust8_operating` + "The grayscale hole, and why the reject class caused it." para | Figure is explicitly relabelled "design-history"; the τ knob it shows "no longer exists" in the shipped no-reject router. | Delete figure; compress the para to 2 lines (the hole existed under reject; no-reject removes it). | 8–10 (+1 figure) |
| V9 | empirical:505–538 §4.3.3 design-evolution block: "A dual-classifier extension (validated, not shipped)" + "Design evolution: OOD confuser zoo (predecessor config)" + Table `tab:classifiers` + Fig `fig:classifier_reversal` + "The surface reversal." | Three predecessor-classifier paragraphs + a table + a figure for classifiers (fnfn, control40, sa32) that are not shipped. The surface-reversal point is made twice (fig caption + para). | Keep the surface-reversal *lesson* (1 short para, motivates statistical selection) + the figure OR table (not both). Drop the dual-classifier paragraph to 2 lines. | 20–28 (+1 fig or tab) |
| V10 | empirical:360–372 "Design-evolution evidence (predecessor stack)" + Fig `fig:cascade_segment_fig` + per-category para | Predecessor-stack temporal numbers (+6.6/+16.5 pp) + a figure, then a per-category bird/airplane para. Production temporal table already above it. | Keep the one durable lesson ("the cascade gained F1 at segment grain; per-frame fell — unit-of-analysis lesson") as 3 lines + keep figure OR cut it. Drop per-category para to a clause. | 10–14 |
| V11 | conclusion:10,14 RQ1/RQ2 answers re-quoting full ablation (every TP/FP number, 2,019→337, 835→39, speeds) | A conclusion restating the evidence chapter number-for-number. | Rewrite RQ answers to assert the answer + cite the table: ~40% shorter. The numbers live in Ch4. | 14–20 |
| V12 | conclusion:30–32 Production Stack — re-lists every component with full rationale | App B already tabulates models+roles; §3.8 gives rationale. | Compress to the stack list + the two carve-outs; drop re-justifications. | 8–12 |
| V13 | related_work:126–136 §2.9 three caveat paragraphs ("Scoring rule." / "Dataset split." / "Task.") + "Two claims survive" | Caveats overlap each other and overlap §3.2.2 + §3.3 + threats §4.6. Author flagged this block "needs revisiting." | Compress three caveats to a 4-line bullet list; "Two claims survive" to 2 lines. | 12–16 |
| V14 | introduction:71–74 §1.6 Thesis Outline — prose paragraph re-listing every chapter's contents | A sentence-form ToC; the ToC already exists. | Cut to 3–4 lines (one sentence per chapter) or delete. | 6–8 |
| V15 | introduction:50,53,55,57 Contributions — 4 long paragraphs pre-printing Ch4 numbers + abstract content | Each enumerated contribution is a mini-results section duplicating the abstract. | Trim each to its claim + a forward-ref; remove the inline headline numbers (they're in Ch4). | 16–22 |
| V16 | methodology:497 + appendices:176–201 (App C) — MRI "audits its own paper trail" + full verbatim MRI report | The verdict block is shown as `fig:mri_report` in §3.7; App C repeats it plus a longer table; the "corrected two figures" story is told in §3.7 AND §4.4. | Keep App C as a short verbatim block (verdict + top features) — drop the duplicated signal table rows already in Fig 3.x. The self-audit anecdote: keep once. | 8–12 |
| V17 | methodology:391 §3.5.1 + 427–448 §3.5.3 — co-evolution problem + loop steps (6 numbered steps then a 6-item Disagree list then "A model FP is either…") | The disciplined-loop idea is stated in §3.5.1, the 6-step loop, the figure caption, AND the "every disagreement is actionable" closer — 3 framings of one loop. | Keep the figure + the 6 steps; cut the §3.5.1 restatement and the closing paragraph to 2 lines. | 8–10 |
| V18 | Throughout: hedge/throat-clearing openers — "It is worth noting", "The size of the difference is the reason this is protocol rather than a footnote", "Three observations, drawn together in…", "Two attributions emerge", "Two readings.", "Three per-surface readings." | Section-scaffolding phrases that announce structure instead of stating content. | Mechanical pass: delete the announcer, keep the content. ~30–40 such openers across Ch3–4. | 15–25 (aggregate) |
| V19 | empirical:355–358 temporal filter-threshold sweep para ("A filter-threshold sweep over the cached probabilities settles whether the veto cost is an operating-point artifact: it is not… smeared across [0.01,0.25)…") | Long methodological aside whose conclusion ("near-zero threshold or alert gate") is already the section's thesis. | Compress to 2 lines; full sweep is in the cited .md. | 6–8 |
| V20 | methodology:543 §3.8.2 conf-floor sweep detail (Anti-UAV flat 0.959–0.963; RGB test peaks 0.25; SelCom 0.591→0.692) | These exact numbers reappear in §4.1.6 (low-conf mode) with the full table. | Keep the principle (deferred suppression); move the sweep numbers to §4.1.6 only. | 6–8 |

**Table 2 subtotal (rough): ~250–330 source lines + ~5–6 figures/tables** removable, i.e. roughly
**10–15 pages** (figures/tables each consume 0.3–0.5pp).

---

## TABLE 3 — Sections to COMPRESS or MERGE

| # | section(s) | rationale | est. pages saved |
|---|---|---|---|
| C1 | **§3.8.4 Alert-Gate + §3.8.10 Patch Filter + §4.3.4 "patch generation" + §4.2 design-evolution + Fig patch_catchbar + Fig cascade_segment + Fig failopen_expanded** — the entire **superseded-component apparatus** | A superseded predecessor (patch CNN + alert gate + fail-open) is documented at full architecture depth in Ch3 AND re-evaluated at full depth in Ch4. One "Design history" note (½pp) + the single comparison table `tab:distill_verifier` carries everything the reader needs. | **3–4** |
| C2 | **§4.3.3 Trust Classifier vs §3.8.9 Statistical Feature Selection** | §3.8.9 builds robust6→robust8; §4.3.3 re-builds the same lineage (robust6→robust8→robust8-nr) + re-derives the grayscale hole + the leakage argument + sa32 leakage. Massive overlap. | **2–3** |
| C3 | **§4.1.1 "Why the shipped router drops the reject class" merged into §4.3.3 "dropping the reject class"** | Two full treatments of the identical decision with identical numbers. | **1–1.5** |
| C4 | **§2.9 Comparison to Prior Work** (two tables + 3 caveats + survivors) | Author flagged for revisiting. Caveats duplicate §3.2.2/§3.3/§4.6. Keep `tab:numerical_comparison` + one caveat bullet list. | **1.5–2** |
| C5 | **Ch1 Contributions + Ch5 RQ answers** — de-duplicate the headline numbers between them and the abstract | The same 6 headline numbers are printed in full in abstract, §1.4, and §5.1. Print full once (Ch4 tables), assert+cite elsewhere. | **1.5–2.5** |
| C6 | **§3.8.8 IR detector subsections** (Grayscale mode + Cross-Modal Filter) | The CBAM held-out numbers (48→6, recall 0.967, −3.7pp) are stated 3× inside §3.8.8 alone, then again in §4.3.4 + App A.4. State once in §3.8.8 (brief), full table in §4.3.4. | **1–1.5** |
| C7 | **§4.1 readings blocks** (five readings / three observations / two readings / three per-surface readings) | Collectively ~80 source lines restating the four ablation tables. Tables are self-contained with rich captions. | **1.5–2** |
| C8 | **§1.6 Outline + §3 chapter preamble (methodology:4) + Ch4 preamble (empirical:4)** | Three "here is what this chapter establishes/consumes" roadmaps. Keep the Ch3/Ch4 one-liners; cut §1.6 to a stub. | **0.5–1** |
| C9 | **App C MRI report + App E glossary entries for robust6/8/nr/sa32/mlp_v5** | App C duplicates `fig:mri_report`; glossary re-states the same caveats already in-text. Glossary can keep 1-line defs without re-arguing leakage etc. | **0.5–1** |

**Table 3 subtotal: ~13–19 pages** of compress/merge potential (overlaps partly with Tables 1–2 —
see reconciliation in the total below).

---

## DO NOT CUT (protect from the plan)

- **The evidence tables** — `tab:ablation_svanstrom`, `tab:ablation_antiuav`, `tab:ablation_dut`,
  `tab:ablation_confusers`, `tab:ablation_solo`, `tab:rq3`, `tab:per_size`, `tab:failure_profile`,
  `tab:ir_evolution`, `tab:rgb_comparison`, `tab:selcom`, `tab:resolution`, `tab:distill_verifier`,
  `tab:ir_aligned`, `tab:gray_threeway`, `tab:realvideo_master`, `tab:clean_split`,
  `tab:svanstrom_audit`, `tab:speed`, `tab:lowconf_selcom`, `tab:temporal_production`. These are the
  thesis's spine. (Trim *captions*, never the data.)
- **The RQ structure** (RQ1/RQ2/RQ3) and the explicit answers section §5.1 — keep the structure;
  only de-duplicate the re-quoted numbers.
- **All headline numbers at their canonical home**: Svanström 0.742→0.946 (Table 4.1), confuser
  30.4%→1.4%/0.11% (Table 4.4), Anti-UAV 0.973→0.984 (Table 4.2), grayscale 0.607/0.187/0.580
  (Table `tab:gray_threeway`), IR HITL 0.503→0.967 (Table 4.x), CBAM 48→6 (`tab:ir_aligned`).
- **The integrity / provenance apparatus**: §3.2 (evaluation protocol), §3.3 (overlap audit +
  `tab:clean_split`), §3.6 (reproducibility), App D (number provenance + `tab:provenance`), the
  `% [source: …]` comments. These answer the author's own repeated "how does the reader verify this"
  concern and must survive.
- **The Model MRI core** (§3.7 instrument + §4.4 findings) — but the self-audit anecdote and the
  verbatim App C can be stated once.
- **The leakage statistic** (§3.8.9, Eq. leakage, `tab:leakage`) — this is a methodological
  contribution; keep the derivation once.
- **The HITL V5-regression case study** (§4.3.2, `tab:ir_evolution`, the loop figure) — the
  "clearest negative result"; keep.
- **Datasets appendix A + the dataset tables** (provenance the author explicitly wants).
- **Ethics §1.5** — required.

---

## TOP-10 prioritized trim actions (biggest page savings first)

1. **Collapse the superseded-component apparatus** (patch CNN + alert-gate + fail-open) to a single
   "Design history" note + keep `tab:distill_verifier`. Removes §3.8.4 + §3.8.10 bulk, §4.3.4 patch
   block, Figs `patch_catchbar` + `failopen_expanded`, and trims §4.2 design-evolution. **≈3–4 pp.**
2. **Merge the two reject-class treatments** (§4.1.1 "Why the shipped router drops the reject class"
   + §4.3.3 "dropping the reject class") and de-duplicate the robust6→robust8→nr lineage between
   §3.8.9 and §4.3.3. **≈3–4 pp.**
3. **Cut the "readings" prose blocks** in §4.1 (five readings / three observations / two readings /
   three per-surface readings) to 1–2 lines each — the tables carry the data. **≈2–3 pp.**
4. **De-duplicate the headline numbers** across abstract / §1.4 Contributions / §5.1 RQ answers:
   print full once (Ch4), assert+cite elsewhere; trim §1.4 paragraphs and §5.1 RQ answers. **≈2–3 pp.**
5. **Compress §2.9 Comparison to Prior Work** (3 caveats → bullet list; "two claims survive" → 2
   lines) and drop the §2.1/§2.2/§2.4 re-derivations of the confuser/collapse story. **≈2 pp.**
6. **Consolidate the CBAM/IR-filter numbers** (stated 3× in §3.8.8 + §4.3.4 + App A.4) and the
   grayscale-finding restatements (3× in §4.5) to one canonical statement each. **≈1.5–2 pp.**
7. **Drop the design-history figures** `fig:robust8_operating`, `fig:failopen_expanded`,
   `fig:classifier_reversal` (or keep one of the classifier artifacts) and the dual-classifier
   paragraph. **≈1.5–2 pp** (3 figures + prose).
8. **Trim the "Stack under test" tutorial** (empirical:11–12) and the conf-floor/threshold semantics
   repeated across §3.8.2 / §4.1.6 / App D / glossary to one home each. **≈1.5 pp.**
9. **Mechanical hedge/announcer pass** across Ch3–4: delete "It is worth noting", "The size of the
   difference is the reason this is protocol rather than a footnote", "Two attributions emerge",
   "Three observations…", etc.; collapse the repeated bootstrap-CI and trust-aware definitions to
   cross-refs. **≈1.5–2.5 pp** (aggregate).
10. **Compress §1.6 Outline, §3.5.1 co-evolution restatement, App C verbatim report, and glossary
    re-arguments** to stubs. **≈1.5 pp.**

### Rough total achievable reduction

| Bucket | Pages |
|---|---|
| Top-10 actions (de-duplicated against each other) | ~20–26 |
| Residual fine-grained prose tightening (sentence level, not double-counted above) | ~3–6 |
| **Estimated total** | **~22–30 pages** (≈17–23% of the 132pp body) |

A **conservative, low-risk** target (repetition + dead design-history + readings-block prose only,
no aggressive rewriting) lands around **15–18 pages**. The **22–30pp** figure assumes the prose
tightening in Table 2 is also done. None of this touches an evidence table, a headline number at its
canonical home, the RQ structure, or the provenance apparatus.

**Biggest single levers (where to start):** (1) the superseded-component apparatus, (2) the
duplicated reject-class / classifier-lineage treatment, (3) the §4.1 readings-block prose.

---

## Notes for the plan author
- Chapters 3 and 4 hold 66% of the body and ~90% of the trim opportunity; Ch1/2/5 are smaller wins.
- Many cuts are *cross-references*, not deletions — the content survives at one canonical site, so
  the thesis stays easy to follow (and arguably easier, since the reader stops re-reading the same
  number). This directly serves the author's "easy to follow" constraint.
- The author's notes file (thesis notes.txt) independently flags several of these as repetition
  ("this was literally mentioned before" at §2.4 re bird/drone separability = item #1; "as i said
  before" at §2.4/§2.9 re scoring swing = item #3; "you're repeating yourself" at §2.4 re hardneg
  numbers = item #1/#2). Items #1, #3, #8, #13, #15 are the highest-confidence repetition cuts.
- A few items in the author's notes ask to *add* material (per-size stats, dataset visuals, speed
  table) — those are out of scope for a trim pass but note that the speed table (`tab:speed`) and
  per-size table (`tab:per_size`) already exist, so the corresponding inline re-quotes can be cut
  in favour of the tables.
