# Verify v2 — Chapter 2 (Related Work) reverification

**Date:** 2026-06-18
**Scope:** `docs/thesis_working_distilling_overleaf/chapters/related_work.tex` (Chapter 2) vs notes lines 251–447 (sessions 1–3).
**Mode:** READ-ONLY. No thesis/results edits. Numbers judged against `thesis_eval/results/{tier1,notes_round1}_results.json`, `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv`, `eval/results/svan_resolution_sweep.json`, `runs/negative_frame_fire.json`, `knowledge/{ledger,evals}.csv`, `references.bib`.
**Audit:** `py thesis_eval/_audit_headline_numbers.py` → **187/187 pass** (147 headline cells + 40 cited paths), 0 failures.

Headline verdict: the chapter has been **substantially rewritten since the notes were written**. Nearly every session-1/2/3 directive is FOLLOWED. No directive is REGRESSED. Two items are OPEN-by-design (cross-paper claims that need the citation agent + the Svanström paper, which is not currently in Downloads). The contested "94.4% bird fire rate" is honestly contextualized and the disproving low-fire evals are now reflected in the prose.

---

## TABLE 1 — Notes reverify

| Note (line) | Directive | Status | Thesis loc (file:line + quote) | Evidence |
|---|---|---|---|---|
| L256-260 | DETR/RT-DETR "not adopted" needs a *better* reason than calibration (downstream built after YOLO) | **FOLLOWED** | rw:7 "downstream stages were trained \emph{after} and \emph{against} the YOLO detectors … replacing the backbone now would mean re-deriving both" | reason now = build-order switching cost, exactly the note's point |
| L262-264 | "imgsz a frequently under-reported hyperparameter" — source? | **FOLLOWED** | rw:9 reframed to "A decisive hyperparameter … YOLO letterboxes … objects smaller than a few pixels … unresolvable" + `\cite{akyon2022sahi}` | claim now cited (SAHI) + geometric, not an un-sourced assertion |
| L266-270 | imgsz R=0.961 claim must say Svan is 640 / model trained at other size ("boxy") | **FOLLOWED** | rw:9 "corpus is natively $640\times512$ while the production RGB detector is trained at \texttt{imgsz=1280} … renders the median 29.8-px drone at half its trained-for scale" | `svan_resolution_sweep.json`: baseline@1280 R=0.9641, @640 R=0.6838 (−28.0pp) |
| L273-277 | Rozantsev "treated how?" — explain | **FOLLOWED** | rw:10 "regression-based motion stabilisation of local patches followed by classification on spatio-temporal image cubes, combining appearance with motion cues" | n/a (prose detail) |
| L285 | "dark silhouettes" — doubtful | **FOLLOWED** | rw:17 now "small, low-texture silhouettes against sky" (word "dark" gone, 0 occurrences) | grep: no "dark" in rw |
| L288-290 | "94.4% bird-frame fire" blown out of proportion; we have evals that disprove it — mention them | **FOLLOWED** | rw:17 frames it as *bird-only Svanström @1280* + pairs with corpus spread "fires on 23--58% of frames per category" on merged OOD @640; rw:25 "the trust classifier and per-frame confuser filter catch what the detector cannot" | 94.4% real (`svanstrom_1280_by_category.csv` BIRD det_rate 94.4%, 807 FP/589 fr); the tempering evals exist & are now reflected: merged airplane 23.4% / bird 39.0% @640 (notes_round1 CAT), and near-domain `negative_frame_fire.json` rgb 0.93% / ir 1.64% |
| L294-299 | C-UAS surveys + Schumann + Drone-vs-Bird "hard-neg mining necessary but not sufficient" — verify, quote verbatim | **FOLLOWED (claims softened; cite-verify deferred)** | rw:20 cites `shi2018counteruas,taha2019drone,samaras2019deep` + `taha2019drone` verbatim "hardly comparable"; `schumann2017deep` two-stage transfer; `coluccia2021dronevsbird` "winning entries relied on explicit negative training images … bird confusion remained the challenge's central difficulty" | all 4 keys in `references.bib`; claims now defensible paraphrases, not the original strong "necessary but not sufficient" — see TABLE 3 NEEDS-CITE for the citation agent |
| L302-308 | Anti-UAV "saturated/in-distribution" — verify P/R/F1, do 640 too, mention test split (avoid leakage suspicion) | **FOLLOWED** | rw:22 "$P=0.9922,R=0.9950,F1=0.9936$ at \texttt{imgsz=1280} ($3{,}178$~TP, $25$~FP, $16$~FN) … production detector reaches $F1=0.986$ at the Tier-1 \texttt{imgsz=640} … IR-only $F1=0.9654$ … no train/test split is claimed; in-distribution sanity floor" | master.csv `rgb_only`: 3178/25/16 → 0.9922/0.995/0.9936 ✓; tier1 ft4/rgb@640 0.9853 (→0.986) ✓; IR 0.9654 = `ir_v3b_antiuav640_may10` full-corpus ✓; 59,413 anti frames disclosed |
| L320 | ESC full name | **FOLLOWED** | rw:30 "electronic speed controllers (ESCs)" | n/a |
| L323-327 | IR F1=0.636 grayscale "ties RGB on hardest bird clip" — name dataset+clip; best-case framing is misleading | **FOLLOWED** | rw:30 "YouTube drone-clip diagnostic surface (9 clips, 1{,}359 frames …) aggregate $F1=0.636$ … single best clip, \texttt{flock\_of\_seagulls\_attack\_drone\_beach}, it ties the RGB baseline ($0.837$ vs $0.840$); **the aggregate, not the best clip, is the representative number**" | `ledger=ir-grayscale-fallback` (0.636 aggregate; 0.837/0.840 tie); best-case caveat explicitly defused |
| L330-333 | REMOVE Roboflow OOD IR brittleness (R=0.519 ir_mixed_cbam / R=0.264 ir_drone_night) — unverified quality | **FOLLOWED (removed)** | rw: 0 occurrences of 0.519 / 0.264 / ir_mixed_cbam / ir_drone_night anywhere in chapter | grep confirms removed; Roboflow now only a dataset-provenance mention (appendices/methodology), not a brittleness number |
| L337-341 | "two detectors hallucinate on different scenes" — do we have a profile? "picks modality whose failure profile is least active" — how, with detection features only? | **FOLLOWED** | rw:38 reframed to a *category* complementarity "RGB residual fire is bird- and helicopter-driven while the thermal detector's is airplane-dominated … makes its four-way trust decision from eight detection-evidence features only" | EVIDENCED: RGB bird 94.4%/heli 66.2% (`svanstrom_1280_by_category.csv`); IR airplane 35.2% / bird 12.2% / heli 0% (notes_round1 `ir_confusers` CAT). "8 detection features only" matches production robust8-nr |
| L343-352 | the "66.2%→41.9% / 74.6%→64.7%" + retrained_v2 birds-3.4%/R=0.306 block is RGB-only + repeated | **FOLLOWED** | rw:38 condensed to "helicopters and partially airplanes respond, birds essentially do not, and the stance that does suppress birds collapses small-drone recall" + "mining experiments … are RGB-side by design"; the verbatim 66.2→41.9 numbers no longer duplicated in Ch2 (live in §sec:problem) | dedup confirmed; numbers themselves still backed by the CSV |
| L354-357 / L390-392 | "swings F1 by 28pp … not comparable across studies" stated as obvious — just explain, don't assert | **FOLLOWED** | rw:44 defers: "how detections … are aggregated into one F1, is defined and justified as evaluation protocol in Section~\ref{sec:scoring_audit}; every multi-modal number … declares its rule" — the flat "28pp" assertion is GONE from Ch2 | methodology §3.2.2 carries it honestly: current swing **2.8pp** (0.921 vs 0.949), the 27.7pp is a *historical May-10* @640 measurement → "anywhere from 3 to 28 points" |
| L361-367 | §2.6 cross-modal "no published study matches this setup" — sure? hedge honestly | **FOLLOWED (hedged)** | rw:52 "we are **not aware of** published work matching the setup here … transfer emerges from shared low-level statistics" | softened from notes' original categorical "None match"; OPEN-by-design for the citation agent to sanity-check |
| L369-376 | cascade "high-recall stage" framing unfair — our detectors are strong (high P+R, low FP), not just recall | **FOLLOWED** | rw:55 "in those designs the first stage is deliberately recall-tuned and weak on precision, **whereas the detectors here are strong on both axes standalone** (Anti-UAV $P=0.989$/$R=0.982$; 2.8\% test-corpus hallucination) and the cascade exists for a \emph{concentrated} residual failure mode"; rw:91 "keeps the detectors strong on both precision and recall in their own right" | matches production-stack ground truth exactly; P/R verified (tier1 ft4 0.9889/0.9817) |
| L378 | "XGBoost on per-frame scene features" — wrong; robust6/8 do NOT use frame features (too slow); that was sa32 | **FOLLOWED** | rw:38 "production router makes its four-way trust decision from **eight detection-evidence features only** … a **superseded variant that additionally consumed per-frame scene statistics is retained as a comparison**"; rw:55 "an XGBoost router on **detection-evidence features**" | `ledger=robust6-speed-feature-efficiency` (scene stats = the slow sa32 38ms pass); production = robust8-nr 8 feats |
| L386 | "a verifier is trained" → a *filter* is trained | **FOLLOWED** | rw:65 "a **filter** is trained only when the probe says the separation exists, and the probe's embeddings become the filter's training data" | terminology aligned; see TABLE 3 for the mild MRI↔filter wording tension |
| L388-399 | §2.9 "scoring audit swings 28pp" relevance; detector ≠ recall stage; filter can be per-frame OR alert-gate | **FOLLOWED** | rw:55 "(ii) the confuser filter **can be applied per frame or at the alert-emit boundary** … the deployed GUI uses the alert gate"; rw:91 "(trust classifier $+$ confuser filter, run per frame or at the alert gate)" | matches locked composition (filt→clf, per-frame shipped; alert-gate in GUI) |
| L401 | Table 2.2: does Svanström share its scoring? + include IR in confuser rows | **FOLLOWED** | rw:103-124 Svanström row now states scoring "IoU@0.5, det.\ thr.\ 0.5, YOLOv2 @416" + caveat rw:128 "MATLAB's \texttt{bboxPrecisionRecall} … IoU $0.5$ … det threshold $0.5$"; IR present in **three** rows: Anti-UAV IR (rw:112-113), Svanström thermal (rw:106-107), **Thermal confuser corpus (no GT, IR)** robust6 29.4%→1.9% (rw:118-119) | `tier1 ir_confusers`: bare 0.2943 → clf→filt[robust6] 0.0192 ✓ |
| L406-435 (S2) | scoring/split caveats — check the actual Svanström paper + GitHub repo; figure out leakage/test-split | **OPEN-by-design** | rw:128-130 now cites paper specifics (eval set 120 IR + 120 visible clips, 5/class/distance-bin, list not published → cannot adopt; own MATLAB→YOLO conversion caveat) + GitHub repo source comment rw:134 | Svanström PDF **not in `C:\Users\User\Downloads`** at verify time → cannot confirm IoU disclosure firsthand; the citation agent + a paper re-check should close this. Claims are concrete and checkable |
| L436-447 (S3) | "two qualitative claims survive" paragraph — "needs revisiting, not revised. skipped." | **OPEN-by-design (user-parked)** | rw:136 "this thesis's per-modality numbers \emph{exceed} … part of that gap is attributable to detector generation … inference resolution … matcher leniency … At minimum … competitive baselines rather than weak strawmen … suppression $30.4\% \to 4.9\% \to 1.4\%$ … cannot be directly benchmarked" | the user explicitly parked this para ("remind me"); it now carries the honest generation/resolution/leniency qualifier. Flag = revisit per the note, not a defect |

---

## TABLE 2 — Numbers

| Number + context | Thesis loc | Source | BACKED? |
|---|---|---|---|
| Bird-only Svanström @1280 fire **94.4%** | rw:17 | `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` baseline BIRD det_rate 94.4% (807 FP / 589 fr) | ✓ |
| Merged OOD confuser fire **23–58% per category** @640 | rw:17 | `notes_round1_results.json` rgb_confuser CAT (airplane 23.4%, bird 39.0%); upper end = heli on gray_confuser 71.8% / svan heli 66.2% | ✓ |
| retrained_v2 collapses Svan drone **R=0.306** | rw:17 | `svanstrom_1280_by_category.csv` retrained_v2 DRONE det_rate 30.8%, R=0.306; `ledger=retrainedv2-recall-collapse` | ✓ |
| baseline loses **28 pp** recall @640 (0.964→0.684) | rw:9 (+ method 0.964→0.684) | `svan_resolution_sweep.json` baseline@1280 R=0.9641 / @640 R=0.6838 = −28.0pp | ✓ |
| median drone **29.8 px** Svan | rw:9 | `notes_round1_results.json` (svanstrom SZ median 29.8px) per source comment; method:228 quotes "median 28 px" | ⚠ (rw:9 says 29.8, method fig says 28 — both ~same, box vs √area rounding; cosmetic mismatch worth one number harmonised) |
| Anti-UAV @1280 **0.9922 / 0.9950 / 0.9936**, 3178 TP / 25 FP / 16 FN | rw:22 | `_ablation/2026-05-18.../master.csv` rgb_only row exact | ✓ |
| Anti-UAV production **F1=0.986** @640 | rw:22 | tier1 ft4/rgb@640 F1=0.9853 (CI [0.982,0.9882]); rounds to 0.985, thesis 0.986 | ⚠ (0.9853→0.986 is a 0.001 up-round; within CI; suggest 0.985) |
| Anti-UAV IR-only **F1=0.9654** | rw:22 | `evals.csv` `ir_v3b_antiuav640_may10` full-corpus F1=0.9654 (TP15910/FP213/FN926) | ✓ (full-corpus eval, not the strided 0.9647 — different cache, both real) |
| Anti-UAV standalone **P=0.989 / R=0.982**; **2.8%** test-corpus halluc | rw:55 (+ intro:8) | tier1 ft4/rgb 0.9889/0.9817 ✓; 2.8% = `eval=v5_rgbds bare halluc 0.028` (rgb_dataset composite) | ✓ (but see TABLE 3: "test-corpus" unattributed *in Ch2*) |
| IR grayscale aggregate **F1=0.636**; clip tie **0.837 vs 0.840** | rw:30 | `ledger=ir-grayscale-fallback`; eval `vid_drone_ir_gray` | ✓ |
| Thermal confuser **29.4% → 1.9%** (robust6) | rw:119 + src comment rw:123 | tier1 ir_confusers bare 0.2943 → clf→filt[robust6] 0.0192 | ✓ |
| RGB confuser suppression **30.4% → 4.9% → 1.4%** | rw:136 + src comment rw:137 | tier1 rgb_confuser bare 0.3035; clf[robust8] 0.0490; **filt_mlp 0.0144** (=shipped filt→clf[robust8_nr_drop] 0.0144) | ✓ (the "→1.4%" = per-frame filter / shipped no-reject composition; clf→filt[robust8] alone is 0.0011 — chain is honestly the router-then-filter staging) |
| Confuser-filter panel **1.4% fire** (RGB), **29.4%→1.9%** (IR robust6) | rw:116, rw:119 | as above | ✓ |
| MRI z-score **beats CORAL** AUROC **0.919 vs 0.707** grayscale→thermal | rw:65 | `ledger=gray-thermal-alignable` (per-modality z 0.919; CORAL 0.707; chance 0.500) | ✓ |
| Svanström published **0.785 visible / 0.760 IR**; this thesis 0.950 / 0.961 | rw:103-107, rw:136 | src comment rw:124 `svanstrom2021real (arXiv 2007.07396) visible avg 0.7849, IR 0.7601` | ⚠ (number self-consistent; the *paper value itself* needs the citation agent / PDF — not in Downloads now) |

---

## TABLE 3 — Claims

| Claim | Thesis loc | Status | Note |
|---|---|---|---|
| YOLO/Ultralytics is the detector | rw:7 | CITED `redmon2016yolo,ultralytics2024` | ✓ |
| Faster/Cascade R-CNN higher-acc, higher-latency | rw:7 | CITED `ren2015fasterrcnn,cai2018cascade` | ✓ |
| DETR/RT-DETR plausible drop-in, deferred for build-order cost | rw:7 | CITED `carion2020detr,zhao2024rtdetr` + EVIDENCED (build order) | ✓ honest reason |
| small-object lit resorts to up-scaled/sliced inference | rw:9 | CITED `akyon2022sahi` | ✓ |
| Rozantsev: motion-stabilised patches + spatio-temporal cubes | rw:10 | CITED `rozantsev2017detecting` | NEEDS-CITE-VERIFY (method detail — confirm "regression-based motion stabilisation" + "spatio-temporal image cubes" are the paper's actual method; plausible) |
| C-UAS surveys: visual attractive, bird/range limits | rw:20 | CITED `shi2018counteruas,taha2019drone,samaras2019deep` | NEEDS-CITE-VERIFY (survey consensus claim) |
| Taha: results "hardly comparable" across papers | rw:20 | CITED `taha2019drone` (verbatim quote) | NEEDS-CITE-VERIFY (quote attribution — verify "hardly comparable" is verbatim in taha2019) |
| Schumann: two-stage proposals+CNN, transfers across domain gap | rw:20 | CITED `schumann2017deep` | NEEDS-CITE-VERIFY (the notes' original "necessary but not sufficient" is *gone*; current claim is the safer "transfers across a substantial domain gap" — confirm) |
| Drone-vs-Bird: winners used explicit negatives + synthetic; bird confusion central | rw:20 | CITED `coluccia2021dronevsbird` | NEEDS-CITE-VERIFY (softened from "hard-neg necessary-but-not-sufficient"; confirm "winning entries relied on explicit negative training images and synthetic data") |
| Svanström / Anti-UAV anchor benchmarks | rw:22 | CITED `svanstrom2021real,svanstrom2022dronedataset,jiang2021antiuav,zhao2023antiuav` + EVIDENCED | ✓ |
| IR hot-spots (motors/ESC/battery) vs uniform-temp birds = discriminative cue | rw:30 | UNSUPPORTED (domain assertion, no cite) | NEEDS-CITE (physical-cue claim has no reference; the *downstream* "IR hallucinates less on birds" IS evidenced by IR bird-fire 12.2% vs RGB 94.4%, but the thermal-physics rationale is uncited) |
| IR ≈ grayscale-RGB low-level stats → cross-modal transfer | rw:30 | EVIDENCED `ledger=gray-thermal-alignable, ir-grayscale-fallback` (§sec:grayscale) | ✓ framed as emergent DETECTOR finding (matches locked decision #1: grayscale = finding, not filter) |
| decision-level fusion w/ XGBoost trust classifier | rw:38 | CITED `ramachandram2017fusion,chen2016xgboost,friedman2001greedy` | ✓ |
| intermediate fusion needs paired training data (multispectral pedestrian) | rw:38 | CITED `wagner2016multispectral` | ✓ |
| RGB bird+heli vs IR airplane complementarity | rw:38 | EVIDENCED (CSV + ir_confusers CAT) | ✓ (source comment points to §sec:failure_profile; per-category numbers actually live in the CAT blocks — fine) |
| OHEM/Focal/confuser-aug is class-asymmetric | rw:38 | CITED `shrivastava2016ohem,lin2017focal` + EVIDENCED | ✓ |
| IoP@0.5 for Svanström (loose GT boxes) | rw:44, rw:128 | EVIDENCED (memory rule: Svanström IoP; loose boxes) | ✓ |
| modality hallucination = closest published setting | rw:52 | CITED `hoffman2016modalityhallucination` | ✓ |
| RGB↔thermal translation / domain-adversarial = adjacent | rw:52 | CITED `berg2018rgb2thermal,kniaz2018thermalgan,ganin2016domain` | ✓ |
| "not aware of published work matching the setup" | rw:52 | EVIDENCED (hedged novelty claim) | ✓ honestly hedged (OPEN-by-design) |
| cascaded-rejection lineage (Viola-Jones, Cascade R-CNN) | rw:55 | CITED `viola2001rapid,cai2018cascade` + EVIDENCED differentiation | ✓ detectors-not-weak framing correct |
| detectors strong standalone (P=0.989/R=0.982; 2.8% halluc) | rw:55 | EVIDENCED (tier1 + v5_rgbds) | ⚠ "2.8% test-corpus" is **unattributed inside Ch2** — an examiner reading Ch2 alone won't know "test-corpus" = composite RGB test split (defined only in intro:8/10). Violates "every number names its dataset." Soft fix: add "(composite RGB test split)" inline |
| active-learning / data-centric lineage | rw:60 | CITED `settles2009active,brust2019active,northcutt2021confident,sambasivan2021data` | ✓ |
| IR F1 0.503→0.967 driven by corpus ops | rw:60 | EVIDENCED `ledger=ir-version-progression` (0.430→0.967; thesis says 0.503→0.967) | ⚠ ledger start = 0.430 (V2); 0.503 may be a different epoch — verify the 0.503 start value (cosmetic, both show the same trajectory) |
| MRI = linear-probe / Mahalanobis lineage; "filter trained only when probe says separation exists" | rw:65 | CITED `alain2016understanding,lee2018simple` + EVIDENCED | ✓ but mild tension w/ locked decision #2 ("filters NOT trained FROM MRI; MRI only diagnoses"): rw:65 "the probe's embeddings become the filter's training data" reads as MRI-supplies-training-data. Defensible (feature reuse ≠ MRI-trains-filter) but worth one clause to keep the "MRI diagnoses, does not train" line crisp |
| distillation / transferable-features / CORAL / CKA framing | rw:65 | CITED `hinton2015distilling,yosinski2014transferable,sun2016return,kornblith2019similarity` | ✓ |
| no system reports separated OOD confuser fire rate | rw:136 | EVIDENCED (architectural-novelty claim) | ✓ OPEN-by-design |

**Citations:** all 39 keys used in `related_work.tex` are defined in `references.bib` (44 keys total). No undefined `\cite`. No broken refs.

---

## TOP FIXES (ranked)

1. **Attribute "2.8% test-corpus hallucination" inline in Ch2 (rw:55).** It is the only Ch2 number that does not name its surface where it appears (defined only in the introduction as the composite RGB test split). One parenthetical — "(2.8% on its composite RGB test split)" — satisfies the "every number names its dataset" rule for a reader who opens Ch2 cold. *(highest: direct rule violation, trivial fix)*

2. **Cite or soften the thermal-physics cue (rw:30).** "drone motors, ESCs, and batteries produce localised hot spots vs uniform-temperature birds" is an uncited domain assertion. The *consequence* (IR fires less on birds: 12.2% vs RGB 94.4%) is evidenced, but the physical rationale wants a reference (e.g. an IR drone-detection / thermal-signature source) or a hedge ("are expected to produce").

3. **Harmonise the Svan median-drone px (rw:9 "29.8" vs method fig "28").** Pick one (box vs √area), state which, so the two figures agree. Cosmetic but examiner-visible.

4. **Round Anti-UAV @640 to 0.985 (rw:22 says 0.986).** tier1 ft4/rgb = 0.9853 → 0.985, not 0.986. Within CI but the headline-audit value is 0.985.

5. **Forward to the citation agent (OPEN-by-design):** verify the four C-UAS/confuser claims (taha "hardly comparable" verbatim; schumann transfer; coluccia winners-used-negatives; rozantsev method) against the actual papers, and re-confirm the Svanström 0.7849/0.7601 baseline values — **the Svanström PDF was not in `C:\Users\User\Downloads` at verify time**, so the IoU/scoring-disclosure claims (rw:128) could not be checked firsthand.

6. **Tighten the MRI↔filter sentence (rw:65)** so it does not read as "MRI trains the filter" (locked decision: MRI diagnoses only; filters reuse the embeddings as data). A one-clause clarification keeps the prescriptive-but-not-training framing clean.

7. **Verify the IR F1 trajectory start value (rw:60 "0.503" vs ledger "0.430").** Same story either way; confirm which epoch is the intended floor.

---

### Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_verify_v2_related.md` (this file)
