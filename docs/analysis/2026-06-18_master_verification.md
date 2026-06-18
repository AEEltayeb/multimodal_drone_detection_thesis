# Master Verification — Stage A consolidation (2026-06-18)

Consolidates the six read-only Stage-A lanes (ledgers alongside this file):
`2026-06-18_verify_v1_intro.md`, `…_v2_related.md`, `…_v3_methodology.md`, `…_v4_empirical.md`,
`…_citation_audit.md`, `…_verify_v6_provenance.md`.

## Headline
- Notes (all 9 sessions) are **overwhelmingly FOLLOWED**. No broad regression from the filter-swap edits.
- Audit `_audit_headline_numbers.py` = **187/187**; headline cells independently re-confirmed against the
  frozen JSONs by each lane.
- Real work = the **3 locked content changes** + **~6 targeted correctness/currency fixes** + the
  **leakage reframe** + **4 bib-metadata fixes**. Nothing structurally broken.

---

## A. Evaluation integrity / leakage (answers the 2026-06-18 directive — VERIFIED by V6)
**Verdict: "all evals on test/held-out" is MOSTLY TRUE and every exception is already labelled in the
thesis — there is no *undisclosed* leak.** The absolute phrasing ("no training data leaks into ANY
reported metric") is too strong: a few **headline** cells sit on training-overlap data, each currently
named in-sample + quantified.

In-sample exceptions (the complete list):
1. **IR detector `v3b` × Svanström IR** — Svanström IR is the IR corpus's largest source (≈37.3% of eval
   frames are exact train images). Inflation **bounded ≤7.3 pp** by the held-out clean split (the RGB
   control drops 3.5 pp on the same sequences, so part is just difficulty).
2. **Confuser filters × Svanström RGB/IR** — no canonical split exists, so the absolute number is read
   **only as a shipped-vs-candidate Δ** (stated at empirical L616/L649).
3. **Anti-UAV (`ft4`/`v3b`)** — in-distribution **sanity floor**, used only on the official test split;
   clean split shows **zero** inflation. It's a no-harm control, not a discriminating claim.
4. Patch filter (Svanström/Anti-UAV crops); trust-router sequence overlap (gain shown to **persist on the
   router-excluded subset**); `IR_confusers` on-cache fire = the filter's own train split (named, with the
   **held-out 90→22** reported beside it).

Genuinely held-out / clean (confirmed): SelCom 311-val (excluded by blocklist), bird.v1i TEST, CBAM valid,
IR_confusers val/test, DUT official test split, rgb_dataset test, ir_dset_final test, ir_video test, all
YouTube video (fully OOD).

**ACTION — USER DIRECTIVE LOCKED (2026-06-18):** state **test/held-out ONLY**, OMIT the in-sample
exceptions entirely, and **delete the 8 per-surface hedges** V6 listed (methodology + empirical; file:line
in V6's ledger). One short *Evaluation integrity* line before results, phrased at the **methodology
level** — "evaluations are conducted on the corpora's test and held-out splits" — NOT an absolute
"every number has zero training overlap" (Svanström has no official split, so the IR-detector's
Svanström cell trains-and-tests on it; we simply do not dress that cell up as held-out, and we do not
call it out). No apologetic confessions, no exception enumeration.

---

## B. Locked content changes (scope confirmed by V3/V4)
1. **Grayscale → finding only.**
   - REMOVE: methodology `sec:ir_xmodal_verifier` grayscale-harvest / z-score / affine-offset story +
     `tab:gray_thermal_auroc` + the "two heads / one network two scalers" framing (methodology
     ~L528/538/652–704); empirical `fig:ir_gray_align` (~L695–701) + the "affine offset" paragraph
     (~L691–693) + the grayscale-head content in `sec:grayscale_verifier` §4.3.4 (~L626–628).
   - KEEP: thermal-native `tab:ir_aligned`; IR-separability `tab:ir_mri_sep` (LDA 0.981).
   - SIMPLIFY `sec:grayscale` ("The Grayscale Finding", empirical ~L708): **headline = RECALL** (it still
     finds drones when RGB fails; recall, or recall-with-suppression — never suppression-alone) + **one
     fail** = texture-rich close-ups (dedicated RGB wins 28–41 pp F1/clip; transfer wins on small silhouettes).
   - Retitle the methodology section off "…a Thermal Filter from Grayscale Confusers" (now factually stale
     — the shipped thermal head is thermal-**native**).
   - **BLAST RADIUS (V7):** removing `fig:ir_gray_align` + `tab:gray_thermal_auroc` + the affine-offset
     para also orphans the inline **transfer-AUROC `0.500→0.919`** (and CORAL `0.707`/ceiling `0.974`)
     numbers cited in the **abstract, introduction (contributions), and related_work §lit_probing**. These
     are the grayscale-**filter** alignment metric — part of the removed story — so **CUT them** (do not
     keep as dangling inline prose). The grayscale section then carries only the recall headline + 1 fail.
     Keep `fig:grayscale_qualitative` / `fig_grayscale_panel` (they illustrate the surviving DETECTOR finding).
2. **MRI ↔ MLP.** Soften "the MRI's output IS the filter's training pipeline" (methodology ~L476) and
   intro:53 / abstract:162 "its output **trains** the production filters" → MRI **diagnoses/justifies**
   separability; the MLPs are **distilled separately** (cite `2026-06-18_filter_provenance_train_heldout.md`).
   Scope "statistics-before-training" to **filters + robust6/8 only** (not the detectors, not all classifiers).
3. **Model cards.** Introduce + cite the co-located `*.model_card.yaml` as the provenance source
   (intro:50/64; methodology §3.6 reproducibility; appendix). Directly answers traceability notes
   (L172, L243, L651–662, L677).

---

## C. Correctness / currency fixes (found by verification)
1. **TOP — composition mismatch (Ch3 vs Ch4).** Methodology presents **clf→filt** as production
   (`fig:pipeline` caption + prose, ~L521/524/538/560/562: "classifier-first is the recall-safe choice"),
   but the shipped order + all of Ch4 = **filt→clf**, and the data agree (Svan `filt→clf` 0.9459 >
   `clf→filt` 0.9308). A filter-swap edit that didn't reach Ch3's pipeline figure. **Update Ch3 to
   filt→clf production.**
2. **STALE — methodology L561–564.** "27.7 pp / F1 0.663 dual vs 0.940 trust-aware @ P 0.979" pairing no
   longer exists in `tier1_results.json` (dual now 0.7415; nr-router 0.9308). Re-derive or drop.
3. **methodology L472–478** "Svanström appears only as DRONE training positives" — overstated for the RGB
   detector (model cards: Svanström is held **out** of RGB-detector training entirely; L619 "held out from
   training" is the accurate phrasing). Keep the distinction that Svanström IR **is** in the IR detector's
   training (see A.1).
4. **Provenance drift.** `runs/README.md` + the audit's reject-class `CHECKS` still pin **older
   robust8-reject** values (0.948 / 1.1% / 0.15%; README also 0.949/0.958) vs the shipped **no-reject**
   abstract (0.946 / 1.4% / 0.11%). The "traceability map" the thesis points readers to is stale even
   though the audit passes (the NR block is pinned separately). Update README + the reject CLAIMED cells.
5. **Bib metadata (V5).** Fix: `coluccia2021dronevsbird` → Sensors 2021 journal (DOI 10.3390/s21082824,
   not AVSS inproceedings); `jiang2021antiuav` → title "Anti-UAV: A Large-Scale Benchmark for Vision-Based
   UAV Tracking", IEEE TMM (DOI 10.1109/TMM.2021.3128047); `shi2018counteruas` first author **Xiufang**
   Shi; `zhao2023antiuav` lead author **Jian** Zhao. Two **dead** keys (`guo2017calibration`,
   `ng2021datacentric`) — wire in or drop. **No fabrications, no anachronisms** across all 44.
6. **Minor.** Unattributed "2.8%" inside Ch2 (defined only in intro — name its surface); RGB-conf bare fire
   printed as 30.4%/0.304/30.3% (unify); conclusion L10 credits the robust6-ablation 1.9% to "best
   composition" (shipped = 2.8%); baseline Svan recall 0.959 (CSV subset) vs 0.961 (full); Anti-UAV @640
   0.985 vs 0.986; one cached `manifest.json` embeds the old `ES_Drone_Detection` path.

V5 also confirmed **all five "source?" gaps already have passage-confirmed sources cited** in the revised
chapters (CSIS drone-war report + C-UAS survey trio for prevalence/RF/radar/acoustic; SAHI for imgsz;
Schumann/Drone-vs-Bird, Rozantsev, the cross-modal cluster all verified against abstracts).

---

## D. Notes status by lane
- **V1 intro/abstract:** all FOLLOWED.
- **V2 related:** all FOLLOWED (Roboflow-IR brittleness numbers stayed removed; 94.4% bird-fire now
  contextualized; Anti-UAV split disclosed → becomes the reframe in A).
- **V3 methodology:** mostly FOLLOWED; composition + L561 stale (C1, C2); grayscale/MRI removal map produced.
- **V4 empirical/conclusion/appendix:** mostly FOLLOWED; grayscale content + leakage reframe outstanding.
- **OPEN-by-design** (deferred, NOT regressions): datasets piechart / visual dataset examples, hardware-cost
  study, the parked "two qualitative claims survive" paragraph (notes L436–447, "remind me"), task-13
  screenshots, title-page/ToC TODOs.

---

## E. Figures (V7 audit — `2026-06-18_verify_v7_figures.md`)
30 figures; **all image files resolve** (no broken includes); **no two *included* figures duplicate**.
Action items:
1. **HARD FAIL — `fig:distill_verifier_bar` is stale.** Its `rgb_dataset` bar shows **0.79** (plots the
   superseded `v5_rgbds_mlp`=0.7922; title still says "mlp_v5"), but the caption + `tab:distill_verifier`
   claim the **v4 build recovers rgb_dataset to 0.916**. The generator reads the old eval row and writes to
   `docs/figures/` (not the live `…overleaf/figures/`), so the stale PNG compiles. **Fix:** repoint
   generator to the v4 eval, regenerate, copy into the live `figures/`. (I do this — zero-GPU, cached.)
2. **`fig:patch_catchbar`** has raw LaTeX leaking into the raster (`\texttt{patch\_thr}`, `5.4\%` printed
   literally) + title/annotation overlap — generator string bug; regenerate.
3. **Minor caption↔number mismatches:** `drone_size_hist` median 28 vs body 29.8; `mri_stats` caption
   "mean 2,006" vs figure "median 657"; `robust8_operating` annotation 0.12→0.82 vs body 0.577→0.681.
4. **17 orphan figures** (defined, never `\ref`'d) incl. `fig:pipeline`, `fig:ir_evolution`,
   `fig:pipeline_ablation`, `fig:3.6 montage` — add a `\ref` in the discussing paragraph (chapter agents).
5. **6 dead raster assets** in `figures/` (unreferenced `fig8_failopen_*`, `fig9_ir_v3b_*`) — archive
   candidates (note for a later `/sweep`, not this pass).
6. **Notes resolved:** fig 3.5 axis already "fraction of GT boxes" (just reconcile 28↔29.8); fig 3.6 =
   orphan (add ref); fig 3.12 has **no model/imgsz confound** (shows each model at both sizes — reassure,
   no change); fig 4.1 whisker=95% CI **already in caption**; label-reviewer is **already two** figures
   (home+launch) but both are unrendered placeholders + unref'd.

---

## Pitstop — what's LOCKED vs still open
**LOCKED by the user (proceed in Stage B):**
- All 3 content changes (grayscale→finding incl. the AUROC blast-radius cut; MRI↔MLP; model cards).
- The 4 correctness fixes: **composition Ch3 → filt→clf**; stale L561; provenance drift
  (`runs/README.md` + reject CLAIMED → shipped no-reject); bib metadata (4 fixes + 2 dead keys).
- Leakage = **test/held-out only, omit exceptions, drop the 8 hedges** (methodology-level phrasing).
- Grayscale "1 fail" = texture-rich close-ups.

**Open (confirm at pitstop):**
- Figure REMOVALS/regens: `fig:distill_verifier_bar` + `fig:patch_catchbar` regenerate; the 6 dead assets
  are archive-only (deferred). No figure other than `fig:ir_gray_align` is cut. → low-risk, recommend proceed.
- Dead citations: wire `ng2021datacentric` into the data-centric framing; `guo2017calibration` — place at
  the calibration/confidence point or drop. → recommend wire both if a natural home exists, else drop guo.
