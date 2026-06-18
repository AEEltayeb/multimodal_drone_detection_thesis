# Verify v3 — `chapters/methodology.tex` (Ch 3), notes lines 450-824

Read-only verification of the BIGGEST chapter slice. Scope: thesis lines 450-824 (sessions 3-7
of `C:\Users\User\Desktop\thesis notes.txt`). Production-stack currency is judged against the
LOCKED ground truth: detectors RGB=`ft4` / IR=`v3b`; router = **`robust8-nr`** (8 feats, argmax,
**no reject**, reads detector outputs ONLY); filters = `mlp_v5_v4` RGB @0.25 + `mlp_aligned_thermalonly`
IR-thermal @0.05; **composition shipped = `filt→clf`**.

Legend — Status: FOLLOWED / REINTRODUCED / REGRESSED / OPEN-by-design / N-A.
Backed: ✓ value matches source · ✗ contradicted by source · ⚠ stale / partial / unattributed.

NO EDITS were made to any thesis/code/results file. This markdown is the only write.

---

## TABLE 1 — Notes reverify (sessions 3-7, lines 450-824)

| Note (line) | Directive | Status | Thesis loc (file:line + quote + label) | Evidence |
|---|---|---|---|---|
| L472-478 | "Svanström frames appear only as DRONE training positives… OOD for baseline" — verify Svan drones NOT in baseline 172k | **REGRESSED** (claim over-stated) | `methodology.tex:619` "Svanstr\"om imagery is held out from training." + L513 "$172{,}022$-frame composite RGB corpus" | `models/rgb/Yolo26n_trained/...model_card.yaml` + `knowledge/models.csv` baseline `train_dataset = "general RGB drone corpus (+ drone-vs-bird subset)"` — Svan **not listed**. So Svan is held OUT entirely; the L472-478-style phrasing "appears as DRONE positives" is *wrong* (it appears in NEITHER baseline nor ft4 train sets). L619 "held out from training" is the accurate version. **Flag any residual "appears only as drone positives" wording** (note's own worry confirmed). |
| L481-482, L484-486 | IR detector: give the *fire rate* (much < RGB) + sea/sky/building negatives fire rate | **OPEN-by-design** | L514, L657 "$1.8\%$ of images" vs RGB "$30.4\%$"; L671 tab `Raw detector hallucination rate $1.8\%$/img` | `mri/results/ir_v3b_report/stats.json raw_halluc_rate=0.01765` ✓. IR near-domain fire **1.8%** is now stated and contrasted with RGB 30.4% — directive FOLLOWED for the headline; building-specific fire rate still not broken out (minor open). |
| L488-491 | tab 3.5 "dv5 53,512 thermal drone sequences" = post-HITL **DV5** version; say so | **OPEN-by-design** | tab in Ch3 datasets (label `tab:ds_ir_*`); not in my 450-824 range but cited | `notebooks/ir_dset_final_results.ipynb:643/689` shows `53,512` rows under `dv5_` prefix → 53,512 IS the DV5 (dataset-version-5, post-HITL) thermal count ✓. Verify the prose explicitly says "version 5 after HITL". |
| L501-522 (sess2 carry-over) | scoring-rule / dataset-split / task caveats — did we resolve? | **N-A (Ch4 site)** | Belongs to Ch4 §results; not in methodology 450-824 | Out of this slice; flag for the Ch4 verifier. |
| L561-564 | "27.7pp / F1 0.663 dual vs 0.940 trust-aware @ P 0.979" — possibly OUTDATED (pre-mlp-filter) | **REGRESSED — STALE** | `methodology.tex:561` (note-numbered; live text in §3.2 protocol, not in 450-824 block but cited here) | **CONFIRMED STALE.** Current `thesis_eval/results/tier1_results.json` Svan: dual/bare F1=**0.7415** (P=0.6088); production `clf->filt[robust8_nr_drop]` F1=**0.9308** (P=0.906); `clf[robust8]` 0.9414 (P=0.897). No row pairs 0.663 vs 0.940 at P 0.979. Qualitative "large swing" survives; **exact numbers must be re-pulled**. |
| L611-615 | "FT4 R3: 300 hard negs, freeze=15, 3 ep … R1/R2/A1-A4 gates" never defined | **REGRESSED (undefined in prose)** | `methodology.tex:634` "a \emph{regression-gated} confuser injection (300 hard negatives at \texttt{freeze}$=$15… Section~\ref{sec:training_recipes})" | Gate names ARE defined in `scripts/auto_confuser_ft4.py:54-104`: R1=600hn/f12, R2=300hn/f12, **R3=300hn/f15=WINNER**, R4; A1-A4=ratio variants. So findable in repo, but **R1/R2/A1-A4 are NOT expanded in the thesis prose** — reader can't decode them. Add a one-line gate legend. |
| L628-630 | Re-quote bird/heli/airplane fire rates per stance | **FOLLOWED** | L627-630 baseline bird 94.4%, heli 66.2→41.9%, airplane 74.6→64.7%, retrained_v2 bird→3.4% R→0.306 | `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` matches every cell ✓ |
| L651-657 (§3.6) | verify manifest.json / git hash / weight SHA / deterministic; locate manifests in ES_Drone_Thesis | **FOLLOWED (1 wrinkle)** | `methodology.tex:453` full §sec:reproducibility | `eval/run_manifest.py` has `manifest_dict()`+`cache_identity_tag()`; computes git `rev-parse HEAD`+dirty (L63-67), weight `sha256[:12]` (L37-40). Real file `eval/cache/raw_detections_svanstrom_..._sz1280_st1.manifest.json` has git.commit=`cee9416…`, dirty=true, weights[].sha256_short=`5e11ed739f7b`, env(python/torch/cuda). **Wrinkle:** that manifest's `repo_root` + weight path point to OLD `ES_Drone_Detection` and rgb=`retrained_v2` (cache predates migration). Mechanism sound; some manifests embed stale absolute paths. |
| L663 (§3.6) | which classifier — robust6 / robust8? | **FOLLOWED** | L453 "the production \texttt{robust8} and its comparisons … GroupShuffleSplit(random\_state=42)" | Names robust8 explicitly ✓. (Could add `-nr` for full currency.) |
| L666-672 (§3.7) | "statistics-before-training" scope = filters + robust6/8 ONLY, not detectors/all clf; name the models | **FOLLOWED** | L459 "applies to exactly three models: …`mlp_v5_v4`… cross-modal IR filters… robust6/robust8. The detectors themselves were trained conventionally, and the hand-engineered `sa32`… predates the discipline." | Scope correctly fenced ✓. Strong fix from prior over-reach. |
| L677 (§3.7.1) | stats.json must be easily found in new dir | **FOLLOWED** | L463 "written under `mri/results/<run>/`… e.g. `mri/results/v5_report_regen/stats.json`" | File resident ✓. |
| L679-685 (§3.7.1) | are z-score / CORAL actually MRI-OUTPUT? (LDA/PCA/ANOVA/AUROC known; rest unsure) | **FOLLOWED** | L470-471 "Per-modality z-score… CORAL. Full-covariance alignment…", L463 "produced by the MRI's modality-alignment module (`mri/modality\_align.py`)" | `mri/modality_align.py:14-15,69-70,135-141` literally computes `permod_z` + `_coral()` and prints transfer AUROC ✓. Note's doubt RESOLVED — both ARE MRI-produced. |
| L689 (§3.7.2) | "small MLP" — expand acronym | **FOLLOWED** | L476 "a small multi-layer perceptron (MLP)" | Expanded ✓ |
| L691-693 (§3.7.2) | "CBAM excluded from training" is new/undefined; introduce CBAM in datasets | **REGRESSED (undefined)** | L657 "held-out CBAM \texttt{valid} split (Section~\ref{sec:ds_ir_confusers})", L699 "CBAM \texttt{train} drones" | CBAM cross-ref points to `sec:ds_ir_confusers` (Ch3 datasets, outside 450-824). **Verify CBAM is actually defined there**; within this slice it is used ~6× as an undefined term. Flag for the datasets section. |
| L695-699 (§3.7.2) | "MRI corrected two figures + one headline" — give the example | **FOLLOWED** | L498 full sentence: 35,098→32,931; conf F≈15,000 (rank 6) → p5 ch F=42,346 first | `mri/docs/mlp_v5_report_regen.md §0` confirms 35,098 vs 32,931 and F=42,346; stats.json max_anova_F=42345.7 ✓. Example now concrete. |
| L705-707, L709-714 (§3.8.1) | RGB "trained on composite… bird negatives" oversimplified — be exact; IR "finetuned"→"trained"; "brittle on OOD"→soften | **FOLLOWED** | L513 names AirBird/FBD-SV/WosDetC + VIRAT/UA-DETRAC/BDD; L645 "trained on iteratively curated thermal"; L514/L645 "stronger single-modality detector on Svanström" | RGB corpus now itemised ✓; IR worded "trained"/"finetuned on curated" (L645 uses "finetuned" once — minor); OOD softened to "airplane-dominated residual" + "stronger on Svanström" ✓. |
| L715-718 (§3.8.1, fig 3.10) | figure depicts filt→clf? we use clf→filt; rename "mlp_v5 verifier"→"confuser filter (mlp)" | **REGRESSED — see TOP FIX #1** | `methodology.tex:524` caption "detectors, then the trust classifier, then the confuser filter" (= **clf→filt**) | **MISMATCH with LOCKED ground truth (shipped = filt→clf).** Ch3 still ships clf→filt prose (L521,524,538,560,562). Data: `filt->clf[robust8_nr_drop]` F1=0.9459 **>** `clf->filt[...]` 0.9308 on Svan. Naming "`mlp_v5_v4` filter" is used (not "verifier") ✓, but composition order is stale. |
| L721-723, L740-742 (§3.8.2) | "fail-open" was the PATCH path; no longer for mlp filter — verify | **FOLLOWED** | L528 "confidence-gated veto… survives only if… $P(\text{drone})\geq$ threshold"; L544 §sec:design_rationale retitled "Deferred Suppression" | "Fail-open" reframed to "deferred suppression"; the *fail-open variant* is explicitly "evaluated and rejected" (L528) ✓. Old "fail-open detection" heading gone. |
| L728-738 (sess5) | "IR filter = one network two scalers" — grayscale filter is a SEPARATE z-shifted net (mlp_v5_aligned + mlp_v4_aligned_grayscale); are we shipping a "worse" IR filter to keep grayscale? | **REGRESSED → maps to GRAYSCALE REMOVAL (TOP FIX #2)** | L538 "The IR filter is \emph{two heads}", L699 "ships \emph{two} 517-D heads… single-net design (one network with two per-modality input scalers) was superseded" | Thesis now says TWO heads (`mlp_aligned_thermalonly` + `mlp_aligned_gray_balanced`), supersedes the one-net-two-scaler design ✓ accurate to `knowledge/models.csv`. BUT per LOCKED decision the **grayscale head story is to be REMOVED** (thermal-native only). Whole §sec:ir_xmodal_verifier + L538 two-head framing + L699-700 grayscale = Stage-B deletions (Table 4). |
| L744-753 (§3.8.2) | "we use no scene conditions (only sa32)"; explain "leakage-aware"; isolate imgsz in fig 3.12 | **FOLLOWED** | L560 "The router does not read the scene… eight features are… scene statistics appear only in the hand-engineered `sa32`"; L562 "screened by the scene-fingerprint ratio… cannot pass by memorising surface identity"; L589/Tab `tab:resolution` "both RGB variants at both sizes under one harness" | Scene-free router stated correctly ✓; "leakage-aware" unpacked via the leakage-ratio (Eq 3, L744) ✓; resolution fig/table now isolate imgsz *within* each model ✓. |
| L757-762 (sess5) | baseline Svan P0.940/R0.961 + 94.4% bird — "are you sure that's BARE, not incl. filter?" | **FOLLOWED — confirmed BARE** | `methodology.tex:627` "the bare detector, no downstream stage applied, reaches $P=0.940, R=0.961$ on drones, with a 94.4\% bird-frame fire rate" | `svanstrom_1280_by_category.csv` baseline DRONE P=0.940 R=**0.959** (by-category subset), BIRD det_rate 94.4% — **bare**, no filter ✓. NOTE: CSV R=0.959; thesis L627 R=0.961 (cites full 28,710 corpus via evals.csv). Tiny 0.2pp subset-vs-full gap — harmonise or footnote. |
| L763 (sess5) | "the lesson that motivated the cascade" → say the **filter**, not whole cascade (classifier still needed) | **FOLLOWED** | L630 "The lesson that motivated the per-detection confuser filter… (The trust router is needed regardless; its job is modality choice, not confuser rejection.)" | Exactly the requested correction ✓. |
| L766-768 (sess5) | "compatible with deployed classifier without recalibration swap" is NOT true (we retrained robust6/8) — drop | **FOLLOWED** | not present in current 450-824 text; L726 instead says "A trust classifier is not portable across RGB detector swaps without re-checking calibration" | The false "no-recalibration" claim is gone; replaced by the honest portability caveat ✓. |
| L771-773, L795-805 (sess6) | "MRI" must be introduced; verify IR sep numbers (0.981 / F=5,370 / 14,697 / 1,386); state IR halluc < RGB + pre/post-filter FP on which confuser | **FOLLOWED** | L655-657 "A Model MRI scan (the feature-space instrument of Section~\ref{sec:model_mri})… 0.981 train accuracy… fires on only $1.8\%$… vs RGB $30.4\%$… recovers $R=0.967$… FP to $6$ (bare $48$)"; tab `tab:ir_mri_sep` | All numbers match `mri/results/ir_v3b_report/stats.json`: lda 0.9813, max_anova_F 5370.1, n=14697/1386, raw_halluc 1.8% ✓. CBAM R 0.967/FP 6 (bare 48) per `2026-06-18_filter_provenance_train_heldout.md §2.4` ✓. MRI introduced + dataset named ✓. |
| L777-793, L807-816 (sess6) | "grayscale mode dissolves data-scarcity blocker"/"per-frame IR filter kept OUT of stack" — correct (IR filter ALWAYS present; grayscale is for a *single* filter we don't truly have); guide reader to the z-shift section | **REGRESSED → GRAYSCALE REMOVAL (TOP FIX #2)** | L777-808 of NOTES point at §sec:ir_xmodal_verifier intro + Proof-2 affine-offset prose | The current §sec:ir_xmodal_verifier (L652-704) still tells the grayscale-harvest + affine-offset + "two heads" story. Per LOCKED decision this is **removed wholesale** in Stage B (Table 4). The "data-scarcity blocker" framing the note objects to is exactly what gets cut. |
| L813-816, L818-821 (sess6) | verify Jaccard 0.71-0.88 / corr 0.93-0.99 / cosine 0.012; then SCAN whole thesis for inline numbers that NEED a table | **FOLLOWED (numbers) / OPEN (pitstop)** | L679 "Jaccard $0.71$--$0.88$; mean-activation correlation $0.93$--$0.99$; drone-centroid cosine distance $0.012$" | `mri/docs/ir_grayscale_verifier_report.md:38-39` matches exactly ✓. The "which numbers need a table" pitstop is a design discussion (open by design) — candidates noted in TOP FIXES. (These specific numbers are slated for REMOVAL anyway, Table 4.) |

---

## TABLE 2 — Numbers (verified against source files)

| Number + context | Thesis loc | Source | Backed? |
|---|---|---|---|
| RGB MRI: LDA 0.952, ANOVA F 42,346, conf rank 6 (F=10,696), halluc 54.4%, FP-cut 97.4%, recall-ret 98.9%, n 19,334/13,597 | `:482-495` fig:mri_report | `mri/results/v5_report_regen/stats.json` (lda 0.9517, max_anova_F 42345.7, raw_halluc 0.544, fp_reduction 0.9741, classifier_recall_retention 0.9894; n 19334/13597) | ✓ |
| MRI self-correction: 35,098→32,931 detections; conf F≈15,000 (rank 6) → p5 F=42,346 | `:498` | `mri/docs/mlp_v5_report_regen.md §0` + stats.json | ✓ |
| `mlp_v5_v4` parent LDA ≈95%, silhouette 0.067, 32,931-det corpus, 46-72× faster, 1.3-2.1ms, 1-4% overhead, 5-fold F1 0.9857±0.0004 | `:818,822` | stats.json (lda 0.9517, silhouette 0.0673); `ledger=embedding-distillation-cv`; `eval/bench_speed.py` | ✓ (CV F1 list-backed; speed via ledger) |
| `mlp_v5_v4` held-out bird 91→30/230; rgb_dataset_test recall 0.694→0.874 | `:818-820` | `2026-06-18_filter_provenance_train_heldout.md §1` (both present) | ✓ |
| IR MRI: LDA 0.981, max ANOVA F 5,370 (p5), median 256, halluc 1.8%, FP-cut 89%, recall-ret 99.7%, n 14,697/1,386 | `:657,668-673` tab:ir_mri_sep | `mri/results/ir_v3b_report/stats.json` (lda 0.9813, max_anova_F 5370.1, median 255.7, raw_halluc 0.01765, fp_reduction 0.8896, recall_retention 0.99687) | ✓ |
| Gray→thermal AUROC: raw 0.500 / CORAL 0.707 / z-score 0.919 / ceiling 0.974 | `:691-693` tab:gray_thermal_auroc | `mri/docs/ir_grayscale_verifier_report.md:45-48` + `mri/modality_align.py` | ✓ (but slated REMOVE, Table 4) |
| Affine offset: Jaccard 0.71-0.88, act-corr 0.93-0.99, centroid cosine 0.012 | `:679` | `mri/docs/ir_grayscale_verifier_report.md:38-39` | ✓ (but slated REMOVE) |
| Thermal-native head: CBAM R 0.967 / FP 6 (bare 48); antiuav_ir 0.937, ir_video 0.971, svanstrom_ir 0.966; ir_dset_final 0.965→0.928 (-3.7pp); shared-net 0.717→0.967 | `:657,699-702` | `2026-06-18_filter_provenance_train_heldout.md §2.1-2.4`; `models.csv mlp_aligned_thermalonly` | ✓ |
| Grayscale head: gray Svan recall 0.55→~0.08 @0.25 | `:699` | `tier1_results.json svanstrom_gray.S4_verifier.filt_mlp` R=0.0789 ✓; GRAY_SWEEP 0.02→0.2965 | ✓ (slated REMOVE) |
| Baseline Svan@1280 bare: P 0.940, R 0.961, bird fire 94.4%; heli 66.2→41.9%, airplane 74.6→64.7%; retrained_v2 bird 3.4% / R 0.306 | `:627-630,544` | `eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv` | ⚠ (R: CSV subset 0.959 vs thesis 0.961 full-corpus — reconcile) |
| Conf-sweep: Anti-UAV bare F1 0.9592@0.05; rgb_dataset_test bare 0.9259@0.25; selcom bare 0.5911@0.25, +filt 0.692@0.05 | `:544-545` | `thesis_eval/results/conf_sweep/conf_sweep_results.json` (exact) | ✓ |
| Resolution: baseline @640 0.684 / @1280 0.964; retrained_v2 @640 0.070 / @1280 0.323; n=4,102 stride-7 IoP@0.5 | `:589-614` tab:resolution | `eval/results/svan_resolution_sweep.json` (0.6838/0.9641; 0.0699/0.3234) | ✓ |
| 27.7pp / dual F1 0.663 vs trust-aware 0.940 @ P 0.979 | `:561` (note ref) | `tier1_results.json svanstrom` — dual 0.7415; nr-router 0.9308; no 0.979 pairing | ✗ STALE |
| Patch filter: 45,917 patches, val acc 0.975, airplane 0.907/0.971, heli 0.988/0.938, bird 0.893/0.712 | `:806` | `classifier/runs/patches/confuser_filter4_rgb_metrics.json` (acc 0.9748; airplane 0.9066/0.971; heli 0.9877/0.9375); `models.csv patch_v2` 45,917 | ✓ |
| DV5 thermal drone sequences 53,512 | tab:ds_ir (cited, outside slice) | `notebooks/ir_dset_final_results.ipynb:643/689` (`dv5_` prefix) | ✓ |
| §3.6 manifest: git hash + weight sha256 + cuda + cmdline + deterministic | `:453` | `eval/run_manifest.py` (manifest_dict/cache_identity_tag/_git_commit/_short_sha256) + real `eval/cache/*.manifest.json` | ✓ (⚠ some manifests embed OLD `ES_Drone_Detection` paths) |
| Composition: `filt->clf[robust8_nr_drop]` 0.9459 > `clf->filt` 0.9308 (Svan) | data check | `tier1_results.json svanstrom.B_pipeline` | ✓ data; ✗ thesis prose still says clf→filt is production |

---

## TABLE 3 — Claims (citation / evidence audit)

| Claim | Thesis loc | Verdict | Note |
|---|---|---|---|
| XGBoost gradient-boosted trust classifier | `:719` | CITED `chen2016xgboost` | ✓ key in references.bib |
| MobileNetV3-Small patch backbone | `:806` | CITED `howard2019mobilenetv3` | ✓ |
| "statistics-before-training discipline… exactly three models" | `:459` | EVIDENCED `mri/*` + models.csv | ✓ scope correct; no cite needed |
| Deferred suppression / asymmetric recoverability (design principle) | `:541-544` | EVIDENCED `svanstrom_1280_by_category.csv` (retrained_v2 R-collapse) | ✓ internal evidence; could cite OHEM `shrivastava2016ohem` for hard-neg context (optional) |
| Leakage ratio (Eq 3.x) = F_domain-in-class / F_class | `:742-747` | EVIDENCED `classifier/fusion_feature_stats.py` + `feature_stats_ranked.csv` | ✓ novel statistic, repo-backed |
| robust8-nr is production router (8 feats, reject dropped) | `:562,710,794` | EVIDENCED `models/routers/robust8_noreject_drop/model.model_card.yaml` (production: true) + models.csv | ✓ |
| Router reads detector outputs ONLY (no scene/OpenCV) | `:560,710` | EVIDENCED `classifier/train_routing_robust.py` (ROBUST6+rgb_mean_conf+is_grayscale) | ✓ — directly fixes old L828-829 "AND the source frames" error (that bad phrasing is GONE in this slice) |
| `sa32` leaks scene statistics → upper bound, not deployable | `:560,562,724` | EVIDENCED tab:leakage + `fusion_feature_stats.py` | ✓ |
| `fusion_no_fn_v1.1` / `control_v3more_40feat` (comparison variants) | `:724` | EVIDENCED `models.csv` | ✓ both DEFINED as comparison set at L724 (not undefined terms) — old note worry resolved |
| MRI only DIAGNOSES; filter then trained from same 517-D embeddings | `:474-476` | EVIDENCED `mri/classifier.py`, `mri/holdout.py` | ⚠ wording risk: L476 "its output \emph{is} the filter's training pipeline" / "The MRI is not only a diagnostic" can read as "trained BY the MRI." LOCKED decision wants strict "MRI diagnoses separability → justifies a lightweight MLP." Soften to avoid the "trained from MRI" implication. |
| Reproducibility: manifest/git/SHA/deterministic | `:453` | EVIDENCED `eval/run_manifest.py` + real manifests | ✓ |
| "Svanström imagery held out from training" | `:619` | EVIDENCED models.csv baseline+ft4 train_dataset | ✓ (and corrects the L472-478 over-statement) |
| Model cards / provenance findable in ES_Drone_Thesis | `:453,463` (§3.6/§3.7 cite paths) | EVIDENCED `models/**/*.model_card.yaml` (ft4, v3b, robust8, robust8_nr_drop, mlp_v5_v4, mlp_aligned_thermalonly, mlp_aligned_gray_balanced all present) | ✓ — §3.6/§3.7 ARE the right citation sites; cards resident. Consider explicit `*.model_card.yaml` mention in §3.6. |
| Production composition = classifier→filter (clf→filt) | `:521,524,538,560,562` | UNSUPPORTED vs ground truth | ✗ LOCKED truth = **filt→clf**; data agrees filt→clf wins. **Stale currency.** |
| CBAM (used as held-out IR confuser set) | `:657,699` | NEEDS-CITE / NEEDS-DEFINE | CBAM never defined in this slice; cross-ref to `sec:ds_ir_confusers` — verify it's defined there + tie to `coluccia2021dronevsbird` (drone-vs-bird challenge) if that is the CBAM source |

---

## TABLE 4 — Grayscale-filter / affine-offset-MRI REMOVAL MAP (for Stage B)

Per LOCKED decision: production IR filter is **thermal-native only** (`mlp_aligned_thermalonly`, keep
`tab:ir_aligned`); the grayscale-confuser-harvest + affine-offset story is REMOVED from Ch3. The
grayscale *DETECTOR* transfer stays a finding elsewhere (§grayscale). The IR-separability `tab:ir_mri_sep`
(LDA 0.981) STAYS. All locations below are in `chapters/methodology.tex`.

| Loc | Label / span | What it is | Action |
|---|---|---|---|
| `:459` | sentence "the cross-modal IR filters (thermal-native `mlp_aligned_thermalonly` **and grayscale-aligned `mlp_aligned_gray_balanced`**) via … **and grayscale↔thermal alignment**" | three-model scope list names the grayscale head + alignment | Trim to thermal-native only |
| `:470-471` | description items **Per-modality z-score** + **CORAL** | MRI diagnostics whose sole purpose is the grayscale↔thermal affine-offset story | Remove both `\item`s (they exist only to justify grayscale transfer) |
| `:476` | "(`mlp_aligned_thermalonly` **and `mlp_aligned_gray_balanced`**) are all products of this path" | grayscale head in the "from stats to filter" para | Drop grayscale head |
| `:498` | "Cross-modal alignment is fit from \emph{drones only}…" (first sentence) | sets up the grayscale-transfer independence argument | Remove the alignment sentence; keep the V5-corpus self-audit half |
| `:528` | "(RGB $0.25$, IR-thermal $0.05$, **grayscale $0.25$**)" | threshold triple includes grayscale | Drop grayscale threshold |
| `:538` | **whole `\paragraph{Two-filter fusion}`** "The IR filter is \emph{two heads}… grayscale-aligned net (`mlp_aligned_gray_balanced`, $P\geq0.25$) on the grayscale-fallback path" | the "two heads"/grayscale-fallback framing | Reduce IR filter to single thermal-native head |
| `:647-651` | `\subsubsection{Grayscale-RGB Operating Mode}` (`sec:ir_grayscale_mode`) | grayscale DETECTOR fallback | **KEEP** (detector transfer is a sanctioned finding) — but it must no longer feed a grayscale *filter* |
| `:652-704` | **entire `\subsection{Cross-Modal Feature Alignment: a Thermal Filter from Grayscale Confusers}` (`sec:ir_xmodal_verifier`)** | the grayscale-harvest narrative ("grayscale mode is what let us build a grayscale-aligned head", Proof-2 affine offset) | **Largest removal.** KEEP only: Proof-1 IR-separability + `tab:ir_mri_sep` + the thermal-native head result (`tab:ir_aligned` content, CBAM R0.967/FP6). DELETE: L655 grayscale-harvest framing, L679-680 affine-offset para, `tab:gray_thermal_auroc` (L682-697), `fig:ir_gray_align` ref, L699-700 grayscale-head para, "two heads" everywhere. Retitle the subsection (drop "from Grayscale Confusers"). |
| `:679-680` | "the grayscale→thermal feature gap is a removable affine offset…" + `[source: ledger=gray-thermal-alignable]` | Proof-2 affine-offset prose | DELETE |
| `:682-697` | `tab:gray_thermal_auroc` (raw/CORAL/z-score/ceiling) + caption + `fig:ir_gray_align` ref | the affine-offset evidence table | DELETE |
| `:699-700` | "Production therefore ships \emph{two} 517-D heads… grayscale-aligned head… built from grayscale-harvested confusers per-modality z-aligned… The grayscale path remains recall-limited (0.55→~0.08)" | the two-head + grayscale-recall para | DELETE; collapse to single thermal-native head |
| `:822` | "Its IR counterparts are \emph{two heads}… the thermal-native `mlp_aligned_thermalonly` **and the grayscale-aligned `mlp_aligned_gray_balanced`**" | closing line of §sec:distill_verifier | Trim to thermal-native counterpart only |

Cross-refs that will dangle after removal (fix in Stage B): every `Section~\ref{sec:ir_xmodal_verifier}`
pointing at the grayscale story (e.g. `:459,470,471,476,514,528,538,650,822`), `\ref{fig:ir_gray_align}`,
`\ref{tab:gray_thermal_auroc}`.

---

## TOP FIXES (ranked)

1. **Composition currency — Ch3 still ships `clf→filt`; LOCKED truth + data say `filt→clf`.**
   `methodology.tex:521,524,538,560,562` describe production as classifier-first ("recall-safe,
   within a point of the alternative"; "trust-first: classifier routes, then the filter screens").
   But `tier1_results.json svanstrom.B_pipeline`: `filt->clf[robust8_nr_drop]` F1 **0.9459** >
   `clf->filt` **0.9308**. Re-point the pipeline figure + prose to filt→clf (this is the single
   most material staleness in the slice).

2. **Execute the grayscale-filter / affine-offset removal (Table 4).** Biggest single edit:
   delete most of `sec:ir_xmodal_verifier` (`:652-704`) + `tab:gray_thermal_auroc` + the "two heads"
   framing + grayscale thresholds, keeping thermal-native (`tab:ir_aligned`) and IR-separability
   (`tab:ir_mri_sep`). Detector grayscale mode (`:647-651`) stays as a finding.

3. **L561-564 is STALE (✗).** Replace the "27.7pp / 0.663 dual vs 0.940 @ P 0.979" pairing with
   current tier1 numbers (dual 0.7415; production nr-router `clf->filt` 0.9308 / `filt->clf` 0.9459).
   No row reproduces P 0.979. Keep the qualitative "scoring choice swings the number" point.

4. **Soften "the MRI's output IS the filter's training pipeline" (`:474-476`).** Reads as "filter
   trained BY the MRI." LOCKED stance: MRI *diagnoses* separability → *justifies* a lightweight MLP;
   the MLP is then trained on the same embeddings. Rephrase to keep the diagnose/justify boundary.

5. **Define the gate names + CBAM.** R1/R2/A1-A4 (`:634→sec:training_recipes`) are real in
   `scripts/auto_confuser_ft4.py` but never expanded in prose — add a one-line legend. CBAM
   (`:657,699`) is used ~6× undefined in this slice; confirm it's introduced in `sec:ds_ir_confusers`
   and cite its source.

6. **Reconcile baseline Svan recall 0.959 (by-category CSV subset) vs 0.961 (full-corpus, L627).**
   Both are real; pick one surface per number or footnote the subset-vs-full distinction
   (rule: every number names its dataset/surface).

7. **§3.6 manifest provenance wrinkle.** Mechanism (`run_manifest.py`) and real manifests are
   resident ✓, but at least one `eval/cache/*.manifest.json` embeds OLD `ES_Drone_Detection`
   repo_root + a `retrained_v2` weight path (cache predates migration). Either regenerate the
   cited manifests in `ES_Drone_Thesis` or note that historical manifests carry pre-migration paths.
   Consider naming `*.model_card.yaml` explicitly in §3.6 since all production cards are resident.

---

### Delivered
- `docs/analysis/2026-06-18_verify_v3_methodology.md` (this file) — the only write.

Evidence files opened (all absolute under the repo root
`C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\`):
`docs/thesis_working_distilling_overleaf/chapters/methodology.tex`;
`mri/results/{v5_report_regen,ir_v3b_report}/stats.json`;
`mri/docs/{mlp_v5_report_regen.md,ir_grayscale_verifier_report.md}`; `mri/modality_align.py`;
`eval/run_manifest.py`; `eval/cache/raw_detections_svanstrom_..._sz1280_st1.manifest.json`;
`eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv`;
`thesis_eval/results/{tier1_results.json,conf_sweep/conf_sweep_results.json}`;
`scripts/auto_confuser_ft4.py`; `classifier/runs/patches/confuser_filter4_rgb_metrics.json`;
`notebooks/ir_dset_final_results.ipynb`; `docs/analysis/2026-06-18_filter_provenance_train_heldout.md`;
`knowledge/{models.csv,evals.csv}`;
`models/**/*.model_card.yaml` (baseline, ft4, v3b, robust8, robust8_noreject_drop, mlp_v5_v4,
mlp_aligned_thermalonly, mlp_aligned_gray_balanced).
