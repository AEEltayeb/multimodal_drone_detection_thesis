# Thesis Final-First-Draft Audit (2026-06-09)

Full audit of `docs/thesis_working_distilling_overleaf/` (main.tex + 6 chapters) against the
knowledge base (`knowledge/{ledger,evals,models,figures}.csv` — 69 findings, 134 evals, 104 models,
27 figures). Four parallel read-only agents, one per dimension you asked for:
(1) data-backing + conflicts, (2) narrative + alternatives, (3) gaps vs the data, (4) experiment worthiness.

Scope decisions you locked: backing bar = **traceability + conflicts**; narrative = **current + 2–3
alternative spines**; unworthy = **stat-weak / redundant / off-narrative / superseded**; replacements
= **prefer existing data, flag re-runs as [NEEDS COMPUTE]**.

---

## 0. Executive summary — the one thing that matters most

**The thesis advocates a "production stack" (`ft4` + `v3b` + `robust8` + `mlp_v5` + `mlp_v5_ir_aligned`)
that is never evaluated end-to-end; the headline cascade numbers were measured on a *superseded*
configuration (`sa32` + the v2 *patch* verifier).** This single issue surfaced independently in all four
audits:

- **Backing:** the abstract/RQ answers report cascade numbers from `clfzoo_fnfn` / `sa32` + patch; the
  conclusion ships `robust8` + per-frame MLPs and concedes the full-cascade run is "pending the
  `mlp_v5` re-run" (`fig:mlp_pipeline_placeholder`).
- **Narrative:** the system the thesis *advocates* is not the system the headline numbers *measure* —
  the central credibility crack a committee will probe.
- **Gaps:** the production-stack numbers **already partly exist** (`offline-verifier-matrix` [ledger 53],
  `email-recompute-robust6-mlp` [66], evals `svan_classifier_robust6_ta`, `antiuav_classifier_robust6_ta`,
  `svanstrom_ir_aligned`) but are component-level only and parked behind the placeholder.
- **Worthiness:** several results are flagged "superseded-config, disclosed"; the prescribed fix is to
  re-run the full cascade with the production stack (the registered open item).

**Action:** (a) surface the production-stack component numbers that already exist — *no compute* — and
(b) schedule the one end-to-end re-run — *[NEEDS COMPUTE]* — to retire the placeholder. Everything else
in this audit is secondary to this.

The other four must-fix-before-supervisor items:
1. **The 0.895 @ `patch_thr=0.9` headline cost number has no matching eval row** (the only `sa32` S3 row
   is thr 0.8 = 0.896/0.868). It is repeated ≥5× incl. the abstract. Fix provenance or correct the prose.
2. **IR "stronger on both benchmarks"** asserts a 1.1-pp margin that is IR@640 vs RGB@1280 (confounded);
   disclosed in §threats but stated bare at `methodology.tex:461`.
3. **Seagull single-clip "tie" (0.837 vs 0.840)** is the most-quoted RQ3 number and is *not* in the CSV
   (only the 0.636 aggregate is). Keep it strictly as illustration; back RQ3 with a bootstrap CI.
4. **Appendix model table labels all three of robust8/mlp_v5/mlp_v5_ir_aligned "production"** without the
   "not-yet-full-pipeline-validated" qualifier — undercuts the otherwise-honest disclosure.

---

## 1. Data-backing audit (traceability + conflicts)

The chapters are unusually disciplined — **~78 result sentences are TRACEABLE-OK** (numbers match the
cited `eval`/`ledger` row and the claim strength matches the finding's outcome). No claim CONTRADICTS a
ledger `contradicts` link (they are all handled as explicit carve-outs). Problems found:

| Location | Claim (quote) | Cited source | Verdict | Issue / fix |
|---|---|---|---|---|
| methodology:726, abstract:159, intro:42, conclusion:14, empirical:189 | "S3 $F1=0.895$ at `patch_thr=0.9` … $R=0.869$" | `svan_s3_sa32_thr08` | **MISMATCH / UNTRACEABLE** | Only `sa32` S3 row is **thr 0.8** = F1 **0.896**/R **0.868**. No thr-0.9 row exists. Headline operating point (≥5×) unbacked. Record the thr-0.9 run or correct prose to 0.896@thr0.8. |
| methodology:248–250 | "FT4 … only config that cleared all gates … 16-pp confuser cut" | `ft4-backbone-freeze; eval=none` | **UNTRACEABLE (number)** | 16-pp figure has no eval row (gate-sweep only). Acknowledge as such or record. |
| methodology:461, empirical:634 | IR "the stronger single-modality detector on both benchmarks" | `ir_v3b_svan640_may10` vs `rgb_svan_baseline` | **OVERSTATED (mitigated)** | IR 0.961@640 vs RGB 0.950@1280 — confounded by imgsz. Disclosed at empirical:644, but asserted bare at methodology:461. Add the caveat at point of claim. |
| empirical:392, 357, abstract:162 | grayscale v3b "fires ~20× more … (37.2% vs 1.8%)" | `ir-grayscale-is-hallucination-mode; eval=none` | **UNTRACEABLE (number)** | 37.2%/1.8% has no eval row; related open item `prov-ir-gray-confuser-src` is flagged UNKNOWN. |
| empirical:325/357, intro:46, abstract:162 | seagull "$F1=0.837$ vs $0.840$" | `vid_drone_ir_gray` | **UNTRACEABLE (per-clip)** | The cited eval is the *aggregate* (0.636). The per-clip tie lives only in the cache. |
| empirical:471–490, 547–565 | per-variant per-frame P/R cells; per-category FPR | `pipe_vid_*_pf`, `pipe_percat_sa32` | **PARTIAL** | Some table cells exceed what the cited rows carry (e.g. `pipe_vid_retrainedv2_pf` has F1 only, no P/R). |
| empirical:620, methodology:452,618 | "selcom_960 … best cross-surface drone-first RGB variant" | `xsurf_selcom_holdout`, `selcom960-cross-surface-winner` | **OVERSTATED** | Ledger outcome **partial**/status **open** ("loses to baseline on Svan medium + clean video; untested in pipeline"). |
| conclusion:33–37, appendices, abstract:159 | production stack = robust8 + mlp_v5 + aligned, shipped | `robust8-grayscale-router` (**partial**), `dual-verifier-fusion-rule` (**unimplemented**) | **OVERSTATED (mitigated)** | robust8 outcome partial; dual-verifier soft-weight "unimplemented". Disclosed via placeholder, but appendix "production" labels lack the qualifier. |
| methodology:268, empirical:474 | LDA "0.952 RGB, 0.981 IR" separability | `mri-v5-report-regen`, `ir-features-separable` | **OK (note)** | Numbers are in ledger *notes*, not an eval P/R/F1 column; the cited `distill_cv_mlp` is a different quantity. Fine, but not the literal source. |

**Highest-risk (a supervisor will challenge):** (1) the 0.895/thr-0.9 operating point; (2) "production
stack" shipped vs never-validated-end-to-end; (3) IR "stronger on both" confound; (4) seagull single-clip
tie not in CSV; (5) the 16-pp and 37.2%/1.8% `eval=none` numbers.

---

## 2. Narrative analysis

### Current narrative (what's being pushed)
A **systems / engineering deliverable**: headline = "a deployable dual-modality drone-detection system"
(Contribution 1, first); spine = *recall belongs to the detector, precision to the downstream cascade*
("the contribution is the *system* … not any single filter"). RQ1 (confuser suppression) is foregrounded
and gets the most space; the cascade/trust-classifier/verifier dominate the methodology chapter.

**Tensions:** (a) *identity crisis* — it calls itself a systems paper but its two most novel, defensible
results are scientific observations (the 28-pp scoring-rule audit; emergent thermal→grayscale transfer)
that it actively downplays; (b) *abstract ≠ conclusion* — the headline numbers measure a superseded
config, the conclusion ships a different one (the §0 issue); (c) *the flagship RQ2 number is a cost*
(0.950→0.895, a 5.5-pp loss) the text has to reframe.

### Alternative spines (same evidence, reordered)

- **Spine A — Scientific findings (RECOMMENDED).** *Two measurement truths govern multi-modal drone
  detection — F1 is not comparable across studies without scoring-rule disclosure (28-pp swing on
  identical detections), and a thermal-only detector transfers zero-shot to grayscale video — and the
  cascade is the engineering consequence.* Headline = the scoring audit (`scoring-rule-swing`, confirmed)
  + cross-modal transfer (`ir-grayscale-fallback`, `gray-thermal-alignable`, confirmed, mechanism-backed).
  System demotes to "application." **Pro:** both findings are confirmed, reproducible, and *independent of
  the unsettled production stack*; converts the two strongest results into the headline and makes the
  pending re-run a limitation of an *application*, not a crack in the *main claim*. **Con:** findings
  theses face a higher generality bar (transfer shown on 9 clips / 1,234 frames; swing on one benchmark).

- **Spine B — Methodology.** *Statistics-before-training (Model MRI + leakage-aware feature selection) and
  HITL co-development produce better components at lower cost than train-and-measure.* Headline = the
  method; cascade = worked example. Strong ledger support (`fusion-feature-leakage`,
  `blurriness-is-corpus-artifact` (within-source AUROC 0.516 = chance), `ir-version-progression`).
  **Pro:** ages well, shows maturity. **Con:** risks reading as "good practice," not a research result.

- **Spine C — Negative-results-done-right.** *Mapping the failure boundaries (bird-mining wall, recall
  collapse, V5 HITL regression, airplane-inert cascade) is more useful than another saturated benchmark.*
  Headline = the boundary map. **Pro:** honest, differentiating. **Con:** risky as the *primary* frame for
  a Master's; best as a strong secondary.

### Recommendation
**Lead with Spine A; keep the system as the delivery vehicle; fold Spine B's rigor in as the method.**
Evidence strength points here: the scoring audit and cross-modal transfer are the most novel, most
reproducible, and least dependent on the `mlp_v5` re-run that the current systems-headline rests on.
Reframing makes the two cleanest results the headline and defuses the production-stack credibility crack.

---

## 3. Gaps — in the data but missing / under-represented in the thesis

| # | Missing item | Backing (ledger / eval) | Why it matters | Where |
|---|---|---|---|---|
| 1 | **Full production-stack numbers (ft4+robust8+mlp_v5/aligned)** end-to-end | `offline-verifier-matrix`(53), `email-recompute`(66); `svan_rgbfilter_mlp_ta`, `svan_classifier_robust6_ta`, `antiuav_classifier_robust6_ta`, `svanstrom_ir_aligned` | The shipped system is never shown whole; data partly exists | replace `fig:mlp_pipeline_placeholder` (empirical §realvideo/§cumulative) |
| 2 | **f16→f32 verifier caching bug** (silently zeroed P(drone)) | `mlp-feats-need-f32`(65) | Measurement-integrity caveat + validates the offline caches | methodology reproducibility |
| 3 | **Verifier P(drone) as a trust-classifier feature** (RGB AUROC 0.949 vs raw-conf 0.842) | `verifier-score-as-classifier-feature`(60), `routing-failure-is-trust-rgb-recall`(61) | The actual *lever* behind the grayscale fix; only the symptom (robust8 patch) is shown | methodology §feature_selection |
| 4 | **robust6/8 speed (404×, 0.095 vs 38.3 ms/frame)** | `robust6-speed-feature-efficiency`(68) | Concrete measured efficiency win for "production-viable" | conclusion §limits |
| 5 | **Embedding-distillation CV precursor** (LogReg 0.71→0.99) | `embedding-distillation-cv`(33); `distill_cv_mlp` | Design lineage of the V5 verifier | methodology §distill_verifier |
| 6 | **MLP-filter beats CNN-patch on OOD** (helis 30%→84%) | `mlp-filter-beats-cnn-ood`(67), `mlp-beats-patch-both-modalities`(57) | Quantitative basis for replacing the patch verifier | empirical/methodology verifier comparison |
| 7 | **v5.2 coverage-boost disproof** (+14.5k drones = net-negative) | `v5-rgbds-ceiling`(9); `v52_rgbds_remine` | Clean "why naïve more-data fails" negative result | methodology carve-out |
| 8 | **IR Anti-UAV bare number** (0.987/0.945/0.965) | `antiuav-saturated`(5); `ir_v3b_antiuav640_may10` | Symmetric IR saturation evidence for §threats | empirical IR / threats |
| 9 | **dual_classifier_v3 missing-modality fallback** (bird 0.247→1.000) | `dual-classifier-v3`(32); `dualclf_v3_vs_sa32` | Strong grayscale/missing-modality result; absent | methodology §grayscale (NOTE: §4 flags it off-narrative — see below) |
| 10 | **ir-mlp-conf-not-loadbearing** (dropping conf is a no-op) | `ir-mlp-conf-not-loadbearing`(46) | Pre-empts "isn't the verifier just re-using confidence?" | methodology §grayscale_verifier |

**Orphaned figures** (have `\includegraphics`+`\label` but no in-text `\ref` — examiners flag this):
`fig:dataset_montage`, `fig:drone_size_hist`, `fig:confuser_problem`, `fig:pyside_gui`, `fig:ood_classifier`,
`fig:robust8_operating`, `fig:patch_sweep`, `fig:patch_catchbar`, `fig:fusion_leakage`, `fig:ir_evolution`,
`fig:v5_regression`, `fig:cumulative_confuser`, `fig:svanstrom_by_cat`, `fig:surface_exchange`,
`fig:realvideo_pareto`, `fig:cascade_segment_fig`, `fig:classifier_reversal`, `fig:label_reviewer`,
`fig:hitl_loop`, `fig:pipeline`. (Add `\ref` anchors; several are load-bearing figures floating unanchored.)

**Warranted-but-missing figures:** full production-cascade exchange figure (replaces the placeholder);
verifier-score-as-feature AUROC plot; classifier speed/feature-cost bar; dual_clf_v3 per-surface F1.

**Under-evidenced RQs:** RQ1 answered by a *stand-in* config not the shipped system (see §0); RQ3's
*verifier/classifier* grayscale half (`gray-thermal-alignable`, `ir-grayscale-harvest…`, `dual-classifier-v3`,
`verifier-score-as-feature`) is the richest, most novel evidence cluster in the base and is only partially
surfaced; deployment/latency is "unmeasured" yet the measured *relative* speed (ledger 68) isn't consolidated.

---

## 4. Experiment worthiness (scored) — cut / trim / strengthen

**Cut list (clearest removals):**
1. **selcom_960 cross-surface winner (§selcom)** — OFF-NARRATIVE; ledger `open`/`partial`, not shipped → one-line future-work.
2. **Patch version history v1–v4 (four near-identical F1s)** — REDUNDANT → "v2 shipped; v3 over-aggressive; v4 ties v2."
3. **control40 column in `tab:classifiers`** — REDUNDANT (thesis itself: "buys nothing over sa32") → footnote.
4. **`tab:cascade_perframe` (full 4-row table)** — REDUNDANT/pedagogical (exists only to be refuted) → one sentence.
5. **Fail-open gate table + `fig:failopen_expanded` (§mlp_recall_drop)** — OFF-NARRATIVE (documents a *rejected* fix at table+figure length) → keep the 1-para diagnosis.
6. **dual-classifier-v3 (§classifier_variants)** — OFF-NARRATIVE (not shipped; several numbers single-clip) → future-work paragraph. *(Tension with Gap #9: the result is novel — decide whether to promote it into the RQ3 story or cut it; don't leave it as an orphan mid-chapter.)*

**Strengthen list (keep, but the number must improve — mostly [NEEDS COMPUTE]):**
1. **`fig:resolution` imgsz curve** — STAT-WEAK: two endpoints from *different* variants (baseline@1280 vs retrained_v2@640). **[NEEDS COMPUTE]** baseline RGB @ imgsz=640 on Svanström for a true single-model sweep.
2. **`tab:realvideo_seagull` + `fig:grayscale_qualitative`** — STAT-WEAK (single clip / single frame). Keep strictly as illustration; **[NEEDS COMPUTE]** a bootstrap CI on the aggregate `vid_drone_ir_gray` vs `vid_drone_baseline` (1,234 GT frames) — the "bootstrap CI not an eyeballed gap" discipline.
3. **`tab:cascade_percategory`** — STAT-WEAK (~20–340 frames/clip; thesis admits wide sampling error). Keep the bird-vs-airplane *asymmetry*, demote per-cell precision.
4. **`tab:cascade_segment` + `tab:cascade_classifier`** — single-run, `sa32`+patch (pre-production). **[NEEDS COMPUTE]** the full-cascade re-run with `robust8` + `mlp_v5`/`mlp_v5_ir_aligned` (the §0 fix).
5. **Cascade headline confuser numbers** — currently anchored on `clfzoo_fnfn` (baseline+patch). Cite `svan_classifier_robust6_ta` / `antiuav_classifier_robust6_ta` alongside so the headline reflects the shipped stack — *no compute, data exists*.

**KEEP (sound as-is):** `tab:ir_evolution`, `tab:rgb_comparison`, Roboflow OOD audits, `tab:ir_grayscale`
(aggregate), `tab:ir_aligned`/`_gray` (CBAM held-out), `tab:scoring` (28-pp audit), Model-MRI stats,
`tab:leakage`/`tab:robust6_pipeline`, `tab:distill_verifier`, Anti-UAV saturation rows.

---

## 5. Prioritized action plan (before it goes to your supervisor)

**Tier 1 — fixes, no compute (do these first):**
- Fix the **0.895 @ thr-0.9** provenance (correct to 0.896@thr0.8 or record the run). [§1]
- Add the caveat to **IR "stronger on both benchmarks"** at point of claim. [§1]
- Re-label the **appendix "production" rows** with "component-validated; full-pipeline re-run pending". [§1]
- **Surface the production-stack component numbers** that already exist (`svan_classifier_robust6_ta`,
  `antiuav_classifier_robust6_ta`, `svanstrom_ir_aligned`, `offline-verifier-matrix`) alongside the
  comparison numbers, so RQ1 is answered by the shipped stack. [§0, §3.1, §4]
- Add **`\ref` anchors** for the ~20 orphaned figures. [§3]
- Add the **cheap missing findings**: f16→f32 caching note (65), robust6/8 speed (68), MLP-beats-CNN-OOD
  (67), IR Anti-UAV row (5), conf-not-loadbearing (46). [§3]
- Execute the **cut list** (§4) and the **redundant-table trims**.

**Tier 2 — narrative (decide, then restructure):**
- Decide whether to **reframe to Spine A** (findings-led). This is the highest-leverage change and it
  *defuses* the production-stack crack. If yes, promote the scoring audit + cross-modal transfer; demote
  the cascade to "application". [§2]

**Tier 3 — [NEEDS COMPUTE] (the runs that move "lower bound" → "result"):**
- **Full-cascade re-run** with `robust8` + `mlp_v5`/`mlp_v5_ir_aligned` on Svanström/Anti-UAV/real-video
  (retires `fig:mlp_pipeline_placeholder`; the single biggest credibility upgrade). [§0]
- **Bootstrap CI** on the RQ3 grayscale aggregate. [§4]
- **baseline RGB @ imgsz=640** on Svanström for the resolution sweep. [§4]

---

## Delivered
- This audit: `docs/analysis/2026-06-09_thesis_final_audit.md`
- Source data: `knowledge/{ledger,evals,models,figures}.csv`; thesis `docs/thesis_working_distilling_overleaf/`
- No thesis edits made (audit only). Backups from prior cleanup: `docs/_ledger_strip_bak_2026-06-08/`,
  `docs/_emdash_bak_2026-06-08/`, `docs/_appendices_bak_2026-06-09.tex`.
