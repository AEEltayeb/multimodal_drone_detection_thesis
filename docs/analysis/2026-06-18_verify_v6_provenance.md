# Verify: "every reported eval surface is test/held-out for the model scored on it" (2026-06-18)

**Auditor mode:** READ-ONLY data-provenance audit. No thesis edits made; this ledger is the only write.
**Question (user's claim, verified not assumed):** *"Every evaluation surface that reports a number in
the thesis is a TEST split or HELD-OUT set relative to the model(s) being evaluated — no training
data leaks into any reported metric."*

**Verdict (one line):** **MOSTLY TRUE, and the thesis already discloses every exception.** All
OOD/benchmark/test-split surfaces are genuinely disjoint from the models scored on them. There are
exactly **three families of IN-SAMPLE pairs**, each already flagged in the thesis: **(1) the IR
detector v3b on Svanström IR and Anti-UAV** (training-corpus overlap, quantified + bounded by a
held-out clean split), **(2) the confuser filters on Svanström RGB/IR** (no split exists → fair only
as a shipped-vs-candidate Δ; stated inline), and **(3) the patch filter on Svanström/Anti-UAV** (crop
sources include both). Plus the structural near-tautologies that are *not* leakage but worth naming:
Anti-UAV RGB is in-distribution for every RGB detector (used as a declared no-harm control, not a
discriminating claim), and the trust router's training rows are mined from both paired surfaces
(disclosed as a residual; the router's own protocol is sequence-stratified).

The user's phrasing "*no training data leaks into any reported metric*" is too strong as literally
stated — some headline cells (Svanström-IR solo, the in-sample filter Δ rows, Anti-UAV) **are** computed
on data overlapping training. But the thesis never *hides* this: it labels each one, quantifies the
overlap per component (Table `tab:svanstrom_audit`), and bounds the inflation on a 57,542+5,557-frame
**held-out clean split** (Table `tab:clean_split`). So the honest restatement is: *every surface is
either fully held-out, or its in-sample status is explicitly declared and its inflation bounded.*

---

## TABLE 1 — Provenance matrix (surface × model scored)

Split class: **TEST** = held-back split of a corpus the model trained on · **HELD-OUT** = corpus
entirely excluded from training (OOD / disjoint) · **VAL** = validation split · **IN-SAMPLE** = scored
data overlaps the model's *training* data (leakage; flagged).

| Surface | Model(s) scored | Split class | Evidence |
|---|---|---|---|
| **Svanström RGB** (paired) | RGB det `ft4` (+ baseline/hardneg/retrained_v2/selcom ablations) | **HELD-OUT** | No Svanström in any RGB corpus. methodology §sec:ds_rgb_corpus L79 ("No Svanström frames of any class appear in this corpus"); `tab:svanstrom_audit` RGB row = "Clean"; ft4 card `trained_on` = selcom+confuser, no Svanström |
| **Svanström IR** | IR det `v3b` | **IN-SAMPLE** | `ir_dset_final` `svan` prefix = 21,637 frames (`tab:ds_ir_components` L148); `tab:svanstrom_audit` IR row: 17,314 train frames, **37.3% of eval frames are exact train images**. Bounded: clean-split solo F1 0.940→0.867 (≤7.3pp inflation) |
| **Svanström grayscale** | IR det `v3b` on grayscale-RGB | **HELD-OUT (cross-domain)** | v3b never trained on visible-light/grayscale input; it is the grayscale *fallback* mode. `tab:gray_threeway` L725-727; finding §sec:grayscale |
| **Anti-UAV RGB** (paired/test) | RGB det `ft4` (+ ablations) | **IN-SAMPLE (declared no-harm control)** | 59,413 anti_uav_* frames in RGB corpus (`tab:ds_rgb_components` L63); `tab:svanstrom_audit` RGB-AntiUAV: corpus holds material from **all 91 eval segments, 16.0% exact frames**. Used as saturated control, methodology §sec:ds_antiuav L181 |
| **Anti-UAV IR** (paired/test) | IR det `v3b` | **IN-SAMPLE (declared no-harm control)** | `tab:svanstrom_audit` IR-AntiUAV: 22,603 train frames, 30/90 eval segments, **6.3% exact train images**. Bounded: clean-split IR solo 0.961→0.966 (no inflation) |
| **DUT Anti-UAV test (2,200/960)** | RGB det `ft4`; IR `v3b` on grayscale | **TEST (official split)** | `dut` is a source in composite `rgb_dataset` (`tab:ds_rgb_components` L69) → ft4 **in-domain** but only the *official test split* is used; empirical L106 "no DUT frame here was a training image for the router or filters"; datasets.csv row 13 caveat. Grayscale-IR side = cross-domain held-out |
| **SelCom val (311)** | RGB det `ft4` (+ selcom_ft2/ft3); RGB filter `mlp_v5_v4` | **VAL (held-out, blocklisted)** | Pure-SelCom 15% val seed 0; ft4 fine-tune trains on remaining frames (methodology §sec:ds_selcom L187); filter card: "the 311 SelCom-val images excluded"; provenance doc §1.3 ✅ DISJOINT by blocklist |
| **rgb_dataset test (17,209)** | RGB det `ft4`/baseline/retrained_v2; RGB filter `mlp_v5_v4` | **TEST** | 80/10/10 seq-disjoint (`tab:ds_rgb_components`); filter trained on `…/{train,val}` only, eval on `…/test` — provenance doc §1.3 ✅ DISJOINT |
| **ir_dset_final test (9,612)** | IR det `v3b` (+ V2–V6/Final); IR filter `mlp_aligned_thermalonly` | **TEST** | 83.5/9.1/7.4 split (`tab:ds_ir_components`); IR-version table on "fixed IR_dset_final test split" (empirical L432); filter trained on `…/train`, eval `…/test` — provenance §2.3 ✅ DISJOINT |
| **ir_video test (831)** | IR det `v3b`; IR filter `mlp_aligned_thermalonly` | **TEST** | filter trained on `IR_video/train`, eval `IR_video/test` — provenance §2.3 ✅ DISJOINT; `tab:ir_aligned` L640 |
| **Anti-UAV IR test (filter ctx, 4,269)** | IR filter `mlp_aligned_thermalonly` | **TEST (val/test disjoint)** | filter trained on Anti-UAV **val**/IR; eval Anti-UAV **test**/IR; val∩test = 0 — provenance §2.3; `tab:svanstrom_audit` IR-filter AntiUAV col = "Clean" |
| **CBAM valid (180)** | IR filter `mlp_aligned_thermalonly` | **HELD-OUT** | filter trained on CBAM **train** (GT-aware); eval CBAM **valid** (disjoint). The recall-recovery gate. provenance §2.3; empirical L638, L657, §sec:ds_ir_confusers L169 ✅ |
| **rgb_confusers_merged test (2,633)** | RGB det `ft4` (headline); RGB filter `mlp_v5_v4` | **HELD-OUT for ft4 (per-model) / TEST for filter** | Per-model status spelled out methodology §sec:ds_confusers L111: for ft4 the 2/3 Svanström is fully OOD, remaining 1/3 sequence-disjoint. Filter trained on `…/{train,val}`, eval `…/test` — provenance §1.3 ✅ |
| **IR_confusers (val/test, 388 fires)** | IR filter `mlp_aligned_thermalonly` | **HELD-OUT** | "honest" held-out suppression 90→22; filter trained on `IR_confusers/train`. evals.csv `ir_confusers_heldout_cbam`; provenance §2.3 ✅ |
| **IR_confusers on-cache (4,000, fire 0.028)** | IR filter `mlp_aligned_thermalonly` | **IN-SAMPLE (disclosed)** | This cache = `IR_confusers/train` = the filter's own train split. empirical L649 names it explicitly; the held-out 90→22 is reported alongside. evals.csv `ir_confusers_thermalonly_final` ("LEAKY CACHE") |
| **bird.v1i TEST (484, 230 fires)** | RGB filter `mlp_v5_v4` | **HELD-OUT** | 60/40 seed-0 name split; filter trains on 728 train names, eval 484 unseen; 30/230 kept vs predecessor 91/230. empirical L616; provenance §1.2-1.3 ✅ |
| **rgb_confusers (on-cache, fire 0.014/39FP)** | RGB det `ft4` + filter `mlp_v5_v4` | **HELD-OUT for ft4; mixed for filter** | The pipeline `tab:ablation_confusers` cell. ft4 OOD; the filter's confuser training shares source pools (`rgb_confusers_merged/{train,val}`) but eval is the **test** split → still disjoint at split level |
| **grayscale confusers (656→15 FP)** | grayscale filter `mlp_aligned_gray_balanced` | **HELD-OUT (cross-domain)** | grayscale-converted `rgb_confusers` through the gray head; §sec:grayscale L776 |
| **YouTube real-video (19 clips, 2,609 fr)** | all components (ft4, v3b, robust8-nr, filters, patch) | **HELD-OUT (fully OOD)** | methodology §sec:ds_youtube L193 "only fully OOD surface for every pipeline component: no clip appears in any training set"; `tab:temporal_production` caption "Fully out-of-distribution for every component" |
| **Svanström clean split (5,557)** | ft4, v3b, robust8/-nr cascade | **HELD-OUT (both detectors)** | 54/279 seqs with zero frames in IR training; RGB clean everywhere. `tab:clean_split`; runs/clean_split/ |
| **Anti-UAV clean split (57,542)** | ft4, v3b, robust8/-nr cascade | **HELD-OUT (IR only; RGB in-sample, disclosed)** | 61/91 segs IR-clean; RGB corpus touches all segments (no RGB-clean subset exists) — stated in `tab:clean_split` caption + L296 |
| **Trust router on Svanström / Anti-UAV** (every `clf`/`clf→filt`/`filt→clf` cascade cell) | `robust8-nr` (prod), `robust8`, `robust6`, `sa32` | **IN-SAMPLE (sequence-overlap, disclosed residual)** | `tab:svanstrom_audit` router row: trained on **214/273** Svanström eval seqs, **61/90** Anti-UAV eval segs. Disclosed L324; stricter router-excluding subset shows gain persists |
| **IR test / RGB test / SelCom (router solo rows)** | `robust8`/`robust8-nr` on single channel | inherits surface class (TEST/VAL) | router rows in `tab:ablation_solo` L196-210; router training overlap is the same sequence-level disclosure |
| **Predecessor design-evolution surfaces** (confuser_zoo_1280, svan stride-9 @1280, pipe_video) | sa32, fusion_no_fn, control40, patch_v2, baseline RGB | HELD-OUT / IN-SAMPLE per surface, all **labelled "design-evolution / predecessor configuration"** | `tab:classifiers`, `tab:robust6_pipeline`, `fig:cascade_segment_fig` all carry explicit predecessor-config captions; not production claims |

---

## TABLE 2 — In-sample exceptions (the complete list), and whether the thesis flags each

| # | Surface × model | Why it is IN-SAMPLE | Thesis flags it? | Where |
|---|---|---|---|---|
| 1 | **Svanström IR × v3b** | `svan` prefix (21,637 fr) is the IR corpus's largest single source; 37.3% of eval frames are exact train images | **YES — loudly** | `tab:ds_ir_components` caption ("the reason Svanström IR evaluation is in-distribution"); `tab:svanstrom_audit` (37.3%); bounded by `tab:clean_split` (≤7.3pp, RGB control −3.5pp); §sec:threats L790 |
| 2 | **Anti-UAV RGB × ft4** (and all RGB dets) | 59,413 anti_uav_* frames in composite RGB corpus; covers all 91 eval segments | **YES — declared as no-harm control, not a discriminating claim** | `tab:ds_rgb_components` L63; methodology §sec:ds_antiuav L181 ("used as a no-harm control"); `tab:svanstrom_audit` 16.0% exact; clean-split shows **0** inflation |
| 3 | **Anti-UAV IR × v3b** | 22,603 train frames, 6.3% exact | **YES** | `tab:svanstrom_audit`; clean-split 0.961→0.966 (none) |
| 4 | **Svanström RGB × confuser filters** (`mlp_v5_v4` RGB; `mlp_aligned_thermalonly` IR) | Svanström drone/confuser crops used in filter training; no Svanström train/test split exists | **YES — "fair only as a Δ"** | empirical L616 ("the one in-sample exception is Svanström RGB"), L649 (Svanström IR "in-sample exception … fair only as a Δ"); `tab:svanstrom_audit` filter rows "In-distribution"; provenance doc §1.3, §2.3, §3 |
| 5 | **Svanström/Anti-UAV × patch filter** (`patch_v2`) | crop training sources include Svanström + Anti-UAV | **YES** | `tab:svanstrom_audit` patch row "In-distribution (crop sources incl. Svanström / Anti-UAV)"; patch is a superseded ablation anyway |
| 6 | **IR_confusers on-cache (fire 0.028) × `mlp_aligned_thermalonly`** | the on-cache surface = `IR_confusers/train` = filter's own train split | **YES — named, with held-out 90→22 reported beside it** | empirical L649; evals.csv `ir_confusers_thermalonly_final` tagged "LEAKY CACHE" |
| 7 | **Trust router cascade cells × robust8-nr/robust8/robust6/sa32** on Svanström & Anti-UAV | router rows mined from 214/273 + 61/90 eval sequences | **YES — disclosed residual** | `tab:svanstrom_audit` router row; L324 ("one residual overlap remains disclosed rather than removed"); stricter subset checked |
| 8 | **Anti-UAV clean split RGB side × ft4** | RGB corpus touches all 91 segments → no RGB-clean subset | **YES** | `tab:clean_split` caption + L296 (IR-clean only) |

Note on #2/#3/#7: these are not "leakage that inflates a discriminating result" — Anti-UAV is the
*saturated control* the thesis explicitly says proves nothing on its own, and the router overlap is at
sequence level with the pipeline gain shown to persist on the router-excluded subset. Only #1 (and the
filter Δ #4) touch a *headline discriminating* number, and both are bounded/Δ-scoped.

---

## VERDICT (precise)

The claim **"every reported surface is TEST or HELD-OUT relative to the model scored"** is **TRUE for
every OOD / benchmark / test-split surface** (Svanström RGB, all confuser test splits, bird.v1i TEST,
CBAM valid, IR_confusers val/test, rgb/ir/selcom test splits, DUT official test, YouTube video, the two
clean splits). It is **FALSE as an absolute** only for the disclosed in-sample set in Table 2 — and in
every one of those cases the thesis **labels the surface in-sample, quantifies the overlap, and (for the
detectors) bounds the inflation on a held-out clean split.** No undisclosed leak was found. The honest,
defensible headline is:

> **Every evaluation surface is either fully held-out from the model scored on it, or its in-sample
> status is explicitly declared and its effect bounded.** The in-sample cases are exactly: the IR
> detector on the two paired benchmarks (overlap quantified; ≤7.3pp Svanström-IR inflation, zero on
> Anti-UAV, measured on a 63k-frame held-out clean split); the confuser filters on Svanström, where no
> split exists and the absolute number is therefore read only as a shipped-vs-candidate Δ; and the trust
> router's sequence overlap on both paired surfaces (gain shown to persist on the router-excluded
> subset). Anti-UAV is in-distribution by design and used as a no-harm control, never as a discriminating
> result.

---

## RECOMMENDED GLOBAL STATEMENT (place once, before the results)

Drop this as a short paragraph at the head of §sec:eval_protocol (or as the first paragraph of
Chapter `ch:hitl`), so the per-surface hedges can be trimmed to one-word labels:

> **Evaluation integrity.** Every number in this thesis is computed on data that is either held out from
> the model producing it or, where it is not, declared in-sample with its overlap quantified. The fully
> held-out surfaces — Svanström RGB (absent from every RGB corpus), the in-distribution test splits
> (`rgb_dataset`, `ir_dset_final`, SelCom-val, the official DUT test split), every confuser test split
> (`rgb_confusers_merged`, `IR_confusers` val/test, bird.v1i test, the CBAM held-out gate), the two
> sequence-level clean splits, and the 19-clip YouTube set (out-of-distribution for every component) —
> carry the discriminating claims. Three in-sample cases remain and are reported as such: the IR detector
> overlaps both paired benchmarks (quantified in Table~\ref{tab:svanstrom_audit} and bounded on a
> 63{,}099-frame held-out clean split in Table~\ref{tab:clean_split}); the confuser filters were trained
> on Svanström crops, so Svanström filter numbers are read only as shipped-versus-candidate deltas; and
> the trust router's training rows were mined from both paired surfaces, with the pipeline gain shown to
> persist on the router-excluded subset. Anti-UAV is in-distribution for the RGB detector by construction
> and serves throughout as a no-harm control, not as a discriminating result.

(Adjust the 63,099 = 57,542 + 5,557 if you prefer to cite the two splits separately.)

---

## Apologetic / hedging passages to consider replacing with the single global note

These are the per-surface disclosures currently scattered through the text. They are all *correct* and
*not over-apologetic* (the thesis tone is matter-of-fact), so this is an optional consolidation, not a
correctness fix. Replace the long inline hedges with a one-word label ("held-out" / "in-sample, Δ-only")
once the global note above is in place.

- `methodology.tex` §sec:ds_confusers **L111**: *"What ''out-of-distribution'' means on this corpus's
  test split is per-model … The headline suppression numbers of Chapter 4 are ft4 numbers and carry this
  status."* — keep (it carries the per-model tally) but can shorten now the global note exists.
- `methodology.tex` §sec:ds_antiuav **L181**: *"Anti-UAV is therefore used as a no-harm control: a strong
  result here is necessary but not sufficient…"* — fold the "necessary but not sufficient" into the global
  note.
- `methodology.tex` §sec:svanstrom_audit **L268**: *"The headline numbers of this thesis remain the
  full-surface figures; the clean split is the control that bounds their inflation."* — this is the anchor;
  keep, reference it from the global note.
- `methodology.tex` **L324**: *"One residual overlap remains disclosed rather than removed: the trust
  router's training rows were mined from both surfaces…"* — keep (it is the router caveat); the global note
  can summarise it.
- `empirical.tex` §sec:verifier_results **L616**: *"the one in-sample exception is Svanström RGB (no
  train/test split exists, so its absolute number is fair only as a shipped-vs-candidate Δ)."* — exact and
  load-bearing; keep, this is the model for how to phrase the filter Δ.
- `empirical.tex` **L649**: *"with Svanström IR the in-sample exception (drone recall held flat at 0.966,
  fair only as a Δ). This held-out 90→22 is distinct from the on-cache IR_confusers fire … whose surface is
  the filter's own train split."* — keep; it correctly separates the leaky cache from the held-out number.
- `empirical.tex` §sec:threats **L790**: *"The IR detector and the trust classifier train on frames and
  sequences that overlap both paired evaluation surfaces; the overlap is quantified per component, and
  bounded by the held-out clean split…"* — this is the threats-to-validity summary; keep, it is the right
  place for the bound.
- `empirical.tex` DUT **L106**: *"Only the official test split is used (no DUT frame here was a training
  image for the router or filters)."* — keep (it is the disjointness guarantee for a corpus where ft4 is
  in-domain).

**One small wording risk to watch (not an error):** `tab:datasets` caption (`methodology.tex` L13) says
*"no OOD surface contributes any training data."* That is true for the *OOD* surfaces as defined, but a
careless reader could over-extend it to "no eval surface overlaps training" — which is false for the
in-sample set above. The global note resolves the ambiguity; no change to L13 strictly required.

---

## Sources cross-referenced
- Model cards: `models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.model_card.yaml`,
  `models/ir/corrective_finetune/finetune_v3b/weights/best.model_card.yaml`,
  `models/routers/robust8_noreject_drop/model.model_card.yaml`,
  `models/verifiers/rgb_v5/mlp_v5_v4.model_card.yaml`,
  `models/verifiers/ir_aligned/mlp_aligned_thermalonly.model_card.yaml`,
  `models/patches/confuser_filter4_rgb_v2_backup.model_card.yaml`
- Filter provenance: `docs/analysis/2026-06-18_filter_provenance_train_heldout.md`
- Knowledge base: `knowledge/models.csv`, `knowledge/datasets.csv` (rows: clean_split #12, dut #13),
  `knowledge/evals.csv` (rows 158-175 leakage/clean-split/held-out)
- Thesis: `docs/thesis_working_distilling_overleaf/chapters/{methodology,empirical,introduction,conclusion}.tex`
- Detector args: `models/ir/corrective_finetune/finetune_v3b/args.yaml` (data.yaml itself lives at the old
  `ES_Drone_Detection/runs/corrective_finetune/dataset_v3/` path and is not resident here; the
  Svanström-IR overlap is authoritatively given instead by `tab:ds_ir_components` + `tab:svanstrom_audit`)

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-18_verify_v6_provenance.md` (this file)
