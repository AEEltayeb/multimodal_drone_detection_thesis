# Confuser-Filter Definition Map — Thesis Audit (READ-ONLY)

Date: 2026-06-17
Purpose: locate EVERY place the two confuser verifiers ("the filters") are NAMED, DEFINED, or
DESCRIBED QUALITATIVELY so they can be swapped for retrained versions. Numbers are owned by a
separate agent; this map captures definitional/descriptive prose only (numbers appear only where
they are load-bearing for the wording a swap might change).

The filters:
- **RGB** = `mlp_v5` (distilled feature-space MLP verifier), P(drone) >= 0.25, per-frame.
- **IR** = `mlp_v5_ir_aligned` = ONE network, two input scalers: thermal `mlp_aligned.pt` (thr 0.05,
  conf 0.40) and grayscale `mlp_aligned_gray.pt` (thr 0.25, conf 0.25). Built from grayscale-harvested
  confusers z-aligned into thermal feature space.
- Predecessor (superseded, still documented): MobileNetV3 4-class **patch verifier** v2
  (`confuser_filter4_{rgb,ir}_v2_backup.pt`).

Files scanned: `docs/thesis_working.tex` (PRIMARY, 2420 lines) and `docs/thesis_chapters.tex`
(older synced copy, 1675 lines — reported separately at the bottom).

---

## PRIMARY FILE — docs/thesis_working.tex

| file | line(s) | what it says (short quote) | category | RGB/IR/gray/general | swap changes this? |
|------|---------|----------------------------|----------|---------------------|--------------------|
| thesis_working.tex | 158 | "A per-frame confuser verifier then filters... a distilled feature-space MLP that reuses the detector's own ROI features (`mlp_v5` on RGB, `mlp_v5_ir_aligned` on IR), which supersedes an earlier MobileNetV3 patch verifier" | definition | general | yes-wording |
| thesis_working.tex | 202 (RQ1) | "...a per-frame confuser-aware verifier (the distilled feature-space MLP, `mlp_v5` for RGB and `mlp_v5_ir_aligned` for IR...) suppress OOD confuser fire rate..." | definition | general | maybe |
| thesis_working.tex | 215 (Contrib 1) | "the trusted modality's per-frame distilled MLP verifier then acts as the confuser-aware filter (`mlp_v5` on the RGB branch, `mlp_v5_ir_aligned` on the IR branch), superseding the earlier MobileNetV3 patch verifier on every surface measured" | definition / qualitative-claim | general | yes-wording |
| thesis_working.tex | 222 (Contrib 5) | "A distilled feature-space confuser verifier that is the production filter stage on both modalities... MLP consumes the detector's own fused `p3`+`p5` features... `mlp_v5`... `mlp_v5_ir_aligned`... recall-safe... carve-out is a structural recall ceiling on `rgb_dataset` ... photo-style RGB is therefore routed to the patch verifier as a fallback" | definition / training-recipe / qualitative-claim | general | yes-wording |
| thesis_working.tex | 237 | Thesis Outline: "...the distilled feature-space verifier, and the cross-modal IR verifier" (chapter map) | header-ref | general | no |
| thesis_working.tex | 248 | "...a trust classifier and a per-frame distilled MLP verifier" (latency rationale) | definition | general | no |
| thesis_working.tex | 264 | "the trust classifier and the per-frame distilled verifier catch what is left" | qualitative-claim | general | no |
| thesis_working.tex | 306 / 316 (Table tab:related_systems) | "the only one that combines learned modality-trust fusion with a downstream per-frame distilled confuser verifier"; row: "Trust classifier + per-frame distilled (`mlp_v5`) confuser verifier" | definition / novelty-claim | general | yes-wording |
| thesis_working.tex | 321 | "separates the recall-stage (detector) from the precision-stage (trust classifier + per-frame distilled verifier)" | qualitative-claim (precision tool) | general | maybe |
| thesis_working.tex | 539 | PLACEHOLDER fig confuser_problem: "the patch-verifier softmax output under each" | figure-placeholder | RGB | yes-wording |
| thesis_working.tex | 580 (Table tab:svanstrom_audit) | "Patch verifier — No / Yes (crops) / Clean" (leakage table row) | training-recipe | general | maybe |
| thesis_working.tex | 619 | FT4 fine-tune "the feature source for the distilled verifier (§distill_verifier)"; freeze=15 recipe | training-recipe | RGB | maybe (feature source) |
| thesis_working.tex | 630–649 (§Model MRI, label sec:model_mri) | Defines the Model MRI tool: hooks `p3`(s8)/`p5`(s32), ROI-pools to 517-feature embedding (512 pyramid + 5 meta); LDA/ANOVA/AUROC/per-modality z-score/CORAL; "the headline IR-verifier numbers come from a held-out evaluation (`mri/holdout.py`) with CBAM excluded from training" | definition / training-recipe | general | yes-wording |
| thesis_working.tex | 663 | IR detector "Also operates as a grayscale-RGB fallback... ties the strongest dedicated RGB detector on the hardest bird-cluttered clip" | qualitative-claim | gray | maybe |
| thesis_working.tex | 665 (cascade list item 4) | "Confuser verifier: in production, the distilled `mlp_v5` feature-space verifier... cheap enough to run per frame... supersedes a 4-class MobileNetV3 patch verifier (bird/airplane/helicopter/other)... expensive enough to require alert-gating" | definition / threshold(per-frame) | RGB | yes-wording |
| thesis_working.tex | 668 | "the per-frame `mlp_v5` verifier removes that coupling, and the alert-gate analysis of §alert_gate now describes the superseded patch path" | definition / cascade-role | general | yes-wording |
| thesis_working.tex | 670–697 (Fig pipeline tikz) | Pipeline node: "`mlp_v5` verifier (per-frame; ROI-feature reuse)"; greyed node "patch verifier (predecessor; alert-gated)"; caption "production confuser verifier is the distilled `mlp_v5`, which runs per frame..." | definition / figure | general | yes-wording |
| thesis_working.tex | 699 | "a downstream confuser verifier — in production the per-frame distilled `mlp_v5`" | definition | RGB | maybe |
| thesis_working.tex | 701 | NOTE para: production verifier = per-frame `mlp_v5` (RGB) + `mlp_v5_ir_aligned` (IR), both in deployed `MLPVerifier`; full-cascade experiments used the OLDER patch verifier; a re-eval is an open item | definition / cascade-role | general | yes-wording |
| thesis_working.tex | 703 | **Two-verifier fusion (trust-first)** para: "the trusted modality's verifier filters. `reject_both` drops the frame; `trust_rgb` keeps it only if the RGB verifier passes; `trust_ir` keeps if IR passes; `trust_both` recall-first (survives if either passes)... IR verifier can run always-on because the grayscale<->thermal alignment makes it recall-safe rather than recall-eroding. ...unchanged whether router is `robust8` or `sa32`" | definition / composition-order / qualitative-claim | general | yes-wording |
| thesis_working.tex | 709 | Fail-open: "A confuser FP raised by the detector can be vetoed by a downstream stage" (filter = veto/precision tool) | qualitative-claim | general | no |
| thesis_working.tex | 714 | "the patch verifier roughly halves the residual (1.6%->0.8%)" cascade role | qualitative-claim | RGB | yes-number-only |
| thesis_working.tex | 738–747 (§Alert-Gate Cascade, sec:alert_gate) | "describes the deployment of the MobileNetV3 patch verifier, the predecessor that the per-frame `mlp_v5` verifier now supersedes. Alert-gating was forced by the patch verifier's cost..."; rejected `filter_then_classifier` / `classifier_then_filter`; production = `alert_gate_only` | definition / composition-order | general | yes-wording |
| thesis_working.tex | 763 | GUI caption: "the patch verifier runs silently at the alert gate" | cascade-role | general | maybe |
| thesis_working.tex | 850 | preprocessing-sweep note ("no OpenCV preprocessing... improves selcom... not pre-inference filtering") — NOT the filter | (excluded — false hit on "filter") | — | no |
| thesis_working.tex | 872–877 (§sec:ir_xmodal_verifier header) | "Cross-Modal Feature Alignment: a Thermal Verifier from Grayscale Confusers"; "dissolves the data-scarcity blocker that had previously kept any per-frame IR confuser verifier out of the production stack"; "the confuser-rejection signal already lives inside the IR detector" — MRI of `v3b` 517-D | header / definition / training-recipe | IR | yes-wording |
| thesis_working.tex | 889 | "a 517-feature near-linear boundary — and not a confidence threshold — is what separates thermal drones from confusers" (what IR verifier reads) | definition / qualitative-claim | IR | maybe |
| thesis_working.tex | 895 (Fig caption) | "the thermal verifier reads the detector's semantic activations, not its score" | qualitative-claim | IR | maybe |
| thesis_working.tex | 900–901 | "What the IR verifier sees, spatially." — on thermal drone p3/p5 bind to airframe; on held-out CBAM confuser they scatter; "thermal verifier reads the same kind of spatial signature its RGB counterpart does" | qualitative-claim | IR | maybe |
| thesis_working.tex | 913 | "the grayscale->thermal feature gap is a removable affine offset, so a confuser filter transfers across the modality"; per-modality z-score lifts transfer AUROC 0.500->0.919; CORAL 0.707 | definition / training-recipe | gray->IR | yes-wording |
| thesis_working.tex | 924 | "the IR confuser-verifier blocker was confuser data scarcity, not the detector... grayscale-RGB confusers are abundant and, once z-aligned, occupy the same thermal feature space... yields a thermal verifier that is simultaneously confuser-rich and recall-safe" | training-recipe / qualitative-claim | IR/gray | yes-wording |
| thesis_working.tex | 932 | robust6/robust8/sa32 classifier feature-set (context: classifier, not filter) | (context) | general | no |
| thesis_working.tex | 980 | classifier comparison harness "v2 patch verifier" baseline condition | cascade-role | general | yes-number-only |
| thesis_working.tex | 1084 (Full-pipeline verdict) | "the classifier and verifier are complementary and the lowest fire rate needs both" — composition order clf->verifier | qualitative-claim / composition-order | general | maybe |
| thesis_working.tex | 1095–1104 (Table tab:robust6_pipeline) | rows: "verifier only", "classifier->verifier [sa32]/[robust6]", "verifier->classifier [robust6]" | composition-order / cascade-role | general | yes-number-only |
| thesis_working.tex | 1106–1108 (para "Closing the grayscale hole: robust8") | Defines robust8 grayscale fix; validated "classifier->verifier" on `ft4`/`v3b` (NOTE: this paragraph my memory said was deleted but is STILL PRESENT) | composition-order | gray | maybe |
| thesis_working.tex | 1111 | "the RGB `mlp_v5` drone probability separates trust-positive from reject at AUROC 0.949... held out of shipped `robust8`... because feeding the verifier into the classifier couples the two stages" | qualitative-claim | RGB | maybe |
| thesis_working.tex | 1114–1136 (§Patch Verifier, sec:patch_verifier_arch) | Patch-verifier ARCHITECTURE: 4-class (airplane/helicopter/bird/other) MobileNetV3; `p_confuser=max(...)` thresholded at `patch_thr`; `other` = OOD outlet; Mahalanobis secondary; "superseded by the distilled `mlp_v5`"; v1–v4 version history | header / definition / training-recipe | general | yes-wording |
| thesis_working.tex | 1138 | "patch verifier... a recall/safety knob with a tunable operating point, not an unconditional precision booster"; bimodal output | qualitative-claim / threshold | general | maybe |
| thesis_working.tex | 1141–1172 (§Threshold Sweep, sec:patch_threshold) | `patch_thr` sweep; **production patch_thr=0.9 on Svanstrom** "at the elbow where drone recall recovery saturates" | threshold | general | yes-wording (only if patch retired) |
| thesis_working.tex | 1174–1219 (§Catch-Rate Audit, sec:patch_audit) | per-bucket catch/veto; "**airplane gap**" — airplanes 52% catch, median patch prob 0.540; "active wrong classifications... recoverable by retraining the verifier with the specific FP crops as confuser-class data"; "motivation for the distilled `mlp_v5` successor" | qualitative-claim (airplane gap) / training-recipe | RGB | YES-wording (airplane-gap claim) |
| thesis_working.tex | 1221–1222 (§sec:distill_verifier header) | **"Distilled Feature-Space Verifier (`mlp_v5`)"** — the core RGB-filter definition section | header | RGB | yes-wording |
| thesis_working.tex | 1224 | patch verifier "abstains to `other`" on unseen surfaces (SelCom CCTV); "This blind spot motivates a structurally different verifier" | qualitative-claim | RGB | maybe |
| thesis_working.tex | 1227 | **CORE DEFINITION**: "a small MLP consumes the RGB detector's own fused intermediate ROI features — `p3`(stride 8) and `p5`(stride 32)... ROI-pooled to a 512-dim embedding plus 5 detection-metadata scalars (**517 features total**)... trained on... 32,931-detection corpus (19,334 drone + 13,597 confuser)... a trained classifier, not a distance threshold, is required" | definition / training-recipe | RGB | yes-wording |
| thesis_working.tex | 1244–1258 (Table tab:distill_verifier) | `mlp_v5` vs patch v2 on FT4; caption "wins on Svanstrom, SelCom, confuser-only; ties Anti-UAV; regresses on `rgb_dataset`" | qualitative-claim | RGB | yes-number-only |
| thesis_working.tex | 1263–1266 (Fig distill_verifier_bar) | "Distilled feature-space verifier (`mlp_v5`, green) vs MobileNetV3 patch verifier (v2)" | figure | RGB | yes-number-only |
| thesis_working.tex | 1269 | "It ties the patch verifier on Anti-UAV... 46–72x faster per detection at 1–4% pipeline overhead, cheap enough to run per frame rather than alert-gated. Alert-gating it would... cost 4.0 pp Svanstrom F1, so **per-frame operation is the correct deployment choice**" | threshold (per-frame) / qualitative-claim | RGB | yes-wording |
| thesis_working.tex | 1273 | carve-out: "regresses to F1=0.792... recall collapse... structural recall ceiling... distilled `mlp_v5` therefore supersedes the v2 patch verifier as the production confuser verifier... shipped at its full-veto operating point" | qualitative-claim / threshold | RGB | yes-wording |
| thesis_working.tex | 1277–1287 (What the verifier sees + Fig mri_activation) | spatial activation maps; "a 517-feature near-linear boundary suffices... rejects 97% of confuser detections while retaining 98.9% of true drones" | qualitative-claim | RGB | maybe |
| thesis_working.tex | 1290–1349 (§Diagnosing Recall Drop, sec:mlp_recall_drop) | full diagnosis: vetoed drones are OOD-from-confuser; fail-open kNN gate (tau), "promising but not adopted"; "shipped at its full-veto operating point"; "**the verifier rejects them because they do not resemble its training drones, not because they resemble birds**" | qualitative-claim / threshold | RGB | yes-wording |
| thesis_working.tex | 1304 | "the same gate... **backfires** on cluttered Svanstrom" (recall-first/abstain decision) | qualitative-claim | RGB | maybe |
| thesis_working.tex | 1348 | "The same diagnostic pattern recovered the deployable thermal IR verifier (§grayscale_verifier)... MRI statistics... localised it and prescribed the recall-safe fix" | qualitative-claim | general | maybe |
| thesis_working.tex | 1382 | label_reviewer "filtered views" — NOT the confuser filter (false hit) | (excluded) | — | no |
| thesis_working.tex | 1503 / 1519 / 1525 / 1541 | cumulative-suppression captions: "v2 patch verifier", "the patch verifier roughly halves the residual", "patch verifier removes the residual on helicopters... most of it on birds" | qualitative-claim / cascade-role | general | yes-number-only |
| thesis_working.tex | 1545–1552 | cumulative cascade prose: "classifier carrying ~98% of the cumulative suppression and the patch verifier halving the residual" | qualitative-claim | RGB | yes-number-only |
| thesis_working.tex | 1563–1607 (§Roboflow OOD) | "The patch verifier is run as a per-frame filter in this evaluation"; "the patch verifier costs net F1 on every drone setting at imgsz=640"; "**verifier is severely distribution-bound**... bird/airplane/helicopter discrimination does not transfer" | qualitative-claim | RGB | YES-wording (distribution-bound) |
| thesis_working.tex | 1599–1606 (Table tab:ood_rgb_confuser + prose) | "airplane 3.1% — **verifier ineffective on OOD airplanes**"; "bird 50.5% — verifier transfers well to OOD birds"; "the verifier's 'OOD ceiling'" | qualitative-claim (airplane ineffective) | RGB | YES-wording |
| thesis_working.tex | 1621–1628 (Table tab:ood_ir + prose) | "ir_mixed_cbam + patch v2" rows (patch verifier on IR OOD) | qualitative-claim | IR | yes-number-only |
| thesis_working.tex | 1761–1762 (§sec:grayscale_verifier header) | **"From Cross-Modal Transfer to a Deployable Thermal Verifier"** — the core IR-filter definition section; "reverses an earlier conclusion of this project" | header | IR | yes-wording |
| thesis_working.tex | 1764 | "The IR detector's grayscale mode is, for confuser frames, its hallucination mode... fires on grayscale confusers ~20x more often... precisely what makes grayscale a rich, abundant source of confuser detections to harvest" | training-recipe / qualitative-claim | gray | yes-wording |
| thesis_working.tex | 1767 | **CORE IR DEFINITION**: "production checkpoint `mlp_v5_ir_aligned` combines thermal drone detections... with grayscale-harvested confusers, per-modality z-aligned into a single **517-D verifier** that loads in the production `MLPVerifier`. It is **one network with two per-modality input scalers**: thermal `mlp_aligned.pt` and grayscale `mlp_aligned_gray.pt` share the same weights and differ only in the (mu,sigma) standardisation... CBAM... held out of training" | definition / training-recipe | IR | yes-wording |
| thesis_working.tex | 1769–1786 (Table tab:ir_aligned) | "Thermal-deploy verifier (`mlp_v5_ir_aligned`, scaler `mlp_aligned.pt`, **conf=0.40, thr=0.05**)... preserving thermal drone recall... The MobileNetV3 patch verifier on the same held-out CBAM cuts only 7 FP" | threshold / qualitative-claim | IR | yes-wording |
| thesis_working.tex | 1789–1805 (Table tab:ir_aligned_gray) | "same network, with its grayscale scaler... cuts held-out grayscale confuser FP by 96%... matching the dedicated grayscale-only filter (`mlp_v5_gray`) at roughly half its drone-recall cost"; caption "scaler `mlp_aligned_gray.pt`, **conf=0.25, thr=0.25**"; row "dedicated `mlp_v5_gray`" | threshold / definition / qualitative-claim | gray | yes-wording |
| thesis_working.tex | 1807 | "a drone-diversity re-mine (adding ~30k corrective-set thermal drones) made the verifier **recall-safe** on held-out thermal... the fix indicates that loss was OOD-drone coverage, not confuser-reliance" | training-recipe / qualitative-claim (recall-safe) | IR | yes-wording |
| thesis_working.tex | 1810 | "Grayscale is therefore a confuser-harvesting surface and a hardware-fallback detection surface, not a primary one... reverses the project's earlier 'ship no per-frame IR verifier' conclusion" | qualitative-claim | gray/IR | maybe |
| thesis_working.tex | 1830 (Fig caption, mlp_pipeline_placeholder) | "the production per-frame `mlp_v5` verifier has component-level evidence (§distill_verifier) but has not yet been run through this full real-video cascade; the numbers in this section use its patch-verifier predecessor" | cascade-role | general | yes-wording |
| thesis_working.tex | 1885 | "The patch veto removes a further 0.7–1.4 pp of drone segment F1 for a 0.7–2.1 pp cut in confuser FPR" | qualitative-claim | general | yes-number-only |
| thesis_working.tex | 1916 | "the trust classifier and the patch verifier exist precisely to convert detector confidence into modality- and scene-conditional alert decisions" | qualitative-claim | general | maybe |
| thesis_working.tex | 1944–1956 (bird/airplane asymmetry) | "The cascade earns its keep on birds... essentially inert on airplanes"; "patch verifier audit had already shown the verifier is genuinely uncertain on airplane crops (median patch prob 0.540...)... the dedicated remediation is a MobileNetV3 patch-verifier retrain with explicit OOD airplane crops (distinct from the distilled `mlp_v5` verifier... which targets the same crops via feature reuse)" | qualitative-claim (airplane gap) / training-recipe | general | YES-wording |
| thesis_working.tex | 1961 / 1965 | real-video classifier comparison "each with the same RGB / IR / patch verifier configuration" | cascade-role | general | yes-number-only |
| thesis_working.tex | 2031 (Threats internal) | "the patch verifier's four versions (§patch_verifier_arch) were iterated against Svanstrom evaluation results" | qualitative-claim (Svanstrom-bias) | general | maybe |
| thesis_working.tex | 2061–2063 (§Limits — Patch verifier and OOD airplanes) | "The patch verifier (predecessor to the production `mlp_v5`)... suppresses 50–52% of bird FPs... but only 2–7% of OOD airplanes... The distilled `mlp_v5` that supersedes it **inherits the same OOD-airplane exposure gap**, so the recommended remediation applies to both: ...collect the specific Roboflow airplane FP crops as confuser-class training data" | qualitative-claim (airplane gap) / training-recipe | general | YES-wording |
| thesis_working.tex | 2087–2091 (RQ1/RQ2 answers) | RQ1 "v2 patch verifier"; RQ2 "the per-frame `mlp_v5`/`mlp_v5_ir_aligned` verifiers... full-cascade real-video re-run with those production verifiers in the alert-gate slot is a registered open item" | qualitative-claim / cascade-role | general | yes-wording |
| thesis_working.tex | 2113 (Production Stack) | **PRODUCTION STATEMENT**: "The confuser verifier is the distilled `mlp_v5` on the RGB branch and its cross-modal counterpart `mlp_v5_ir_aligned` (one network, two per-modality scalers)... both running per-frame in the deployed `MLPVerifier` and superseding the MobileNetV3 v2 patch verifier... The two verifiers are composed trust-classifier-first (classifier->verifier), the order that is most recall-safe at equal confuser suppression... superseded patch verifier (at patch_thr=0.9 Svanstrom / 0.7 real-video) remains the documented fallback on the photo-style `rgb_dataset` surface" | definition / composition-order / threshold / shipped-weight | general | yes-wording |
| thesis_working.tex | 2116 | carve-out: "**on grayscale the aligned IR verifier over-vetoes**, so the grayscale path is **gated rather than run unconditionally per-frame**" | qualitative-claim (grayscale over-veto) / threshold | gray | YES-wording |
| thesis_working.tex | 2119 (Future work) | "closing the 2–7% OOD-airplane gap... a remediation that applies to the distilled `mlp_v5` as much as to the superseded patch verifier; a grayscale-aware routing gate for the `mlp_v5_ir_aligned` verifier, which still over-vetoes on grayscale" | qualitative-claim (airplane gap, grayscale over-veto) / training-recipe | general | YES-wording |
| thesis_working.tex | 2174 | Appendix model-table head context (patch verifiers / distilled MLP verifiers) | reference | general | no |
| thesis_working.tex | 2342–2366 (Appendix model table) | rows: `confuser_filter4_live`, `patch_v2` "Production patch verifier (fallback under V5 proposal)", `mlp_v5` "PRODUCTION verifier (signed off 2026-05-30, V5 pure_1x8 per-frame)", `mlp_v5_ir_aligned` "PRODUCTION IR verifier. CBAM held-out F1 0.699->0.841, FP 48->13", `mlp_v5_gray`, `mlp_v5_ir_dronediv`, plus superseded variants | glossary / shipped-weight | general | yes-wording |
| thesis_working.tex | 2383 (Glossary CNN) | "The patch verifier is a MobileNetV3-class CNN" | glossary | general | maybe |
| thesis_working.tex | 2390 (Glossary S1/S2/S3) | "S3 = + patch verifier alert gate" | glossary | general | yes-wording |
| thesis_working.tex | 2398 (Glossary v2 patch verifier) | "`confuser_filter4_{rgb,ir}_v2_backup.pt`. The MobileNetV3 patch verifier; the audited predecessor, superseded in production by `mlp_v5`" | glossary / shipped-weight | general | yes-wording |
| thesis_working.tex | 2399 (Glossary mlp_v5) | "The distilled feature-space confuser verifier (§distill_verifier): an MLP on the detector's fused `p3`+`p5` ROI features that supersedes the v2 patch verifier on confuser-rich surfaces and runs per-frame" | glossary / definition | RGB | yes-wording |
| thesis_working.tex | 2400 (Glossary mlp_v5_ir_aligned) | "The cross-modal IR confuser verifier (§grayscale_verifier): one network with two per-modality input scalers, serving the thermal-deploy and grayscale-fallback paths" | glossary / definition | IR | yes-wording |
| thesis_working.tex | 2404 (Glossary patch_thr) | "The decision threshold on p_confuser at the patch-verifier alert gate. Production: 0.9 on Svanstrom-like inputs, 0.7 on real-video" | glossary / threshold | general | yes-wording (if patch retired) |

---

## SECONDARY FILE — docs/thesis_chapters.tex (OLDER SYNCED COPY)

IMPORTANT: `thesis_chapters.tex` (1675 lines) is a STALE predecessor. It contains **NO `mlp_v5`,
NO `mlp_v5_ir_aligned`, NO distilled feature-space verifier, NO Model MRI, NO cross-modal/grayscale
aligned IR verifier** (0 matches for any of those terms). The ONLY confuser-filter content it carries
is the legacy MobileNetV3 **patch verifier**. If a swap touches this file, only patch-verifier prose
exists to update; the new filters were never written into it.

| file | line(s) | what it says (short quote) | category | RGB/IR/gray/general | swap changes this? |
|------|---------|----------------------------|----------|---------------------|--------------------|
| thesis_chapters.tex | 583–601 (§Patch Verifier, sec:patch_verifier_arch) | 4-class MobileNetV3 (airplane/helicopter/bird/other); `p_confuser=max(...)` at `patch_thr`; `other`=OOD outlet; Mahalanobis secondary; v1–v4 history; v2 = `confuser_filter4_{rgb,ir}_v2_backup.pt` | header / definition / training-recipe | general | yes-wording |
| thesis_chapters.tex | 603 | "a recall/safety knob with a tunable operating point, not an unconditional precision booster"; bimodal output | qualitative-claim | general | maybe |
| thesis_chapters.tex | 605–636 (§Threshold Sweep, sec:patch_threshold) | `patch_thr` sweep, production point | threshold | general | yes-number-only |
| thesis_chapters.tex | 638+ (§Catch-Rate Audit, sec:patch_audit) | per-bucket catch/veto; airplane gap (52%, median 0.540) | qualitative-claim (airplane gap) | RGB | yes-wording |
| thesis_chapters.tex | 670, 1522 | (omitted long lines within patch-audit / grayscale §sec:grayscale) — patch-verifier cascade references | qualitative-claim | general | yes-number-only |
| thesis_chapters.tex | 1228 (§sec:grayscale) | grayscale cross-modal DETECTION transfer (detector, not the aligned verifier) | (context) | gray | no |

---

## CLAIMS A SWAP MIGHT FALSIFY

These are qualitative statements the retrained filters could contradict. If the new filters behave
differently, the WORDING (not just numbers) must change. Ordered by exposure.

1. **"Airplane gap" / "verifier ineffective on OOD airplanes" / "essentially inert on airplanes."**
   (working.tex 1218, 1599–1606, 1944–1956, 2061–2063, 2116/2119; chapters.tex 638+). The thesis
   repeatedly states BOTH the patch verifier AND the distilled `mlp_v5` fail on airplanes / large
   in-frame airplanes, and names "retrain with explicit OOD airplane crops" as the remedy. If a
   retrained filter now removes airplanes, every "airplane gap / inert on airplanes / inherits the
   same OOD-airplane exposure gap" sentence is false and the future-work remediation must be deleted.

2. **"On grayscale the aligned IR verifier over-vetoes, so the grayscale path is gated rather than
   run unconditionally per-frame."** (working.tex 2116, 2119; ledger `grayscale-trust-classifier-degrades`).
   If a retrained grayscale scaler is recall-safe, this carve-out and the "grayscale-aware routing
   gate" future-work item must be removed, and the "per-frame always-on" claim extends to grayscale.

3. **"`mlp_v5` regresses on the photo-style `rgb_dataset` split (−11 pp F1) — structural recall
   ceiling; photo-style RGB is routed to the patch verifier as a fallback."** (working.tex 222, 1273,
   1290–1349, 2113). A retrained RGB filter that closes this carve-out would invalidate the entire
   §sec:mlp_recall_drop diagnosis, the fail-open discussion, AND the "patch verifier remains the
   documented fallback" production statement.

4. **"The patch verifier is severely distribution-bound / its discrimination does not transfer to
   OOD."** (working.tex 1586, 1606). A retrained patch verifier (if patch is what's swapped) that
   generalises would falsify the "OOD ceiling" framing.

5. **Composition order: "classifier->verifier (clf->filt) is the most recall-safe at equal confuser
   suppression."** (working.tex 703, 1084, 1095–1104, 2113). NOTE: the user's memory/handover says the
   shipped order was switched to `filt->clf` for the no-reject router, but thesis_working.tex STILL
   states classifier->verifier as the shipped order. A swap that changes the optimal order would
   require rewording here. (Flagged as a possible pre-existing inconsistency, not introduced by the swap.)

6. **"`mlp_v5_ir_aligned` is recall-safe on in-distribution thermal (drone-diversity re-mine fixed
   the recall loss)."** (working.tex 222, 703, 1767, 1807). If a retrained IR filter trades recall,
   the "recall-safe / can run always-on" claim (which underwrites the two-verifier trust-first fusion
   rule at line 703) breaks.

7. **"The thermal/RGB verifier reads the detector's semantic activations (p5), not its confidence
   score — 517-feature near-linear boundary, not a confidence threshold."** (working.tex 889, 895,
   1227, 1277). If the retrained filter is architecturally different (e.g., a different feature set,
   not 517-D p3+p5, or a thresholded score), these mechanism claims and the Model MRI figures change.

8. **"Distilled verifier supersedes the patch verifier on EVERY confuser-rich surface and on BOTH
   modalities" (the headline supersession / novelty claim).** (working.tex 215, 222, 306/316, 1114,
   2113; ledger `mlp-beats-patch-both-modalities`). Any surface where a retrained filter loses to the
   patch verifier would weaken this and the Table tab:related_systems novelty claim.

9. **Operating-point / threshold claims tied to specific values:** RGB P(drone)>=0.25, IR thermal
   thr=0.05/conf=0.40, IR grayscale thr=0.25/conf=0.25, "shipped at its full-veto operating point",
   patch_thr=0.9/0.7. (working.tex 1269, 1273, 1311, 1767, 1769, 1789, 2113, 2404.) A retrained filter
   with a different calibrated threshold (or a non-full-veto operating point) changes these.

10. **"The RGB filter does all the work / IR-thermal contributes little" type readings.** Not found as
    a standalone sentence in `thesis_working.tex` (the memory note about "RGB filter does all the work
    on Svanstrom, IR-thermal nothing" lives in docs/analysis, not the live .tex). Closest live claim:
    "the classifier carries ~98% of cumulative suppression and the patch verifier halves the residual"
    (1545) — a swap that makes the filter carry more of the suppression would shift this attribution.

---

## TOTALS

- **docs/thesis_working.tex (PRIMARY):** 56 definitional/descriptive locations (table rows above).
- **docs/thesis_chapters.tex (STALE COPY):** 6 locations — patch verifier ONLY; contains none of the
  new filters.
- **Total mapped locations:** 62.
- **Filter-definition SECTION headers in working.tex (4):** §sec:distill_verifier (1221, RGB `mlp_v5`),
  §sec:ir_xmodal_verifier (872, IR feature alignment), §sec:grayscale_verifier (1761, deployable IR
  `mlp_v5_ir_aligned`), §sec:patch_verifier_arch (1114, predecessor) — plus §sec:model_mri (630, the
  tool that produces the filter evidence) and §sec:mlp_recall_drop (1290, RGB carve-out diagnosis).
- **Glossary entries (working.tex 2383–2404):** CNN, S1/S2/S3, v2 (patch verifier), `mlp_v5`,
  `mlp_v5_ir_aligned`, `patch_thr` = 6 entries.
- **"Claims a swap might falsify": 10** (airplane gap is the single highest-exposure claim, appearing
  in 5+ distinct passages).
