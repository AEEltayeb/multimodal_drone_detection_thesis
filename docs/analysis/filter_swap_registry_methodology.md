# Confuser-Filter Swap Registry (exhaustive)

Date: 2026-06-17. READ-ONLY scan. Purpose: a complete map of every place the **confuser
filters** (RGB `mlp_v5`, IR `mlp_v5_ir_aligned` = thermal `aligned`/`mlp_aligned` + grayscale
`aligned_gray`/`mlp_aligned_gray`) appear in the live thesis, so a filter swap can update every
number, claim, definition, figure, and glossary entry without corrupting the document.

## Scope note (IMPORTANT)

The task named four chapters (`methodology`, `introduction`, `conclusion`, `appendices`) + `main.tex`.
But the bulk of filter **numbers** live in **`empirical.tex`** (the results chapter, 103 KB — this is the
file the task's "~103KB methodology" and filter metrics actually describe; `methodology.tex` is also
103 KB but holds *design/recipe* prose). `empirical.tex` is the CBAM-table home and is co-regexed by
the audit alongside `methodology.tex`. **Omitting it would corrupt the thesis on swap, so it is included
here.** `related_work.tex` also carries two filter cells. All files scanned:
`introduction.tex`, `methodology.tex`, `empirical.tex`, `related_work.tex`, `conclusion.tex`,
`appendices.tex`, `main.tex`, `figures/fig_pipeline.tex`.

Paths are relative to `docs/thesis_working_distilling_overleaf/`. Absolute repo root:
`C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis`.

Audit backbone: `thesis_eval/_audit_headline_numbers.py`. Its CHECK labels are quoted below per cell.
Cells with no audit CHECK are marked **UN-AUDITED** (a swap can silently break them).

---

## CBAM number — every occurrence (the known discrepancy, resolved)

Canonical = `knowledge/evals.csv` row `ir_aligned_cbam_heldout` (line 122): **bare F1 0.699 / 48 FP /
R 0.967 → aligned F1 0.846 / 15 FP / R 0.917**. The audit (`_audit_headline_numbers.py` lines 112-121)
regexes BOTH the methodology prose (`48 to 15`) and the empirical table (`0.846}$ ... (\textbf{15})`)
and pins both to **15**. The task's "0.841/13-FP" is the **superseded** value from older memory; it does
**NOT** appear anywhere in the current live thesis. **Live thesis is internally consistent at 0.846/15FP**,
but every CBAM occurrence is listed so a swap updates all of them:

| file | line | CBAM value as written | audit CHECK |
|---|---|---|---|
| empirical.tex | 628 | caption: "cuts held-out novel-confuser FPs by **69%**"; "patch ... cuts only **7** FP" | indirectly (table) |
| empirical.tex | 634 | table row: bare `0.547/0.967/0.699 (48)` → aligned **`0.786/0.917/0.846 (15)`**; ΔR/ΔF1 `-0.050/+0.147` | **"CBAM aligned FP (empirical table)" = 15** |
| empirical.tex | 639 | patch on CBAM `0.564/0.883/0.688 (41)` (cut only 7 FP) | UN-AUDITED |
| methodology.tex | 654 | prose: "on the held-out CBAM set it cuts false positives **$48 \to 15$** at no recall cost" | **"CBAM aligned FP (methodology prose)" = 15** |
| methodology.tex | 698 | prose: "cutting CBAM confuser false positives from **$48$ to $15$**" | (same regex target; pinned to 15) |
| appendices.tex | 163 | `tab:models_evaluated`: "held-out CBAM $F1\;0.699\to0.846$, FP $48\to15$, recall-safe" | UN-AUDITED |
| methodology.tex | 169 | defines the 180-frame CBAM probe / 1,775-frame CBAM source (no FP number) | UN-AUDITED |
| appendices.tex | 38 | defines the 180-frame CBAM probe as held-out gate (no FP number) | UN-AUDITED |

Swap action: if filter FP changes, update lines 628/634/639 + 654/698 + 163 **together** (audit only
guards 634 + 654/698). Lines 163 and 639 are UN-AUDITED and are the silent-drift risk.

---

## (1) NUMBERS table

Surface key: SVAN=Svanström, AUV=Anti-UAV, DUT=DUT-AntiUAV, RGBconf/IRconf/GRAYconf=confuser corpora,
RGBtest/IRtest=in-dist test splits, SEL=SelCom, VID=real-video. Filter col: RGB=`mlp_v5`,
IR-th=`mlp_v5_ir_aligned` thermal scaler, gray=`mlp_v5_ir_aligned` grayscale scaler.

### empirical.tex — Table `tab:ablation_svanstrom` (SVAN, filt rows)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 31 | 3043/353/271, P0.896 R0.918 **F1 0.907** | TP/FP/FN/P/R/F1 | SVAN | RGB | tab:ablation_svanstrom | "svan filt_mlp_rgb F1"=0.907 |
| empirical.tex | 32 | 3142/2018/172, **F1 0.742** (filter inert) | " | SVAN | IR-th `aligned` | tab:ablation_svanstrom | "svan filt_mlp_ir F1"=0.742 |
| empirical.tex | 38 | clf→filt[robust8-nr] **F1 0.930** | F1 | SVAN | RGB+IR composed | tab:ablation_svanstrom | "NR svan composed F1"=0.930 |
| empirical.tex | 39 | clf→filt[robust8] **F1 0.949** | F1 | SVAN | composed | tab:ablation_svanstrom | "svan composed F1"=0.9485 |
| empirical.tex | 40 | clf→filt[sa32] F1 0.955 | F1 | SVAN | composed | tab:ablation_svanstrom | UN-AUDITED (sa32 composed) |
| empirical.tex | 41 | **filt→clf[robust8-nr] F1 0.944** (SHIPPED, bold) P0.901 R0.991 | F1/P/R | SVAN | composed | tab:ablation_svanstrom | "NR svan filt->clf F1"=0.944 |
| empirical.tex | 42 | filt→clf[robust8] F1 0.963 | F1 | SVAN | composed | tab:ablation_svanstrom | "svan filt->clf F1"=0.9629 |

### empirical.tex — Table `tab:ablation_antiuav` (AUV, filt rows)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 60 | filt(mlp,RGB) **F1 0.973** | F1 | AUV | RGB | tab:ablation_antiuav | "antiuav filt_mlp_rgb F1"=0.973 |
| empirical.tex | 61 | filt(aligned,IR) **F1 0.973** | F1 | AUV | IR-th | tab:ablation_antiuav | "antiuav filt_mlp_ir F1"=0.973 |
| empirical.tex | 67 | clf→filt[robust8-nr] F1 0.984 | F1 | AUV | composed | tab:ablation_antiuav | "NR antiuav composed F1"=0.984 |
| empirical.tex | 70 | filt→clf[robust8-nr] **F1 0.984** (SHIPPED bold) | F1 | AUV | composed | tab:ablation_antiuav | (covered by NR antiuav composed) |

### empirical.tex — Table `tab:rq3` (RQ3)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 99 | Routed (robust8-nr, filt→clf) SVAN **0.944** / AUV **0.984** | F1 | SVAN+AUV | composed | tab:rq3 | "NR svan filt->clf F1"=0.944 / NR antiuav |

### empirical.tex — Table `tab:ablation_dut` (DUT, filt rows)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 120 | filt(mlp,RGB) **F1 0.726** | F1 | DUT | RGB | tab:ablation_dut | "dut filt_mlp_rgb F1"=0.726 |
| empirical.tex | 121 | filt(`aligned_gray`,IR) **F1 0.725** | F1 | DUT | gray | tab:ablation_dut | "dut filt_mlp_ir F1"=0.725 |
| empirical.tex | 127 | clf→filt[robust8-nr] F1 0.792 | F1 | DUT | composed | tab:ablation_dut | "NR dut composed F1"=0.792 |
| empirical.tex | 130 | **filt→clf[robust8-nr] F1 0.837** (SHIPPED bold) P0.869 R0.808 | F1/P/R | DUT | composed | tab:ablation_dut | "NR dut filt->clf F1"=0.837 |
| empirical.tex | 131 | filt→clf[robust8] F1 0.707 R0.583 | F1/R | DUT | composed | tab:ablation_dut | "DUT filt->clf[robust8] P"=0.900 |

### empirical.tex — Table `tab:ablation_confusers` (confuser fire rates) — CORE FILTER NUMBERS
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 160 | filt only (mlp) **RGB 0.011 / IR 0.237 / gray 0.008** | fire rate | RGBconf/IRconf/GRAYconf | RGB / IR-th / gray | tab:ablation_confusers | "rgbconf mlp fire"=0.0106; "NR ir_conf fire"=0.237; "FIG gray fire@0.25"=0.008 |
| empirical.tex | 162 | filt→clf[robust8-nr] **RGB 0.011 / IR 0.237** (SHIPPED bold) | fire rate | RGBconf/IRconf | composed | tab:ablation_confusers | "NR rgb_conf fire"=0.0106; "NR ir_conf fire"=0.237 |
| empirical.tex | 163 | clf→filt[robust6] RGB 0.0011 / IR 0.179 | fire rate | RGBconf/IRconf | composed | tab:ablation_confusers | "irconf composed r6 fire"=0.1792 |
| empirical.tex | 164 | clf→filt[robust8] RGB 0.0015 / IR 0.217 | fire rate | RGBconf/IRconf | composed | tab:ablation_confusers | "rgbconf composed fire"=0.0015; "irconf composed r8 fire"=0.2167 |
| empirical.tex | 161 | filt only (patch) RGB 0.102 / IR 0.246 / gray 0.204 | fire rate | confusers | patch (predecessor) | tab:ablation_confusers | "rgbconf patch fire"=0.1022 |

### empirical.tex — Table `tab:ablation_solo` (single-modality, filt rows)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 199 | IR test filt(aligned,thermal) P0.943 R0.973 **F1 0.958** | P/R/F1 | IRtest | IR-th | tab:ablation_solo | UN-AUDITED |
| empirical.tex | 204 | RGB test filt(mlp) P0.976 R0.691 **F1 0.809** | P/R/F1 | RGBtest | RGB | tab:ablation_solo | "rgbtest mlp F1"=0.8092 / "NR rgb_test composed F1"=0.809 |
| empirical.tex | 209 | SelCom filt(mlp) P0.950 R0.451 **F1 0.612** | P/R/F1 | SEL | RGB | tab:ablation_solo | "selcom mlp F1"=0.6115 |

### empirical.tex — `tab:per_size` (per-size +filt recall) — see lines 230-247 (filt recall columns)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | ~231-247 | bare→+filt per-bucket recall (e.g. <16px 0.782→0.256, 16-32 0.865→0.447, ≥64 0.956→0.951) | recall | RGBtest/SVAN | RGB | tab:per_size | "SZ rgbtest <16 filt R"=0.2562; "...16-32 filt R"=0.4465; "...>=64 filt R"=0.9506 |

### empirical.tex — `tab:temporal_production` (segment-level, filt rows)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 339 | filt only (mlp) win-F1 0.690, fire 0.226 (−35%) | win-F1/fire | VID | RGB+gray | tab:temporal_production | "NR video composed F1"=0.687 / "video composed r8 F1"=0.56 |
| empirical.tex | 345 | clf→filt[robust8-nr] win-F1 0.687, fire 0.225 (−36%) | win-F1/fire | VID | composed | tab:temporal_production | "NR video composed F1"=0.687; "NR video_conf fire"=0.2252 |
| empirical.tex | 340 | filt only (patch,per-frame) F1 0.801, fire 0.182 | win-F1/fire | VID | patch | tab:temporal_production | UN-AUDITED |

### empirical.tex — `tab:distill_verifier` (mlp_v5 vs patch, per surface) — UN-AUDITED design table
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 589 | SVAN +mlp_v5 **F1 0.869**, halluc **0.037** | F1/halluc | SVAN | RGB | tab:distill_verifier | UN-AUDITED (predecessor-config) |
| empirical.tex | 590 | AUV +mlp_v5 F1 0.985, halluc 0.010 | F1/halluc | AUV | RGB | tab:distill_verifier | UN-AUDITED |
| empirical.tex | 591 | SEL +mlp_v5 F1 0.607, halluc 0.019 | F1/halluc | SEL | RGB | tab:distill_verifier | UN-AUDITED |
| empirical.tex | 592 | rgb_dataset +mlp_v5 **F1 0.792** (carve-out), halluc 0.010 | F1/halluc | RGBtest | RGB | tab:distill_verifier | UN-AUDITED (rgbtest mlp 0.8092 is the Tier-1 cell) |
| empirical.tex | 593 | confuser-only +mlp_v5 halluc **0.008** | halluc | RGBconf | RGB | tab:distill_verifier | UN-AUDITED |

### empirical.tex — `tab:ir_aligned` (CBAM held-out filter table) — see CBAM section above
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 634 | CBAM bare `(48)` → aligned **`0.786/0.917/0.846 (15)`** | P/R/F1/FP | CBAM (IR) | IR-th | tab:ir_aligned | **"CBAM aligned FP (empirical table)"=15** |
| empirical.tex | 635 | ir_dset_final bare 0.965(109) → 0.962(108), ΔR −0.007 | P/R/F1/FP/ΔR | IRtest | IR-th | tab:ir_aligned | UN-AUDITED |
| empirical.tex | 636 | ir_video 0.942(80) → 0.942(80), Δ 0 | " | IR video | IR-th | tab:ir_aligned | UN-AUDITED |
| empirical.tex | 637 | AUV 0.962(68) → 0.962(68), Δ 0 | " | AUV (IR) | IR-th | tab:ir_aligned | UN-AUDITED |
| empirical.tex | 639 | patch on CBAM 0.564/0.883/0.688 (41), cut 7 FP | P/R/F1/FP | CBAM | patch | tab:ir_aligned | UN-AUDITED |

### empirical.tex — `fig:filter_operating` caption (operating-point sweep numbers)
| file | line | value | metric | surface | filter | label | audit CHECK |
|---|---|---|---|---|---|---|---|
| empirical.tex | 649 | RGB @0.25 keeps **95.6%** drone dets at **1.1%** fire; recall-transparent ≥98% below | recall/fire | RGBconf | RGB | fig:filter_operating | "FIG rgb recall@0.25"=0.956; "FIG rgb fire@0.25"=0.011 |
| empirical.tex | 649 | IR-thermal fire flat ≈0.21–0.24 at every threshold | fire | IRconf | IR-th | fig:filter_operating | UN-AUDITED (range) |
| empirical.tex | 649 | gray @0.25 retains **46.7%**; @0.05 0.47→0.62 recall, fire 0.008→0.036 | recall/fire | GRAYconf | gray | fig:filter_operating | "FIG gray recall@0.25"=0.467; "...@0.05"=0.618; "...fire@0.25"=0.008; "...fire@0.05"=0.036 |

### Introduction / Conclusion / Abstract / Methodology — headline restatements of filter numbers
| file | line | value | metric | surface | filter | label/inline | audit CHECK |
|---|---|---|---|---|---|---|---|
| introduction.tex | 50 | SVAN F1 0.742→**0.944**, R 0.948→0.991, P 0.609→0.901; RGBconf fire **30.4%→1.1%** (reject 0.15%); AUV 0.973→0.984; filter 1.3–2.1 ms/det 37–72× | F1/R/P/fire/latency | SVAN/RGBconf/AUV | RGB+IR composed | inline (contribution 1) | "NR svan filt->clf F1"=0.944; "rgbconf bare fire"=0.3035; "NR rgb_conf fire"=0.0106 |
| introduction.tex | 26 | "per-detection MLP confuser filter ... re-reads the **517-dimensional** ROI feature vector" | dim | — | RGB | inline | UN-AUDITED (def) |
| introduction.tex | 53 | MRI: LDA 0.952 RGB / 0.981 IR; AUROC 0.500→0.919 vs CORAL 0.707 | LDA/AUROC | — | both filters | inline (contribution 2) | "MRI ir LDA"=0.981 (RGB 0.952 UN-AUDITED) |
| introduction.tex | 61 | grayscale finding F1 0.580 vs 0.607; clip 0.837 vs 0.840 | F1 | SVAN/VID | gray (detector, not filter) | inline (finding) | "3way gray F1"=0.5796 |
| conclusion.tex | 10 | SVAN 0.742→**0.944** R0.948→0.991 P0.609→0.901; RGBconf **30.4%→1.1%** reject 0.15%; thermal "cuts only **39%**"; AUV 0.973→0.984; 41 FP | F1/R/P/fire | multi | composed | RQ1 | "NR svan filt->clf F1"=0.944; "irconf composed r6 fire"=0.1792 (39% derived) |
| conclusion.tex | 14 | filter "removes **82%** of bare FPs (2,019→353)" SVAN; confuser "835→29 FP" vs patch "835→282"; filter 1.3–2.1 ms 37–404× | FP counts/latency | SVAN/RGBconf | RGB | RQ2 | derived from tab:ablation_svanstrom 353 / confusers 29 |
| conclusion.tex | 30 | production-stack: filters `mlp_v5` (RGB) + `mlp_v5_ir_aligned` (IR); SelCom +10pp F1 at floor 0.05 | naming/F1 | — | both | sec:production_stack | UN-AUDITED (prose) |
| conclusion.tex | 32 | carve-out (i) `mlp_v5` costs **11 pp** F1 on rgb_dataset; (iii) gray aligned **−96.8%** FP | F1/FP | RGBtest/GRAYconf | RGB/gray | carve-outs | UN-AUDITED (−96.8% = 656→21) |
| main.tex (abstract) | 156 | RGBconf "fires on **30.4%**"; SVAN bird "up to **94%**"; retrain collapse 0.961→0.306 | fire/R | RGBconf/SVAN | (detector) | abstract | "rgbconf bare fire"=0.3035 |
| main.tex (abstract) | 159 | SVAN **0.742→0.944** R 0.948→0.991; RGBconf **30.4%→1.1%** (reject 0.15%); AUV 0.973→0.984; stages 0.095 ms / ~2 ms 37–404× | F1/R/fire/latency | multi | composed | abstract | "NR svan filt->clf F1"=0.944; "NR rgb_conf fire"=0.0106 |
| main.tex (abstract) | 162 | MRI LDA 0.952 RGB / 0.981 IR; AUROC 0.500→0.919; gray finding 2.7 pp | LDA/AUROC | — | both | abstract | "MRI ir LDA"=0.981 |
| methodology.tex | 127 | fig caption: detector conf 0.82–0.86 vs filter P(drone) **0.001–0.077**, all suppressed at **0.25** | P(drone) | RGBconf | RGB | fig:confuser_fp_examples | UN-AUDITED (illustrative) |
| methodology.tex | 360 | FT4 R3 "16-pp confuser-hallucination cut" (feature source for the filter) | halluc | — | (detector, feeds filter) | tab:ft4_variants | UN-AUDITED |
| methodology.tex | 473/495/815-818 | MRI verdict: LDA 0.952, FP cut 97.4%, recall ret 98.9%, max ANOVA F 42,346; 5-fold CV F1 0.9857±0.0004; 46–72× / 1.3–2.1 ms; corpus 32,931 (19,334 drone/13,597 conf) | various | — | RGB | fig:mri_report / sec:distill_verifier | UN-AUDITED (in-corpus; not a Tier-1 cell) |
| methodology.tex | 654/665-674 | IR-MRI: LDA **0.981**, max ANOVA F **5,370**, median **256**, halluc **1.8%**, FP cut **89%**, recall ret **99.7%**, 14,697 drone/1,386 conf | various | — | IR-th | tab:ir_mri_sep | "MRI ir LDA"=0.981; "MRI ir maxF"=5370; "MRI ir medianF"=256; "MRI ir halluc"=0.018; "MRI ir fp_cut"=0.89; "MRI ir recall_ret"=0.997; "MRI ir n_drone"=14697; "MRI ir n_confuser"=1386 |
| methodology.tex | 689/694 | gray→thermal AUROC: raw 0.500 / CORAL 0.707 / z-score **0.919** / ceiling 0.974 | AUROC | — | IR (alignment) | tab:gray_thermal_auroc | UN-AUDITED (canonical = ledger gray-thermal-alignable) |
| methodology.tex | 696 | gray-only net over-vetoes (recall 0.55→0.16) vs aligned recall-safe (0.55→0.51) | recall | gray SVAN | gray | sec:ir_xmodal_verifier | UN-AUDITED |
| methodology.tex | 698-699 | aligned ΔR −0.007 IRtest / 0.000 video / 0.000 AUV; CBAM 48→15 | ΔR/FP | IR | IR-th | sec:ir_xmodal_verifier | CBAM=15 (audited); ΔR UN-AUDITED |
| empirical.tex (related) | — | (related_work) | | | | | |
| related_work.tex | 116 | "This thesis (confuser filter, @640) ... **1.1% fire**" | fire | RGBconf | RGB | tab:numerical_comparison | "NR rgb_conf fire"=0.0106 |
| related_work.tex | 119 | "best ablation, robust6 ... **29.4%→17.9%** fire" | fire | IRconf | composed | tab:numerical_comparison | "irconf bare fire"=0.2943; "irconf composed r6 fire"=0.1792 |
| empirical.tex | 469/483-486 | robust6 vs sa32 design table: clf→filt[robust6] F1 0.9957 / fire 0.066; filt→clf 0.057; filter only 0.964/0.114 | F1/fire | SVAN/zoo | composed | tab:robust6_pipeline | UN-AUDITED (predecessor 5k-strided) |
| empirical.tex | 510 | predecessor zoo: fnfn 52.1%→1.6%(0.8%); sa32 20.5%(10.3%) | fire | zoo | patch composed | inline | UN-AUDITED |
| empirical.tex | 769 | gray "IR hallucinates ~20× more on gray confusers than thermal (37.2% vs 1.8%)" | halluc | GRAYconf | (detector→filter source) | inline | UN-AUDITED |
| empirical.tex | 772 | gray aligned filter cuts FP **96.8%** (656→21); over-vetoes drones (recall ≤0.27 even @0.02) | FP/recall | GRAYconf/gray SVAN | gray | inline | UN-AUDITED |

---

## (2) DEFINITIONS / TRAINING-RECIPE / THRESHOLD prose

| file | line | what it says (short) | category | swap = wording or number? |
|---|---|---|---|---|
| methodology.tex | 513 | filter def: "distilled `mlp_v5` feature-space filter ... reuses ROI features ... per frame; supersedes 4-class MobileNetV3 patch" | definition | wording (if architecture/name changes) |
| methodology.tex | 525 | "production filter is a confidence-gated veto: survives only if MLP assigns P(drone) ≥ production threshold"; fail-open rejected | def + threshold | wording + number (threshold) |
| methodology.tex | 535 | **two-filter fusion (trust-first)** rule: reject_both drop / trust_rgb→RGB filter / trust_ir→IR filter / trust_both per-modality recall-first; IR filter = one net two scalers | definition (fusion) | wording |
| methodology.tex | 541 | deferred suppression; SelCom 0.591→0.692 with filter at floor 0.05 | rationale + number | number |
| methodology.tex | 649-700 | **sec:ir_xmodal_verifier** — full IR-filter build recipe: 517-D, one net two scalers (thermal ≥0.05 / gray ≥0.25), z-align, ~30k drone re-mine, CBAM held out | recipe | wording + numbers |
| methodology.tex | 696 | "genuinely ONE shipped network, not two; separate grayscale-only net was rejected" | definition | wording |
| methodology.tex | 798-806 | **patch filter** (predecessor) def: 4-class MobileNetV3-Small, 45,917 patches, p_confuser=max(...)≥patch_thr, `other` OOD outlet, v2 shipped | definition (predecessor) | wording |
| methodology.tex | 809-819 | **sec:distill_verifier (`mlp_v5`)** — RGB filter recipe: p3+p5 ROI → 512+5=517-D; 32,931-det corpus; LDA ~95%; PCA silhouette 0.067; 5-fold CV F1 0.9857; 46–72× faster | recipe | wording + numbers |
| methodology.tex | 12 (empirical) | filter threshold def: mlp_v5 (RGB) 0.25; mlp_v5_ir_aligned thermal 0.05 / grayscale 0.25; "lower bar keeps more"; thermal 0.05 most permissive (recall-safe, not aggressive) | threshold def | wording + numbers |
| empirical.tex | 280 | floors: rgb_conf 0.25, ir_conf 0.40 "exist to keep bare FP in check ... with FP control delegated to filter, that reason weakens" | rationale | wording |
| empirical.tex | 287 | "filter is recall-transparent at every floor; lowering floor converts directly into recall" (SelCom sweep) | threshold claim | wording (could falsify) |
| empirical.tex | 628 | thermal scaler `conf=0.40, thr=0.05` | threshold | number |
| empirical.tex | 352 | temporal replay: `patch_thr=0.7, mlp rgb@0.25/gray@0.25` | threshold | number |
| appendices.tex | 268 | glossary `mlp_v5`: distilled p3+p5 ROI MLP, supersedes patch, per-frame | definition | wording |
| appendices.tex | 269 | glossary `mlp_v5_ir_aligned`: one net, two per-modality scalers, thermal+grayscale | definition | wording |
| appendices.tex | 273 | glossary `patch_thr`: threshold on p_confuser; production 0.9 SVAN-like, 0.7 real-video | threshold def | number |
| methodology.tex | 388 | "mlp_v5 filter of Section ... distinct from the V5 IR-detector regression" (disambiguation) | definition | wording |
| methodology.tex | 262 | unified cache stores 517-D fused p3+p5 ROI vector "the MLP confuser filter consumes (float32: float16 vetoes everything)" | recipe constraint | wording |

---

## (3) FIGURES / TABLES with filter content

| file | line | \label | caption (short) | generating script |
|---|---|---|---|---|
| methodology.tex | 124-129 | fig:confuser_fp_examples | ft4 hallucinations + filter P(drone) 0.001–0.077, suppressed @0.25 | thesis_eval/gen_dataset_figures.py |
| methodology.tex | 477-493 | fig:mri_report | MRI verdict block (RGB filter): LDA 0.952, FP cut 97.4%, recall 98.9% | mri/cli.py → mri/results/v5_report_regen/report.md |
| methodology.tex | 518-523 | fig:pipeline | pipeline diagram; production filter = `mlp_v5` per-frame, patch greyed | figures/fig_pipeline.tex (pre-rendered PDF) |
| methodology.tex | 528-533 | fig:confuser_problem | filter reads ROI, P(drone)=0.00 bird / 0.96 drone | thesis_eval/gen_dataset_figures.py |
| methodology.tex | 657-674 | tab:ir_mri_sep | IR-MRI separability (LDA 0.981, F 5370, halluc 1.8%) | mri/cli.py → mri/results/ir_v3b_report/stats.json |
| methodology.tex | 679-693 | tab:gray_thermal_auroc | gray→thermal AUROC 0.500/0.707/0.919/0.974 | mri/modality_align.py |
| empirical.tex | 19-45 | tab:ablation_svanstrom | SVAN full ablation (filt rows) | thesis_eval/pipeline_eval_unified.py |
| empirical.tex | 48-74 | tab:ablation_antiuav | AUV full ablation (filt rows) | thesis_eval/pipeline_eval_unified.py |
| empirical.tex | 89-102 | tab:rq3 | RQ3 routed vs solo | thesis_eval/pipeline_eval_unified.py |
| empirical.tex | 108-134 | tab:ablation_dut | DUT ablation (filt rows incl. aligned_gray) | thesis_eval/pipeline_eval_unified.py (results_dut) |
| empirical.tex | 147-167 | tab:ablation_confusers | confuser fire rates (CORE filter table) | thesis_eval/pipeline_eval_unified.py |
| empirical.tex | 176-181 | fig:pipeline_ablation | ablation-at-a-glance (drawn for robust8; note nr=1.1%) | thesis_eval/gen_ablation_figure.py |
| empirical.tex | 188-213 | tab:ablation_solo | solo-surface (filt mlp/aligned/patch rows) | thesis_eval/pipeline_eval_unified.py |
| empirical.tex | 224-247 | tab:per_size | per-size bare→+filt recall | thesis_eval/notes_round1_replays.py |
| empirical.tex | ~330-352 | tab:temporal_production | segment-level filt rows | thesis_eval/temporal_replay.py |
| empirical.tex | 307-319 | tab:speed | filter latency mlp_v5 1.3–2.1 ms 37–72× | eval/bench_speed.py |
| empirical.tex | 549-564 | tab:patch_audit | patch v2 per-bucket catch @0.5 | (predecessor; eval) |
| empirical.tex | 567-573 | fig:patch_catchbar | patch catch bar vs 0.90 bar | fig8_patch_catchbar |
| empirical.tex | 579-596 | tab:distill_verifier | mlp_v5 vs patch per surface | eval/eval_v4_vs_patch.py |
| empirical.tex | 599-605 | fig:distill_verifier_bar | mlp_v5 vs patch vs bare bars | fig8_distill_verifier (from knowledge/evals.csv) |
| empirical.tex | 615-620 | fig:failopen_expanded | fail-open frontier (why not adopted) | eval/failopen_expanded_ref.py |
| empirical.tex | 626-642 | tab:ir_aligned | **CBAM held-out filter table (0.846/15FP)** | mri_train_aligned / eval_run_aligned_full |
| empirical.tex | 652-657 | fig:filter_operating | 3-panel P(drone) sweep, shipped points 0.25/0.05/0.25 | eval/filter_operating_sweep.py |
| empirical.tex | 665-672 | fig:mri_stats | FT4 MRI LDA/ANOVA (filter feature space) | mri/cli.py |
| empirical.tex | 678-685 | fig:mri_activation | "brain scan" of neurons the filter reads | mri |
| empirical.tex | 691-697 | fig:ir_gray_align | gray↔thermal alignment bars (filter transfer) | mri/modality_align.py |
| related_work.tex | 95-122 | tab:numerical_comparison | confuser filter 1.1% fire; robust6 29.4→17.9% | thesis_eval/results/tier1_results.json |
| related_work.tex | ~76-86 | (architectural map table) | "per-frame distilled (mlp_v5) confuser filter" | — |
| figures/fig_pipeline.tex | 17 | (TikZ) | "confuser filter (MLP; per-frame; ROI-feature reuse)" node | self (pdflatex) |

---

## (4) QUALITATIVE CLAIMS A SWAP COULD FALSIFY

Each is a verbal claim about filter *behaviour*; a swap that changes the behaviour requires a prose
rewrite, not just a number. Quote — file:line — why.

1. **Thermal filter inert on Svanström** — "filt only (`aligned`, IR) ... 3142 2018 172 ... 0.742"
   (identical to bare) — empirical.tex:32 + reading at empirical.tex:81. Memory `ir_mlp_necessity`:
   on Svanström the RGB filter does all the work, IR-thermal nothing. A swap that makes the IR filter
   fire on Svanström breaks this row AND the "filter suppresses the FPs / reject is the complementary
   net" RQ2 reading.

2. **Airplane hole / "thermal confusers resist"** — "the best composition (robust6, 0.179) cuts fire
   by only **39%**" + "airplane-dominated (76%)" — empirical.tex:173; restated conclusion.tex:10
   ("cuts only 39%"), empirical.tex:649 ("IR-thermal fire-rate is threshold-insensitive ... airplane
   coverage gap"), empirical.tex:802 + 808 (airplane gap = data not architecture), methodology.tex:802.
   A better IR filter on airplanes falsifies "every filter generation suppresses airplanes worst".

3. **Grayscale over-veto / recall ≤0.27** — "over-vetoes drones on grayscale Svanström ... no operating
   point preserves useful recall (recall ≤0.27 even at threshold 0.02)" — empirical.tex:772; gray scaler
   @0.25 retains only 46.7% (empirical.tex:649); carve-out (iii) "confuser-suppression tool, not
   recall-safe; gated there; grayscale detection runs unfiltered" — conclusion.tex:32. Methodology:
   gray-only net 0.55→0.16 vs aligned 0.55→0.51 (methodology.tex:696).

4. **RGB filter recall-safe / recall-transparent** — "RGB `mlp_v5` sits on a flat high-recall shoulder:
   @0.25 keeps 95.6% ... stays recall-transparent (≥98%) below" — empirical.tex:649; "filter is
   recall-transparent at every floor" SelCom — empirical.tex:287; "raises recall (0.948→0.991)"
   — introduction.tex:50, conclusion.tex:10. A swap that loses RGB recall breaks RQ1's headline claim.

5. **"Filter reads p5 activations, not confidence"** — "the filter reads *what the detector saw*, not
   *how sure it was*"; strongest discriminator a p5 channel F=42,346, 4× the confidence scalar
   — empirical.tex:675 + fig:mri_stats:669; methodology.tex:495 (confidence ranks 6th, F=10,696);
   appendices.tex:199 (top features p5 ch=154 ... meta:conf). A retrained filter on different features
   falsifies the "p5 not confidence" through-line and the MRI brain-scan figure.

6. **Composition order (filt→clf shipped, recall-safe)** — "the shipped filter-then-classify keeps the
   router's recall (R=0.991, F1=0.944); classify-then-filter trades a little recall (F1=0.930), within
   1.5 pp" — conclusion.tex:14; empirical.tex:83; "no-reject router never vetoes a confuser frame, so
   the two orders give *identical* confuser fire" — empirical.tex:83/170/149. A filter swap that changes
   composition sensitivity rewrites the production-order argument.

7. **mlp_v5 beats patch ~10× on confusers, 37–72× faster** — "the per-frame mlp_v5 cuts FP 835→29 where
   the patch CNN manages 835→282 ... ~10× stronger on the surface that matters" — empirical.tex:170;
   "835→29 FP, where patch managed 835→282" — conclusion.tex:14; latency 37–72× — empirical.tex:316.

8. **The rgb_dataset carve-out (−11 pp, OOD coverage not overlap)** — "mlp_v5 costs 11 pp F1 on the
   photo-style rgb_dataset split, an OOD coverage gap ... patch filter remains the documented fallback"
   — conclusion.tex:32; full diagnosis empirical.tex:607-613 (vetoed drones smaller, centroid 16.5 vs
   11.1, not low-confidence Δ0.000); tab:distill_verifier:592 (0.792). A swap that closes the carve-out
   removes a stated carve-out and the patch fallback rationale.

9. **Production-stack weight names (shipped filters)** — "per-frame filters `mlp_v5` (RGB) and
   `mlp_v5_ir_aligned` (IR), the latter one network with two per-modality input scalers" —
   conclusion.tex:30; appendices.tex:162-164 (tab:models_evaluated production rows); fig_pipeline.tex:17.
   A weight swap must rename these or they cite a retired weight.

10. **"Genuinely one network, two scalers"** — "This is genuinely *one* shipped network, not two; a
    separately trained grayscale-only filter was the rejected alternative" — methodology.tex:696;
    "one network with two per-modality input scalers" repeated methodology.tex:535/818, conclusion.tex:30,
    appendices.tex:163/269. A two-net swap falsifies this architectural claim everywhere.

11. **Fail-open rejected** — "A fail-open filter variant that releases uncertain detections was evaluated
    and rejected" — methodology.tex:525; empirical.tex:612 (precision 0.887→0.631 on Svanström),
    fig:failopen_expanded:615. A swap changing the veto policy rewrites this.

12. **Grayscale-harvested confusers are what made the IR filter trainable** — "grayscale-harvested
    confusers, aligned into thermal feature space, are what made the thermal filter trainable at all"
    — conclusion.tex:25; main.tex:162; methodology.tex:696; empirical.tex:769. Tied to the
    gray-thermal AUROC alignment claim (tab:gray_thermal_auroc).

13. **−96.8% grayscale FP cut (656→21)** — empirical.tex:772, conclusion.tex:32, empirical.tex:645.
    A grayscale-scaler swap changes this exact count.

---

## (5) GLOSSARY / NOMENCLATURE entries (appendices.tex)

| line | entry | content |
|---|---|---|
| 251 | CNN | "The patch filter is a MobileNetV3-class CNN" |
| 257 | IoP | (scoring; not filter but used by filter eval rows) |
| 267 | v2 (patch filter) | `confuser_filter4_{rgb,ir}_v2_backup.pt`; MobileNetV3 patch, predecessor, superseded by `mlp_v5` |
| 268 | `mlp_v5` | "distilled feature-space confuser filter ... MLP on detector's fused p3+p5 ROI features ... supersedes v2 patch ... per-frame" |
| 269 | `mlp_v5_ir_aligned` | "cross-modal IR confuser filter ... one network with two per-modality input scalers, serving thermal-deploy and grayscale-fallback paths" |
| 273 | `patch_thr` | "decision threshold on p_confuser at the patch-filter alert gate. Production: 0.9 SVAN-like, 0.7 real-video" |
| (appendices.tex 162-165) | tab:models_evaluated | production rows `mlp_v5`, `mlp_v5_ir_aligned` (0.699→0.846, 48→15), comparison `mlp_v5_gray` (~96% FP cut), `patch_v2` |

Note: glossary `mlp_v5_ir_aligned` (line 269) forward-references `Section~\ref{sec:grayscale_verifier}`
(empirical.tex:623), while the build appendix (line 163) references the methodology build section.

---

## TOTALS

- Files scanned: **8** (`introduction`, `methodology`, `empirical`, `related_work`, `conclusion`,
  `appendices`, `main.tex`, `figures/fig_pipeline.tex`).
- **NUMBERS** rows catalogued: **~70** filter-bearing cells (tables + inline restatements) across the
  files; the CORE filter table is `tab:ablation_confusers` (empirical.tex:147-167).
- **DEFINITIONS / recipe / threshold** prose blocks: **~20**.
- **FIGURES / TABLES** with filter content: **30** labelled floats.
- **QUALITATIVE claims a swap could falsify: 13.**
- **GLOSSARY** filter entries: **6** (+ tab:models_evaluated production/comparison rows).
- **CBAM occurrences: 8** (2 audited at FP=15; 6 UN-AUDITED). Live thesis consistent at **0.846 / 15 FP**;
  the "0.841 / 13 FP" from older notes does NOT appear in the live thesis.

### Audit-coverage warning (the swap risk surface)
Audited filter cells (guarded by `_audit_headline_numbers.py`): the Tier-1 NR composed/filt-only cells,
the split `filt_mlp_rgb/ir` cells, the `fig:filter_operating` sweep numbers, the IR-MRI table, and the
**two** CBAM restatements. **UN-AUDITED filter content a swap can silently break:** `tab:distill_verifier`
(empirical.tex:579-596, predecessor-config mlp_v5-vs-patch F1/halluc), `tab:ir_aligned` non-CBAM rows
(635-639), `tab:gray_thermal_auroc` (methodology.tex:689), the MRI verdict figure (fig:mri_report),
all threshold prose (0.25 / 0.05 / patch_thr 0.7-0.9), the production-stack prose (conclusion.tex:30-32),
all glossary entries, the `tab:robust6_pipeline` predecessor table, and the grayscale −96.8% / 656→21
counts. Update these by hand on any swap — the audit will not catch a stale value.
