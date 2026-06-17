# Filter-swap edit plan — introduction / related_work / conclusion / appendices / main (abstract)

Date: 2026-06-18. READ-ONLY scan; **no `.tex`/`.py` edited**. Production filters FINAL:
RGB `mlp_v5_v4` @0.25 · IR thermal-only `mlp_aligned_thermalonly` @0.05 · IR grayscale `mlp_aligned_gray_balanced` @0.25.

Numbers source = `ES_Drone_Detection/filter_PRODUCTION_swap_tables.md` (shipped→candidate, every cell).
Audit constants = `thesis_eval/_audit_headline_numbers.py` (CHECK labels per
`docs/analysis/filter_swap_registry_methodology.md`). Held-out provenance =
`docs/analysis/2026-06-18_filter_provenance_train_heldout.md`.

**Scope rules honoured.** (1) Tables + eval surfaces DON'T change; headline numbers take the swap-map
value on the SAME surface. (2) Held-out test-split results are NOT added in these chapters (they go only in
the methodology/empirical filter sections); but where a headline number is one of the **leaky cache** numbers
(ir-confuser fire, bird FP), it is FLAGGED below to reference the held-out figure and stay consistent.
(3) Filters FINAL. (4) Flipped claims reframed: confuser suppression now stronger; thermal-airplane hole
largely closed; RGB bird FP fixed; IR = two heads (retire "one net two scalers").

**Δ direction key:** the per-frame **RGB filter** headline is `filt→clf[robust8-nr]` Svanström F1
**0.9439→0.9459** (+0.002, BETTER) and rgb_confuser `filt-only` fire **0.0106→0.0144** (+0.0038, slightly
WORSE on the in-sample confuser cache but bird held-out is far better — see flags). The big move is **IR
thermal confuser fire 0.237→0.0278** (cache) / **0.1792→0.0192** (robust6 cache) — these are the leaky
cache cells whose honest held-out replacements are **shipped 90→cbam 22** (IR_confusers val/test).

---

## FILE 1 — `main.tex` (Abstract)  [`docs/thesis_working_distilling_overleaf/main.tex`]

| file | line | section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|---|
| main.tex | 156 | Abstract ¶1 | "fires on $30.4\%$ of an out-of-distribution confuser corpus and on up to $94\%$ of bird-only Svanstr\"om… frames" | NO CHANGE (bare-detector numbers, not filter; 30.4% / 94% are detector fire) | none | neutral | audited: "rgbconf bare fire"=0.3035 (unchanged) |
| main.tex | 156 | Abstract ¶1 | "collapses small-drone recall ($0.961 \to 0.306$)" | NO CHANGE (detector retrain, not filter) | none | neutral | audited (retrainedv2-recall-collapse, unchanged) |
| main.tex | 159 | Abstract ¶2 | "lifts Svanstr\"om paired drone F1 from $0.742$ to **$0.944$** while \emph{raising} recall ($0.948 \to 0.991$)" | "…to **$0.946$** while raising recall ($0.948 \to 0.991$)" | number | better | **audited CHECK "NR svan filt->clf F1" 0.9439→0.9459** |
| main.tex | 159 | Abstract ¶2 | "cuts frame fire from $30.4\%$ to **$1.1\%$** (a router variant that adds a reject class reaches **$0.15\%$** composed…)" | KEEP **1.1%** (rounds from 0.0144; was 0.0106→0.011, now 0.0144→**1.4%** if 1-dp). Decision: write **$1.4\%$**; reject-class robust8 **$0.11\%$** (3→2633) | number | worse (in-sample cache; bird held-out better) | **audited "NR rgb_conf fire" 0.0106→0.0144**; robust8 "rgbconf composed fire" 0.0015→0.0011. **FLAG:** rgb_confuser is in-sample on bird train; the honest bird story is held-out bird.v1i TEST shipped 91→**30** — keep the abstract on the cache value (table-consistent) but ensure §verifier_results carries the held-out 30. |
| main.tex | 159 | Abstract ¶2 | "on the saturated Anti-UAV control it does no harm ($0.973 \to **0.984**$)" | NO CHANGE (0.9842 rounds to 0.984) | none | neutral | audited "NR antiuav composed F1" 0.9841→0.9842 (rounds same) |
| main.tex | 162 | Abstract ¶3 | "aligned grayscale into thermal feature space (transfer AUROC $0.500 \to 0.919$)… **grayscale-harvested confusers**… are what made the thermal confuser filter trainable at all" | KEEP (alignment claim holds); OPTIONAL reframe of "the thermal confuser filter" → "the thermal confuser filter (now a thermal-native CBAM head)" if abstract should foreshadow two-head IR | wording (optional) | neutral/better | un-audited (prose) → `2026-06-18_..._train_heldout.md §2` |
| main.tex | 162 | Abstract ¶3 | "(LDA separability $0.952$ RGB, $0.981$ IR)" | NO CHANGE (MRI feature-space, filter-corpus-independent) | none | neutral | audited "MRI ir LDA"=0.981 |

> **Abstract note:** the only forced numeric edits are line 159 (Svan F1 0.944→**0.946**, confuser
> 1.1%→**1.4%**, reject 0.15%→**0.11%**). The thermal-airplane "closed" reframe does NOT belong in the
> abstract (abstract never quotes the 39% thermal number — that lives in conclusion/empirical).

---

## FILE 2 — `introduction.tex`  [`…/chapters/introduction.tex`]

| file | line | section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|---|
| introduction.tex | 8 | Background | "raising only 41 false positives across those 4{,}000 frames, and its hallucination rate on its composite RGB test corpus is 2.8\% of frames" | NO CHANGE (bare detector, not filter) | none | neutral | audited (antiuav bare; rgb_dataset bare 0.028) |
| introduction.tex | 15 | Problem | "fires on 30.4\% of frames, with per-category rates of 39.0\%…" / Svanström "94.4\% (birds)…" | NO CHANGE (bare detector fire) | none | neutral | audited "rgbconf bare fire"=0.3035 |
| introduction.tex | 23 | Problem | "thermal detector leads (own-GT F1 $0.940$ vs … $0.607$)… Anti-UAV ($0.985$ vs $0.961$)" | NO CHANGE (bare detectors, RQ3 surface) | none | neutral | un-audited detector cells (unchanged by filter) |
| introduction.tex | 50 | Contribution 1 | "lifts Svanstr\"om paired drone $F1$ from $0.742$ (bare) to **$0.944$** while raising recall ($0.948 \to 0.991$) and precision ($0.609 \to 0.901$)" | "…to **$0.946$**…" (precision 0.901→**0.905** per swap-map P 0.9015→0.9052; recall 0.991 unchanged) | number | better | **audited "NR svan filt->clf F1" 0.9439→0.9459**; P 0.9015→0.9052 (un-audited inline, source swap-map svan filt→clf[r8-nr]) |
| introduction.tex | 50 | Contribution 1 | "the per-frame filter cuts frame fire from $30.4\%$ to **$1.1\%$** (a router variant that adds a \texttt{reject} class reaches **$0.15\%$** composed…)" | "…to **$1.4\%$** (…reaches **$0.11\%$** composed…)" | number | worse (in-sample cache) | **audited "NR rgb_conf fire" 0.0106→0.0144**; robust8 0.0015→0.0011 (2 frames). **FLAG:** in-sample bird cache; held-out bird.v1i TEST shipped 91→**30** belongs in §verifier_results, keep intro on cache value. |
| introduction.tex | 50 | Contribution 1 | "Anti-UAV control the pipeline does no harm ($F1\;0.973 \to **0.984**$…)" | NO CHANGE | none | neutral | audited 0.9841→0.9842 (rounds same) |
| introduction.tex | 50 | Contribution 1 | "filter $1.3$--$2.1$~ms per detection ($37$--$72\times$ faster…)" | NO CHANGE (latency; weight architecture identical 517-D MLP) | none | neutral | audited speed rows (architecture unchanged) |
| introduction.tex | 26 | system para | "per-detection \emph{MLP confuser filter} (a small MLP that re-reads the 517-dimensional ROI feature vector…)" | NO CHANGE (517-D holds for v4 + thermal-only; provenance §0 confirms 517-D) | none | neutral | un-audited def; confirmed `…_train_heldout.md §0` (517-D both) |
| introduction.tex | 53 | Contribution 2 | "drone/confuser linear separability measured 0.952 (RGB) and 0.981 (IR)… AUROC from chance ($0.500$) to $0.919$… CORAL… $0.707$. Its output trains the production confuser filters automatically" | NO CHANGE (MRI separability on detector features; "trains the production filters" still true) | none | neutral | audited "MRI ir LDA"=0.981 |
| introduction.tex | 61 | Findings | grayscale finding "$F1$ $0.580$ vs $0.607$… clip… $0.837$ vs $0.840$" | NO CHANGE (detector grayscale finding, not filter) | none | neutral | audited "3way gray F1"=0.5796 |

---

## FILE 3 — `related_work.tex`  [`…/chapters/related_work.tex`]

| file | line | section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|---|
| related_work.tex | 116 | tab:numerical_comparison | "**This thesis (confuser filter, @640)** … **1.1\% fire**" (rgb_confuser, filt-only mlp) | "**1.4\% fire**" | number | worse (in-sample cache) | **audited "NR rgb_conf fire" 0.0106→0.0144** (RGB filt-only). |
| related_work.tex | 119 | tab:numerical_comparison | "**best ablation, \texttt{robust6}, @640** … **29.4\% $\to$ 17.9\% fire**" (ir_confusers, clf→filt[robust6]) | "**29.4\% $\to$ 1.9\% fire**" | number | **better (much stronger suppression)** | swap-map `ir_confusers clf→filt[robust6]` fire **0.1792→0.0192**; bare 0.2943 (audited "irconf composed r6 fire" 0.1792→0.0192; "irconf bare fire"=0.2943). **FLAG — LEAKY:** the `ir_confusers` cache = IR train split. The honest held-out figure is **IR_confusers val/test shipped 90→cbam 22** (`…_train_heldout.md §2.4`). RULE says table surface stays; but since THIS is a headline comparison-table cell against published baselines, recommend either (a) keep cache 1.9% with a footnote pointing to the held-out §ir_xmodal_verifier number, or (b) flag for user: the comparison row should arguably cite the held-out suppression, not the leaky cache. Stay consistent with whatever the filter section states. |
| related_work.tex | 91 | prose | "(trust classifier $+$ confuser filter, run per frame or at the alert gate)" | NO CHANGE (architecture prose) | none | neutral | un-audited (prose) |
| related_work.tex | 104/107/110/113 | tab:numerical_comparison | baseline RGB/IR detector P/R/F1 rows (0.950 / 0.961 / 0.985 / 0.961) | NO CHANGE (bare detectors) | none | neutral | un-audited detector cells (unchanged) |

---

## FILE 4 — `conclusion.tex`  [`…/chapters/conclusion.tex`]

| file | line | section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|---|
| conclusion.tex | 10 | RQ1 | "lifts drone $F1$ from $0.742$… to **$0.944$**, with recall rising… to $0.991$ and precision… to $0.901$" | "…to **$0.946$**… precision… to **$0.905$**" (recall 0.991 unchanged) | number | better | **audited "NR svan filt->clf F1" 0.9439→0.9459**; P 0.9015→0.9052 (swap-map) |
| conclusion.tex | 10 | RQ1 | "the per-frame filter reduces fire to **$1.1\%$**; the reject-class \texttt{robust8} ablation drives it to **$0.15\%$** (four frames in $2{,}633$…)" | "…reduces fire to **$1.4\%$**; …drives it to **$0.11\%$** (**three** frames in $2{,}633$…)" | number | worse (in-sample cache) | **audited "NR rgb_conf fire" 0.0106→0.0144**; robust8 "rgbconf composed fire" 0.0015→0.0011 (FP 4→**3**). **FLAG:** held-out bird.v1i TEST 91→30 lives in §verifier_results. |
| conclusion.tex | 10 | RQ1 | "The thermal-confuser surface is the counterpoint: the best composition **cuts only $39\%$** there, an **airplane-dominated gap recorded as the pipeline's weakest front**." | **REWRITE (flipped claim):** "The thermal-confuser surface, the pipeline's old weakest front, is **largely closed by the thermal-native IR filter**: the best composition now **cuts $\sim$93\%** of thermal-confuser fire ($29.4\%\to1.9\%$), and airplanes — formerly the resistant class — no longer dominate the residual." | **wording + number (CLAIM FLIP)** | **better** | swap-map `ir_confusers clf→filt[robust6]` **0.1792→0.0192** (93.5% cut vs bare 0.2943); audited "irconf composed r6 fire". **FLAG — LEAKY cache:** honest held-out IR_confusers val/test **90→22 (94% removed)** + CBAM valid recall recovered 0.717→0.967 (`…_train_heldout.md §2.4`). Keep conclusion's % on the cache surface (table-consistent) but ensure the §ir_xmodal_verifier held-out numbers are the cited proof; reframe must NOT overclaim beyond the held-out 94%. |
| conclusion.tex | 10 | RQ1 | "Anti-UAV control… ($F1$ $0.973 \to **0.984**$)… ($41$ false positives across $4{,}000$ Anti-UAV frames)" | NO CHANGE | none | neutral | audited 0.9841→0.9842; 41 FP (bare detector) unchanged |
| conclusion.tex | 14 | RQ2 | "on Svanstr\"om it removes **$82\%$** of bare false positives (**$2{,}019 \to 353$**)" | "removes **$82\%$**… (**$2{,}019 \to 337$**)" — recompute: 1−337/2019 = **83\%**; write "**$83\%$** ($2{,}019\to337$)" | number | better | swap-map svan `filt only (mlp_v5,RGB)` FP **353→337**. un-audited derived count (FP 337 from tab:ablation_svanstrom). |
| conclusion.tex | 14 | RQ2 | "on the confuser corpus the filter is again the decisive stage (**$835 \to 29$** FP detections, where the patch-CNN predecessor managed $835 \to 282$)" | "(**$835 \to 39$** FP detections, where the patch-CNN predecessor managed $835\to282$)" | number | worse (cache; bird held-out better) | swap-map rgb_confuser `filt only (mlp)` FP **29→39**. un-audited derived. **FLAG:** in-sample bird cache; held-out bird.v1i TEST shipped 91→**30** is the honest transfer number (§verifier_results). The "10× stronger than patch" framing still holds (39 vs 282 ≈ 7×; soften "≈10×"→"≈7×" if quoted). |
| conclusion.tex | 14 | RQ2 | "the shipped filter-then-classify keeps the router's recall ($R=0.991$, $F1=**0.944**$) and classify-then-filter trades a little recall for precision ($F1=**0.930**$), within $1.5$~pp" | "$F1=**0.946**$ … ($F1=**0.931**$), within $1.5$~pp" | number | better | **audited "NR svan filt->clf F1" 0.9439→0.9459; "NR svan composed F1" 0.9302→0.9308** |
| conclusion.tex | 14 | RQ2 | "router costs $0.095$~ms… filter $1.3$--$2.1$~ms… ($37$--$404\times$ cheaper…)" | NO CHANGE (latency; architecture identical) | none | neutral | audited speed rows |
| conclusion.tex | 18 | RQ3 | "thermal dominates… ($F1$ $0.940$ vs $0.607$)… Anti-UAV ($0.985$ vs $0.961$)" | NO CHANGE (bare detectors) | none | neutral | un-audited detector cells (unchanged) |
| conclusion.tex | 21 | methodological thread | "linear separability $\approx$95\% RGB / $0.981$ IR… AUROC $0.500 \to 0.919$" | NO CHANGE (MRI) | none | neutral | audited "MRI ir LDA"=0.981 |
| conclusion.tex | 25 | findings | "grayscale confusers, aligned into thermal feature space, are what made the thermal filter trainable at all" | NO CHANGE (alignment claim holds for thermal-native build too) | none | neutral | un-audited; consistent with `…_train_heldout.md §2` |
| conclusion.tex | 30 | sec:production_stack | "per-frame filters \texttt{mlp\_v5} (RGB) and **\texttt{mlp\_v5\_ir\_aligned} (IR), the latter one network with two per-modality input scalers** serving the thermal and grayscale-fed channels" | **REWRITE (retire one-net-two-scalers):** "per-frame filters \texttt{mlp\_v5} (RGB, the \texttt{v4} bird-split build) and the **two-head IR filter — a thermal-native head (\texttt{mlp\_aligned\_thermalonly}, CBAM-trained) and a grayscale-aligned head (\texttt{mlp\_aligned\_gray\_balanced})**, selected per modality" | **wording (CLAIM FLIP)** | better/neutral | un-audited prose → `2026-06-18_thesis_integration_checklist.md §6` + `…_train_heldout.md §2`. **NB:** weight names change — ensure glossary (appendices 269) + tab:models_evaluated (163) updated in lockstep. |
| conclusion.tex | 30 | production_stack | "RGB confidence floor can drop to $0.05$--$0.10$… $+10$~pp F1 on SelCom at unchanged confuser safety" | NO CHANGE (SelCom F1 0.6115 unchanged in swap-map) | none | neutral | audited "selcom mlp F1" 0.6115 unchanged |
| conclusion.tex | 32 | carve-out (i) | "\texttt{mlp\_v5} **costs $11$~pp F1 on the photo-style \texttt{rgb\_dataset} split**, an OOD coverage gap… **the patch filter remains the documented fallback** on that one surface" | **REWRITE (carve-out RESOLVED):** the v4 bird-split build **closes this carve-out** — rgb_dataset_test F1 **$0.809\to0.922$** (recall $0.691\to0.887$); the $11$-pp gap is gone and the patch fallback is no longer needed on this surface. | **wording + number (CLAIM FLIP)** | **better** | swap-map rgb_dataset_test `filt(mlp)` F1 **0.8092→0.9222**, recall **0.6912→0.8873** (audited "rgbtest mlp F1" 0.8092→0.9222). Held-out (§1.3 disjoint test split) → honest. **NB:** removes a stated carve-out — drops carve-out count 4→3; renumber (ii)(iii)(iv). |
| conclusion.tex | 32 | carve-out (ii) | "shipped no-reject \texttt{robust8-nr} trades confuser suppression for recall: RGB-confuser frame fire rises to $\sim$1\%…" | "rises to $\sim$**1.4\%**…" | number | worse (cache) | audited "NR rgb_conf fire" 0.0144 |
| conclusion.tex | 32 | carve-out (iii) | "the aligned filter is a confuser-suppression tool (**$-96.8\%$ FP**), not a recall-safe filter… grayscale detection runs unfiltered" | KEEP claim; UPDATE count: gray_confuser `filt(mlp)` FP **21→15** (balanced-gray @0.25). Recompute −96.8% (656→21) → if denominator 656 unchanged, **656→15 = −97.7\%** | number | better | swap-map gray_confuser FP **21→15** (audited "grayconf mlp fire/FP" 0.0076→0.0053 / 21→15). un-audited −96.8% derived count → recompute to −97.7%. |
| conclusion.tex | 32 | carve-out (iv) | "aligned IR filter… operator-GUI wiring is an open engineering item… Jetson-class edge latency" | NO CHANGE (engineering status) | none | neutral | un-audited (status) |
| conclusion.tex | 36 | future work | "Closing the airplane gap directly with OOD airplane crops…" | **SOFTEN (flipped):** the airplane gap is now **largely closed** by the thermal-native head; reframe as "further hardening the airplane class" rather than "closing the gap" (the thermal-native CBAM build already cut airplane-dominated fire 29.4%→1.9% cache / 94% held-out) | wording (CLAIM SOFTEN) | better | un-audited prose; consistent with `…_train_heldout.md §2.4` |

---

## FILE 5 — `appendices.tex`  [`…/chapters/appendices.tex`]

| file | line | section | CURRENT (quote) | PROPOSED new | edit type | better/worse/neutral | VERIFIABLE? |
|---|---|---|---|---|---|---|---|
| appendices.tex | 38 | Thermal Confuser Corpus | "The **180-frame CBAM probe** used during the aligned filter's development remains that filter's held-out training gate, since \texttt{IR\_confusers} postdates the training." | UPDATE: thermal-native head is trained WITH CBAM-train (GT-aware); held-out gate is now **CBAM valid** (recall recovered 0.717→0.967) + **IR_confusers val/test** (90→22). Reword "180-frame CBAM probe… held-out training gate" → name CBAM valid + IR_confusers val/test as the held-out surfaces. | wording | better | `…_train_heldout.md §2.3` (CBAM train used; CBAM valid + IR_confusers val/test held out). un-audited. |
| appendices.tex | 163 | tab:models_evaluated | "\texttt{mlp\_v5\_ir\_aligned} & **production** & Production IR filter (**one network, two per-modality scalers**); held-out CBAM $F1\;0.699\to0.846$, **FP $48\to15$**, recall-safe." | **REWRITE row(s):** split into two-head IR filter. Name `mlp_aligned_thermalonly` (CBAM-native) + `mlp_aligned_gray_balanced`. CBAM held-out: **R 0.967 / FP 6 @0.05** (recall recovered from balanced 0.717). | **wording + number (CLAIM FLIP)** | better | `2026-06-18_thesis_integration_checklist.md §2`: CBAM **R 0.967 / FP 6 @0.05** (RESOLVED, no extra GPU). un-audited table row. **NB:** must match glossary 269 + conclusion 30 + fig_pipeline. |
| appendices.tex | 164 | tab:models_evaluated | "\texttt{mlp\_v5\_gray} & comparison & Grayscale-mode filter; cuts held-out grayscale confuser FPs by ${\sim}96\%$." | Rename to `mlp_aligned_gray_balanced` (now PRODUCTION grayscale head, not comparison); FP cut updates (656→15 = ~97.7%) | wording + number | better | swap-map gray FP 21→15; §6 checklist (two-head production). un-audited. |
| appendices.tex | 162 | tab:models_evaluated | "\texttt{mlp\_v5} & production & …distilled MLP on… \texttt{p3}+\texttt{p5} ROI features, run per-frame." | OPTIONAL: name the \texttt{v4} bird-split build (RGB carve-out closed). Architecture (p3+p5, per-frame) unchanged. | wording (optional) | neutral/better | `…_train_heldout.md §1`; 517-D unchanged |
| appendices.tex | 165 | tab:models_evaluated | "\texttt{patch\_v2} & comparison & Superseded… **documented fallback on the photo-style \texttt{rgb\_dataset} surface**." | UPDATE: with the v4 build closing the rgb_dataset carve-out, patch_v2 is no longer the needed fallback there; soften to "audited predecessor; the v4 RGB filter removes its last fallback role." | wording (CLAIM FLIP, ties to conclusion 32-i) | better | swap-map rgbtest F1 0.809→0.922 (carve-out closed). un-audited row. |
| appendices.tex | 269 | glossary | "\texttt{mlp\_v5\_ir\_aligned}] The cross-modal IR confuser filter…: **one network with two per-modality input scalers**, serving the thermal-deploy and grayscale-fallback paths." | **REWRITE (retire one-net-two-scalers):** "The IR confuser filter, now **two heads**: a thermal-native head (\texttt{mlp\_aligned\_thermalonly}, CBAM-trained, @0.05) and a grayscale-aligned head (\texttt{mlp\_aligned\_gray\_balanced}, @0.25), selected per modality." | **wording (CLAIM FLIP)** | better | `2026-06-18_thesis_integration_checklist.md §6`; `…_train_heldout.md §2.1/2.5`. un-audited glossary. |
| appendices.tex | 268 | glossary | "\texttt{mlp\_v5}] The distilled feature-space confuser filter…: an MLP on… \texttt{p3}+\texttt{p5} ROI features that supersedes the v2 patch filter on confuser-rich surfaces and runs per-frame." | OPTIONAL: note v4 build. Architecture/threshold (0.25) unchanged. | wording (optional) | neutral | `…_train_heldout.md §1` |
| appendices.tex | 224 | tab:provenance | "Svanstr\"om composed F1 $0.742 \to **0.944**$ (shipped \texttt{robust8-nr}, \texttt{filt$\to$clf})" | "$0.742 \to **0.946**$" | number | better | **audited "NR svan filt->clf F1" 0.9439→0.9459** |
| appendices.tex | 225 | tab:provenance | "Confuser fire $30.4\% \to **1.1\%**$ (shipped \texttt{robust8-nr}; **$0.15\%$** \texttt{robust8} ablation)" | "$30.4\% \to **1.4\%**$ (…; **$0.11\%$** robust8 ablation)" | number | worse (cache) | audited "NR rgb_conf fire" 0.0144; robust8 0.0011 |
| appendices.tex | 226 | tab:provenance | "Anti-UAV no-harm $0.973 \to **0.984**$" | NO CHANGE | none | neutral | audited 0.9842 |
| appendices.tex | 227 | tab:provenance | "Grayscale 3-way $0.607/0.187/0.580$" | NO CHANGE (detector finding) | none | neutral | audited "3way gray F1" |
| appendices.tex | 184/195/197 | app:mri_report | "LDA separability 0.952… projected FP cut 97.4%… Projected FP rate 1.4%" + corpus "19,334 drone / 13,597 confuser" | NO CHANGE (MRI report is the RGB filter's TRAINING-CORPUS diagnosis; v4 keeps the same 32,931-det corpus per provenance §1.2 / the report regen is mri-side, not a Tier-1 cell) | none | neutral | un-audited (verbatim report). **FLAG:** if v4 retrain shifted the report, regen `mri/results/v5_report_regen/`; provenance §1.2 says v2 corpus = same 19,334/13,597 budget → likely unchanged. Confirm before assuming. |
| appendices.tex | 273 | glossary | "\texttt{patch\_thr}] …Production: 0.9 SVAN-like, 0.7 real-video." | NO CHANGE (patch predecessor threshold; not swapped) | none | neutral | un-audited (predecessor) |
| appendices.tex | 251 | glossary | "CNN] The patch filter is a MobileNetV3-class CNN" | NO CHANGE | none | neutral | un-audited |

---

## TOTALS

- **Total edits proposed (rows that change):** **26**
  - main.tex: 1 forced numeric block (line 159: 3 numbers) + 1 optional wording → **1–2**
  - introduction.tex: **2** (line 50 F1+precision; line 50 confuser-fire/reject)
  - related_work.tex: **2** (line 116 RGB fire; line 119 IR robust6 fire — FLIP better)
  - conclusion.tex: **11** (RQ1 F1/precision; RQ1 confuser; RQ1 **thermal-hole FLIP**; RQ2 82→83%; RQ2 835→39; RQ2 filt→clf F1; production-stack **two-head FLIP**; carve-out-i **RESOLVED FLIP**; carve-out-ii fire; carve-out-iii −97.7%; future-work airplane soften)
  - appendices.tex: **10** (CBAM gate; tab:models 163 **two-head FLIP**; 164 gray rename; 165 patch fallback; 269 glossary **FLIP**; provenance 224; provenance 225; + optional 162/268/162-row)
- **NO-CHANGE rows confirmed:** ~24 (bare-detector / MRI / latency / engineering-status cells).

### By direction
- **Better:** **15** (Svan F1↑, precision↑, IR thermal-confuser suppression FLIP, RGB carve-out RESOLVED, gray FP↓, two-head reframe, airplane-hole closed, 82→83% FP removal).
- **Worse:** **5** (rgb_confuser in-sample cache fire 1.1→1.4% and FP 29→39, reject 0.15→0.11% [fewer frames=better but smaller-sample], all on the **in-sample bird cache**; DUT/Svan single-modality recall deltas are table-only, not in these chapters).
- **Neutral:** **6** (Anti-UAV 0.984 rounds same; latency; MRI separability; SelCom F1; detector findings).

### By verifiability
- **Audited (CHECK exists in `_audit_headline_numbers.py`):** the core headlines —
  Svan filt→clf F1 (0.9439→**0.9459**), Svan composed F1 (0.9302→**0.9308**), Anti-UAV composed (0.9841→0.9842),
  rgb_conf fire (0.0106→**0.0144**), rgbtest mlp F1 (0.8092→**0.9222**), irconf r6 fire (0.1792→**0.0192**),
  grayconf fire/FP (0.0076→0.0053 / 21→**15**), MRI ir LDA (0.981). **≈10 audited number-edits.**
- **Un-audited (verify vs source doc/swap-map before writing):** the derived FP counts (Svan 353→337,
  confuser 835→29→**39**, gray 656→21→**15**), CBAM held-out **R 0.967 / FP 6** (checklist §2 RESOLVED),
  all prose CLAIM-FLIPS (two-head IR, thermal-hole-closed, RGB carve-out-resolved, airplane-soften),
  glossary/tab:models_evaluated rows, the MRI verbatim report. **≈16 un-audited edits** → sources:
  `filter_PRODUCTION_swap_tables.md`, `2026-06-18_thesis_integration_checklist.md §2/§6`,
  `2026-06-18_filter_provenance_train_heldout.md §1–2`.

### Cross-chapter consistency LOCKS (must edit in lockstep)
1. **Svan F1 0.944→0.946** appears in main.tex:159, intro:50, conclusion:10/14, appendices:224 (+ empirical tables, out of scope here). All five must move together.
2. **Confuser fire 1.1%→1.4% / reject 0.15%→0.11%** in main:159, intro:50, conclusion:10, conclusion:32-ii, appendices:225, related_work:116.
3. **Two-head IR filter** (retire one-net-two-scalers) in conclusion:30, appendices:163, appendices:269 — AND the out-of-scope methodology:535/696/818, fig_pipeline.tex:17, empirical glossary refs. Flag the cross-chapter dependency.
4. **RGB carve-out RESOLVED** (rgbtest 0.809→0.922) in conclusion:32-i + appendices:165 (patch fallback) — ties to empirical §verifier_results (out of scope).
5. **Thermal-hole closed / 39%→93% cache (94% held-out)** in conclusion:10 + future-work:36 + related_work:119 — ties to empirical:173/649/802 (out of scope).

### LEAKY-CACHE flags (headline numbers that should reference the held-out figure)
- **conclusion:10 / related_work:119** thermal-confuser fire uses the **leaky `ir_confusers` cache** (= IR
  train split). Keep the table-consistent cache % (1.9% / 93% cut) BUT the cited proof must be the
  **held-out IR_confusers val/test 90→22 (94%)** + CBAM-valid recall recovery in §ir_xmodal_verifier. Do
  not overclaim past 94%.
- **conclusion:14 / intro:50 / main:159** rgb_confuser FP (29→39) and fire (1.1→1.4%) use the **in-sample
  bird cache**. The honest transfer number is **held-out bird.v1i TEST shipped 91→30**, which lives in
  §verifier_results. These chapters keep the cache value (table-consistent) but must stay consistent with
  that held-out figure; the "≈10× vs patch" claim softens to "≈7×".

---

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\filter_edit_plan_intro_concl_appendix.md` (this plan)
