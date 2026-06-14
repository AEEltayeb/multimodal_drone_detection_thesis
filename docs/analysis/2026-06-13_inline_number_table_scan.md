# Inline-number → table scan (round-6 pitstop, 2026-06-13)

**Question (notes l.818–821):** which inline numbers in the thesis lack a table and need one?
**Bar (locked):** **T** = make a table (≥3 related/comparative numbers, or a comparison spanning models/surfaces, with no existing table); **X** = numbers already live in a table/figure → keep as cross-referenced prose, don't duplicate; **L** = single/standalone or one-off contextual number → leave inline.

**Headline finding:** the thesis is already table-dense. The biggest genuine gap (the §3.8.8 MRI number blobs) was **fixed this session** (new `tab:ir_mri_sep` + `tab:gray_thermal_auroc`). The scan finds only **two** further real candidates; almost everything else is **X** (a deliberate restatement of an already-tabled headline number — good practice) or **L**.

---

## NEW table candidates (T) — for decision

### T1 (recommended) — Resolution 2×2 numeric table  ·  §3.2.3 / §3.8 `sec:resolution_arch`
The within-model resolution sweep (baseline & `retrained_v2` × imgsz 640/1280 Svanström recall: 0.964 / 0.684 / 0.323 / 0.070) currently exists **only as a figure** (`fig6_6_resolution`, grouped bars) and is restated inline in **three** places (related_work l.9, introduction l.23, methodology §rgb_results l.362). A figure of bars is not a table; the numbers are compared across model×size. → add a compact 2×2 table beside `fig:resolution`; convert the 3 inline restatements to cross-refs.
Source: `eval/results/svan_resolution_sweep.json` (already audited: RES cells).

### T2 (optional) — RGB-confusers per-category fire rate  ·  §1.2 `sec:problem` / §3.x
The `rgb_confusers_merged` @640 per-category split (bird 39.0% / helicopter 58.0% / airplane 23.4%; aggregate 30.4%) is stated **inline only** (introduction l.15). The *Svanström* @1280 per-category rates are already tabled (`tab:rgb_comparison`), but this @640 surface is not. 3-number cluster, single occurrence in motivation prose → borderline. → either a small per-category table or fold a column into an existing confuser table; or accept as L (intro motivation).
Source: `thesis_eval/results/notes_round1_results.json` (CAT; already audited: CAT rgbconf cells).

---

## Already tabled → keep as cross-referenced prose (X)
These recur inline but each has a home table; the inline mentions are correct restatements (intro/conclusion/related rhetoric):
- Svanström per-category fire rates (94.4 / 74.6 / 66.2; hardneg 94.2 / 64.7 / 41.9; retrained_v2 3.4 / 5.6 / 4.5) → `tab:rgb_comparison`. (intro l.15,18; related l.17; methodology l.611)
- Pipeline headline (0.742→0.949, R 0.948→0.958, P 0.609→0.939) → `tab:ablation_svanstrom`. (intro l.50; conclusion l.10)
- Confuser suppression ladder (30.4% → 4.9% → 1.1% → 0.15%) → `tab:ablation_confusers`. (intro l.50; conclusion l.10; related l.136)
- Stage runtimes (0.095 ms; 1.3–2.1 ms; 404×; 37–72×) → `tab:speed`. (intro l.50; conclusion l.14)
- IR evolution F1 0.503→0.967 → `tab:ir_evolution`. (intro l.55; related l.60; conclusion l.21)
- Modality A/B (RGB 0.458 / IR 0.632 / routed 0.921) → `tab:modality_ab`. (intro l.23)
- Per-size recall (29.8 px; 0.94–0.96 → 0.63) → `tab:per_size`. (intro l.23)
- Anti-UAV saturation (P 0.9922 / R 0.9950 / F1 0.9936; F1 0.986 @640) → `tab:numerical_comparison`. (related l.22; intro l.8)
- gray→thermal AUROC ladder (0.500/0.707/0.919/0.974) → now `tab:gray_thermal_auroc` (this session); de-dup'd in empirical §4 + related l.65 (related l.65 keeps the 0.919 vs 0.707 as a one-line lit-comparison — acceptable L, not a duplicate cluster).
- CBAM 48→15 + per-surface ΔR → `tab:ir_aligned`. (methodology §3.8.8 prose cross-refs it)

## Leave inline (L) — single/standalone or contextual
- Baseline training-time acceptance check P=0.978/R=0.915/mAP 0.951 (methodology l.356) — one-off training note, explicitly pre-protocol.
- SelCom ft4 lift 0.145→0.591, in-domain 0.942→0.929 (intro l.28) — two paired numbers, motivational.
- scene-fingerprint lean13→lean10 recovers 18–26 pp (methodology l.712) — single range, points to `sec:feature_selection`.
- Conclusion RQ-answer numbers — all carry explicit table cross-refs; summary prose, correct as-is.
- Detector-confidence sweep / low-conf mode numbers — already in `tab:lowconf_selcom`.

---

## Recommendation
Build **T1** (resolution 2×2 table — clear win, numbers currently figure-only and thrice-restated). Treat **T2** as optional (lean toward folding into an existing confuser table or leaving as intro motivation). Everything else: no new tables — the thesis is already well-tabled, and the remaining inline numbers are correct cross-referenced restatements or one-offs.
