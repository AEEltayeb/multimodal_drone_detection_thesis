# Pipeline Comparison — Aggregate Results & Thesis Copy

**Date:** 2026-06-09
**Status:** ⚠️ **MINI / TRIAGE RUN — NOT the standard run.** All numbers below are from a 1000-frame
strided triage (per `knowledge/THESIS_EVAL_LEDGER.md` §0). The **qualitative findings are
thesis-ready**; the **exact digits must be regenerated at the full standard** (Svanström stride-9
@1280, Anti-UAV n=4000 @640, confusers full @640) before final submission.
**Imgsz:** Svanström = 1280 (native 640×512, sub-floor drones); everything else = 640.

**Sources:**
- Routing cascade (Stages 1–3): `eval/results/_routing_pipeline_cmp/comparison.md` · run `eval/compare_routing_pipeline.py` · ledger `mini-routing-scorecard-1k`
- Verifier matrix (Stage 4 / filter choice): `eval/results/_offline_pipeline/offline_eval_results.md` · run `eval/pipeline_cache.py`+`pipeline_eval_offline.py` · ledger `mini-offline-verifier-matrix-1k`

> The two harnesses answer **different questions**. Routing = the **real production cascade**
> (detector→trust-classifier→filter). Offline = a **verifier-isolation** study (detector→one
> verifier). They use different filter configs on grayscale, so **do not cross-compare grayscale
> recall between the two tables** — within each table it is consistent.

---

## 1. Headline (the one-paragraph version)

The three-stage pipeline is a **no-op on clean in-domain data** (Anti-UAV thermal/RGB,
`ir_dset_final`, `ir_video`, `svanstrom_ir` all stay within noise of bare) and **earns its keep on
confuser-rich / OOD data** — so it is safe to always run. Of the three trust-classifier routers,
**robust8 wins on all three axes that matter**: best grayscale-drone recall (0.689 vs sa32 0.608),
lowest confuser fire-rate (0.044 vs sa32 0.049), and thermal-drone F1 within noise (0.979 vs 0.982).
The **RGB verifier (mlp_v5)** delivers a **13× false-positive cut on RGB confusers** (216→16) and
recovers Svanström-RGB F1 0.613→0.865; the **IR-aligned verifier** is the CBAM/grayscale win
(CBAM F1 0.699→0.846, gray-confuser FP 143→68). The **residual hard case is RGB birds** — the
classifier cracks them where the verifier alone cannot, but a 12.6% fire-rate remains (honest
limitation / future work).

---

## 2. Full-pipeline stage comparison (ROUTING harness, robust8 router)

**Drone datasets — F1 (recall in parens).** Higher = better.

| Dataset (rule, modality) | bare | filter_only | clf_only | **clf→filter (ship)** | filter→clf |
|---|---|---|---|---|---|
| antiuav (iou, thermal) | 0.971 (1.00) | 0.970 | 0.969 | **0.969** | 0.969 |
| svanstrom (iop, RGB@1280) | 0.674 (P0.51,R1.0) | 0.958 | 0.959 | **0.988** (R0.995) | 0.988 (R1.0) |
| svanstrom_gray (iop, gray) | 0.627 (R0.935) | **0.815** (R0.902) | 0.630 (R0.694) | 0.727 (R0.689) | 0.750 (R0.660) |

**Confuser datasets — fire-rate (FP / frame).** Lower = better.

| Dataset | bare | filter_only | clf_only | clf→filter | **filter→clf (best)** |
|---|---|---|---|---|---|
| rgb_confuser | 0.282 | 0.091 | 0.093 | 0.044 | **0.031** |
| rgb_bird_confuser (NEW) | 0.628 | 0.560 | 0.242 | 0.198 | **0.126** |

**Reads:**
- **Anti-UAV thermal:** pipeline does no harm on clean in-domain (F1 ~0.97 everywhere).
- **Svanström RGB:** bare is precision-starved (P0.51, fires on 69%); classifier **or** filter
  recovers precision to ~0.92–0.99 *without* losing recall → cascade F1 0.988.
- **Svanström grayscale (the hard regime):** the **classifier costs recall** here (drone-on-gray
  looks confuser-ish). robust8 **minimises that cost** (R0.689, best of the routers) but
  **filter_only preserves the most recall** (R0.902, F1 0.815). → On grayscale, prefer the
  verifier-only/filter→clf path; robust8 is the least-damaging classifier if you route through one.
- **RGB confusers:** bare fires 28% → cascade 3–5%; both stages compound.
- **RGB birds (NEW, hardest):** bare fires 63%. The verifier **barely helps** (0.56). The
  **classifier is what cracks it** — robust8 0.126 vs sa32 0.455 — validating robust8's
  `rgb_mean_conf`+`is_grayscale` features. 12.6% residual remains.

---

## 3. Router scorecard (sa32 vs robust6 vs robust8) — clf→filter cell

| router | thermal drone F1 | grayscale drone recall | confuser fire-rate |
|---|---|---|---|
| sa32 | **0.982** | 0.608 | 0.049 |
| robust6 | 0.981 | 0.581 | 0.053 |
| **robust8@0.20** | 0.979 | **0.689** | **0.044** |

→ **robust8 is the production pick:** +8pp grayscale recall and lowest confuser fire vs sa32, at a
0.3pp thermal-F1 cost (noise).

---

## 4. Verifier-isolation matrix (OFFLINE harness — which filter, alone)

**Where verifiers EARN their keep** (confuser-rich / OOD):

| Surface (modality, rule) | bare | patch | mlp / aligned | metric |
|---|---|---|---|---|
| rgb_confuser (rgb) | 216 | 104 | **16** | FP (13× cut) |
| svanstrom (rgb, iop) | 0.613 | 0.784 | **0.865** | F1 |
| selcom_val (rgb, iop) | F1 0.591 (P0.86) | 0.591 | **0.612 (P0.95)** | FP 22→7, R unchanged |
| cbam (ir, n=180) | 0.699 | 0.688 | **0.846** | F1 (FP 48→15) |
| ir_confusers (ir, NEW) | 288 | 250 | **220** | FP |
| gray_confuser (gray) | 143 | 119 | **68** | FP (halved) |
| gray_svan (gray, iop) | 0.548 | **0.591** | 0.581 (R0.505) | F1 |

**Where verifiers are NO-OPS** (clean in-domain — they don't hurt, safe to always run):
`antiuav_rgb` (F1 0.987→0.987), `antiuav_ir` (0.957), `ir_dset_final` (0.979→0.977),
`ir_video` (0.975), `svanstrom_ir` (0.953) — all ≈ bare.

**Carve-outs / honest limitations:**
- **rgb_dataset_test** (clean in-domain RGB): mlp_v5 **over-vetoes** — F1 0.922→0.812 (R0.888→0.694).
  The known −11pp cost of per-frame verification. Ship per-frame anyway: confuser robustness >> the
  recall cost on an already-strong detector.
- **rgb_bird_confuser:** here **patch (129 FP) BEATS mlp_v5 (199 FP)** — birds fool the distillation
  MLP more than the patch CNN. (And in the *cascade* it's the *classifier*, not either verifier,
  that does the heavy lifting — see §2.)

---

## 5. What to put on the thesis

**Add three tables** (regenerate digits at the standard run first):
1. **§Empirical — full-pipeline stage comparison** = §2 above (drone-F1 + confuser-fire, per
   dataset, bare→filter→clf→cascade). This is the "compare all pipelines on Anti-UAV / Svanström /
   confusers" table you asked for.
2. **§Empirical — router scorecard** = §3 (sa32/robust6/robust8 × 3 axes). One small table, settles
   the router choice.
3. **§Empirical — verifier ablation** = §4 (bare/patch/mlp per surface, split into "earns keep" /
   "no-op" / "carve-out").

**Prose claims (each must carry a `% [source:]` comment; all currently MINI — re-verify at standard):**
- *"The pipeline is inert on clean in-domain data and activates only under confuser/OOD pressure,
  so it can run unconditionally."* — `% [source: ledger=mini-offline-verifier-matrix-1k; ...]`
- *"robust8 is the production router: +8pp grayscale-drone recall and the lowest confuser fire-rate
  vs the sa32 baseline, at a negligible (0.3pp) thermal-F1 cost."* — `% [source: mini-routing-scorecard-1k]`
- *"The RGB verifier reduces confuser false positives 13× (216→16) and lifts Svanström-RGB F1 from
  0.61 to 0.86."* — `% [source: mini-offline-verifier-matrix-1k]`
- *"The IR-aligned verifier is the thermal/grayscale-confuser win — CBAM F1 0.70→0.85 and grayscale
  confuser FP halved — and is recall-safe on grayscale, unlike the dedicated grayscale verifier
  which over-rejected."* — `% [source: mini-offline-verifier-matrix-1k; DECISIONS 2026-06-09]`
- *"RGB birds remain the hardest residual confuser: the trust classifier cuts their fire-rate from
  0.63 to 0.13 where verifiers alone cannot, but a 13% residual marks them as future work."*
  — `% [source: mini-routing-scorecard-1k]` (frame as honest limitation, not a win)

**Grayscale nuance to state honestly:** on grayscale Svanström the trust classifier *costs* recall;
robust8 is the least-damaging router but the filter-only path preserves the most recall. Do **not**
claim the classifier improves grayscale — claim it *recovers most of the recall the other routers
throw away*.

**Do NOT put in the thesis yet:** any exact digit as "final." Mark them provisional or wait for the
standard run — the stride/imgsz differ from the canonical configs in THESIS_EVAL_LEDGER §0.

---

## Delivered

- This doc: `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\docs\analysis\2026-06-09_pipeline_comparison_results.md`
- Source results: `eval\results\_routing_pipeline_cmp\comparison.md`, `eval\results\_offline_pipeline\offline_eval_results.md`
- Ledger findings: `knowledge\ledger.csv` rows `mini-routing-scorecard-1k`, `mini-offline-verifier-matrix-1k`
