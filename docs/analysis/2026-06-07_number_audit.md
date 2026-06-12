# 2026-06-07 — Full thesis number audit (`thesis_working_distilling.tex`)

**Trigger:** after the `0.585` scare, verify *every* result-number against the evidence ledger.
**Method:** (1) `thesis_audit.py` deterministic pass (matches each number to `evals.csv`); (2) five
parallel per-chapter auditor agents checking each number against its `% [source:]` citation; (3)
manual resolution of the concrete candidates against `master.csv` / `models.csv` / `ledger.csv`.

## Headline
**No fabricated numbers.** Of ~1,093 result-numbers, the engine matched 1,090 to a real eval value;
the 3 it didn't are sourced non-eval stats (27.5% dataset split → `dataset.yaml`; 37.2% → ledger
`ir-grayscale-is-hallucination-mode`). Every headline result (F1 0.869, LDA 0.952/0.981, ANOVA
42 346/5 370, gray-align AUROC 0.50→0.919, the distill-verifier table's 15 cells, the classifier
zoo, the cascade tables) traces correctly. The problems below are **misattributions, missing eval
rows, and a few stale/wrong labels — not invented data.**

---

## TIER 1 — fixed (2 real errors; #1 was a FALSE ALARM)
| # | location | problem | resolution |
|---|---|---|---|
| ~~1~~ | §sys-arch + Intro | `R=0.072`/`P=0.837` @640 attributed to `retrained_v2` (an agent thought it was baseline) | **FALSE ALARM — thesis is CORRECT.** The May-10 ablation `cell.json` records `rgb_weights = "RGB model/Yolo26n_retrained_v2/weights/best.pt"`, so R=0.072@640 genuinely is `retrained_v2`. The *ledger note* `prov-may10-imgsz` ("…1280 would be ~0.959") wrongly assumed baseline — **the ledger note is the misleading item, not the thesis.** No thesis edit. (Optional: correct that ledger note.) |
| 2 | app:datasets L2113 | confuser-clips Total cell = `1{,}270` (rows sum to 1,250; used everywhere else) | **FIXED → `1{,}250`.** Verified Extracted column 249+55+20+271+20+21+20+554+20+20 = 1,250. |
| 3 | app:models caption/rows | marked baseline `production`, said "6 production"; models.csv has 5 | **FIXED — regenerated `app:models` from models.csv (now 5 production, baseline excluded).** Also fixed a real `gen_models_appendix.py` bug: its idempotent-replace fed the LaTeX block as a regex *replacement* → `re.error: bad escape \c` on every re-run (only first-insert worked → why it went stale). Now uses a function replacement. |

## TIER 2 — numbers REAL but MISSING from the ledger (no eval row, or cited id doesn't exist)
| # | location | number | issue | fix |
|---|---|---|---|---|
| 4 | L259/341/502/846/1931 (×5) | IR **Anti-UAV F1 = 0.965/0.9654** | **no IR-Anti-UAV eval row** in evals.csv; value lives only in May-10 `master.csv`; citations point at the Svan-IR or RGB-Anti-UAV eval | record an `evals` row for IR Anti-UAV (P0.987/R0.945/F1 0.9654) + fix the `% [source:]` |
| 5 | ~25 source-comments | `eval=clf_own_holdout` | **dead eval-cite** — `clf_own_holdout` is a *config* id, no eval row exists | record the classifier-holdout eval row(s), or change `eval=` to the real ids |
| 6 | §exp L1461/1462 (tab:cumulative_svanstrom) | S2 `0.915/0.912/0.914`, S3 `0.927/0.868` | Svanström-**paired** cumulative values cited to `clfzoo_fnfn` (confuser-zoo only); no paired eval row | record the Svan-paired cumulative eval, or re-cite |
| 7 | §exp L1857 | patch median probs `0.540/0.904/0.987` | in `tab:patch_audit` (sourced) but the prose mention mis-cites `pipe_percat_sa32` | re-cite to `patch_catch_v2_svan` |

## TIER 3 — MISATTRIBUTED citations (number correct, `% [source: eval=]` points to the wrong/▲different eval)
| location | number | cited | should be |
|---|---|---|---|
| L703/788 | Roboflow OOD `R=0.726/0.746` | `rgb_svan_retrainedv2` / `rob_rgb_baseline` | `rob_rgb_retrainedv2` / `rob_rgb_baseline` |
| L1221 | distill CV `F1=0.9857` | `distill_cv_mlp` (=0.9955, non-shipped corpus) | `mri-v5-report-regen` (shipped corpus) |
| L1925 | `R=0.072` | `clf3_retrainedv2` (a classifier-accuracy eval, F1 0.842) | the May-10 rgb_only row |
| L183/188 | per-category halluc `94.4/74.6/66.2`, `41.9/94.2/64.7` | `rgb_svan_baseline`/`_hardneg` (overall only) | failure-diagnosis cache `svanstrom_1280_by_category.csv` |
| L214/358/1980 | S1 zoo fire `52.1%` | `clfzoo_fnfn` (S2/S3 only) | the `confuser_*_v1.1/summary.json` S1 cell |
| L1477/1480/1486 | production patch point `F1≈0.895` labelled **`patch_thr=0.9`** | eval `svan_s3_sa32_thr08` is at **thr=0.8** (R0.868/F1 0.896 — matches the thr=0.9 *sweep* row, not the thr=0.8 sweep row 0.856/0.889) | reconcile the thr label between the eval id and `tab:patch_sweep` |

## Careful-pass resolution (2026-06-08) — TIER-3 + dead-cites
- **Fixed (3 genuine Roboflow eval gaps):** L704 added `rob_rgb_retrainedv2,rob_rgb_baseline` (sources the R=0.726/0.746 in-sentence); L789 `rob_rgb_baseline`→`rob_rgb_retrainedv2` (claim is retrained_v2's R=0.726); L1516 added `rob_rgb_retrainedv2` (table compares both models).
- **Left as false-flag / imprecise-but-sourced:** L1925 (`R=0.072` is a back-reference to §3.1, comment serves the classifier threat); L1536/L1956/L2010 (`rob_rgb_baseline` cited for a confuser-suppression claim, but `ledger=patch-verifier-distribution-bound` correctly sources the finding and no confuser-eval exists to substitute); distill `F1=0.9857` (`ledger=mri-v5-report-regen`+`cache=v5_report_regen/stats.json` already source the shipped corpus); per-category halluc + S1-zoo `52.1%` (granularity, sourced via the by-category cache / cumulative ledger).
- **`clf_own_holdout` dead-cites (23):** all removed from `eval=` slots (21 cleared, 2 stripped to keep the real `svan_gray_robust8_t20_clffilter`); the 27 valid `config=clf_own_holdout` references preserved.
- **Typo:** L259 IR Anti-UAV `F1=0.9647`→`0.9654`. **Recorded:** `ir_v3b_antiuav640_may10` eval row.

## Verified clean (sample)
tab:rgb_comparison, tab:selcom, tab:distill_verifier (all 15 cells), tab:classifiers, tab:patch_audit,
patch v1–v4 F1, robust8 τ-sweep (0.577/0.681/0.738), real-video F1 (0.826/0.644/0.219), IR Svan
(0.950/0.973/0.961), IR OOD (0.519/0.264), confuser 13× (0.107→0.008), MRI/LDA/ANOVA/gray-align,
scoring-swing — all match their cited source within rounding.

## Delivered
- `docs/analysis/2026-06-07_number_audit.md` (this file)
- engine report: `docs/analysis/THESIS_AUDIT_2026-06-07.md`
