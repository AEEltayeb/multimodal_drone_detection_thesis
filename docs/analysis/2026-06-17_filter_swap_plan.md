# Filter Swap Kit — plan + runbook (RGB `mlp_v5` / IR `mlp_v5_ir_aligned`)

**Date:** 2026-06-17 · **Purpose:** make swapping the two confuser filters into the thesis a
repeatable, traceable operation, so whichever pair is finally locked (v2 here, or a future
v3-RGB + new-IR) drops in with one regeneration pass + one guided prose pass.

**Decisions locked (user, 2026-06-17):** (1) build the kit now, **defer the actual swap (Phase 3)**
until the final pair is chosen; (2) ship new weights under **versioned filenames** (keep the old
ones); (3) **I edit chapter prose**, guided by the registry, at integration time.

---

## 0. The one finding that reshaped this
The **canonical live thesis** is `docs/thesis_working_distilling_overleaf/chapters/{methodology,empirical,
introduction,conclusion,appendices}.tex` → `main.pdf` (146 pp, rebuilt 2026-06-17). The repo-root
`docs/thesis_working.tex` is a **stale patch-verifier-era snapshot** (zero `filt`/`clf→filt` cells) —
**do not edit it for the swap.** `docs/thesis_chapters.tex` is older still (no distilled filters at all).

## 1. Traceability spine (how a thesis number ties to a filter)
```
new weights ─► THESIS_* env ─► every filter-dependent replay (all import load_verifiers)
                                   │
   pipeline_eval_unified.py ──────┼─► thesis_eval/results/            (robust8 main)
   temporal_replay.py             ├─► thesis_eval/results_noreject/   (robust8-nr = SHIPPED router)
   notes_round1_replays.py        ├─► thesis_eval/results_clean/      (clean splits)
   video_thr_sweep.py             ├─► runs/results_dut/               (DUT test split)
   leakage_controlled_replay.py   ├─► thesis_eval/results/{temporal,notes_round1,video_thr_sweep,...}.json
   eval/filter_operating_sweep.py ├─► docs/.../figures/fig_filter_operating.{pdf,png}
   (CBAM 48→15, IR-HITL, latency)─┘   GPU-only → knowledge/evals.csv + ledger.csv (NOT replay JSON)
                                   │
   _audit_headline_numbers.py  ── CLAIMED constant == JSON cell (fails if Δ>5e-4) + CITED_PATHS exists
                                   │
   chapters/*.tex  (numbers + qualitative claims)  +  knowledge/ (models, evals, ledger rows)
```
Full number→file→command map already exists at **`runs/README.md`** — that is the integration command list.

**Critical risk the scan found:** most filter numbers in the thesis are **UN-AUDITED** (the audit pins
only the F1 cell of each filter row + a handful of FIG/CBAM cells; all TP/FP/FN/P/R and most prose FP
counts are unguarded). **The swap registry — not the audit gate — is the real safety net.** Every
registry row must be checked by hand on a swap.

## 2. The registries (the swap checklist)
- `docs/analysis/filter_swap_registry_empirical.md` — empirical.tex: ~120 numbers (≈38 audited, rest
  UN-AUDITED), 9 figures, 13 tables, 13 falsifiable claims. **Use this file.**
- `docs/analysis/filter_swap_registry_methodology.md` — methodology + intro + conclusion + appendices
  + related_work: ~70 numbers, 20 definitions/recipes, 30 figs/tables, 13 claims, 6 glossary entries.
  **Use this file.** (CBAM confirmed internally consistent at **0.846 F1 / 15 FP**; the 0.841/13 in old
  notes is superseded — no reconciliation needed.)
- `..._map_numbers.md` / `..._map_definitions.md` — **STALE**: scanned the wrong file (thesis_working.tex).
  Ignore for the swap; kept only as a record.

**Claims a swap may falsify (must reword, not just renumber):** "RGB filter does all the work on
Svanström" / "IR-thermal filter contributes nothing"; the **thermal-airplane hole** ("thermal
confusers resist ~39%"); **grayscale over-veto** (recall ≤ 0.27); "recall-safe"; "reads p5 activations
not the confidence score"; composition order `filt→clf` vs `clf→filt`; production-stack sentences
naming the shipped weight. `fig_pipeline.tex` also still labels the router `robust8` (not `robust8-nr`).

## 3. The kit (built now)
- **`thesis_eval/run_filter_bundle.py`** — sets `THESIS_*`, runs the decision-relevant replays into
  `thesis_eval/_filter_swap/<tag>/`, writes `swap_manifest.json` (weight SHA-256 + thresholds + git
  SHA + timing), optionally auto-diffs vs a prior tag. Zero-GPU; never writes committed dirs/weights.
  `--only <surface>` for a fast smoke. Defaults reproduce the shipped stack.
- **`thesis_eval/results/_filter_ab/diff_filters.py`** — per-surface shipped-vs-candidate delta over
  every filter-bearing cell (now takes `--shipped/--candidate`). This is the decision + changelog artifact.
- **Env override in `pipeline_eval_unified.py`** — `THESIS_MLP_V5 / THESIS_ALIGNED / THESIS_ALIGNED_GRAY
  / THESIS_RGB_THR_MLP / THESIS_IR_THR_MLP / THESIS_GRAY_THR_MLP`; defaults = shipped (committed numbers
  unchanged). Inherited by ALL sibling replays + `eval/filter_operating_sweep.py`.

### Evaluate any future pair (the repeatable command)
```powershell
py -u thesis_eval/run_filter_bundle.py --tag shipped                 # baseline once
py -u thesis_eval/run_filter_bundle.py --tag v3rgb `
   --rgb <new_rgb.pt> --ir <new_ir.pt> --ir-gray <new_gray.pt> --ir-thr <thr> --diff-against shipped
```
→ tagged JSONs + manifest + a full delta table. That is the entire "is the new pair better" loop.

## 4. Integration runbook (Phase 3 — DEFERRED, run once per locked pair)
Not yet executed. When the pair is locked:
1. **Promote weights (versioned):** copy → `models/verifiers/rgb_v5/mlp_v5_<ver>.pt`,
   `models/verifiers/ir_aligned/mlp_aligned_<ver>.pt` (+ `_gray`). Keep the old files.
2. **Repoint defaults:** `pipeline_eval_unified.py` weight constants → new filenames; set
   `IR_THR_MLP` default to the new IR thr (e.g. 0.01); `gui/pyside_engine.py` pointer.
3. **Regenerate every canonical evidence file** (zero-GPU; commands per `runs/README.md`): the four
   dirs (`results/`, `results_noreject/`, `results_clean/`, `runs/results_dut/`) + `temporal_results`,
   `notes_round1_results`, `video_thr_sweep`, `leakage_controlled`, and `eval/filter_operating_sweep.py`
   (the figure). **GPU-only, gated:** CBAM held-out (`mri.cli --holdout-eval`) and any IR-HITL/latency
   rows in `knowledge/evals.csv`.
4. **Move the audit gate:** update CLAIMED constants + `FS` dict + `CITED_PATHS` in
   `_audit_headline_numbers.py` to the new values/paths.
5. **Edit chapters** (methodology + empirical + intro + conclusion + appendices + glossary) from the
   registries — numbers AND the falsifiable claims in §2. (I do this, you review the diff.)
6. **Record traceability:** `kb.py record` new `models` (provenance: trained_from_script, dataset,
   threshold) + `evals` + `ledger` rows; commit `swap_manifest.json` + the diff.
7. **Build + audit:** recompile `main.pdf`; `py thesis_eval/_audit_headline_numbers.py` must be all-pass.

## 5. Open / to confirm at integration
- Exact `--out` + cache-subset invocation for `results_clean` and `runs/results_dut` (the clean/DUT
  caches are resident in `thesis_eval/cache/`, so replays are zero-GPU — confirm per-dir command before
  the in-place regen). `run_filter_bundle.py` currently automates the two core replays into a scratch
  tag; the remaining canonical-dir regens are listed in `runs/README.md` and will be wired into an
  `--integrate` mode (or run by hand) when Phase 3 is triggered.
- Whether `results_noreject` differs from `results` by anything other than which cells the audit reads.

## Delivered
- `…\ES_Drone_Thesis\thesis_eval\run_filter_bundle.py` (new — Phase-1 driver)
- `…\ES_Drone_Thesis\thesis_eval\results\_filter_ab\diff_filters.py` (now CLI-parameterized)
- `…\ES_Drone_Thesis\thesis_eval\results\_filter_ab\{shipped,candidate,*_temporal}\` + `DIFF.txt` (v2 A/B)
- `…\ES_Drone_Thesis\thesis_eval\pipeline_eval_unified.py` (THESIS_* env override; defaults = shipped)
- `…\ES_Drone_Thesis\docs\analysis\filter_swap_registry_empirical.md`
- `…\ES_Drone_Thesis\docs\analysis\filter_swap_registry_methodology.md`
- `…\ES_Drone_Thesis\docs\analysis\2026-06-17_filter_swap_plan.md` (this file)
