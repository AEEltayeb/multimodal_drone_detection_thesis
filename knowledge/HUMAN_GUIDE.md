# Human operating guide — Ahmed

The point of `knowledge/` is that you don't memorize the project — you look it up. This is the map.

## Where to look (in order, start of any session)
1. **`knowledge/PROJECT_STATE.md` → 📍 Resume Here** — where you left off, next action, open threads. READ FIRST.
2. **`knowledge/views/rankings.md`** — "what's best for X" (drone-detection / confusion-filter / trust). Generated; never edit.
3. **`knowledge/views/comparisons.md`** — apples-to-apples model comparison per eval_config.
4. **`knowledge/ledger.md`** — every finding + verdict + evidence (the old EVIDENCE_LEDGER, now structured). EVIDENCE_LEDGER.md is ARCHIVED — don't look for it in docs/.
5. **`knowledge/DECISIONS.md`** — why the system/structure changed over time.
6. Design specs: `knowledge/README.md` (the system), `THESIS_AUTOMATION.md`, `RESTRUCTURE_PLAN.md`.

## What to do — when — with what
| When | Do | Tool |
|---|---|---|
| Wrote a keeper script / trained a model / got a metric / found something | record it | `/record` (or it's automatic; the Stop-hook nags if you forget) |
| About to re-run an eval | check it's not cached first | `py knowledge/_tools/kb.py check-eval --target T --config C` |
| Need to find/edit a table row | use the tool, never hand-edit CSVs or `*.md` views | `kb.py record` / `set` / `mv` |
| Done with a script/model | mark `safe-to-archive`, then sweep | `kb.py set … lifecycle=safe-to-archive` → `/sweep` (you green-light the move) |
| Moving a file | move + update its row atomically | `kb.py mv <table> <id> <newpath>` |
| End of session | update your portal | `/resume` |
| Thesis work | fresh chat, the skill | `thesis` skill: audit / draft / novelty / email / plot / table / compile / refs / hygiene / diff |
| Codebase restructure | fresh chat | point it at `knowledge/RESTRUCTURE_PLAN.md` |
| Changed the schema/org | log it | append to `knowledge/DECISIONS.md` |

## Only YOU can do these (decisions / GPU runs)
- **Run the e2e inference-time measurement** (config `e2e_latency` is staged) → `/record` results.
- **Resolve the 2 unverified thesis claims**: run the 28pp dual-vs-trust scoring ablation + grab the `ir_detected` feature importance, OR soften those sentences.
- **Verify** which patch weights the GUI actually loads (`confuser_filter4_live` vs `patch_v2`).
- **Stack decision**: `selcom_960` looks like a better production RGB than the current pick — revisit (`ledger.selcom960-cross-surface-winner`, status=open).
- **Author the unwritten chapters** (Ch2/5/6/7/8/9) — the `thesis` skill assists, but the voice is yours.
- **Green-light** every `/sweep` physical move.

## Two launch-ready hand-offs (fresh chats, full context)
- **Restructure** → "read `knowledge/RESTRUCTURE_PLAN.md` + scripts.csv + models.csv, execute from Phase 0."
- **Full thesis pass** → "run `thesis` skill: full audit of `docs/thesis_working.tex` (verdict every claim, add `% [source: …]` run-info behind each), then draft the novelty gaps from `docs/analysis/2026-05-31_thesis_novelty_gaps.md`."

## Gotchas
- Never hand-edit `knowledge/*.csv` or `*.md` views — `kb.py` regenerates views; manual edits get clobbered.
- The thesis is in good shape; audit against the actual `.tex`, NOT `thesis_deliverables.md` (its claim paraphrases are stale, and its "every number in EVIDENCE_LEDGER" rule should now read "in knowledge/ledger").
- Do the restructure BEFORE big new work, so paths don't churn under you.
- Thesis files: **`docs/thesis_chapters.tex` is the real thesis**; the skill edits the synced COPY
  `docs/thesis_working.tex` (port accepted edits back to it / Overleaf); `docs/tesi_master.tex` is the
  professor's template (structure target only, never the content).
- The Stop-hook only catches git-tracked script changes — new files in gitignored dirs (e.g. `scratch/`) won't nag.
