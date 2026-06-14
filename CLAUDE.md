When tasks involve 3+ independent subtasks, use parallel sub-agents automatically.

# Knowledge system — the method (MANDATORY)

This repo has a structured knowledge system under `knowledge/`. It is the source of truth for
scripts, models, evals, and findings. Full spec: `knowledge/README.md`. The CSVs are the
database; `knowledge/_tools/kb.py` is the ONLY sanctioned way to write them.

Follow these rules every session:

1. **Search before you write a script.** Before creating any new `.py`, grep `knowledge/scripts.csv`
   for an existing `canonical` script with the same purpose. If one exists, **extend it** instead
   of writing a near-duplicate. Only write new when nothing fits; then record it.

2. **Record after you produce a number or an artifact.**
   - New/changed script → `/record` (or `py knowledge/_tools/kb.py record scripts ...`).
   - New model / weights → record a `models` row with provenance (`trained_from_script`, dataset).
   - Any eval that emits metrics → **first** `py knowledge/_tools/kb.py check-eval --target T --config C`
     to avoid reruns; if new, record an `evals` row (+ a `ledger` row if it's a finding).
   - Verify each number against its source file before recording it (no guessed metrics).

3. **Never hand-edit** the `knowledge/*.csv` files or the generated `.md` views (`knowledge/*.md`,
   `knowledge/views/*.md`). Use `kb.py record` / `kb.py set`. Views regenerate automatically.

4. **Cleanup is incremental and reversible.** When a script/model is absorbed or dead, mark it
   `kb.py set <table> <id> lifecycle=safe-to-archive` (+ `absorbed_into=`). Never delete or move
   files yourself — physical archiving happens only via `/sweep` on the user's green light.

5. **Schema/organization changes** to the knowledge system go in `knowledge/DECISIONS.md`.

6. **At session end**, update the `📍 Resume Here` block in `knowledge/PROJECT_STATE.md` (or run
   `/resume`) so the next session re-orients fast.

`docs/EVIDENCE_LEDGER.md` is being migrated into `knowledge/ledger.csv` — do not treat it as the
long-term home for new metrics; record into the knowledge system instead.
