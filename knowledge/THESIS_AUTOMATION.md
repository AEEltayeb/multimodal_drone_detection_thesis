# Thesis automation — DESIGN SPEC (not built)

**Status:** blueprint, 2026-05-30. Build AFTER the `knowledge/` population is complete and
`docs/EVIDENCE_LEDGER.md` is retired — the thesis audit is only as trustworthy as the ledger
it checks against. Prereq: confirm the path of `thesis.tex` (locate it first).

## Confirmed decisions (2026-05-30)
- **Mode = audit + author-assist (aggressive):** rewrites overstated/wrong claims in the working
  copy AND drafts missing sections from evidence — every edit cited to a `ledger`/`evals` id.
- **Target = a working COPY, full edit authority, no git branch.** Source lives in BOTH Overleaf
  and local-in-repo, so: keep `thesis_working.tex` in-repo as the canonical working copy Claude
  edits; for the Overleaf side it's a copy-in / copy-out paste workflow (Claude can't reach Overleaf).
  You diff the working copy and port what you accept back to your real thesis / Overleaf.
- **Delivery = ONE `thesis` skill with modes** (`audit` / `draft` / `novelty` / `email`) so any chat
  invokes it automatically. Not separate skills.
- **Integrity principle (hard-wired):** only writes claims that trace to a `ledger`/`evals` row,
  cites the evidence inline (LaTeX comment), never invents numbers/findings; you own final voice.

## Core principle
The thesis is a **claim database**, and verification is a **layer over `knowledge/`** — NOT a
new `research/` tree. Every thesis claim resolves to existing rows: `ledger.csv` (findings:
`outcome`, `evidence_evals`, `contradicts`, `thesis_contribution`) and `evals.csv` (numbers
with `source_script` + `cache_path`). Add only what's missing: a `claims` table and a
thesis-scoped `figures` table. Reuse `kb.py` (record/set/views/validate) — same discipline.

## New tables (CSV source-of-truth + generated `.md` view, via kb.py)
**`claims`** — one row per extracted thesis assertion:
`id, chapter, tex_location, claim_text, kind{metric|qualitative|figure|table}, status{supported|partial|unsupported|contradicted|unverified}, evidence (ledger/eval ids), confidence{high|med|low}, suggested_rewording, notes`

**`figures`** — one row per `\includegraphics` / thesis table:
`id, tex_path, kind{figure|table}, generated_by (scripts.id), source_eval (evals.id), status{verified|stale|orphan}, notes`

## Commands to build (slash commands → engine `knowledge/_tools/thesis_audit.py`)
- **`/audit-thesis`** — parse `thesis.tex`; extract claims, numbers, `\includegraphics`, tables.
  1. **Number check (deterministic, highest value):** every number in the tex must trace to an
     `evals` row (within tolerance); unmatched → flag as *stale/unverified* (catches copied old numbers).
  2. **Figure/table provenance:** each `\includegraphics` path → generating script → eval; mark verified/stale/orphan.
  3. **Claim status (agent reasoning):** match each qualitative claim to `ledger` findings →
     SUPPORTED / PARTIAL / UNSUPPORTED / CONTRADICTED, with a suggested rewording when overstated
     (e.g. a `conditional` finding behind an absolute claim).
  Writes `claims.csv` + `figures.csv` + a dated `docs/analysis/THESIS_AUDIT_<date>.md` report.
- **`/find-novelty`** — `ledger` rows with `thesis_contribution` set but **not** cited in `thesis.tex`
  → "evidence exists, mentioned in 0 chapters." (Half-built: contributions already flagged in ledger.)
- **`/progress-email`** — `git log` since last email + recent `ledger`/`evals`/`PROJECT_STATE` deltas
  → drafts a supervisor email in the user's voice (signature: Ahmed). Stored under `knowledge/progress/`.

## Tractable vs judgment
- **Deterministic, do first:** number verification, figure provenance, progress email.
- **Agent-reasoning each run:** qualitative claim status, novelty scoring, contradiction phrasing.

## v2 modes (designed 2026-05-31 — to implement)
Pattern: **analyze → record (consumable table) → act**. Mode taxonomy:
- **check** (read-only, write a table): `audit`, `coherence`, `hygiene`, `novelty`
- **act** (edit the working copy): `draft`, `edit`, `plot`, `table`, `structure`, `readability`
- **util**: `compile`, `diff`, `refs`, `email`
- **orchestrator**: `do-all`

New modes + decisions:
- **`do-all`** — runs check→act→verify, **no-op-skipping** (no figures needed → none added).
  **Autonomous**, then presents **one diff** at the end. Run iteratively as the thesis matures
  (readability/coherence are wasted on stub chapters).
- **`structure`** (folds coherence + organization + template-conformance; writes a `coherence`/
  structure-gap table the act-modes consume). **Target = `docs/tesi_master.tex` skeleton (verified
  IMRAD, 5 chapters):** Intro · Background & Related Work · Contribution/System · Empirical
  Evaluation (Study Design→Goal/**Research Questions**, Context, Data Analysis · Result · Discussion ·
  **Threats to Validity**) · Conclusion. Consolidation map: current Ch3+Ch4→tesi Ch3; current
  Ch5+6+7+8→tesi Ch4. Add explicit RQs + Threats (missing today). Leverages ledger `contradicts`
  for claim-vs-claim incoherence.
- **`readability`** (prose micro-pass, separate from `structure`) — **applies clarity edits directly**
  (you review the diff). HARD rule: never alters a number, claim, citation, or the author's voice/meaning;
  formal-academic register, just clearer.
- **`coherence`** output is a **consumable table** (location, issue, type, severity) usable by
  `edit`/`structure` later — per the user's "usable by the skill itself" requirement.
- **`examiner`** (added 2026-05-31) — adversarial committee read (Reviewer #2): unsupported claims,
  logical jumps, weak arguments, overclaims, **results-not-interpreted**. Writes a NEW **`review`** table
  (section, weakness, weakness_type, severity, suggestion) and **autofixes** in the working copy ONLY
  where a `ledger`/`evals` row backs the fix; otherwise leaves `status=open`. Linter: `thesis_tools.py examiner`.
- **`humanify`** (added 2026-05-31, SEPARATE from readability) — de-AI voice: strip filler, vary sentence
  length, drop marketing, active voice. **Key edge: certainty calibrated to the ledger** — `supported`+`high`
  stays firm; `conditional`/`partial`/`low` → hedge. Never alters numbers/claims/citations/meaning. Framing =
  researcher register, not detection-evasion. Linter: `thesis_tools.py humanify`.

## Sequencing
0. Locate `thesis.tex`; confirm it's the live document (vs `archive/report/main.tex`).
1. Finish `knowledge/` population (detector + classifier census + ledger detail-sweep reconciliation).
2. Retire `EVIDENCE_LEDGER.md` (mark safe-to-archive).
3. Build `claims`/`figures` tables + `thesis_audit.py` + the 3 commands.
4. First `/audit-thesis` pass → triage the report with the supervisor's eye.
