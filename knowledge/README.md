# `knowledge/` — Project Knowledge System (DESIGN SPEC)

**Status:** blueprint, locked 2026-05-30. Not yet built. This file is the contract we
implement against. Nothing in `knowledge/` is operational until the bootstrap audit runs.

---

## 1. Why this exists

The repo has real knowledge infrastructure (MEMORY.md, EVIDENCE_LEDGER, docs/analysis)
but it **rotted** because there was no *enforced method*. Symptoms and root cause:

| Symptom | Root cause |
|---|---|
| Drift across sessions | No canonical schema for agents to conform to |
| No provenance ("recreate data X from script 105") | No enforced link script→data→finding |
| No reusability | Outputs aren't in a uniform, findable shape |
| Agents rewrite instead of reusing code | No script registry to check first |
| Script sprawl / overlap | No "extend the canonical script" rule |
| EVIDENCE_LEDGER is a wall of text | No structured store; free prose |

The fix is **upstream of where docs live**: a small relational model (CSV source-of-truth
+ generated views) plus a *method that is enforced* (CLAUDE.md law + commands + a hook).
Memory (`MEMORY.md`) stays a **thin index**; full understanding lives here, on disk, read
on demand.

---

## 2. The model (star schema)

Noun-things are **tables**. Anything you'd otherwise hand-maintain (rankings, comparisons,
project state) is a **generated view** over those tables — so it cannot drift.

```
                 eval_configs ─┐
                               │ (comparison protocol: dataset, n_frames, imgsz, scoring)
   scripts ──produces──▶ models│
      │                    │   ▼
      └──produces──▶ evals ◀┘   (one eval run; metrics + cache_path)
                       │
                       ▼
                    ledger        (findings/claims, cite eval ids, contradict ids)
```

- **scripts** — code provenance (the *how to reproduce*)
- **models** — what we have + per-purpose role (the *what's best for X*)
- **eval_configs** — the uniform comparison protocols (apples-to-apples guarantee)
- **evals** — what we measured + cache pointer (the *don't-rerun* guard)
- **ledger** — what we concluded (the *what we learned*)

Contradictions and thesis-contributions are **columns/sections**, not separate registries
(deliberate anti-sprawl).

---

## 3. Table schemas

Each table is `name.csv` (source of truth) + `name.md` (generated, equally-structured,
human-readable view). **Never hand-edit the `.md`.**

### `scripts.csv`
| col | meaning |
|---|---|
| id | slug, stable |
| path | repo-relative |
| purpose | one line — what it does |
| inputs | datasets/models/files consumed |
| outputs | data/plots/models produced |
| role | `canonical` \| `one-off` \| `library` (agent designates) |
| lifecycle | `active` \| `superseded` \| `absorbed` \| `safe-to-archive` \| `archived` |
| supersedes / absorbed_into | id(s) this replaced or was folded into |
| produces_models | model id(s) |
| produces_evals | eval id(s) |
| reproduce_cmd | exact command line |
| last_run | date |

### `models.csv`
| col | meaning |
|---|---|
| id | slug |
| name | human name |
| type | `rgb_yolo` \| `ir_yolo` \| `classifier` \| `verifier` \| `fusion` \| `mlp` \| … |
| purpose_tags | `drone-detection`, `confusion-filter`, `full-pipeline`, … (multi) |
| trained_from_script | script id |
| train_dataset | corpus name |
| weights_path | repo-relative |
| provenance_notes | free text |
| production | flag — currently shipped? |
| lifecycle | same enum as scripts |

### `eval_configs.csv` — the uniform comparison protocol
| col | meaning |
|---|---|
| id | slug, e.g. `svan_iop_1280` |
| dataset | corpus/subset |
| n_frames / n_clips | sample size |
| imgsz | inference size |
| scoring_rule | `iou` \| `iop` + threshold |
| conf_thr | detector confidence |
| notes | caveats |

> Encodes existing conventions: Svanström → `imgsz=1280` + `iop@0.5`; Anti-UAV → `iou`.

### `evals.csv` — one row per eval run
| col | meaning |
|---|---|
| id | slug |
| date | run date |
| target | model id **or** pipeline/stack id |
| config_id | eval_configs.id |
| precision / recall / f1 / fpr / halluc_rate / latency_ms | core metrics (blank if N/A) |
| extra | JSON blob for non-core metrics |
| cache_path | where predictions/results cache lives |
| source_script | script id that produced it |
| ledger_ids | finding(s) this feeds |

**Rerun-guard:** before running, query `evals` for `(target, config_id)`; if a row exists
with a live `cache_path`, **reuse — do not rerun the model.**

### `ledger.csv` — findings/claims (replaces EVIDENCE_LEDGER.md)
| col | meaning |
|---|---|
| id | slug |
| date | |
| claim | the hypothesis/finding, one line |
| outcome | `supported` \| `partial` \| `refuted` \| `conditional` |
| condition | for `conditional`: the boundary ("holds on Svanström, not rgb_test") |
| evidence_evals | eval id(s) backing it |
| contradicts | ledger id(s) it tensions with |
| thesis_contribution | flag + short note if candidate contribution |
| status | `open` \| `confirmed` |

---

## 4. Generated views (read these, don't maintain them)

Under `knowledge/views/`, regenerated from the CSVs:

- **`rankings.md`** — for each `purpose_tag` (confusion-filter / drone-detection /
  full-pipeline), models ranked by the relevant metric, restricted to comparable
  `eval_configs`. "Models that go well together" = full-pipeline rows.
- **`comparisons.md`** — per `eval_config`, all targets side-by-side. Uniform by
  construction (same n_frames, scoring, imgsz).

You **query** for "best confusion filter" — you never keep a hand-edited list.

---

## 5. `PROJECT_STATE.md` — the human pane

Advisor-facing single source of "where the project is." Pinned at the **top**, a brief
personal portal:

```
📍 Resume Here   (≤10 lines, updated every session-end by the hook)
- Last goal:
- Last step done:
- Next action:
- Open threads (incomplete; when to return):
```

Below that: current production stack, in-progress work, candidate thesis contributions
(pulled from ledger flags), recently resolved.

---

## 6. Lifecycle & slow cleanup (`safe-to-archive`)

Nothing is ever deleted directly. Lifecycle is a field on **scripts, models, docs,
notebooks, plots**:

```
active → superseded / absorbed → safe-to-archive → archived
```

- An artifact earns **`safe-to-archive`** only when confirmed unneeded **or** absorbed
  into a canonical thing.
- **Agent marks** `safe-to-archive` (a proposal, with a reason). **User green-lights**
  the sweep.
- The sweep physically moves files into the existing **`archive/<date>/<original-path>`**
  (one graveyard, not two). Git history makes every move reversible.

**First use case:** migrate `docs/EVIDENCE_LEDGER.md` → `knowledge/ledger.{csv,md}`, then
mark the old file `safe-to-archive` (reason: *absorbed into knowledge/ledger*). It stays
visible until the next green-lit sweep.

---

## 7. The method (enforcement)

This is the crux — without enforcement, the structure rots again in three sessions.

**`CLAUDE.md` = the law** (to be added when we build):
1. Before writing any new script: search `scripts.csv` by purpose. If a `canonical` one
   exists, **extend it**; else add a row (designate `role`).
2. After any run that emits a number: add an `evals` row (+ `ledger` row if it's a
   finding). First check `evals.csv` to avoid reruns.
3. Never hand-edit `*.md` views or generated tables.
4. Schema/organization changes → log in `DECISIONS.md`.
5. Update `📍 Resume Here` at session end.

**Commands = path of least resistance** (to build):
- `/record` — the funnel that keeps every fact uniform and every view fresh; the *only*
  sanctioned way to write to the tables. Its purpose is to make the correct action the
  easiest action — that's what actually enforces the method. It:
  - appends a correctly-shaped row to the right table (`scripts`/`models`/`evals`/`ledger`)
    and **validates the enums** (`role`, `lifecycle`, `outcome`) so no malformed entries;
  - **auto-fills derived fields** (id slug, date; for evals, checks whether `cache_path`
    exists);
  - **regenerates the `.md` view + dependent views** (`rankings`, `comparisons`) so they're
    never stale;
  - gives the Stop-hook one thing to check — "did a `/record` happen this session?".
  - *Example:* finish an eval → `/record eval` → asks target/config, reads metrics, writes
    the `evals` row, refreshes `comparisons.md` + `rankings.md`. You never touch a CSV.
- `/sweep` — list `safe-to-archive`, relocate on green light.
- `/resume` — show/update the portal.

**Hook = the backstop:** on session Stop, if `*.py` changed or new numbers were produced
but no `knowledge/*.csv` row was added → nag.

---

## 8. Evolution (`DECISIONS.md`)

The organization is allowed to evolve — but every schema/structure change is **recorded**,
or "evolving" just becomes drift again. Format:

```
2026-05-30 — added `eval_configs` table — reason: uniform model comparison without rerun.
```

---

## 9. Layout

```
knowledge/
  README.md            ← this spec
  PROJECT_STATE.md     ← human pane (📍 Resume Here on top)
  DECISIONS.md         ← schema/org change log
  ledger.csv  ledger.md
  models.csv  models.md
  scripts.csv scripts.md
  evals.csv   evals.md
  eval_configs.csv  eval_configs.md
  views/
    rankings.md        ← generated
    comparisons.md     ← generated
```

---

## 10. Bootstrap — the one-time forensic audit (NOT YET RUN)

> **The audit only inventories and *tags* — it archives NOTHING.** Almost everything is
> tagged `lifecycle: active`. Archiving is never a big-bang; it happens incrementally as
> you work (an item is marked `safe-to-archive` only once it's genuinely absorbed/dead,
> and moves only on your green light — see §6). "Audit" ≠ "archive the repo."

Build the schema/method first (this spec), *then* populate so outputs land already-shaped:

1. **Census** — sweep scripts, models, notebooks, plots, docs, **and `archive/`** (treat
   archived as first-class; archived ≠ obsolete). Classify each → populate `scripts` &
   `models` with `role` + `lifecycle` (default `active`). Flag duplicates / orphans /
   overlaps for *later* review — do not archive them now.
2. **Migrate** EVIDENCE_LEDGER → `ledger.csv`; mark old file `safe-to-archive` (the one
   archival action seeded by bootstrap, and it still waits for your green light).
3. **Backfill** `evals` + `eval_configs` from existing result CSVs / caches (so the
   rerun-guard works immediately).
4. **Write** `PROJECT_STATE.md` + first `DECISIONS.md` entry; generate first views.
5. Continue until repository coverage is high; surface contradictions and candidate
   contributions as ledger flags.
```
