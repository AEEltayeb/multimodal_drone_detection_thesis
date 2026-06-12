#!/usr/bin/env python3
"""kb.py — the knowledge/ system backbone (canonical script).

The ONLY sanctioned writer to the knowledge/ tables. Makes the correct action the
easiest action: validates schema + enums, auto-fills derived fields, and regenerates
the .md views so they are never stale. See knowledge/README.md for the design.

Subcommands
-----------
  record   <table> k=v [k=v ...]   append a validated row, then regen views
  set      <table> <id> k=v ...    update an existing row (e.g. lifecycle), regen
  mv       <table> <id> <newpath>  move file on disk (git mv / Move) + update row path, regen
  views                            regenerate all .md views from the CSVs
  validate                         check every CSV against schema + enums
  check-eval --target T --config C rerun-guard: is there an eval (+ live cache)?
  sweep                            list rows with lifecycle == safe-to-archive

Stdlib only (csv, argparse) — no pandas dependency, so it always runs.
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

KB = Path(__file__).resolve().parent.parent          # knowledge/
REPO = KB.parent                                       # repo root
VIEWS = KB / "views"

# --- Schema -----------------------------------------------------------------
# table -> ordered columns. First column is always the primary id.
SCHEMA = {
    "scripts": ["id", "path", "purpose", "inputs", "outputs", "role", "lifecycle",
                "supersedes", "absorbed_into", "produces_models", "produces_evals",
                "reproduce_cmd", "last_run"],
    "models": ["id", "name", "type", "purpose_tags", "trained_from_script",
               "train_dataset", "weights_path", "provenance_notes", "production",
               "lifecycle"],
    "eval_configs": ["id", "dataset", "n_samples", "imgsz", "scoring_rule",
                     "conf_thr", "notes"],
    "evals": ["id", "date", "target", "config_id", "precision", "recall", "f1",
              "fpr", "halluc_rate", "latency_ms", "extra", "cache_path",
              "source_script", "ledger_ids"],
    "ledger": ["id", "date", "claim", "outcome", "condition", "evidence_evals",
               "contradicts", "thesis_contribution", "status", "notes"],
    # thesis-automation tables (see knowledge/THESIS_AUTOMATION.md)
    "claims": ["id", "chapter", "tex_location", "claim_text", "kind", "verdict",
               "evidence", "confidence", "suggested_rewording", "notes"],
    "figures": ["id", "tex_path", "kind", "generated_by", "source_eval", "fig_status", "notes"],
    "coherence": ["id", "location", "issue", "issue_type", "severity", "related", "status", "notes"],
    "review": ["id", "section", "weakness", "weakness_type", "severity", "suggestion", "status", "notes"],
    # dataset registry (see knowledge/DECISIONS.md 2026-06-05; eval_configs.dataset FKs to datasets.id)
    "datasets": ["id", "name", "modality", "usage", "size", "physical_root", "scoring",
                 "provenance", "redistributable", "thesis_ref", "notes"],
}

# column -> allowed values (validated; case-sensitive)
ENUMS = {
    "role": {"canonical", "one-off", "library"},
    "lifecycle": {"active", "superseded", "absorbed", "safe-to-archive", "archived"},
    "outcome": {"supported", "partial", "refuted", "conditional"},
    "status": {"open", "confirmed"},
    "scoring_rule": {"iou", "iop"},
    "kind": {"metric", "qualitative", "figure", "table"},
    "verdict": {"supported", "partial", "unsupported", "contradicted", "unverified"},
    "fig_status": {"verified", "stale", "orphan"},
    "confidence": {"high", "med", "low"},
    "issue_type": {"contradiction", "gap", "flow", "undefined", "redundancy", "structure"},
    "severity": {"high", "med", "low"},
    "weakness_type": {"unsupported", "missing-citation", "logical-jump", "weak-argument",
                      "uninterpreted-result", "overclaim"},
}

# required (non-empty) columns per table
REQUIRED = {
    "scripts": ["path", "purpose", "role", "lifecycle"],
    "models": ["name", "type", "lifecycle"],
    "eval_configs": ["dataset"],
    "evals": ["target", "config_id"],
    "ledger": ["claim", "outcome"],
    "claims": ["claim_text"],
    "figures": ["tex_path"],
    "coherence": ["issue"],
    "review": ["weakness"],
    "datasets": ["name", "modality"],
}

DATE_COLS = {"last_run", "date"}

# table -> the column that holds an on-disk path (used by `mv`)
PATH_COL = {"scripts": "path", "models": "weights_path"}


def _today() -> str:
    return _dt.date.today().isoformat()


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return s[:48] or "row"


def _csv_path(table: str) -> Path:
    if table not in SCHEMA:
        sys.exit(f"unknown table '{table}'. known: {', '.join(SCHEMA)}")
    return KB / f"{table}.csv"


def _read(table: str) -> list[dict]:
    p = _csv_path(table)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(table: str, rows: list[dict]) -> None:
    cols = SCHEMA[table]
    with _csv_path(table).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# --- validation -------------------------------------------------------------
def _validate_row(table: str, row: dict) -> list[str]:
    errs = []
    cols = SCHEMA[table]
    for k in row:
        if k not in cols:
            errs.append(f"unknown column '{k}'")
    for c in REQUIRED.get(table, []):
        if not (row.get(c) or "").strip():
            errs.append(f"missing required '{c}'")
    for c, allowed in ENUMS.items():
        v = (row.get(c) or "").strip()
        if c in cols and v and v not in allowed:
            errs.append(f"'{c}'='{v}' not in {sorted(allowed)}")
    return errs


def cmd_validate(_args) -> int:
    bad = 0
    for table in SCHEMA:
        for i, row in enumerate(_read(table), start=2):  # +1 header, +1 1-index
            errs = _validate_row(table, row)
            if errs:
                bad += 1
                print(f"[{table}.csv:{i}] {row.get('id','?')}: {'; '.join(errs)}")
    print("OK — all rows valid." if not bad else f"{bad} invalid row(s).")
    return 1 if bad else 0


# --- record -----------------------------------------------------------------
def cmd_record(args) -> int:
    table = args.table
    fields = {}
    for kv in args.fields:
        if "=" not in kv:
            sys.exit(f"bad field '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        fields[k.strip()] = v

    # auto-fill derived fields
    for dc in DATE_COLS:
        if dc in SCHEMA[table] and not fields.get(dc):
            fields[dc] = _today()
    if not fields.get("id"):
        seed = fields.get("path") or fields.get("name") or fields.get("claim") \
            or fields.get("target") or fields.get("dataset") or "row"
        base = _slug(Path(seed).stem if "/" in seed or "\\" in seed else seed)
        existing = {r["id"] for r in _read(table)}
        cand, n = base, 2
        while cand in existing:
            cand = f"{base}_{n}"; n += 1
        fields["id"] = cand

    errs = _validate_row(table, fields)
    if errs:
        sys.exit(f"refusing to record — {'; '.join(errs)}")

    rows = _read(table)
    if any(r["id"] == fields["id"] for r in rows):
        sys.exit(f"id '{fields['id']}' already exists in {table}.csv (use a unique id)")

    # eval cache sanity check (non-fatal warning)
    if table == "evals" and fields.get("cache_path"):
        cp = (REPO / fields["cache_path"])
        if not cp.exists():
            print(f"WARN: cache_path does not exist: {fields['cache_path']}")

    rows.append(fields)
    _write(table, rows)
    print(f"recorded {table}.{fields['id']}")
    _regen_views()
    return 0


# --- set (update existing row) ---------------------------------------------
def cmd_set(args) -> int:
    table = args.table
    updates = {}
    for kv in args.fields:
        if "=" not in kv:
            sys.exit(f"bad field '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        updates[k.strip()] = v
    rows = _read(table)
    target = next((r for r in rows if r["id"] == args.id), None)
    if target is None:
        sys.exit(f"id '{args.id}' not found in {table}.csv")
    merged = {**target, **updates}
    errs = _validate_row(table, {k: v for k, v in merged.items() if k in SCHEMA[table]})
    if errs:
        sys.exit(f"refusing to update — {'; '.join(errs)}")
    target.update(updates)
    _write(table, rows)
    print(f"updated {table}.{args.id}: {', '.join(updates)}")
    _regen_views()
    return 0


# --- check-eval (rerun guard) ----------------------------------------------
def cmd_check_eval(args) -> int:
    hits = [r for r in _read("evals")
            if r.get("target") == args.target and r.get("config_id") == args.config]
    if not hits:
        print(f"NO EXISTING EVAL for target={args.target} config={args.config} — safe to run.")
        return 0
    for r in hits:
        cp = r.get("cache_path", "")
        live = "LIVE" if cp and (REPO / cp).exists() else "MISSING"
        print(f"FOUND eval '{r['id']}' (f1={r.get('f1','')}) cache={cp or '-'} [{live}] "
              f"— REUSE, do not rerun.")
    return 0


# --- sweep ------------------------------------------------------------------
def cmd_sweep(_args) -> int:
    found = False
    for table in ("scripts", "models"):
        for r in _read(table):
            if r.get("lifecycle") == "safe-to-archive":
                found = True
                print(f"[{table}] {r['id']}: {r.get('path') or r.get('name')} "
                      f"-> {r.get('absorbed_into') or '(unneeded)'}")
    if not found:
        print("nothing marked safe-to-archive.")
    print("\n(physical move waits for your green light; sweep relocates to archive/<date>/)")
    return 0


# --- mv (move file on disk + update its row atomically) ---------------------
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO),
                          capture_output=True, text=True)


def _is_tracked(rel: str) -> bool:
    return _git("ls-files", "--error-unmatch", "--", rel).returncode == 0


def cmd_mv(args) -> int:
    table = args.table
    col = args.col or PATH_COL.get(table)
    if not col:
        sys.exit(f"no path column known for table '{table}'; pass --col")
    if col not in SCHEMA[table]:
        sys.exit(f"column '{col}' not in {table} schema")

    rows = _read(table)
    target = next((r for r in rows if r["id"] == args.id), None)
    if target is None:
        sys.exit(f"id '{args.id}' not found in {table}.csv")

    old = (target.get(col) or "").strip().replace("\\", "/")
    if not old:
        sys.exit(f"row '{args.id}' has empty '{col}' — nothing to move")

    new = args.new_path.replace("\\", "/")
    # If the destination is an existing dir or ends with '/', keep the basename.
    if args.new_path.endswith("/") or (REPO / new).is_dir():
        new = f"{new.rstrip('/')}/{Path(old).name}"

    if new == old:
        sys.exit("new path == old path; nothing to do")

    old_abs, new_abs = REPO / old, REPO / new

    if old_abs.exists():
        if new_abs.exists():
            sys.exit(f"refusing to overwrite existing dest: {new}")
        new_abs.parent.mkdir(parents=True, exist_ok=True)
        if _is_tracked(old):
            r = _git("mv", "--", old, new)
            if r.returncode != 0:
                sys.exit(f"git mv failed: {r.stderr.strip() or r.stdout.strip()}")
        else:
            shutil.move(str(old_abs), str(new_abs))
        print(f"moved {old} -> {new}")
    elif new_abs.exists():
        print(f"WARN: source missing but dest already present — updating row only")
    else:
        sys.exit(f"source path does not exist: {old}")

    target[col] = new
    _write(table, rows)
    print(f"updated {table}.{args.id}: {col}={new}")
    _regen_views()
    return 0


# --- view generation --------------------------------------------------------
def _md_cell(v: str) -> str:
    return (v or "").replace("|", "\\|").replace("\n", "<br>")


def _md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_md_cell(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _gen_table_view(table: str) -> None:
    rows = _read(table)
    body = (f"<!-- GENERATED from {table}.csv by knowledge/_tools/kb.py — do not hand-edit. -->\n\n"
            f"# {table} ({len(rows)} rows)\n\n")
    body += _md_table(rows, SCHEMA[table]) if rows else "_empty_\n"
    (KB / f"{table}.md").write_text(body + "\n", encoding="utf-8")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _gen_rankings() -> None:
    models = _read("models")
    evals = _read("evals")
    by_target = {}
    for e in evals:
        by_target.setdefault(e.get("target"), []).append(e)

    purposes = sorted({t.strip() for m in models for t in
                       (m.get("purpose_tags") or "").split(";") if t.strip()})
    out = ["<!-- GENERATED by knowledge/_tools/kb.py — do not hand-edit. -->\n",
           "# Model rankings (generated)\n",
           "Per purpose, the meaningful metric differs: detection purposes rank by best F1 "
           "(higher=better); confusion-filter ranks by lowest hallucination rate "
           "(lower=better, the point of a filter).\n"]
    if not purposes:
        out.append("_No models with purpose_tags yet._")
    # purpose -> (metric_col, higher_is_better, label)
    metric_for = {"confusion-filter": ("halluc_rate", False, "best_halluc")}
    for p in purposes:
        col, higher, label = metric_for.get(p, ("f1", True, "best_f1"))
        out.append(f"\n## {p}  _(by {label})_\n")
        ranked = []
        for m in models:
            if p in [t.strip() for t in (m.get("purpose_tags") or "").split(";")]:
                vals = [_f(e.get(col)) for e in by_target.get(m["id"], [])]
                vals = [x for x in vals if x is not None]
                best = (max(vals) if higher else min(vals)) if vals else None
                ranked.append((best, m["id"], len(by_target.get(m["id"], []))))
        # sort: rows with a value first; best value by direction
        ranked.sort(key=lambda t: (t[0] is None,
                                   (-(t[0] or 0)) if higher else (t[0] if t[0] is not None else 1e9)))
        rows = [{"model": mid, label: "" if b is None else f"{b:.4f}".rstrip("0").rstrip("."),
                 "n_evals": str(n)} for b, mid, n in ranked]
        out.append(_md_table(rows, ["model", label, "n_evals"]) if rows else "_none_")
    VIEWS.mkdir(exist_ok=True)
    (VIEWS / "rankings.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def _gen_comparisons() -> None:
    evals = _read("evals")
    by_cfg = {}
    for e in evals:
        by_cfg.setdefault(e.get("config_id"), []).append(e)
    metrics = ["precision", "recall", "f1", "fpr", "halluc_rate", "latency_ms"]
    out = ["<!-- GENERATED by knowledge/_tools/kb.py — do not hand-edit. -->\n",
           "# Model comparisons by eval_config (generated)\n",
           "Same config_id = apples-to-apples (same dataset, n_samples, imgsz, scoring).\n"]
    if not by_cfg:
        out.append("_No evals yet._")
    for cfg in sorted(by_cfg):
        out.append(f"\n## config: `{cfg}`\n")
        rows = [{"target": e.get("target", ""), **{m: e.get(m, "") for m in metrics}}
                for e in sorted(by_cfg[cfg],
                                key=lambda e: _f(e.get("f1")) or -1, reverse=True)]
        out.append(_md_table(rows, ["target"] + metrics))
    VIEWS.mkdir(exist_ok=True)
    (VIEWS / "comparisons.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def _regen_views() -> None:
    for table in SCHEMA:
        _gen_table_view(table)
    _gen_rankings()
    _gen_comparisons()


def cmd_views(_args) -> int:
    _regen_views()
    print("regenerated: *.md table views + views/rankings.md + views/comparisons.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="kb.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="append a validated row")
    pr.add_argument("table", choices=list(SCHEMA))
    pr.add_argument("fields", nargs="+", help="key=value pairs")
    pr.set_defaults(func=cmd_record)

    ps = sub.add_parser("set", help="update an existing row")
    ps.add_argument("table", choices=list(SCHEMA))
    ps.add_argument("id")
    ps.add_argument("fields", nargs="+", help="key=value pairs")
    ps.set_defaults(func=cmd_set)

    pm = sub.add_parser("mv", help="move a file on disk + update its row atomically")
    pm.add_argument("table", choices=list(SCHEMA))
    pm.add_argument("id")
    pm.add_argument("new_path", help="destination path (or dir) relative to repo root")
    pm.add_argument("--col", help="path column to update (default: path/weights_path)")
    pm.set_defaults(func=cmd_mv)

    sub.add_parser("views", help="regenerate .md views").set_defaults(func=cmd_views)
    sub.add_parser("validate", help="validate all CSVs").set_defaults(func=cmd_validate)
    sub.add_parser("sweep", help="list safe-to-archive").set_defaults(func=cmd_sweep)

    pc = sub.add_parser("check-eval", help="rerun guard")
    pc.add_argument("--target", required=True)
    pc.add_argument("--config", required=True)
    pc.set_defaults(func=cmd_check_eval)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
