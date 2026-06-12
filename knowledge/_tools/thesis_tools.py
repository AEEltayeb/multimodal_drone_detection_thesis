#!/usr/bin/env python3
"""thesis_tools.py — deterministic engines behind the `thesis` skill's act/util modes.
(The reasoning modes — draft/structure/readability/novelty/coherence-analysis/do-all — are
agent-driven per SKILL.md; this module is the tooling they and the user call.)

Subcommands:
  hygiene  [tex]                 LaTeX lint: undefined \\ref, duplicate \\label, undefined \\cite,
                                 and prose result-numbers missing a `% [source: ...]` comment (rule #2)
  table    <config_id> [--out f] generate a booktabs LaTeX results table from evals.csv (no drift)
  plot     <config_id> [--kind bar|scatter] [--out f]   matplotlib chart from a config's evals
  compile  [tex]                 smoke-build the thesis (latexmk/pdflatex+biber); report errors
  diff                           git diff thesis_chapters.tex (original) vs thesis_working.tex (copy)

Stdlib only except `plot` (needs matplotlib). Pairs with thesis_audit.py (numbers+figures).
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

REPO = kb.REPO
DOCS = REPO / "docs"
DEFAULT_TEX = [DOCS / "thesis_working.tex", DOCS / "thesis_chapters.tex"]


def _tex(argv_path=None) -> Path:
    if argv_path:
        return Path(argv_path)
    for p in DEFAULT_TEX:
        if p.exists():
            return p
    sys.exit("no thesis .tex found")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --- hygiene ----------------------------------------------------------------
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:auto|c|C|eq|page|name)?ref\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")
BIBKEY_RE = re.compile(r"@\w+\{([^,\s]+)\s*,")
RESULT_RE = re.compile(r"(?<![\d.])0?\.\d{2,}\b|\b\d{1,3}\.\d+\s*\\?%|\b\d{1,3}\s*\\?%")

# humanify / examiner word lists (AI-tells, over-certainty, marketing, interpretation cues)
FILLER = ["furthermore", "moreover", "additionally", "notably", "significantly", "importantly",
          "crucially", "it is important to note", "it should be noted", "in conclusion",
          "overall,", "as such,", "in today's world", "delve", "leverage"]
OVERCERTAIN = ["proves", "proven", "demonstrates", "clearly", "obviously", "undoubtedly",
               "definitively", "guarantees", "ensures", "certainly", "without doubt", "always"]
MARKETING = ["state-of-the-art", "cutting-edge", "revolutionary", "groundbreaking", "seamless",
             "powerful", "best-in-class", "world-class", "game-chang"]
INTERP = ["because", "since", "due to", "suggests", "indicates", "implies", "explains",
          "which means", "attributable", "driven by", "consistent with", "reflects",
          "owing to", "we attribute", "this is because", "as a result of"]
CLAIM_VERBS = ["outperforms", "improves", "achieves", "reduces", "increases", "beats",
               "surpasses", "exceeds", "boosts"]


def _prose_lines(text):
    """Yield (lineno, stripped_line) for prose only (skip tables, comments, command lines)."""
    in_tab = False
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if "\\begin{tab" in s or "\\begin{longtable" in s:
            in_tab = True
        if "\\end{tab" in s or "\\end{longtable" in s:
            in_tab = False; continue
        if in_tab or not s or s.startswith("%") or s.startswith("\\") or "&" in s:
            continue
        yield i, s


def _hits(s, words):
    low = s.lower()
    return [w for w in words if w in low]


def cmd_hygiene(args) -> int:
    tex = _tex(args.path)
    text = tex.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    labels, dup = set(), []
    for m in LABEL_RE.finditer(text):
        if m.group(1) in labels:
            dup.append(m.group(1))
        labels.add(m.group(1))
    refs = {k for m in REF_RE.finditer(text) for k in [m.group(1)]}
    undefined_refs = sorted(r for r in refs if r not in labels)

    cites = {c.strip() for m in CITE_RE.finditer(text) for c in m.group(1).split(",")}
    bibkeys = set()
    for bib in DOCS.glob("*.bib"):
        bibkeys |= set(BIBKEY_RE.findall(bib.read_text(encoding="utf-8", errors="replace")))
    undefined_cites = sorted(c for c in cites if c and bibkeys and c not in bibkeys) if bibkeys else []

    # prose result-numbers missing a source comment (rule #2). skip table rows + comment lines.
    in_tab = False
    missing_src = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if "\\begin{tab" in s or "\\begin{longtable" in s:
            in_tab = True
        if "\\end{tab" in s or "\\end{longtable" in s:
            in_tab = False; continue
        if in_tab or s.startswith("%") or "&" in s or "\\usepackage" in s or "\\includegraphics" in s:
            continue
        if RESULT_RE.search(ln):
            window = " ".join(lines[i-1:i+2])
            if "% [source" not in window:
                missing_src.append((i, s[:80]))

    print(f"# hygiene — {tex.name}")
    print(f"undefined \\ref ({len(undefined_refs)}): {undefined_refs[:20]}")
    print(f"duplicate \\label ({len(dup)}): {sorted(set(dup))[:20]}")
    if bibkeys:
        print(f"undefined \\cite ({len(undefined_cites)}): {undefined_cites[:20]}")
    else:
        print("undefined \\cite: (no .bib found in docs/ — skipped)")
    print(f"prose result-numbers missing a % [source:] comment ({len(missing_src)}) — rule #2:")
    for ln, ctx in missing_src[:40]:
        print(f"  L{ln}: {ctx}")
    return 0


# --- table ------------------------------------------------------------------
def cmd_table(args) -> int:
    cfg = args.config_id
    evals = [e for e in kb._read("evals") if e.get("config_id") == cfg]
    if not evals:
        sys.exit(f"no evals for config_id={cfg}")
    names = {m["id"]: m.get("name", m["id"]) for m in kb._read("models")}
    cols = [c for c in ("precision", "recall", "f1", "fpr", "halluc_rate", "latency_ms")
            if any((e.get(c) or "").strip() for e in evals)]
    hdr = {"precision": "P", "recall": "R", "f1": "F1", "fpr": "FPR",
           "halluc_rate": "Halluc", "latency_ms": "ms"}
    out = [f"% generated from knowledge/evals.csv by thesis_tools.py table {cfg} — do not hand-edit",
           "\\begin{table}[H]\\centering",
           f"\\caption{{Results on \\texttt{{{cfg}}}.}}\\label{{tab:{cfg}}}",
           "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
           "Target & " + " & ".join(hdr[c] for c in cols) + " \\\\", "\\midrule"]
    for e in sorted(evals, key=lambda e: _num(e.get("f1")) or -1, reverse=True):
        name = names.get(e.get("target", ""), e.get("target", "")).replace("_", "\\_")
        row = [name] + [(e.get(c) or "--") for c in cols]
        out.append(" & ".join(row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex = "\n".join(out)
    if args.out:
        Path(args.out).write_text(tex + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(tex)
    return 0


# --- plot -------------------------------------------------------------------
def cmd_plot(args) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed — cannot plot")
    cfg = args.config_id
    evals = [e for e in kb._read("evals") if e.get("config_id") == cfg]
    if not evals:
        sys.exit(f"no evals for config_id={cfg}")
    out = Path(args.out) if args.out else (DOCS / "figures" / f"{cfg}_{args.kind}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if args.kind == "scatter":
        xs = [_num(e.get("recall")) for e in evals]
        ys = [_num(e.get("halluc_rate")) for e in evals]
        for e, x, y in zip(evals, xs, ys):
            if x is not None and y is not None:
                ax.scatter(x, y); ax.annotate(e.get("target", ""), (x, y), fontsize=7)
        ax.set_xlabel("recall"); ax.set_ylabel("halluc rate")
    else:  # bar of F1
        pairs = [(e.get("target", ""), _num(e.get("f1"))) for e in evals]
        pairs = [(t, v) for t, v in pairs if v is not None]
        pairs.sort(key=lambda p: p[1], reverse=True)
        ax.bar([t for t, _ in pairs], [v for _, v in pairs])
        ax.set_ylabel("F1"); plt.xticks(rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{cfg}")
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"wrote {out.relative_to(REPO)} — add \\includegraphics + caption + % [source: config={cfg}]")
    return 0


# --- compile ----------------------------------------------------------------
def cmd_compile(args) -> int:
    tex = _tex(args.path)
    import shutil
    tool = shutil.which("latexmk") or shutil.which("pdflatex")
    if not tool:
        print("no LaTeX toolchain (latexmk/pdflatex) on PATH — cannot compile here. "
              "Compile in Overleaf or install TeX.")
        return 0
    cmd = ([tool, "-pdf", "-interaction=nonstopmode", tex.name] if "latexmk" in tool
           else [tool, "-interaction=nonstopmode", tex.name])
    r = subprocess.run(cmd, cwd=tex.parent, capture_output=True, text=True, timeout=300)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("!") or "Undefined" in l]
    print(f"compile via {Path(tool).name}: exit={r.returncode}; {len(errs)} error line(s)")
    for e in errs[:30]:
        print("  " + e)
    return 0 if r.returncode == 0 else 1


# --- diff -------------------------------------------------------------------
def cmd_diff(_args) -> int:
    orig, work = DOCS / "thesis_chapters.tex", DOCS / "thesis_working.tex"
    if not work.exists():
        sys.exit("docs/thesis_working.tex missing")
    r = subprocess.run(["git", "diff", "--no-index", "--stat", str(orig), str(work)],
                       cwd=REPO, capture_output=True, text=True)
    print(r.stdout or "(no differences)")
    print("\nfull diff: git diff --no-index docs/thesis_chapters.tex docs/thesis_working.tex")
    return 0


# --- humanify (lint AI-tells; agent rewrites, ledger-calibrated) -----------
def cmd_humanify(args) -> int:
    import statistics
    tex = _tex(args.path)
    text = tex.read_text(encoding="utf-8", errors="replace")
    filler, certain, marketing, prose = [], [], [], []
    for i, s in _prose_lines(text):
        for w in _hits(s, FILLER):
            filler.append((i, w))
        for w in _hits(s, OVERCERTAIN):
            certain.append((i, w))
        for w in _hits(s, MARKETING):
            marketing.append((i, w))
        prose.append(s)
    sents = [x for x in re.split(r"(?<=[.!?])\s+", " ".join(prose)) if len(x.split()) > 2]
    lens = [len(x.split()) for x in sents]
    mean = round(statistics.mean(lens), 1) if lens else 0
    sd = round(statistics.pstdev(lens), 1) if len(lens) > 1 else 0
    runons = sum(1 for x in lens if x > 40)
    print(f"# humanify lint — {tex.name}")
    print(f"AI-filler words ({len(filler)}): " + ", ".join(f"L{i}:{w}" for i, w in filler[:25]))
    print(f"over-certain language ({len(certain)}): " + ", ".join(f"L{i}:{w}" for i, w in certain[:25]))
    print(f"marketing language ({len(marketing)}): " + ", ".join(f"L{i}:{w}" for i, w in marketing[:25]))
    print(f"sentence length: mean={mean} sd={sd} (low sd => uniform/robotic); run-ons>40w: {runons}")
    print("AGENT: rewrite to remove filler, vary length, drop marketing; CALIBRATE certainty to the "
          "ledger — supported+high stays firm, conditional/partial -> hedge (suggests/indicates). "
          "Preserve every number/claim/citation; keep the author's voice.")
    return 0


# --- examiner (adversarial read; agent records to review.csv + autofixes) --
def cmd_examiner(args) -> int:
    tex = _tex(args.path)
    text = tex.read_text(encoding="utf-8", errors="replace")
    uninterp, maybe_uncited = [], []
    for i, s in _prose_lines(text):
        if RESULT_RE.search(s) and not _hits(s, INTERP):
            uninterp.append((i, s[:85]))
        if _hits(s, CLAIM_VERBS) and "\\cite" not in s:
            maybe_uncited.append((i, s[:85]))
    print(f"# examiner lint — {tex.name}")
    print(f"results stated without interpretation ({len(uninterp)}) — weakness_type=uninterpreted-result:")
    for i, s in uninterp[:40]:
        print(f"  L{i}: {s}")
    print(f"comparative claims with no nearby \\cite ({len(maybe_uncited)}) — verify supported/cited:")
    for i, s in maybe_uncited[:25]:
        print(f"  L{i}: {s}")
    print("AGENT: read each section adversarially (logical jumps, weak arguments, overclaims, "
          "uninterpreted results). Record findings: kb.py record review section=.. weakness=.. "
          "weakness_type=.. severity=.. suggestion=.. . Then AUTOFIX in thesis_working.tex ONLY where "
          "the fix is backed by a ledger/eval row (add interpretation + % [source:]); leave flagged "
          "(status=open) anything you cannot back with evidence — never invent.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="thesis_tools.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hygiene"); p.add_argument("path", nargs="?"); p.set_defaults(func=cmd_hygiene)
    p = sub.add_parser("table"); p.add_argument("config_id"); p.add_argument("--out"); p.set_defaults(func=cmd_table)
    p = sub.add_parser("plot"); p.add_argument("config_id"); p.add_argument("--kind", choices=["bar", "scatter"], default="bar"); p.add_argument("--out"); p.set_defaults(func=cmd_plot)
    p = sub.add_parser("compile"); p.add_argument("path", nargs="?"); p.set_defaults(func=cmd_compile)
    p = sub.add_parser("diff"); p.set_defaults(func=cmd_diff)
    p = sub.add_parser("humanify"); p.add_argument("path", nargs="?"); p.set_defaults(func=cmd_humanify)
    p = sub.add_parser("examiner"); p.add_argument("path", nargs="?"); p.set_defaults(func=cmd_examiner)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
