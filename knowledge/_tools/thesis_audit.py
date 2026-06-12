#!/usr/bin/env python3
"""thesis_audit.py — deterministic half of the `thesis` skill's `audit` mode.

Checks a thesis .tex against the knowledge base:
  1. NUMBER verification: every %/decimal in the tex that LOOKS like a result metric is
     matched against values in knowledge/evals.csv. Unmatched -> flagged "verify" (may be
     stale/copied, or derived/cited). Catches the classic 'pasted an old number' bug.
  2. FIGURE provenance: each \\includegraphics -> resolve via \\graphicspath -> exists? ->
     guess generating script from scripts.csv. Writes knowledge/figures.csv rows.
  3. Writes a dated report under docs/analysis/.

Qualitative claim verification (SUPPORTED/CONTRADICTED + rewording) and drafting are done by
the agent in the skill — this engine only does the deterministic checks. Stdlib only.

Usage: py knowledge/_tools/thesis_audit.py [path/to/thesis.tex]
       (default: docs/thesis_working.tex, else docs/thesis_chapters.tex)
"""
from __future__ import annotations
import csv
import datetime as _dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

REPO = kb.REPO
DEFAULT_TEX = [REPO / "docs" / "thesis_working.tex", REPO / "docs" / "thesis_chapters.tex"]

NUM_COLS = ["precision", "recall", "f1", "fpr", "halluc_rate", "latency_ms"]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# tex result-number tokens: percentages, decimals 0.xx, and N.N (with optional %, pp, x)
TOKEN_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s*(\\?%|pp|×|x\b)?", re.IGNORECASE)
INCG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def _floats_from(s: str):
    out = []
    for m in NUM_RE.finditer(s or ""):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def eval_value_index():
    """Set of known result values (rounded), in both decimal and percentage form."""
    vals = set()
    for r in kb._read("evals"):
        nums = []
        for c in NUM_COLS:
            v = (r.get(c) or "").strip()
            if v:
                nums += _floats_from(v)
        nums += _floats_from(r.get("extra", ""))
        for x in nums:
            vals.add(round(x, 3))
            vals.add(round(x * 100, 1))   # 0.869 -> 86.9
            vals.add(round(x / 100, 4))   # 86.9 -> 0.869
    return vals


def _matches(v, idx):
    for cand in (round(v, 3), round(v, 1), round(v / 100, 4), round(v * 100, 1)):
        for known in idx:
            if abs(known - cand) <= 0.005:
                return True
    return False


def audit_numbers(text, idx):
    """Return (n_total, matched, [unmatched (value, line, context)]) for result-like numbers."""
    matched, unmatched, total = 0, [], 0
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("%") or "\\usepackage" in s or "geometry" in s or "fontsize" in s:
            continue
        for m in TOKEN_RE.finditer(line):
            raw, suffix = m.group(1), (m.group(2) or "")
            try:
                v = float(raw)
            except ValueError:
                continue
            is_pct = "%" in suffix
            # only treat as a result metric: a percentage, or a decimal in (0,1), or x.x with pp/x
            looks_metric = is_pct or ("." in raw and 0 < v < 1) or suffix in ("pp", "x", "×") \
                or ("." in raw and 1 < v < 100 and is_pct)
            if not looks_metric:
                continue
            total += 1
            if _matches(v, idx):
                matched += 1
            else:
                ctx = s[:90]
                unmatched.append((f"{raw}{suffix}", i, ctx))
    return total, matched, unmatched


# explicit figure -> (generated_by script id, representative source_eval, note)
# from docs/thesis_deliverables.md §3 (the thesis figures are exported by generate_thesis_figures.py)
FIG_MAP = {
    "fig4_ir_evolution": ("thesis_figures_gen", "ir_final_v3b", "IR model evolution (Tab 4.2)"),
    "fig6_1_cumulative_confuser": ("thesis_figures_gen", "", "cumulative confuser suppression §7"),
    "fig6_2_svanstrom_by_category": ("thesis_figures_gen", "rgb_svan_baseline", "Svanstrom by-category x by-stage"),
    "fig6_3_threshold_sweep": ("thesis_figures_gen", "", "patch threshold sweep §7.6"),
    "fig6_6_resolution": ("thesis_figures_gen", "selcom_ft2_1280", "resolution sensitivity 640 vs 1280"),
    "fig7_1_ood_classifier": ("thesis_figures_gen", "rob_rgb_baseline", "OOD classifier comparison"),
    "fig5_realvideo_pareto": ("thesis_figures_gen", "vid_drone_baseline", "real-video Pareto §9.4.5"),
    "fig5_cascade_percategory": ("thesis_figures_gen", "pipe_percat_sa32", "per-category cascade §9.5.9"),
    "unisa-logo": ("(institutional)", "", "institutional logo, not a result figure"),
}


def audit_figures(text):
    rows, report = [], []
    scripts = {s["id"] for s in kb._read("scripts")}
    for m in INCG_RE.finditer(text):
        ref = m.group(1).strip()
        stem = Path(ref).stem
        cands = [REPO / "docs" / ref, REPO / "docs" / "figures" / ref, REPO / ref]
        for ext in (".pdf", ".png", ".jpg"):
            cands.append(REPO / "docs" / "figures" / (stem + ext))
        exists = any(c.exists() for c in cands)
        gen, src, note = FIG_MAP.get(stem, ("", "", ""))
        if stem == "unisa-logo":
            status = "verified"   # institutional asset; provenance N/A
        elif gen and gen in scripts and exists:
            status = "verified"
        elif not exists:
            status = "stale"
        else:
            status = "orphan"
        rows.append(dict(id=kb._slug("fig_" + stem), tex_path=ref, kind="figure",
                         generated_by=gen, source_eval=src, fig_status=status,
                         notes=(note + f"; exists={exists}").strip("; ")))
        report.append((ref, exists, gen or "(unknown)", status))
    return rows, report


def main():
    tex = Path(sys.argv[1]) if len(sys.argv) > 1 else next((p for p in DEFAULT_TEX if p.exists()), None)
    if not tex or not tex.exists():
        sys.exit("no thesis .tex found (looked for docs/thesis_working.tex, docs/thesis_chapters.tex)")
    text = tex.read_text(encoding="utf-8", errors="replace")
    idx = eval_value_index()

    n_total, matched, unmatched = audit_numbers(text, idx)
    fig_rows, fig_report = audit_figures(text)

    # persist figures.csv (overwrite — it's derived from the tex)
    kb._write("figures", fig_rows)

    date = _dt.date.today().isoformat()
    out = REPO / "docs" / "analysis" / f"THESIS_AUDIT_{date}.md"
    L = [f"# Thesis audit — {tex.name} ({date})", "",
         "_Deterministic checks (numbers + figures). Qualitative claim verdicts are produced by "
         "the `thesis` skill's agent pass, recorded in `knowledge/claims.csv`._", "",
         "## Numbers vs knowledge/evals.csv",
         f"- result-like numbers scanned: **{n_total}**",
         f"- matched to an eval value: **{matched}**",
         f"- **not found ({len(unmatched)})** — verify (stale/copied, OR derived/cited/aggregate):", ""]
    if unmatched:
        L.append("| value | line | context |")
        L.append("|---|---|---|")
        for raw, ln, ctx in unmatched[:200]:
            L.append(f"| {raw} | {ln} | {ctx.replace('|', '\\|')} |")
    else:
        L.append("_all result-like numbers matched._")
    L += ["", "## Figures (\\includegraphics)", "",
          "| tex_path | exists | generated_by | status |", "|---|---|---|---|"]
    for ref, ex, gen, st in fig_report:
        L.append(f"| {ref} | {ex} | {gen} | {st} |")
    L += ["", f"_figures.csv updated ({len(fig_rows)} rows)._"]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    kb._regen_views()
    print(f"audited {tex.name}: {matched}/{n_total} numbers matched, "
          f"{len(unmatched)} to verify; {len(fig_rows)} figures. Report: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
