"""
diff_filters.py — shipped-vs-candidate filter A/B differ over the Tier-1 harness JSON.

Loads two tier1_results.json (shipped/, candidate/) produced by pipeline_eval_unified.py with
identical caches/stride/seed and only the filter weights+IR thr swapped (THESIS_* env overrides).
For every surface it walks the filter-bearing cells (filt_mlp*, clf->filt*, filt->clf*, GRAY_SWEEP,
Part-D) and prints shipped vs candidate (R/F1 on drone surfaces; FP/fire_rate on confusers) + delta.
Bare / patch / clf-only cells are filter-independent and are emitted only as a sanity check.

  py thesis_eval/results/_filter_ab/diff_filters.py
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--shipped", default=str(HERE / "shipped" / "tier1_results.json"))
_ap.add_argument("--candidate", default=str(HERE / "candidate" / "tier1_results.json"))
_args, _ = _ap.parse_known_args()
SHIP = json.load(open(_args.shipped))
CAND = json.load(open(_args.candidate))

FILT = ("filt_mlp", "clf->filt", "filt->clf")          # cells whose value depends on the mlp filter


def is_filter_cell(name):
    return any(k in name for k in FILT)


def scalars(cell):
    """Return the comparable metrics present in a cell dict."""
    out = {}
    for k in ("recall", "precision", "f1", "FP", "fire_rate"):
        if k in cell:
            out[k] = cell[k]
    return out


def fmt_delta(a, b, invert=False):
    d = b - a
    arrow = "" if abs(d) < 1e-9 else ("[better]" if (d > 0) != invert else "[worse]")
    return f"{a:>7.4g} -> {b:>7.4g}  ({d:+.4g}) {arrow}"


def walk_section(surf, sect, ship_sec, cand_sec, lines, only_filter=True):
    for cell in ship_sec:
        if cell not in cand_sec:
            continue
        if only_filter and not is_filter_cell(cell):
            continue
        sa, sb = scalars(ship_sec[cell]), scalars(cand_sec[cell])
        # FP/fire_rate => lower is better (confuser); recall/f1 => higher better.
        for m in ("recall", "f1", "FP", "fire_rate", "precision"):
            if m in sa and m in sb and abs(sa[m] - sb[m]) > 1e-9:
                invert = m in ("FP", "fire_rate")
                lines.append(f"  {surf:<18} {sect:<11} {cell:<28} {m:<10} {fmt_delta(sa[m], sb[m], invert)}")


def main():
    lines = []
    for surf in sorted(SHIP):
        s, c = SHIP[surf], CAND[surf]
        for sect in ("B_pipeline", "S4_verifier", "C_confuser"):
            if sect in s and sect in c:
                walk_section(surf, sect, s[sect], c[sect], lines)
        # GRAY_SWEEP: dict of thr -> metrics (whole sweep is filter-driven)
        if "GRAY_SWEEP" in s and "GRAY_SWEEP" in c:
            for thr in s["GRAY_SWEEP"]:
                if thr in c["GRAY_SWEEP"]:
                    sa, sb = scalars(s["GRAY_SWEEP"][thr]), scalars(c["GRAY_SWEEP"][thr])
                    for m in ("recall", "f1", "FP", "fire_rate"):
                        if m in sa and m in sb and abs(sa[m] - sb[m]) > 1e-9:
                            lines.append(f"  {surf:<18} GRAY_SWEEP  thr={thr:<22} {m:<10} "
                                         f"{fmt_delta(sa[m], sb[m], m in ('FP','fire_rate'))}")
    # sanity: confirm A_bare identical everywhere (filter must not touch bare)
    bare_changed = []
    for surf in SHIP:
        sb_, cb_ = SHIP[surf].get("A_bare", {}), CAND[surf].get("A_bare", {})
        for mod in sb_:
            if mod in cb_ and abs(sb_[mod].get("f1", 0) - cb_[mod].get("f1", 0)) > 1e-9:
                bare_changed.append(f"{surf}/{mod}")
    out = ["=" * 100,
           "FILTER A/B DIFF - shipped vs candidate (only cells that CHANGED; filter-bearing)",
           "=" * 100,
           "\n".join(lines) if lines else "  (no differences)",
           "\nSANITY - A_bare cells that changed (should be EMPTY): " +
           (", ".join(bare_changed) if bare_changed else "none OK")]
    txt = "\n".join(out)
    (HERE / "DIFF.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
