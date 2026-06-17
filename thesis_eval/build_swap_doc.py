"""
build_swap_doc.py — reconstruct every thesis table that contains an mlp-filter cell as a SHIPPED vs
CANDIDATE side-by-side, for a ship/no-ship decision. Reads tier1 + temporal + notes_round1 JSON from
a shipped dir and a candidate dir (same caches/seed; only the filter weights differ). Writes ONE
markdown doc. Pure read/format — regenerates nothing.

  py thesis_eval/build_swap_doc.py --shipped <dir> --candidate <dir> --tag v4 --out <doc.md>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(d: Path, name: str):
    p = d / name
    return json.load(open(p)) if p.exists() else {}


def cell(d, surface, section, key):
    return d.get(surface, {}).get(section, {}).get(key)


def fnum(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def darrow(s, c, lower_better=False):
    if s is None or c is None:
        return ""
    d = c - s
    if abs(d) < 1e-9:
        return "="
    good = (d < 0) if lower_better else (d > 0)
    return f"{d:+.4g} {'OK' if good else 'X'}"


# ── drone-surface table (B_pipeline / S4_verifier): P/R/F1/FP per cell ───────────────────────────
def drone_table(S, C, surface, section, cells, title):
    L = [f"### {title}  (`{surface}.{section}`)", "",
         "| cell | metric | shipped | candidate | Δ |", "|---|---|---|---|---|"]
    for ckey, label in cells:
        s, c = cell(S, surface, section, ckey), cell(C, surface, section, ckey)
        if s is None and c is None:
            continue
        for m, lb in (("recall", False), ("f1", False), ("FP", True), ("precision", False)):
            sv = s.get(m) if s else None
            cv = c.get(m) if c else None
            L.append(f"| {label} | {m} | {fnum(sv)} | {fnum(cv)} | {darrow(sv, cv, lb)} |")
    L.append("")
    return L


# ── confuser table (C_confuser): FP + fire ───────────────────────────────────────────────────────
def confuser_table(S, C, surfaces, cells, title):
    L = [f"### {title}  (`*.C_confuser`)", "",
         "| surface | cell | metric | shipped | candidate | Δ |", "|---|---|---|---|---|---|"]
    for surface in surfaces:
        for ckey, label in cells:
            s, c = cell(S, surface, "C_confuser", ckey), cell(C, surface, "C_confuser", ckey)
            if s is None and c is None:
                continue
            for m in ("FP", "fire_rate"):
                sv = s.get(m) if s else None
                cv = c.get(m) if c else None
                L.append(f"| {surface} | {label} | {m} | {fnum(sv)} | {fnum(cv)} | {darrow(sv, cv, True)} |")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--tag", default="candidate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SD, CD = Path(a.shipped), Path(a.candidate)
    St, Ct = load(SD, "tier1_results.json"), load(CD, "tier1_results.json")

    paired_cells = [("filt_mlp_rgb", "filt only (mlp_v5,RGB)"),
                    ("filt_mlp_ir", "filt only (aligned,IR)"),
                    ("clf->filt[robust8_nr_drop]", "clf→filt [robust8-nr]"),
                    ("filt->clf[robust8_nr_drop]", "filt→clf [robust8-nr] (SHIPPED)"),
                    ("clf->filt[robust8]", "clf→filt [robust8]"),
                    ("filt->clf[robust8]", "filt→clf [robust8]")]
    solo_cells = [("bare", "bare detector"), ("filt_mlp", "filt (mlp)")]
    conf_cells = [("filt_mlp", "filt only (mlp)"),
                  ("clf->filt[robust8_nr_drop]", "clf→filt [robust8-nr] (SHIPPED)"),
                  ("clf->filt[robust8]", "clf→filt [robust8]"),
                  ("clf->filt[robust6]", "clf→filt [robust6]")]

    L = [f"# Filter swap — SHIPPED vs `{a.tag}` (decision tables)", "",
         f"shipped = `{SD}` · candidate = `{CD}`. Own-GT / trust-aware. Filter removes detections only.",
         "Δ flag: OK = improves (recall/F1 up, FP/fire down), X = regresses, = unchanged.", ""]

    L += ["## Paired full-pipeline tables", ""]
    L += drone_table(St, Ct, "svanstrom", "B_pipeline", paired_cells, "tab:ablation_svanstrom")
    L += drone_table(St, Ct, "antiuav", "B_pipeline", paired_cells, "tab:ablation_antiuav")
    L += drone_table(St, Ct, "dut_antiuav_960", "B_pipeline", paired_cells, "tab:ablation_dut (@960)")

    L += ["## Solo-surface table", ""]
    for surf in ("ir_dset_final", "rgb_dataset_test", "selcom_val", "svanstrom_gray"):
        L += drone_table(St, Ct, surf, "S4_verifier", solo_cells, f"tab:ablation_solo — {surf}")

    L += ["## Confuser FP-reduction table", ""]
    L += confuser_table(St, Ct, ["rgb_confuser", "ir_confusers", "gray_confuser"], conf_cells,
                        "tab:ablation_confusers")

    # temporal (video) — frame+window per cell if present
    Sv, Cv = load(SD, "temporal_results.json"), load(CD, "temporal_results.json")
    if Sv and Cv:
        tcells = ["filt_mlp", "clf->filt[robust8_nr_drop]", "clf->filt[robust8]", "clf->filt[robust6]"]
        L += ["## Real-video temporal table (`tab:temporal_production`)", "",
              "video_drone = frame P/R/F1; video_confuser = frame fire-rate.", "",
              "| surface | cell | metric | shipped | candidate | Δ |", "|---|---|---|---|---|---|"]
        sd, cd = Sv.get("video_drone", {}), Cv.get("video_drone", {})
        for ckey in tcells:
            s, c = sd.get(ckey), cd.get(ckey)
            if not s and not c:
                continue
            for i, m in ((1, "recall"), (2, "f1")):     # frame = [P, R, F1]
                sv = s["frame"][i] if s else None
                cv = c["frame"][i] if c else None
                L.append(f"| video_drone | {ckey} | {m} | {fnum(sv)} | {fnum(cv)} | {darrow(sv, cv, False)} |")
        scf, ccf = Sv.get("video_confuser", {}), Cv.get("video_confuser", {})
        for ckey in tcells:
            s, c = scf.get(ckey), ccf.get(ckey)
            if not s and not c:
                continue
            sv = s.get("frame_fire") if s else None
            cv = c.get("frame_fire") if c else None
            L.append(f"| video_confuser | {ckey} | fire | {fnum(sv)} | {fnum(cv)} | {darrow(sv, cv, True)} |")
        L.append("")

    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
