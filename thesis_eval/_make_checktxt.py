"""
thesis_eval/_make_checktxt.py — emit clean nr_drop (robust8_noreject_drop) rows for the thesis ablation
tables, in each table's exact LaTeX column order, into ES_Drone_Detection/check.txt (handoff for the
thesis-agent). Reads thesis_eval/results_noreject/tier1_results.json. Cross-prints the robust8 row so the
agent can confirm the source matches the existing thesis row (consistency check). Cascade/real-video rows
are appended separately after temporal_replay.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WS = REPO.parent
CHECK = WS / "ES_Drone_Detection" / "check.txt"
J = json.load(open(REPO / "thesis_eval/results_noreject/tier1_results.json"))
T = json.load(open(REPO / "thesis_eval/results_noreject/temporal_results.json"))
NR = "robust8_nr_drop"


def prf_row(label, c):
    ci = c.get("f1_ci", [c["f1"], c["f1"]])
    return (f"{label} & {c['TP']} & {c['FP']} & {c['FN']} & {c['precision']:.3f} & "
            f"{c['recall']:.3f} & {c['f1']:.3f} [{ci[0]:.3f}--{ci[1]:.3f}] \\\\")


def fire_cell(c):
    if not c:
        return "---"
    ci = c.get("fire_ci")
    return f"{c['fire_rate']:.4f}" + (f" [{ci[0]:.4f}--{ci[1]:.4f}]" if ci else "")


L = []
L += ["================================================================",
      "  nr_drop (robust8 NO-REJECT, drop variant) -> thesis ablation rows",
      "  HANDOFF for the thesis agent. ADD these rows; do not change existing rows.",
      "================================================================",
      "",
      "WHAT nr_drop IS: robust8's exact recipe (full56, F8 = robust6 + rgb_mean_conf + is_grayscale,",
      "  same XGBoost hyperparams + seq-split) but the 4th class `reject` is REMOVED (rows with neither",
      "  modality correct are dropped in training). The router always routes rgb/ir/both (argmax, no tau);",
      "  the per-frame verifier does all FP rejection. Stored with label_map {0:1,1:2,2:3}.",
      "  Model: models/routers/robust8_noreject.joblib   (ES_Drone_Thesis)",
      "",
      "SOURCE of every number below: thesis_eval/results_noreject/tier1_results.json",
      "  run = thesis_eval/pipeline_eval_unified.py  (SAME unified caches + code as the canonical",
      "  thesis_eval/results/; the robust8/sa32/robust6 cells are byte-identical to the canonical run,",
      "  so these nr_drop rows are drop-in consistent with the existing tables).",
      "  Per-surface F1 summary + narrative: docs/analysis/2026-06-14_robust8_noreject.md",
      "  Column order = TP & FP & FN & P & R & F1 [95% CI]   (same as the thesis rows).",
      "  '(cross-check)' lines = the existing robust8 row from this source; it MUST match the thesis",
      "  table already — if it does, the nr_drop row beside it is consistent.",
      "  (A cross-check F1 may differ by +/-0.001 from the thesis from .3f rounding at a boundary; the",
      "   TP/FP/FN match exactly, which is the real consistency proof.)",
      ""]

# paired drone tables: B_pipeline, three nr_drop arms
PAIRED = [("tab:ablation_svanstrom", "svanstrom", "svanstrom B_pipeline"),
          ("tab:ablation_antiuav", "antiuav", "antiuav B_pipeline"),
          ("tab:ablation_dut", "dut_antiuav_960", "dut_antiuav_960 B_pipeline (full split n=2200)")]
for tab, surf, src in PAIRED:
    p = J[surf]["B_pipeline"]
    L += [f"% ====== {tab}  ({src}) ======",
          f"% [source: thesis_eval/results_noreject/tier1_results.json ({surf} B_pipeline); run=thesis_eval/pipeline_eval_unified.py]",
          "ADD:",
          "  " + prf_row(r"clf only [nr\_drop]", p[f"clf[{NR}]"]),
          "  " + prf_row(r"clf$\to$filt [nr\_drop]", p[f"clf->filt[{NR}]"]),
          "  " + prf_row(r"filt$\to$clf [nr\_drop]", p[f"filt->clf[{NR}]"]),
          "(cross-check robust8 from this source — should equal the existing thesis row):",
          "  " + prf_row("clf only [robust8]", p["clf[robust8]"]),
          "  " + prf_row(r"clf$\to$filt [robust8]", p["clf->filt[robust8]"]),
          ""]

# solo drone table: S4_verifier
L += ["% ====== tab:ablation_solo  (S4_verifier clf rows) ======",
      "% [source: thesis_eval/results_noreject/tier1_results.json (ir_dset_final, rgb_dataset_test, selcom_val S4_verifier)]"]
for surf in ("ir_dset_final", "rgb_dataset_test", "selcom_val"):
    s4 = J[surf].get("S4_verifier", {})
    if not s4:
        continue
    L += [f"  -- {surf} --",
          "  ADD " + prf_row(r"clf only [nr\_drop]", s4[f"clf[{NR}]"]),
          "  ADD " + prf_row(r"clf$\to$filt [nr\_drop]", s4[f"clf->filt[{NR}]"]),
          "  (xc) " + prf_row("clf only [robust8]", s4["clf[robust8]"])]
L += [""]

# confuser table: C_confuser fire rate, columns RGB | IR | gray(---)
L += ["% ====== tab:ablation_confusers  (C_confuser; fire rate [CI]; cols: RGB | IR | grayscale) ======",
      "% [source: thesis_eval/results_noreject/tier1_results.json (rgb_confuser, ir_confusers C_confuser; gray router-bypassed=---)]"]
rc, ic = J["rgb_confuser"]["C_confuser"], J["ir_confusers"]["C_confuser"]
for arm, key in (("clf only [nr\\_drop]", f"clf[{NR}]"), ("clf$\\to$filt [nr\\_drop]", f"clf->filt[{NR}]")):
    L += [f"  ADD {arm:<26}& {fire_cell(rc.get(key))} & {fire_cell(ic.get(key))} & --- \\\\"]
for arm, key in (("clf only [robust8]", "clf[robust8]"), ("clf$\\to$filt [robust8]", "clf->filt[robust8]")):
    L += [f"  (xc) {arm:<26}& {fire_cell(rc.get(key))} & {fire_cell(ic.get(key))} & --- \\\\"]
vd, vc = T["video_drone"], T["video_confuser"]
def seg(c):
    w = c["window"]; return f"{w[0]:.3f} / {w[1]:.3f} / {w[2]:.3f}"
def segfire(c):
    return f"{c['window_fire']:.4f}"
L += ["",
      "% ====== tab:cascade_* / real-video SEGMENT grain (2-of-3 window vote) ======",
      "% [source: thesis_eval/results_noreject/temporal_results.json; run=thesis_eval/temporal_replay.py --out thesis_eval/results_noreject]",
      "% CAVEAT (from the harness): the OLD cascade_segment rows used baseline RGB + ALERT-gated patch;",
      "%   this replay is ft4 + v3b-on-gray + per-frame patch_thr=0.70 — compare DIRECTIONS, not decimals.",
      "% video_drone (9 clips) — segment P / R / F1:",
      f"  ADD  clf [nr\\_drop]            {seg(vd['clf[' + NR + ']'])}",
      f"  ADD  clf$\\to$filt [nr\\_drop]   {seg(vd['clf->filt[' + NR + ']'])}",
      f"  (xc) clf [robust8]            {seg(vd['clf[robust8]'])}",
      f"  (xc) clf$\\to$filt [robust8]   {seg(vd['clf->filt[robust8]'])}",
      "% video_confuser (10 clips) — segment fire-rate (lower=better):",
      f"  ADD  clf [nr\\_drop]            {segfire(vc['clf[' + NR + ']'])}",
      f"  ADD  clf$\\to$filt [nr\\_drop]   {segfire(vc['clf->filt[' + NR + ']'])}",
      f"  (xc) clf [robust8]            {segfire(vc['clf[robust8]'])}",
      f"  (xc) clf$\\to$filt [robust8]   {segfire(vc['clf->filt[robust8]'])}",
      "",
      "% ====== tab:modality_ab ======",
      "% NOT regenerated here — it uses thesis_eval/notes_round1_replays.py (coverage/dual scoring, a",
      "%   different harness). load_classifiers now includes robust8_nr_drop, so if an nr_drop row is",
      "%   wanted there, run notes_round1_replays.py and read M_modality_ab.",
      ""]

CHECK.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {CHECK}  ({len(L)} lines)")
