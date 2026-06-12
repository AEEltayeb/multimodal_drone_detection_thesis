"""Aggregate OOD results with FP rates (FPPI + frame FP%)."""
import json
from pathlib import Path
from collections import defaultdict

base = Path("eval/results/roboflow_ood")
agg = defaultdict(lambda: {"imgs":0, "fp_frames_raw":0, "fp_frames_filt":0,
                            "raw_fp":0, "filt_fp":0,
                            "tp":0, "fn":0, "ftp":0, "ffn":0})

for jp in sorted(base.rglob("*_results.json")):
    d = json.loads(jp.read_text())
    rel = jp.relative_to(base)
    ds = rel.parts[0]
    model = d.get("model", "")
    k = (ds, model)

    dm = d.get("detection_metrics", [])
    fm = d.get("filtered_metrics", [])
    iop = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
    fiop = fm[1] if len(fm) > 1 else (fm[0] if fm else {})
    frm = d.get("frame_metrics", {})
    fs = d.get("filter_summary", {})

    agg[k]["tp"] += iop.get("TP", 0)
    agg[k]["fn"] += iop.get("FN", 0)
    agg[k]["raw_fp"] += iop.get("FP", 0)
    agg[k]["filt_fp"] += fiop.get("FP", 0) if fiop else 0
    agg[k]["ftp"] += fiop.get("TP", 0) if fiop else 0
    agg[k]["ffn"] += fiop.get("FN", 0) if fiop else 0
    agg[k]["imgs"] += frm.get("tp", 0) + frm.get("fp", 0) + frm.get("fn", 0) + frm.get("tn", 0)
    agg[k]["fp_frames_raw"] += fs.get("raw_det_frames", 0)
    agg[k]["fp_frames_filt"] += fs.get("filt_det_frames", 0)

hdr = (f"{'Dataset':<22s} {'Model':<18s} {'Imgs':>5s} "
       f"{'rawFP':>6s} {'FPPI':>6s} {'FP%frm':>7s} "
       f"{'filtFP':>6s} {'fFPPI':>6s} {'fFP%':>7s} "
       f"| {'R':>5s} {'fR':>5s}")
print(hdr)
print("-" * len(hdr))

for (ds, m), v in sorted(agg.items()):
    n = v["imgs"]
    fppi = v["raw_fp"] / n if n else 0
    fpr = v["fp_frames_raw"] / n * 100 if n else 0
    ffppi = v["filt_fp"] / n if n else 0
    ffpr = v["fp_frames_filt"] / n * 100 if n else 0
    rec = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0
    frec = v["ftp"] / (v["ftp"] + v["ffn"]) if v["ftp"] + v["ffn"] else 0
    r_s = f"{rec:.3f}" if v["tp"] + v["fn"] else "  -"
    fr_s = f"{frec:.3f}" if v["ftp"] + v["ffn"] else "  -"
    rfp = v["raw_fp"]
    ffp = v["filt_fp"]
    print(f"{ds:<22s} {m:<18s} {n:>5d} "
          f"{rfp:>6d} {fppi:>6.3f} {fpr:>6.1f}% "
          f"{ffp:>6d} {ffppi:>6.3f} {ffpr:>6.1f}% "
          f"| {r_s:>5s} {fr_s:>5s}")
