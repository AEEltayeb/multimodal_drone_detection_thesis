"""Pull test-split IoP drone stats."""
import json
from pathlib import Path

base = Path("eval/results/roboflow_ood")
targets = [
    ("rgb_drone", "rgb_baseline", "test"),
    ("rgb_drone", "rgb_retrained_v2", "test"),
    ("ir_mixed_cbam", "ir_model", "valid"),  # may not have test
    ("ir_mixed_cbam", "ir_model", "test"),
    ("ir_drone_night", "ir_model", "test"),
]

print(f"{'Dataset':<22s} {'Model':<18s} {'Split':<6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Imgs':>5s} {'FPPI':>6s}")
print("-" * 105)

for ds, model, split in targets:
    jp = base / ds / model / split / f"{model}_results.json"
    if not jp.exists():
        print(f"{ds:<22s} {model:<18s} {split:<6s} -- not found --")
        continue
    d = json.loads(jp.read_text())
    dm = d.get("detection_metrics", [])
    iop = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
    frm = d.get("frame_metrics", {})
    n = frm.get("tp",0) + frm.get("fp",0) + frm.get("fn",0) + frm.get("tn",0)
    fppi = iop.get("FP",0) / n if n else 0
    print(f"{ds:<22s} {model:<18s} {split:<6s} "
          f"{iop.get('TP',0):>5d} {iop.get('FP',0):>5d} {iop.get('FN',0):>5d} "
          f"{iop.get('precision',0):>6.3f} {iop.get('recall',0):>6.3f} {iop.get('f1',0):>6.3f} "
          f"{n:>5d} {fppi:>6.3f}")
