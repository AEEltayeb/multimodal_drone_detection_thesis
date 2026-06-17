"""One-time patch: update the CLAIMED constants in _audit_headline_numbers.py to the v4/thermal-only
values (= the frozen canonical json). Label-anchored regex so each constant is replaced exactly once.
CBAM is handled separately (regex + canonical). Run from repo root; review the git diff after."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
F = REPO / "thesis_eval/_audit_headline_numbers.py"

UP = {
    "svan composed F1": "0.948", "svan composed P": "0.941", "svan composed R": "0.9552",
    "svan filt->clf F1": "0.9636",
    "rgbconf mlp fire": "0.0144", "rgbconf mlp FP": "39", "rgbconf composed FP": "3",
    "irconf composed r8 fire": "0.0243", "irconf composed r6 fire": "0.0192",
    "grayconf mlp fire": "0.0053", "grayconf mlp FP": "15",
    "rgbtest mlp F1": "0.9222", "irtest mlp F1": "0.9421",
    "video composed r8 F1": "0.5436", "video composed r8 fire": "0.0756",
    "SZ rgbtest <16 filt R": "0.7672", "SZ rgbtest 16-32 filt R": "0.8435", "SZ rgbtest >=64 filt R": "0.9513",
    "SWING trust-aware pipeline F1": "0.948", "SWING identical precision": "0.941",
    "SWING trust-aware precision": "0.941", "SWEEP selcom filt@0.05": "0.6993",
    "CLEAN svan pipeline": "0.934", "CLEAN auv pipeline": "0.9861",
    "DUT clf->filt[robust8] P": "0.9", "DUT filt->clf[robust8] P": "0.901",
    "NR svan composed F1": "0.931", "NR svan composed R": "0.957", "NR svan composed P": "0.906",
    "NR svan filt->clf F1": "0.946",
    "NR dut composed F1": "0.79", "NR dut composed R": "0.721", "NR dut filt->clf F1": "0.835",
    "NR rgb_conf fire": "0.0144", "NR ir_conf fire": "0.028",
    "NR rgb_conf fire filt->clf": "0.0144", "NR ir_conf fire filt->clf": "0.028",
    "NR rgb_test composed F1": "0.922", "NR video composed F1": "0.646", "NR video_conf fire": "0.213",
    "dut filt_mlp_rgb F1": "0.722", "dut filt_mlp_ir F1": "0.728",
    "FIG rgb recall@0.25": "0.949", "FIG rgb fire@0.25": "0.014",
    "FIG gray recall@0.25": "0.476", "FIG gray recall@0.05": "0.633",
    "FIG gray fire@0.25": "0.005", "FIG gray fire@0.05": "0.033",
}

t = F.read_text(encoding="utf-8")
patched, missed = 0, []
for label, val in UP.items():
    pat = r'(\("' + re.escape(label) + r'",\s*)[0-9.]+(\s*,)'
    t, c = re.subn(pat, r'\g<1>' + val + r'\g<2>', t)
    patched += c
    if c != 1:
        missed.append(f"{label} (matched {c})")
F.write_text(t, encoding="utf-8")
print(f"patched {patched}/{len(UP)} CLAIMED constants")
if missed:
    print("CHECK THESE:", "; ".join(missed))
