"""Step 2: Train classifier on paired + real grayscale data."""
import subprocess, sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "reliability" / "fusion" / "train_fusion.py"

cmd = [
    sys.executable, str(SCRIPT),
    "--no-fn",
    "--in-suffix", "_v3more_realgray",
    "--out-suffix", "_v3more_realgray",
    "--max-rows-per-dataset", "60000",
    "--exclude-features",
    "rgb_n_dets,ir_n_dets,rgb_detected,ir_detected,both_detect,neither_detect,rgb_only_detect,ir_only_detect",
]
print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, check=True)
print("Step 2 done.")
