"""Quick check why labels fail to parse."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_confuser_filter import _find_original_frame, _get_box_from_label
import pandas as pd
import cv2

m = pd.read_csv(Path(__file__).resolve().parent / "runs" / "patches" / "manifest.csv")
n = 0
for _, r in m.iterrows():
    fp, lp = _find_original_frame(r["stem"], r["modality"], r["category"])
    if fp is None or not fp.exists():
        continue
    img = cv2.imread(str(fp))
    if img is None:
        continue
    h, w = img.shape[:2]
    box = _get_box_from_label(lp, r["stem"], w, h)
    if box is None:
        content = lp.read_text().strip() if lp and lp.exists() else "NO_FILE"
        lp_exists = lp.exists() if lp else False
        print(f"SKIP stem={r['stem']}")
        print(f"  label_path={lp}")
        print(f"  exists={lp_exists}  content_len={len(content)}")
        if content:
            print(f"  first_line={repr(content.split(chr(10))[0][:120])}")
        n += 1
        if n >= 15:
            break

print(f"\nChecked up to {n} failures")
