"""Print clip -> YouTube ID from the two extraction manifests (notes round 2, N7)."""
import json, re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "datasets"
for mf in (BASE / "drone detection video tests/rgb/extraction_manifest.json",
           BASE / "confuser_videos/extraction_manifest.json"):
    print("==", mf.parent.name)
    for e in json.load(open(mf, encoding="utf-8")):
        m = re.search(r"_Media_([A-Za-z0-9_-]{11})", e.get("video", ""))
        print(f"{e['dataset_name']:60s} {m.group(1) if m else 'NOT PRESERVED'}")
