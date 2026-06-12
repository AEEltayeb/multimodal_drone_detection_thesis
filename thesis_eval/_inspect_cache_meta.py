"""Read-only: print each cached surface's meta + ir_confusers category mix."""
import pickle, collections
from pathlib import Path

for p in sorted(Path(__file__).parent.glob("cache/*.pkl")):
    d = pickle.load(open(p, "rb")); m = d["meta"]
    print(f"{m['name']:18} n={m['n']:5} stride={m['stride']:2} kind={m['kind']:6} "
          f"rule={m['rule']} imgsz_rgb={m['rgb_imgsz']} ir={m['ir_imgsz']} drones={m['has_drones']}")
    if m["name"] == "ir_confusers":
        c = collections.Counter(f["key"].split("_")[0].lower() for f in d["frames"])
        print("    ir_confusers key prefixes:", dict(c.most_common(8)))
