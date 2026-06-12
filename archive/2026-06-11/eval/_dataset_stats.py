"""Get image counts + FPPI for confuser tables, and per-category FP for ir_mixed_cbam."""
import json
from pathlib import Path
from collections import defaultdict

base = Path("eval/results/roboflow_ood")

# 1) Image counts per dataset (all splits combined)
print("=== Image counts per dataset (all splits) ===")
for ds in sorted(base.iterdir()):
    if not ds.is_dir():
        continue
    total = 0
    for jp in ds.rglob("*_results.json"):
        d = json.loads(jp.read_text())
        frm = d.get("frame_metrics", {})
        total += frm.get("tp",0) + frm.get("fp",0) + frm.get("fn",0) + frm.get("tn",0)
    # deduplicate by model - count once per split
    models = set()
    imgs_per_split = {}
    for jp in ds.rglob("*_results.json"):
        d = json.loads(jp.read_text())
        rel = jp.relative_to(base)
        split = rel.parts[2] if len(rel.parts) > 3 else "root"
        model = d.get("model","")
        frm = d.get("frame_metrics", {})
        n = frm.get("tp",0)+frm.get("fp",0)+frm.get("fn",0)+frm.get("tn",0)
        if split not in imgs_per_split:
            imgs_per_split[split] = n
    unique_imgs = sum(imgs_per_split.values())
    print(f"  {ds.name}: {unique_imgs} images ({imgs_per_split})")

# 2) Per-dataset FP rate (using unique images, one model at a time)
print("\n=== Confuser FP rates (all splits) ===")
confuser_ds = ["rgb_airplane","rgb_bird","rgb_helicopter","ir_airplane_hors2","ir_airplane_plane","ir_bird"]
for dsn in confuser_ds:
    dp = base / dsn
    if not dp.exists():
        continue
    models = defaultdict(lambda: {"fp":0, "ffp":0, "imgs":0})
    for jp in dp.rglob("*_results.json"):
        d = json.loads(jp.read_text())
        m = d.get("model","")
        dm = d.get("detection_metrics",[])
        fm = d.get("filtered_metrics",[])
        iop = dm[1] if len(dm)>1 else (dm[0] if dm else {})
        fiop = fm[1] if len(fm)>1 else (fm[0] if fm else {})
        frm = d.get("frame_metrics",{})
        n = frm.get("tp",0)+frm.get("fp",0)+frm.get("fn",0)+frm.get("tn",0)
        models[m]["fp"] += iop.get("FP",0)
        models[m]["ffp"] += fiop.get("FP",0) if fiop else 0
        models[m]["imgs"] += n
    for m, v in sorted(models.items()):
        fp_rate = v["fp"]/v["imgs"]*100 if v["imgs"] else 0
        ffp_rate = v["ffp"]/v["imgs"]*100 if v["imgs"] else 0
        print(f"  {dsn:22s} {m:18s} {v['imgs']:>5d} imgs  {v['fp']:>5d} FP ({fp_rate:5.1f}%)  -> {v['ffp']:>5d} filt ({ffp_rate:5.1f}%)")

# 3) ir_mixed_cbam valid split: per-category GT counts
print("\n=== ir_mixed_cbam class distribution (valid split labels) ===")
from collections import Counter
cbam_root = Path("G:/drone/roboflow_eval/ir_mixed_cbam")
class_names = {0: "Bird", 1: "Drone", 2: "Plane"}
for split in ("train","valid","test"):
    lbl_dir = cbam_root / split / "labels"
    if not lbl_dir.exists():
        continue
    c = Counter()
    n_files = 0
    for f in lbl_dir.glob("*.txt"):
        n_files += 1
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                cls = int(line.split()[0])
                c[cls] += 1
    print(f"  {split}: {n_files} files, {dict({class_names.get(k,k): v for k,v in sorted(c.items())})}")
