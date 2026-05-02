import json, os

d = r"G:/drone/Anti-UAV-RGBT/val/20190926_103046_1_2"
with open(os.path.join(d, "visible.json")) as f:
    data = json.load(f)

print("Keys:", list(data.keys()))
for k in list(data.keys()):
    v = data[k]
    if isinstance(v, str):
        print(f"  {k}: {v}")
    elif isinstance(v, list):
        print(f"  {k}: list[{len(v)}], first 3 = {v[:3]}")
    else:
        print(f"  {k}: {type(v).__name__} = {str(v)[:80]}")

if "exist" in data:
    exist = data["exist"]
    total = len(exist)
    absent = sum(1 for e in exist if e == 0)
    present = sum(1 for e in exist if e == 1)
    print(f"\nExist flags: total={total}, present={present}, absent={absent}")
    print(f"Target visible in {present/total*100:.1f}% of frames")

if "gt_rect" in data:
    rects = data["gt_rect"]
    nonzero = [(i, r) for i, r in enumerate(rects) if any(v != 0 for v in r)]
    print(f"\nGT rects: total={len(rects)}, with bbox={len(nonzero)}, empty={len(rects)-len(nonzero)}")
    if nonzero:
        print(f"First 3 bbox frames: {nonzero[:3]}")
        print(f"Last 3 bbox frames: {nonzero[-3:]}")

# Check all val sequences for class labels
print("\n\n=== All val sequences ===")
val_dir = r"G:/drone/Anti-UAV-RGBT/val"
for seq in sorted(os.listdir(val_dir)):
    seq_path = os.path.join(val_dir, seq)
    if not os.path.isdir(seq_path):
        continue
    vj = os.path.join(seq_path, "visible.json")
    if os.path.exists(vj):
        with open(vj) as f:
            d = json.load(f)
        exist = d.get("exist", [])
        present = sum(1 for e in exist if e == 1)
        total = len(exist)
        cls = d.get("class", d.get("label", d.get("category", "?")))
        print(f"  {seq}: {total} frames, target in {present}/{total}, class={cls}")
