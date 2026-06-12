import json

old = json.load(open("fusion_models/scene_aware_v3more_32feat/metrics.json"))
new = json.load(open("fusion_models/retrained_v2_32feat/metrics.json"))

old_imp = old["feature_importance"]
new_imp = new["feature_importance"]

# Sort by new importance
all_feats = sorted(set(old_imp) | set(new_imp), key=lambda f: new_imp.get(f, 0), reverse=True)

print(f"{'Feature':<32s} {'OLD':>8s} {'NEW':>8s} {'Delta':>8s}")
print("-" * 62)
for f in all_feats:
    o = old_imp.get(f, 0)
    n = new_imp.get(f, 0)
    d = n - o
    arrow = "^" if d > 0.005 else ("v" if d < -0.005 else " ")
    print(f"{f:<32s} {o:>8.4f} {n:>8.4f} {d:>+8.4f} {arrow}")
