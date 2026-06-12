import json

old = json.load(open("fusion_models/scene_aware_v3more_32feat/metrics.json"))
new = json.load(open("fusion_models/retrained_v2_32feat/metrics.json"))

print("METRIC                    OLD (production)    NEW (retrained_v2)")
print("=" * 65)
print(f"Accuracy                  {old['accuracy']:.4f}              {new['accuracy']:.4f}")
print(f"F1 macro                  {old['f1_macro']:.4f}              {new['f1_macro']:.4f}")
print(f"F1 weighted               {old['f1_weighted']:.4f}              {new['f1_weighted']:.4f}")
print(f"Train rows                {old['n_train']:,}            {new['n_train']:,}")
print(f"Test rows                 {old['n_test']:,}             {new['n_test']:,}")
print()

old_imp = sorted(old["feature_importance"].items(), key=lambda x: -x[1])
new_imp = sorted(new["feature_importance"].items(), key=lambda x: -x[1])
print("TOP 5 FEATURES:")
print(f"  {'OLD':<30s} {'imp':>6s}   {'NEW':<30s} {'imp':>6s}")
for i in range(8):
    of, ov = old_imp[i]
    nf, nv = new_imp[i]
    print(f"  {of:<30s} {ov:>6.4f}   {nf:<30s} {nv:>6.4f}")

if "per_dataset" in old:
    print("\nPER-DATASET (old):")
    for ds, v in old["per_dataset"].items():
        print(f"  {ds:<20s} acc={v['acc']:.4f}  f1m={v['f1_macro']:.4f}")

if "per_dataset" in new:
    print("\nPER-DATASET (new):")
    for ds, v in new["per_dataset"].items():
        print(f"  {ds:<20s} n={v['n']:>6d}  acc={v['acc']:.4f}  f1m={v['f1_macro']:.4f}")
