import json

ab = json.load(open("runs/patches/confuser_filter4_ab_comparison.json"))
new_rgb = json.load(open("runs/patches/confuser_filter4_rgb_metrics.json"))
new_ir = json.load(open("runs/patches/confuser_filter4_ir_metrics.json"))

for mod in ["rgb", "ir"]:
    new = new_rgb if mod == "rgb" else new_ir
    old_acc = ab[mod]["old_acc"]
    new_acc = new["best_val_acc"]
    print(f"\n=== {mod.upper()} CONFUSER FILTER: OLD (v2) vs NEW (v3) ===")
    print(f"  Overall acc:  OLD={old_acc:.4f}  NEW={new_acc:.4f}  delta={new_acc-old_acc:+.4f}")
    print(f"  {'Class':10s} {'OLD_P':>7s} {'OLD_R':>7s} {'NEW_P':>7s} {'NEW_R':>7s}  Delta-R")
    for cls in ["airplane", "helicopter", "bird", "other"]:
        op = ab[mod]["old_per_class"][cls]
        np_ = new["final"]["per_class"][cls]
        dr = np_["R"] - op["R"]
        arrow = "^" if dr > 0.01 else ("v" if dr < -0.01 else "=")
        print(f"  {cls:10s} {op['P']:>7.3f} {op['R']:>7.3f} {np_['P']:>7.3f} {np_['R']:>7.3f}  {dr:+.3f} {arrow}")

    print(f"\n  Veto sweep (old vs new):")
    print(f"  {'Thr':>5s} {'OLD_vP':>7s} {'OLD_vR':>7s} {'NEW_vP':>7s} {'NEW_vR':>7s} {'OLD_pass':>8s} {'NEW_pass':>8s}")
    for thr in ["thr=0.5", "thr=0.7", "thr=0.9"]:
        os = ab[mod]["old_reject_sweep"][thr]
        ns = new["final"]["reject_sweep"][thr]
        print(f"  {thr:>5s} {os['precision_veto']:>7.3f} {os['recall_veto']:>7.3f} {ns['precision_veto']:>7.3f} {ns['recall_veto']:>7.3f} {os['pass_acc_on_drones']:>8.3f} {ns['pass_acc_on_drones']:>8.3f}")
