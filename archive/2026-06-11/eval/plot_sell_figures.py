"""plot_sell_figures.py - two email-ready figures for the Pietro reply.
Values are the measured points from this cycle (sweep_rgb_filter_ood.py / bench_speed.py /
2026-06-01 feature study). py eval/plot_sell_figures.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ── Fig 1: RGB confuser-filter Pareto (same FT4 detector; suppression vs drone recall) ──
mlp = [(63.9, .907), (78.2, .893), (85.6, .879), (91.7, .850), (93.5, .841),
       (94.0, .819), (94.4, .798), (96.3, .757), (98.6, .710)]
cnn = [(33.8, .905), (39.8, .902), (44.4, .890), (48.1, .883), (51.9, .874)]
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.plot([p[0] for p in mlp], [p[1] for p in mlp], "-o", color="#27ae60", lw=2.2, ms=6,
        label="V5 MLP filter (swept)")
ax.plot([p[0] for p in cnn], [p[1] for p in cnn], "-X", color="#e74c3c", lw=2.2, ms=9,
        label="CNN patch (swept)")
ax.axvspan(52, 100, color="#27ae60", alpha=0.06)
ax.annotate("CNN cannot exceed ~52%\nsuppression", (51.9, .874), xytext=(34, .80),
            arrowprops=dict(arrowstyle="->", color="#c0392b"), color="#c0392b", fontsize=9)
ax.annotate("MLP: 64–99% suppression\nat equal-or-higher recall", (92, .798), xytext=(57, .725),
            arrowprops=dict(arrowstyle="->", color="#1e8449"), color="#1e8449", fontsize=9)
ax.set_xlabel("Confuser suppression  (%)  →  fewer false alarms")
ax.set_ylabel("Drone recall  →  keeps real drones")
ax.set_title("RGB confuser filter: V5 MLP vs CNN patch (same detector)", pad=12)
ax.set_xlim(30, 101); ax.set_ylim(.68, .94); ax.grid(alpha=.3); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "email_filter_pareto.png", dpi=150); plt.close(fig)

# ── Fig 2: AUROC vs leakage — which features carry signal vs memorise scenes ──
keep = [(0.002, 0.965, "ir_max_conf"), (0.002, 0.952, "ir_best_aspect"),
        (0.005, 0.946, "ir_best_log_area"), (0.002, 0.905, "rgb conf/geom (×3)")]
drop = [(349.6, 0.502, "rgb_img_std"), (307.4, 0.510, "rgb_img_entropy"),
        (11.7, 0.795, "ir_blurriness"), (3.0, 0.708, "ir_img_entropy"), (1.75, 0.525, "rgb_edge_density")]
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.scatter([p[0] for p in keep], [p[1] for p in keep], s=110, color="#27ae60",
           edgecolor="k", zorder=5, label="KEEP → robust6 (6 feats)")
ax.scatter([p[0] for p in drop], [p[1] for p in drop], s=110, color="#e74c3c",
           marker="X", edgecolor="k", zorder=5, label="DROP → scene fingerprints (in fusion_no_fn/sa32)")
ax.annotate("robust6's 6 features\nAUROC 0.91–0.97, leakage ≈0.002", (0.002, 0.905),
            xytext=(0.011, 0.83), fontsize=9, color="#1e7a3f",
            arrowprops=dict(arrowstyle="->", color="#27ae60"))
for x, y, t in drop:
    ax.annotate(t, (x, y), xytext=(5, -12), textcoords="offset points", fontsize=8)
ax.axhline(0.5, color="gray", ls="--", lw=.8)
ax.set_xscale("log"); ax.set_xlabel("Leakage  F_domain-in-class / F_class  (→ scene memorisation)")
ax.set_ylabel("AUROC-alone  (→ drone-vs-confuser signal)")
ax.set_title("Why 6 features ≈ 40: keep high-signal / low-leakage, drop the rest", pad=12)
ax.set_ylim(0.45, 1.02); ax.grid(alpha=.3); ax.legend(loc="upper center")
ax.text(0.5, 0.03, "robust6: 6 vs 40 feats · 404× faster · 99.8% in-domain · −30% OOD false alarms",
        transform=ax.transAxes, ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#eafaf1", ec="#27ae60"))
fig.tight_layout(); fig.savefig(OUT / "email_robust6_stats.png", dpi=150); plt.close(fig)
print("saved:", OUT / "email_filter_pareto.png", "+", OUT / "email_robust6_stats.png")
