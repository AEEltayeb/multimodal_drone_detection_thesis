"""
mri.examples — per-dataset spatial activation panels.

For one representative detection per dataset, render the 3-panel figure
(detection crop + P3 spatial-activation heatmap + P5 spatial-activation heatmap)
overlaid on the image — the intuitive "where the discriminative neurons fire"
view. The example chosen per dataset is the detection that MAXIMIZES activation
on the top discriminative neurons (highest contrast).

Requires the live feature maps, so this only runs during a real --pos/--neg scan
(not --resume on a cached feature corpus). Rendering style mirrors
scripts/visualize_active_neurons.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def top_channels_by_layer(schema, top_idx, n=8) -> dict[str, list[int]]:
    """Split the top-n discriminative flat indices into per-layer channel lists."""
    out: dict[str, list[int]] = {l: [] for l in schema.layers}
    for idx in top_idx[:n]:
        layer, chan, _cell = schema.locate(int(idx))
        if layer in out and chan is not None and chan not in out[layer]:
            out[layer].append(int(chan))
    return out


def spatial_heatmap(fmap, channels, img_shape) -> np.ndarray | None:
    """Abs-mean activation of `channels` from a (1,C,H,W) map, resized to image."""
    if fmap is None or not channels:
        return None
    arr = fmap[0].detach().cpu().numpy()           # (C, H, W)
    chans = [c for c in channels if c < arr.shape[0]]
    if not chans:
        return None
    heat = np.abs(arr[chans]).mean(axis=0)          # (H, W)
    ih, iw = img_shape
    return cv2.resize(heat, (iw, ih), interpolation=cv2.INTER_LINEAR)


def pick_best_per_spec(provenance, X, y, top_idx, n=8):
    """Per spec name, pick the row index maximizing activation on the top
    neurons. Drones score on signed positive activation; confusers on absolute
    (the discriminative-pattern magnitude). Returns {spec: row_index}."""
    sel = [int(i) for i in top_idx[:n]]
    best: dict[str, tuple[float, int]] = {}
    for row, prov in enumerate(provenance):
        feat = X[row]
        if y[row] == 1:
            score = float(sum(feat[i] for i in sel if feat[i] > 0))
        else:
            score = float(sum(abs(feat[i]) for i in sel))
        spec = prov["spec"]
        if spec not in best or score > best[spec][0]:
            best[spec] = (score, row)
    return {spec: row for spec, (_s, row) in best.items()}


def render_example(extractor, prov, channels_by_layer, title, out_path,
                   conf_thr=0.25, imgsz=640, device="cuda", pad=40):
    """Re-run YOLO on the example image, build P3/P5 heatmaps, render 3 panels."""
    img_bgr = cv2.imread(prov["path"])
    if img_bgr is None:
        return None
    ih, iw = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in prov["box"]]

    extractor.hook.clear()
    extractor.model.predict(img_bgr, imgsz=imgsz, conf=conf_thr,
                            verbose=False, device=device)

    heat_p3 = spatial_heatmap(extractor.hook.get("p3"),
                              channels_by_layer.get("p3", []), (ih, iw))
    heat_p5 = spatial_heatmap(extractor.hook.get("p5"),
                              channels_by_layer.get("p5", []), (ih, iw))

    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(iw, x2 + pad), min(ih, y2 + pad)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    crop = rgb[cy1:cy2, cx1:cx2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(crop)
    axes[0].add_patch(plt.Rectangle((x1 - cx1, y1 - cy1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor="lime", facecolor="none"))
    axes[0].set_title(f"Detection (conf={prov['conf']:.2f})")
    axes[0].axis("off")

    for ax, heat, ttl in [
        (axes[1], heat_p3, "P3 activation (stride 8 — spatial detail)"),
        (axes[2], heat_p5, "P5 activation (stride 32 — semantic depth)"),
    ]:
        ax.imshow(crop)
        if heat is not None:
            ax.imshow(heat[cy1:cy2, cx1:cx2], cmap="jet", alpha=0.5,
                      vmin=0, vmax=max(np.percentile(heat, 95), 1e-6))
        else:
            ax.text(0.5, 0.5, "no neurons\nin this layer", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(ttl)
        ax.axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def generate_examples(extractor, provenance, X, y, schema, top_idx, specs,
                      out_dir, conf_thr=0.25, device="cuda", n_top=8):
    """One activation panel per dataset (spec). Returns a list of dicts:
    {path, spec, role, conf} — role is 'DRONE' or 'CONFUSER'."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = top_channels_by_layer(schema, top_idx, n_top)
    best = pick_best_per_spec(provenance, X, y, top_idx, n_top)
    imgsz_by_spec = {s.name: s.imgsz for s in specs}
    out = []
    for spec_name, row in best.items():
        prov = provenance[row]
        role = "DRONE" if y[row] == 1 else "CONFUSER"
        title = f"{spec_name} — {role}: top discriminative neuron activation"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in spec_name)
        p = out_dir / f"example_{safe}.png"
        r = render_example(extractor, prov, channels, title, p,
                           conf_thr=conf_thr,
                           imgsz=imgsz_by_spec.get(spec_name, 640), device=device)
        if r:
            out.append({"path": r, "spec": spec_name, "role": role,
                        "conf": float(prov["conf"])})
            print(f"    example: {role:8s} {spec_name} -> {p.name}")
    return out
