"""Headless smoke: the GUI FusionEngine.classify() with the no-reject router (label_map path).
Skips detector/verifier loading via __new__; tests only the classifier wiring."""
import sys, random, joblib
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gui.fusion.engine import FusionEngine   # also verifies engine.py imports cleanly (my edits)

b = joblib.load(REPO / "models/routers/robust8_noreject.joblib")
print("bundle:", len(b["features"]), "features | tau:", b.get("tau"), "| label_map:", b.get("label_map"))
F8 = b["features"]
stub = FusionEngine.__new__(FusionEngine)
stub.fusion_clf = b["model"]; stub.feature_names = F8; stub.fusion_tau = b.get("tau")
stub.fusion_label_map = {int(k): int(v) for k, v in b["label_map"].items()} if b.get("label_map") else None

random.seed(0); seen = set()
for _ in range(300):
    feats = {f: random.uniform(0, 1) for f in F8}
    feats["rgb_best_log_bbox_area"] = random.uniform(2, 9)
    feats["ir_best_log_bbox_area"] = random.uniform(2, 9)
    feats["is_grayscale"] = random.choice([0, 1])
    label, probs = FusionEngine.classify(stub, feats)
    assert label in (1, 2, 3), f"INVALID/REJECT label {label}"
    assert len(probs) == 4 and probs[0] == 0.0, f"probs not 4-vec with reject=0: {probs}"
    assert abs(sum(probs) - 1.0) < 1e-6, f"probs don't sum to 1: {probs}"
    seen.add(label)
print(f"OK: 300 synthetic frames -> labels seen {sorted(seen)} (never 0/reject); "
      f"probs length 4, reject-prob 0, sums to 1.")

# cross-check classify() vs raw model.predict + label_map
X = np.array([[feats[f] for f in F8]])
raw = int(stub.fusion_clf.predict(X)[0]); mapped = stub.fusion_label_map[raw]
lab, pr = FusionEngine.classify(stub, feats)
print(f"cross-check: model.predict={raw} -> label_map -> {mapped} ; classify -> {lab} ; match={mapped==lab}")
print(f"probs[trust] indexing: P(label={lab}) = {pr[lab]:.3f} (4-vec {[round(p,3) for p in pr]})")
