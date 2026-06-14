"""Headless cascade smoke (zero-GPU): FusionEngine.predict_paired / predict_grayscale in BOTH cascade
orders with the REAL no-reject router (models/routers/robust8_noreject.joblib), stubbing YOLO + patch
verifiers. Proves the filter->classifier path the user asked about works with robust8_noreject:
  - filter_then_classifier: trust_label in {1,2,3}, 4-vec probs (reject=0), NEVER rejects.
  - classifier_then_filter: same router contract; the patch-veto layer MAY still revoke to reject_both
    (label 0) when it vetoes every trusted modality -- that is the patch veto's job, independent of the
    (no-reject) router.
Validates control-flow + the {1,2,3}+4-vec contract that pyside_engine's filter-first branch also relies on
(probs[orig_trust]); it does NOT exercise real feature numerics (extract_features is stubbed).

Run:  py thesis_eval/_cascade_noreject_smoke.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gui.fusion.engine import FusionEngine, Detection, TRUST_LABELS  # noqa: E402

bundle = joblib.load(REPO / "models/routers/robust8_noreject.joblib")
F8 = bundle["features"]


class StubVerifier:
    """Mimics PatchVerifier.predict_boxes_with_ood -> (P(confuser) per box, ood flags, _)."""
    def __init__(self, prob):
        self.prob = float(prob)
        self.last_labels = []

    def predict_boxes_with_ood(self, img, boxes):
        n = len(boxes)
        self.last_labels = [f"bird:{self.prob:.2f}"] * n
        return np.full(n, self.prob), np.zeros(n, dtype=bool), None


def make_engine(cascade_order, veto_prob):
    e = FusionEngine.__new__(FusionEngine)            # bypass __init__ (no YOLO load)
    e.cascade_order = cascade_order
    e.rgb_conf, e.ir_conf = 0.05, 0.05
    e.rgb_model = e.ir_model = None                   # only passed to the stubbed _run_yolo
    e.use_patch_verifier = True
    e.patch_threshold = 0.9
    e.grayscale_run_ir_filter = True
    e.fusion_clf = bundle["model"]
    e.feature_names = F8
    e.fusion_tau = bundle.get("tau")
    e.fusion_label_map = {int(k): int(v) for k, v in bundle["label_map"].items()}
    e.rgb_verifier = StubVerifier(veto_prob)
    e.ir_verifier = StubVerifier(veto_prob)
    e._run_yolo = lambda model, img, conf: [Detection((10, 10, 60, 60), 0.9),
                                            Detection((80, 80, 140, 150), 0.7)]
    e.extract_features = lambda *a, **k: {f: 0.5 for f in F8}
    return e


img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)


def check(tag, res, allow_reject):
    valid = (0, 1, 2, 3) if allow_reject else (1, 2, 3)
    assert res.trust_label in valid, f"{tag}: unexpected label {res.trust_label}"
    assert len(res.trust_probs) == 4, f"{tag}: probs not len-4: {res.trust_probs}"
    assert res.trust_probs[0] == 0.0, f"{tag}: router assigned reject prob: {res.trust_probs}"
    print(f"  OK {tag:46s} label={res.trust_label} ({TRUST_LABELS[res.trust_label]:11s}) "
          f"probs={[round(p, 3) for p in res.trust_probs]} trusted={len(res.trusted_dets)} veto={res.patch_veto}")


print(f"bundle: {len(F8)} features | tau={bundle.get('tau')} | label_map={bundle.get('label_map')}\n")
print("cascade smoke (real no-reject router, stubbed YOLO + verifiers):")
for order in ("filter_then_classifier", "classifier_then_filter"):
    allow_reject = (order == "classifier_then_filter")   # only the patch-veto layer can reject
    for veto_prob, vlabel in ((0.10, "filter-pass"), (0.99, "veto-all")):
        e = make_engine(order, veto_prob)
        check(f"paired    {order} {vlabel}", e.predict_paired(img, img), allow_reject)
        check(f"grayscale {order} {vlabel}", e.predict_grayscale(img), allow_reject)
print("\nPASS: filter_then_classifier NEVER rejects with robust8_noreject (label in {1,2,3}, 4-vec probs).")
print("      classifier_then_filter keeps the same router contract; reject (label 0) only ever comes from")
print("      the patch-veto layer, not the no-reject router.")
