import joblib
from pathlib import Path

REPO = Path("c:/Users/User/Desktop/UNISA projects/Drone detection/es proj 3 thesis workspace/es_drone_detection")
clf_data = joblib.load(REPO / "classifier/fusion_models/scene_aware_v3more_32feat/model.joblib")
classifier = clf_data["model"]
print("Classifier type:", type(classifier))
print("Classes:", classifier.classes_)
if hasattr(classifier, "feature_importances_"):
    print("Feature importances count:", len(classifier.feature_importances_))
