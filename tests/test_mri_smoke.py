"""
Synthetic smoke test for the mri package — no GPU, no YOLO, no real data.

Builds two Gaussian blobs as a fake feature corpus and exercises the numeric +
plotting + classifier + diagnosis layers end-to-end, asserting the contract the
CLI relies on. Guards against schema/return-shape drift in refactors.

    python -m pytest tests/test_mri_smoke.py        (or just run the file)
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))  # for `from metrics import ...`

from mri.extract import FeatureSchema
from mri import stats, plots, diagnose
from mri.classifier import MLPWrapper, cross_val_score_f1


def _fake_corpus(n=300, dim=37, seed=0):
    """Separable-ish two-class corpus. dim = 5 meta + 32 'yolo'."""
    rng = np.random.RandomState(seed)
    n_pos = n // 2
    drones = rng.normal(0.0, 1.0, (n_pos, dim)) + 1.2
    confs = rng.normal(0.0, 1.0, (n - n_pos, dim)) - 1.2
    X = np.vstack([drones, confs]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(np.float32)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _schema(dim=37):
    s = FeatureSchema(layers=("p5",), grids={"p5": (1, 1)})
    s.layer_dims["p5"] = dim - s.meta_dim
    return s


def test_stats_contract():
    X, y = _fake_corpus()
    schema = _schema(X.shape[1])
    summary, F, auroc, top = stats.separability_summary(X, y, schema)
    assert F.shape[0] == X.shape[1]
    assert auroc.shape[0] == X.shape[1]
    assert 0.0 <= summary["lda_train_accuracy"] <= 1.0
    assert len(top) == X.shape[1]
    assert summary["lda_train_accuracy"] > 0.8  # blobs are separable


def test_plots_written(tmp_path):
    X, y = _fake_corpus()
    schema = _schema(X.shape[1])
    _, F, _, top = stats.separability_summary(X, y, schema)
    figs = plots.generate_all(X, y, schema, F, top, tmp_path,
                              want=["pca", "lda", "anova", "heatmap", "neurons"])
    assert figs and all(Path(f).exists() for f in figs)


def test_classifier_and_diagnosis():
    X, y = _fake_corpus()
    schema = _schema(X.shape[1])
    summary, _, _, _ = stats.separability_summary(X, y, schema)
    f1, sd, best, oof = cross_val_score_f1(
        MLPWrapper, {"input_dim": X.shape[1], "device": "cpu", "epochs": 8}, X, y)
    assert 0.0 <= f1 <= 1.0
    assert oof.shape[0] == len(y)
    raws = [
        {"name": "pos", "role": "pos", "n_images": 100, "n_dets": 120,
         "tp": 90, "fp": 10, "fn": 10, "mined_drones": 90, "mined_confusers": 10},
        {"name": "neg", "role": "neg", "n_images": 100, "n_dets": 60,
         "tp": 0, "fp": 60, "fn": 0, "mined_drones": 0, "mined_confusers": 60},
    ]
    d = diagnose.diagnose(raws, summary, oof=oof, y=y)
    assert d["verdict"] in diagnose.VERDICTS
    assert d["raw_halluc_rate"] == 0.6  # 60 FP / 100 neg imgs
    assert "recall_cost" in d


if __name__ == "__main__":
    test_stats_contract()
    test_plots_written(Path("./_mri_smoke_figs"))
    test_classifier_and_diagnosis()
    print("OK — all mri smoke checks passed")
