"""Fixed target-aware traditional baselines, fitted on inner training only.

The PCA image projection and both scalers are part of the serialized pipeline.
No validation parameter search or train+validation refit is implemented.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from microcture.task1.evaluation import grouped_predictions, metrics, save_prediction_csv


BASELINES = ("length_only", "mask_only", "mask_length", "pca_lr")
IMAGE_FEATURE_COUNT = 2 * 3 * 64 * 64
MASK_FEATURE_COUNT = 2 * 64 * 64


def baseline_features(samples: dict, name: str) -> np.ndarray:
    """Select geometry or flattened full I3 image plus the length column.

    Geometry baselines do not inspect signal or validity values. This matters
    when auditing that they cannot use either contact or missingness patterns.
    """
    if name not in BASELINES:
        raise ValueError(f"Unknown formal baseline: {name}; choose {BASELINES}")
    n = len(samples["y"])
    if samples["x"].shape != (n, 2, 3, 64, 64) or samples["length"].shape != (n, 1):
        raise ValueError("Unexpected formal baseline input layout")
    if name == "length_only":
        features = samples["length"].copy()
    elif name == "pca_lr":
        features = np.concatenate((samples["x"].reshape(n, -1), samples["length"]), axis=1)
    else:
        features = samples["x"][:, :, 2].reshape(n, -1).copy()
        if name == "mask_length":
            features = np.concatenate((features, samples["length"]), axis=1)
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite selected {name} features")
    return np.ascontiguousarray(features)


def _settings(config: dict) -> dict:
    baseline = config["baseline"]
    expected = {"pca_components": 16, "C": 1.0, "max_iter": 3000, "tol": 1e-5, "seed": 42}
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise ValueError(f"Formal baseline.{key} must remain preregistered at {value}")
    if config["training"].get("augmentation") != "reverse":
        raise ValueError("Formal baselines require the frozen reverse augmentation")
    return baseline


def _build_pipeline(name: str, settings: dict) -> Pipeline:
    steps = []
    if name == "pca_lr":
        # Only image columns enter PCA. The existing inner-training normalized
        # length remains an explicit feature and cannot be lost in projection.
        image = Pipeline([
            ("image_scaler", StandardScaler()),
            ("pca", PCA(n_components=settings["pca_components"], svd_solver="randomized",
                        random_state=settings["seed"])),
        ])
        features = ColumnTransformer([
            ("image_pca", image, slice(0, IMAGE_FEATURE_COUNT)),
            ("target_length", "passthrough", [IMAGE_FEATURE_COUNT]),
        ], remainder="drop", sparse_threshold=0)
        steps.append(("features", features))
    steps.extend([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=settings["C"], solver="lbfgs", class_weight="balanced",
                                  max_iter=settings["max_iter"], tol=settings["tol"],
                                  random_state=settings["seed"])),
    ])
    return Pipeline(steps)


def predict_baseline(estimator: Pipeline, name: str, samples: dict) -> np.ndarray:
    """Predict with the fitted pipeline without modifying its fitted state."""
    if not np.array_equal(estimator.classes_, [0, 1, 2]):
        raise ValueError("Unexpected formal baseline class ordering")
    probabilities = np.asarray(estimator.predict_proba(baseline_features(samples, name)), dtype=np.float64)
    # The shared evaluator checks shape, finite bounds and probability sums.
    grouped_predictions(samples, probabilities)
    return probabilities


def fit_baseline(name: str, train: dict, validation: dict, config: dict,
                 output: Path) -> tuple[Pipeline, dict]:
    """Fit one fixed baseline and retain validation evidence, without outer data."""
    if name not in BASELINES:
        raise ValueError(f"Unknown formal baseline: {name}; choose {BASELINES}")
    settings = _settings(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.joblib").exists() or (output / "validation_predictions.csv").exists():
        raise FileExistsError("Baseline output already contains evidence; use a fresh job directory")
    y = np.asarray(train["y"])
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or not np.array_equal(np.unique(y), [0, 1, 2]):
        raise ValueError("Every formal baseline training subset must contain fixed classes 0, 1, 2")
    augmented = {**train,
                 "x": np.concatenate((train["x"], train["x"][..., ::-1, ::-1]), axis=0),
                 "length": np.concatenate((train["length"], train["length"]), axis=0),
                 "y": np.concatenate((y, y), axis=0)}
    features = baseline_features(augmented, name)
    if name == "pca_lr" and settings["pca_components"] > min(features.shape[0], IMAGE_FEATURE_COUNT):
        raise ValueError("Inner-training examples are insufficient for the fixed 16-component PCA")
    pipeline = _build_pipeline(name, settings)
    with warnings.catch_warnings(), threadpool_limits(limits=config["training"].get("cpu_threads", 4)):
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features, augmented["y"])
    probabilities = predict_baseline(pipeline, name, validation)
    grouped = save_prediction_csv(output / "validation_predictions.csv", validation, probabilities)
    joblib.dump(pipeline, output / "model.joblib")
    feature_spec = {
        "name": name, "input_condition": "I3", "raw_feature_count": int(features.shape[1]),
        "signal_features_used": name == "pca_lr", "validity_features_used": name == "pca_lr",
        "mask_features_used": name != "length_only", "length_features_used": name != "mask_only",
        "image_layout": "scale,channel(signal/valid_fraction/target_interval_fraction),row,column",
        "fit_scope": "inner-training replicate examples and their fixed reverse augmentation only",
        "length_scaler_scope": "unique inner-training structures only, already fitted by prepare_inner",
        "pipeline": "StandardScaler -> balanced LogisticRegression",
    }
    if name == "pca_lr":
        feature_spec.update({
            "image_features_before_pca": IMAGE_FEATURE_COUNT,
            "pca_components": settings["pca_components"],
            "features_after_pca_and_length": settings["pca_components"] + 1,
            "length_bypasses_pca": True,
            "pipeline": "image: StandardScaler -> PCA(16); concatenate length; StandardScaler -> balanced LogisticRegression",
            "pca_solver": "randomized", "pca_random_state": settings["seed"],
        })
    result = {
        "status": "completed", "validation_metrics": {**metrics(grouped["y"], grouped["probabilities"]),
                                                       "replicate_disagreement": grouped["replicate_disagreement"]},
        "feature_spec": feature_spec,
        "selection": {"rule": "fixed preregistered parameters; no hyperparameter search",
                      "C": settings["C"], "seed": settings["seed"], "max_iter": settings["max_iter"],
                      "tol": settings["tol"], "solver": "lbfgs", "class_weight": "balanced",
                      "refit_on_train_plus_validation": False, "outer_used_for_selection": False},
        "outer_images_used": False, "acceptance": None,
        "training_examples_before_augmentation": int(len(y)),
        "training_examples_after_augmentation": int(len(augmented["y"])),
    }
    return pipeline, result
