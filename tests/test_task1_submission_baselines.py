"""Traditional-baseline contracts using synthetic arrays and mocked fits."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from microcture.tasks.task1.evaluation import _read_prediction_csv
from microcture.tasks.task1 import comparison as baselines


class FakeEstimator:
    """Serializable estimator fixture; no optimization or fitted model code."""
    classes_ = np.array([0, 1, 2])

    def fit(self, features, labels):
        self.fit_features = features.copy()
        self.fit_labels = labels.copy()
        return self

    def predict_proba(self, features):
        self.last_prediction_features = features.copy()
        return np.tile(np.array([0.2, 0.5, 0.3]), (len(features), 1))


class FormalBaselineChecks(unittest.TestCase):
    def setUp(self):
        n = 12
        self.samples = {
            "x": (np.arange(n * 2 * 3 * 64 * 64, dtype=np.float32) % 19).reshape(n, 2, 3, 64, 64),
            "length": np.tile(np.arange(6, dtype=np.float32)[:, None], (2, 1)),
            "y": np.tile(np.array([0, 0, 1, 1, 2, 2]), 2),
            "structure_id": np.tile(np.array([f"S{i}" for i in range(6)]), 2),
            "group_id": np.tile(np.arange(6), 2), "replicate": np.repeat([1, 2], 6),
        }
        self.config = {"baseline": {"pca_components": 16, "C": 1.0, "max_iter": 3000,
                                    "tol": 1e-5, "seed": 42},
                       "training": {"augmentation": "reverse", "cpu_threads": 1}}

    def test_geometry_features_cannot_use_signal_or_validity(self):
        changed = {**self.samples, "x": self.samples["x"].copy()}
        changed["x"][:, :, :2] = np.nan
        for name in baselines.BASELINES[:3]:
            np.testing.assert_array_equal(baselines.baseline_features(changed, name),
                                          baselines.baseline_features(self.samples, name))
        with self.assertRaisesRegex(ValueError, "Nonfinite selected pca_lr"):
            baselines.baseline_features(changed, "pca_lr")

    def test_full_pca_features_include_all_three_channels_and_explicit_length(self):
        features = baselines.baseline_features(self.samples, "pca_lr")
        self.assertEqual(features.shape, (12, 24577))
        np.testing.assert_array_equal(features[:, :-1], self.samples["x"].reshape(12, -1))
        np.testing.assert_array_equal(features[:, -1:], self.samples["length"])
        self.assertEqual(baselines.baseline_features(self.samples, "length_only").shape, (12, 1))
        self.assertEqual(baselines.baseline_features(self.samples, "mask_only").shape, (12, 8192))
        self.assertEqual(baselines.baseline_features(self.samples, "mask_length").shape, (12, 8193))

    def test_pipeline_routes_length_around_pca_and_keeps_fixed_parameters(self):
        pipeline = baselines._build_pipeline("pca_lr", self.config["baseline"])
        transformer = pipeline.named_steps["features"]
        image_name, image_pipeline, image_columns = transformer.transformers[0]
        length_name, passthrough, length_columns = transformer.transformers[1]
        self.assertEqual(image_name, "image_pca")
        self.assertEqual(image_columns, slice(0, 24576))
        pca = image_pipeline.named_steps["pca"]
        self.assertEqual(pca.n_components, 16)
        self.assertEqual(pca.svd_solver, "randomized")
        self.assertEqual(pca.random_state, 42)
        self.assertEqual(length_name, "target_length")
        self.assertEqual(passthrough, "passthrough")
        self.assertEqual(length_columns, [24576])
        lr = pipeline.named_steps["lr"]
        self.assertEqual((lr.C, lr.class_weight, lr.solver, lr.max_iter, lr.tol, lr.random_state),
                         (1.0, "balanced", "lbfgs", 3000, 1e-5, 42))
        self.assertEqual(list(baselines._build_pipeline("mask_only", self.config["baseline"]).named_steps),
                         ["scaler", "lr"])

    def test_fitting_receives_only_augmented_train_and_serializes_reusable_predictor(self):
        validation = {**self.samples, "x": self.samples["x"] + 200,
                      "length": self.samples["length"] + 100}
        for name in baselines.BASELINES:
            fake = FakeEstimator()
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="microcture-formal-baseline-test-", dir="/tmp") as temporary:
                output = Path(temporary)
                with patch.object(baselines, "_build_pipeline", return_value=fake):
                    estimator, result = baselines.fit_baseline(name, self.samples, validation, self.config, output)
                self.assertIs(estimator, fake)
                self.assertEqual(result["status"], "completed")
                self.assertFalse(result["outer_images_used"])
                self.assertFalse(result["selection"]["refit_on_train_plus_validation"])
                self.assertFalse(result["selection"]["outer_used_for_selection"])
                n = len(self.samples["y"])
                np.testing.assert_array_equal(fake.fit_labels, np.concatenate((self.samples["y"], self.samples["y"])))
                np.testing.assert_array_equal(fake.fit_features[:n], baselines.baseline_features(self.samples, name))
                reversed_samples = {**self.samples, "x": self.samples["x"][..., ::-1, ::-1]}
                np.testing.assert_array_equal(fake.fit_features[n:], baselines.baseline_features(reversed_samples, name))
                np.testing.assert_array_equal(fake.last_prediction_features, baselines.baseline_features(validation, name))
                self.assertEqual(result["validation_metrics"]["n_structures"], 6)
                self.assertEqual(_read_prediction_csv(output / "validation_predictions.csv")["metrics"]["n_structures"], 6)
                restored = joblib.load(output / "model.joblib")
                np.testing.assert_array_equal(baselines.predict_baseline(restored, name, validation),
                                              np.tile([0.2, 0.5, 0.3], (12, 1)))

    def test_configuration_is_fixed_without_validation_search(self):
        for field, value in (("C", 2), ("pca_components", 8), ("seed", 43), ("tol", 1e-4)):
            config = copy.deepcopy(self.config)
            config["baseline"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "preregistered"):
                baselines._settings(config)
        changed = copy.deepcopy(self.config)
        changed["training"]["augmentation"] = "none"
        with self.assertRaisesRegex(ValueError, "reverse augmentation"):
            baselines._settings(changed)

    def test_missing_training_class_or_existing_evidence_is_not_silently_fitted(self):
        changed = {**self.samples, "y": self.samples["y"] % 2}
        with tempfile.TemporaryDirectory(prefix="microcture-formal-baseline-test-", dir="/tmp") as temporary:
            output = Path(temporary)
            with patch.object(baselines, "_build_pipeline", side_effect=AssertionError("Must not fit")):
                with self.assertRaisesRegex(ValueError, "fixed classes"):
                    baselines.fit_baseline("length_only", changed, self.samples, self.config, output)
                (output / "model.joblib").write_text("existing evidence")
                with self.assertRaises(FileExistsError):
                    baselines.fit_baseline("length_only", self.samples, self.samples, self.config, output)

    def test_convergence_warning_and_invalid_predictor_are_rejected(self):
        fake = FakeEstimator()
        with tempfile.TemporaryDirectory(prefix="microcture-formal-baseline-test-", dir="/tmp") as temporary:
            with patch.object(baselines, "_build_pipeline", return_value=fake), \
                    patch.object(fake, "fit", side_effect=ConvergenceWarning("did not converge")):
                with self.assertRaises(ConvergenceWarning):
                    baselines.fit_baseline("length_only", self.samples, self.samples, self.config, Path(temporary))
        fake.classes_ = np.array([1, 0, 2])
        with self.assertRaisesRegex(ValueError, "class ordering"):
            baselines.predict_baseline(fake, "length_only", self.samples)


if __name__ == "__main__":
    unittest.main()
