"""Tiny deterministic forwards and hand-written traces; no optimizer training."""

import copy
import json
import unittest

import numpy as np
import torch
from torch import nn

from microcture.task1_controlled.diagnostics import diagnose_model, learning_acceptance


def samples(values, ids=None, labels=None):
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    ids = ids or [str(i // 2) for i in range(n)]
    labels = labels or [i // 2 % 3 for i in range(n)]
    groups = {sid: i for i, sid in enumerate(dict.fromkeys(ids))}
    return {"x": values, "structure_id": np.asarray(ids), "y": np.asarray(labels, dtype=np.int64),
            "group_id": np.asarray([groups[sid] for sid in ids], dtype=np.int64),
            "replicate": np.asarray([str(i % 2 + 1) for i in range(n)])}


class PassLogits(nn.Module):
    def forward(self, x):
        return x.reshape(len(x), 3)


class TinyHead(nn.Module):
    def __init__(self, activation, dropout=0.0):
        super().__init__()
        self.input_linear = nn.Linear(3, 3)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(3, 3)
        with torch.no_grad():
            self.input_linear.weight.copy_(torch.eye(3))
            self.input_linear.bias.zero_()
            self.output_linear.weight.copy_(torch.eye(3))
            self.output_linear.bias.zero_()

    def forward(self, x):
        return self.output_linear(self.dropout(self.activation(self.input_linear(x))))


class TinyModel(nn.Module):
    def __init__(self, activation, dropout=0.0):
        super().__init__()
        self.head = TinyHead(activation, dropout)
        self.inputs_seen = []

    def forward(self, x):
        self.inputs_seen.append(x.detach().cpu().clone())
        return self.head(x.reshape(len(x), 3))


class TinyLengthModel(TinyModel):
    def __init__(self):
        super().__init__(nn.GELU())
        self.lengths_seen = []

    def forward(self, x, length):
        self.lengths_seen.append(length.detach().cpu().clone())
        return super().forward(x + length)


class DiagnosticForwardTests(unittest.TestCase):
    def test_across_class_variation_is_not_between_structure_variation(self):
        data = samples([[10, -2, 1]] * 6)
        result = diagnose_model(PassLogits(), data, "cpu", 2)
        self.assertEqual(result["n_structures"], 3)
        self.assertEqual(result["logit_between_structure_var"], 0)
        self.assertAlmostEqual(result["probability_between_structure_var"], 0, delta=1e-20)
        self.assertTrue(result["logits_constant"])
        self.assertEqual(result["num_predicted_classes"], 1)
        self.assertEqual(result["majority_prediction_fraction"], 1)

    def test_replicate_logits_and_probabilities_are_averaged_separately(self):
        data = samples([[-1, 0, 1], [1, 0, -1], [-2, 0, 2], [2, 0, -2]])
        result = diagnose_model(PassLogits(), data, "cpu", 3)
        self.assertEqual(result["n_structures"], 2)
        self.assertEqual(result["replicate_count_min"], 2)
        self.assertEqual(result["logit_between_structure_var"], 0)
        self.assertGreater(result["probability_between_structure_var"], 0)
        self.assertFalse(result["probabilities_constant"])
        self.assertEqual(result["metrics"]["n_structures"], 2)

    def test_negative_gelu_values_are_effective_units(self):
        model = TinyModel(nn.GELU())
        result = diagnose_model(model, samples([[-1, -2, -0.5]] * 6), "cpu", 4)
        self.assertEqual(result["head"]["positive_fraction"], 0)
        self.assertEqual(result["head"]["effective_fraction"], 1)
        self.assertEqual(result["head"]["inactive_units"], 0)
        self.assertFalse(result["head"]["positive_fraction_is_dead_unit_test"])
        self.assertEqual(result["head"]["preactivation"]["positive_fraction"], 0)

    def test_relu_completely_inactive_units_are_reported(self):
        model = TinyModel(nn.ReLU())
        result = diagnose_model(model, samples([[-1, -2, -0.5]] * 6), "cpu", 4)
        self.assertEqual(result["head"]["inactive_units"], 3)
        self.assertEqual(result["head"]["effective_fraction"], 0)
        self.assertTrue(result["logits_constant"])

    def test_dropout_eval_masks_preserved_and_original_modes_restored(self):
        model = TinyModel(nn.GELU(), dropout=0.8)
        model.train()
        model.head.input_linear.eval()  # Preserve a deliberately mixed mode tree.
        data = samples([[1, 0, 1], [1, 0, 1], [2, 1, 0], [2, 1, 0]])
        initial_modes = [module.training for module in model.modules()]
        result1 = diagnose_model(model, data, "cpu", 3)
        np.testing.assert_array_equal(torch.cat(model.inputs_seen).numpy(), data["x"])
        model.inputs_seen.clear()
        result2 = diagnose_model(model, data, "cpu", 3)
        self.assertEqual(result1, result2)
        self.assertEqual([module.training for module in model.modules()], initial_modes)
        self.assertEqual(len(model.head.activation._forward_hooks), 0)
        json.dumps(result1, allow_nan=False)

    def test_common_logit_offset_is_not_discriminative_variation(self):
        data = samples([[2, 1, 0], [2, 1, 0], [12, 11, 10], [12, 11, 10]])
        result = diagnose_model(PassLogits(), data, "cpu", 4)
        self.assertGreater(result["logit_between_structure_var"], 1)
        self.assertFalse(result["logits_constant"])
        self.assertTrue(result["centered_logits_constant"])
        self.assertTrue(result["probabilities_constant"])

    def test_length_side_channel_is_passed_without_mutation(self):
        model = TinyLengthModel()
        data = samples([[1, 0, 1], [1, 0, 1], [2, 1, 0], [2, 1, 0]])
        data["length"] = np.asarray([[0], [0], [1.2], [1.2]], dtype=np.float32)
        expected = data["length"].copy()
        result = diagnose_model(model, data, "cpu", 3)
        np.testing.assert_array_equal(torch.cat(model.lengths_seen).numpy(), expected)
        np.testing.assert_array_equal(data["length"], expected)
        self.assertEqual(result["n_structures"], 2)

    def test_classwise_loss_support_recovers_weighted_replicate_ce(self):
        data = samples([[2, 1, 0], [2, 1, 0], [0, 1, 2], [0, 1, 2], [1, 0, 2], [1, 0, 2]])
        result = diagnose_model(PassLogits(), data, "cpu", 3)
        support = result["loss_support"]
        weights = np.asarray([1.2, 0.5, 4.0])
        recovered = np.dot(weights, support["unweighted_nll_sum_by_class"]) / np.dot(weights, support["examples_count_by_class"])
        direct = torch.nn.functional.cross_entropy(torch.tensor(data["x"]), torch.tensor(data["y"]),
                                                  weight=torch.tensor(weights, dtype=torch.float32)).item()
        self.assertAlmostEqual(recovered, direct, delta=1e-6)
        self.assertEqual(support["examples_count_by_class"], [2, 2, 2])

    def test_loss_support_remains_finite_for_extreme_finite_logits(self):
        data = samples([[1000, 0, -1000]] * 6)
        result = diagnose_model(PassLogits(), data, "cpu", 4)
        self.assertTrue(np.isfinite(result["loss_support"]["unweighted_nll_sum_by_class"]).all())
        json.dumps(result, allow_nan=False)


def trace_diagnostic(majority=0.5):
    return {"logit_between_structure_var": 0.1, "centered_logit_between_structure_var": 0.08,
            "logits": {"max_coordinate_range": 1.0},
            "centered_logits": {"max_coordinate_range": 0.9},
            "probabilities": {"max_coordinate_range": 0.3},
            "head": {"available": True, "active_units": 4, "total_units": 4,
                     "effective_fraction": 0.9, "positive_fraction": 0.0, "epsilon": 1e-6},
            "majority_prediction_fraction": majority,
            "metrics": {"per_class": {"CHID": {"f1": 0.0}}}}


def history():
    rows = [{"epoch": 0, "weighted_train_ce": 1.0, "gradient_norms": None,
             "train_diagnostics": trace_diagnostic(), "validation_diagnostics": trace_diagnostic()}]
    for epoch in range(1, 16):
        gradients = {part: {"mean": 0.1, "min": 0.08, "max": 0.12, "finite": True, "nonzero_fraction": 1.0}
                     for part in ("total", "encoder", "head")}
        rows.append({"epoch": epoch, "weighted_train_ce": 0.8 - epoch * 0.01,
                     "train_diagnostics": trace_diagnostic(), "validation_diagnostics": trace_diagnostic(),
                     "gradient_norms": gradients})
    return rows


class LearningAcceptanceTests(unittest.TestCase):
    def test_sustained_learning_can_pass_without_high_chid_f1(self):
        result = learning_acceptance(history(), {})
        self.assertTrue(result["accepted"])
        self.assertEqual(result["primary_window"], 10)
        self.assertEqual(result["initial_reference_epoch"], 0)
        self.assertFalse(result["CHID_F1_used_as_acceptance_threshold"])
        json.dumps(result, allow_nan=False)

    def test_variable_logits_all_one_class_are_separate_from_constant(self):
        rows = history()
        for row in rows:
            row["train_diagnostics"]["majority_prediction_fraction"] = 1.0
            row["validation_diagnostics"]["majority_prediction_fraction"] = 1.0
        result = learning_acceptance(rows, {})
        self.assertTrue(result["learning_detected"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["primary"]["views"]["train"]["logits_constant_epochs"], 0)
        self.assertIn("train_prediction_concentrated", result["primary"]["failed_checks"])

    def test_constant_logits_cannot_pass_despite_loss_decline(self):
        rows = history()
        for row in rows:
            for name in ("train", "validation"):
                d = row[f"{name}_diagnostics"]
                d["logit_between_structure_var"] = 0
                d["logits"]["max_coordinate_range"] = 0
        result = learning_acceptance(rows, {})
        self.assertFalse(result["learning_detected"])

    def test_nonfinite_or_zero_encoder_gradient_rejects(self):
        for value, finite in ((0.0, True), (float("nan"), False)):
            rows = history()
            rows[-1]["gradient_norms"]["encoder"]["mean"] = value
            rows[-1]["gradient_norms"]["encoder"]["finite"] = finite
            self.assertFalse(learning_acceptance(rows, {})["accepted"])

    def test_negative_gelu_activation_is_not_rejected_for_positive_fraction_zero(self):
        rows = history()
        self.assertTrue(all(row["train_diagnostics"]["head"]["positive_fraction"] == 0 for row in rows))
        self.assertTrue(learning_acceptance(rows, {})["accepted"])

    def test_terminal_regression_rejects_early_transient_learning(self):
        rows = history()
        for row in rows[1:6]:
            row["weighted_train_ce"] = 0.3
        for row in rows[6:]:
            row["weighted_train_ce"] = 0.8
        result = learning_acceptance(rows, {"acceptance": {"window": 5}})
        self.assertFalse(result["accepted"])
        self.assertFalse(result["primary"]["checks"]["terminal_not_degraded"])

    def test_last_epoch_must_still_show_loss_drop(self):
        rows = history()
        rows[-1]["weighted_train_ce"] = 1.0
        result = learning_acceptance(rows, {})
        self.assertFalse(result["accepted"])
        self.assertFalse(result["primary"]["checks"]["terminal_loss_drop"])

    def test_insufficient_window_is_explicit_and_history_is_not_mutated(self):
        rows = history()[:5]
        before = copy.deepcopy(rows)
        result = learning_acceptance(rows, {})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["accepted"])
        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
