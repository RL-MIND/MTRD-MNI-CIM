from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from image_workflows.workflow import (
    aggregate_rows,
    equation_forward,
    evaluate_segmentation_model,
    feedback_metric,
    resolve_segmentation_metric_contract,
    resolve_segmentation_mtrd_bn_update_policy,
    segmentation_nominal_kd_forward,
)


class _MetricFixture(Dataset):
    def __init__(self) -> None:
        self.images = [
            torch.ones(1, 2, 2),
            torch.ones(1, 2, 2),
            -torch.ones(1, 2, 2),
        ]
        self.masks = [
            torch.ones(2, 2),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.masks[index], f"sample-{index}"


class _InputLogitModel(nn.Module):
    def forward(self, inputs, _level=0.0, _noise_type="none"):
        return inputs[:, :1]


class _BatchNormFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1, momentum=1.0)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()

    def forward(self, inputs, _level=0.0, _noise_type="none"):
        return self.bn(self.conv(inputs))


def _metric_config(aggregation: str) -> dict[str, object]:
    return {
        "seed": 19,
        "protocol": {
            "segmentation": {
                "batch_size": 2,
                "num_workers": 0,
                "metric_aggregation": aggregation,
            }
        },
    }


def _backend_info() -> dict[str, object]:
    return {
        "name": "fixture",
        "realization_scope": "fixed_trial",
        "weight_bits": 8,
        "dac_bits": 8,
        "adc_bits": 8,
    }


class UnetFeedbackAndBatchNormTest(unittest.TestCase):
    def test_feedback_metric_matches_final_per_image_and_batch_global_semantics(self) -> None:
        expected = {
            "per_image_mean": (3, 2.0 / 3.0),
            "released_batch_global": (2, 1.0 / 3.0),
        }
        for aggregation, (observation_count, expected_value) in expected.items():
            with self.subTest(aggregation=aggregation):
                config = _metric_config(aggregation)
                model = _InputLogitModel()
                dataset = _MetricFixture()
                feedback = feedback_metric(
                    model,
                    dataset,
                    task="segmentation",
                    device_model="rram",
                    level=0.0,
                    config=config,
                    device=torch.device("cpu"),
                    seed_namespace="feedback-test",
                )
                rows = evaluate_segmentation_model(
                    model,
                    dataset,
                    backend="neurosim",
                    backend_info=_backend_info(),
                    device_model="rram",
                    level=0.0,
                    trial=0,
                    trial_seed=23,
                    config=config,
                    method="nominal",
                    checkpoint=Path("/tmp/unet-feedback-fixture.pth"),
                    checkpoint_hash="fixture",
                )
                trial_rows, _summary_rows = aggregate_rows(rows)

                self.assertEqual(len(rows), observation_count)
                self.assertEqual(trial_rows[0]["observation_count"], observation_count)
                self.assertAlmostEqual(feedback, expected_value, places=7)
                self.assertAlmostEqual(
                    feedback,
                    float(trial_rows[0]["metric_mean"]),
                    places=7,
                )
                self.assertEqual(
                    feedback,
                    feedback_metric(
                        model,
                        dataset,
                        task="segmentation",
                        device_model="rram",
                        level=0.0,
                        config=config,
                        device=torch.device("cpu"),
                        seed_namespace="feedback-test",
                    ),
                )

    def test_metric_contract_defaults_and_rejects_unknown_aggregation(self) -> None:
        default = resolve_segmentation_metric_contract({"protocol": {"segmentation": {}}})
        self.assertEqual(default["metric_aggregation"], "per_image_mean")
        self.assertFalse(default["configured_explicitly"])

        invalid = _metric_config("not-a-metric")
        with self.assertRaisesRegex(ValueError, "metric_aggregation"):
            resolve_segmentation_metric_contract(invalid)

    def test_noisy_task_only_preserves_persistent_buffers_and_gradients(self) -> None:
        model = _BatchNormFixture().train()
        inputs = torch.tensor(
            [
                [[[0.0, 1.0], [2.0, 3.0]]],
                [[[4.0, 5.0], [6.0, 7.0]]],
            ]
        )
        policy = resolve_segmentation_mtrd_bn_update_policy({
            "protocol": {"segmentation": {"mtrd_bn_update_policy": "noisy_task_only"}}
        })
        mean_before = model.bn.running_mean.detach().clone()
        variance_before = model.bn.running_var.detach().clone()
        batches_before = model.bn.num_batches_tracked.detach().clone()

        nominal = segmentation_nominal_kd_forward(
            model,
            inputs,
            bn_update_policy=policy,
        )
        self.assertTrue(torch.equal(model.bn.running_mean, mean_before))
        self.assertTrue(torch.equal(model.bn.running_var, variance_before))
        self.assertTrue(torch.equal(model.bn.num_batches_tracked, batches_before))
        nominal.square().mean().backward()
        self.assertIsNotNone(model.bn.weight.grad)
        self.assertGreater(model.bn.weight.grad.abs().sum().item(), 0.0)

        with torch.no_grad():
            equation_forward(model, inputs, "rram", 0.1, seed=29)
        self.assertEqual(model.bn.num_batches_tracked.item(), batches_before.item() + 1)
        self.assertFalse(torch.equal(model.bn.running_mean, mean_before))

    def test_legacy_policy_updates_buffers_for_both_forwards(self) -> None:
        model = _BatchNormFixture().train()
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
        policy = resolve_segmentation_mtrd_bn_update_policy({"protocol": {"segmentation": {}}})

        self.assertEqual(policy["policy"], "legacy_dual_branch")
        self.assertFalse(policy["configured_explicitly"])
        segmentation_nominal_kd_forward(model, inputs, bn_update_policy=policy)
        self.assertEqual(model.bn.num_batches_tracked.item(), 1)
        with torch.no_grad():
            equation_forward(model, inputs, "rram", 0.1, seed=31)
        self.assertEqual(model.bn.num_batches_tracked.item(), 2)

    def test_batch_norm_policy_rejects_unknown_value(self) -> None:
        invalid = {
            "protocol": {
                "segmentation": {"mtrd_bn_update_policy": "automatic"}
            }
        }
        with self.assertRaisesRegex(ValueError, "mtrd_bn_update_policy"):
            resolve_segmentation_mtrd_bn_update_policy(invalid)


if __name__ == "__main__":
    unittest.main()
