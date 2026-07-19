from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from classification.model import PaperVGG16
from classification.evaluation import _backend_identity
from simulators.neurosim import neurosim_functional_status
from simulators.neurosim_functional import (
    FunctionalCIMConv2d,
    FunctionalCIMConvTranspose2d,
    FunctionalCIMLinear,
    program_rram_fixed_trial,
    symmetric_fake_quantize,
)


def _neurosim_root() -> Path:
    configured = os.environ.get("MTRD_NEUROSIM_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "NeuroSim"


class TinyWeightedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.up = nn.ConvTranspose2d(4, 2, 2, stride=2)
        self.head = nn.Linear(8, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(torch.relu(self.conv(value)))


class NeuroSimFunctionalAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = TinyWeightedModel().eval()

    def test_paper_profile_has_complete_weighted_operator_coverage(self) -> None:
        programmed, manifest = program_rram_fixed_trial(
            self.model, model_name="unet", sigma=0.3, seed=19,
            profile="paper-8bit", weight_bits=8, dac_bits=8, adc_bits=8,
        )

        self.assertIsInstance(programmed.conv, FunctionalCIMConv2d)
        self.assertIsInstance(programmed.up, FunctionalCIMConvTranspose2d)
        self.assertIsInstance(programmed.head, FunctionalCIMLinear)
        self.assertTrue(manifest["operator_coverage_complete"])
        self.assertEqual(manifest["programmed_layer_count"], 3)
        self.assertEqual(manifest["remaining_weighted_operators"], [])
        self.assertAlmostEqual(manifest["theta_std_population"], 0.3, delta=0.08)
        self.assertFalse(manifest["profile"]["profile_author_verified"])
        self.assertFalse(manifest["profile"]["uses_upstream_cim_array_kernel"])
        self.assertEqual(
            manifest["profile"]["bias_mapping"],
            "digital after ADC-output fake quantization",
        )

    def test_fixed_trial_replays_same_seed_and_changes_with_new_seed(self) -> None:
        first, first_manifest = program_rram_fixed_trial(
            self.model, model_name="unet", sigma=0.2, seed=7,
            profile="paper-8bit",
        )
        replay, replay_manifest = program_rram_fixed_trial(
            self.model, model_name="unet", sigma=0.2, seed=7,
            profile="paper-8bit",
        )
        changed, changed_manifest = program_rram_fixed_trial(
            self.model, model_name="unet", sigma=0.2, seed=8,
            profile="paper-8bit",
        )
        value = torch.randn(2, 1, 8, 8)

        self.assertTrue(torch.equal(first(value), replay(value)))
        self.assertFalse(torch.equal(first(value), changed(value)))
        self.assertEqual(
            first_manifest["layers"][0]["programmed_sha256"],
            replay_manifest["layers"][0]["programmed_sha256"],
        )
        self.assertNotEqual(
            first_manifest["layers"][0]["programmed_sha256"],
            changed_manifest["layers"][0]["programmed_sha256"],
        )

    def test_released_profile_reproduces_conv_only_mapping_boundary(self) -> None:
        programmed, manifest = program_rram_fixed_trial(
            self.model, model_name="unet", sigma=0.1, seed=3,
            profile="released-legacy",
        )

        self.assertIsInstance(programmed.conv, FunctionalCIMConv2d)
        self.assertIsInstance(programmed.up, nn.ConvTranspose2d)
        self.assertIsInstance(programmed.head, FunctionalCIMLinear)
        self.assertFalse(manifest["operator_coverage_complete"])
        self.assertEqual(
            manifest["remaining_weighted_operators"],
            [{"name": "up", "type": "ConvTranspose2d"}],
        )

    def test_released_profile_implements_equation_one_exactly(self) -> None:
        source = nn.Sequential(nn.Linear(4, 3, bias=False)).eval()
        original = source[0].weight.detach().clone()
        sigma = 0.25
        seed = 29
        generator = torch.Generator(device="cpu").manual_seed(seed)
        theta = torch.randn(original.shape, generator=generator) * sigma
        expected = original * torch.exp(theta)

        programmed, _manifest = program_rram_fixed_trial(
            source, model_name="dncnn", sigma=sigma, seed=seed,
            profile="released-legacy",
        )

        self.assertTrue(torch.equal(programmed[0].weight, expected))

    def test_released_profile_completely_maps_vgg_weighted_operators(self) -> None:
        source = nn.Sequential(
            nn.Conv2d(1, 2, 1),
            nn.Flatten(),
            nn.Linear(8, 3),
        ).eval()

        _programmed, manifest = program_rram_fixed_trial(
            source, model_name="vgg16", sigma=0.1, seed=13,
            profile="released-legacy",
        )

        self.assertTrue(manifest["operator_coverage_complete"])
        self.assertEqual(manifest["programmed_layer_count"], 2)
        self.assertIsNone(manifest["weight_bits"])
        self.assertIsNone(manifest["dac_bits"])
        self.assertIsNone(manifest["adc_bits"])

    def test_activation_fake_quantization_is_finite_and_bounded(self) -> None:
        value = torch.tensor([-2.0, -0.9, 0.0, 0.7, 2.0])
        quantized = symmetric_fake_quantize(value, 8)

        self.assertTrue(torch.isfinite(quantized).all())
        self.assertEqual(float(quantized.abs().max()), 2.0)
        self.assertLessEqual(len(torch.unique(quantized)), 255)

    def test_unsupported_weighted_operator_fails_complete_mapping(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "weighted operators"):
            program_rram_fixed_trial(
                nn.Sequential(nn.Conv3d(1, 1, 1)), model_name="unet",
                sigma=0.1, seed=5, profile="paper-8bit",
            )

    def test_grouped_transpose_convolution_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "grouped ConvTranspose2d"):
            program_rram_fixed_trial(
                nn.Sequential(nn.ConvTranspose2d(4, 4, 2, groups=2)),
                model_name="unet", sigma=0.1, seed=5, profile="paper-8bit",
            )

    def test_empty_model_is_not_complete_coverage(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not find"):
            program_rram_fixed_trial(
                nn.ReLU(), model_name="unet", sigma=0.1, seed=5,
                profile="paper-8bit",
            )

    def test_public_adapter_requires_and_accepts_the_pinned_tree(self) -> None:
        root = _neurosim_root()
        status = neurosim_functional_status(root, "unet")

        self.assertTrue(status["source_tree_matches"])
        self.assertTrue(status["public_functional_adapter_ready"])
        self.assertTrue(status["functional_ready"])
        self.assertEqual(
            status["functional_execution_engine"],
            "source-gated-pytorch-extension",
        )
        self.assertTrue(status["neurosim_source_gate_only"])
        self.assertFalse(status["upstream_native_cim_array_kernel_used"])

    def test_source_gated_extension_does_not_claim_native_neurosim(self) -> None:
        root = _neurosim_root()
        identity = _backend_identity(
            SimpleNamespace(
                backend="neurosim",
                neurosim_home=str(root),
                neurosim_functional_profile="released-legacy",
                realization_scope="fixed-trial",
                quantization_bits=8,
            ),
            [],
        )

        self.assertTrue(identity["neurosim_source_gate_verified"])
        self.assertTrue(identity["neurosim_source_gate_only"])
        self.assertFalse(identity["upstream_native_cim_array_kernel_used"])
        self.assertIsNone(identity["weight_bits"])
        self.assertIsNone(identity["dac_bits"])
        self.assertIsNone(identity["adc_bits"])

    def test_vgg16_has_complete_mapping_and_valid_forward(self) -> None:
        source = PaperVGG16(num_classes=100).eval()
        programmed, manifest = program_rram_fixed_trial(
            source, model_name="vgg16", sigma=0.1, seed=31,
            profile="paper-8bit",
        )
        output = programmed(torch.randn(2, 3, 32, 32))

        self.assertEqual(output.shape, (2, 100))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(manifest["programmed_layer_count"], 16)
        self.assertTrue(manifest["operator_coverage_complete"])
        self.assertTrue(
            neurosim_functional_status(
                _neurosim_root(), "vgg16",
            )["functional_ready"]
        )


if __name__ == "__main__":
    unittest.main()
