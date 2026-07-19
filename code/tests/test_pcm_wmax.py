from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from models.noisy_layers import NoisyLinear, pcm_layerwise_wmax
from classification.pytorch import _temporary_fixed_weight_noise
from image_workflows.workflow import equation_forward
from classification.aihwkit_backend import (
    _fixed_programming_noise_model,
    build_additive_config,
    convert_model,
    require_replayable_realization_scope,
)
from classification.model import (
    _perturbed_parameters,
    quantize_weight_tensor,
)


ASYMMETRIC_WEIGHTS = torch.tensor([[-4.0, 1.0]])


class EquationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(ASYMMETRIC_WEIGHTS)

    def forward(self, inputs, _noise=0.0, _noise_type="none"):
        return self.linear(inputs)


class PcmWmaxEquationTest(unittest.TestCase):
    def test_wmax_is_signed_layerwise_maximum(self) -> None:
        self.assertEqual(pcm_layerwise_wmax(ASYMMETRIC_WEIGHTS).item(), 1.0)
        self.assertEqual(ASYMMETRIC_WEIGHTS.abs().amax().item(), 4.0)

    def test_negative_wmax_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative Wmax"):
            pcm_layerwise_wmax(torch.tensor([[-4.0, -1.0]]))

    @patch("torch.randn_like", side_effect=lambda tensor: torch.ones_like(tensor))
    def test_noisy_layer_uses_signed_wmax(self, _random) -> None:
        layer = NoisyLinear(2, 1, bias=False)
        with torch.no_grad():
            layer.linear.weight.copy_(ASYMMETRIC_WEIGHTS)
        output = layer(torch.ones(1, 2), lamda=0.1, noise_type="pcm")
        self.assertTrue(torch.allclose(output, torch.tensor([[-2.8]])))

    @patch("torch.randn_like", side_effect=lambda tensor: torch.ones_like(tensor))
    def test_classification_fixed_trial_uses_signed_wmax_and_restores_weight(self, _random) -> None:
        model = nn.Sequential(nn.Linear(2, 1, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(ASYMMETRIC_WEIGHTS)
        original = model[0].weight.detach().clone()
        with _temporary_fixed_weight_noise([model], "pcm", 0.1):
            self.assertTrue(
                torch.allclose(model[0].weight, ASYMMETRIC_WEIGHTS + 0.1)
            )
        self.assertTrue(torch.equal(model[0].weight, original))

    @patch("torch.randn_like", side_effect=lambda tensor: torch.ones_like(tensor))
    def test_classification_training_uses_signed_wmax(self, _random) -> None:
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(ASYMMETRIC_WEIGHTS)
        replacement = _perturbed_parameters(model, "pcm", 0.1)["weight"]
        self.assertTrue(torch.allclose(replacement, ASYMMETRIC_WEIGHTS + 0.1))

    @patch("torch.randn_like", side_effect=lambda tensor: torch.ones_like(tensor))
    def test_image_workflow_training_uses_signed_wmax(self, _random) -> None:
        model = EquationModel()
        output = equation_forward(model, torch.ones(1, 2), "pcm", 0.1, seed=7)
        self.assertTrue(torch.allclose(output, torch.tensor([[-2.8]])))

    def test_symmetric_quantization_keeps_absolute_maximum(self) -> None:
        quantized = quantize_weight_tensor(ASYMMETRIC_WEIGHTS, bits=3)
        self.assertEqual(quantized[0, 0].item(), -4.0)


@unittest.skipUnless(importlib.util.find_spec("aihwkit"), "AIHWKit is not installed")
class AihwkitSignedWmaxTest(unittest.TestCase):
    def test_formal_evaluation_rejects_unseeded_per_mac_rng(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no public seed"):
            require_replayable_realization_scope("per-mac")

    def test_per_mac_config_requires_explicit_layer_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer-specific"):
            build_additive_config(0.1, realization_scope="per-mac")

    @patch("torch.randn_like", side_effect=lambda tensor: torch.ones_like(tensor))
    def test_fixed_programming_noise_uses_signed_wmax(self, _random) -> None:
        noise_model = _fixed_programming_noise_model(0.1)
        programmed, _ = noise_model.apply_programming_noise(ASYMMETRIC_WEIGHTS)
        self.assertTrue(torch.allclose(programmed, ASYMMETRIC_WEIGHTS + 0.1))

    def test_per_mac_conversion_compensates_aihwkit_absmax_mapping(self) -> None:
        model = nn.Sequential(nn.Linear(2, 1, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(ASYMMETRIC_WEIGHTS)
        analog = convert_model(
            model,
            0.1,
            input_bits=8,
            output_bits=8,
            seed=17,
            realization_scope="per-mac",
        )
        self.assertEqual(analog._mtrd_pcm_signed_max_ratios, {"0": 0.25})
        tile = next(analog.analog_tiles())
        self.assertAlmostEqual(tile.rpu_config.forward.w_noise, 0.025)
        self.assertAlmostEqual(tile.get_mapping_scales().item(), 4.0)

    def test_dac_and_adc_precision_are_configured_independently(self) -> None:
        config = build_additive_config(
            0.1,
            input_bits=6,
            output_bits=7,
            seed=17,
            realization_scope="fixed-trial",
        )
        self.assertAlmostEqual(config.forward.inp_res, 1.0 / 62.0)
        self.assertAlmostEqual(config.forward.out_res, 1.0 / 126.0)


if __name__ == "__main__":
    unittest.main()
