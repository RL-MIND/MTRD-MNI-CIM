from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from models.dncnn import DnCNN
from models.unet import UNet
from image_workflows.workflow import (
    CarvanaRecord,
    CarvanaSplitDataset,
    aggregate_rows,
    configured_carvana_resize_backend,
    configured_neurosim_ptq,
    evaluate_segmentation_model,
)
from simulators.static_ptq import (
    released_static_qconfig,
    static_quantize_calibrated,
)
from simulators.neurosim_functional import program_rram_fixed_trial


class _SegmentationFixture(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        return torch.ones(3, 16, 16), torch.ones(16, 16), f"sample-{index}"


class _PositiveSegmentationModel(torch.nn.Module):
    def forward(self, images, _level=0.0, _noise_type="none"):
        return torch.ones(images.shape[0], 1, images.shape[2], images.shape[3])


class StaticPTQTest(unittest.TestCase):
    def test_released_qconfig_uses_requested_integer_ranges(self) -> None:
        qconfig = released_static_qconfig(activation_bits=6, weight_bits=6)
        activation = qconfig.activation()
        weight = qconfig.weight()

        self.assertEqual((activation.quant_min, activation.quant_max), (0, 63))
        self.assertEqual((weight.quant_min, weight.quant_max), (-32, 31))
        with self.assertRaisesRegex(ValueError, "activation_bits"):
            released_static_qconfig(activation_bits=1, weight_bits=6)

    def test_dncnn_static_ptq_converts_all_compute_convolutions(self) -> None:
        model = DnCNN(channels=1, num_of_layers=5, features=8).eval()
        converted, manifest = static_quantize_calibrated(
            model,
            [torch.rand(2, 1, 16, 16)],
            activation_bits=6,
            weight_bits=6,
        )

        output = converted(torch.rand(2, 1, 16, 16), 0.0, "none")
        self.assertEqual(output.shape, (2, 1, 16, 16))
        converted_types = manifest["quantized_module_types"]
        self.assertEqual(
            converted_types["torch.ao.nn.quantized.modules.conv.Conv2d"], 5
        )
        self.assertEqual(manifest["calibration_sample_count"], 2)

    def test_programmed_dncnn_preserves_convertible_modules_for_ptq(self) -> None:
        model = DnCNN(channels=1, num_of_layers=5, features=8).eval()
        programmed, programming = program_rram_fixed_trial(
            model,
            model_name="dncnn",
            sigma=0.3,
            seed=23,
            profile="released-legacy",
            preserve_standard_modules=True,
        )
        converted, manifest = static_quantize_calibrated(
            programmed,
            [torch.rand(2, 1, 16, 16)],
            activation_bits=6,
            weight_bits=6,
        )

        output = converted(torch.rand(2, 1, 16, 16), 0.0, "none")
        self.assertEqual(output.shape, (2, 1, 16, 16))
        self.assertTrue(programming["preserved_standard_modules_for_ptq"])
        self.assertEqual(
            manifest["quantized_module_types"][
                "torch.ao.nn.quantized.modules.conv.Conv2d"
            ],
            5,
        )

    def test_unet_static_ptq_converts_transposed_convolutions(self) -> None:
        model = UNet(features=(4, 8)).eval()
        converted, manifest = static_quantize_calibrated(
            model,
            [torch.rand(1, 3, 16, 16)],
            activation_bits=6,
            weight_bits=6,
        )

        output = converted(torch.rand(1, 3, 16, 16), 0.0, "none")
        self.assertEqual(output.shape, (1, 1, 16, 16))
        converted_types = manifest["quantized_module_types"]
        self.assertEqual(
            converted_types[
                "torch.ao.nn.quantized.modules.conv.ConvTranspose2d"
            ],
            2,
        )

    def test_released_unet_programming_uses_source_registration_order(self) -> None:
        model = UNet(features=(4, 8)).eval()
        _programmed, manifest = program_rram_fixed_trial(
            model,
            model_name="unet",
            sigma=0.1,
            seed=23,
            profile="released-legacy",
            preserve_standard_modules=True,
        )

        names = [record["name"] for record in manifest["layers"]]
        self.assertEqual(
            names,
            [
                "ups_conv.0.conv1.conv",
                "ups_conv.0.conv2.conv",
                "ups_conv.1.conv1.conv",
                "ups_conv.1.conv2.conv",
                "downs.0.conv1.conv",
                "downs.0.conv2.conv",
                "downs.1.conv1.conv",
                "downs.1.conv2.conv",
                "bottleneck.conv1.conv",
                "bottleneck.conv2.conv",
                "final_conv.conv",
            ],
        )

    def test_ptq_configuration_is_explicit_and_validated(self) -> None:
        config = {
            "simulation": {
                "neurosim_functional_profile": "released-legacy",
                "neurosim_post_training_quantization": {
                    "mode": "released-eager-static",
                    "activation_bits": 6,
                    "weight_bits": 6,
                    "calibration_data": "evaluation",
                    "engine": "fbgemm",
                }
            }
        }
        self.assertEqual(configured_neurosim_ptq(config)["weight_bits"], 6)
        config["simulation"]["neurosim_post_training_quantization"][
            "calibration_data"
        ] = "training"
        with self.assertRaisesRegex(ValueError, "calibration_data=evaluation"):
            configured_neurosim_ptq(config)

    def test_released_carvana_resize_matches_opencv(self) -> None:
        import cv2

        image_array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        mask_array = np.zeros((5, 7), dtype=np.uint8)
        mask_array[:, 3:] = 255
        with tempfile.TemporaryDirectory(prefix="mtrd-carvana-resize-") as temporary:
            root = Path(temporary)
            image_path = root / "sample.png"
            mask_path = root / "sample_mask.png"
            Image.fromarray(image_array).save(image_path)
            Image.fromarray(mask_array).save(mask_path)
            dataset = CarvanaSplitDataset(
                {
                    "sample": CarvanaRecord(
                        sample_id="sample", image=image_path, mask=mask_path
                    )
                },
                ["sample"],
                height=3,
                width=4,
                train=False,
                seed=17,
                resize_backend="released-opencv",
            )
            image, mask, sample_id = dataset[0]

        expected_image = cv2.resize(
            image_array, (4, 3), interpolation=cv2.INTER_LINEAR
        )
        expected_mask = cv2.resize(
            mask_array, (4, 3), interpolation=cv2.INTER_NEAREST
        )
        self.assertEqual(sample_id, "sample")
        self.assertTrue(
            torch.equal(
                image,
                torch.from_numpy(expected_image.transpose(2, 0, 1).copy()).float()
                / 255.0,
            )
        )
        self.assertTrue(
            torch.equal(mask, torch.from_numpy((expected_mask > 127).astype(np.float32)))
        )

    def test_carvana_resize_backend_must_be_explicit(self) -> None:
        config = {"protocol": {"segmentation": {"resize_backend": "released-opencv"}}}
        self.assertEqual(
            configured_carvana_resize_backend(config), "released-opencv"
        )
        config["protocol"]["segmentation"]["resize_backend"] = "unknown"
        with self.assertRaisesRegex(ValueError, "resize_backend"):
            configured_carvana_resize_backend(config)

    def test_released_batch_global_dice_emits_one_value_per_batch(self) -> None:
        config = {
            "seed": 17,
            "protocol": {
                "segmentation": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "metric_aggregation": "released_batch_global",
                }
            },
        }
        backend_info = {
            "name": "fixture",
            "realization_scope": "fixed_trial",
            "weight_bits": 6,
            "dac_bits": 6,
            "adc_bits": 6,
        }
        with tempfile.TemporaryDirectory(prefix="mtrd-static-ptq-") as temporary:
            checkpoint = Path(temporary) / "fixture.pth"
            checkpoint.write_bytes(b"fixture")
            rows = evaluate_segmentation_model(
                _PositiveSegmentationModel(),
                _SegmentationFixture(),
                backend="neurosim",
                backend_info=backend_info,
                device_model="rram",
                level=0.1,
                trial=0,
                trial_seed=19,
                config=config,
                method="nominal",
                checkpoint=checkpoint,
                checkpoint_hash="fixture",
            )

        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(row["metric"] == "Dice_batch_global_fraction" for row in rows)
        )
        self.assertTrue(all(row["metric_value"] == 1.0 for row in rows))
        trial_rows, summary_rows = aggregate_rows(rows)
        self.assertEqual(trial_rows[0]["observation_count"], 2)
        self.assertEqual(len(summary_rows), 1)


if __name__ == "__main__":
    unittest.main()
