"""Tests for public external-asset and portable-checkpoint interfaces."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from capsule import assets
from classification import inference
from classification.run import parse_args as parse_classification_args


class _TinyClassifier(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 32 * 32, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(1))


def _load_tiny_checkpoint(model: nn.Module, path: str, _location: object) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old PyTorch compatibility.
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload, strict=True)


class PublicInterfaceTest(unittest.TestCase):
    def test_classification_cli_exposes_portable_checkpoint_test(self) -> None:
        args = parse_classification_args(
            [
                "test-checkpoint",
                "--dataset", "cifar100",
                "--checkpoint", "/data/checkpoints/vgg16.pt",
            ]
        )
        self.assertEqual(args.command, "test-checkpoint")
        self.assertEqual(args.dataset, "cifar100")
        self.assertFalse(args.require_checkpoint_manifest)

    def test_portable_checkpoint_test_exports_complete_prediction_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-portable-checkpoint-") as temporary:
            root = Path(temporary)
            checkpoint = root / "portable.pt"
            original = _TinyClassifier(10)
            with torch.no_grad():
                original.linear.weight.zero_()
                original.linear.bias.zero_()
                original.linear.bias[0] = 1.0
            torch.save(original.state_dict(), checkpoint)

            inputs = torch.zeros(3, 3, 32, 32)
            targets = torch.tensor([0, 1, 0])
            loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
            arguments = Namespace(
                checkpoint=str(checkpoint),
                dataset="cifar10",
                normalization_profile="dataset-native",
                require_checkpoint_manifest=True,
                output_dir=str(root / "result"),
                overwrite=False,
                device="cpu",
                seed=7,
                data_root=str(root / "cifar"),
                batch_size=2,
                workers=0,
                download=False,
            )
            normalization = {"verified": True, "manifest_present": True}
            identity = {"dataset": "cifar10", "present": True}
            with (
                patch.object(inference, "PaperVGG16", _TinyClassifier),
                patch.object(inference, "load_checkpoint_strict", _load_tiny_checkpoint),
                patch.object(inference, "build_cifar_loaders", return_value=(None, loader, None)),
                patch.object(inference, "checkpoint_normalization_identity", return_value=normalization),
                patch.object(inference, "cifar_identity", return_value=identity),
                patch.object(inference, "environment_identity", return_value={"fixture": True}),
                patch.object(inference, "git_identity", return_value={"fixture": True}),
            ):
                predictions, manifest = inference.test_checkpoint(arguments)

            with predictions.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["prediction"] for row in rows], ["0", "0", "0"])
            self.assertEqual(payload["result"]["correct"], 2)
            self.assertEqual(payload["result"]["total"], 3)
            self.assertTrue(payload["checkpoint"]["strict_load_valid"])
            self.assertEqual(
                payload["execution_engine"], "PyTorch clean digital inference; no CIM simulator"
            )

    def test_portable_checkpoint_test_can_require_provenance_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-portable-provenance-") as temporary:
            checkpoint = Path(temporary) / "portable.pt"
            torch.save(_TinyClassifier(10).state_dict(), checkpoint)
            arguments = Namespace(
                checkpoint=str(checkpoint),
                dataset="cifar10",
                normalization_profile="dataset-native",
                require_checkpoint_manifest=True,
                output_dir=str(Path(temporary) / "result"),
                overwrite=False,
                device="cpu",
                seed=7,
                data_root=str(Path(temporary) / "cifar"),
                batch_size=2,
                workers=0,
                download=False,
            )
            with patch.object(
                inference,
                "checkpoint_normalization_identity",
                return_value={"verified": False, "manifest_present": False},
            ):
                with self.assertRaisesRegex(RuntimeError, "require-checkpoint-manifest"):
                    inference.test_checkpoint(arguments)

    def test_cifar_asset_validator_reports_verified_layout(self) -> None:
        fixture = {"dataset": "cifar100", "present": True, "sha256_tree": "a" * 64}
        with patch.object(assets, "cifar_identity", return_value=fixture):
            report = assets.validate_cifar("cifar100", "/data/cifar", hash_content=True)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["ready_for_training_or_clean_test"])
        self.assertEqual(report["identity"], fixture)

    def test_denoising_asset_validator_rejects_duplicate_train_test_content(self) -> None:
        fixture = {
            "training_count": 400,
            "set12_count": 12,
            "train_test_duplicate_content_hashes": ["b" * 64],
        }
        with patch.object(assets, "denoising_asset_identity", return_value=fixture):
            report = assets.validate_denoising(
                "/data/berkeley400", "/data/berkeley400/Set12", hash_content=True
            )
        self.assertEqual(report["status"], "fail")
        self.assertIn("overlaps", report["errors"][0])

    def test_carvana_asset_validator_distinguishes_raw_and_paper_ready_states(self) -> None:
        index = {f"car_{item}": object() for item in range(5088)}
        with (
            patch.object(assets, "build_carvana_index", return_value=index),
            patch.object(
                assets,
                "carvana_asset_identity",
                return_value={"sample_count": 5088, "paired_content_manifest_sha256": "c" * 64},
            ),
        ):
            report = assets.validate_carvana(
                ["/data/carvana/images"],
                ["/data/carvana/masks"],
                split_manifest=None,
                allow_derived_split=False,
                require_paper_split=False,
                hash_content=True,
            )
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["ready_for_raw_asset_upload"])
        self.assertFalse(report["ready_for_author_verified_paper_run"])


if __name__ == "__main__":
    unittest.main()
