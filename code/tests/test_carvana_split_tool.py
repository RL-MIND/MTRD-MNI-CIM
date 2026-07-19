from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from image_workflows import workflow as dncnn_unet
from tools import build_carvana_split, build_released_carvana_split


class CarvanaSplitToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = {
            f"car{car_index:03d}_{view_index:02d}": object()
            for car_index in range(318)
            for view_index in range(1, 17)
        }

    def _build(self, *, seed: int = 17, policy: str = "fixed-view"):
        with mock.patch.object(
            build_carvana_split, "build_carvana_index", return_value=self.index,
        ), mock.patch.object(
            build_carvana_split,
            "carvana_asset_identity",
            return_value={"sample_count": 5088},
        ):
            return build_carvana_split.build_split(
                image_dirs=["images"], mask_dirs=["masks"], seed=seed,
                test_policy=policy, test_view=3,
            )

    def test_fixed_view_builds_the_declared_partition(self) -> None:
        split = self._build()

        self.assertEqual(len(split["train_ids"]), 4700)
        self.assertEqual(len(split["test_ids"]), 318)
        self.assertEqual(len(split["excluded_ids"]), 70)
        self.assertTrue(all(value.endswith("_03") for value in split["test_ids"]))
        self.assertFalse(split["author_verified"])
        self.assertFalse(split["derivation"]["paper_claim"])
        self.assertEqual(
            set(split["train_ids"])
            | set(split["test_ids"])
            | set(split["excluded_ids"]),
            set(self.index),
        )

    def test_seeded_policy_is_replayable(self) -> None:
        first = self._build(seed=23, policy="seeded-random")
        replay = self._build(seed=23, policy="seeded-random")
        changed = self._build(seed=24, policy="seeded-random")

        self.assertEqual(first, replay)
        self.assertNotEqual(first["test_ids"], changed["test_ids"])
        self.assertEqual(
            len({value.rsplit("_", 1)[0] for value in first["test_ids"]}),
            318,
        )

    def test_released_directory_contract_preserves_4588_500_assignment(self) -> None:
        train = dict(list(self.index.items())[:4588])
        test = dict(list(self.index.items())[4588:])
        with mock.patch.object(
            build_released_carvana_split,
            "_index",
            side_effect=[train, test],
        ), mock.patch.object(
            build_released_carvana_split,
            "carvana_asset_identity",
            side_effect=lambda value, include_hash=False: {
                "sample_count": len(value)
            },
        ):
            split = build_released_carvana_split.build_released_split(
                train_image_dir="train-images",
                train_mask_dir="train-masks",
                test_image_dir="test-images",
                test_mask_dir="test-masks",
            )

        self.assertEqual(split["split_contract"], "released-directory-4588-500")
        self.assertEqual(len(split["train_ids"]), 4588)
        self.assertEqual(len(split["test_ids"]), 500)
        self.assertEqual(split["excluded_ids"], [])
        self.assertFalse(split["derivation"]["paper_claim"])

    def test_segmentation_preflight_does_not_validate_dncnn_protocol(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation.template.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        config["protocol"]["segmentation"]["feedback_split"] = "carvana_training"
        self.assertNotIn("denoising", config["protocol"])

        with mock.patch.object(
            dncnn_unet, "build_carvana_index", side_effect=FileNotFoundError("fixture"),
        ):
            report = dncnn_unet.run_preflight(
                config,
                scope="train",
                tasks=("segmentation",),
                device_models=("rram",),
                stage="clean",
                write_report=False,
            )

        protocol_check = next(
            item for item in report["checks"] if item["name"] == "image-task protocol constants"
        )
        self.assertEqual(protocol_check["status"], "pass")
        self.assertFalse(any("DnCNN" in value for value in report["errors"]))
        self.assertEqual(
            report["resolved_protocol"]["segmentation_optimizer"]["profile"],
            "manuscript-stated-sgd",
        )
        self.assertEqual(
            report["resolved_protocol"]["segmentation_metric_contract"][
                "metric_aggregation"
            ],
            "per_image_mean",
        )
        self.assertEqual(
            report["resolved_protocol"]["segmentation_mtrd_bn_update_policy"]["policy"],
            "legacy_dual_branch",
        )

    def test_segmentation_preflight_records_released_adam_profile(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation.template.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        config["protocol"]["segmentation"]["optimizer_profile"] = "released-source-adam"
        config["protocol"]["segmentation"]["feedback_split"] = "carvana_training"

        with mock.patch.object(
            dncnn_unet, "build_carvana_index", side_effect=FileNotFoundError("fixture"),
        ):
            report = dncnn_unet.run_preflight(
                config,
                scope="train",
                tasks=("segmentation",),
                device_models=("rram",),
                stage="clean",
                write_report=False,
            )

        protocol_check = next(
            item for item in report["checks"] if item["name"] == "image-task protocol constants"
        )
        self.assertEqual(protocol_check["status"], "pass")
        self.assertEqual(
            report["resolved_protocol"]["segmentation_optimizer"]["profile"],
            "released-source-adam",
        )
        self.assertTrue(
            any("optimizer-only reconstruction" in value for value in report["warnings"])
        )

    def test_segmentation_preflight_rejects_unknown_optimizer_profile(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation.template.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        config["protocol"]["segmentation"]["optimizer_profile"] = "automatic"
        config["protocol"]["segmentation"]["feedback_split"] = "carvana_training"

        with mock.patch.object(
            dncnn_unet, "build_carvana_index", side_effect=FileNotFoundError("fixture"),
        ):
            report = dncnn_unet.run_preflight(
                config,
                scope="train",
                tasks=("segmentation",),
                device_models=("rram",),
                stage="clean",
                write_report=False,
            )

        protocol_check = next(
            item for item in report["checks"] if item["name"] == "image-task protocol constants"
        )
        self.assertEqual(protocol_check["status"], "fail")
        self.assertTrue(any("optimizer_profile" in value for value in report["errors"]))

    def test_segmentation_preflight_rejects_unknown_batch_norm_policy(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "segmentation.template.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        config["protocol"]["segmentation"]["feedback_split"] = "carvana_training"
        config["protocol"]["segmentation"]["mtrd_bn_update_policy"] = "automatic"

        with mock.patch.object(
            dncnn_unet, "build_carvana_index", side_effect=FileNotFoundError("fixture"),
        ):
            report = dncnn_unet.run_preflight(
                config,
                scope="train",
                tasks=("segmentation",),
                device_models=("rram",),
                stage="clean",
                write_report=False,
            )

        protocol_check = next(
            item for item in report["checks"] if item["name"] == "image-task protocol constants"
        )
        self.assertEqual(protocol_check["status"], "fail")
        self.assertTrue(any("mtrd_bn_update_policy" in value for value in report["errors"]))


if __name__ == "__main__":
    unittest.main()
