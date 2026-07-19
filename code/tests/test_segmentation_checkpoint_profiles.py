from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from image_workflows import workflow as dncnn_unet
from image_workflows.workflow import (
    checkpoint_path,
    checkpoint_payload,
    mtrd_prerequisite_specs,
    segmentation_checkpoint_training_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def segmentation_config(
    *, optimizer: str = "released-source-adam",
    aggregation: str = "released_batch_global",
    batch_norm_policy: str = "legacy_dual_branch",
) -> dict[str, object]:
    return {
        "seed": 1,
        "protocol": {
            "eq4_alpha": 0.7,
            "distillation_temperature": 5.0,
            "eq6_direction": "equation_literal_current_minus_previous",
            "eq6_temperature": 1.0,
            "include_clean_teacher": True,
            "rram_levels": [0.1, 0.2, 0.3, 0.4, 0.5],
            "pcm_levels": [0.02, 0.04, 0.06, 0.08, 0.1],
            "rram_student_level": 0.3,
            "pcm_student_level": 0.06,
            "segmentation": {
                "teacher_epochs": 20,
                "student_epochs": 30,
                "batch_size": 16,
                "num_workers": 0,
                "image_height": 160,
                "image_width": 240,
                "resize_backend": "pil-bilinear",
                "optimizer_profile": optimizer,
                "metric_aggregation": aggregation,
                "mtrd_bn_update_policy": batch_norm_policy,
                "feedback_split": "carvana_training",
                "feedback_sample_count": 0,
                "checkpoint_selection": "final_epoch",
            }
        }
    }


def write_segmentation_checkpoint(
    path: Path, config: dict[str, object], *, role: str, device_model: str,
    level: float | None = None, epoch: int | None = None,
    payload_updates: dict[str, object] | None = None,
) -> None:
    expected_level = 0.0 if device_model == "none" else (0.3 if level is None else level)
    expected_epoch = 30 if role == "mtrd_student" else 20
    payload = checkpoint_payload(
        nn.Linear(2, 1),
        task="segmentation",
        role=role,
        device_model=device_model,
        level=expected_level,
        epoch=expected_epoch if epoch is None else epoch,
        metric=0.5,
        config=config,
    )
    if payload_updates:
        payload.update(payload_updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class SegmentationCheckpointProfileTest(unittest.TestCase):
    def test_enveloped_mtrd_checkpoint_requires_exact_training_contract(self) -> None:
        config = segmentation_config()
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-contract-") as temporary:
            path = Path(temporary) / "mtrd.pth"
            write_segmentation_checkpoint(
                path, config, role="mtrd_student", device_model="rram",
            )
            identity = segmentation_checkpoint_training_identity(
                path, config, device_model="rram", role="mtrd",
            )

        self.assertEqual(identity["verification"], "exact_resolved_contract")
        self.assertEqual(identity["expected_checkpoint_role"], "mtrd_student")
        self.assertIsNotNone(identity["embedded_batch_norm_policy_sha256"])

    def test_mtrd_checkpoint_rejects_optimizer_metric_and_batch_norm_mismatches(self) -> None:
        config = segmentation_config()
        variants = {
            "optimizer_contract": segmentation_config(optimizer="manuscript-stated-sgd"),
            "segmentation_metric_contract": segmentation_config(aggregation="per_image_mean"),
            "segmentation_mtrd_bn_update_policy": segmentation_config(
                batch_norm_policy="noisy_task_only"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-contract-mismatch-") as temporary:
            path = Path(temporary) / "mtrd.pth"
            write_segmentation_checkpoint(
                path, config, role="mtrd_student", device_model="rram",
            )
            for expected_field, candidate in variants.items():
                with self.subTest(field=expected_field):
                    with self.assertRaisesRegex(ValueError, expected_field):
                        segmentation_checkpoint_training_identity(
                            path, candidate, device_model="rram", role="mtrd",
                        )

    def test_legacy_checkpoint_is_explicitly_unverified(self) -> None:
        config = segmentation_config()
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-legacy-") as temporary:
            path = Path(temporary) / "legacy.pth"
            torch.save(nn.Linear(2, 1).state_dict(), path)
            identity = segmentation_checkpoint_training_identity(
                path, config, device_model="rram", role="mtrd",
            )

        self.assertEqual(identity["verification"], "unverified_legacy_checkpoint")
        self.assertIn("does not embed", str(identity["reason"]))

    def test_teacher_and_nominal_roles_are_checked_before_mtrd_training(self) -> None:
        config = segmentation_config()
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-teacher-") as temporary:
            root = Path(temporary)
            teacher_path = root / "teacher.pth"
            clean_path = root / "clean.pth"
            write_segmentation_checkpoint(
                teacher_path, config, role="variation_teacher", device_model="rram",
            )
            write_segmentation_checkpoint(
                clean_path, config, role="clean_teacher", device_model="none",
            )
            teacher = segmentation_checkpoint_training_identity(
                teacher_path, config, device_model="rram", role="teacher",
                expected_level=0.3,
            )
            nominal = segmentation_checkpoint_training_identity(
                clean_path, config, device_model="rram", role="nominal",
            )

        self.assertEqual(teacher["verification"], "exact_resolved_contract")
        self.assertEqual(nominal["verification"], "exact_resolved_contract")

    def test_mtrd_preflight_records_and_rejects_prerequisite_contracts(self) -> None:
        template = ROOT / "configs" / "segmentation.template.json"
        config = json.loads(template.read_text(encoding="utf-8"))
        config["protocol"]["segmentation"]["feedback_split"] = "carvana_training"
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-preflight-") as temporary:
            config["checkpoint_root"] = str(Path(temporary) / "checkpoints")
            for spec in mtrd_prerequisite_specs(config, "segmentation", "rram"):
                path = checkpoint_path(
                    config,
                    "segmentation",
                    "clean" if spec.clean else "teacher",
                    None if spec.clean else "rram",
                    None if spec.clean else spec.level,
                )
                write_segmentation_checkpoint(
                    path,
                    config,
                    role="clean_teacher" if spec.clean else "variation_teacher",
                    device_model="none" if spec.clean else "rram",
                    level=spec.level,
                )

            with mock.patch.object(
                dncnn_unet, "build_carvana_index", side_effect=FileNotFoundError("fixture"),
            ), mock.patch.object(dncnn_unet, "load_model_strict", return_value=nn.Identity()):
                report = dncnn_unet.run_preflight(
                    config,
                    scope="train",
                    tasks=("segmentation",),
                    device_models=("rram",),
                    stage="mtrd",
                    write_report=False,
                )
            contracts = report["mtrd_training_prerequisite_contracts"]
            self.assertEqual(len(contracts), 6)
            self.assertTrue(
                all(item["verification"] == "exact_resolved_contract" for item in contracts.values())
            )

            mismatched = copy.deepcopy(config)
            mismatched["protocol"]["segmentation"]["metric_aggregation"] = "released_batch_global"
            initial = checkpoint_path(
                config, "segmentation", "teacher", "rram", 0.3,
            )
            write_segmentation_checkpoint(
                initial, mismatched, role="variation_teacher", device_model="rram",
            )
            with mock.patch.object(
                dncnn_unet, "build_carvana_index", side_effect=FileNotFoundError("fixture"),
            ), mock.patch.object(dncnn_unet, "load_model_strict", return_value=nn.Identity()):
                rejected = dncnn_unet.run_preflight(
                    config,
                    scope="train",
                    tasks=("segmentation",),
                    device_models=("rram",),
                    stage="mtrd",
                    write_report=False,
                )

        self.assertTrue(
            any("segmentation_metric_contract" in error for error in rejected["errors"])
        )

    def test_mtrd_contract_rejects_changed_objective_and_feedback_semantics(self) -> None:
        config = segmentation_config()
        variants = {
            "eq4": {"eq4_alpha": 0.2},
            "temperature": {"distillation_temperature": 2.0},
            "eq6": {
                "eq6_direction": "narrative_underperformance_previous_minus_current",
                "eq6_temperature": 0.5,
            },
            "feedback": {"feedback_sample_count": 3},
            "resize": {"resize_backend": "released-opencv"},
        }
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-objective-") as temporary:
            path = Path(temporary) / "mtrd.pth"
            write_segmentation_checkpoint(
                path, config, role="mtrd_student", device_model="rram",
            )
            for name, updates in variants.items():
                candidate = copy.deepcopy(config)
                for key, value in updates.items():
                    target = (
                        candidate["protocol"]["segmentation"]
                        if key in {"feedback_sample_count", "resize_backend"}
                        else candidate["protocol"]
                    )
                    target[key] = value
                with self.subTest(semantic=name):
                    with self.assertRaisesRegex(ValueError, "segmentation_mtrd_contract"):
                        segmentation_checkpoint_training_identity(
                            path, candidate, device_model="rram", role="mtrd",
                        )

    def test_outer_identity_rejects_role_device_level_and_partial_epoch(self) -> None:
        config = segmentation_config()
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-outer-") as temporary:
            root = Path(temporary)
            teacher = root / "teacher.pth"
            write_segmentation_checkpoint(
                teacher, config, role="variation_teacher", device_model="rram", level=0.2,
            )
            with self.assertRaisesRegex(ValueError, "noise-level mismatch"):
                segmentation_checkpoint_training_identity(
                    teacher, config, device_model="rram", role="teacher", expected_level=0.1,
                )
            with self.assertRaisesRegex(ValueError, "device-model mismatch"):
                segmentation_checkpoint_training_identity(
                    teacher, config, device_model="pcm", role="teacher", expected_level=0.02,
                )

            partial = root / "partial.pth"
            write_segmentation_checkpoint(
                partial, config, role="variation_teacher", device_model="rram", epoch=1,
            )
            with self.assertRaisesRegex(ValueError, "final-epoch mismatch"):
                segmentation_checkpoint_training_identity(
                    partial, config, device_model="rram", role="teacher", expected_level=0.3,
                )

            student = root / "student.pth"
            write_segmentation_checkpoint(
                student, config, role="mtrd_student", device_model="rram",
            )
            with self.assertRaisesRegex(ValueError, "role mismatch"):
                segmentation_checkpoint_training_identity(
                    student, config, device_model="rram", role="teacher", expected_level=0.3,
                )

    def test_default_and_explicit_metric_and_batch_norm_contracts_are_semantically_equal(self) -> None:
        explicit = segmentation_config(aggregation="per_image_mean")
        legacy_default = segmentation_config(aggregation="per_image_mean")
        del legacy_default["protocol"]["segmentation"]["metric_aggregation"]
        del legacy_default["protocol"]["segmentation"]["mtrd_bn_update_policy"]
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-semantic-default-") as temporary:
            path = Path(temporary) / "mtrd.pth"
            write_segmentation_checkpoint(
                path, legacy_default, role="mtrd_student", device_model="rram",
            )
            identity = segmentation_checkpoint_training_identity(
                path, explicit, device_model="rram", role="mtrd",
            )

        self.assertEqual(identity["verification"], "exact_resolved_contract")

    def test_old_mtrd_envelope_is_unverified_instead_of_exact(self) -> None:
        config = segmentation_config()
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-old-envelope-") as temporary:
            path = Path(temporary) / "mtrd.pth"
            write_segmentation_checkpoint(
                path,
                config,
                role="mtrd_student",
                device_model="rram",
                payload_updates={"segmentation_mtrd_contract": None},
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["segmentation_mtrd_contract"]
            torch.save(payload, path)
            identity = segmentation_checkpoint_training_identity(
                path, config, device_model="rram", role="mtrd",
            )

        self.assertEqual(identity["verification"], "unverified_legacy_checkpoint")

    def test_mtrd_preflight_rejects_unverified_legacy_prerequisites(self) -> None:
        template = ROOT / "configs" / "segmentation.template.json"
        config = json.loads(template.read_text(encoding="utf-8"))
        config["protocol"]["segmentation"]["feedback_split"] = "carvana_training"
        with tempfile.TemporaryDirectory(prefix="mtrd-unet-legacy-prerequisite-") as temporary:
            config["checkpoint_root"] = str(Path(temporary) / "checkpoints")
            specs = mtrd_prerequisite_specs(config, "segmentation", "rram")
            for spec in specs:
                path = checkpoint_path(
                    config,
                    "segmentation",
                    "clean" if spec.clean else "teacher",
                    None if spec.clean else "rram",
                    None if spec.clean else spec.level,
                )
                if spec.clean:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(nn.Linear(2, 1).state_dict(), path)
                    continue
                write_segmentation_checkpoint(
                    path,
                    config,
                    role="variation_teacher",
                    device_model="rram",
                    level=spec.level,
                )
            with mock.patch.object(
                dncnn_unet, "build_carvana_index", side_effect=FileNotFoundError("fixture"),
            ), mock.patch.object(dncnn_unet, "load_model_strict", return_value=nn.Identity()):
                report = dncnn_unet.run_preflight(
                    config,
                    scope="train",
                    tasks=("segmentation",),
                    device_models=("rram",),
                    stage="mtrd",
                    write_report=False,
                )

        self.assertTrue(any("unverified legacy checkpoint" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
