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
    ReleasedBH4RankBalancer,
    build_training_plan,
    checkpoint_payload,
    checkpoint_path,
    dncnn_checkpoint_profile_identity,
    dncnn_mtrd_contract_payload,
    learning_rate,
    mtrd_prerequisite_specs,
    mtrd_teacher_specs,
    released_bh4_source_loss,
    resolve_dncnn_mtrd_contract,
    run_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def load_template(name: str) -> dict[str, object]:
    with (ROOT / "configs" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_dncnn_checkpoint(
    path: Path, config: dict[str, object], contract, *, role: str,
    device_model: str, level: float, epoch: int,
) -> None:
    payload = checkpoint_payload(
        nn.Linear(2, 1),
        task="denoising",
        role=role,
        device_model=device_model,
        level=level,
        epoch=epoch,
        metric=1.0,
        config=config,
        extra={"dncnn_mtrd_contract": dncnn_mtrd_contract_payload(contract)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class DnCNNProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_config = load_template("denosing_released_ptq.template.json")
        self.paper_config = load_template("denosing.template.json")
        self.paper_config["protocol"]["denoising"]["feedback_split"] = "set12_test"

    def test_profiles_have_isolated_teacher_pools_and_documented_invocations(self) -> None:
        source_rram = resolve_dncnn_mtrd_contract(self.source_config, "rram")
        source_pcm = resolve_dncnn_mtrd_contract(self.source_config, "pcm")
        paper_rram = resolve_dncnn_mtrd_contract(self.paper_config, "rram")

        self.assertEqual(source_rram.name, "released_bh4_source_semantics")
        self.assertEqual(
            [spec.level for spec in source_rram.mtrd_teacher_specs],
            [0.1, 0.2, 0.4, 0.5],
        )
        self.assertFalse(any(spec.clean for spec in source_rram.mtrd_teacher_specs))
        self.assertIn("--w_noiseL 0.3", source_rram.source_invocation or "")
        self.assertIn("--w_noiseL 0.06", source_pcm.source_invocation or "")
        self.assertIn("defaults --w_noiseL to 0.0", source_rram.source_default_ambiguity or "")
        self.assertIn("intermediate_variation", source_rram.student_training_forward)
        self.assertIn("denosing/README.md", source_rram.source_files)
        self.assertIn("denosing/test_quant.py", source_rram.source_files)

        self.assertEqual(len(paper_rram.mtrd_teacher_specs), 6)
        self.assertTrue(paper_rram.mtrd_teacher_specs[0].clean)
        self.assertEqual(paper_rram.source_invocation, None)
        self.assertFalse(paper_rram.kd_is_degenerate)

    def test_source_preflight_pool_includes_all_preparation_teachers_but_no_clean(self) -> None:
        pool = mtrd_teacher_specs(self.source_config, "denoising", "rram")
        prerequisites = mtrd_prerequisite_specs(
            self.source_config, "denoising", "rram"
        )
        self.assertEqual(len(pool), 4)
        self.assertEqual([spec.level for spec in prerequisites], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertFalse(any(spec.clean for spec in prerequisites))

    def test_source_rank_balancer_uses_zero_baseline_and_four_raw_weights(self) -> None:
        balancer = ReleasedBH4RankBalancer()
        self.assertTrue(torch.allclose(balancer.previous, torch.zeros(4, dtype=torch.float64)))
        self.assertTrue(torch.allclose(balancer.beta, torch.full((4,), 0.25, dtype=torch.float64)))

        beta = balancer.update([11.0, 10.0, 12.0, 9.0])
        expected = torch.tensor([1.5, 2.0, 1.0, 2.5], dtype=torch.float64) / 7.0
        self.assertTrue(torch.allclose(beta, expected))
        self.assertTrue(torch.allclose(
            balancer.previous, torch.tensor([11.0, 10.0, 12.0, 9.0], dtype=torch.float64)
        ))

    def test_source_loss_preserves_degenerate_single_channel_kl_and_sum_mse(self) -> None:
        student = torch.tensor([[[[2.0]]]], requires_grad=True)
        target = torch.zeros_like(student)
        teachers = [torch.full_like(student, value) for value in (1.0, 2.0, 3.0, 4.0)]
        loss, parts = released_bh4_source_loss(
            student,
            target,
            teachers,
            torch.full((4,), 0.25),
        )

        self.assertAlmostEqual(loss.item(), 0.6, places=7)
        self.assertTrue(torch.allclose(parts["kd"], torch.zeros_like(parts["kd"])))
        self.assertTrue(parts["kd_is_degenerate"])
        self.assertEqual(parts["loss_reduction"], "kl_mean_plus_mse_sum_divided_by_2_batch")
        loss.backward()
        self.assertAlmostEqual(student.grad.item(), 0.6, places=7)

    def test_profiles_choose_their_own_student_learning_rate_schedule(self) -> None:
        self.assertAlmostEqual(
            learning_rate(
                self.source_config, "denoising", 50, student=True, device_model="rram"
            ),
            1e-4,
        )
        self.assertAlmostEqual(
            learning_rate(
                self.paper_config, "denoising", 50, student=True, device_model="rram"
            ),
            1e-5,
        )

    def test_source_plan_declares_nominal_reference_and_resolved_contract(self) -> None:
        plan = build_training_plan(
            self.source_config, ("denoising",), ("rram",), "all"
        )
        clean = next(item for item in plan if item["action"] == "train_clean_nominal_reference")
        student = next(item for item in plan if item["action"] == "train_mtrd_student")
        self.assertEqual(clean["dncnn_mtrd_contract"]["name"], "released_bh4_source_semantics")
        self.assertEqual(student["dncnn_mtrd_contract"]["student_lr_milestones"], [30])

    def test_source_profile_rejects_a_paper_interpreted_student_schedule(self) -> None:
        invalid = copy.deepcopy(self.source_config)
        invalid["protocol"]["denoising"]["student_lr_milestones"] = [30, 50]
        with self.assertRaisesRegex(ValueError, "BH4 DnCNN student_lr_milestones"):
            resolve_dncnn_mtrd_contract(invalid, "rram")

        invalid = copy.deepcopy(self.paper_config)
        invalid["protocol"]["denoising"]["mtrd_profile_parameters"]["extra"] = 1
        with self.assertRaisesRegex(ValueError, "does not accept unknown"):
            resolve_dncnn_mtrd_contract(invalid, "rram")

    def test_checkpoint_profile_identity_allows_shared_clean_but_requires_exact_mtrd(self) -> None:
        source_rram = resolve_dncnn_mtrd_contract(self.source_config, "rram")
        source_pcm = resolve_dncnn_mtrd_contract(self.source_config, "pcm")
        paper_rram = resolve_dncnn_mtrd_contract(self.paper_config, "rram")
        with tempfile.TemporaryDirectory(prefix="mtrd-dncnn-profile-") as directory:
            root = Path(directory)
            clean = root / "clean.pt"
            teacher = root / "teacher.pt"
            student = root / "student.pt"
            write_dncnn_checkpoint(
                clean,
                self.source_config,
                source_rram,
                role="clean_teacher",
                device_model="none",
                level=0.0,
                epoch=source_rram.teacher_epochs,
            )
            write_dncnn_checkpoint(
                teacher,
                self.source_config,
                source_rram,
                role="variation_teacher",
                device_model="rram",
                level=0.1,
                epoch=source_rram.teacher_epochs,
            )
            write_dncnn_checkpoint(
                student,
                self.source_config,
                source_rram,
                role="mtrd_student",
                device_model="rram",
                level=source_rram.student_initial_level,
                epoch=source_rram.student_epochs,
            )
            self.assertEqual(
                dncnn_checkpoint_profile_identity(clean, source_pcm, role="nominal")["verification"],
                "clean_training_contract",
            )
            self.assertEqual(
                dncnn_checkpoint_profile_identity(
                    teacher, source_rram, role="teacher", expected_level=0.1,
                )["verification"],
                "exact_resolved_contract",
            )
            self.assertEqual(
                dncnn_checkpoint_profile_identity(student, source_rram, role="mtrd")["verification"],
                "exact_resolved_contract",
            )
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                dncnn_checkpoint_profile_identity(student, paper_rram, role="mtrd")
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                dncnn_checkpoint_profile_identity(
                    teacher, paper_rram, role="teacher", expected_level=0.1,
                )

    def test_checkpoint_profile_identity_rejects_outer_role_level_and_partial_epoch(self) -> None:
        source = resolve_dncnn_mtrd_contract(self.source_config, "rram")
        with tempfile.TemporaryDirectory(prefix="mtrd-dncnn-outer-") as directory:
            root = Path(directory)
            teacher = root / "teacher.pt"
            write_dncnn_checkpoint(
                teacher,
                self.source_config,
                source,
                role="variation_teacher",
                device_model="rram",
                level=0.2,
                epoch=source.teacher_epochs,
            )
            with self.assertRaisesRegex(ValueError, "noise-level mismatch"):
                dncnn_checkpoint_profile_identity(
                    teacher, source, role="teacher", expected_level=0.1,
                )
            with self.assertRaisesRegex(ValueError, "role mismatch"):
                dncnn_checkpoint_profile_identity(teacher, source, role="mtrd")

            partial = root / "partial.pt"
            write_dncnn_checkpoint(
                partial,
                self.source_config,
                source,
                role="variation_teacher",
                device_model="rram",
                level=0.3,
                epoch=1,
            )
            with self.assertRaisesRegex(ValueError, "final-epoch mismatch"):
                dncnn_checkpoint_profile_identity(
                    partial, source, role="teacher", expected_level=0.3,
                )

    def test_mtrd_preflight_validates_teacher_and_initial_checkpoint_profiles(self) -> None:
        config = copy.deepcopy(self.source_config)
        source = resolve_dncnn_mtrd_contract(config, "rram")
        paper = resolve_dncnn_mtrd_contract(self.paper_config, "rram")
        with tempfile.TemporaryDirectory(prefix="mtrd-dncnn-preflight-") as directory:
            config["checkpoint_root"] = str(Path(directory) / "checkpoints")
            for spec in mtrd_prerequisite_specs(config, "denoising", "rram"):
                path = checkpoint_path(
                    config, "denoising", "teacher", "rram", spec.level,
                )
                write_dncnn_checkpoint(
                    path,
                    config,
                    source,
                    role="variation_teacher",
                    device_model="rram",
                    level=spec.level,
                    epoch=source.teacher_epochs,
                )

            with mock.patch.object(dncnn_unet, "load_model_strict", return_value=nn.Identity()):
                report = run_preflight(
                    config,
                    scope="train",
                    tasks=("denoising",),
                    device_models=("rram",),
                    stage="mtrd",
                    write_report=False,
                )
            contracts = report["mtrd_training_prerequisite_contracts"]
            self.assertEqual(len(contracts), 5)
            self.assertTrue(
                all(item["verification"] == "exact_resolved_contract" for item in contracts.values())
            )

            initial = checkpoint_path(config, "denoising", "teacher", "rram", 0.3)
            write_dncnn_checkpoint(
                initial,
                config,
                paper,
                role="variation_teacher",
                device_model="rram",
                level=0.3,
                epoch=paper.teacher_epochs,
            )
            with mock.patch.object(dncnn_unet, "load_model_strict", return_value=nn.Identity()):
                rejected = run_preflight(
                    config,
                    scope="train",
                    tasks=("denoising",),
                    device_models=("rram",),
                    stage="mtrd",
                    write_report=False,
                )

        self.assertTrue(any("profile mismatch" in error for error in rejected["errors"]))

    def test_mtrd_preflight_requires_a_strictly_loadable_prerequisite(self) -> None:
        config = copy.deepcopy(self.source_config)
        source = resolve_dncnn_mtrd_contract(config, "rram")
        with tempfile.TemporaryDirectory(prefix="mtrd-dncnn-loadable-") as directory:
            config["checkpoint_root"] = str(Path(directory) / "checkpoints")
            for spec in mtrd_prerequisite_specs(config, "denoising", "rram"):
                path = checkpoint_path(config, "denoising", "teacher", "rram", spec.level)
                write_dncnn_checkpoint(
                    path,
                    config,
                    source,
                    role="variation_teacher",
                    device_model="rram",
                    level=spec.level,
                    epoch=source.teacher_epochs,
                )
            invalid = checkpoint_path(config, "denoising", "teacher", "rram", 0.3)
            payload = torch.load(invalid, map_location="cpu", weights_only=False)
            payload["state_dict"] = {"not_a_dncnn_weight": torch.ones(1)}
            torch.save(payload, invalid)
            report = run_preflight(
                config,
                scope="train",
                tasks=("denoising",),
                device_models=("rram",),
                stage="mtrd",
                write_report=False,
            )

        self.assertTrue(any("strict model-state load" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
