from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from classification.pytorch import (
    TrainingJob,
    build_parser,
    cifar_integrity_checks,
    sha256_file,
    training_job_identity,
    update_balancing_weights,
    validate_completed_job,
)
from image_workflows.workflow import (
    EpochDeltaBalancer as ImageEpochDeltaBalancer,
    Set12Dataset,
    _denoising_png_files,
    build_training_plan,
    build_task_datasets,
    checkpoint_payload,
    build_parser as build_image_parser,
    optimizer_for as image_optimizer_for,
    paper_mtrd_loss as image_mtrd_loss,
    resolve_evaluation_trial_selection,
    resolve_unet_optimizer_profile,
    run_noise_scale_validation,
    resolve_realization_scope,
    resolve_execution_device,
    training_manifest as image_training_manifest,
    unet_optimizer_contract,
)
from classification.evaluation import evaluate as evaluate_classification
from classification.model import PaperVGG16, load_checkpoint_strict
from classification.repro import (
    checkpoint_normalization_identity,
    cifar_identity,
    cifar_normalization,
    sha256_file as classification_sha256_file,
)
from classification.run import (
    _write_evaluation_checkpoint_roles,
    parse_args as parse_classification_args,
)
from classification.aihwkit_backend import convert_model
from classification.protocol import EpochDeltaBalancer, paper_mtrd_loss
from utils.checkpoint_roles import validate_checkpoint_roles, write_checkpoint_roles
from utils.checkpoints import (
    remap_legacy_unet_state_dict,
    remap_legacy_vgg_state_dict,
)
from utils.source_tree import source_tree_sha256


class ProtocolTest(unittest.TestCase):
    @staticmethod
    def _unet_optimizer_config(profile: str) -> dict[str, object]:
        return {
            "checkpoint_root": "/tmp/mtrd-unet-checkpoints",
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
                    "image_height": 160,
                    "image_width": 240,
                    "resize_backend": "pil-bilinear",
                    "metric_aggregation": "per_image_mean",
                    "mtrd_bn_update_policy": "legacy_dual_branch",
                    "optimizer_profile": profile,
                    "feedback_split": "carvana_training",
                    "feedback_sample_count": 0,
                    "checkpoint_selection": "final_epoch",
                }
            },
        }

    def test_unet_optimizer_profiles_are_explicit_and_pinned(self) -> None:
        model = nn.Linear(2, 1)
        sgd_config = self._unet_optimizer_config("manuscript-stated-sgd")
        sgd_contract = resolve_unet_optimizer_profile(sgd_config)
        sgd = image_optimizer_for(sgd_config, "segmentation", model, student=False)

        self.assertEqual(sgd_contract["profile"], "manuscript-stated-sgd")
        self.assertEqual(sgd_contract["roles"], {"teacher": "SGD", "student": "SGD"})
        self.assertIsInstance(sgd, torch.optim.SGD)
        self.assertEqual(sgd.param_groups[0]["lr"], 1e-4)
        self.assertEqual(sgd.param_groups[0]["momentum"], 0.0)
        self.assertEqual(sgd.param_groups[0]["weight_decay"], 0.0)

        adam_config = self._unet_optimizer_config("released-source-adam")
        adam_contract = unet_optimizer_contract(adam_config, student=True)
        adam = image_optimizer_for(adam_config, "segmentation", model, student=True)

        self.assertEqual(adam_contract["profile"], "released-source-adam")
        self.assertEqual(adam_contract["optimization_role"], "student")
        self.assertEqual(adam_contract["role_optimizer"], "Adam")
        self.assertIsInstance(adam, torch.optim.Adam)
        self.assertEqual(adam.param_groups[0]["lr"], 1e-4)
        self.assertEqual(adam.param_groups[0]["betas"], (0.9, 0.999))
        self.assertEqual(adam.param_groups[0]["eps"], 1e-8)
        self.assertEqual(adam.param_groups[0]["weight_decay"], 0.0)
        self.assertFalse(adam.param_groups[0]["amsgrad"])

        dncnn_config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "denosing.template.json"
        )
        dncnn_config = json.loads(dncnn_config_path.read_text(encoding="utf-8"))
        dncnn_config["protocol"]["denoising"]["feedback_split"] = "set12_test"
        denoising = image_optimizer_for(
            dncnn_config,
            "denoising",
            model,
            student=False,
            device_model="rram",
        )
        self.assertIsInstance(denoising, torch.optim.Adam)
        self.assertEqual(denoising.param_groups[0]["lr"], 1e-3)

    def test_image_evaluation_device_is_explicit_and_fails_without_cuda(self) -> None:
        parser = build_image_parser()
        args = parser.parse_args(["--config", "fixture.json", "evaluate"])
        self.assertEqual(args.torch_device, "cpu")
        self.assertEqual(resolve_execution_device("cpu").type, "cpu")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA evaluation was requested"):
                resolve_execution_device("cuda")

    def test_image_selected_trial_is_explicit_and_not_an_aggregate(self) -> None:
        config = {
            "simulation": {
                "trial_count": 5,
                "trial_count_author_verified": True,
            }
        }
        resolved, indices, selection = resolve_evaluation_trial_selection(
            config, trials=None, trial_index=3,
        )
        self.assertEqual(indices, (3,))
        self.assertEqual(selection["policy"], "explicit-single-trial")
        self.assertTrue(selection["single_trial_exemplar"])
        self.assertFalse(resolved["simulation"]["trial_count_author_verified"])
        self.assertEqual(config["simulation"]["trial_count_author_verified"], True)

        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            resolve_evaluation_trial_selection(config, trials=2, trial_index=3)
        with self.assertRaisesRegex(ValueError, "zero or greater"):
            resolve_evaluation_trial_selection(config, trials=None, trial_index=-1)

    def test_unet_optimizer_profile_rejects_legacy_or_unknown_settings(self) -> None:
        missing = self._unet_optimizer_config("manuscript-stated-sgd")
        del missing["protocol"]["segmentation"]["optimizer_profile"]
        with self.assertRaisesRegex(ValueError, "optimizer_profile"):
            resolve_unet_optimizer_profile(missing)

        unknown = self._unet_optimizer_config("automatic")
        with self.assertRaisesRegex(ValueError, "optimizer_profile"):
            resolve_unet_optimizer_profile(unknown)

        legacy = self._unet_optimizer_config("released-source-adam")
        legacy["protocol"]["segmentation"]["weight_decay"] = 0.2
        with self.assertRaisesRegex(ValueError, "legacy keys"):
            resolve_unet_optimizer_profile(legacy)

    def test_unet_optimizer_contract_is_written_to_plan_checkpoint_and_manifest(self) -> None:
        config = self._unet_optimizer_config("released-source-adam")
        plan = build_training_plan(config, ("segmentation",), ("rram",), "all")

        self.assertEqual(len(plan), 7)
        self.assertEqual(plan[0]["optimizer_contract"]["optimization_role"], "teacher")
        self.assertEqual(plan[-1]["optimizer_contract"]["optimization_role"], "student")
        self.assertEqual(plan[-1]["optimizer_contract"]["profile"], "released-source-adam")
        self.assertEqual(
            plan[-1]["segmentation_metric_contract"]["metric_aggregation"],
            "per_image_mean",
        )
        self.assertEqual(
            plan[-1]["segmentation_mtrd_bn_update_policy"]["policy"],
            "legacy_dual_branch",
        )
        self.assertEqual(
            plan[-1]["segmentation_mtrd_contract"]["eq4_alpha"],
            0.7,
        )
        self.assertEqual(
            plan[-1]["segmentation_mtrd_contract"]["student_epochs"],
            30,
        )

        with tempfile.TemporaryDirectory(prefix="mtrd-unet-optimizer-manifest-") as temporary:
            root = Path(temporary)
            checkpoint = root / "student.pth"
            epoch_log = root / "student.epochs.csv"
            checkpoint.write_bytes(b"checkpoint fixture")
            epoch_log.write_text("epoch,metric\n1,0.5\n", encoding="ascii")
            payload = checkpoint_payload(
                nn.Linear(2, 1),
                task="segmentation",
                role="mtrd_student",
                device_model="rram",
                level=0.3,
                epoch=30,
                metric=0.5,
                config=config,
            )
            manifest = image_training_manifest(
                checkpoint,
                config=config,
                task="segmentation",
                role="mtrd_student",
                device_model="rram",
                level=0.3,
                metric=0.5,
                dataset={"fixture": True},
                log_path=epoch_log,
            )

        self.assertEqual(payload["optimizer_contract"]["profile"], "released-source-adam")
        self.assertEqual(payload["optimizer_contract"]["optimization_role"], "student")
        self.assertEqual(manifest["optimizer_contract"], payload["optimizer_contract"])
        self.assertEqual(
            payload["segmentation_metric_contract"]["observation_metric"],
            "Dice_fraction",
        )
        self.assertEqual(
            manifest["segmentation_mtrd_bn_update_policy"],
            payload["segmentation_mtrd_bn_update_policy"],
        )
        self.assertEqual(
            manifest["segmentation_mtrd_contract"],
            payload["segmentation_mtrd_contract"],
        )

    def test_released_image_rank_balancer_prioritizes_smallest_delta(self) -> None:
        balancer = ImageEpochDeltaBalancer(
            5, "released_rank_underperformance", 1.0,
        )
        first = balancer.update([10.0, 10.0, 10.0, 10.0, 10.0])
        second = balancer.update([11.0, 10.0, 12.0, 9.0, 13.0])

        expected = torch.tensor([1.5, 2.0, 1.0, 2.5, 0.5], dtype=torch.float64)
        expected /= expected.sum()
        self.assertTrue(torch.allclose(first, torch.full((5,), 0.2, dtype=torch.float64)))
        self.assertTrue(torch.allclose(second, expected))
        self.assertEqual(int(torch.argmax(second)), 3)
        self.assertEqual(int(torch.argmin(second)), 4)

    def test_released_image_rank_balancer_rejects_other_pool_sizes(self) -> None:
        balancer = ImageEpochDeltaBalancer(
            6, "released_rank_underperformance", 1.0,
        )
        balancer.update([0.0] * 6)
        with self.assertRaisesRegex(ValueError, "exactly five robust teachers"):
            balancer.update([1.0] * 6)

    def test_unet_checkpoint_remap_is_idempotent_for_unified_convolutions(self) -> None:
        state = {
            "downs.0.conv1.conv.weight": torch.ones(1),
            "downs.0.conv2.conv.weight": torch.ones(1),
            "bottleneck.conv1.conv.weight": torch.ones(1),
            "ups_conv.0.conv1.conv.weight": torch.ones(1),
        }

        self.assertEqual(remap_legacy_unet_state_dict(state), state)

    def test_set12_evaluation_does_not_require_training_hdf5(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-set12-eval-") as temporary:
            root = Path(temporary) / "berkeley400"
            (root / "train").mkdir(parents=True)
            set12 = root / "Set12"
            set12.mkdir()
            (root / "train" / "train001.png").write_bytes(b"training fixture")
            (set12 / "01.png").write_bytes(b"test fixture")
            missing_h5 = Path(temporary) / "missing-h5"
            config = {
                "data": {
                    "berkeley_root": str(root),
                    "set12_dir": str(set12),
                    "denoising_h5_dir": str(missing_h5),
                }
            }

            training, evaluation, identity = build_task_datasets(
                config, "denoising", include_training=False,
            )

            self.assertIsNone(training)
            self.assertIsInstance(evaluation, Set12Dataset)
            self.assertEqual(len(evaluation), 1)
            self.assertNotIn("preprocessed_h5", identity)
            with self.assertRaises(FileNotFoundError):
                build_task_datasets(config, "denoising", include_training=True)

    def test_classification_normalization_defaults_are_dataset_specific(self) -> None:
        cifar10 = cifar_normalization("cifar10")
        cifar100 = cifar_normalization("cifar100")
        cifar100_legacy = cifar_normalization("cifar100", "released-legacy")

        self.assertEqual(cifar10["mean"], (0.4914, 0.4822, 0.4465))
        self.assertEqual(cifar10["std"], (0.2023, 0.1994, 0.2010))
        self.assertEqual(cifar100["mean"], (0.5071, 0.4867, 0.4408))
        self.assertEqual(cifar100["std"], (0.2675, 0.2565, 0.2761))
        self.assertEqual(cifar100_legacy["mean"], cifar10["mean"])
        self.assertEqual(cifar100_legacy["std"], cifar10["std"])
        self.assertNotEqual(cifar100["mean"], cifar100_legacy["mean"])

    def test_classification_normalization_rejects_unknown_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dataset"):
            cifar_normalization("cifar1000")
        with self.assertRaisesRegex(ValueError, "unsupported normalization profile"):
            cifar_normalization("cifar100", "automatic")

    def test_classification_cli_defaults_to_dataset_native_normalization(self) -> None:
        native = parse_classification_args([
            "train-teachers", "--dataset", "cifar100", "--device-type", "rram",
        ])
        legacy = parse_classification_args([
            "train-teachers", "--dataset", "cifar100", "--device-type", "rram",
            "--normalization-profile", "released-legacy",
        ])
        self.assertEqual(native.normalization_profile, "dataset-native")
        self.assertEqual(legacy.normalization_profile, "released-legacy")

    def test_classification_identity_records_normalization_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-cifar-identity-") as temporary:
            identity = cifar_identity(
                "cifar100", Path(temporary), include_hash=False,
            )
        self.assertEqual(identity["normalization"], {
            "profile": "dataset-native",
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
        })

    def test_checkpoint_normalization_manifest_prevents_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-normalization-manifest-") as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            manifest = {
                "schema": "mtrd.classification.checkpoint.v1",
                "checkpoint_sha256": classification_sha256_file(checkpoint),
                "dataset": {
                    "dataset": "cifar100",
                    "normalization": {
                        "profile": "dataset-native",
                        "mean": [0.5071, 0.4867, 0.4408],
                        "std": [0.2675, 0.2565, 0.2761],
                    },
                },
            }
            checkpoint.with_suffix(".manifest.json").write_text(
                json.dumps(manifest), encoding="ascii",
            )
            identity = checkpoint_normalization_identity(
                checkpoint, dataset="cifar100", requested_profile="dataset-native",
            )
            self.assertTrue(identity["verified"])
            with self.assertRaisesRegex(ValueError, "normalization mismatch"):
                checkpoint_normalization_identity(
                    checkpoint, dataset="cifar100", requested_profile="released-legacy",
                )

    def test_checkpoint_without_training_manifest_is_explicitly_unverified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-normalization-unverified-") as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            identity = checkpoint_normalization_identity(
                checkpoint, dataset="cifar10", requested_profile="dataset-native",
            )
        self.assertFalse(identity["manifest_present"])
        self.assertFalse(identity["verified"])

    def test_classification_strict_loader_accepts_lossless_modular_vgg_layout(self) -> None:
        torch.manual_seed(17)
        source = PaperVGG16(num_classes=100).eval()
        modular_state = remap_legacy_vgg_state_dict(source.state_dict())
        self.assertEqual(len(modular_state), len(source.state_dict()))

        with tempfile.TemporaryDirectory(prefix="mtrd-modular-vgg-") as temporary:
            checkpoint = Path(temporary) / "candidate.pth"
            torch.save({"state_dict": modular_state}, checkpoint)
            restored = PaperVGG16(num_classes=100).eval()
            load_checkpoint_strict(restored, str(checkpoint), "cpu")

        for key, expected in source.state_dict().items():
            self.assertTrue(torch.equal(restored.state_dict()[key], expected), key)
        inputs = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(source(inputs), restored(inputs)))

    @patch("classification.evaluation.build_cifar_loaders")
    def test_classification_formal_evaluation_rejects_per_mac_before_data(
        self, build_loaders,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "per-MAC forward-noise RNG"):
            evaluate_classification(Namespace(
                backend="aihwkit-additive",
                device_type="pcm",
                realization_scope="per-mac",
            ))
        build_loaders.assert_not_called()

    def test_image_formal_evaluation_rejects_per_mac(self) -> None:
        config = {"simulation": {"pcm_realization_scope": "per_mac"}}
        with self.assertRaisesRegex(RuntimeError, "cannot be seeded or replayed"):
            resolve_realization_scope(config, "pcm", None)

    def test_generated_checkpoint_roles_are_complete_and_unverified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-generated-roles-") as temporary:
            root = Path(temporary)
            first = root / "first.pth"
            second = root / "second.pth"
            first.write_bytes(b"first checkpoint")
            second.write_bytes(b"second checkpoint")
            manifest = root / "roles.json"
            identity = write_checkpoint_roles(
                manifest,
                {"fixture.nominal": first, "fixture.mtrd": second},
                group_id="generated-fixture",
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertFalse(payload["role_assignments_author_verified"])
            self.assertEqual(set(payload["roles"]), {"fixture.nominal", "fixture.mtrd"})
            self.assertEqual(set(identity["validated_roles"]), set(payload["roles"]))

    def test_classification_training_writes_evaluation_role_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-classification-roles-") as temporary:
            root = Path(temporary)
            directory = root / "cifar10" / "pcm"
            directory.mkdir(parents=True)
            (directory / "teacher_clean.pt").write_bytes(b"nominal checkpoint")
            (directory / "student_mtrd_pcm.pt").write_bytes(b"mtrd checkpoint")
            path = _write_evaluation_checkpoint_roles(Namespace(
                checkpoint_root=str(root), dataset="cifar10", device_type="pcm", seed=2025,
            ))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload["roles"]),
                {
                    "classification.cifar10.pcm.nominal",
                    "classification.cifar10.pcm.mtrd",
                },
            )

    def test_checkpoint_roles_bind_path_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-checkpoint-roles-") as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint fixture")
            manifest = root / "roles.json"
            manifest.write_text(
                json.dumps({
                    "schema_version": 1,
                    "group_id": "fixture",
                    "asset_root": str(root),
                    "role_assignments_author_verified": False,
                    "roles": {
                        "fixture.nominal": {
                            "path": "model.pth",
                            "size_bytes": checkpoint.stat().st_size,
                            "sha256": sha256_file(checkpoint),
                        }
                    },
                }),
                encoding="ascii",
            )
            identity = validate_checkpoint_roles(
                manifest, {"fixture.nominal": checkpoint},
            )
            self.assertFalse(identity["role_assignments_author_verified"])
            self.assertEqual(
                identity["validated_roles"]["fixture.nominal"]["path"],
                str(checkpoint.resolve()),
            )

            swapped = root / "other.pth"
            swapped.write_bytes(b"checkpoint fixture")
            with self.assertRaisesRegex(ValueError, "resolves to"):
                validate_checkpoint_roles(
                    manifest, {"fixture.nominal": swapped},
                )

    def test_classification_public_help_excludes_internal_worker(self) -> None:
        help_text = build_parser().format_help()
        self.assertNotIn("_train-mtrd", help_text)
        self.assertIn("preflight", help_text)
        self.assertIn("train", help_text)
        self.assertIn("evaluate", help_text)

    def test_empty_cifar_directories_fail_both_preflights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-empty-cifar-") as temporary:
            root = Path(temporary)
            (root / "cifar-10-batches-py").mkdir()
            checks = cifar_integrity_checks(root)
            self.assertTrue(checks)
            self.assertFalse(any(item["ok"] for item in checks))
            self.assertFalse(cifar_identity("cifar10", root)["present"])

    def test_denoising_preflight_rejects_non_png_source_images(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-denoising-source-") as temporary:
            root = Path(temporary)
            (root / "sample.jpg").write_bytes(b"not consumed by preprocessing")
            with self.assertRaisesRegex(ValueError, "PNG files only"):
                _denoising_png_files(root)

    def test_set12_dataset_uses_the_png_asset_filter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-set12-dataset-") as temporary:
            root = Path(temporary)
            image = root / "01.png"
            image.write_bytes(b"png fixture")
            dataset = Set12Dataset(root)
            self.assertEqual(dataset.paths, [image])

    def test_source_tree_identity_ignores_runtime_bytecode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-source-tree-") as temporary:
            root = Path(temporary)
            (root / "module.py").write_text("value = 1\n", encoding="ascii")
            before = source_tree_sha256(root)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "module.pyc").write_bytes(b"runtime cache")
            self.assertEqual(before, source_tree_sha256(root))
            (root / "module.py").write_text("value = 2\n", encoding="ascii")
            self.assertNotEqual(before, source_tree_sha256(root))

    def test_source_tree_identity_includes_tests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-source-tree-tests-") as temporary:
            root = Path(temporary)
            tests = root / "tests"
            tests.mkdir()
            test_file = tests / "test_contract.py"
            test_file.write_text("assert True\n", encoding="ascii")
            before = source_tree_sha256(root)
            test_file.write_text("assert False\n", encoding="ascii")
            self.assertNotEqual(before, source_tree_sha256(root))

    def test_classification_completed_job_rejects_changed_configuration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-classification-sentinel-") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint fixture")
            job = TrainingJob(
                name="teacher_rram_0.1",
                command=("python", "train.py", "--epochs", "200"),
                expected_checkpoint=checkpoint,
                log_path=root / "train.log",
                done_path=root / "done.json",
            )
            preflight = {
                "dataset_fingerprint": {"sha256": "dataset-fixture"},
                "environment": {"torch": "fixture"},
                "git": {"commit": "fixture", "dirty": False},
                "protocol": {"epochs": 200},
            }
            identity = training_job_identity(job, preflight)
            job.done_path.write_text(
                json.dumps({
                    "base_command": list(job.command),
                    "job_identity_sha256": identity,
                    "checkpoint_sha256": sha256_file(checkpoint),
                }),
                encoding="ascii",
            )
            validate_completed_job(job, preflight, identity)

            changed = replace(
                job,
                command=("python", "train.py", "--epochs", "300"),
            )
            changed_identity = training_job_identity(changed, preflight)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                validate_completed_job(changed, preflight, changed_identity)

    def test_classification_balancing_policies_have_opposite_direction(self) -> None:
        current = [0.8, 0.7, 0.6]
        previous = [0.7, 0.7, 0.7]
        positive = update_balancing_weights(
            current, previous, "positive_delta_softmax", 1.0
        )
        negative = update_balancing_weights(
            current, previous, "negative_delta_softmax", 1.0
        )
        self.assertGreater(positive[0], positive[2])
        self.assertLess(negative[0], negative[2])
        self.assertAlmostEqual(sum(positive), 1.0)
        self.assertAlmostEqual(sum(negative), 1.0)

    def test_classification_balancer_uses_accuracy_fractions(self) -> None:
        balancer = EpochDeltaBalancer(3)
        first = balancer.update([0.6, 0.7, 0.8])
        second = balancer.update([0.7, 0.68, 0.81])
        expected = torch.softmax(
            torch.tensor([0.1, -0.02, 0.01], dtype=torch.float64), dim=0
        )
        self.assertTrue(torch.allclose(first, torch.full((3,), 1 / 3, dtype=torch.float64)))
        self.assertTrue(torch.allclose(second, expected))
        with self.assertRaisesRegex(ValueError, "fractions"):
            balancer.update([60.0, 70.0, 80.0])

    def test_eq4_propagates_through_both_student_paths(self) -> None:
        nominal = torch.randn(4, 10, requires_grad=True)
        noisy = torch.randn(4, 10, requires_grad=True)
        teachers = [torch.randn(4, 10) for _ in range(3)]
        targets = torch.tensor([0, 1, 2, 3])
        beta = torch.tensor([0.2, 0.3, 0.5])
        loss, parts = paper_mtrd_loss(
            nominal, noisy, teachers, targets, beta, alpha=0.7, temperature=5.0
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(nominal.grad)
        self.assertIsNotNone(noisy.grad)
        self.assertTrue(torch.isfinite(parts["kd"]))
        self.assertTrue(torch.isfinite(parts["task"]))

    def test_eq4_uses_weighted_individual_teacher_losses(self) -> None:
        nominal = torch.tensor([[1.0, -0.5, 0.25]], requires_grad=True)
        noisy = torch.zeros((1, 3), requires_grad=True)
        teachers = [
            torch.tensor([[2.0, -1.0, 0.0]]),
            torch.tensor([[-0.5, 1.5, 0.5]]),
        ]
        beta = torch.tensor([0.25, 0.75])
        temperature = 2.0
        loss, parts = paper_mtrd_loss(
            nominal, noisy, teachers, torch.tensor([0]), beta,
            alpha=1.0, temperature=temperature,
        )
        student_log = torch.log_softmax(nominal / temperature, dim=1)
        individual = torch.stack([
            torch.nn.functional.kl_div(
                student_log,
                torch.softmax(teacher / temperature, dim=1),
                reduction="batchmean",
            )
            for teacher in teachers
        ])
        expected = torch.sum(beta * individual) * temperature**2
        mixture = sum(
            weight * torch.softmax(teacher / temperature, dim=1)
            for weight, teacher in zip(beta, teachers)
        )
        mixture_kl = torch.nn.functional.kl_div(
            student_log, mixture, reduction="batchmean",
        ) * temperature**2
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(parts["kd"], expected.detach()))
        self.assertFalse(torch.allclose(expected, mixture_kl))

    def test_image_eq4_reports_weighted_individual_mse(self) -> None:
        nominal = torch.zeros((1, 1, 1, 1), requires_grad=True)
        noisy = torch.zeros_like(nominal, requires_grad=True)
        teachers = [torch.ones_like(nominal), torch.full_like(nominal, 3.0)]
        loss, parts = image_mtrd_loss(
            "denoising", nominal, noisy, teachers, torch.zeros_like(nominal),
            torch.tensor([0.25, 0.75]), alpha=1.0, temperature=5.0,
        )
        self.assertAlmostEqual(loss.detach().item(), 7.0)
        self.assertAlmostEqual(parts["kd"].item(), 7.0)

    def test_aihwkit_rejects_incomplete_operator_coverage_first(self) -> None:
        model = nn.Sequential(nn.ConvTranspose2d(2, 2, kernel_size=2))
        with self.assertRaisesRegex(TypeError, "ConvTranspose2d"):
            convert_model(
                model,
                0.06,
                input_bits=8,
                output_bits=8,
                seed=2025,
                realization_scope="fixed-trial",
            )

    @patch("simulators.aihwkit.validate_noise_scale")
    def test_image_aihwkit_validation_covers_every_pcm_eta(self, validate) -> None:
        validate.side_effect = lambda eta, **_kwargs: {
            "eta": eta,
            "passed": True,
            "relative_std_error": 0.0,
        }
        config = {
            "seed": 2025,
            "protocol": {"pcm_levels": [0.02, 0.04, 0.06, 0.08, 0.10]},
            "simulation": {
                "runtime_validation_samples": 32,
                "runtime_validation_relative_std_tolerance": 0.1,
            },
        }
        results = run_noise_scale_validation(
            config, backend="aihwkit-additive", realization_scope="fixed_trial",
        )
        self.assertEqual([item["eta"] for item in results], [0.02, 0.04, 0.06, 0.08, 0.10])
        self.assertEqual(validate.call_count, 5)


if __name__ == "__main__":
    unittest.main()
