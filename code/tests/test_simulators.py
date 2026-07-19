from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from capsule import run as capsule_run
from simulators import cli as simulator_cli
from simulators.neurosim import (
    EXPECTED_NEUROSIM_COMMIT,
    EXPECTED_NEUROSIM_SOURCE_TREE_SHA256,
    FunctionalAdapterUnavailable,
    NeuroSimLayerFiles,
    NeuroSimPPARequest,
    neurosim_functional_status,
    parse_neurosim_output,
    neurosim_source_tree_sha256,
    require_functional_adapter,
    run_neurosim_ppa,
    run_neurosim_smoke,
    validate_neurosim_inputs,
)


NEUROSIM_OUTPUT = """+ChipArea : 100um^2
Chip total CIM array : 40um^2
Chip clock period is: 2ns
Chip layer-by-layer readLatency (per image) is: 20ns
Chip total readDynamicEnergy is: 30pJ
Chip total leakage Energy is: 2pJ
Energy Efficiency TOPS/W (total leakage energy is excluded): 4
Throughput TOPS (total): 5
Throughput FPS (total): 6
Compute efficiency TOPS/mm^2 (total): 7
Total Run-time of NeuroSim: 0.01 seconds
"""


class NeuroSimInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mtrd-neurosim-test-")
        self.root = Path(self.temporary.name) / "neurosim"
        source = self.root / "NeuroSIM"
        source.mkdir(parents=True)
        (self.root / "NEUROSIM_COMMIT").write_text(
            EXPECTED_NEUROSIM_COMMIT + "\n", encoding="ascii"
        )
        (source / "Param.cpp").write_text("// fixture\n", encoding="ascii")
        binary = source / "main"
        binary.write_text(
            "#!/bin/sh\n"
            "cat <<'EOF'\n"
            + NEUROSIM_OUTPUT
            + "EOF\n",
            encoding="ascii",
        )
        binary.chmod(0o755)

        self.network = Path(self.temporary.name) / "network.csv"
        self.weights = Path(self.temporary.name) / "weights.csv"
        self.inputs = Path(self.temporary.name) / "inputs.csv"
        self.network.write_text("1,1,64,1,1,128,0,1\n", encoding="ascii")
        weight_row = ",".join(["0.5"] * 128)
        self.weights.write_text((weight_row + "\n") * 64, encoding="ascii")
        self.inputs.write_text("1\n" * 64, encoding="ascii")
        self.request = NeuroSimPPARequest(
            network_csv=self.network,
            layers=(NeuroSimLayerFiles(self.weights, self.inputs),),
            neurosim_root=self.root,
            subarray_rows=64,
            parallel_rows=32,
            expected_source_tree_sha256=neurosim_source_tree_sha256(self.root),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validates_non_128_row_fixture(self) -> None:
        identity = validate_neurosim_inputs(self.request)
        self.assertEqual(identity["layer_count"], 1)
        self.assertEqual(identity["layers"][0]["weight_shape"], [64, 128])
        self.assertEqual(identity["layers"][0]["input_shape"], [64, 1])

    def test_rejects_fractional_activation(self) -> None:
        self.inputs.write_text("0.5\n" + "1\n" * 63, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "non-integer activation"):
            validate_neurosim_inputs(self.request)

    def test_rejects_zero_stride(self) -> None:
        self.network.write_text("1,1,64,1,1,128,0,0\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "invalid dimensions"):
            validate_neurosim_inputs(self.request)

    def test_rejects_stride_dimensions_that_diverge_from_pinned_cpp(self) -> None:
        self.network.write_text("5,5,1,3,3,1,0,2\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "divisible by stride"):
            validate_neurosim_inputs(self.request)

    def test_source_tree_hash_detects_edited_cpp_with_unchanged_marker(self) -> None:
        expected = neurosim_source_tree_sha256(self.root)
        request = replace(self.request, expected_source_tree_sha256=expected)
        validate_neurosim_inputs(request)
        (self.root / "NeuroSIM" / "Param.cpp").write_text(
            "// edited fixture\n", encoding="ascii"
        )
        with self.assertRaisesRegex(RuntimeError, "source-tree hash mismatch"):
            validate_neurosim_inputs(request)
        self.assertNotEqual(expected, EXPECTED_NEUROSIM_SOURCE_TREE_SHA256)

    def test_rejects_revision_mismatch(self) -> None:
        request = replace(self.request, expected_commit="0" * 40)
        with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
            validate_neurosim_inputs(request)

    def test_parses_required_metrics_and_derivations(self) -> None:
        metrics = parse_neurosim_output(NEUROSIM_OUTPUT)
        self.assertEqual(metrics["area_um2"], 100.0)
        self.assertEqual(metrics["latency_ns"], 20.0)
        self.assertEqual(metrics["dynamic_energy_pj"], 30.0)
        self.assertEqual(metrics["energy_pj"], 32.0)
        self.assertEqual(metrics["edp_pj_ns"], 640.0)

    def test_runner_writes_hashed_manifest(self) -> None:
        output = Path(self.temporary.name) / "result"
        result = run_neurosim_ppa(self.request, output)
        self.assertEqual(result["return_code"], 0)
        self.assertGreater(result["metrics"]["area_um2"], 0)
        manifest = json.loads((output / "neurosim_ppa.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["input_identity"]["revision"], EXPECTED_NEUROSIM_COMMIT)
        self.assertTrue(manifest["raw_log_sha256"])
        self.assertEqual(manifest["request"]["subarray_rows"], 64)

    def test_runner_refuses_to_reuse_success_output(self) -> None:
        output = Path(self.temporary.name) / "immutable-result"
        run_neurosim_ppa(self.request, output)
        original_manifest = (output / "neurosim_ppa.json").read_bytes()

        with self.assertRaisesRegex(FileExistsError, "will not be reused"):
            run_neurosim_ppa(self.request, output)

        self.assertEqual((output / "neurosim_ppa.json").read_bytes(), original_manifest)

    def test_runner_failure_has_no_success_manifest(self) -> None:
        binary = self.root / "NeuroSIM" / "main"
        binary.write_text("#!/bin/sh\necho simulated failure\nexit 9\n", encoding="ascii")
        binary.chmod(0o755)
        output = Path(self.temporary.name) / "failed-result"

        with self.assertRaisesRegex(RuntimeError, "exited with code 9"):
            run_neurosim_ppa(self.request, output)

        self.assertFalse((output / "neurosim_ppa.json").exists())
        self.assertFalse((output / "neurosim_raw.log").exists())
        self.assertEqual(
            (output / "neurosim_failed.log").read_text(encoding="utf-8"),
            "simulated failure\n",
        )

    def test_functional_metrics_fail_closed(self) -> None:
        for model in ("vgg16", "unet", "dncnn"):
            with self.subTest(model=model):
                with self.assertRaises(FunctionalAdapterUnavailable):
                    require_functional_adapter(self.root, model)

    def test_functional_status_reports_architecture_and_equation_gaps(self) -> None:
        (self.root / "inference.py").write_text(
            "# Native fixture supports vgg8 and cifar100 only.\n"
            "model = 'vgg8'\n"
            "dataset = 'cifar100'\n",
            encoding="ascii",
        )
        modules = (
            self.root
            / "pytorch-quantization"
            / "pytorch_quantization"
            / "cim"
            / "modules"
        )
        modules.mkdir(parents=True)
        (modules / "macro.py").write_text(
            "def conductance_sampling():\n"
            "    return torch.normal(mean=0.0, std=1.0)\n",
            encoding="ascii",
        )

        status = neurosim_functional_status(self.root, "vgg16")

        self.assertFalse(status["functional_ready"])
        self.assertTrue(status["native_frontend_vgg8_detected"])
        self.assertTrue(status["native_frontend_cifar100_detected"])
        self.assertFalse(status["paper_model_adapter_detected"])
        self.assertFalse(status["paper_rram_equation_supported"])
        self.assertEqual(status["native_device_sampling_distribution"], "gaussian-conductance")
        self.assertGreaterEqual(len(status["reasons"]), 2)

    def test_functional_status_requires_explicit_paper_equation_marker(self) -> None:
        (self.root / "mtrd_functional_adapter.py").write_text(
            "MTRD_PAPER_EQ1_LOGNORMAL = True\n"
            "def vgg16():\n"
            "    return 'cifar100'\n",
            encoding="ascii",
        )

        status = neurosim_functional_status(self.root, "vgg16")

        self.assertTrue(status["functional_ready"])
        self.assertEqual(
            status["paper_equation_marker_sources"],
            ["mtrd_functional_adapter.py"],
        )
        self.assertTrue(status["upstream_native_cim_array_kernel_used"])
        self.assertEqual(require_functional_adapter(self.root, "vgg16"), status)

    def test_smoke_runs_a_deterministic_64_by_128_fixture(self) -> None:
        output = Path(self.temporary.name) / "smoke-result"
        result = run_neurosim_smoke(
            self.root, output,
            expected_source_tree_sha256=neurosim_source_tree_sha256(self.root),
        )

        identity = result["input_identity"]
        self.assertEqual(identity["layers"][0]["weight_shape"], [64, 128])
        self.assertEqual(identity["layers"][0]["input_shape"], [64, 1])
        self.assertEqual(result["smoke_test"]["fixture"], "deterministic_64x128_ppa")
        manifest = json.loads((output / "neurosim_ppa.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["smoke_test"]["functional_inference"])
        self.assertTrue(Path(manifest["smoke_test"]["input_dir"]).is_dir())

    def test_smoke_refuses_to_reuse_its_input_directory(self) -> None:
        output = Path(self.temporary.name) / "smoke-immutable"
        input_dir = output.with_name(output.name + "-inputs")
        input_dir.mkdir()

        with self.assertRaisesRegex(FileExistsError, "will not be reused"):
            run_neurosim_smoke(
                self.root, output,
                expected_source_tree_sha256=neurosim_source_tree_sha256(self.root),
            )

    def test_smoke_rejects_invalid_timeout_before_writing_inputs(self) -> None:
        output = Path(self.temporary.name) / "smoke-invalid-timeout"

        with self.assertRaisesRegex(ValueError, "must be positive"):
            run_neurosim_smoke(
                self.root, output, timeout_seconds=0,
                expected_source_tree_sha256=neurosim_source_tree_sha256(self.root),
            )

        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + "-inputs").exists())


class SimulatorCliTest(unittest.TestCase):
    def test_aihwkit_probe_returns_nonzero_when_runtime_is_not_ready(self) -> None:
        payload = {"runtime_ready": False, "error": "fixture"}
        stdout = io.StringIO()
        with mock.patch("simulators.aihwkit.runtime_probe", return_value=payload):
            with contextlib.redirect_stdout(stdout):
                return_code = simulator_cli.main(["aihwkit-probe"])

        self.assertEqual(return_code, 2)
        self.assertEqual(json.loads(stdout.getvalue()), payload)

    def test_aihwkit_probe_returns_zero_when_runtime_is_ready(self) -> None:
        payload = {"runtime_ready": True}
        with mock.patch("simulators.aihwkit.runtime_probe", return_value=payload):
            with contextlib.redirect_stdout(io.StringIO()):
                return_code = simulator_cli.main(["aihwkit-probe"])

        self.assertEqual(return_code, 0)


class CapsuleStatusTest(unittest.TestCase):
    def test_local_status_uses_capsule_data_and_results_directories(self) -> None:
        local_root = Path(self.id().replace(".", "-"))
        local_code = local_root / "code"
        stdout = io.StringIO()
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"MTRD_DATA_ROOT", "MTRD_RESULTS_ROOT"}
        }
        with mock.patch.object(capsule_run, "CODE_ROOT", local_code):
            with mock.patch.object(capsule_run, "CAPSULE_ROOT", local_root):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch(
                        "simulators.neurosim.neurosim_capabilities",
                        return_value={"ppa_ready": False},
                    ):
                        with contextlib.redirect_stdout(stdout):
                            return_code = capsule_run.status()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(return_code, 0)
        self.assertEqual(payload["data_root"], str(local_root / "data"))
        self.assertEqual(payload["results_root"], str(local_root / "results"))


if __name__ == "__main__":
    unittest.main()
