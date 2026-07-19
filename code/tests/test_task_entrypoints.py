from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest

from capsule.run import _usage
from classification.cli import main as classification_main
from image_workflows.cli import run_task
from image_workflows.workflow import build_task_parser


class TaskEntrypointTest(unittest.TestCase):
    def test_denoising_parser_fixes_the_task(self) -> None:
        args = build_task_parser("denoising").parse_args(["train"])
        self.assertEqual(args.task, "denoising")

    def test_segmentation_parser_hides_denoising_h5_option(self) -> None:
        parser = build_task_parser("segmentation")
        train = next(
            action.choices["train"]
            for action in parser._actions
            if isinstance(getattr(action, "choices", None), dict)
            and "train" in action.choices
        )
        overwrite_h5 = next(
            action for action in train._actions if action.dest == "overwrite_h5"
        )
        self.assertEqual(overwrite_h5.help, "==SUPPRESS==")

    def test_segmentation_entrypoint_rejects_denoising_only_h5_option(self) -> None:
        with self.assertRaisesRegex(SystemExit, "only available from the denosing"):
            run_task("segmentation", ["train", "--overwrite-h5"])

    def test_public_task_directories_are_available(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        for directory in ("classification", "denosing", "segmentation"):
            self.assertTrue((code_root / directory).is_dir())
        usage = _usage()
        self.assertIn("classification", usage)
        self.assertIn("denosing", usage)
        self.assertIn("segmentation", usage)

    def test_task_templates_do_not_mix_task_specific_assets_or_protocols(self) -> None:
        config_root = Path(__file__).resolve().parents[1] / "configs"
        denoising = json.loads(
            (config_root / "denosing.template.json").read_text(encoding="utf-8")
        )
        segmentation = json.loads(
            (config_root / "segmentation.template.json").read_text(encoding="utf-8")
        )
        self.assertIn("denoising", denoising["protocol"])
        self.assertNotIn("segmentation", denoising["protocol"])
        self.assertNotIn("carvana_image_dirs", denoising["data"])
        self.assertIn("segmentation", segmentation["protocol"])
        self.assertNotIn("denoising", segmentation["protocol"])
        self.assertNotIn("berkeley_root", segmentation["data"])

    def test_classification_dispatcher_exposes_both_public_subworkflows(self) -> None:
        for workflow in ("pytorch", "cim"):
            output = StringIO()
            with redirect_stdout(output):
                with self.assertRaises(SystemExit) as exit_context:
                    classification_main([workflow, "--help"])
            self.assertEqual(exit_context.exception.code, 0)
            self.assertIn("usage:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
