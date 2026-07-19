from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from image_workflows import workflow as dncnn_unet
from image_workflows.workflow import (
    denoising_h5_identity,
    prepare_denoising_h5,
    resolve_denoising_h5_input,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_flat_h5(path: Path, values: list[np.ndarray], *, keys: list[str] | None = None) -> None:
    with h5py.File(path, "w") as handle:
        for index, value in enumerate(values):
            handle[(keys or [str(item) for item in range(len(values))])[index]] = value


class DnCNNH5ProvenanceTest(unittest.TestCase):
    def _config(
        self, root: Path, *, train_h5: Path, val_h5: Path,
        train_sha256: str | None = None, val_sha256: str | None = None,
    ) -> dict[str, object]:
        h5_config: dict[str, object] = {
            "mode": "source-provided",
            "train_h5": str(train_h5),
            "val_h5": str(val_h5),
        }
        if train_sha256 is not None:
            h5_config["train_h5_sha256"] = train_sha256
            h5_config["val_h5_sha256"] = val_sha256
        return {
            "seed": 1,
            "data": {
                "berkeley_root": str(root / "berkeley400"),
                "set12_dir": str(root / "berkeley400" / "Set12"),
                "denoising_h5": h5_config,
            },
            "protocol": {
                "denoising": {
                    "patch_size": 40,
                    "patch_stride": 10,
                },
            },
        }

    @contextmanager
    def _mock_asset_relation(self, expected_values: list[np.ndarray], train_count: int):
        source = {
            "source": {
                "training_content_manifest_sha256": "training-content",
                "set12_content_manifest_sha256": "set12-content",
            }
        }
        with mock.patch.object(
            dncnn_unet, "denoising_preprocessing_spec", return_value=source,
        ), mock.patch.object(
            dncnn_unet, "_expected_dncnn_train_patch_count", return_value=train_count,
        ), mock.patch.object(
            dncnn_unet, "_expected_set12_h5_values", return_value=expected_values,
        ):
            yield

    def test_source_provided_h5_records_verified_hashes_and_set12_relation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-source-h5-") as temporary:
            root = Path(temporary)
            train_h5 = root / "train.h5"
            val_h5 = root / "val.h5"
            train_values = [
                np.full((1, 40, 40), value, dtype=np.float32)
                for value in (0.25, 0.75)
            ]
            set12_values = [np.full((1, 5, 7), 0.5, dtype=np.float32)]
            _write_flat_h5(train_h5, train_values)
            _write_flat_h5(val_h5, set12_values)
            config = self._config(
                root,
                train_h5=train_h5,
                val_h5=val_h5,
                train_sha256=_sha256(train_h5),
                val_sha256=_sha256(val_h5),
            )

            with self._mock_asset_relation(set12_values, len(train_values)):
                identity = denoising_h5_identity(config)

        self.assertEqual(identity["mode"], "source-provided")
        self.assertEqual(identity["train_h5_sha256"], config["data"]["denoising_h5"]["train_h5_sha256"])
        self.assertEqual(
            identity["h5_structure"]["validation"]["set12_content_relation"],
            "exact_released_bgr_channel_match",
        )
        self.assertEqual(
            identity["input_write_policy"],
            "source-provided-artifacts-never-rewritten",
        )

    def test_source_provided_h5_rejects_bad_digest_and_set12_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-source-h5-invalid-") as temporary:
            root = Path(temporary)
            train_h5 = root / "train.h5"
            val_h5 = root / "val.h5"
            train_values = [np.zeros((1, 40, 40), dtype=np.float32)]
            actual_set12 = [np.zeros((1, 4, 4), dtype=np.float32)]
            expected_set12 = [np.ones((1, 4, 4), dtype=np.float32)]
            _write_flat_h5(train_h5, train_values)
            _write_flat_h5(val_h5, actual_set12)
            config = self._config(
                root,
                train_h5=train_h5,
                val_h5=val_h5,
                train_sha256="0" * 64,
                val_sha256="1" * 64,
            )
            with self._mock_asset_relation(expected_set12, len(train_values)):
                with self.assertRaisesRegex(ValueError, "digest does not match"):
                    denoising_h5_identity(config)

            valid_digest = copy.deepcopy(config)
            valid_digest["data"]["denoising_h5"]["train_h5_sha256"] = _sha256(train_h5)
            valid_digest["data"]["denoising_h5"]["val_h5_sha256"] = _sha256(val_h5)
            with self._mock_asset_relation(expected_set12, len(train_values)):
                with self.assertRaisesRegex(ValueError, "does not match the configured Set12"):
                    denoising_h5_identity(valid_digest)

    def test_source_provided_h5_rejects_nonsequential_wrong_shape_or_dtype(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-source-h5-layout-") as temporary:
            root = Path(temporary)
            train_h5 = root / "train.h5"
            val_h5 = root / "val.h5"
            _write_flat_h5(
                train_h5,
                [
                    np.zeros((1, 40, 40), dtype=np.float32),
                    np.zeros((1, 39, 40), dtype=np.float32),
                ],
                keys=["0", "2"],
            )
            set12_values = [np.zeros((1, 3, 3), dtype=np.float32)]
            _write_flat_h5(val_h5, set12_values)
            config = self._config(root, train_h5=train_h5, val_h5=val_h5)
            with self._mock_asset_relation(set12_values, 2):
                with self.assertRaisesRegex(ValueError, "sequential decimal dataset keys"):
                    denoising_h5_identity(config)

            _write_flat_h5(
                train_h5,
                [
                    np.zeros((1, 40, 40), dtype=np.float32),
                    np.zeros((1, 39, 40), dtype=np.float64),
                ],
            )
            with self._mock_asset_relation(set12_values, 2):
                with self.assertRaisesRegex(ValueError, "shape|dtype"):
                    denoising_h5_identity(config)

    def test_source_provided_mode_never_overwrites_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-source-h5-no-write-") as temporary:
            root = Path(temporary)
            train_h5 = root / "train.h5"
            val_h5 = root / "val.h5"
            _write_flat_h5(train_h5, [np.zeros((1, 40, 40), dtype=np.float32)])
            _write_flat_h5(val_h5, [np.zeros((1, 2, 2), dtype=np.float32)])
            config = self._config(root, train_h5=train_h5, val_h5=val_h5)
            before = (_sha256(train_h5), _sha256(val_h5))
            with self.assertRaisesRegex(ValueError, "forbidden for source-provided"):
                prepare_denoising_h5(config, overwrite=True)
            self.assertEqual(before, (_sha256(train_h5), _sha256(val_h5)))

    def test_explicit_modes_and_digest_pairing_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mtrd-source-h5-config-") as temporary:
            root = Path(temporary)
            config = self._config(
                root, train_h5=root / "train.h5", val_h5=root / "val.h5",
            )
            config["data"]["denoising_h5"]["train_h5_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "requires both"):
                resolve_denoising_h5_input(config)

            invalid_mode = copy.deepcopy(config)
            invalid_mode["data"]["denoising_h5"].pop("train_h5_sha256")
            invalid_mode["data"]["denoising_h5"]["mode"] = "automatic"
            with self.assertRaisesRegex(ValueError, "must be one of"):
                resolve_denoising_h5_input(invalid_mode)


if __name__ == "__main__":
    unittest.main()
