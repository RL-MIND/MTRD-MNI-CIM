from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

from image_workflows.workflow import Set12Dataset
from utils.data import prepare_denoising_data


def _non_gray_bgr_image(height: int, width: int) -> np.ndarray:
    rows, columns = np.indices((height, width))
    blue = (7 * rows + 11 * columns + 3) % 256
    green = (13 * rows + 5 * columns + 29) % 256
    red = (3 * rows + 17 * columns + 101) % 256
    return np.stack((blue, green, red), axis=-1).astype(np.uint8)


class DnCNNSourcePreprocessingTest(unittest.TestCase):
    def test_training_resize_uses_released_literal_height_width_dsize(self) -> None:
        source_image = _non_gray_bgr_image(61, 79)
        with tempfile.TemporaryDirectory(prefix="mtrd-dncnn-resize-") as temporary:
            root = Path(temporary) / "berkeley400"
            train = root / "train"
            set12 = root / "Set12"
            output = Path(temporary) / "h5"
            train.mkdir(parents=True)
            set12.mkdir()
            self.assertTrue(cv2.imwrite(str(train / "train.png"), source_image))
            self.assertTrue(cv2.imwrite(str(set12 / "01.png"), source_image))

            prepare_denoising_data(
                str(root), patch_size=40, stride=100, h5_dir=str(output), seed=17,
            )

            with h5py.File(output / "train.h5", "r") as handle:
                actual = np.asarray(handle["0"])

        released = cv2.resize(
            source_image,
            (source_image.shape[0], source_image.shape[1]),
            interpolation=cv2.INTER_CUBIC,
        )
        conventional = cv2.resize(
            source_image,
            (source_image.shape[1], source_image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        expected = released[:, :, 0][None, :40, :40].astype(np.float32) / 255.0
        conventional_patch = conventional[:, :, 0][None, :40, :40].astype(np.float32) / 255.0

        self.assertEqual(released.shape[:2], (79, 61))
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, conventional_patch))

    def test_set12_uses_opencv_bgr_first_channel_not_luminance(self) -> None:
        source_image = _non_gray_bgr_image(7, 11)
        with tempfile.TemporaryDirectory(prefix="mtrd-set12-bgr-") as temporary:
            directory = Path(temporary)
            path = directory / "01.png"
            self.assertTrue(cv2.imwrite(str(path), source_image))

            actual, sample_id = Set12Dataset(directory)[0]
            expected_bgr = cv2.imread(str(path))[:, :, 0].astype(np.float32) / 255.0
            previous_luminance = (
                np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            )

        self.assertEqual(sample_id, "01")
        np.testing.assert_array_equal(actual.numpy(), expected_bgr[None, :, :])
        self.assertFalse(np.array_equal(actual.numpy()[0], previous_luminance))
