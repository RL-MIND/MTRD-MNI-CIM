"""CIFAR loading and Berkeley400/Set12 preprocessing utilities."""

from __future__ import annotations

import glob
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as udata
import torchvision.transforms as transforms
from torchvision import datasets

from .paths import is_code_ocean


CIFAR_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
    },
}


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _remove_corrupt_cifar_cache(dataset_class, data_root):
    root = Path(data_root)
    archive = root / getattr(dataset_class, "filename", "")
    extracted = root / getattr(dataset_class, "base_folder", "")
    for path in (archive, extracted):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"[Classification] removed corrupt CIFAR cache: {path}")


def _build_cifar_dataset(dataset_class, data_root, train, transform):
    download = not is_code_ocean()
    try:
        return dataset_class(
            data_root, train=train, download=download, transform=transform
        )
    except RuntimeError as error:
        if is_code_ocean():
            raise FileNotFoundError(
                "CIFAR is not available in the read-only Code Ocean data asset "
                f"{data_root}. Mount an extracted CIFAR asset; runtime downloads "
                "to /data are disabled."
            ) from error
        if "not found or corrupted" not in str(error).lower():
            raise
        _remove_corrupt_cifar_cache(dataset_class, data_root)
        try:
            return dataset_class(
                data_root, train=train, download=True, transform=transform
            )
        except RuntimeError as retry_error:
            raise RuntimeError(
                "CIFAR download failed after the corrupt cache was removed. "
                "Rerun with network access or place the extracted CIFAR folder "
                "under --data_root."
            ) from retry_error


def get_classification_loaders(
    dataset_name,
    data_root="./data",
    batch_size=64,
    num_workers=4,
    seed=42,
):
    """Return deterministic CIFAR train and official test DataLoaders.

    The train transform is padding, random crop, and random horizontal flip.
    The returned evaluation loader is the official CIFAR test split; it is not
    an independent validation split.
    """
    if dataset_name not in CIFAR_STATS:
        available = ", ".join(sorted(CIFAR_STATS))
        raise ValueError(
            f"unsupported classification dataset {dataset_name!r}; expected {available}"
        )
    statistics = CIFAR_STATS[dataset_name]
    train_transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(statistics["mean"], statistics["std"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(statistics["mean"], statistics["std"]),
        ]
    )
    dataset_class = (
        datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    )
    train_set = _build_cifar_dataset(
        dataset_class, data_root, train=True, transform=train_transform
    )
    test_set = _build_cifar_dataset(
        dataset_class, data_root, train=False, transform=test_transform
    )
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
    }
    train_loader = udata.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=_loader_generator(seed),
        **common,
    )
    test_loader = udata.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        generator=_loader_generator(seed),
        **common,
    )
    print(
        f"[Classification] {dataset_name} | train={len(train_set)} "
        f"| test={len(test_set)} | classes={statistics['num_classes']}"
    )
    return train_loader, test_loader, statistics["num_classes"]


def _im2patch(image, window, stride=1):
    channels, width, height = image.shape
    if width < window or height < window:
        return np.zeros([channels, window, window, 0], np.float32)
    base = image[:, 0 : width - window + 1 : stride, 0 : height - window + 1 : stride]
    total = base.shape[1] * base.shape[2]
    patches = np.zeros([channels, window * window, total], np.float32)
    index = 0
    for row in range(window):
        for column in range(window):
            view = image[
                :,
                row : width - window + row + 1 : stride,
                column : height - window + column + 1 : stride,
            ]
            patches[:, index, :] = view.reshape(channels, total)
            index += 1
    return patches.reshape([channels, window, window, total])


def _data_augmentation(image, mode):
    output = np.transpose(image, (1, 2, 0))
    if mode == 1:
        output = np.flipud(output)
    elif mode == 2:
        output = np.rot90(output)
    elif mode == 3:
        output = np.flipud(np.rot90(output))
    elif mode == 4:
        output = np.rot90(output, k=2)
    elif mode == 5:
        output = np.flipud(np.rot90(output, k=2))
    elif mode == 6:
        output = np.rot90(output, k=3)
    elif mode == 7:
        output = np.flipud(np.rot90(output, k=3))
    return np.transpose(output, (2, 0, 1))


def prepare_denoising_data(
    data_path,
    patch_size=40,
    stride=10,
    aug_times=1,
    h5_dir=".",
    seed=42,
):
    """Create DnCNN HDF5 inputs from ``train/*.png`` and ``Set12/*.png``.

    This preserves the released preprocessing behavior: OpenCV's first BGR
    channel is used as the single image channel, and four resize scales
    (1.0, 0.9, 0.8, 0.7) are extracted for training. In particular, the
    released source passes ``(int(height * scale), int(width * scale))``
    directly to ``cv2.resize``. This is intentionally its literal argument
    order, even though OpenCV names the tuple ``(width, height)``.
    """
    import cv2
    import h5py

    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if aug_times < 1:
        raise ValueError("aug_times must be at least one")
    source = Path(data_path)
    train_files = sorted(glob.glob(str(source / "train" / "*.png")))
    test_files = sorted(glob.glob(str(source / "Set12" / "*.png")))
    if not train_files:
        raise FileNotFoundError(f"no Berkeley training PNG files found in {source / 'train'}")
    if not test_files:
        raise FileNotFoundError(f"no Set12 PNG files found in {source / 'Set12'}")

    output = Path(h5_dir)
    output.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(seed)
    scales = (1.0, 0.9, 0.8, 0.7)

    training_count = 0
    with h5py.File(output / "train.h5", "w") as h5_file:
        for file_name in train_files:
            image = cv2.imread(file_name)
            if image is None:
                raise ValueError(f"OpenCV could not decode training image: {file_name}")
            height, width, _ = image.shape
            for scale in scales:
                resized = cv2.resize(
                    image,
                    (int(height * scale), int(width * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
                grayscale = np.expand_dims(
                    resized[:, :, 0].copy(), 0
                ).astype(np.float32) / 255.0
                patches = _im2patch(grayscale, window=patch_size, stride=stride)
                for patch_index in range(patches.shape[3]):
                    patch = patches[:, :, :, patch_index].copy()
                    h5_file.create_dataset(str(training_count), data=patch)
                    training_count += 1
                    for augmentation_index in range(aug_times - 1):
                        augmented = _data_augmentation(
                            patch, int(random_generator.integers(1, 8))
                        )
                        key = f"{training_count}_aug_{augmentation_index + 1}"
                        h5_file.create_dataset(key, data=augmented)
                        training_count += 1

    with h5py.File(output / "val.h5", "w") as h5_file:
        for index, file_name in enumerate(test_files):
            image = cv2.imread(file_name)
            if image is None:
                raise ValueError(f"OpenCV could not decode Set12 image: {file_name}")
            grayscale = np.expand_dims(
                image[:, :, 0].copy(), 0
            ).astype(np.float32) / 255.0
            h5_file.create_dataset(str(index), data=grayscale)
    print(
        f"[Denoising] HDF5 ready: train={training_count}, test={len(test_files)}"
    )


class DenoisingDataset(udata.Dataset):
    """Lazily read either batched or per-sample DnCNN HDF5 files."""

    def __init__(self, h5_path):
        super().__init__()
        import h5py

        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as h5_file:
            if "data" in h5_file:
                self.batch_mode = True
                self.length = h5_file["data"].shape[0]
                self.keys = None
            else:
                self.batch_mode = False
                self.keys = sorted(
                    h5_file.keys(),
                    key=lambda key: (0, int(key)) if key.isdigit() else (1, key),
                )
                self.length = len(self.keys)
        self._h5_file = None

    def __len__(self):
        return self.length

    def _open(self):
        import h5py

        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, index):
        h5_file = self._open()
        if self.batch_mode:
            data = h5_file["data"][index]
        else:
            data = np.array(h5_file[self.keys[index]])
        return torch.as_tensor(data, dtype=torch.float32)


def get_denoising_loaders(h5_dir=".", batch_size=128, num_workers=4, seed=42):
    """Return a shuffled train loader and an indexable Set12 dataset."""
    root = Path(h5_dir)
    train_h5 = root / "train.h5"
    test_h5 = root / "val.h5"
    if not train_h5.is_file():
        raise FileNotFoundError(f"missing {train_h5}; run prepare_denoising_data first")
    if not test_h5.is_file():
        raise FileNotFoundError(f"missing {test_h5}; run prepare_denoising_data first")
    train_dataset = DenoisingDataset(train_h5)
    test_dataset = DenoisingDataset(test_h5)
    train_loader = udata.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=_loader_generator(seed),
    )
    print(
        f"[Denoising] train={len(train_dataset)} patches | test={len(test_dataset)} images"
    )
    return train_loader, test_dataset
