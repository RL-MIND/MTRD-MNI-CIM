"""Lazy exports for the small set of shared public utilities."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "DenoisingDataset": ("data", "DenoisingDataset"),
    "get_classification_loaders": ("data", "get_classification_loaders"),
    "get_denoising_loaders": ("data", "get_denoising_loaders"),
    "prepare_denoising_data": ("data", "prepare_denoising_data"),
    "format_time": ("helpers", "format_time"),
    "get_device": ("helpers", "get_device"),
    "progress_bar": ("helpers", "progress_bar"),
    "set_seed": ("helpers", "set_seed"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
