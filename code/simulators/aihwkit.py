"""Stable public API for the manuscript-oriented AIHWKit PCM backend."""

from classification.aihwkit_backend import (
    FORMAL_EVALUATION_SCOPES,
    PER_MAC_FORWARD_RNG_REPLAYABLE,
    PINNED_AIHWKIT_VERSION,
    RealizationScope,
    build_additive_config,
    convert_model,
    installed_version,
    runtime_probe,
    require_replayable_realization_scope,
    validate_noise_scale,
)

__all__ = [
    "FORMAL_EVALUATION_SCOPES",
    "PER_MAC_FORWARD_RNG_REPLAYABLE",
    "PINNED_AIHWKIT_VERSION",
    "RealizationScope",
    "build_additive_config",
    "convert_model",
    "installed_version",
    "runtime_probe",
    "require_replayable_realization_scope",
    "validate_noise_scale",
]
