"""Immutable protocol values and paper MTRD loss helpers.

The functions in this module deliberately implement Eqs. (4) and (6) from the
supplied manuscript.  They are separate from the repository's learned router,
which is a different method and must not be used for a paper reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


RRAM_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
PCM_GRID = (0.02, 0.04, 0.06, 0.08, 0.10)
STUDENT_NOISE = {"rram": 0.3, "pcm": 0.06}


@dataclass(frozen=True)
class TeacherSpec:
    label: str
    noise: float
    clean: bool = False


def teacher_specs(device_type: str, include_clean: bool = True) -> tuple[TeacherSpec, ...]:
    if device_type not in STUDENT_NOISE:
        raise ValueError(f"unsupported device type: {device_type}")
    grid = RRAM_GRID if device_type == "rram" else PCM_GRID
    specs: list[TeacherSpec] = []
    if include_clean:
        specs.append(TeacherSpec(label="clean", noise=0.0, clean=True))
    specs.extend(TeacherSpec(label=f"{value:g}", noise=value) for value in grid)
    return tuple(specs)


def paper_mtrd_loss(
    nominal_student_logits: torch.Tensor,
    noisy_student_logits: torch.Tensor,
    teacher_logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    beta: torch.Tensor,
    *,
    alpha: float = 0.7,
    temperature: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Implement the separation required by manuscript Eq. (4).

    KD is applied to the nominal student prediction.  The supervised task loss
    is applied to the independently noise-injected student prediction.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(teacher_logits) == 0:
        raise ValueError("at least one teacher is required")
    if beta.ndim != 1 or beta.numel() != len(teacher_logits):
        raise ValueError("beta must contain one scalar per teacher")

    beta = beta.to(device=nominal_student_logits.device, dtype=nominal_student_logits.dtype)
    beta = beta / beta.sum().clamp_min(torch.finfo(beta.dtype).eps)
    student_log_probabilities = F.log_softmax(
        nominal_student_logits / temperature, dim=1,
    )
    individual_kd = torch.stack([
        F.kl_div(
            student_log_probabilities,
            F.softmax(logits.detach() / temperature, dim=1),
            reduction="batchmean",
        )
        for logits in teacher_logits
    ])
    kd = torch.sum(beta * individual_kd) * temperature * temperature
    task = F.cross_entropy(noisy_student_logits, targets)
    total = alpha * kd + (1.0 - alpha) * task
    return total, {"kd": kd.detach(), "task": task.detach()}


class EpochDeltaBalancer:
    """Literal Eq. (6): beta = softmax(p_t - p_(t-1)).

    Performance values must be fractions in [0, 1], not percentages.  The
    manuscript prose says underperforming teachers should receive more weight,
    while the printed equation has the opposite sign.  This class follows the
    printed equation and exposes that identity in manifests produced by the
    runner.
    """

    equation = "beta_i=softmax(p_s(t,noise_i)-p_s(t-1,noise_i))"

    def __init__(self, count: int):
        if count <= 0:
            raise ValueError("count must be positive")
        self.count = count
        self.previous: torch.Tensor | None = None
        self.beta = torch.full((count,), 1.0 / count, dtype=torch.float64)

    def update(self, performance: Sequence[float]) -> torch.Tensor:
        current = torch.as_tensor(performance, dtype=torch.float64)
        if current.shape != (self.count,):
            raise ValueError(f"expected {self.count} performance values")
        if not torch.isfinite(current).all():
            raise ValueError("performance contains non-finite values")
        if ((current < 0) | (current > 1)).any():
            raise ValueError("performance values must be fractions in [0, 1]")
        if self.previous is not None:
            self.beta = torch.softmax(current - self.previous, dim=0)
        self.previous = current.clone()
        return self.beta.clone()

    def state_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "previous": None if self.previous is None else self.previous.tolist(),
            "beta": self.beta.tolist(),
            "equation": self.equation,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["count"]) != self.count:
            raise ValueError("balancer teacher count mismatch")
        previous = state.get("previous")
        self.previous = None if previous is None else torch.as_tensor(previous, dtype=torch.float64)
        self.beta = torch.as_tensor(state["beta"], dtype=torch.float64)
