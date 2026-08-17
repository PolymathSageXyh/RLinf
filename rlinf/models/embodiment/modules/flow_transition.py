# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure PyTorch transition kernels for Flow-T policies.

The functions in this module operate on already-predicted fields. They do not
depend on a policy implementation, OpenPI, JAX, or Flax. In particular,
Improved MeanFlow transitions always consume the interval-average field
``U(x_t, t_to, t_from)`` for the exact interval being integrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

RECTIFIED_FLOW = "rectified_flow"
IMPROVED_MEANFLOW = "improved_meanflow"
FLOW_ODE = "flow_ode"
FLOW_NOISE = "flow_noise"
FLOW_SDE = "flow_sde"


@dataclass(frozen=True)
class FlowTransitionMoments:
    """Mean and standard deviation of one next-flow-state transition."""

    x_mean: torch.Tensor
    x_std: torch.Tensor


def normalize_flow_objective(objective: str) -> str:
    """Return the canonical objective name."""

    aliases = {
        "rf": RECTIFIED_FLOW,
        "rectified_flow": RECTIFIED_FLOW,
        "imf": IMPROVED_MEANFLOW,
        "improved_mean_flow": IMPROVED_MEANFLOW,
        "improved_meanflow": IMPROVED_MEANFLOW,
    }
    try:
        return aliases[objective.strip().lower().replace("-", "_")]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "flow objective must be 'rectified_flow' or "
            f"'improved_meanflow', got {objective!r}."
        ) from exc


def normalize_sampler_method(method: str) -> str:
    """Return a validated flow sampler method."""

    try:
        normalized = method.strip().lower().replace("-", "_")
    except AttributeError as exc:
        raise ValueError(
            f"flow sampler method must be a string, got {method!r}."
        ) from exc
    valid_methods = {FLOW_ODE, FLOW_NOISE, FLOW_SDE}
    if normalized not in valid_methods:
        raise ValueError(
            "flow sampler method must be 'flow_ode', 'flow_noise', or "
            f"'flow_sde', got {method!r}."
        )
    return normalized


def as_batch_time(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Normalize a scalar-like time to one value per batch element."""

    if torch.is_tensor(value):
        time = value.to(device=reference.device, dtype=reference.dtype)
    else:
        time = torch.as_tensor(
            value,
            device=reference.device,
            dtype=reference.dtype,
        )
    batch_size = reference.shape[0]
    if time.numel() == 1:
        time = time.reshape(1).expand(batch_size)
    elif time.numel() == batch_size:
        time = time.reshape(batch_size)
    else:
        raise ValueError(
            f"{name} must contain one value or one per batch item; got shape "
            f"{tuple(time.shape)} for batch size {batch_size}."
        )
    if not torch.isfinite(time).all():
        raise ValueError(f"{name} must contain only finite values.")
    return time


def expand_time_like(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[batch]`` time tensor across the flow-state dimensions."""

    return time.reshape(time.shape[0], *([1] * (reference.ndim - 1)))


def rectified_flow_ode_moments(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t_from: float | torch.Tensor,
    t_to: float | torch.Tensor,
) -> FlowTransitionMoments:
    """Euler-integrate an instantaneous Rectified Flow field from 0 toward 1."""

    _validate_matching_field(x_t, velocity)
    from_time = as_batch_time(t_from, x_t, name="t_from")
    to_time = as_batch_time(t_to, x_t, name="t_to")
    dt = to_time - from_time
    if torch.any(from_time < 0) or torch.any(to_time > 1) or torch.any(dt < 0):
        raise ValueError("Rectified Flow requires 0 <= t_from <= t_to <= 1.")
    x_mean = x_t + expand_time_like(dt, x_t) * velocity
    return FlowTransitionMoments(x_mean=x_mean, x_std=torch.zeros_like(x_mean))


def improved_meanflow_ode_moments(
    x_t: torch.Tensor,
    average_field: torch.Tensor,
    t_from: float | torch.Tensor,
    t_to: float | torch.Tensor,
) -> FlowTransitionMoments:
    """Integrate native iMF with the interval field ``U(t_from, t_to)``.

    iMF uses noise at time 1 and data at time 0. The update is therefore
    ``x_to = x_from + (t_to - t_from) * U(x_from, t_to, t_from)``.
    """

    _validate_matching_field(x_t, average_field)
    from_time = as_batch_time(t_from, x_t, name="t_from")
    to_time = as_batch_time(t_to, x_t, name="t_to")
    dt = to_time - from_time
    if torch.any(to_time < 0) or torch.any(from_time > 1) or torch.any(dt > 0):
        raise ValueError("Improved MeanFlow requires 0 <= t_to <= t_from <= 1.")
    x_mean = x_t + expand_time_like(dt, x_t) * average_field
    return FlowTransitionMoments(x_mean=x_mean, x_std=torch.zeros_like(x_mean))


def bounded_next_state_std(
    raw_log_std: torch.Tensor,
    log_std_min: float,
    log_std_max: float,
) -> torch.Tensor:
    """Map an unconstrained noise-head output to next-state standard deviation."""

    if not math.isfinite(log_std_min) or not math.isfinite(log_std_max):
        raise ValueError("next-state log-std bounds must be finite.")
    if log_std_min > log_std_max:
        raise ValueError(
            "log_std_min must not exceed log_std_max, got "
            f"{log_std_min} > {log_std_max}."
        )
    normalized = torch.tanh(raw_log_std)
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (normalized + 1.0)
    return torch.exp(log_std)


def rectified_flow_sde_moments(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t_from: float | torch.Tensor,
    t_to: float | torch.Tensor,
    *,
    noise_level: float = 0.1,
) -> FlowTransitionMoments:
    """Compute the SACFlow corrected-drift RF-SDE transition moments."""

    _validate_matching_field(x_t, velocity)
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError(
            f"RF-SDE noise_level must be finite and non-negative, got {noise_level}."
        )
    from_time = as_batch_time(t_from, x_t, name="t_from")
    to_time = as_batch_time(t_to, x_t, name="t_to")
    dt = to_time - from_time
    if torch.any(from_time < 0) or torch.any(to_time > 1) or torch.any(dt < 0):
        raise ValueError("RF-SDE requires 0 <= t_from <= t_to <= 1.")

    t_expanded = expand_time_like(from_time, x_t)
    dt_expanded = expand_time_like(dt, x_t)
    sigma = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigma_sq = sigma.square()
    denominator = (1.0 - t_expanded).clamp_min(torch.finfo(x_t.dtype).eps)
    corrected_drift = (
        1.0 + t_expanded * sigma_sq / (2.0 * denominator)
    ) * velocity - sigma_sq * x_t / (2.0 * denominator)
    corrected_drift = torch.where(t_expanded == 0, velocity, corrected_drift)

    x_mean = x_t + dt_expanded * corrected_drift
    x_std = sigma * torch.sqrt(dt_expanded).expand_as(x_t)
    return FlowTransitionMoments(x_mean=x_mean, x_std=x_std)


def improved_meanflow_sde_moments(
    x_t: torch.Tensor,
    interval_field: torch.Tensor,
    t_from: float | torch.Tensor,
    t_to: float | torch.Tensor,
    *,
    noise_level: float = 0.1,
    std_min: float = 0.01,
    std_max: float = 0.1,
    safe_initial_time: float = 0.99,
) -> FlowTransitionMoments:
    """Compute OpenPI-compatible native-iMF Flow-SDE moments.

    This is a local PyTorch implementation of the established formula. It has
    no runtime dependency on OpenPI. Crucially, ``interval_field`` is the same
    ``U(x_t, t_to, t_from)`` used by the deterministic interval update; this
    function never substitutes a boundary field ``U(t_from, t_from)``.
    """

    _validate_matching_field(x_t, interval_field)
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError(
            f"iMF-SDE noise_level must be finite and non-negative, got {noise_level}."
        )
    if not math.isfinite(safe_initial_time) or not 0.0 < safe_initial_time < 1.0:
        raise ValueError(
            "iMF-SDE safe_initial_time must be finite and in (0, 1), got "
            f"{safe_initial_time}."
        )
    if (
        not math.isfinite(std_min)
        or not math.isfinite(std_max)
        or std_min < 0
        or std_min > std_max
    ):
        raise ValueError(
            "iMF-SDE std bounds must satisfy 0 <= std_min <= std_max, got "
            f"({std_min}, {std_max})."
        )

    from_time = as_batch_time(t_from, x_t, name="t_from")
    to_time = as_batch_time(t_to, x_t, name="t_to")
    if (
        torch.any(to_time < 0)
        or torch.any(from_time > 1)
        or torch.any(to_time > from_time)
    ):
        raise ValueError("iMF-SDE requires 0 <= t_to <= t_from <= 1.")

    # The logarithmic endpoint calculation is intentionally performed in FP32
    # even when the actor uses fp16/bf16 mixed precision.
    from_time_fp32 = from_time.float()
    to_time_fp32 = to_time.float()
    safe_from_time = torch.where(
        from_time_fp32 >= 1.0,
        torch.full_like(from_time_fp32, safe_initial_time),
        from_time_fp32,
    )
    midpoint = (from_time_fp32 + to_time_fp32) / 2.0
    time_difference = from_time_fp32 - to_time_fp32
    log_difference = torch.log1p(-to_time_fp32) - torch.log1p(-safe_from_time)
    midpoint_log_difference = torch.log1p(-to_time_fp32) - torch.log1p(-midpoint)

    sigma = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigma_sq = sigma.square()
    log_difference = expand_time_like(log_difference, x_t).to(x_t.dtype)
    midpoint_log_difference = expand_time_like(midpoint_log_difference, x_t).to(
        x_t.dtype
    )
    time_difference = expand_time_like(time_difference, x_t).to(x_t.dtype)

    x_mean = x_t * (1.0 - sigma_sq / 2.0 * log_difference) - (
        time_difference
        * interval_field
        * (1.0 + sigma_sq / 2.0 * (1.0 - midpoint_log_difference))
    )
    x_std = torch.sqrt(
        (sigma_sq * (log_difference - time_difference)).clamp_min(0.0)
    ).expand_as(x_t)
    x_std = x_std.clamp(min=std_min, max=std_max)
    return FlowTransitionMoments(x_mean=x_mean, x_std=x_std)


def _validate_matching_field(x_t: torch.Tensor, field: torch.Tensor) -> None:
    if x_t.shape != field.shape:
        raise ValueError(
            "flow field must match flow-state shape, got "
            f"{tuple(field.shape)} and {tuple(x_t.shape)}."
        )


__all__ = [
    "FLOW_NOISE",
    "FLOW_ODE",
    "FLOW_SDE",
    "IMPROVED_MEANFLOW",
    "RECTIFIED_FLOW",
    "FlowTransitionMoments",
    "as_batch_time",
    "bounded_next_state_std",
    "expand_time_like",
    "improved_meanflow_ode_moments",
    "improved_meanflow_sde_moments",
    "normalize_flow_objective",
    "normalize_sampler_method",
    "rectified_flow_ode_moments",
    "rectified_flow_sde_moments",
]
