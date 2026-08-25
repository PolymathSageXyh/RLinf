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

"""Supervised Rectified Flow and Improved MeanFlow objectives."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

PredictField = Callable[[Tensor, Tensor, Tensor | None], Any]


@dataclass(frozen=True)
class FlowObjectiveOutput:
    """Scalar objective loss and detached metrics for logging."""

    loss: Tensor
    metrics: dict[str, Tensor]


def _adaptive_weighted_mse(
    prediction: Tensor,
    target: Tensor,
    *,
    power: float,
    epsilon: float,
    element_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    error = torch.where(
        element_mask,
        prediction - target,
        torch.zeros_like(prediction),
    )
    squared_error = error.square()
    valid_count = element_mask.flatten(start_dim=1).sum(dim=-1).to(prediction.dtype)
    per_sample_error_sum = squared_error.flatten(start_dim=1).sum(dim=-1)
    per_sample_mse = per_sample_error_sum / valid_count
    denominator = (per_sample_mse + epsilon).pow(power).detach()
    total_valid_count = valid_count.sum()
    return (
        (valid_count * per_sample_mse / denominator).sum() / total_valid_count,
        per_sample_error_sum.sum() / total_valid_count,
        (valid_count * denominator).sum() / total_valid_count,
    )


def _resolve_predict_field(
    actor_or_predict_field: Any,
    obs: Any,
    *,
    update_stats: bool,
) -> PredictField:
    encode_condition = getattr(actor_or_predict_field, "encode_condition", None)
    predict_field = getattr(actor_or_predict_field, "predict_field", None)
    if callable(encode_condition) and callable(predict_field):
        # Encoding happens exactly once, outside any JVP closure.
        if (
            torch.is_tensor(obs)
            and obs.ndim == 3
            and obs.shape[1] == 1
            and obs.shape[-1] == getattr(actor_or_predict_field, "d_model", -1)
        ):
            condition = obs
        else:
            condition = encode_condition(obs, update_stats=update_stats)

        def predict_from_condition(
            flow_state: Tensor,
            t_from: Tensor,
            t_to: Tensor | None,
        ) -> Any:
            public_shape = flow_state.shape
            flat_state = flow_state.reshape(flow_state.shape[0], -1)
            value = _extract_field_value(
                predict_field(
                    condition,
                    flat_state,
                    t_from=t_from,
                    t_to=t_to,
                )
            )
            if value.shape != flat_state.shape:
                raise ValueError(
                    "FlowTActor field output must match flattened flow-state shape: "
                    f"got {tuple(value.shape)} and {tuple(flat_state.shape)}."
                )
            return value.reshape(public_shape)

        return predict_from_condition
    if callable(actor_or_predict_field):
        # A lightweight analytic test double may directly implement the same
        # resolved ``(flow_state, t_from, t_to)`` contract.
        return actor_or_predict_field
    else:
        raise TypeError(
            "actor_or_predict_field must expose encode_condition() and "
            "predict_field(), or be a resolved field callable"
        )


def _extract_field_value(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping):
        value = output.get("value", output.get("field", output.get("mean")))
    else:
        value = getattr(output, "value", None)
        if value is None:
            value = getattr(output, "field", getattr(output, "mean", None))
    if not isinstance(value, Tensor):
        raise TypeError(
            "predict_field must return a Tensor or an output containing a Tensor "
            "named 'value'"
        )
    return value


def _call_predict_field(
    predict_field: PredictField,
    flow_state: Tensor,
    t_from: Tensor,
    t_to: Tensor | None,
) -> Tensor:
    value = _extract_field_value(predict_field(flow_state, t_from, t_to))
    if value.shape != flow_state.shape:
        raise ValueError(
            "predict_field output must match the flow-state shape: "
            f"got {tuple(value.shape)} and {tuple(flow_state.shape)}"
        )
    return value


def _make_generator(
    reference: Tensor,
    generator: torch.Generator | None,
    seed: int | None,
) -> torch.Generator | None:
    if generator is not None and seed is not None:
        raise ValueError("Specify either generator or seed, not both.")
    if seed is None:
        return generator
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}.")
    seeded_generator = torch.Generator(device=reference.device)
    seeded_generator.manual_seed(seed)
    return seeded_generator


def _sample_like(reference: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _sample_batch_uniform(
    reference: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.rand(
        (reference.shape[0], 1),
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _normalize_time(
    value: Tensor | float,
    reference: Tensor,
    name: str,
) -> Tensor:
    time = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    batch_size = reference.shape[0]
    if time.ndim == 0:
        time = time.expand(batch_size).unsqueeze(-1)
    elif time.numel() == 1:
        time = time.reshape(1).expand(batch_size).unsqueeze(-1)
    elif time.numel() == batch_size:
        time = time.reshape(batch_size, 1)
    else:
        raise ValueError(
            f"{name} must be scalar or have one value per batch item; got "
            f"shape {tuple(time.shape)} for batch size {batch_size}."
        )
    if not torch.isfinite(time).all():
        raise ValueError(f"{name} must contain only finite values.")
    if torch.any((time < 0) | (time > 1)):
        raise ValueError(f"{name} must lie in [0, 1].")
    return time


def _time_for_flow_state(time: Tensor, flow_state: Tensor) -> Tensor:
    return time.reshape(time.shape[0], *([1] * (flow_state.ndim - 1)))


def _validate_action(action: Tensor) -> None:
    if not isinstance(action, Tensor):
        raise TypeError(f"action must be a Tensor, got {type(action).__name__}.")
    if action.ndim != 3:
        raise ValueError(
            "Flow BC action must have canonical shape [B, H, A], got "
            f"{tuple(action.shape)}."
        )
    if not action.is_floating_point():
        raise TypeError("action must use a floating-point dtype.")
    if not torch.isfinite(action).all():
        raise ValueError("action must contain only finite values.")


def _normalize_valid_mask(
    action: Tensor,
    valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Validate a prefix mask and expand it over per-step action coordinates."""

    if valid_mask is None:
        raise ValueError("Flow BC requires valid_mask with shape [B, H].")
    if not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a bool Tensor.")
    if valid_mask.dtype != torch.bool or valid_mask.ndim != 2:
        raise ValueError(
            "valid_mask must be a bool Tensor with shape [B, H], got "
            f"dtype={valid_mask.dtype}, shape={tuple(valid_mask.shape)}."
        )
    expected_shape = action.shape[:2]
    if tuple(valid_mask.shape) != expected_shape:
        raise ValueError(
            f"valid_mask must have shape {expected_shape}, got "
            f"{tuple(valid_mask.shape)}."
        )
    if torch.any(~valid_mask.any(dim=1)):
        raise ValueError("Each sample must contain at least one valid action step.")
    if valid_mask.shape[1] > 1 and torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1]):
        raise ValueError("valid_mask must be prefix-valid for every sample.")

    valid_mask = valid_mask.to(device=action.device)
    element_mask = valid_mask.unsqueeze(-1).expand_as(action)
    return valid_mask, element_mask


def _per_horizon_mse_metrics(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    """Report stable per-step MSEs without counting padded coordinates."""

    horizon = valid_mask.shape[1]
    if prediction.ndim != 3 or prediction.shape[1] != horizon:
        raise ValueError("Per-horizon metrics require [B, H, A].")

    squared_error = (prediction - target).square()
    metrics = {}
    action_dim = squared_error.shape[-1]
    for step in range(horizon):
        step_mask = valid_mask[:, step].unsqueeze(-1)
        numerator = torch.where(
            step_mask,
            squared_error[:, step],
            torch.zeros_like(squared_error[:, step]),
        ).sum()
        denominator = valid_mask[:, step].sum() * action_dim
        # A horizon with no valid samples is reported as zero. Keeping a stable
        # metric schema avoids distributed logger mismatches near episode tails.
        value = numerator / denominator.clamp_min(1)
        metrics[f"field_mse_h{step}"] = value.detach()
        metrics[f"field_count_h{step}"] = denominator.detach()
    return metrics


def _masked_rms(value: Tensor, element_mask: Tensor) -> Tensor:
    selected = torch.where(element_mask, value, torch.zeros_like(value))
    count = element_mask.sum().to(value.dtype)
    return torch.sqrt(selected.square().sum() / count)


def _prepare_noise(
    action: Tensor,
    noise: Tensor | None,
    generator: torch.Generator | None,
) -> Tensor:
    if noise is None:
        return _sample_like(action, generator)
    if not isinstance(noise, Tensor) or noise.shape != action.shape:
        shape = (
            tuple(noise.shape) if isinstance(noise, Tensor) else type(noise).__name__
        )
        raise ValueError(
            f"noise must match action shape {tuple(action.shape)}, got {shape}."
        )
    noise = noise.to(device=action.device, dtype=action.dtype)
    if not torch.isfinite(noise).all():
        raise ValueError("noise must contain only finite values.")
    return noise


class RectifiedFlowObjective(nn.Module):
    """RF behavior cloning on the noise-to-data path ``0 -> 1``."""

    def __init__(
        self,
        adaptive_power: float = 0.0,
        adaptive_epsilon: float = 0.01,
    ) -> None:
        super().__init__()
        if not math.isfinite(adaptive_power) or adaptive_power < 0:
            raise ValueError("adaptive_power must be finite and non-negative.")
        if not math.isfinite(adaptive_epsilon) or adaptive_epsilon <= 0:
            raise ValueError("adaptive_epsilon must be finite and positive.")
        self.adaptive_power = float(adaptive_power)
        self.adaptive_epsilon = float(adaptive_epsilon)

    def forward(
        self,
        actor_or_predict_field: Any,
        obs: Any,
        action: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | float | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        update_stats: bool = False,
        valid_mask: Tensor | None = None,
    ) -> FlowObjectiveOutput:
        _validate_action(action)
        valid_mask, element_mask = _normalize_valid_mask(action, valid_mask)
        generator = _make_generator(action, generator, seed)
        noise = _prepare_noise(action, noise, generator)
        time = (
            _sample_batch_uniform(action, generator)
            if time is None
            else _normalize_time(time, action, "time")
        )
        time_broadcast = _time_for_flow_state(time, action)
        flow_state = (1.0 - time_broadcast) * noise + time_broadcast * action
        target = action - noise
        prediction = _call_predict_field(
            _resolve_predict_field(
                actor_or_predict_field,
                obs,
                update_stats=update_stats,
            ),
            flow_state,
            time,
            None,
        )
        loss, field_mse, adaptive_denominator_mean = _adaptive_weighted_mse(
            prediction,
            target,
            power=self.adaptive_power,
            epsilon=self.adaptive_epsilon,
            element_mask=element_mask,
        )
        return FlowObjectiveOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "field_mse": field_mse.detach(),
                "valid_coordinate_count": element_mask.sum().detach(),
                "adaptive_denominator_mean": adaptive_denominator_mean.detach(),
                "t_mean": time.detach().mean(),
                "target_norm": _masked_rms(target.detach(), element_mask),
                "field_norm": _masked_rms(prediction.detach(), element_mask),
                **_per_horizon_mse_metrics(
                    prediction.detach(),
                    target.detach(),
                    valid_mask,
                ),
            },
        )


class ImprovedMeanFlowObjective(nn.Module):
    """Native iMF objective with the material derivative computed by JVP."""

    def __init__(
        self,
        boundary_pair_probability: float = 0.5,
        *,
        r_not_equal_t_ratio: float | None = None,
        logit_mean: float = -0.4,
        logit_std: float = 1.0,
        adaptive_power: float = 1.0,
        adaptive_epsilon: float = 0.01,
    ) -> None:
        super().__init__()
        if r_not_equal_t_ratio is not None:
            if boundary_pair_probability != 0.5:
                raise ValueError(
                    "Specify boundary_pair_probability or r_not_equal_t_ratio, "
                    "not both."
                )
            boundary_pair_probability = 1.0 - float(r_not_equal_t_ratio)
        if not 0.0 <= boundary_pair_probability <= 1.0:
            raise ValueError("boundary_pair_probability must lie in [0, 1].")
        if not math.isfinite(logit_mean):
            raise ValueError("logit_mean must be finite.")
        if not math.isfinite(logit_std) or logit_std <= 0:
            raise ValueError("logit_std must be finite and positive.")
        if not math.isfinite(adaptive_power) or adaptive_power < 0:
            raise ValueError("adaptive_power must be finite and non-negative.")
        if not math.isfinite(adaptive_epsilon) or adaptive_epsilon <= 0:
            raise ValueError("adaptive_epsilon must be finite and positive.")
        self.boundary_pair_probability = float(boundary_pair_probability)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.adaptive_power = float(adaptive_power)
        self.adaptive_epsilon = float(adaptive_epsilon)

    @property
    def r_not_equal_t_ratio(self) -> float:
        """Compatibility alias for the probability of a nonzero interval."""

        return 1.0 - self.boundary_pair_probability

    def _sample_time_pair(
        self,
        reference: Tensor,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        logits = torch.randn(
            (reference.shape[0], 2),
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        sampled_times = torch.sigmoid(self.logit_mean + self.logit_std * logits)
        t_from = sampled_times.max(dim=-1, keepdim=True).values
        t_to = sampled_times.min(dim=-1, keepdim=True).values
        if self.boundary_pair_probability == 1.0:
            return t_from, t_from.clone()
        if self.boundary_pair_probability != 0.0:
            use_boundary = (
                torch.rand(
                    (reference.shape[0], 1),
                    device=reference.device,
                    dtype=reference.dtype,
                    generator=generator,
                )
                < self.boundary_pair_probability
            )
            t_to = torch.where(use_boundary, t_from, t_to)
        return t_from, t_to

    def forward(
        self,
        actor_or_predict_field: Any,
        obs: Any,
        action: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | float | None = None,
        r_time: Tensor | float | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        update_stats: bool = False,
        valid_mask: Tensor | None = None,
    ) -> FlowObjectiveOutput:
        _validate_action(action)
        valid_mask, element_mask = _normalize_valid_mask(action, valid_mask)
        generator = _make_generator(action, generator, seed)
        noise = _prepare_noise(action, noise, generator)
        if time is None and r_time is None:
            time, r_time = self._sample_time_pair(action, generator)
        elif time is None or r_time is None:
            raise ValueError(
                "Explicit Improved MeanFlow times must provide both time "
                "(t_from) and r_time (t_to)."
            )
        else:
            time = _normalize_time(time, action, "time")
            r_time = _normalize_time(r_time, action, "r_time")
        if torch.any(r_time > time):
            raise ValueError("Improved MeanFlow requires r_time <= time.")

        time_broadcast = _time_for_flow_state(time, action)
        flow_state = (1.0 - time_broadcast) * action + time_broadcast * noise
        target = noise - action
        predict_field = _resolve_predict_field(
            actor_or_predict_field,
            obs,
            update_stats=update_stats,
        )

        # Boundary velocity is needed only by the supervised JVP objective.
        # Online ODE/noise/SDE sampling never substitutes this boundary field
        # for the interval field U(x_t, r, t).
        boundary_field = _call_predict_field(
            predict_field,
            flow_state,
            time,
            time,
        )

        def interval_field(
            state_arg: Tensor,
            r_arg: Tensor,
            time_arg: Tensor,
        ) -> Tensor:
            return _call_predict_field(
                predict_field,
                state_arg,
                time_arg,
                r_arg,
            )

        with sdpa_kernel(SDPBackend.MATH):
            # ``torch.autograd.functional.jvp`` defaults to
            # ``create_graph=False``, which would detach both the primal output
            # and the JVP. The loss needs the primal ``average_field`` graph to
            # update the actor, even though the material derivative is detached
            # below by the MeanFlow objective. Keep ``strict=False`` so a field
            # that is mathematically independent of one endpoint contributes a
            # zero directional derivative instead of raising.
            average_field, material_derivative = torch.autograd.functional.jvp(
                interval_field,
                (flow_state, r_time, time),
                (
                    boundary_field.detach(),
                    torch.zeros_like(r_time),
                    torch.ones_like(time),
                ),
                create_graph=True,
                strict=False,
            )
        time_gap = _time_for_flow_state(time - r_time, action)
        regression_field = average_field + time_gap * material_derivative.detach()
        loss, field_mse, adaptive_denominator_mean = _adaptive_weighted_mse(
            regression_field,
            target,
            power=self.adaptive_power,
            epsilon=self.adaptive_epsilon,
            element_mask=element_mask,
        )
        return FlowObjectiveOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "field_mse": field_mse.detach(),
                "valid_coordinate_count": element_mask.sum().detach(),
                "adaptive_denominator_mean": adaptive_denominator_mean.detach(),
                "t_mean": time.detach().mean(),
                "r_mean": r_time.detach().mean(),
                "time_gap_mean": (time - r_time).detach().mean(),
                "target_norm": _masked_rms(target.detach(), element_mask),
                "field_norm": _masked_rms(regression_field.detach(), element_mask),
                "boundary_field_norm": _masked_rms(
                    boundary_field.detach(), element_mask
                ),
                "material_derivative_norm": _masked_rms(
                    material_derivative.detach(), element_mask
                ),
                **_per_horizon_mse_metrics(
                    regression_field.detach(),
                    target.detach(),
                    valid_mask,
                ),
            },
        )


def build_flow_objective(name: str, **kwargs: Any) -> nn.Module:
    """Build a supervised objective from its YAML-facing name."""

    if not isinstance(name, str):
        raise TypeError(
            f"Flow objective name must be a str, got {type(name).__name__}."
        )
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"rectified_flow", "rf"}:
        return RectifiedFlowObjective(**kwargs)
    if normalized in {"improved_meanflow", "improved_mean_flow", "imf"}:
        return ImprovedMeanFlowObjective(**kwargs)
    raise ValueError(
        f"Unknown flow objective {name!r}; expected 'rectified_flow' or "
        "'improved_meanflow'."
    )


__all__ = [
    "FlowObjectiveOutput",
    "ImprovedMeanFlowObjective",
    "RectifiedFlowObjective",
    "build_flow_objective",
]
