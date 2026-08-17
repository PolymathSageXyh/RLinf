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
) -> tuple[Tensor, Tensor, Tensor]:
    per_sample_mse = (prediction - target).square().flatten(start_dim=1).mean(dim=-1)
    denominator = (per_sample_mse + epsilon).pow(power).detach()
    return (
        (per_sample_mse / denominator).mean(),
        per_sample_mse.mean(),
        denominator.mean(),
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
            return predict_field(
                condition,
                flow_state,
                t_from=t_from,
                t_to=t_to,
            )

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
    if action.ndim < 2:
        raise ValueError("action must contain batch and action dimensions.")
    if not action.is_floating_point():
        raise TypeError("action must use a floating-point dtype.")


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
    return noise.to(device=action.device, dtype=action.dtype)


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
    ) -> FlowObjectiveOutput:
        _validate_action(action)
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
        )
        return FlowObjectiveOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "field_mse": field_mse.detach(),
                "adaptive_denominator_mean": adaptive_denominator_mean.detach(),
                "t_mean": time.detach().mean(),
                "target_norm": target.detach().flatten(start_dim=1).norm(dim=-1).mean(),
                "field_norm": (
                    prediction.detach().flatten(start_dim=1).norm(dim=-1).mean()
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
    ) -> FlowObjectiveOutput:
        _validate_action(action)
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
        )
        return FlowObjectiveOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "field_mse": field_mse.detach(),
                "adaptive_denominator_mean": adaptive_denominator_mean.detach(),
                "t_mean": time.detach().mean(),
                "r_mean": r_time.detach().mean(),
                "time_gap_mean": (time - r_time).detach().mean(),
                "target_norm": target.detach().flatten(start_dim=1).norm(dim=-1).mean(),
                "field_norm": (
                    regression_field.detach().flatten(start_dim=1).norm(dim=-1).mean()
                ),
                "boundary_field_norm": (
                    boundary_field.detach().flatten(start_dim=1).norm(dim=-1).mean()
                ),
                "material_derivative_norm": (
                    material_derivative.detach()
                    .flatten(start_dim=1)
                    .norm(dim=-1)
                    .mean()
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
