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

"""Typed sampling helpers for differentiable Flow-T paths."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .flow_transition import FLOW_ODE, IMPROVED_MEANFLOW, normalize_flow_objective


@dataclass(frozen=True)
class FlowTransitionSample:
    """One next-flow-state sample and the noise that produced it."""

    x_next: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    log_prob: torch.Tensor | None
    noise: torch.Tensor


@dataclass(frozen=True)
class FlowNoiseTrace:
    """Common random numbers used to sample one complete flow path."""

    initial_noise: torch.Tensor
    step_noise: tuple[torch.Tensor, ...]

    def detach(self) -> "FlowNoiseTrace":
        """Return a trace disconnected from its sampling graph."""

        return FlowNoiseTrace(
            initial_noise=self.initial_noise.detach(),
            step_noise=tuple(noise.detach() for noise in self.step_noise),
        )


@dataclass(frozen=True)
class FlowPathTrace:
    """Fixed Markov-chain states and transition metadata for score gradients."""

    states: tuple[torch.Tensor, ...]
    means: tuple[torch.Tensor, ...]
    stds: tuple[torch.Tensor, ...]
    times: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def detach(self) -> "FlowPathTrace":
        """Return a fixed trace suitable for likelihood recomputation."""

        return FlowPathTrace(
            states=tuple(state.detach() for state in self.states),
            means=tuple(mean.detach() for mean in self.means),
            stds=tuple(std.detach() for std in self.stds),
            times=tuple(
                (t_from.detach(), t_to.detach()) for t_from, t_to in self.times
            ),
        )


@dataclass(frozen=True)
class FlowSample:
    """Public sample consumed by SACFlow rollout and actor/alpha updates."""

    action: torch.Tensor
    latent_action: torch.Tensor
    path_trace: FlowPathTrace
    path_log_prob: torch.Tensor | None
    temperature_log_prob: torch.Tensor | None
    actor_entropy_surrogate: torch.Tensor | None

    @property
    def trace(self) -> FlowPathTrace:
        """Short compatibility alias for the fixed Markov path trace."""

        return self.path_trace


@dataclass(frozen=True)
class FlowPathSample(FlowSample):
    """A public ``FlowSample`` with stacked compatibility/debug traces."""

    states: torch.Tensor
    timesteps: torch.Tensor
    x_means: torch.Tensor
    x_stds: torch.Tensor
    step_log_probs: torch.Tensor | None
    initial_noise: torch.Tensor
    step_noises: torch.Tensor
    noise_trace: FlowNoiseTrace

    @property
    def pre_tanh_action(self) -> torch.Tensor:
        """Compatibility alias for the public ``latent_action`` field."""

        return self.latent_action

    @property
    def joint_path_log_prob(self) -> torch.Tensor | None:
        """Compatibility alias for the public ``path_log_prob`` field."""

        return self.path_log_prob

    def detach_trace(self) -> FlowPathTrace:
        """Return the Markov path as fixed data for score-gradient replay."""

        return self.path_trace.detach()


def build_time_schedule(
    objective: str,
    num_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a native RF ``0->1`` or iMF ``1->0`` K-step schedule."""

    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}.")
    objective = normalize_flow_objective(objective)
    if objective == IMPROVED_MEANFLOW:
        start, end = 1.0, 0.0
    else:
        start, end = 0.0, 1.0
    return torch.linspace(
        start,
        end,
        num_steps + 1,
        device=device,
        dtype=dtype,
    )


def sample_normal_transition(
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
) -> FlowTransitionSample:
    """Reparameterize ``Normal(x_mean, x_std)`` using elementwise noise."""

    if x_mean.shape != x_std.shape:
        raise ValueError(
            f"x_mean and x_std shapes differ: {x_mean.shape} and {x_std.shape}."
        )
    if not torch.isfinite(x_mean).all() or not torch.isfinite(x_std).all():
        raise ValueError("stochastic transition moments must be finite.")
    if torch.any(x_std <= 0):
        raise ValueError("a stochastic transition requires x_std > 0.")
    if noise is None:
        noise = torch.randn_like(x_mean)
    else:
        if noise.shape != x_mean.shape:
            raise ValueError(
                f"transition noise must have shape {tuple(x_mean.shape)}, got "
                f"{tuple(noise.shape)}."
            )
        noise = noise.to(device=x_mean.device, dtype=x_mean.dtype)
    if not torch.isfinite(noise).all():
        raise ValueError("transition noise must contain only finite values.")

    x_next = x_mean + x_std * noise
    distribution = Normal(x_mean, x_std)
    log_prob = (
        distribution.log_prob(x_next).flatten(start_dim=1).sum(dim=1, keepdim=True)
    )
    return FlowTransitionSample(
        x_next=x_next,
        x_mean=x_mean,
        x_std=x_std,
        log_prob=log_prob,
        noise=noise,
    )


def deterministic_transition(
    x_mean: torch.Tensor,
) -> FlowTransitionSample:
    """Return a singular deterministic transition with no likelihood."""

    return FlowTransitionSample(
        x_next=x_mean,
        x_mean=x_mean,
        x_std=torch.zeros_like(x_mean),
        log_prob=None,
        noise=torch.zeros_like(x_mean),
    )


def normalize_step_noises(
    step_noises: Sequence[torch.Tensor] | torch.Tensor | None,
    *,
    num_steps: int,
    reference: torch.Tensor,
    stochastic: bool,
) -> tuple[torch.Tensor | None, ...]:
    """Validate optional common-random-number noise for all K transitions."""

    if step_noises is None:
        return (None,) * num_steps
    if not stochastic:
        raise ValueError("step_noises are valid only for a stochastic flow sampler.")
    if torch.is_tensor(step_noises):
        expected_shape = (reference.shape[0], num_steps, *reference.shape[1:])
        if tuple(step_noises.shape) != expected_shape:
            raise ValueError(
                f"step_noises tensor must have shape {expected_shape}, got "
                f"{tuple(step_noises.shape)}."
            )
        sequence: Sequence[torch.Tensor] = step_noises.unbind(dim=1)
    else:
        sequence = tuple(step_noises)
        if len(sequence) != num_steps:
            raise ValueError(
                f"step_noises must contain {num_steps} tensors, got {len(sequence)}."
            )

    normalized = []
    for index, noise in enumerate(sequence):
        if not torch.is_tensor(noise) or noise.shape != reference.shape:
            shape = (
                tuple(noise.shape) if torch.is_tensor(noise) else type(noise).__name__
            )
            raise ValueError(
                f"step_noises[{index}] must have shape {tuple(reference.shape)}, "
                f"got {shape}."
            )
        normalized.append(noise.to(device=reference.device, dtype=reference.dtype))
        if not torch.isfinite(normalized[-1]).all():
            raise ValueError(f"step_noises[{index}] must contain only finite values.")
    return tuple(normalized)


def squash_action(
    pre_tanh_action: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply tanh/affine action mapping and return its log absolute Jacobian."""

    scale = action_scale.to(
        device=pre_tanh_action.device,
        dtype=pre_tanh_action.dtype,
    )
    bias = action_bias.to(
        device=pre_tanh_action.device,
        dtype=pre_tanh_action.dtype,
    )
    if torch.any(scale == 0):
        raise ValueError("action_scale must be non-zero for path likelihoods.")
    action = torch.tanh(pre_tanh_action) * scale + bias
    log_tanh_jacobian = 2.0 * (
        math.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action)
    )
    log_scale_jacobian = torch.log(scale.abs())
    log_abs_jacobian = (
        (log_tanh_jacobian + log_scale_jacobian)
        .flatten(start_dim=1)
        .sum(dim=1, keepdim=True)
    )
    return action, log_abs_jacobian


def initial_normal_log_prob(initial_noise: torch.Tensor) -> torch.Tensor:
    """Compute the standard-normal prior term for a latent path."""

    return (
        Normal(torch.zeros_like(initial_noise), torch.ones_like(initial_noise))
        .log_prob(initial_noise)
        .flatten(start_dim=1)
        .sum(dim=1, keepdim=True)
    )


def assemble_flow_path_sample(
    *,
    states: list[torch.Tensor],
    timesteps: torch.Tensor,
    transitions: list[FlowTransitionSample],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    sampler_method: str,
) -> FlowPathSample:
    """Assemble stacked traces and the stochastic joint path log-probability."""

    pre_tanh_action = states[-1]
    action, log_abs_jacobian = squash_action(
        pre_tanh_action,
        action_scale,
        action_bias,
    )
    stochastic = sampler_method != FLOW_ODE
    if stochastic:
        step_log_probs = torch.stack(
            [transition.log_prob for transition in transitions],
            dim=1,
        )
        joint_path_log_prob = (
            initial_normal_log_prob(states[0])
            + step_log_probs.sum(dim=1)
            - log_abs_jacobian
        )
    else:
        step_log_probs = None
        joint_path_log_prob = None

    path_trace = FlowPathTrace(
        states=tuple(states),
        means=tuple(transition.x_mean for transition in transitions),
        stds=tuple(transition.x_std for transition in transitions),
        times=tuple(
            (timesteps[index], timesteps[index + 1])
            for index in range(len(transitions))
        ),
    )
    noise_trace = FlowNoiseTrace(
        initial_noise=states[0],
        step_noise=tuple(transition.noise for transition in transitions),
    )
    return FlowPathSample(
        action=action,
        latent_action=pre_tanh_action,
        path_trace=path_trace,
        path_log_prob=joint_path_log_prob,
        temperature_log_prob=(
            None if joint_path_log_prob is None else joint_path_log_prob.detach()
        ),
        actor_entropy_surrogate=None,
        states=torch.stack(states, dim=1),
        timesteps=timesteps,
        x_means=torch.stack(
            [transition.x_mean for transition in transitions],
            dim=1,
        ),
        x_stds=torch.stack(
            [transition.x_std for transition in transitions],
            dim=1,
        ),
        step_log_probs=step_log_probs,
        initial_noise=states[0],
        step_noises=torch.stack(
            [transition.noise for transition in transitions],
            dim=1,
        ),
        noise_trace=noise_trace,
    )


__all__ = [
    "FlowNoiseTrace",
    "FlowPathSample",
    "FlowPathTrace",
    "FlowSample",
    "FlowTransitionSample",
    "assemble_flow_path_sample",
    "build_time_schedule",
    "deterministic_transition",
    "initial_normal_log_prob",
    "normalize_step_noises",
    "sample_normal_transition",
    "squash_action",
]
