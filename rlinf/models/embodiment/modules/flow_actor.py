# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .batch_renorm import BatchRenorm
from .explore_noise_net import ExploreNoiseNet
from .flow_sampler import (
    FlowNoiseTrace,
    FlowPathSample,
    FlowPathTrace,
    FlowTransitionSample,
    assemble_flow_path_sample,
    build_time_schedule,
    deterministic_transition,
    initial_normal_log_prob,
    normalize_step_noises,
    sample_normal_transition,
    squash_action,
)
from .flow_transition import (
    FLOW_NOISE,
    FLOW_ODE,
    FLOW_SDE,
    IMPROVED_MEANFLOW,
    RECTIFIED_FLOW,
    FlowTransitionMoments,
    as_batch_time,
    improved_meanflow_ode_moments,
    improved_meanflow_sde_moments,
    normalize_flow_objective,
    normalize_sampler_method,
    rectified_flow_ode_moments,
    rectified_flow_sde_moments,
)


@dataclass(frozen=True)
class FlowFieldOutput:
    """A predicted flow field and its shared transformer representation."""

    value: torch.Tensor
    hidden: torch.Tensor
    field_kind: Literal["instantaneous_velocity", "interval_average"]

    @property
    def field(self) -> torch.Tensor:
        """Compatibility alias for callers that name the output ``field``."""

        return self.value

    @property
    def mean(self) -> torch.Tensor:
        """Compatibility alias for legacy velocity terminology."""

        return self.value


class FlowTActor(nn.Module):
    """
    Transformer-based Flow Matching Actor for SAC
    Uses transformer architecture with cross-attention between action and observation
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model=64,
        n_head=4,
        n_layers=2,
        denoising_steps=4,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        action_scale=None,
        action_bias=None,
        objective="rectified_flow",
        sampler_method="flow_ode",
        add_flow_noise_head=None,
        noise_level=0.1,
        x_log_std_min=-5.0,
        x_log_std_max=-2.0,
        sde_std_min=0.01,
        sde_std_max=0.1,
        safe_initial_time=0.99,
        action_horizon=1,
    ):
        super().__init__()
        if (
            not isinstance(action_horizon, int)
            or isinstance(action_horizon, bool)
            or action_horizon <= 0
        ):
            raise ValueError("action_horizon must be a positive integer.")

        self.objective = normalize_flow_objective(objective)
        self.flow_objective = self.objective
        self.sampler_method = normalize_sampler_method(sampler_method)
        if add_flow_noise_head is None:
            add_flow_noise_head = self.sampler_method == FLOW_NOISE
        self.add_flow_noise_head = bool(add_flow_noise_head)
        if (
            not isinstance(denoising_steps, int)
            or isinstance(denoising_steps, bool)
            or denoising_steps <= 0
        ):
            raise ValueError("denoising_steps must be a positive integer.")
        if (
            not math.isfinite(x_log_std_min)
            or not math.isfinite(x_log_std_max)
            or x_log_std_min > x_log_std_max
        ):
            raise ValueError("x log-std bounds must satisfy finite min <= max.")
        if (
            not math.isfinite(sde_std_min)
            or not math.isfinite(sde_std_max)
            or sde_std_min < 0
            or sde_std_min > sde_std_max
        ):
            raise ValueError("SDE std bounds must satisfy 0 <= min <= max.")
        if not math.isfinite(noise_level) or noise_level < 0:
            raise ValueError("noise_level must be finite and non-negative.")
        if self.sampler_method == FLOW_SDE and noise_level <= 0:
            raise ValueError("flow_sde requires noise_level > 0.")
        if not math.isfinite(safe_initial_time) or not 0 < safe_initial_time < 1:
            raise ValueError("safe_initial_time must be finite and in (0, 1).")
        self.noise_level = float(noise_level)
        self.x_log_std_min = float(x_log_std_min)
        self.x_log_std_max = float(x_log_std_max)
        self.sde_std_min = float(sde_std_min)
        self.sde_std_max = float(sde_std_max)
        self.safe_initial_time = float(safe_initial_time)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flow_state_dim = self.action_dim * self.action_horizon
        if d_model < self.flow_state_dim:
            raise ValueError(
                "Chunked FlowTActor requires d_model >= H * action_dim; got "
                f"d_model={d_model}, H={self.action_horizon}, "
                f"action_dim={self.action_dim}."
            )
        self.denoising_steps = denoising_steps
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.log_std_min = -5
        self.log_std_max = 2
        self.use_batch_norm = use_batch_norm

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model // 2),
            nn.SiLU(),  # SiLU is PyTorch's Swish/silu
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # A complete chunk is one joint flow state, not an autoregressive token
        # sequence.
        self.action_proj = nn.Linear(self.flow_state_dim, self.d_model)

        # Time embedding (projects 1 -> d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.SiLU(),
            nn.Linear(self.d_model // 4, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Native iMF conditions U(x_t, t_to, t_from) on two independent time
        # encoders.
        if self.objective == IMPROVED_MEANFLOW:
            self.time_to_embedding = nn.Sequential(
                nn.Linear(1, self.d_model // 4),
                nn.SiLU(),
                nn.Linear(self.d_model // 4, self.d_model // 2),
                nn.SiLU(),
                nn.Linear(self.d_model // 2, self.d_model),
            )
            self.time_fusion = nn.Linear(self.d_model * 2, self.d_model)

        # Transformer decoder layers
        # We use nn.TransformerDecoderLayer which includes self-attn, cross-attn, and FFN
        decoder_layers = []
        for _ in range(self.n_layers):
            decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
            )
        self.transformer_layers = nn.ModuleList(decoder_layers)

        # Velocity output heads
        self.velocity_mean_head = nn.Linear(self.d_model, self.flow_state_dim)
        if self.add_flow_noise_head:
            self.flow_noise_head = ExploreNoiseNet(
                in_dim=self.d_model,
                out_dim=self.flow_state_dim,
                hidden_dims=[],
                activation_type="silu",
                noise_logvar_range=[
                    math.exp(self.x_log_std_min),
                    math.exp(self.x_log_std_max),
                ],
                noise_scheduler_type="learned",
            )

        if self.use_batch_norm:
            self.bn_obs = BatchRenorm(self.obs_dim, momentum=batch_norm_momentum)
            self.bn_action = BatchRenorm(
                self.flow_state_dim, momentum=batch_norm_momentum
            )

        # --- Action Scaling ---
        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            # Default to [-1, 1] range
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

        self._init_weights()
        if self.add_flow_noise_head:
            # This head is intentionally absent from BC artifacts. A neutral,
            # deterministic initialization keeps the live actor and its frozen
            # anchor identical on every distributed rank before FSDP wrapping,
            # while still allowing both weight and bias gradients to train it.
            for parameter in self.flow_noise_head.mlp_logvar.parameters():
                nn.init.zeros_(parameter)

        self.grad_norms = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _expanded_action_affine(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def expand_per_action(value: torch.Tensor, name: str) -> torch.Tensor:
            value = value.to(device=reference.device, dtype=reference.dtype).reshape(-1)
            if value.numel() == 1:
                value = value.expand(self.action_dim)
            elif value.numel() != self.action_dim:
                raise ValueError(
                    f"{name} must be scalar or have {self.action_dim} entries, "
                    f"got {value.numel()}."
                )
            return value.repeat(self.action_horizon)

        return (
            expand_per_action(self.action_scale, "action_scale"),
            expand_per_action(self.action_bias, "action_bias"),
        )

    def _flatten_public_flow_state(
        self,
        value: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        expected = (value.shape[0], self.action_horizon, self.action_dim)
        if value.ndim != 3 or tuple(value.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(value.shape)}."
            )
        return value.reshape(value.shape[0], self.flow_state_dim)

    def _public_flow_state(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != self.flow_state_dim:
            raise ValueError(
                "Internal Flow-T state must have shape [B, H*A], got "
                f"{tuple(value.shape)}."
            )
        return value.reshape(value.shape[0], self.action_horizon, self.action_dim)

    def _flatten_path_trace(self, trace: FlowPathTrace) -> FlowPathTrace:
        def flatten(value: torch.Tensor, name: str) -> torch.Tensor:
            if value.ndim == 2 and value.shape[1] == self.flow_state_dim:
                return value
            return self._flatten_public_flow_state(value, name=name)

        return FlowPathTrace(
            states=tuple(
                flatten(value, f"path_trace.states[{index}]")
                for index, value in enumerate(trace.states)
            ),
            means=tuple(
                flatten(value, f"path_trace.means[{index}]")
                for index, value in enumerate(trace.means)
            ),
            stds=tuple(
                flatten(value, f"path_trace.stds[{index}]")
                for index, value in enumerate(trace.stds)
            ),
            times=trace.times,
        )

    def _publicize_sample(self, sample: FlowPathSample) -> FlowPathSample:
        """Expose all action-valued sampling outputs with explicit H and A axes."""

        def public_tuple(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            return tuple(self._public_flow_state(value) for value in values)

        public_trace = FlowPathTrace(
            states=public_tuple(sample.path_trace.states),
            means=public_tuple(sample.path_trace.means),
            stds=public_tuple(sample.path_trace.stds),
            times=sample.path_trace.times,
        )
        public_noise_trace = FlowNoiseTrace(
            initial_noise=self._public_flow_state(sample.noise_trace.initial_noise),
            step_noise=public_tuple(sample.noise_trace.step_noise),
        )
        batch_size = sample.action.shape[0]
        return replace(
            sample,
            action=self._public_flow_state(sample.action),
            latent_action=self._public_flow_state(sample.latent_action),
            path_trace=public_trace,
            states=sample.states.reshape(
                batch_size, -1, self.action_horizon, self.action_dim
            ),
            x_means=sample.x_means.reshape(
                batch_size, -1, self.action_horizon, self.action_dim
            ),
            x_stds=sample.x_stds.reshape(
                batch_size, -1, self.action_horizon, self.action_dim
            ),
            initial_noise=self._public_flow_state(sample.initial_noise),
            step_noises=sample.step_noises.reshape(
                batch_size, -1, self.action_horizon, self.action_dim
            ),
            noise_trace=public_noise_trace,
        )

    def encode_condition(
        self,
        obs: torch.Tensor,
        *,
        update_stats: bool,
    ) -> torch.Tensor:
        """Encode observations once for field evaluation or path sampling.

        ``update_stats`` controls the existing BatchRenorm behavior. Improved
        MeanFlow objectives call this once before entering
        ``torch.autograd.functional.jvp``; the JVP closure therefore contains no
        observation encoding or mutable running-stat updates.
        """

        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                "FlowTActor expects obs with shape [B, obs_dim], got "
                f"{tuple(obs.shape)} for obs_dim={self.obs_dim}."
            )
        if self.use_batch_norm:
            obs = self.bn_obs(obs, update_stats)
        return self.obs_encoder(obs).unsqueeze(1)

    def _format_time_embedding_input(
        self,
        time: float | torch.Tensor,
        reference: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        batch_time = as_batch_time(time, reference, name=name)
        return batch_time.reshape(batch_time.shape[0], 1, 1)

    def _embed_action_time(
        self,
        flow_state: torch.Tensor,
        *,
        t_from: float | torch.Tensor,
        t_to: float | torch.Tensor | None,
    ) -> torch.Tensor:
        if flow_state.ndim != 2 or flow_state.shape[-1] != self.flow_state_dim:
            raise ValueError(
                "FlowTActor expects flow_state with shape [B, flow_state_dim], got "
                f"{tuple(flow_state.shape)} for "
                f"flow_state_dim={self.flow_state_dim}."
            )
        normalized_state = (
            self.bn_action(flow_state, False) if self.use_batch_norm else flow_state
        )
        action_embedding = self.action_proj(normalized_state.unsqueeze(1))
        from_embedding = self.time_embedding(
            self._format_time_embedding_input(
                t_from,
                flow_state,
                name="t_from",
            )
        )

        if self.objective == IMPROVED_MEANFLOW:
            if t_to is None:
                raise ValueError(
                    "Improved MeanFlow requires the explicit interval endpoint "
                    "t_to; it must not be inferred from t_from."
                )
            to_embedding = self.time_to_embedding(
                self._format_time_embedding_input(
                    t_to,
                    flow_state,
                    name="t_to",
                )
            )
            time_embedding = self.time_fusion(
                torch.cat([from_embedding, to_embedding], dim=-1)
            )
        else:
            if t_to is not None:
                raise ValueError(
                    "Rectified Flow predicts an instantaneous field and does "
                    "not accept t_to."
                )
            time_embedding = from_embedding
        return action_embedding + time_embedding

    def _decode_field_output(
        self,
        token: torch.Tensor,
        condition: torch.Tensor,
    ) -> FlowFieldOutput:
        if condition.ndim != 3 or condition.shape[1:] != (1, self.d_model):
            raise ValueError(
                "condition must have shape [B, 1, d_model], got "
                f"{tuple(condition.shape)}."
            )
        if token.shape[0] != condition.shape[0]:
            raise ValueError("condition and flow_state batch sizes must match.")
        diagonal_mask = torch.zeros(
            token.shape[1],
            token.shape[1],
            device=token.device,
            dtype=token.dtype,
        )
        hidden = token
        for layer in self.transformer_layers:
            hidden = layer(hidden, condition, tgt_mask=diagonal_mask)
        hidden = hidden.squeeze(1)
        return FlowFieldOutput(
            value=self.velocity_mean_head(hidden),
            hidden=hidden,
            field_kind=(
                "interval_average"
                if self.objective == IMPROVED_MEANFLOW
                else "instantaneous_velocity"
            ),
        )

    def predict_field(
        self,
        condition: torch.Tensor,
        flow_state: torch.Tensor,
        *,
        t_from: float | torch.Tensor,
        t_to: float | torch.Tensor | None = None,
    ) -> FlowFieldOutput:
        """Predict RF ``v(x,t)`` or interval iMF ``U(x,t_to,t_from)``."""

        token = self._embed_action_time(
            flow_state,
            t_from=t_from,
            t_to=t_to,
        )
        return self._decode_field_output(token, condition)

    def transition_moments(
        self,
        condition: torch.Tensor,
        flow_state: torch.Tensor,
        *,
        t_from: float | torch.Tensor,
        t_to: float | torch.Tensor,
        sampler_method: str | None = None,
    ) -> FlowTransitionMoments:
        """Compute one transition from a single objective-correct field call."""

        method = normalize_sampler_method(sampler_method or self.sampler_method)
        field_kwargs = {
            "t_from": t_from,
            "t_to": t_to if self.objective == IMPROVED_MEANFLOW else None,
        }
        field_output = self.predict_field(condition, flow_state, **field_kwargs)
        if self.objective == RECTIFIED_FLOW:
            deterministic = rectified_flow_ode_moments(
                flow_state,
                field_output.value,
                t_from,
                t_to,
            )
        else:
            deterministic = improved_meanflow_ode_moments(
                flow_state,
                field_output.value,
                t_from,
                t_to,
            )

        if method == FLOW_ODE:
            return deterministic
        if method == FLOW_NOISE:
            if not self.add_flow_noise_head:
                raise ValueError(
                    "flow_noise requires add_flow_noise_head=True on FlowTActor."
                )
            x_std = self.flow_noise_head(field_output.hidden)
            return FlowTransitionMoments(
                x_mean=deterministic.x_mean,
                x_std=x_std,
            )
        if self.objective == RECTIFIED_FLOW:
            return rectified_flow_sde_moments(
                flow_state,
                field_output.value,
                t_from,
                t_to,
                noise_level=self.noise_level,
            )
        return improved_meanflow_sde_moments(
            flow_state,
            field_output.value,
            t_from,
            t_to,
            noise_level=self.noise_level,
            std_min=self.sde_std_min,
            std_max=self.sde_std_max,
            safe_initial_time=self.safe_initial_time,
        )

    def sample_next_flow_state(
        self,
        condition: torch.Tensor,
        flow_state: torch.Tensor,
        *,
        t_from: float | torch.Tensor,
        t_to: float | torch.Tensor,
        sampler_method: str | None = None,
        noise: torch.Tensor | None = None,
    ) -> FlowTransitionSample:
        """Sample one state transition with optional common random noise."""

        method = normalize_sampler_method(sampler_method or self.sampler_method)
        moments = self.transition_moments(
            condition,
            flow_state,
            t_from=t_from,
            t_to=t_to,
            sampler_method=method,
        )
        if method == FLOW_ODE:
            if noise is not None:
                raise ValueError("noise is invalid for deterministic flow_ode.")
            return deterministic_transition(moments.x_mean)
        return sample_normal_transition(
            moments.x_mean,
            moments.x_std,
            noise=noise,
        )

    def recompute_log_prob(
        self,
        condition: torch.Tensor,
        detached_trace: FlowPathTrace,
        *,
        sampler_method: str | None = None,
    ) -> torch.Tensor:
        """Recompute a stochastic path likelihood on a fixed detached chain.

        Unlike the reparameterized sampling likelihood, every next state is a
        constant target here. Gaussian mean gradients therefore remain present
        and provide the score-function entropy term used by the SAC actor.
        """

        method = normalize_sampler_method(sampler_method or self.sampler_method)
        if method == FLOW_ODE:
            raise ValueError("flow_ode has no tractable transition likelihood.")
        trace = self._flatten_path_trace(detached_trace.detach())
        num_steps = len(trace.times)
        if len(trace.states) != num_steps + 1:
            raise ValueError("FlowPathTrace must contain K+1 states for K times.")
        if len(trace.means) != num_steps or len(trace.stds) != num_steps:
            raise ValueError("FlowPathTrace moments must contain K entries.")

        total_log_prob = initial_normal_log_prob(trace.states[0])
        for step, (t_from, t_to) in enumerate(trace.times):
            moments = self.transition_moments(
                condition,
                trace.states[step],
                t_from=t_from,
                t_to=t_to,
                sampler_method=method,
            )
            if torch.any(moments.x_std <= 0):
                raise ValueError("stochastic path recomputation requires x_std > 0.")
            transition_distribution = Normal(moments.x_mean, moments.x_std)
            total_log_prob = total_log_prob + (
                transition_distribution.log_prob(trace.states[step + 1])
                .flatten(start_dim=1)
                .sum(dim=1, keepdim=True)
            )

        action_scale, action_bias = self._expanded_action_affine(trace.states[-1])
        _, log_abs_jacobian = squash_action(
            trace.states[-1],
            action_scale,
            action_bias,
        )
        return total_log_prob - log_abs_jacobian

    def sample_path(
        self,
        condition: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        step_noises: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        sampler_method: str | None = None,
        num_steps: int | None = None,
        train: bool = False,
        log_grad: bool = False,
        compute_actor_entropy_surrogate: bool = False,
    ) -> FlowPathSample:
        """Sample an RF ``0->1`` or iMF ``1->0`` path in ``[B, H, A]`` form."""

        # ``train`` is retained in the public sampler contract for FlowPolicy.
        # Batch-stat behavior has already been chosen by encode_condition().
        del train
        if log_grad:
            self.grad_norms.clear()
        method = normalize_sampler_method(sampler_method or self.sampler_method)
        if num_steps is None:
            num_steps = self.denoising_steps
        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or not 1 <= num_steps <= 99
        ):
            raise ValueError("num_steps must be an integer in [1, 99].")
        if method == FLOW_SDE and self.noise_level <= 0:
            raise ValueError("flow_sde requires noise_level > 0.")
        if (
            method == FLOW_SDE
            and self.objective == IMPROVED_MEANFLOW
            and self.safe_initial_time <= 1.0 - 1.0 / num_steps
        ):
            raise ValueError(
                "Improved MeanFlow flow_sde requires safe_initial_time to be "
                "strictly greater than the first t_to=1-1/num_steps endpoint."
            )
        if (
            method == FLOW_SDE
            and self.objective == IMPROVED_MEANFLOW
            and self.sde_std_min <= 0
        ):
            raise ValueError(
                "SAC Improved MeanFlow flow_sde requires sde_std_min > 0 for "
                "a finite transition log-probability."
            )
        if condition.ndim != 3 or condition.shape[1:] != (1, self.d_model):
            raise ValueError(
                "condition must be produced by encode_condition() and have "
                f"shape [B, 1, {self.d_model}], got {tuple(condition.shape)}."
            )
        public_noise_shape = (
            condition.shape[0],
            self.action_horizon,
            self.action_dim,
        )
        if initial_noise is None:
            public_initial_noise = torch.randn(
                public_noise_shape,
                device=condition.device,
                dtype=condition.dtype,
            )
        else:
            if tuple(initial_noise.shape) != public_noise_shape:
                raise ValueError(
                    f"initial_noise must have shape {public_noise_shape}, got "
                    f"{tuple(initial_noise.shape)}."
                )
            public_initial_noise = initial_noise.to(
                device=condition.device,
                dtype=condition.dtype,
            )
        if not torch.isfinite(public_initial_noise).all():
            raise ValueError("initial_noise must contain only finite values.")
        flow_state = public_initial_noise.reshape(
            condition.shape[0], self.flow_state_dim
        )
        timesteps = build_time_schedule(
            self.objective,
            num_steps,
            device=condition.device,
            dtype=flow_state.dtype,
        )
        normalized_noises = normalize_step_noises(
            step_noises,
            num_steps=num_steps,
            reference=public_initial_noise,
            stochastic=method != FLOW_ODE,
        )
        states = [flow_state]
        transitions = []
        for step in range(num_steps):
            transition = self.sample_next_flow_state(
                condition,
                flow_state,
                t_from=timesteps[step],
                t_to=timesteps[step + 1],
                sampler_method=method,
                noise=(
                    None
                    if normalized_noises[step] is None
                    else normalized_noises[step].reshape(
                        condition.shape[0], self.flow_state_dim
                    )
                ),
            )
            flow_state = transition.x_next
            states.append(flow_state)
            transitions.append(transition)
            if log_grad and flow_state.requires_grad:
                current_step = step
                flow_state.register_hook(
                    lambda grad, s=current_step: self.grad_norms.update(
                        {s: grad.norm().item()}
                    )
                )

        action_scale, action_bias = self._expanded_action_affine(flow_state)
        sample = assemble_flow_path_sample(
            states=states,
            timesteps=timesteps,
            transitions=transitions,
            action_scale=action_scale,
            action_bias=action_bias,
            sampler_method=method,
        )
        if compute_actor_entropy_surrogate:
            if method == FLOW_ODE:
                raise ValueError(
                    "actor entropy-score recomputation requires a stochastic sampler."
                )
            entropy_surrogate = self.recompute_log_prob(
                condition,
                sample.path_trace,
                sampler_method=method,
            )
            sample = replace(
                sample,
                actor_entropy_surrogate=entropy_surrogate,
            )
        return self._publicize_sample(sample)

    def sample_path_from_obs(
        self,
        obs: torch.Tensor,
        *,
        train: bool = False,
        **sample_kwargs,
    ) -> FlowPathSample:
        """Compatibility wrapper that encodes observations before sampling."""

        condition = self.encode_condition(obs, update_stats=train)
        return self.sample_path(condition, **sample_kwargs)

    def forward(self, obs, train=False, log_grad=False, **sample_kwargs):
        sample = self.sample_path_from_obs(
            obs,
            train=train,
            log_grad=log_grad,
            **sample_kwargs,
        )
        return sample.action, sample.path_log_prob


class JaxFlowTActor(nn.Module):
    """
    JAX-style Flow Matching Actor (uses noise sampling instead of distribution sampling)
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        d_model=64,
        n_head=4,
        n_layers=2,
        denoising_steps=4,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        action_scale=None,
        action_bias=None,
        noise_std_head=False,
        log_std_min_train=-5,
        log_std_max_train=2,
        log_std_min_rollout=-5,
        log_std_max_rollout=2,
        noise_std_train=0.3,
        noise_std_rollout=0.02,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.use_batch_norm = use_batch_norm
        # Whether to use fixed noise std, otherwise predict std via velocity_log_std_head
        self.noise_std_head = noise_std_head
        # Different noise std for train/rollout, smaller noise during rollout.
        self.log_std_min_train = log_std_min_train
        self.log_std_max_train = log_std_max_train
        self.log_std_min_rollout = log_std_min_rollout
        self.log_std_max_rollout = log_std_max_rollout
        # Fixed noise std added directly to actions
        self.noise_std_train = noise_std_train
        self.noise_std_rollout = noise_std_rollout

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model // 2),
            nn.SiLU(),  # SiLU is PyTorch's Swish/silu
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Action input projection (projects action_dim -> d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)

        # Time embedding (projects 1 -> d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.SiLU(),
            nn.Linear(self.d_model // 4, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

        # Transformer decoder layers
        # We use nn.TransformerDecoderLayer which includes self-attn, cross-attn, and FFN
        decoder_layers = []
        for _ in range(self.n_layers):
            decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
            )
        self.transformer_layers = nn.ModuleList(decoder_layers)

        # Velocity output heads
        self.velocity_mean_head = nn.Linear(self.d_model, self.action_dim)
        # Use a specific head to predict velocity_log_std
        if self.noise_std_head:
            self.velocity_log_std_head = nn.Linear(self.d_model, self.action_dim)

        if self.use_batch_norm:
            self.bn_obs = BatchRenorm(self.obs_dim, momentum=batch_norm_momentum)
            self.bn_action = BatchRenorm(self.action_dim, momentum=batch_norm_momentum)

        # --- Action Scaling ---
        if action_scale is not None and action_bias is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            # Default to [-1, 1] range
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

        self._init_weights()

        self.grad_norms = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, train=False, log_grad=False):
        if log_grad:
            self.grad_norms.clear()

        batch_size = obs.shape[0]
        device = obs.device

        # Flow Matching time step size
        DELTA_T = 1.0 / self.denoising_steps

        # 1. Observation encoding (memory)
        # obs: [batch_size, obs_dim] -> obs_emb: [batch_size, d_model]
        if self.use_batch_norm:
            obs = self.bn_obs(obs, train)
        obs_emb = self.obs_encoder(obs)
        # Add sequence dimension: [batch_size, 1, d_model]
        obs_emb = obs_emb.unsqueeze(1)

        x_current = torch.randn((batch_size, self.action_dim), device=device)

        # Calculate x0 log probability under N(0, I) using torch.distributions
        initial_dist = Normal(torch.zeros_like(x_current), torch.ones_like(x_current))
        total_log_prob = initial_dist.log_prob(x_current).sum(dim=1, keepdim=True)

        if self.noise_std_head:
            log_std_min = self.log_std_min_train if train else self.log_std_min_rollout
            log_std_max = self.log_std_max_train if train else self.log_std_max_rollout
        else:
            noise_std = self.noise_std_train if train else self.noise_std_rollout

        # 3. Flow Matching iterative refinement
        for step in range(self.denoising_steps):
            # 3a. Project current action to embedding space
            # x_current: [batch_size, action_dim] -> x_input: [batch_size, 1, action_dim]
            if self.use_batch_norm:
                x_bn = self.bn_action(x_current, train)
            else:
                x_bn = x_current
            x_input_bn = x_bn.unsqueeze(1)
            # action_emb: [batch_size, 1, d_model]
            action_emb = self.action_proj(x_input_bn)

            # 3b. Add time embedding
            time_value = torch.full(
                (batch_size, 1, 1),
                step / self.denoising_steps,
                device=device,
                dtype=torch.float32,
            )
            # time_emb: [batch_size, 1, d_model]
            time_emb = self.time_embedding(time_value)

            # 3c. Combine action and time to form query (tgt)
            # input_emb: [batch_size, 1, d_model]
            input_emb = action_emb + time_emb

            # 3d. Create diagonal mask
            # For a single query position (seq_len=1), we don't need
            # to mask future positions. A mask of 0s is fine.
            # PyTorch's mask should be (L, L) -> (1, 1)
            diagonal_mask = torch.zeros(1, 1, device=device)

            # 3e. Transformer forward pass
            output = input_emb
            for layer in self.transformer_layers:
                # tgt=output, memory=obs_emb, tgt_mask=diagonal_mask
                output = layer(output, obs_emb, tgt_mask=diagonal_mask)

            # Output is [batch_size, 1, d_model], squeeze to [batch_size, d_model]
            output = output.squeeze(1)

            # Choice A: use NN predicted velocity_log_std, add noise to velocity
            if self.noise_std_head:
                # 3f. Predict velocity
                velocity_mean = self.velocity_mean_head(output)
                velocity_log_std = self.velocity_log_std_head(output)

                # Clamp log_std
                velocity_log_std = torch.tanh(velocity_log_std)
                velocity_log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (
                    velocity_log_std + 1
                )
                velocity_std = torch.exp(velocity_log_std)

                # 3g. Sample velocity (JAX style: sample noise first, then add)
                noise_dist = Normal(0, 1)
                noise = noise_dist.rsample()

                predicted_velocity = velocity_mean + velocity_std * noise

                velocity_log_prob = noise_dist.log_prob(noise).sum(dim=-1, keepdim=True)
                total_log_prob += velocity_log_prob

                # 3i. Flow Matching update: x_{t+1} = x_t + v_t * Δt
                x_current = x_current + predicted_velocity * DELTA_T

            # Choice B: use fixed noise_std, add noise to action
            else:
                # 3f. Predict velocity
                velocity_mean = self.velocity_mean_head(output)

                # 3g. Euler step (no noise, deterministic)
                x_next_mean = x_current + velocity_mean * DELTA_T

                # 3h. Add noise to actions (not velocity)
                noise_dist = Normal(0, 1)
                noise = noise_dist.rsample((batch_size, self.action_dim)).to(device)
                x_current = x_next_mean + noise_std * noise

                # 3i. calculate log prob
                step_log_prob = (
                    Normal(x_next_mean, noise_std)
                    .log_prob(x_current)
                    .sum(dim=-1, keepdim=True)
                )
                total_log_prob += step_log_prob

            # Add gradient logging hook in style of Actor
            if log_grad:
                current_step_for_hook = step
                x_current.register_hook(
                    lambda grad, s=current_step_for_hook: self.grad_norms.update(
                        {s: grad.norm().item()}
                    )
                )

        # 4. Apply tanh transformation and scaling
        y_t = torch.tanh(x_current)
        action = y_t * self.action_scale + self.action_bias

        # 5. Add Jacobian correction for tanh
        tanh_correction = torch.sum(
            torch.log(self.action_scale * (1 - y_t**2) + 1e-6), dim=-1, keepdim=True
        )
        total_log_prob -= tanh_correction

        return action, total_log_prob
