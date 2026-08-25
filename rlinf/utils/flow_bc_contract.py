# Copyright 2026 The RLinf Authors.
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

"""Canonical configuration contract for PyTorch Flow-T behavior cloning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


class FlowBCConfigError(ValueError):
    """Raised when a Flow-T model does not satisfy the unified contract."""


@dataclass(frozen=True)
class FlowSamplingSpec:
    """One configured Flow-T sampling use case."""

    method: str
    num_steps: int


@dataclass(frozen=True)
class FlowSDESpec:
    """Objective-specific parameters for stochastic Flow-T sampling."""

    noise_level: float
    noise_std_range: tuple[float, float] | None
    safe_initial_time: float | None
    joint_path_logprob: bool


@dataclass(frozen=True)
class FlowBCSpec:
    """Resolved, versionless Flow-T model and sampling contract."""

    implementation: str
    action_horizon: int
    action_dim: int
    objective: str
    action_transform: str
    action_layout: str
    latent_normalization: str
    evaluation: FlowSamplingSpec | None
    actor_update: FlowSamplingSpec | None
    online_rollout: FlowSamplingSpec | None
    flow_sde: FlowSDESpec | None
    flow_noise_std_range: tuple[float, float] | None

    @property
    def flow_state_dim(self) -> int:
        """Flattened action-chunk width used inside ``FlowTActor``."""
        return self.action_horizon * self.action_dim


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FlowBCConfigError(f"{name} must be a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise FlowBCConfigError(f"Unknown {name} keys: {unknown}")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise FlowBCConfigError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FlowBCConfigError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FlowBCConfigError(f"{name} must be a finite number")
    return result


def _sampling_section(
    sampling: Mapping[str, Any],
    name: str,
) -> FlowSamplingSpec | None:
    raw = sampling.get(name)
    if raw is None:
        return None
    section = _mapping(raw, f"flow_sampling.{name}")
    if not section:
        raise FlowBCConfigError(f"flow_sampling.{name} must not be empty")
    _reject_unknown(section, {"method", "num_steps"}, f"flow_sampling.{name}")
    method = str(section.get("method", ""))
    if method not in {"flow_ode", "flow_sde", "flow_noise"}:
        raise FlowBCConfigError(
            f"flow_sampling.{name}.method must be flow_ode, flow_sde, or flow_noise"
        )
    num_steps = _positive_integer(
        section.get("num_steps"), f"flow_sampling.{name}.num_steps"
    )
    if num_steps >= 100:
        raise FlowBCConfigError(f"flow_sampling.{name}.num_steps must be in [1, 99]")
    return FlowSamplingSpec(method=method, num_steps=num_steps)


def _positive_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise FlowBCConfigError(f"{name} must contain [min, max]")
    minimum = _finite_float(value[0], f"{name}[0]")
    maximum = _finite_float(value[1], f"{name}[1]")
    if minimum <= 0.0 or minimum > maximum:
        raise FlowBCConfigError(f"{name} requires 0 < min <= max")
    return minimum, maximum


def resolve_flow_bc_spec(
    model_cfg: Mapping[str, Any],
    *,
    task_type: str | None = None,
) -> FlowBCSpec:
    """Resolve and validate the single PyTorch Flow-T configuration path.

    Args:
        model_cfg: The ``actor.model`` or ``rollout.model`` mapping.
        task_type: ``"sft"`` requires static evaluation sampling;
            ``"embodied"`` additionally validates the H=1 online sampling
            contract. ``None`` validates whichever sampling sections are present.

    Returns:
        An immutable canonical contract shared by data, model, and checkpoint code.

    Raises:
        FlowBCConfigError: If the configuration uses a legacy or inconsistent
            Flow-T contract.
    """
    if not isinstance(model_cfg, Mapping):
        raise FlowBCConfigError("Flow-T model config must be a mapping")
    if task_type not in {None, "sft", "embodied"}:
        raise FlowBCConfigError(f"Unsupported Flow-T task_type={task_type!r}")
    if "num_action_chunks" in model_cfg:
        raise FlowBCConfigError(
            "actor.model.num_action_chunks is unsupported for Flow-T; "
            "use actor.model.action_horizon"
        )

    action_horizon = _positive_integer(
        model_cfg.get("action_horizon"), "actor.model.action_horizon"
    )
    action_dim = _positive_integer(
        model_cfg.get("action_dim"), "actor.model.action_dim"
    )

    matching = _mapping(model_cfg.get("flow_matching"), "flow_matching")
    if not matching:
        raise FlowBCConfigError("actor.model.flow_matching is required")
    _reject_unknown(
        matching,
        {
            "implementation",
            "objective",
            "action_transform",
            "action_chunking",
            "rectified_flow",
            "improved_meanflow",
        },
        "flow_matching",
    )
    implementation = str(matching.get("implementation", ""))
    if implementation != "pytorch_flow_t":
        raise FlowBCConfigError("flow_matching.implementation must be 'pytorch_flow_t'")
    action_transform = str(matching.get("action_transform", ""))
    if action_transform != "tanh_latent":
        raise FlowBCConfigError("flow_matching.action_transform must be 'tanh_latent'")

    action_chunking = _mapping(
        matching.get("action_chunking"), "flow_matching.action_chunking"
    )
    if not action_chunking:
        raise FlowBCConfigError("flow_matching.action_chunking is required")
    _reject_unknown(
        action_chunking,
        {"action_layout", "latent_normalization"},
        "flow_matching.action_chunking",
    )
    action_layout = str(action_chunking.get("action_layout", ""))
    if action_layout != "chunk_major":
        raise FlowBCConfigError(
            "flow_matching.action_chunking.action_layout must be 'chunk_major'"
        )
    latent_normalization = str(action_chunking.get("latent_normalization", ""))
    if latent_normalization != "identity":
        raise FlowBCConfigError(
            "flow_matching.action_chunking.latent_normalization must be 'identity'"
        )

    objective = str(matching.get("objective", ""))
    if objective not in {"rectified_flow", "improved_meanflow"}:
        raise FlowBCConfigError(
            "flow_matching.objective must be 'rectified_flow' or 'improved_meanflow'"
        )
    rectified_flow = _mapping(
        matching.get("rectified_flow"), "flow_matching.rectified_flow"
    )
    _reject_unknown(rectified_flow, set(), "flow_matching.rectified_flow")
    improved_meanflow = _mapping(
        matching.get("improved_meanflow"), "flow_matching.improved_meanflow"
    )
    _reject_unknown(
        improved_meanflow,
        {
            "boundary_pair_probability",
            "logit_mean",
            "logit_std",
            "adaptive_power",
            "adaptive_epsilon",
            "time_conditioning",
        },
        "flow_matching.improved_meanflow",
    )
    if objective == "improved_meanflow":
        boundary_probability = _finite_float(
            improved_meanflow.get("boundary_pair_probability", 0.5),
            "flow_matching.improved_meanflow.boundary_pair_probability",
        )
        if not 0.0 <= boundary_probability <= 1.0:
            raise FlowBCConfigError(
                "improved_meanflow.boundary_pair_probability must be in [0, 1]"
            )
        logit_std = _finite_float(
            improved_meanflow.get("logit_std", 1.0),
            "flow_matching.improved_meanflow.logit_std",
        )
        adaptive_epsilon = _finite_float(
            improved_meanflow.get("adaptive_epsilon", 0.01),
            "flow_matching.improved_meanflow.adaptive_epsilon",
        )
        for key, default in (
            ("logit_mean", -0.4),
            ("adaptive_power", 1.0),
        ):
            _finite_float(
                improved_meanflow.get(key, default),
                f"flow_matching.improved_meanflow.{key}",
            )
        if logit_std <= 0.0:
            raise FlowBCConfigError("improved_meanflow.logit_std must be positive")
        if adaptive_epsilon <= 0.0:
            raise FlowBCConfigError(
                "improved_meanflow.adaptive_epsilon must be positive"
            )
        time_conditioning = _mapping(
            improved_meanflow.get("time_conditioning"),
            "flow_matching.improved_meanflow.time_conditioning",
        )
        _reject_unknown(
            time_conditioning,
            {"from_encoder", "to_encoder", "fusion"},
            "flow_matching.improved_meanflow.time_conditioning",
        )
        expected_time_conditioning = {
            "from_encoder": "independent_mlp",
            "to_encoder": "independent_mlp",
            "fusion": "concat_linear_2d_to_d",
        }
        for key, expected in expected_time_conditioning.items():
            if time_conditioning.get(key) != expected:
                raise FlowBCConfigError(
                    "flow_matching.improved_meanflow.time_conditioning."
                    f"{key} must be {expected!r}"
                )

    sampling = _mapping(model_cfg.get("flow_sampling"), "flow_sampling")
    _reject_unknown(
        sampling,
        {
            "actor_update",
            "online_rollout",
            "evaluation",
            "flow_noise",
            "flow_sde",
        },
        "flow_sampling",
    )
    actor_update = _sampling_section(sampling, "actor_update")
    online_rollout = _sampling_section(sampling, "online_rollout")
    evaluation = _sampling_section(sampling, "evaluation")

    if task_type == "sft":
        if evaluation is None:
            raise FlowBCConfigError("flow_sampling.evaluation is required for Flow BC")
        if actor_update is not None or online_rollout is not None:
            raise FlowBCConfigError(
                "Flow BC accepts only flow_sampling.evaluation sampling sections"
            )
        if evaluation.method not in {"flow_ode", "flow_sde"}:
            raise FlowBCConfigError(
                "Flow BC flow_sampling.evaluation.method must be flow_ode or flow_sde"
            )
    elif task_type == "embodied":
        if action_horizon != 1:
            raise FlowBCConfigError(
                "Online/non-SFT PyTorch Flow-T currently requires "
                "actor.model.action_horizon=1"
            )
        missing = [
            name
            for name, section in (
                ("actor_update", actor_update),
                ("online_rollout", online_rollout),
                ("evaluation", evaluation),
            )
            if section is None
        ]
        if missing:
            raise FlowBCConfigError(
                f"Online Flow-T requires flow_sampling sections: {missing}"
            )
        if evaluation.method not in {"flow_ode", "flow_sde"}:
            raise FlowBCConfigError(
                "flow_sampling.evaluation.method must be flow_ode or flow_sde"
            )
        for name, section in (
            ("actor_update", actor_update),
            ("online_rollout", online_rollout),
        ):
            if section.method == "flow_ode":
                raise FlowBCConfigError(
                    f"SAC {name} requires stochastic flow_noise or flow_sde"
                )

    configured_sections = tuple(
        section
        for section in (evaluation, actor_update, online_rollout)
        if section is not None
    )
    used_methods = {section.method for section in configured_sections}
    flow_noise = _mapping(sampling.get("flow_noise"), "flow_sampling.flow_noise")
    _reject_unknown(flow_noise, {"noise_std_range"}, "flow_sampling.flow_noise")
    flow_noise_std_range: tuple[float, float] | None = None
    if "flow_noise" in used_methods:
        flow_noise_std_range = _positive_range(
            flow_noise.get("noise_std_range"),
            "flow_sampling.flow_noise.noise_std_range",
        )
    elif flow_noise:
        raise FlowBCConfigError(
            "flow_sampling.flow_noise is configured but no section uses flow_noise"
        )

    flow_sde_cfg = _mapping(sampling.get("flow_sde"), "flow_sampling.flow_sde")
    _reject_unknown(
        flow_sde_cfg,
        {
            "noise_level",
            "noise_std_range",
            "safe_initial_time",
            "joint_path_logprob",
        },
        "flow_sampling.flow_sde",
    )
    flow_sde: FlowSDESpec | None = None
    if "flow_sde" in used_methods:
        noise_level = _finite_float(
            flow_sde_cfg.get("noise_level"), "flow_sampling.flow_sde.noise_level"
        )
        if noise_level <= 0.0:
            raise FlowBCConfigError(
                "flow_sampling.flow_sde.noise_level must be positive"
            )
        joint_path_logprob = flow_sde_cfg.get("joint_path_logprob", True)
        if not isinstance(joint_path_logprob, bool):
            raise FlowBCConfigError(
                "flow_sampling.flow_sde.joint_path_logprob must be a boolean"
            )
        noise_std_range: tuple[float, float] | None = None
        safe_initial_time: float | None = None
        if objective == "improved_meanflow":
            noise_std_range = _positive_range(
                flow_sde_cfg.get("noise_std_range"),
                "flow_sampling.flow_sde.noise_std_range",
            )
            safe_initial_time = _finite_float(
                flow_sde_cfg.get("safe_initial_time"),
                "flow_sampling.flow_sde.safe_initial_time",
            )
            if not 0.0 < safe_initial_time < 1.0:
                raise FlowBCConfigError(
                    "flow_sampling.flow_sde.safe_initial_time must be in (0, 1)"
                )
            sde_steps = [
                section.num_steps
                for section in configured_sections
                if section.method == "flow_sde"
            ]
            if any(
                safe_initial_time <= 1.0 - 1.0 / num_steps for num_steps in sde_steps
            ):
                raise FlowBCConfigError(
                    "iMF flow_sde.safe_initial_time must be strictly greater "
                    "than 1-1/num_steps for every SDE sampling section"
                )
        elif any(
            key in flow_sde_cfg for key in ("noise_std_range", "safe_initial_time")
        ):
            raise FlowBCConfigError(
                "RF corrected flow_sde derives std from noise_level and step size; "
                "remove iMF-only noise_std_range and safe_initial_time"
            )
        flow_sde = FlowSDESpec(
            noise_level=noise_level,
            noise_std_range=noise_std_range,
            safe_initial_time=safe_initial_time,
            joint_path_logprob=joint_path_logprob,
        )
    elif flow_sde_cfg:
        raise FlowBCConfigError(
            "flow_sampling.flow_sde is configured but no section uses flow_sde"
        )

    return FlowBCSpec(
        implementation=implementation,
        action_horizon=action_horizon,
        action_dim=action_dim,
        objective=objective,
        action_transform=action_transform,
        action_layout=action_layout,
        latent_normalization=latent_normalization,
        evaluation=evaluation,
        actor_update=actor_update,
        online_rollout=online_rollout,
        flow_sde=flow_sde,
        flow_noise_std_range=flow_noise_std_range,
    )
