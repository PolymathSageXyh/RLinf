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

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.flow_actor import FlowTActor, JaxFlowTActor
from rlinf.models.embodiment.modules.flow_objectives import build_flow_objective
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.resnet_utils import ResNetEncoder
from rlinf.models.embodiment.modules.utils import init_mlp_weights, layer_init, make_mlp
from rlinf.models.embodiment.modules.value_head import ValueHead

_FLOW_V2_IMPLEMENTATION = "pytorch_flow_t_v2"


def _is_flow_v2(cfg) -> bool:
    matching = getattr(cfg, "flow_matching", {}) or {}
    return matching.get("implementation") == _FLOW_V2_IMPLEMENTATION


def _flow_sampling_section(cfg, section: str) -> dict[str, Any]:
    sampling = getattr(cfg, "flow_sampling", {}) or {}
    return dict(sampling.get(section, {}) or {})


def _flow_t_actor_kwargs(cfg) -> dict[str, Any]:
    """Build only the opt-in v2 kwargs, preserving the legacy constructor call."""
    if not _is_flow_v2(cfg):
        return {}

    matching = cfg.flow_matching
    sampling = cfg.flow_sampling or {}
    actor_sampling = dict(sampling.get("actor_update", {}) or {})
    sampler_method = actor_sampling.get("method", "flow_ode")
    denoising_steps = int(actor_sampling.get("num_steps", cfg.denoising_steps))
    configured_methods = {
        section.get("method")
        for name in ("actor_update", "online_rollout", "evaluation")
        if (section := sampling.get(name, None)) is not None
    }

    flow_noise = dict(sampling.get("flow_noise", {}) or {})
    noise_std_range = flow_noise.get("noise_std_range", [0.01, 0.1])
    flow_sde = dict(sampling.get("flow_sde", {}) or {})
    sde_std_range = flow_sde.get("noise_std_range", [0.01, 0.1])
    return {
        "api_version": "v2",
        "objective": matching["objective"],
        "sampler_method": sampler_method,
        "denoising_steps": denoising_steps,
        "add_flow_noise_head": "flow_noise" in configured_methods,
        "noise_level": float(flow_sde.get("noise_level", 0.1)),
        "x_log_std_min": float(torch.tensor(noise_std_range[0]).log()),
        "x_log_std_max": float(torch.tensor(noise_std_range[1]).log()),
        "sde_std_min": float(sde_std_range[0]),
        "sde_std_max": float(sde_std_range[1]),
        "safe_initial_time": float(flow_sde.get("safe_initial_time", 0.99)),
    }


def _build_flow_bc_objective(cfg):
    if not _is_flow_v2(cfg):
        return None
    matching = cfg.flow_matching
    objective = matching["objective"]
    if objective == "rectified_flow":
        return build_flow_objective(objective)
    objective_cfg = dict(matching.get("improved_meanflow", {}) or {})
    objective_cfg.pop("time_conditioning", None)
    return build_flow_objective(objective, **objective_cfg)


def _images_are_preprocessed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("images_preprocessed must not be empty.")
        flattened = value.to(dtype=torch.bool).reshape(-1)
        if not torch.equal(flattened, flattened[:1].expand_as(flattened)):
            raise ValueError(
                "A Flow BC batch cannot mix preprocessed and raw image ranges."
            )
        return bool(flattened[0].item())
    raise TypeError("images_preprocessed must be a bool or boolean tensor.")


def _actions_to_tanh_latent(
    actions: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Invert the policy's affine-tanh action transform for supervised BC."""
    actions_fp32 = actions.float()
    scale = action_scale.to(device=actions.device, dtype=torch.float32)
    bias = action_bias.to(device=actions.device, dtype=torch.float32)
    if torch.any(scale == 0):
        raise ValueError("Flow BC action_scale entries must be non-zero.")
    unit_action = (actions_fp32 - bias) / scale
    if not torch.isfinite(unit_action).all():
        raise ValueError("Flow BC actions contain non-finite values.")
    # GELLO actions are clipped before RelativeFrame transforms them back to
    # the policy frame. That inverse transform can move a small number of
    # recorded rotation values outside [-1, 1]. The BC contract intentionally
    # clamps those legacy samples before atanh; expose the amount of clipping
    # so it is visible in training logs rather than silently discarded.
    outside_range = (unit_action < -1.0) | (unit_action > 1.0)
    clipped_unit_action = unit_action.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    latent = torch.atanh(clipped_unit_action)
    metrics = {
        "action_clamp_fraction": outside_range.float().mean().detach(),
        "action_clamp_abs_max": (
            (unit_action.abs() - 1.0).clamp_min(0.0).max().detach()
        ),
    }
    return latent.to(dtype=actions.dtype), metrics


@dataclass
class FlowConfig:
    input_type: str = "mixed"
    image_size: list[int] = field(default_factory=list)
    image_num: int = 1
    action_dim: int = 4
    state_dim: int = 29
    num_action_chunks: int = 1
    backbone: str = "resnet"
    model_path: Optional[str] = None  # used as dir actually!
    encoder_config: dict[str, Any] = field(
        default_factory=dict
    )  # 'extra_config' rename to 'encoder_config'
    add_value_head: bool = False
    add_q_head: bool = False
    q_head_type: str = "default"  # same as cnn_policy.py

    state_latent_dim: int = 64
    action_scale: Any = None
    final_tanh = True
    std_range = None  # same as cnn_policy.py
    logstd_range = None  # same as cnn_policy.py

    num_q_heads: int = 2  # same as cnn_policy.py

    # -- Flow Matching specific parameters --##
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "JaxFlowTActor"  # "FlowTActor" or "JaxFlowTActor"
    # Whether to use a separate head to predict noise_std
    noise_std_head: bool = False
    # Min/Max log std for training (if using noise_std_head)
    log_std_min_train: float = -5
    log_std_max_train: float = 2
    # Min/Max log std for rollout (if using noise_std_head)
    log_std_min_rollout: float = -20
    log_std_max_rollout: float = 0
    # Fixed noise std for training (if not using noise_std_head)
    noise_std_train: float = 0.3
    # Fixed noise std for rollout (if not using noise_std_head)
    noise_std_rollout: float = 0.02
    # Opt-in PyTorch Flow-T v2 blocks. Empty mappings preserve legacy behavior.
    flow_matching: dict[str, Any] = field(default_factory=dict)
    flow_sampling: dict[str, Any] = field(default_factory=dict)
    pretrained_actor: dict[str, Any] = field(default_factory=dict)

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True
            if self.backbone == "resnet":
                self.std_range = (1e-5, 5)

        assert self.model_path is not None, "Please specify the model_path."
        assert "ckpt_name" in self.encoder_config, (
            "Please specify the ckpt_name in encoder_config to load pretrained encoder weights."
        )
        ckpt_path = os.path.join(self.model_path, self.encoder_config["ckpt_name"])
        assert os.path.exists(ckpt_path), (
            f"Pretrained encoder weights not found at {ckpt_path} with model path {self.model_path} and encoder ckpt name {self.encoder_config['ckpt_name']}"
        )
        self.encoder_config["ckpt_path"] = ckpt_path


class FlowPolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.flow_v2 = _is_flow_v2(cfg)
        self.flow_bc_objective = _build_flow_bc_objective(cfg)
        self.in_channels = self.cfg.image_size[0]

        # Step1: Init Image encoders (same as CNNPolicy)
        self.encoders = nn.ModuleList()
        encoder_out_dim = 0
        if self.cfg.backbone == "resnet":
            sample_x = torch.randn(1, *self.cfg.image_size)
            for img_id in range(self.cfg.image_num):
                self.encoders.append(
                    ResNetEncoder(
                        sample_x, out_dim=256, encoder_cfg=self.cfg.encoder_config
                    )
                )
                encoder_out_dim += self.encoders[img_id].out_dim
        else:
            raise NotImplementedError

        if self.cfg.backbone == "resnet":
            self.state_proj = nn.Sequential(
                *make_mlp(
                    in_channels=self.cfg.state_dim,
                    mlp_channels=[
                        self.cfg.state_latent_dim,
                    ],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            init_mlp_weights(self.state_proj, nonlinearity="tanh")
            self.mix_proj = nn.Sequential(
                *make_mlp(
                    in_channels=encoder_out_dim + self.cfg.state_latent_dim,
                    mlp_channels=[256, 256],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            init_mlp_weights(self.mix_proj, nonlinearity="tanh")

        # --- Step2: Create flow actor --- #
        # FlowTActor will receive mix_feature (256 dim) as obs input
        # So we set obs_dim to 256 (output of mix_proj)
        flow_obs_dim = 256

        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.tensor((h - l) / 2.0, dtype=torch.float32)
            action_bias = torch.tensor((h + l) / 2.0, dtype=torch.float32)
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(self.cfg.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.cfg.action_dim, dtype=torch.float32)

        if self.cfg.flow_actor_type == "FlowTActor":
            flow_actor_kwargs = {
                "obs_dim": flow_obs_dim,
                "action_dim": self.cfg.action_dim,
                "d_model": self.cfg.d_model,
                "n_head": self.cfg.n_head,
                "n_layers": self.cfg.n_layers,
                "denoising_steps": self.cfg.denoising_steps,
                "use_batch_norm": self.cfg.use_batch_norm,
                "batch_norm_momentum": self.cfg.batch_norm_momentum,
                "action_scale": action_scale,
                "action_bias": action_bias,
            }
            flow_actor_kwargs.update(_flow_t_actor_kwargs(self.cfg))
            self.flow_actor = FlowTActor(**flow_actor_kwargs)
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
                noise_std_head=self.cfg.noise_std_head,
                log_std_min_train=self.cfg.log_std_min_train,
                log_std_max_train=self.cfg.log_std_max_train,
                log_std_min_rollout=self.cfg.log_std_min_rollout,
                log_std_max_rollout=self.cfg.log_std_max_rollout,
                noise_std_train=self.cfg.noise_std_train,
                noise_std_rollout=self.cfg.noise_std_rollout,
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # --- Step3: Create Q-head for SAC --- #
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )
        if self.cfg.add_q_head:
            if self.cfg.backbone == "resnet":  # Now only "resnet" backbone is supported
                hidden_size = encoder_out_dim + self.cfg.state_latent_dim
                hidden_dims = [256, 256, 256]
            if self.cfg.q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=hidden_size,
                    hidden_dims=hidden_dims,
                    num_q_heads=self.cfg.num_q_heads,  # pass from actor.model.num_q_heads
                    action_feature_dim=self.cfg.action_dim,
                )

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer(
                "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
            )
        else:
            self.action_scale = None

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs, *, images_preprocessed: bool = False):
        device = next(self.parameters()).device
        processed_env_obs = {}
        processed_env_obs["states"] = env_obs["states"].clone().to(device)
        processed_env_obs["main_images"] = (
            env_obs["main_images"].clone().to(device).float()
        )
        if not images_preprocessed:
            processed_env_obs["main_images"] = processed_env_obs["main_images"] / 255.0
        if env_obs.get("extra_view_images", None) is not None:
            processed_env_obs["extra_view_images"] = (
                env_obs["extra_view_images"].clone().to(device).float()
            )
            if not images_preprocessed:
                processed_env_obs["extra_view_images"] = (
                    processed_env_obs["extra_view_images"] / 255.0
                )
        return processed_env_obs

    def get_feature(self, obs):
        """Extract features from observations (images + states)"""
        visual_features = []
        # from image_keys to image_num
        for img_id in range(self.cfg.image_num):
            if img_id == 0:
                images = obs["main_images"]
            else:
                images = obs["extra_view_images"][:, img_id - 1]
            if images.shape[3] == 3:
                # [B, H, W, C] -> [B, C, H, W]
                images = images.permute(0, 3, 1, 2)
            visual_features.append(self.encoders[img_id](images))
        visual_feature = torch.cat(visual_features, dim=-1)

        state_feature = self.state_proj(obs["states"])
        full_feature = torch.cat([visual_feature, state_feature], dim=-1)

        return full_feature, visual_feature

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)

        obs = kwargs.get("obs")
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs", None)
        if next_obs is not None:
            next_obs = self.preprocess_env_obs(next_obs)
            kwargs.update({"next_obs": next_obs})

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def _sample_v2_flow(
        self,
        condition,
        *,
        sampling_section: str,
        train: bool,
        initial_noise=None,
        step_noises=None,
        sampler_method=None,
        num_steps=None,
        log_grad: bool = False,
        compute_actor_entropy_surrogate: bool = False,
    ):
        section = _flow_sampling_section(self.cfg, sampling_section)
        method = sampler_method or section.get("method", "flow_ode")
        if num_steps is None:
            num_steps = int(section.get("num_steps", self.flow_actor.denoising_steps))
        return self.flow_actor.sample_path(
            condition,
            initial_noise=initial_noise,
            step_noises=step_noises,
            sampler_method=method,
            num_steps=num_steps,
            train=train,
            log_grad=log_grad,
            compute_actor_entropy_surrogate=compute_actor_entropy_surrogate,
        )

    @staticmethod
    def _select_log_prob(flow_sample, mode: str):
        """Select the distinct SAC actor, alpha, or numeric path contract."""
        if mode == "actor_surrogate":
            value = getattr(flow_sample, "actor_entropy_surrogate", None)
        elif mode == "temperature":
            value = getattr(flow_sample, "temperature_log_prob", None)
        elif mode == "path":
            value = getattr(
                flow_sample,
                "path_log_prob",
                getattr(flow_sample, "joint_path_log_prob", None),
            )
        else:
            raise ValueError(
                "log_prob_mode must be 'actor_surrogate', 'temperature', or "
                f"'path', got {mode!r}."
            )
        if value is None and mode != "path":
            value = getattr(
                flow_sample,
                "path_log_prob",
                getattr(flow_sample, "joint_path_log_prob", None),
            )
        return value

    def sft_forward(self, data, **kwargs):
        """Compute PyTorch RF/iMF behavior-cloning loss on a LeRobot batch."""
        del kwargs
        if not self.flow_v2 or self.flow_bc_objective is None:
            raise NotImplementedError(
                "FlowPolicy SFT requires flow_matching.implementation="
                f"{_FLOW_V2_IMPLEMENTATION!r}."
            )
        if not isinstance(data, dict) or "obs" not in data or "actions" not in data:
            raise ValueError("FlowPolicy SFT data requires 'obs' and 'actions'.")
        images_preprocessed = _images_are_preprocessed(
            data.get("images_preprocessed", False)
        )
        obs = self.preprocess_env_obs(
            data["obs"], images_preprocessed=images_preprocessed
        )
        full_feature, _ = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)
        condition = self.flow_actor.encode_condition(
            mix_feature, update_stats=self.training
        )

        actions = data["actions"].to(device=mix_feature.device, dtype=mix_feature.dtype)
        latent_actions, action_metrics = _actions_to_tanh_latent(
            actions,
            self.flow_actor.action_scale,
            self.flow_actor.action_bias,
        )
        objective_output = self.flow_bc_objective(
            self.flow_actor,
            condition,
            latent_actions,
        )
        return {
            **objective_output.metrics,
            **action_metrics,
            "loss": objective_output.loss,
        }

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)

        if self.flow_v2:
            train = bool(kwargs.get("train", True))
            log_prob_mode = kwargs.get("log_prob_mode", "actor_surrogate")
            condition = self.flow_actor.encode_condition(
                mix_feature, update_stats=train
            )
            flow_sample = self._sample_v2_flow(
                condition,
                sampling_section="actor_update",
                train=train,
                initial_noise=kwargs.get("initial_noise"),
                step_noises=kwargs.get("step_noises"),
                sampler_method=kwargs.get("sampler_method"),
                num_steps=kwargs.get("num_steps"),
                log_grad=bool(kwargs.get("log_grad", False)),
                compute_actor_entropy_surrogate=log_prob_mode == "actor_surrogate",
            )
            log_prob = self._select_log_prob(flow_sample, log_prob_mode)
            if log_prob is None:
                raise RuntimeError(
                    "SAC Flow-T requires a stochastic sampler with path log-probability."
                )
            return flow_sample.action, log_prob, full_feature

        # Use flow actor to generate actions
        # FlowTActor expects obs as input, we pass mix_feature as the observation
        action, log_prob = self.flow_actor(mix_feature, train=True, log_grad=False)

        return action, log_prob, full_feature

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    # use get_q_values() as sac_q_forward()
    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    def default_forward(
        self,
        forward_inputs,
        compute_entropy=False,
        compute_values=False,
        **kwargs,
    ):
        """Default forward pass"""

        obs = {
            "main_images": forward_inputs["main_images"],
            "states": forward_inputs["states"],
        }
        if "extra_view_images" in forward_inputs:
            obs["extra_view_images"] = forward_inputs["extra_view_images"]
        obs = self.preprocess_env_obs(obs)

        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)

        if self.flow_v2:
            condition = self.flow_actor.encode_condition(
                mix_feature, update_stats=False
            )
            flow_sample = self._sample_v2_flow(
                condition,
                sampling_section="evaluation",
                train=False,
            )
            log_prob = self._select_log_prob(flow_sample, "path")
            if log_prob is None:
                log_prob = torch.zeros(
                    (flow_sample.action.shape[0], 1),
                    device=flow_sample.action.device,
                    dtype=flow_sample.action.dtype,
                )
            output_dict = {"action": flow_sample.action, "log_prob": log_prob}
            if compute_entropy:
                output_dict["entropy"] = -log_prob
            if compute_values:
                if getattr(self, "value_head", None):
                    output_dict["values"] = self.value_head(mix_feature)
                else:
                    raise NotImplementedError
            return output_dict

        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)

        output_dict = {
            "action": action,
            "log_prob": log_prob,  # key 'log_prob' or 'logprobs' as used in both cnn_policy.py??
        }

        if compute_entropy:
            # For flow matching, entropy is computed from log_prob
            # Approximate entropy as negative log_prob (this is a simplification)
            entropy = -log_prob
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(mix_feature)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        return_shared_feature=False,
        **kwargs,
    ):
        """Predict actions in batch"""
        env_obs = self.preprocess_env_obs(env_obs)

        full_feature, visual_feature = self.get_feature(env_obs)
        mix_feature = self.mix_proj(full_feature)

        if self.flow_v2:
            mode = kwargs.get("mode", "train")
            section = "online_rollout" if mode == "train" else "evaluation"
            condition = self.flow_actor.encode_condition(
                mix_feature, update_stats=False
            )
            flow_sample = self._sample_v2_flow(
                condition,
                sampling_section=section,
                train=False,
                initial_noise=kwargs.get("initial_noise"),
                step_noises=kwargs.get("step_noises"),
            )
            action = flow_sample.action
            log_prob = getattr(flow_sample, "temperature_log_prob", None)
            if log_prob is None:
                log_prob = getattr(
                    flow_sample,
                    "path_log_prob",
                    getattr(flow_sample, "joint_path_log_prob", None),
                )
            if log_prob is None:
                log_prob = torch.zeros(
                    (action.shape[0], 1), device=action.device, dtype=action.dtype
                )
        else:
            # Use flow actor
            action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)

        # chunk_actions is always torch tensor
        chunk_actions = action.reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(mix_feature)
        else:
            chunk_values = torch.zeros_like(log_prob[..., :1])

        forward_inputs = {"action": action}
        if return_obs:
            # x1. image indexing logic changed
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["states"] = env_obs["states"]
            if "extra_view_images" in env_obs:
                forward_inputs["extra_view_images"] = env_obs["extra_view_images"]

        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = visual_feature
        return chunk_actions, result


@dataclass
class FlowStateConfig:
    input_type: str = "state"
    action_dim: int = 4
    obs_dim: int = 29
    num_action_chunks: int = 1
    encoder_config: dict[str, Any] = field(default_factory=dict)
    add_value_head: bool = False  # No visual_feature -> No mix_feature -> No value_head -> add_value_head must be false !
    add_q_head: bool = False
    q_head_type: str = "default"
    num_q_heads: int = 2

    action_scale: Any = None
    final_tanh = True

    # Flow Matching specific parameters
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "JaxFlowTActor"  # "FlowTActor" or "JaxFlowTActor"
    # Whether to use a separate head to predict noise_std
    noise_std_head: bool = False
    # Min/Max log std for training (if using noise_std_head)
    log_std_min_train: float = -5
    log_std_max_train: float = 2
    # Min log std for rollout (if using noise_std_head)
    log_std_min_rollout: float = -20
    log_std_max_rollout: float = 0
    # Fixed noise std for training (if not using noise_std_head)
    noise_std_train: float = 0.3
    # Fixed noise std for rollout (if not using noise_std_head)
    noise_std_rollout: float = 0.02
    flow_matching: dict[str, Any] = field(default_factory=dict)
    flow_sampling: dict[str, Any] = field(default_factory=dict)
    pretrained_actor: dict[str, Any] = field(default_factory=dict)

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True


class FlowStatePolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: FlowStateConfig):
        super().__init__()
        self.cfg = cfg
        self.flow_v2 = _is_flow_v2(cfg)
        self.flow_bc_objective = _build_flow_bc_objective(cfg)

        # 3 layer MLP encoder for obs
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.cfg.obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        # Create flow actor
        # FlowTActor will receive mix_feature (256 dim) as obs input
        # So we set obs_dim to 256 (output of mix_proj)
        flow_obs_dim = 256

        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.tensor((h - l) / 2.0, dtype=torch.float32)
            action_bias = torch.tensor((h + l) / 2.0, dtype=torch.float32)
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(self.cfg.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.cfg.action_dim, dtype=torch.float32)

        if self.cfg.flow_actor_type == "FlowTActor":
            flow_actor_kwargs = {
                "obs_dim": flow_obs_dim,
                "action_dim": self.cfg.action_dim,
                "d_model": self.cfg.d_model,
                "n_head": self.cfg.n_head,
                "n_layers": self.cfg.n_layers,
                "denoising_steps": self.cfg.denoising_steps,
                "use_batch_norm": self.cfg.use_batch_norm,
                "batch_norm_momentum": self.cfg.batch_norm_momentum,
                "action_scale": action_scale,
                "action_bias": action_bias,
            }
            flow_actor_kwargs.update(_flow_t_actor_kwargs(self.cfg))
            self.flow_actor = FlowTActor(**flow_actor_kwargs)
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias,
                noise_std_head=self.cfg.noise_std_head,
                log_std_min_train=self.cfg.log_std_min_train,
                log_std_max_train=self.cfg.log_std_max_train,
                log_std_min_rollout=self.cfg.log_std_min_rollout,
                log_std_max_rollout=self.cfg.log_std_max_rollout,
                noise_std_train=self.cfg.noise_std_train,
                noise_std_rollout=self.cfg.noise_std_rollout,
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # Q-head for SAC
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )
        if self.cfg.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=self.cfg.obs_dim,
                hidden_dims=[256, 256, 256],
                num_q_heads=self.cfg.num_q_heads,
                action_feature_dim=self.cfg.action_dim,
            )

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer(
                "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
            )
        else:
            self.action_scale = None

    # added num_action_chunks property
    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        return {"states": env_obs["states"].to(device)}

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        feat = self.backbone(obs["states"])

        if self.flow_v2:
            train = bool(kwargs.get("train", True))
            log_prob_mode = kwargs.get("log_prob_mode", "actor_surrogate")
            condition = self.flow_actor.encode_condition(feat, update_stats=train)
            section = _flow_sampling_section(self.cfg, "actor_update")
            sample = self.flow_actor.sample_path(
                condition,
                initial_noise=kwargs.get("initial_noise"),
                step_noises=kwargs.get("step_noises"),
                sampler_method=kwargs.get("sampler_method")
                or section.get("method", "flow_ode"),
                num_steps=int(
                    kwargs.get("num_steps")
                    or section.get("num_steps", self.cfg.denoising_steps)
                ),
                train=train,
                log_grad=bool(kwargs.get("log_grad", False)),
                compute_actor_entropy_surrogate=log_prob_mode == "actor_surrogate",
            )
            log_prob = FlowPolicy._select_log_prob(sample, log_prob_mode)
            if log_prob is None:
                raise RuntimeError(
                    "SAC Flow-T requires a stochastic sampler with path log-probability."
                )
            return sample.action, log_prob, None

        # Use flow actor to generate actions
        # FlowTActor expects obs as input, we pass mix_feature as the observation
        action, log_prob = self.flow_actor(feat, train=True, log_grad=False)

        return action, log_prob, None

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        return self.q_head(obs["states"], actions)

    # use get_q_values() as sac_q_forward()
    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        return self.q_head(obs["states"], actions)

    # 10. add unified forward()
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        obs = kwargs.get("obs")
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs", None)
        if next_obs is not None:
            next_obs = self.preprocess_env_obs(next_obs)
            kwargs.update({"next_obs": next_obs})

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)  # originally exists
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)  # use get_q_values()
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)  # NOT USED (NO get_feature)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        """Compute state-only PyTorch RF/iMF behavior-cloning loss."""
        del kwargs
        if not self.flow_v2 or self.flow_bc_objective is None:
            raise NotImplementedError("FlowStatePolicy SFT requires PyTorch Flow-T v2.")
        obs = self.preprocess_env_obs(data["obs"])
        feat = self.backbone(obs["states"])
        condition = self.flow_actor.encode_condition(feat, update_stats=self.training)
        actions = data["actions"].to(device=feat.device, dtype=feat.dtype)
        latent_actions, action_metrics = _actions_to_tanh_latent(
            actions,
            self.flow_actor.action_scale,
            self.flow_actor.action_bias,
        )
        output = self.flow_bc_objective(self.flow_actor, condition, latent_actions)
        return {**output.metrics, **action_metrics, "loss": output.loss}

    def default_forward(
        self, obs, compute_entropy=False, compute_values=False, **kwargs
    ):
        """
        Default forward pass for FlowStatePolicy.

        This method is not supported for FlowStatePolicy because it relies on features
        (e.g., get_feature, mix_proj) that are not defined for this class.
        It should not be used; kept only for compatibility.
        """
        raise NotImplementedError(
            "FlowStatePolicy.default_forward is not supported. "
            "Use FlowStatePolicy.forward with the appropriate forward_type instead."
        )

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,  # NOT USED, unlike FlowPolicy
        return_obs=True,
        return_shared_feature=False,  # NOT USED, unlike FlowPolicy
        **kwargs,
    ):
        """
        Predict actions in batch.
        Called by MultiStepRolloutWorker for rollout
        """
        env_obs = self.preprocess_env_obs(env_obs)

        feat = self.backbone(env_obs["states"])  # encode obs using the 3 layer MLP

        if self.flow_v2:
            mode = kwargs.get("mode", "train")
            section_name = "online_rollout" if mode == "train" else "evaluation"
            section = _flow_sampling_section(self.cfg, section_name)
            condition = self.flow_actor.encode_condition(feat, update_stats=False)
            sample = self.flow_actor.sample_path(
                condition,
                initial_noise=kwargs.get("initial_noise"),
                step_noises=kwargs.get("step_noises"),
                sampler_method=section.get("method", "flow_ode"),
                num_steps=int(section.get("num_steps", self.cfg.denoising_steps)),
                train=False,
            )
            action = sample.action
            log_prob = getattr(sample, "temperature_log_prob", None)
            if log_prob is None:
                log_prob = getattr(
                    sample,
                    "path_log_prob",
                    getattr(sample, "joint_path_log_prob", None),
                )
            if log_prob is None:
                log_prob = torch.zeros(
                    (action.shape[0], 1), device=action.device, dtype=action.dtype
                )
        else:
            # Use flow actor
            action, log_prob = self.flow_actor(feat, train=False, log_grad=False)

        # chunk_actions is always torch tensor
        chunk_actions = action.reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )

        chunk_values = torch.zeros_like(log_prob[..., :1])

        forward_inputs = {"action": action}
        if return_obs:
            forward_inputs["states"] = env_obs[
                "states"
            ]  # add 'states' to forward_inputs instead of 'obs/{key}'

        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
