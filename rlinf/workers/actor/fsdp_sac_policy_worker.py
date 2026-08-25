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


import copy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.config import SupportedModel
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.flow_actor_checkpoint import (
    build_flow_actor_metadata_from_config,
    load_flow_actor_checkpoint,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.model_config import resolve_model_action_horizon
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory, collect_param_names_need_sync
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.actor.sacflow_finetune import (
    SACFlowFinetuneController,
    SACFlowFinetunePhase,
    compute_frozen_anchor_actor_loss,
    make_common_flow_noise,
)


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

        finetune_cfg = self.cfg.algorithm.get("sacflow_finetune", {})
        self.sacflow_finetune_enabled = bool(finetune_cfg.get("enabled", False))
        self.sacflow_finetune_cfg = finetune_cfg
        self.sacflow_finetune_controller = None
        self.sacflow_anchor_policy = None
        if self.sacflow_finetune_enabled:
            self.sacflow_finetune_controller = SACFlowFinetuneController(
                warmup_transitions=finetune_cfg.get("warmup_transitions", 10_000),
                updates_per_transition=finetune_cfg.get("updates_per_transition", 1.0),
            )
            self.sacflow_behavior_beta = float(
                finetune_cfg.get("behavior_beta", 1_000.0)
            )
            self.sacflow_warmup_behavior_beta = float(
                finetune_cfg.get("warmup_behavior_beta", self.sacflow_behavior_beta)
            )
            if (
                not np.isfinite(self.sacflow_behavior_beta)
                or not np.isfinite(self.sacflow_warmup_behavior_beta)
                or self.sacflow_behavior_beta < 0
                or self.sacflow_warmup_behavior_beta < 0
            ):
                raise ValueError(
                    "SACFlow anchor regularization coefficients must be finite "
                    "and non-negative."
                )
            self.sacflow_common_noise = bool(finetune_cfg.get("common_noise", True))

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()
        if self.sacflow_finetune_enabled:
            if not initialize_target:
                raise ValueError(
                    "Frozen-anchor SACFlow initialization requires a fresh target "
                    "critic (initialize_target=True)."
                )
            if not self.cfg.actor.fsdp_config.get("use_orig_params", False):
                raise ValueError(
                    "Frozen-anchor SACFlow requires actor.fsdp_config."
                    "use_orig_params=true so actor and Q-head parameters retain "
                    "separate, verifiable optimizer ownership."
                )
            self._setup_sacflow_finetune_modules(module, target_module)

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # Record the original trainable parameter names before FSDP wrapping.
        # Persistent buffer names are also recorded for selective weight syncing.
        self.param_names_need_sync = collect_param_names_need_sync(module)

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_dsrl = self.use_dsrl
        if self.sacflow_finetune_enabled:
            # In fine-tuning, the online actor owns the complete observation-to-
            # action path and the critic owns only Q heads. Critic forwards detach
            # actor features below, avoiding optimizer overlap by construction.
            param_filters = {"critic": ["q_head"]}
        elif use_dsrl:
            # DSRL: separate actor/critic encoders into different optimizer groups
            param_filters = {
                "critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]
            }
        else:
            param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]
        if self.sacflow_finetune_enabled:
            self._validate_sacflow_optimizer_ownership()

        # SAC alpha
        # Initialize temperature parameter for automatic entropy tuning
        alpha_type = self.cfg.algorithm.entropy_tuning.get(
            "alpha_type", "softplus"
        )  # supported type: ["softplus","exp","fixed_alpha"]
        self.entropy_temp = EntropyTemperature(
            initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
            alpha_type=alpha_type,
            device=self.device,
            dtype=self.torch_dtype,
        )
        if alpha_type != "fixed_alpha":
            self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                "target_entropy",
                -self.cfg.actor.model.action_dim,
            )

            self.alpha_optimizer = torch.optim.Adam(
                self.entropy_temp.parameters(),
                lr=self.cfg.algorithm.entropy_tuning.optim.lr,
            )

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

        if self.sacflow_finetune_enabled and not self.cfg.actor.get(
            "enable_offload", False
        ):
            self._onload_sacflow_anchor()

    def _setup_sacflow_finetune_modules(self, module, target_module) -> None:
        """Load BC weights and create a standalone frozen full-policy anchor."""
        pretrained_cfg = self.cfg.actor.model.get("pretrained_actor", {})
        checkpoint_path = pretrained_cfg.get("path", None)
        is_full_resume = bool(self.cfg.runner.get("resume_dir", None))
        if checkpoint_path and not is_full_resume:
            expected_metadata = build_flow_actor_metadata_from_config(
                self.cfg.actor.model
            )
            sampling_cfg = self.cfg.actor.model.get("flow_sampling", {})
            configured_methods = {
                section.get("method")
                for name in ("actor_update", "online_rollout", "evaluation")
                if (section := sampling_cfg.get(name, None)) is not None
            }
            expected_missing_prefixes = (
                ("flow_actor.flow_noise_head",)
                if "flow_noise" in configured_methods
                else ()
            )
            report = load_flow_actor_checkpoint(
                module,
                checkpoint_path,
                load_mode=str(pretrained_cfg.get("load_mode", "actor_only")),
                strict_actor=bool(pretrained_cfg.get("strict_actor", True)),
                expected_metadata=expected_metadata,
                expected_missing_prefixes=expected_missing_prefixes,
                require_manifest=bool(pretrained_cfg.get("require_manifest", True)),
            )
            # The target is hard-copied after setup, but initializing it here also
            # avoids a transient mismatch before FSDP wrapping.
            target_module.load_state_dict(module.state_dict())
            self.logger.info(
                "Loaded %d pretrained policy tensors from %s",
                len(report.loaded_keys),
                checkpoint_path,
            )
            if report.expected_missing_keys:
                self.logger.warning(
                    "The pretrained checkpoint intentionally omits %d online-only "
                    "actor tensors.",
                    len(report.expected_missing_keys),
                )
        elif not self.cfg.runner.get("ckpt_path", None) and not is_full_resume:
            self.logger.warning(
                "SACFlow fine-tuning is enabled without a pretrained actor file; "
                "the current initialized policy will become the frozen anchor."
            )

        # Keep the anchor outside the FSDP tree: it must not be flattened,
        # synchronized, exposed to either optimizer, or updated by target EMA.
        self.sacflow_anchor_policy = copy.deepcopy(module)
        self.sacflow_anchor_policy.requires_grad_(False)
        self.sacflow_anchor_policy.eval()

    def _validate_sacflow_optimizer_ownership(self) -> None:
        """Fail fast unless online actor and Q parameters have one owner each."""
        actor_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        critic_ids = {
            id(parameter)
            for group in self.qf_optimizer.param_groups
            for parameter in group["params"]
        }
        overlap = actor_ids & critic_ids
        named_trainable = {
            id(parameter): name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        missing = set(named_trainable) - actor_ids - critic_ids
        invalid_actor = sorted(
            named_trainable.get(parameter_id, "<unknown>")
            for parameter_id in actor_ids
            if "q_head" in named_trainable.get(parameter_id, "")
        )
        invalid_critic = sorted(
            named_trainable.get(parameter_id, "<unknown>")
            for parameter_id in critic_ids
            if "q_head" not in named_trainable.get(parameter_id, "")
        )
        if overlap or missing or invalid_actor or invalid_critic:
            raise ValueError(
                "Invalid frozen-anchor SACFlow optimizer ownership: "
                f"overlap={len(overlap)}, missing="
                f"{[named_trainable[key] for key in missing]}, "
                f"actor_q_params={invalid_actor}, critic_non_q_params="
                f"{invalid_critic}. Set FSDP use_orig_params=true and keep Q "
                "heads in a dedicated q_head module."
            )

    def _onload_sacflow_anchor(self) -> None:
        """Move the frozen anchor beside the online model when it is needed."""
        if self.sacflow_anchor_policy is None:
            return
        model_device = next(self.model.parameters()).device
        anchor_device = next(self.sacflow_anchor_policy.parameters()).device
        if anchor_device != model_device:
            self.sacflow_anchor_policy.to(model_device)
        self.sacflow_anchor_policy.eval()

    def _offload_sacflow_anchor(self) -> None:
        """Mirror actor parameter offload for the standalone frozen anchor."""
        if self.sacflow_anchor_policy is not None:
            self.sacflow_anchor_policy.to("cpu")

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        if (
            self.sacflow_finetune_enabled
            and self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        ):
            raise NotImplementedError(
                "Frozen-anchor SACFlow currently requires the default Q head so "
                "critic forwards can detach the actor-owned observation encoder."
            )
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        demo_fraction = self.cfg.algorithm.get("demo_fraction", 0.5)
        if self.sacflow_finetune_enabled:
            demo_fraction = self.sacflow_finetune_cfg.get(
                "demo_fraction", demo_fraction
            )
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
            demo_fraction=demo_fraction,
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        if self.sacflow_finetune_enabled and self.critic_actor_ratio != 1:
            raise ValueError(
                "Frozen-anchor SACFlow requires algorithm.critic_actor_ratio=1 "
                "so every transition-budgeted critic update has its matching "
                "warm-up or online actor update."
            )
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        self._target_shadow_f32 = {}
        for name, param in self.target_model.named_parameters():
            self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
            else:
                # DSRL path: float32 shadow buffer for bf16 precision
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1 and self.target_update_type != "all":
                        shadow = self._target_shadow_f32[name1]
                        shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    else:
                        shadow = self._target_shadow_f32[name1]
                        shadow.mul_(1.0 - tau).add_(
                            online_param.data.float(), alpha=tau
                        )
                        target_param.data.copy_(shadow.to(target_param.data.dtype))

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self._add_received_trajectories(recv_list)

    def _add_received_trajectories(self, recv_list: list[Trajectory]) -> int:
        """Insert online data and advance the shared fine-tune transition clock."""
        samples_before = self.replay_buffer.total_samples
        self.replay_buffer.add_trajectories(recv_list)
        added_samples = self.replay_buffer.total_samples - samples_before
        if self.sacflow_finetune_controller is not None:
            self.sacflow_finetune_controller.record_online_transitions(added_samples)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)
        return added_samples

    def _claim_sacflow_update_phases(
        self,
    ) -> tuple[SACFlowFinetunePhase, ...]:
        """Claim one rank-consistent UTD budget for sync or async training."""
        if self.sacflow_finetune_controller is None:
            return ()
        available_updates = self.sacflow_finetune_controller.available_updates
        if self._world_size > 1:
            available_tensor = torch.tensor(
                available_updates,
                device=self.device,
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(
                available_tensor, op=torch.distributed.ReduceOp.MIN
            )
            available_updates = int(available_tensor.item())
        if available_updates == 0:
            return ()
        return self.sacflow_finetune_controller.claim_update_phases(
            max_updates=available_updates
        )

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        if use_dsrl:
            action_horizon = resolve_model_action_horizon(self.cfg.actor.model)
            discount = self.cfg.algorithm.gamma**action_horizon
            rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            if self.sacflow_finetune_enabled:
                # Critic bootstrap must not mutate actor BatchRenorm/statistics.
                kwargs["train"] = False
                kwargs.update(
                    self._sacflow_actor_sampler_kwargs(log_prob_mode="temperature")
                )
            next_state_actions, next_state_log_pi, shared_feature = self.model(
                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
            )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=None,
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            if self.sacflow_finetune_enabled:
                dsrl_kwargs["detach_encoder"] = True
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    def _sacflow_sampling_kwargs(self, curr_obs) -> dict:
        """Create common latent and per-step noises for policy/anchor pairing."""
        if not self.sacflow_finetune_enabled or not self.sacflow_common_noise:
            return {}
        reference_tensor = next(
            value for value in curr_obs.values() if isinstance(value, torch.Tensor)
        )
        anchor_dtype = next(self.sacflow_anchor_policy.parameters()).dtype
        sampling_cfg = self.cfg.actor.model.get("flow_sampling", {})
        actor_update_cfg = sampling_cfg.get("actor_update", {})
        denoising_steps = actor_update_cfg.get(
            "num_steps", self.cfg.actor.model.get("denoising_steps", 1)
        )
        return make_common_flow_noise(
            batch_size=reference_tensor.shape[0],
            action_horizon=self.cfg.actor.model.action_horizon,
            action_dim=self.cfg.actor.model.action_dim,
            denoising_steps=denoising_steps,
            device=reference_tensor.device,
            dtype=anchor_dtype,
        )

    def _sacflow_actor_sampler_kwargs(
        self, *, log_prob_mode: str | None = "actor_surrogate"
    ) -> dict:
        """Select the configured differentiable actor-update sampler profile."""
        if not self.sacflow_finetune_enabled:
            return {}
        actor_update_cfg = self.cfg.actor.model.flow_sampling.actor_update
        kwargs = {
            "sampler_method": actor_update_cfg.method,
            "num_steps": actor_update_cfg.num_steps,
        }
        if log_prob_mode is not None:
            kwargs["log_prob_mode"] = log_prob_mode
        return kwargs

    def _forward_frozen_sacflow_anchor(self, curr_obs, sampling_kwargs):
        """Sample the immutable full-policy anchor with an explicitly shared path."""
        if self.sacflow_anchor_policy is None:
            raise RuntimeError("SACFlow frozen anchor has not been initialized.")
        anchor_kwargs = dict(sampling_kwargs)
        anchor_kwargs["train"] = False
        # The anchor contributes actions only. Avoid the extra K fixed-trace
        # field evaluations needed by the live actor's entropy-score surrogate.
        anchor_kwargs["log_prob_mode"] = "temperature"
        self.sacflow_anchor_policy.eval()
        with torch.no_grad(), self.amp_context:
            anchor_pi, _, _ = self.sacflow_anchor_policy(
                forward_type=ForwardType.SAC,
                obs=curr_obs,
                **anchor_kwargs,
            )
        return anchor_pi

    @Worker.timer("forward_actor")
    def forward_actor(
        self,
        batch,
        finetune_phase: SACFlowFinetunePhase | None = None,
    ):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        if finetune_phase is not None:
            kwargs["train"] = True
        sampling_kwargs = self._sacflow_sampling_kwargs(curr_obs)
        actor_log_prob_mode = (
            "temperature"
            if finetune_phase is SACFlowFinetunePhase.WARMUP
            else "actor_surrogate"
        )
        sampling_kwargs.update(
            self._sacflow_actor_sampler_kwargs(log_prob_mode=actor_log_prob_mode)
        )
        kwargs.update(sampling_kwargs)
        pi, log_pi, shared_feature = self.model(
            forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)  # sum over the chunk dimension
        anchor_pi = None
        if finetune_phase is not None:
            anchor_pi = self._forward_frozen_sacflow_anchor(curr_obs, sampling_kwargs)

        # Warm-up deliberately excludes Q and entropy from the actor objective.
        if finetune_phase is SACFlowFinetunePhase.WARMUP:
            actor_loss_parts = compute_frozen_anchor_actor_loss(
                policy_actions=pi,
                anchor_actions=anchor_pi,
                behavior_beta=self.sacflow_warmup_behavior_beta,
                phase=finetune_phase,
            )
            entropy = -log_pi.mean()
            return (
                actor_loss_parts.total,
                entropy,
                {
                    "anchor_loss": actor_loss_parts.behavior.item(),
                    "sac_loss": actor_loss_parts.sac.item(),
                },
            )

        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=None,
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        if finetune_phase is SACFlowFinetunePhase.ONLINE:
            actor_loss_parts = compute_frozen_anchor_actor_loss(
                policy_actions=pi,
                anchor_actions=anchor_pi,
                behavior_beta=self.sacflow_behavior_beta,
                phase=finetune_phase,
                joint_log_prob=log_pi,
                q_value=qf_pi,
                alpha=self.entropy_temp.alpha,
            )
            actor_loss = actor_loss_parts.total
            metrics["anchor_loss"] = actor_loss_parts.behavior.item()
            metrics["sac_loss"] = actor_loss_parts.sac.item()
        else:
            actor_loss = ((self.entropy_temp.alpha * log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
            if self.use_dsrl:
                kwargs["train"] = True
            if self.sacflow_finetune_enabled:
                kwargs["train"] = False
                kwargs.update(
                    self._sacflow_actor_sampler_kwargs(log_prob_mode="temperature")
                )
            _, log_pi, _ = self.model(
                forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
            )
            if log_pi.ndim == 1:
                log_pi = log_pi.unsqueeze(-1)
            log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    @Worker.timer("update_one_epoch")
    def update_one_epoch(
        self,
        train_actor: bool = True,
        finetune_phase: SACFlowFinetunePhase | None = None,
    ):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # move train_micro_batch_list to device and apply DRQ for critic/actor/alpha passes
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        if finetune_phase is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            if finetune_phase is not None:
                # Q is differentiable with respect to sampled actions, but Q-head
                # gradients from the actor loss do not belong to either actor
                # clipping or its optimizer step.
                self.qf_optimizer.zero_grad(set_to_none=True)
            gbs_actor_loss = []
            gbs_entropy = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, entropy, q_metrics = self.forward_actor(
                    batch, finetune_phase=finetune_phase
                )
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                append_to_dict(all_actor_metrics, q_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            if finetune_phase is not None:
                self.qf_optimizer.zero_grad(set_to_none=True)
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update temperature parameter if using automatic entropy tuning
            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            train_alpha = finetune_phase is not SACFlowFinetunePhase.WARMUP
            if self.alpha_optimizer is not None and train_alpha:
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                for batch in train_micro_batch_list:
                    alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
                    alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad, op=torch.distributed.ReduceOp.AVG
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                self.alpha_lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "sac/alpha_loss": np.mean(gbs_alpha_loss),
                    "sac/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/entropy": np.mean(gbs_entropy),
                    "alpha/grad_norm": alpha_grad_norm,
                    **all_actor_metrics,
                }
            )
            if finetune_phase is not None:
                metrics_data["finetune/phase"] = float(
                    finetune_phase is SACFlowFinetunePhase.ONLINE
                )
        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        if self.sacflow_finetune_controller is not None:
            append_to_dict(
                metrics,
                {
                    "finetune/online_transitions": self.sacflow_finetune_controller.online_transitions,
                    "finetune/update_credit": self.sacflow_finetune_controller.update_credit,
                    "finetune/completed_updates": self.sacflow_finetune_controller.completed_updates,
                },
            )
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @Worker.timer("run_training")
    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        if self.sacflow_finetune_controller is not None:
            finetune_phases = self._claim_sacflow_update_phases()
            if not finetune_phases:
                return {}
            train_actor = True
            self._onload_sacflow_anchor()
        else:
            num_updates = self.cfg.algorithm.get("update_epoch", 1)
            finetune_phases = (None,) * num_updates
            # Delay actor training until buffer has enough samples.
            train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
            train_actor_steps = max(min_buffer_size, train_actor_steps)
            train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        for finetune_phase in finetune_phases:
            metrics_data = self.update_one_epoch(
                train_actor=train_actor,
                finetune_phase=finetune_phase,
            )
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        if self.sacflow_finetune_enabled and self.cfg.actor.get(
            "enable_offload", False
        ):
            self._offload_sacflow_anchor()
        return mean_metric_dict

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

        if self.sacflow_finetune_controller is not None:
            anchor_save_path = os.path.join(save_base_path, "sac_components/anchor")
            os.makedirs(anchor_save_path, exist_ok=True)
            anchor_state_dict = {
                key: value.detach().cpu()
                for key, value in self.sacflow_anchor_policy.state_dict().items()
            }
            torch.save(
                anchor_state_dict,
                os.path.join(anchor_save_path, f"checkpoint_rank_{self._rank}.pt"),
            )

            finetune_save_path = os.path.join(
                save_base_path, "sac_components/sacflow_finetune"
            )
            os.makedirs(finetune_save_path, exist_ok=True)
            torch.save(
                {
                    "update_step": self.update_step,
                    "controller": self.sacflow_finetune_controller.state_dict(),
                },
                os.path.join(finetune_save_path, f"state_rank_{self._rank}.pt"),
            )

    def load_checkpoint(self, load_base_path):
        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)

        if self.sacflow_finetune_controller is not None:
            anchor_state_file = os.path.join(
                load_base_path,
                "sac_components/anchor",
                f"checkpoint_rank_{self._rank}.pt",
            )
            finetune_state_file = os.path.join(
                load_base_path,
                "sac_components/sacflow_finetune",
                f"state_rank_{self._rank}.pt",
            )
            if not os.path.isfile(anchor_state_file) or not os.path.isfile(
                finetune_state_file
            ):
                raise FileNotFoundError(
                    "A frozen-anchor SACFlow resume requires both the anchor and "
                    "phase state saved by the fine-tuning worker. Missing one of: "
                    f"{anchor_state_file}, {finetune_state_file}."
                )

            anchor_state_dict = torch.load(
                anchor_state_file, map_location="cpu", weights_only=True
            )
            self.sacflow_anchor_policy.load_state_dict(anchor_state_dict, strict=True)
            self.sacflow_anchor_policy.requires_grad_(False)
            self.sacflow_anchor_policy.eval()

            finetune_state = torch.load(
                finetune_state_file, map_location="cpu", weights_only=True
            )
            self.update_step = int(finetune_state["update_step"])
            self.sacflow_finetune_controller.load_state_dict(
                finetune_state["controller"]
            )
            if self.sacflow_finetune_controller.completed_updates != self.update_step:
                raise ValueError(
                    "SACFlow checkpoint is inconsistent: worker update_step="
                    f"{self.update_step}, controller completed_updates="
                    f"{self.sacflow_finetune_controller.completed_updates}."
                )
            if self.cfg.actor.get("enable_offload", False):
                self._offload_sacflow_anchor()
            else:
                self._onload_sacflow_anchor()
