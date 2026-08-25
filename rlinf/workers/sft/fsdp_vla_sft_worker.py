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
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.data.utils import forward_set_epoch
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.utils import clear_memory, get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(data_paths)
            if repo_id is None:
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif (
            SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.FLOW_POLICY
        ):
            from rlinf.data.datasets.flow import (
                build_franka_gello_flow_dataloader,
            )

            return build_franka_gello_flow_dataloader(
                self.cfg,
                self._world_size,
                self._rank,
                data_paths,
                eval_dataset,
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def model_provider_func(self):
        model = super().model_provider_func()
        if self._uses_canonical_flow_bc():
            self._prepare_flow_bc_model(model)
        return model

    def _prepare_flow_bc_model(self, model: torch.nn.Module) -> None:
        """Validate the canonical data/model contract before FSDP wrapping."""
        from rlinf.utils.flow_actor_checkpoint import (
            build_flow_actor_metadata_from_config,
        )
        from rlinf.utils.flow_bc_contract import resolve_flow_bc_spec

        model_cfg = self.cfg.actor.model
        spec = resolve_flow_bc_spec(model_cfg, task_type="sft")
        data_config = getattr(self, "data_config", None)
        if not isinstance(data_config, Mapping):
            raise RuntimeError(
                "Flow BC requires training data_config before model/FSDP "
                "initialization."
            )
        configured_horizon = data_config.get("action_horizon", None)
        if configured_horizon != spec.action_horizon:
            raise RuntimeError(
                "Flow BC data/model action horizon mismatch: "
                f"data={configured_horizon!r}, model={spec.action_horizon}."
            )
        expected_shape = [spec.action_horizon, spec.action_dim]
        if data_config.get("action_shape") != expected_shape:
            raise RuntimeError(
                "Flow BC data/model action shape mismatch: "
                f"data={data_config.get('action_shape')!r}, model={expected_shape}."
            )
        if data_config.get("action_layout") != "chunk_major":
            raise RuntimeError("Flow BC requires data action_layout='chunk_major'.")
        if data_config.get("action_padding") != "zero":
            raise RuntimeError("Flow BC requires zero-padded invalid actions.")
        if data_config.get("action_valid_mask") is not True:
            raise RuntimeError("Flow BC requires action_valid_mask for every horizon.")

        metadata = build_flow_actor_metadata_from_config(model_cfg)
        self._flow_actor_checkpoint_metadata = metadata

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def _uses_canonical_flow_bc(self) -> bool:
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            != SupportedModel.FLOW_POLICY
        ):
            return False
        matching = self.cfg.actor.model.get("flow_matching", {}) or {}
        return matching.get("implementation") == "pytorch_flow_t"

    def _next_training_batch(self) -> Any:
        try:
            batch = next(self.data_iter)
            self._data_iter_offset += 1
            return batch
        except StopIteration:
            self._data_epoch += 1
            logging.info(
                "[INFO] data_iter exhausted, reset iterator self._data_epoch %s",
                self._data_epoch,
            )
            forward_set_epoch(self.data_loader, self._data_epoch)
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
            self._data_iter_offset = 1
            return batch

    @staticmethod
    def _flow_bc_batch_counts(batch: Mapping[str, Any]) -> tuple[int, int]:
        action = batch.get("action")
        valid_mask = batch.get("action_valid_mask")
        if not torch.is_tensor(action) or action.ndim != 3:
            raise ValueError("Flow BC batch action must have shape [B, H, A].")
        if not torch.is_floating_point(action):
            raise ValueError("Flow BC batch action must be floating point.")
        if (
            not torch.is_tensor(valid_mask)
            or valid_mask.dtype != torch.bool
            or tuple(valid_mask.shape) != tuple(action.shape[:2])
        ):
            raise ValueError(
                "Flow BC action_valid_mask must be bool with shape [B, H]."
            )
        if torch.any(~valid_mask.any(dim=1)):
            raise ValueError("Every Flow BC sample must contain a valid h0 action.")
        if valid_mask.shape[1] > 1 and torch.any(
            valid_mask[:, 1:] & ~valid_mask[:, :-1]
        ):
            raise ValueError("Flow BC action_valid_mask must be prefix-valid.")
        element_mask = valid_mask.unsqueeze(-1).expand_as(action)
        if not torch.isfinite(action.masked_select(element_mask)).all():
            raise ValueError("Flow BC valid action coordinates must be finite.")
        if torch.any(action.masked_select(~element_mask) != 0):
            raise ValueError("Flow BC invalid action coordinates must be zero.")
        valid_coordinate_count = int(valid_mask.sum().item()) * int(action.shape[-1])
        return valid_coordinate_count, int(action.shape[0])

    @staticmethod
    def _distributed_sum(value: float, device: torch.device) -> float:
        total = torch.tensor(value, dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return float(total.item())

    @staticmethod
    def _distributed_max(value: float, device: torch.device) -> float:
        maximum = torch.tensor(value, dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        return float(maximum.item())

    def _flow_bc_reduce_device(self, batch: Mapping[str, Any]) -> torch.device:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.device(
                self.torch_device_type,
                self.torch_platform.current_device(),
            )
        action = batch.get("action")
        if torch.is_tensor(action):
            return action.device
        return torch.device("cpu")

    def _aggregate_flow_bc_metrics(
        self,
        metric_batches: list[tuple[dict[str, Any], int, int]],
        *,
        reduce_device: torch.device,
    ) -> dict[str, float]:
        numerators: dict[str, float] = {}
        denominators: dict[str, float] = {}
        explicit_counts: dict[str, float] = {}
        maxima: dict[str, float] = {}
        valid_weighted = {
            "loss",
            "field_mse",
            "adaptive_denominator_mean",
            "action_clamp_fraction",
        }
        rms_weighted = {
            "target_norm",
            "field_norm",
            "boundary_field_norm",
            "material_derivative_norm",
            "target_rms",
            "field_rms",
        }
        max_reduced = {"action_clamp_abs_max"}

        def count_name(metric_name: str) -> str | None:
            if metric_name.startswith("field_mse_h"):
                horizon = metric_name.removeprefix("field_mse_h")
                return f"field_count_h{horizon}"
            return f"{metric_name}_count"

        def is_count(metric_name: str) -> bool:
            return metric_name.endswith("_count") or metric_name.startswith(
                "field_count_h"
            )

        for metrics, valid_count, batch_size in metric_batches:
            for name, value in metrics.items():
                if is_count(name):
                    explicit_counts[name] = explicit_counts.get(name, 0.0) + float(
                        value
                    )
                    continue
                if name in max_reduced:
                    maxima[name] = max(maxima.get(name, -math.inf), float(value))
                    continue
                metric_count_name = count_name(name)
                if metric_count_name is not None and metric_count_name in metrics:
                    weight = float(metrics[metric_count_name])
                elif name in valid_weighted or name in rms_weighted:
                    weight = float(valid_count)
                else:
                    weight = float(batch_size)
                weighted_value = float(value)
                if name in rms_weighted:
                    weighted_value = weighted_value**2
                numerators[name] = numerators.get(name, 0.0) + weighted_value * weight
                denominators[name] = denominators.get(name, 0.0) + weight

        reduced: dict[str, float] = {}
        for name, numerator in numerators.items():
            total_numerator = self._distributed_sum(numerator, reduce_device)
            total_denominator = self._distributed_sum(denominators[name], reduce_device)
            value = total_numerator / max(total_denominator, 1.0)
            reduced[name] = math.sqrt(value) if name in rms_weighted else value
        for name, value in explicit_counts.items():
            reduced[name] = self._distributed_sum(value, reduce_device)
        for name, value in maxima.items():
            reduced[name] = self._distributed_max(value, reduce_device)
        return reduced

    def run_training(self):
        if not self._uses_canonical_flow_bc():
            return super().run_training()

        with self.worker_timer():
            self.model.train()
            batches = [
                self._next_training_batch() for _ in range(self.gradient_accumulation)
            ]
            counts = [self._flow_bc_batch_counts(batch) for batch in batches]
            reduce_device = self._flow_bc_reduce_device(batches[0])
            global_valid_count = self._distributed_sum(
                sum(valid_count for valid_count, _ in counts),
                reduce_device,
            )
            if global_valid_count <= 0:
                raise RuntimeError("Flow BC optimizer batch has no valid actions.")
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )

            metric_batches: list[tuple[dict[str, Any], int, int]] = []
            for idx, (batch, (valid_count, batch_size)) in enumerate(
                zip(batches, counts, strict=True)
            ):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )
                loss, step_metrics = self.get_train_model_output(batch)
                reported_valid_count = step_metrics.get("valid_coordinate_count")
                if reported_valid_count is None:
                    raise RuntimeError(
                        "Flow BC objective must report valid_coordinate_count."
                    )
                if float(reported_valid_count) != float(valid_count):
                    raise RuntimeError(
                        "Flow BC objective valid-coordinate count does not match "
                        f"the batch contract: objective={reported_valid_count}, "
                        f"batch={valid_count}."
                    )
                metric_batches.append((step_metrics, valid_count, batch_size))
                loss_scale = world_size * valid_count / global_valid_count
                with backward_ctx:
                    self.grad_scaler.scale(loss * loss_scale).backward()

            grad_norm, _ = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            train_metrics = self._aggregate_flow_bc_metrics(
                metric_batches,
                reduce_device=reduce_device,
            )
            train_metrics.update(
                all_reduce_dict(
                    {
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": (
                            float(grad_norm)
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        ),
                    },
                    op=torch.distributed.ReduceOp.AVG,
                )
            )

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()
            return train_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        is_flow_bc = self._uses_canonical_flow_bc()
        if is_flow_bc and not self.cfg.actor.fsdp_config.get(
            "save_full_model_weights", True
        ):
            raise ValueError(
                "Flow BC requires actor.fsdp_config.save_full_model_weights=true "
                "to export its portable actor checkpoint."
            )
        super().save_checkpoint(save_path, step)

        if is_flow_bc:
            export_error = [None]
            if self._rank == 0:
                try:
                    from rlinf.utils.flow_actor_checkpoint import (
                        export_flow_actor_checkpoint,
                    )

                    metadata = getattr(
                        self,
                        "_flow_actor_checkpoint_metadata",
                        None,
                    )
                    if metadata is None:
                        raise RuntimeError(
                            "Flow BC runtime checkpoint metadata was not "
                            "initialized before FSDP wrapping."
                        )
                    export_flow_actor_checkpoint(
                        os.path.join(save_path, "model_state_dict", "full_weights.pt"),
                        os.path.join(save_path, "flow_actor"),
                        metadata=metadata,
                        overwrite=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    export_error[0] = f"{type(exc).__name__}: {exc}"
            torch.distributed.broadcast_object_list(export_error, src=0)
            if export_error[0] is not None:
                raise RuntimeError(
                    f"Failed to export Flow BC actor-only checkpoint: {export_error[0]}"
                )

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        if self._uses_canonical_flow_bc():
            from rlinf.utils.flow_actor_checkpoint import (
                read_flow_actor_checkpoint_metadata,
            )

            runtime_metadata = getattr(
                self,
                "_flow_actor_checkpoint_metadata",
                None,
            )
            if runtime_metadata is None:
                raise RuntimeError(
                    "Flow BC runtime checkpoint metadata must be initialized before "
                    "resuming."
                )
            read_flow_actor_checkpoint_metadata(
                Path(load_path) / "flow_actor",
                expected_metadata=runtime_metadata,
            )

        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
