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

"""Evaluate a trained Franka + GELLO Flow BC checkpoint on a static split.

The evaluation is static: it does not start Ray, construct an optimizer, or
modify the checkpoint. Flow policies are generative, so the report separates
the expected performance of one random deployment sample, the mean of repeated
samples, a zero-noise prediction, and a whole-chunk best-of-K diagnostic.

Example:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
     examples/sft/evaluate_franka_gello_flow_bc.py \
     --checkpoint /path/to/global_step_N
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlinf.data.datasets.flow import FrankaGelloFlowDataset
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.flow_policy import get_model
from rlinf.utils.flow_actor_checkpoint import (
    build_flow_actor_metadata_from_config,
    load_flow_actor_checkpoint,
)
from rlinf.utils.flow_bc_contract import FlowBCSpec, resolve_flow_bc_spec

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples/sft/config/franka_gello_flow_bc.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/flow_bc/evaluations/flow_bc_static_eval_val.json"
ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Portable actor directory, actor checkpoint, or global_step_N root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root; defaults to the config's data.train_data_paths.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Episode split to evaluate; validation is the default.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-observation", type=int, default=8)
    parser.add_argument(
        "--sampler-method",
        choices=("flow_ode", "flow_sde"),
        default=None,
        help="Override actor.model.flow_sampling.evaluation.method.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override actor.model.flow_sampling.evaluation.num_steps.",
    )
    parser.add_argument("--sde-noise-level", type=float, default=None)
    parser.add_argument(
        "--sde-noise-std-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Override the iMF SDE transition standard-deviation range.",
    )
    parser.add_argument("--sde-safe-initial-time", type=float, default=None)
    parser.add_argument(
        "--objective-draws",
        type=int,
        default=1,
        help="Random objective evaluations per batch; use 0 to skip field metrics.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N frames; defaults to the complete dataset.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--initial-noise-seed",
        type=int,
        default=None,
        help="Initial latent seed; defaults to --seed.",
    )
    parser.add_argument(
        "--step-noise-seed",
        type=int,
        default=None,
        help="SDE transition-noise seed; defaults to --seed + 1.",
    )
    parser.add_argument("--good-fit-r2", type=float, default=0.5)
    parser.add_argument("--good-fit-pose-r2", type=float, default=0.5)
    parser.add_argument("--good-fit-pose-rmse", type=float, default=0.1)
    parser.add_argument("--good-fit-gripper-accuracy", type=float, default=0.95)
    parser.add_argument(
        "--require-good-fit",
        action="store_true",
        help="Exit nonzero when the configured good-fit checks do not all pass.",
    )
    return parser.parse_args()


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


@dataclass(frozen=True)
class StaticSamplingSettings:
    """Resolved runtime sampler settings for one static evaluation."""

    method: str
    num_steps: int
    noise_level: float | None = None
    noise_std_range: tuple[float, float] | None = None
    safe_initial_time: float | None = None


def _resolve_sampling_settings(
    spec: FlowBCSpec,
    args: argparse.Namespace,
) -> StaticSamplingSettings:
    """Apply CLI overrides without making sampling part of checkpoint identity."""
    if spec.evaluation is None:
        raise ValueError("Flow BC requires flow_sampling.evaluation.")
    method = args.sampler_method or spec.evaluation.method
    num_steps = args.num_steps or spec.evaluation.num_steps
    if not 1 <= num_steps <= 99:
        raise ValueError(f"num_steps must be in [1, 99], got {num_steps}.")
    sde_overridden = any(
        value is not None
        for value in (
            args.sde_noise_level,
            args.sde_noise_std_range,
            args.sde_safe_initial_time,
        )
    )
    if method == "flow_ode":
        if sde_overridden:
            raise ValueError("SDE overrides require --sampler-method=flow_sde.")
        return StaticSamplingSettings(method=method, num_steps=num_steps)

    configured_sde = spec.flow_sde
    noise_level = (
        args.sde_noise_level
        if args.sde_noise_level is not None
        else (configured_sde.noise_level if configured_sde is not None else 0.1)
    )
    if not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("flow_sde noise_level must be finite and positive.")
    if spec.objective == "rectified_flow":
        if (
            args.sde_noise_std_range is not None
            or args.sde_safe_initial_time is not None
        ):
            raise ValueError(
                "Rectified Flow SDE derives its standard deviation from noise_level; "
                "noise_std_range and safe_initial_time are iMF-only."
            )
        return StaticSamplingSettings(
            method=method,
            num_steps=num_steps,
            noise_level=noise_level,
        )

    configured_range = (
        configured_sde.noise_std_range if configured_sde is not None else None
    )
    raw_range = args.sde_noise_std_range or configured_range or (0.01, 0.1)
    std_min, std_max = (float(raw_range[0]), float(raw_range[1]))
    if not (
        math.isfinite(std_min) and math.isfinite(std_max) and 0 < std_min <= std_max
    ):
        raise ValueError("iMF flow_sde requires 0 < std_min <= std_max.")
    safe_initial_time = (
        args.sde_safe_initial_time
        if args.sde_safe_initial_time is not None
        else (configured_sde.safe_initial_time if configured_sde is not None else 0.99)
    )
    if not (
        math.isfinite(safe_initial_time)
        and 0 < safe_initial_time < 1
        and safe_initial_time > 1 - 1 / num_steps
    ):
        raise ValueError(
            "iMF flow_sde safe_initial_time must be in (0, 1) and strictly "
            "greater than 1-1/num_steps."
        )
    return StaticSamplingSettings(
        method=method,
        num_steps=num_steps,
        noise_level=noise_level,
        noise_std_range=(std_min, std_max),
        safe_initial_time=safe_initial_time,
    )


def _configure_runtime_sampler(
    model: torch.nn.Module,
    settings: StaticSamplingSettings,
) -> None:
    """Apply validated SDE overrides to non-parameter actor attributes."""
    if settings.method != "flow_sde":
        return
    model.flow_actor.noise_level = settings.noise_level
    if settings.noise_std_range is not None:
        model.flow_actor.sde_std_min, model.flow_actor.sde_std_max = (
            settings.noise_std_range
        )
    if settings.safe_initial_time is not None:
        model.flow_actor.safe_initial_time = settings.safe_initial_time


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested through --device={requested}, but unavailable."
        )
    return device


def _resolve_dataset_path(data_root: Path) -> Path:
    """Resolve a collection root or finalized LeRobot path."""
    root = data_root.expanduser().resolve()
    collected_data = root / "collected_data"
    if collected_data.is_dir():
        return collected_data
    if root.is_dir():
        return root
    raise FileNotFoundError(f"Training data path does not exist: {root}")


def _configured_data_root(requested: Path | None, cfg: Any) -> Path:
    """Resolve one dataset root, with an explicit CLI path taking precedence."""
    if requested is not None:
        return requested

    configured = cfg.data.get("train_data_paths", None)
    if isinstance(configured, Mapping):
        configured = configured.get("dataset_path", configured.get("data_path"))
    elif isinstance(configured, Sequence) and not isinstance(configured, str):
        if len(configured) != 1:
            raise ValueError(
                "Static evaluation requires exactly one data.train_data_paths entry "
                "when --data-root is omitted."
            )
        configured = configured[0]
        if isinstance(configured, Mapping):
            configured = configured.get(
                "dataset_path",
                configured.get("data_path"),
            )
    if not isinstance(configured, (str, Path)) or not str(configured):
        raise ValueError(
            "Could not resolve data.train_data_paths; pass --data-root explicitly."
        )
    return Path(str(configured))


def _resolve_portable_checkpoint(checkpoint: Path) -> Path:
    """Resolve global-step, actor, or portable actor checkpoint directories."""
    root = checkpoint.expanduser().resolve()
    candidates = (
        root / "actor" / "flow_actor",
        root / "flow_actor",
        root,
    )
    for candidate in candidates:
        if (candidate / "flow_actor_manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find flow_actor_manifest.json under checkpoint path "
        f"{root}; tried {[str(path) for path in candidates]}."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _slice_batch(batch: Mapping[str, Any], size: int) -> dict[str, Any]:
    def slice_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value[:size]
        if isinstance(value, Mapping):
            return {key: slice_value(item) for key, item in value.items()}
        return value

    return {key: slice_value(value) for key, value in batch.items()}


def _canonical_action_chunk(
    actions: torch.Tensor,
    *,
    action_dim: int,
    action_horizon: int,
    name: str,
) -> torch.Tensor:
    expected = (action_horizon, action_dim)
    if actions.ndim != 3 or tuple(actions.shape[1:]) != expected:
        raise ValueError(
            f"{name} must have shape [B, {action_horizon}, {action_dim}] "
            f"for every action horizon, got {tuple(actions.shape)}."
        )
    return actions


def _canonical_valid_mask(
    valid_mask: torch.Tensor | None,
    *,
    batch_size: int,
    action_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(
            batch_size,
            action_horizon,
            dtype=torch.bool,
            device=device,
        )
    expected_shape = (batch_size, action_horizon)
    if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != expected_shape:
        raise ValueError(
            f"valid_mask must be bool with shape {expected_shape}, got "
            f"dtype={valid_mask.dtype}, shape={tuple(valid_mask.shape)}."
        )
    valid_mask = valid_mask.to(device=device)
    if torch.any(~valid_mask.any(dim=1)):
        raise ValueError("Every observation must have at least one valid action step.")
    if torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1]):
        raise ValueError("valid_mask must be prefix-valid with tail padding only.")
    return valid_mask


def _canonical_sample_chunk(
    samples: torch.Tensor,
    *,
    action_dim: int,
    action_horizon: int,
) -> torch.Tensor:
    expected = (action_horizon, action_dim)
    if samples.ndim != 4 or tuple(samples.shape[2:]) != expected:
        raise ValueError(
            f"samples must have shape [B, K, {action_horizon}, {action_dim}] "
            f"for every action horizon, got {tuple(samples.shape)}."
        )
    if samples.shape[1] <= 0:
        raise ValueError("samples must contain at least one random draw.")
    return samples


def _select_best_chunk_sample(
    samples: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one K sample per observation using whole-valid-chunk MSE."""
    action_dim = target.shape[-1]
    action_horizon = target.shape[1] if target.ndim == 3 else 1
    target = _canonical_action_chunk(
        target,
        action_dim=action_dim,
        action_horizon=action_horizon,
        name="target",
    )
    samples = _canonical_sample_chunk(
        samples,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    if samples.shape[0] != target.shape[0]:
        raise ValueError("samples and target batch sizes must match.")
    valid_mask = _canonical_valid_mask(
        valid_mask,
        batch_size=target.shape[0],
        action_horizon=action_horizon,
        device=target.device,
    )
    coordinate_mask = valid_mask[:, None, :, None]
    squared_error = torch.where(
        coordinate_mask,
        (samples - target[:, None]).square(),
        torch.zeros((), device=samples.device, dtype=samples.dtype),
    )
    denominator = valid_mask.sum(dim=1) * action_dim
    chunk_mse = squared_error.sum(dim=(2, 3)) / denominator[:, None]
    best_indices = chunk_mse.argmin(dim=1)
    best = samples[torch.arange(target.shape[0], device=samples.device), best_indices]
    return best, best_indices


@dataclass
class TargetStatistics:
    """Streaming valid-target moments for aggregate and horizon baselines."""

    action_dim: int
    action_horizon: int = 1
    count: int = 0
    horizon_count: torch.Tensor = field(init=False)
    value_sum: torch.Tensor = field(init=False)
    square_sum: torch.Tensor = field(init=False)
    minimum: torch.Tensor = field(init=False)
    maximum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        _require_positive(self.action_dim, "action_dim")
        _require_positive(self.action_horizon, "action_horizon")
        shape = (self.action_horizon, self.action_dim)
        self.horizon_count = torch.zeros(self.action_horizon, dtype=torch.int64)
        self.value_sum = torch.zeros(shape, dtype=torch.float64)
        self.square_sum = torch.zeros(shape, dtype=torch.float64)
        self.minimum = torch.full(shape, math.inf, dtype=torch.float64)
        self.maximum = torch.full(shape, -math.inf, dtype=torch.float64)

    def update(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        target = _canonical_action_chunk(
            target.detach().to(device="cpu", dtype=torch.float64),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            name="target",
        )
        valid_mask = _canonical_valid_mask(
            None if valid_mask is None else valid_mask.detach().to(device="cpu"),
            batch_size=target.shape[0],
            action_horizon=self.action_horizon,
            device=target.device,
        )
        mask = valid_mask.unsqueeze(-1)
        masked_target = torch.where(mask, target, torch.zeros_like(target))
        counts = valid_mask.sum(dim=0)
        self.count += int(counts.sum())
        self.horizon_count += counts
        self.value_sum += masked_target.sum(dim=0)
        self.square_sum += masked_target.square().sum(dim=0)
        for horizon in range(self.action_horizon):
            if counts[horizon] > 0:
                values = target[valid_mask[:, horizon], horizon]
                self.minimum[horizon] = torch.minimum(
                    self.minimum[horizon], values.amin(dim=0)
                )
                self.maximum[horizon] = torch.maximum(
                    self.maximum[horizon], values.amax(dim=0)
                )

    def moments(
        self, horizon: int | None = None
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if horizon is None:
            return (
                self.count,
                self.value_sum.sum(dim=0),
                self.square_sum.sum(dim=0),
                self.minimum.amin(dim=0),
                self.maximum.amax(dim=0),
            )
        return (
            int(self.horizon_count[horizon]),
            self.value_sum[horizon],
            self.square_sum[horizon],
            self.minimum[horizon],
            self.maximum[horizon],
        )

    def centered_square_sum(self, horizon: int | None = None) -> torch.Tensor:
        count, value_sum, square_sum, _, _ = self.moments(horizon)
        if count == 0:
            raise ValueError("Cannot summarize empty target statistics.")
        return (square_sum - value_sum.square() / count).clamp_min(0.0)


@dataclass
class RegressionAccumulator:
    """Accumulate valid chunk prediction errors without retaining samples."""

    action_dim: int
    action_horizon: int = 1
    count: int = 0
    horizon_count: torch.Tensor = field(init=False)
    absolute_error_sum: torch.Tensor = field(init=False)
    square_error_sum: torch.Tensor = field(init=False)
    maximum_absolute_error: torch.Tensor = field(init=False)
    within_005: torch.Tensor = field(init=False)
    within_010: torch.Tensor = field(init=False)
    gripper_sign_correct: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        _require_positive(self.action_dim, "action_dim")
        _require_positive(self.action_horizon, "action_horizon")
        shape = (self.action_horizon, self.action_dim)
        self.horizon_count = torch.zeros(self.action_horizon, dtype=torch.int64)
        self.absolute_error_sum = torch.zeros(shape, dtype=torch.float64)
        self.square_error_sum = torch.zeros(shape, dtype=torch.float64)
        self.maximum_absolute_error = torch.zeros(shape, dtype=torch.float64)
        self.within_005 = torch.zeros(shape, dtype=torch.int64)
        self.within_010 = torch.zeros(shape, dtype=torch.int64)
        self.gripper_sign_correct = torch.zeros(self.action_horizon, dtype=torch.int64)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        prediction = _canonical_action_chunk(
            prediction.detach().to(device="cpu", dtype=torch.float64),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            name="prediction",
        )
        target = _canonical_action_chunk(
            target.detach().to(device="cpu", dtype=torch.float64),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            name="target",
        )
        if prediction.shape != target.shape:
            raise ValueError("prediction and target chunk shapes must match.")
        valid_mask = _canonical_valid_mask(
            None if valid_mask is None else valid_mask.detach().to(device="cpu"),
            batch_size=target.shape[0],
            action_horizon=self.action_horizon,
            device=target.device,
        )
        absolute_error = (prediction - target).abs()
        mask = valid_mask.unsqueeze(-1)
        masked_error = torch.where(
            mask,
            absolute_error,
            torch.zeros_like(absolute_error),
        )
        counts = valid_mask.sum(dim=0)
        self.count += int(counts.sum())
        self.horizon_count += counts
        self.absolute_error_sum += masked_error.sum(dim=0)
        self.square_error_sum += masked_error.square().sum(dim=0)
        self.within_005 += ((absolute_error <= 0.05) & mask).sum(dim=0)
        self.within_010 += ((absolute_error <= 0.10) & mask).sum(dim=0)
        for horizon in range(self.action_horizon):
            if counts[horizon] > 0:
                horizon_error = absolute_error[valid_mask[:, horizon], horizon]
                self.maximum_absolute_error[horizon] = torch.maximum(
                    self.maximum_absolute_error[horizon],
                    horizon_error.amax(dim=0),
                )
        if self.action_dim == len(ACTION_NAMES):
            sign_correct = (prediction[..., -1] >= 0) == (target[..., -1] >= 0)
            self.gripper_sign_correct += (sign_correct & valid_mask).sum(dim=0)

    @staticmethod
    def _r2(error_ss: float, target_ss: float) -> float | None:
        if target_ss <= 0.0:
            return None
        return 1.0 - error_ss / target_ss

    def _group_summary(
        self,
        indices: slice,
        *,
        count: int,
        absolute_error_sum: torch.Tensor,
        square_error_sum: torch.Tensor,
        maximum_absolute_error: torch.Tensor,
        within_005: torch.Tensor,
        within_010: torch.Tensor,
        target_centered_ss: torch.Tensor,
    ) -> dict[str, float | None]:
        absolute_sum = float(absolute_error_sum[indices].sum())
        square_sum = float(square_error_sum[indices].sum())
        centered_sum = float(target_centered_ss[indices].sum())
        dimensions = absolute_error_sum[indices].numel()
        denominator = count * dimensions
        mse = square_sum / denominator
        return {
            "mae": absolute_sum / denominator,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "max_absolute_error": float(maximum_absolute_error[indices].max()),
            "within_0.05_fraction": float(within_005[indices].sum()) / denominator,
            "within_0.10_fraction": float(within_010[indices].sum()) / denominator,
            "mean_baseline_mse": centered_sum / denominator,
            "r2": self._r2(square_sum, centered_sum),
        }

    def _summarize_scope(
        self,
        target_statistics: TargetStatistics,
        horizon: int | None,
    ) -> dict[str, Any]:
        target_count, target_sum, _, target_min, target_max = target_statistics.moments(
            horizon
        )
        if horizon is None:
            count = self.count
            absolute_error_sum = self.absolute_error_sum.sum(dim=0)
            square_error_sum = self.square_error_sum.sum(dim=0)
            maximum_absolute_error = self.maximum_absolute_error.amax(dim=0)
            within_005 = self.within_005.sum(dim=0)
            within_010 = self.within_010.sum(dim=0)
            gripper_sign_correct = int(self.gripper_sign_correct.sum())
        else:
            count = int(self.horizon_count[horizon])
            absolute_error_sum = self.absolute_error_sum[horizon]
            square_error_sum = self.square_error_sum[horizon]
            maximum_absolute_error = self.maximum_absolute_error[horizon]
            within_005 = self.within_005[horizon]
            within_010 = self.within_010[horizon]
            gripper_sign_correct = int(self.gripper_sign_correct[horizon])
        if count == 0 or count != target_count:
            raise ValueError(
                "Prediction and target scope counts must match and be nonzero."
            )
        target_centered_ss = target_statistics.centered_square_sum(horizon)
        per_dimension = []
        for index in range(self.action_dim):
            mse = float(square_error_sum[index] / count)
            target_ss = float(target_centered_ss[index])
            per_dimension.append(
                {
                    "name": (
                        ACTION_NAMES[index]
                        if self.action_dim == len(ACTION_NAMES)
                        else str(index)
                    ),
                    "target_mean": float(target_sum[index] / count),
                    "target_min": float(target_min[index]),
                    "target_max": float(target_max[index]),
                    "mae": float(absolute_error_sum[index] / count),
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "mean_baseline_mse": float(target_centered_ss[index] / count),
                    "r2": self._r2(float(square_error_sum[index]), target_ss),
                }
            )
        group_kwargs = {
            "count": count,
            "absolute_error_sum": absolute_error_sum,
            "square_error_sum": square_error_sum,
            "maximum_absolute_error": maximum_absolute_error,
            "within_005": within_005,
            "within_010": within_010,
            "target_centered_ss": target_centered_ss,
        }
        summary: dict[str, Any] = {
            "valid_steps": count,
            "overall": self._group_summary(slice(0, self.action_dim), **group_kwargs),
            "per_dimension": per_dimension,
        }
        if self.action_dim == len(ACTION_NAMES):
            summary["pose_6d"] = self._group_summary(slice(0, 6), **group_kwargs)
            summary["gripper"] = {
                **self._group_summary(slice(6, 7), **group_kwargs),
                "sign_accuracy": gripper_sign_correct / count,
            }
        return summary

    def summarize(self, target_statistics: TargetStatistics) -> dict[str, Any]:
        if (
            self.action_horizon != target_statistics.action_horizon
            or self.count == 0
            or self.count != target_statistics.count
            or not torch.equal(self.horizon_count, target_statistics.horizon_count)
        ):
            raise ValueError(
                "Prediction and target statistics must have matching nonzero "
                "aggregate and per-horizon counts."
            )
        summary = self._summarize_scope(target_statistics, None)
        per_horizon = []
        for horizon in range(self.action_horizon):
            if self.horizon_count[horizon] == 0:
                per_horizon.append(
                    {"horizon_index": horizon, "valid_steps": 0, "available": False}
                )
            else:
                per_horizon.append(
                    {
                        "horizon_index": horizon,
                        "available": True,
                        **self._summarize_scope(target_statistics, horizon),
                    }
                )
        summary["per_horizon"] = per_horizon
        return summary


@dataclass
class AllValidChunkRegressionAccumulator:
    """Accumulate regression metrics for samples with a complete valid chunk."""

    action_dim: int
    action_horizon: int = 1
    valid_chunks: int = 0
    target_statistics: TargetStatistics = field(init=False)
    regression: RegressionAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self.target_statistics = TargetStatistics(
            self.action_dim,
            self.action_horizon,
        )
        self.regression = RegressionAccumulator(
            self.action_dim,
            self.action_horizon,
        )

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        prediction = _canonical_action_chunk(
            prediction.detach().to(device="cpu"),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            name="prediction",
        )
        target = _canonical_action_chunk(
            target.detach().to(device="cpu"),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            name="target",
        )
        if prediction.shape != target.shape:
            raise ValueError("prediction and target chunk shapes must match.")
        valid_mask = _canonical_valid_mask(
            valid_mask.detach().to(device="cpu"),
            batch_size=target.shape[0],
            action_horizon=self.action_horizon,
            device=target.device,
        )
        all_valid = valid_mask.all(dim=1)
        chunk_count = int(all_valid.sum())
        if chunk_count == 0:
            return

        prediction = prediction[all_valid]
        target = target[all_valid]
        full_mask = torch.ones(
            chunk_count,
            self.action_horizon,
            dtype=torch.bool,
        )
        self.target_statistics.update(target, full_mask)
        self.regression.update(prediction, target, full_mask)
        self.valid_chunks += chunk_count

    def summarize(self) -> dict[str, Any]:
        if self.valid_chunks == 0:
            return {
                "available": False,
                "valid_chunks": 0,
                "valid_steps": 0,
            }
        return {
            "available": True,
            "valid_chunks": self.valid_chunks,
            **self.regression.summarize(self.target_statistics),
        }


@dataclass
class SampleDiagnostics:
    """Accumulate valid uncertainty and coverage for chunked Flow samples."""

    action_dim: int
    action_horizon: int = 1
    count: int = 0
    horizon_count: torch.Tensor = field(init=False)
    standard_deviation_sum: torch.Tensor = field(init=False)
    envelope_covered: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.action_horizon, self.action_dim)
        self.horizon_count = torch.zeros(self.action_horizon, dtype=torch.int64)
        self.standard_deviation_sum = torch.zeros(shape, dtype=torch.float64)
        self.envelope_covered = torch.zeros(shape, dtype=torch.int64)

    def update(
        self,
        samples: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        samples = _canonical_sample_chunk(
            samples.detach().to(device="cpu", dtype=torch.float64),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )
        target = _canonical_action_chunk(
            target.detach().to(device="cpu", dtype=torch.float64),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            name="target",
        )
        if samples.shape[0] != target.shape[0]:
            raise ValueError("samples and target batch sizes must match.")
        valid_mask = _canonical_valid_mask(
            None if valid_mask is None else valid_mask.detach().to(device="cpu"),
            batch_size=target.shape[0],
            action_horizon=self.action_horizon,
            device=target.device,
        )
        mask = valid_mask.unsqueeze(-1)
        counts = valid_mask.sum(dim=0)
        self.count += int(counts.sum())
        self.horizon_count += counts
        sample_std = samples.std(dim=1, unbiased=False)
        covered = (target >= samples.amin(dim=1)) & (target <= samples.amax(dim=1))
        masked_std = torch.where(mask, sample_std, torch.zeros_like(sample_std))
        self.standard_deviation_sum += masked_std.sum(dim=0)
        self.envelope_covered += (covered & mask).sum(dim=0)

    def summarize(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("Cannot summarize empty sample diagnostics.")
        aggregate_std = self.standard_deviation_sum.sum(dim=0) / self.count
        aggregate_coverage = (
            self.envelope_covered.sum(dim=0).to(dtype=torch.float64) / self.count
        )
        per_horizon = []
        for horizon in range(self.action_horizon):
            count = int(self.horizon_count[horizon])
            if count == 0:
                per_horizon.append(
                    {"horizon_index": horizon, "valid_steps": 0, "available": False}
                )
                continue
            mean_std = self.standard_deviation_sum[horizon] / count
            coverage = self.envelope_covered[horizon].to(torch.float64) / count
            per_horizon.append(
                {
                    "horizon_index": horizon,
                    "valid_steps": count,
                    "available": True,
                    "mean_predictive_std": float(mean_std.mean()),
                    "mean_envelope_coverage": float(coverage.mean()),
                    "per_dimension_predictive_std": mean_std.tolist(),
                    "per_dimension_envelope_coverage": coverage.tolist(),
                }
            )
        return {
            "valid_steps": self.count,
            "mean_predictive_std": float(aggregate_std.mean()),
            "mean_envelope_coverage": float(aggregate_coverage.mean()),
            "per_dimension_predictive_std": aggregate_std.tolist(),
            "per_dimension_envelope_coverage": aggregate_coverage.tolist(),
            "per_horizon": per_horizon,
        }


def _batch_action_valid_mask(
    batch: Mapping[str, Any],
    target: torch.Tensor,
) -> torch.Tensor:
    """Validate the public loss-only mask and zero-padded target."""
    valid_mask = batch.get("action_valid_mask", None)
    if valid_mask is None:
        raise ValueError("Static Flow BC evaluation requires action_valid_mask.")
    valid_mask = _canonical_valid_mask(
        valid_mask,
        batch_size=target.shape[0],
        action_horizon=target.shape[1],
        device=target.device,
    )
    invalid_coordinates = (~valid_mask).unsqueeze(-1).expand_as(target)
    if torch.any(target[invalid_coordinates] != 0):
        raise ValueError("Padded action coordinates must be exactly zero.")
    return valid_mask


def _sample_actions(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    random_samples: int,
    sampler_method: str,
    num_steps: int,
    initial_noise_generator: torch.Generator,
    step_noise_generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample complete chunks without exposing target-tail validity to the actor."""
    obs = model.preprocess_env_obs(batch["obs"], images_preprocessed=True)
    full_feature, _ = model.get_feature(obs)
    mix_feature = model.mix_proj(full_feature)
    condition = model.flow_actor.encode_condition(mix_feature, update_stats=False)
    batch_size = condition.shape[0]
    action_dim = model.flow_actor.action_dim
    action_horizon = int(model.flow_actor.action_horizon)
    flow_state_dim = int(model.flow_actor.flow_state_dim)
    if flow_state_dim != action_dim * action_horizon:
        raise ValueError(
            "Flow actor has inconsistent action_dim, action_horizon, and "
            "flow_state_dim."
        )
    initial_random_noise = torch.randn(
        (batch_size, random_samples, action_horizon, action_dim),
        generator=initial_noise_generator,
        dtype=torch.float32,
    ).to(device=condition.device, dtype=condition.dtype)
    initial_noise = torch.cat(
        (
            torch.zeros(
                (batch_size, 1, action_horizon, action_dim),
                device=condition.device,
                dtype=condition.dtype,
            ),
            initial_random_noise,
        ),
        dim=1,
    )
    sample_count = random_samples + 1
    repeated_condition = (
        condition[:, None]
        .expand(batch_size, sample_count, *condition.shape[1:])
        .reshape(batch_size * sample_count, *condition.shape[1:])
    )
    flat_initial_noise = initial_noise.reshape(
        batch_size * sample_count,
        action_horizon,
        action_dim,
    )
    step_noises = None
    if sampler_method == "flow_sde":
        if step_noise_generator is None:
            raise ValueError("flow_sde requires an explicit step-noise generator.")
        step_noises = torch.randn(
            (
                batch_size,
                sample_count,
                num_steps,
                action_horizon,
                action_dim,
            ),
            generator=step_noise_generator,
            dtype=torch.float32,
        ).reshape(
            batch_size * sample_count,
            num_steps,
            action_horizon,
            action_dim,
        )
        step_noises = step_noises.to(
            device=condition.device,
            dtype=condition.dtype,
        )
    sample = model.flow_actor.sample_path(
        repeated_condition,
        initial_noise=flat_initial_noise,
        step_noises=step_noises,
        sampler_method=sampler_method,
        num_steps=num_steps,
        train=False,
    )
    actions = sample.action.reshape(
        batch_size,
        sample_count,
        action_horizon,
        action_dim,
    )
    return actions[:, 0], actions[:, 1:]


def _build_verdict(
    headline_metrics: Mapping[str, Any],
    *,
    minimum_r2: float,
    minimum_pose_r2: float,
    maximum_pose_rmse: float,
    minimum_gripper_accuracy: float,
) -> dict[str, Any]:
    overall_r2 = headline_metrics["overall"]["r2"]
    pose_r2 = headline_metrics["pose_6d"]["r2"]
    checks = {
        "overall_r2": overall_r2 is not None and overall_r2 >= minimum_r2,
        "pose_6d_r2": pose_r2 is not None and pose_r2 >= minimum_pose_r2,
        "pose_6d_rmse": (headline_metrics["pose_6d"]["rmse"] <= maximum_pose_rmse),
        "gripper_sign_accuracy": (
            headline_metrics["gripper"]["sign_accuracy"] >= minimum_gripper_accuracy
        ),
    }
    return {
        "well_fitted": all(checks.values()),
        "prediction": "expected_single_sample",
        "checks": checks,
        "thresholds": {
            "minimum_overall_r2": minimum_r2,
            "minimum_pose_6d_r2": minimum_pose_r2,
            "maximum_pose_6d_rmse": maximum_pose_rmse,
            "minimum_gripper_sign_accuracy": minimum_gripper_accuracy,
        },
    }


def _build_dataset(
    cfg: Any,
    dataset_path: Path,
    *,
    split: str,
) -> FrankaGelloFlowDataset:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}.")
    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    action_horizon = int(model_cfg.get("action_horizon", 1))
    parameters = inspect.signature(FrankaGelloFlowDataset).parameters
    if action_horizon > 1 and "action_horizon" not in parameters:
        raise RuntimeError(
            "This FrankaGelloFlowDataset version does not support action chunks; "
            f"cannot evaluate action_horizon={action_horizon}."
        )
    episode_split = data_cfg.get("episode_split", None)
    if split == "val" and episode_split is None:
        raise ValueError(
            "--split=val requires data.episode_split so validation episodes are "
            "disjoint from training. Configure seed and val_fraction, or use "
            "--split=train for an unsplit dataset."
        )
    if episode_split is not None and not {
        "episode_split",
        "eval_dataset",
    }.issubset(parameters):
        raise RuntimeError(
            "This FrankaGelloFlowDataset version does not support episode-level "
            "train/val selection."
        )
    kwargs: dict[str, Any] = {
        "state_dim": int(model_cfg.state_dim),
        "action_dim": int(model_cfg.action_dim),
        "image_keys": tuple(data_cfg.get("image_keys", ["image"])),
        "image_shape": tuple(model_cfg.image_size),
        "state_key": str(data_cfg.get("state_key", "state")),
        "action_key": str(data_cfg.get("action_key", "actions")),
        "image_value_range": str(data_cfg.get("image_value_range", "zero_one")),
        "fps": data_cfg.get("fps", 10),
        "min_frames": int(data_cfg.get("min_frames", 1)),
        "load_workers": int(data_cfg.get("load_workers", 0)),
    }
    optional = {
        "action_horizon": action_horizon,
        "action_scale": model_cfg.get("action_scale", None),
        "episode_split": episode_split,
        "eval_dataset": split == "val",
    }
    kwargs.update({key: value for key, value in optional.items() if key in parameters})
    dataset = FrankaGelloFlowDataset(dataset_path, **kwargs)
    actual_split = getattr(dataset, "split_name", None)
    expected_split = split if episode_split is not None else "all"
    if actual_split is not None and actual_split != expected_split:
        raise RuntimeError(
            "Dataset returned an unexpected episode split: "
            f"requested={split!r}, actual={actual_split!r}."
        )
    return dataset


def _objective_metrics(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    if draws == 0:
        return {}
    totals: dict[str, float] = {}
    for draw in range(draws):
        torch.manual_seed(seed + draw)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + draw)
        with torch.enable_grad():
            output = model(forward_type=ForwardType.SFT, data=batch)
        for key, value in output.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        del output
    model.zero_grad(set_to_none=True)
    return {key: value / draws for key, value in totals.items()}


_PER_HORIZON_FIELD_MSE = re.compile(r"field_mse_h([0-9]+)\Z")


def _accumulate_objective_metrics(
    weighted_sums: dict[str, float],
    denominators: dict[str, int],
    metrics: Mapping[str, float],
    *,
    batch_size: int,
    valid_mask: torch.Tensor,
) -> None:
    """Accumulate batch means with the denominator used by each metric."""
    valid_coordinate_count = int(
        metrics.get(
            "valid_coordinate_count",
            float(valid_mask.sum().detach().cpu()),
        )
    )
    for key, value in metrics.items():
        if key == "valid_coordinate_count" or key.startswith("field_count_h"):
            weighted_sums[key] = weighted_sums.get(key, 0.0) + value
            denominators.setdefault(key, 1)
            continue
        match = _PER_HORIZON_FIELD_MSE.fullmatch(key)
        if match is not None:
            horizon = int(match.group(1))
            if horizon >= valid_mask.shape[1]:
                raise ValueError(
                    f"Objective metric {key!r} exceeds action horizon "
                    f"{valid_mask.shape[1]}."
                )
            weight = int(
                metrics.get(
                    f"field_count_h{horizon}",
                    float(valid_mask[:, horizon].sum().detach().cpu()),
                )
            )
        elif key.endswith("_mean") and key in {
            "t_mean",
            "r_mean",
            "time_gap_mean",
        }:
            weight = batch_size
        else:
            weight = valid_coordinate_count
        if weight == 0:
            continue
        weighted_sums[key] = weighted_sums.get(key, 0.0) + value * weight
        denominators[key] = denominators.get(key, 0) + weight


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _require_positive(args.batch_size, "batch_size")
    _require_positive(args.samples_per_observation, "samples_per_observation")
    if args.num_steps is not None:
        _require_positive(args.num_steps, "num_steps")
    if args.objective_draws < 0:
        raise ValueError(
            f"objective_draws must be non-negative, got {args.objective_draws}."
        )
    if args.max_samples is not None:
        _require_positive(args.max_samples, "max_samples")

    started_at = time.time()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = _resolve_portable_checkpoint(args.checkpoint)
    device = _resolve_device(args.device)
    cfg = OmegaConf.load(config_path)
    configured_data_root = _configured_data_root(args.data_root, cfg)
    dataset_path = _resolve_dataset_path(configured_data_root)
    flow_spec = resolve_flow_bc_spec(cfg.actor.model, task_type="sft")
    sampling = _resolve_sampling_settings(flow_spec, args)
    initial_noise_seed = (
        args.seed if args.initial_noise_seed is None else args.initial_noise_seed
    )
    step_noise_seed = (
        args.seed + 1 if args.step_noise_seed is None else args.step_noise_seed
    )

    LOGGER.info("Loading %s dataset split from %s", args.split, dataset_path)
    dataset_started_at = time.time()
    dataset = _build_dataset(cfg, dataset_path, split=args.split)
    dataset_load_seconds = time.time() - dataset_started_at
    evaluated_samples = len(dataset)
    if args.max_samples is not None:
        evaluated_samples = min(evaluated_samples, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.data.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    LOGGER.info("Building model and loading %s", checkpoint_path)
    model = get_model(cfg.actor.model, torch_dtype=torch.float32)
    action_horizon = flow_spec.action_horizon
    _configure_runtime_sampler(model, sampling)
    load_report = load_flow_actor_checkpoint(
        model,
        checkpoint_path,
        expected_metadata=build_flow_actor_metadata_from_config(cfg.actor.model),
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()

    action_dim = int(cfg.actor.model.action_dim)
    if action_dim != len(ACTION_NAMES):
        raise ValueError(
            f"Franka + GELLO evaluation requires action_dim=7, got {action_dim}."
        )
    metadata = load_report.metadata or {}
    if int(metadata["action_horizon"]) != action_horizon:
        raise ValueError(
            "Checkpoint and config action horizons differ: "
            f"checkpoint={metadata['action_horizon']} config={action_horizon}."
        )
    action_min, action_max = (float(value) for value in metadata["action_range"])
    num_steps = sampling.num_steps
    target_statistics = TargetStatistics(action_dim, action_horizon)
    expected_target_statistics = TargetStatistics(action_dim, action_horizon)
    raw_target_statistics = TargetStatistics(action_dim, action_horizon)
    accumulators = {
        name: RegressionAccumulator(action_dim, action_horizon)
        for name in (
            "zero_noise",
            "expected_single_sample",
            "sample_mean",
            "best_of_k",
        )
    }
    all_valid_accumulators = {
        name: AllValidChunkRegressionAccumulator(action_dim, action_horizon)
        for name in accumulators
    }
    raw_sample_mean = RegressionAccumulator(action_dim, action_horizon)
    raw_all_valid_sample_mean = AllValidChunkRegressionAccumulator(
        action_dim,
        action_horizon,
    )
    sample_diagnostics = SampleDiagnostics(action_dim, action_horizon)
    objective_sums: dict[str, float] = {}
    objective_denominators: dict[str, int] = {}
    raw_outside_count = 0
    raw_value_count = 0
    raw_max_overshoot = 0.0
    initial_noise_generator = torch.Generator(device="cpu")
    initial_noise_generator.manual_seed(initial_noise_seed)
    step_noise_generator = torch.Generator(device="cpu")
    step_noise_generator.manual_seed(step_noise_seed)
    processed = 0
    inference_started_at = time.time()

    for batch_index, cpu_batch in enumerate(loader):
        remaining = evaluated_samples - processed
        if remaining <= 0:
            break
        batch_size = int(cpu_batch["action"].shape[0])
        if batch_size > remaining:
            cpu_batch = _slice_batch(cpu_batch, remaining)
            batch_size = remaining
        batch = _move_to_device(cpu_batch, device)
        raw_target = _canonical_action_chunk(
            batch["action"].float(),
            action_dim=action_dim,
            action_horizon=action_horizon,
            name="batch actions",
        )
        valid_mask = _batch_action_valid_mask(batch, raw_target)
        policy_target = raw_target.clamp(action_min, action_max)
        valid_coordinates = valid_mask.unsqueeze(-1).expand_as(raw_target)
        outside = (
            (raw_target < action_min) | (raw_target > action_max)
        ) & valid_coordinates
        raw_outside_count += int(outside.sum().detach().cpu())
        raw_value_count += int(valid_coordinates.sum().detach().cpu())
        overshoot = torch.where(
            valid_coordinates,
            torch.maximum(
                (action_min - raw_target).clamp_min(0.0),
                (raw_target - action_max).clamp_min(0.0),
            ),
            torch.zeros_like(raw_target),
        )
        raw_max_overshoot = max(
            raw_max_overshoot,
            float(overshoot.max().detach().cpu()),
        )

        with torch.inference_mode():
            zero_noise, random_actions = _sample_actions(
                model,
                batch,
                random_samples=args.samples_per_observation,
                sampler_method=sampling.method,
                num_steps=num_steps,
                initial_noise_generator=initial_noise_generator,
                step_noise_generator=(
                    step_noise_generator if sampling.method == "flow_sde" else None
                ),
            )
        sample_mean = random_actions.mean(dim=1)
        best_of_k, _ = _select_best_chunk_sample(
            random_actions,
            policy_target,
            valid_mask,
        )
        sample_count = random_actions.shape[1]
        expected_predictions = random_actions.reshape(
            batch_size * sample_count,
            action_horizon,
            action_dim,
        )
        expected_targets = (
            policy_target[:, None]
            .expand_as(random_actions)
            .reshape(batch_size * sample_count, action_horizon, action_dim)
        )
        expected_mask = (
            valid_mask[:, None]
            .expand(batch_size, sample_count, action_horizon)
            .reshape(batch_size * sample_count, action_horizon)
        )

        target_statistics.update(policy_target, valid_mask)
        expected_target_statistics.update(expected_targets, expected_mask)
        raw_target_statistics.update(raw_target, valid_mask)
        accumulators["zero_noise"].update(zero_noise, policy_target, valid_mask)
        all_valid_accumulators["zero_noise"].update(
            zero_noise,
            policy_target,
            valid_mask,
        )
        accumulators["expected_single_sample"].update(
            expected_predictions,
            expected_targets,
            expected_mask,
        )
        all_valid_accumulators["expected_single_sample"].update(
            expected_predictions,
            expected_targets,
            expected_mask,
        )
        accumulators["sample_mean"].update(sample_mean, policy_target, valid_mask)
        all_valid_accumulators["sample_mean"].update(
            sample_mean,
            policy_target,
            valid_mask,
        )
        accumulators["best_of_k"].update(best_of_k, policy_target, valid_mask)
        all_valid_accumulators["best_of_k"].update(
            best_of_k,
            policy_target,
            valid_mask,
        )
        raw_sample_mean.update(sample_mean, raw_target, valid_mask)
        raw_all_valid_sample_mean.update(sample_mean, raw_target, valid_mask)
        sample_diagnostics.update(random_actions, policy_target, valid_mask)

        objective = _objective_metrics(
            model,
            batch,
            draws=args.objective_draws,
            seed=args.seed + 1_000_000 + batch_index * max(args.objective_draws, 1),
        )
        _accumulate_objective_metrics(
            objective_sums,
            objective_denominators,
            objective,
            batch_size=batch_size,
            valid_mask=valid_mask,
        )
        processed += batch_size
        if batch_index == 0 or processed == evaluated_samples or batch_index % 10 == 0:
            LOGGER.info("Evaluated %d/%d frames", processed, evaluated_samples)

    if processed != evaluated_samples:
        raise RuntimeError(
            f"Expected {evaluated_samples} frames, evaluated {processed}."
        )
    inference_seconds = time.time() - inference_started_at
    regression_metrics = {}
    for name, accumulator in accumulators.items():
        statistics = (
            expected_target_statistics
            if name == "expected_single_sample"
            else target_statistics
        )
        regression_metrics[name] = accumulator.summarize(statistics)
        regression_metrics[name]["all_valid_full_chunks"] = all_valid_accumulators[
            name
        ].summarize()
    verdict = _build_verdict(
        regression_metrics["expected_single_sample"],
        minimum_r2=args.good_fit_r2,
        minimum_pose_r2=args.good_fit_pose_r2,
        maximum_pose_rmse=args.good_fit_pose_rmse,
        minimum_gripper_accuracy=args.good_fit_gripper_accuracy,
    )
    weights_path = load_report.checkpoint.weights_path
    summary = {
        "verdict": verdict,
        "metrics": regression_metrics,
        "raw_target_sample_mean": {
            **raw_sample_mean.summarize(raw_target_statistics),
            "all_valid_full_chunks": raw_all_valid_sample_mean.summarize(),
        },
        "sampling_diagnostics": sample_diagnostics.summarize(),
        "objective_metrics": (
            {
                key: value / objective_denominators[key]
                for key, value in sorted(objective_sums.items())
            }
            if objective_denominators
            else {}
        ),
        "evaluation": {
            "static": True,
            "sampler": sampling.method,
            "num_steps": num_steps,
            "sde": {
                "noise_level": sampling.noise_level,
                "noise_std_range": sampling.noise_std_range,
                "safe_initial_time": sampling.safe_initial_time,
            },
            "samples_per_observation": args.samples_per_observation,
            "objective_draws": args.objective_draws,
            "requested_split": args.split,
            "evaluated_split": dataset.split_name,
            "seed": args.seed,
            "initial_noise_seed": initial_noise_seed,
            "step_noise_seed": (
                step_noise_seed if sampling.method == "flow_sde" else None
            ),
            "objective_seed_base": args.seed + 1_000_000,
            "device": str(device),
            "batch_size": args.batch_size,
            "evaluated_samples": evaluated_samples,
            "complete_dataset": evaluated_samples == len(dataset),
            "dataset_load_seconds": dataset_load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.time() - started_at,
        },
        "data": {
            "configured_root": str(configured_data_root.expanduser().resolve()),
            "configured_root_source": (
                "command_line" if args.data_root is not None else "config"
            ),
            "resolved_lerobot_path": str(dataset_path),
            "dataset_samples": len(dataset),
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "action_shape": [action_horizon, action_dim],
            "action_layout": metadata["action_layout"],
            "action_names": list(ACTION_NAMES),
            "source_fps": dataset.source_fps,
            "source_fingerprint": dataset.source_fingerprint,
            "episode_split": dataset.episode_split,
            "episode_split_fingerprint": dataset.episode_split_fingerprint,
            "num_source_episodes": dataset.num_source_episodes,
            "num_selected_episodes": dataset.num_selected_episodes,
            "valid_action_steps": target_statistics.count,
            "padded_action_steps": processed * action_horizon - target_statistics.count,
            "per_horizon_valid_steps": target_statistics.horizon_count.tolist(),
            "policy_action_range": [action_min, action_max],
            "raw_action_outside_range_count": raw_outside_count,
            "raw_action_value_count": raw_value_count,
            "raw_action_outside_range_fraction": raw_outside_count / raw_value_count,
            "raw_action_max_overshoot": raw_max_overshoot,
        },
        "checkpoint": {
            "requested_path": str(args.checkpoint.expanduser().resolve()),
            "portable_path": str(checkpoint_path),
            "weights_path": str(weights_path),
            "weights_sha256": _sha256(weights_path),
            "manifest_path": str(load_report.checkpoint.manifest_path),
            "loaded_tensor_count": len(load_report.loaded_keys),
            "metadata": metadata,
        },
        "config_path": str(config_path),
        "notes": {
            "headline_target": "Actions clipped to the checkpoint policy range.",
            "expected_single_sample": (
                f"All K random {sampling.method} chunks are treated as independent "
                "deployment samples; this metric is used for the verdict."
            ),
            "sample_mean": (
                f"Mean of K random {sampling.method} chunks; diagnostic only."
            ),
            "zero_noise": (
                "Sample starting from zero latent noise; SDE transition noise is "
                "still active when sampler=flow_sde."
            ),
            "best_of_k": (
                "One oracle sample is selected per observation by MSE over the whole "
                "valid chunk; this is not deployable performance."
            ),
            "masking": (
                "The target mask is used only for metrics and never passed to the "
                "actor or sampler; all aggregate and per-horizon metrics exclude "
                "zero-padded tail steps."
            ),
            "normalization": "All horizons use identity latent normalization.",
            "mean_baseline": "Constant per-dimension mean of the evaluated split.",
        },
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")

    headline = regression_metrics["expected_single_sample"]
    LOGGER.info(
        "expected_single_sample: RMSE=%.6f R2=%s pose_RMSE=%.6f "
        "pose_R2=%s gripper_acc=%.4f",
        headline["overall"]["rmse"],
        headline["overall"]["r2"],
        headline["pose_6d"]["rmse"],
        headline["pose_6d"]["r2"],
        headline["gripper"]["sign_accuracy"],
    )
    LOGGER.info("Well-fitted verdict: %s", verdict["well_fitted"])
    LOGGER.info("Wrote evaluation report to %s", output_path)
    if args.require_good_fit and not verdict["well_fitted"]:
        failed = [name for name, passed in verdict["checks"].items() if not passed]
        raise RuntimeError(f"Flow BC checkpoint did not pass good-fit checks: {failed}")


if __name__ == "__main__":
    main()
