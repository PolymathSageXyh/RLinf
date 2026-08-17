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

"""Evaluate a trained Franka + GELLO Flow BC checkpoint on its training set.

The evaluation is static: it does not start Ray, construct an optimizer, or
modify the checkpoint. Flow policies are generative, so the report separates a
single random sample, the mean of repeated samples, a zero-noise prediction,
and a diagnostic best-of-K upper bound.

Example:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
     examples/sft/evaluate_franka_gello_flow_bc.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlinf.config import _validate_flow_policy_v2_cfg
from rlinf.data.datasets.flow import FrankaGelloFlowDataset
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.flow_policy import get_model
from rlinf.utils.flow_actor_checkpoint import (
    build_flow_actor_metadata_from_config,
    load_flow_actor_checkpoint,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples/sft/config/franka_gello_flow_bc.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "results/flow_bc/franka_gello_flow_bc/checkpoints/global_step_12000"
)
DEFAULT_DATA_ROOT = Path("/nas/xyh/data/data_pick_cube")
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/flow_bc/franka_gello_flow_bc/evaluations/"
    "global_step_12000_train_fit.json"
)
ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-observation", type=int, default=8)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Flow ODE steps; defaults to actor.model.denoising_steps.",
    )
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


@dataclass
class TargetStatistics:
    """Streaming target moments used to define the constant-mean baseline."""

    action_dim: int
    count: int = 0
    value_sum: torch.Tensor = field(init=False)
    square_sum: torch.Tensor = field(init=False)
    minimum: torch.Tensor = field(init=False)
    maximum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.value_sum = torch.zeros(self.action_dim, dtype=torch.float64)
        self.square_sum = torch.zeros(self.action_dim, dtype=torch.float64)
        self.minimum = torch.full((self.action_dim,), math.inf, dtype=torch.float64)
        self.maximum = torch.full((self.action_dim,), -math.inf, dtype=torch.float64)

    def update(self, target: torch.Tensor) -> None:
        target = target.detach().to(device="cpu", dtype=torch.float64)
        if target.ndim != 2 or target.shape[1] != self.action_dim:
            raise ValueError(
                f"target must have shape [B, {self.action_dim}], got {tuple(target.shape)}."
            )
        self.count += target.shape[0]
        self.value_sum += target.sum(dim=0)
        self.square_sum += target.square().sum(dim=0)
        self.minimum = torch.minimum(self.minimum, target.amin(dim=0))
        self.maximum = torch.maximum(self.maximum, target.amax(dim=0))

    def centered_square_sum(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("Cannot summarize empty target statistics.")
        centered = self.square_sum - self.value_sum.square() / self.count
        return centered.clamp_min(0.0)


@dataclass
class RegressionAccumulator:
    """Accumulate prediction errors without retaining dataset-sized tensors."""

    action_dim: int
    count: int = 0
    absolute_error_sum: torch.Tensor = field(init=False)
    square_error_sum: torch.Tensor = field(init=False)
    maximum_absolute_error: torch.Tensor = field(init=False)
    within_005: torch.Tensor = field(init=False)
    within_010: torch.Tensor = field(init=False)
    gripper_sign_correct: int = 0

    def __post_init__(self) -> None:
        self.absolute_error_sum = torch.zeros(self.action_dim, dtype=torch.float64)
        self.square_error_sum = torch.zeros(self.action_dim, dtype=torch.float64)
        self.maximum_absolute_error = torch.zeros(self.action_dim, dtype=torch.float64)
        self.within_005 = torch.zeros(self.action_dim, dtype=torch.int64)
        self.within_010 = torch.zeros(self.action_dim, dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to(device="cpu", dtype=torch.float64)
        target = target.detach().to(device="cpu", dtype=torch.float64)
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError(
                "prediction and target must have identical [B, D] shapes, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {prediction.shape[1]}."
            )
        absolute_error = (prediction - target).abs()
        self.count += target.shape[0]
        self.absolute_error_sum += absolute_error.sum(dim=0)
        self.square_error_sum += absolute_error.square().sum(dim=0)
        self.maximum_absolute_error = torch.maximum(
            self.maximum_absolute_error, absolute_error.amax(dim=0)
        )
        self.within_005 += (absolute_error <= 0.05).sum(dim=0)
        self.within_010 += (absolute_error <= 0.10).sum(dim=0)
        if self.action_dim == len(ACTION_NAMES):
            self.gripper_sign_correct += int(
                ((prediction[:, -1] >= 0) == (target[:, -1] >= 0)).sum()
            )

    @staticmethod
    def _r2(error_ss: float, target_ss: float) -> float | None:
        if target_ss <= 0.0:
            return None
        return 1.0 - error_ss / target_ss

    def _group_summary(
        self,
        indices: slice,
        target_centered_ss: torch.Tensor,
    ) -> dict[str, float | None]:
        absolute_sum = float(self.absolute_error_sum[indices].sum())
        square_sum = float(self.square_error_sum[indices].sum())
        centered_sum = float(target_centered_ss[indices].sum())
        dimensions = self.absolute_error_sum[indices].numel()
        denominator = self.count * dimensions
        mse = square_sum / denominator
        baseline_mse = centered_sum / denominator
        return {
            "mae": absolute_sum / denominator,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "max_absolute_error": float(self.maximum_absolute_error[indices].max()),
            "within_0.05_fraction": float(self.within_005[indices].sum()) / denominator,
            "within_0.10_fraction": float(self.within_010[indices].sum()) / denominator,
            "mean_baseline_mse": baseline_mse,
            "r2": self._r2(square_sum, centered_sum),
        }

    def summarize(self, target_statistics: TargetStatistics) -> dict[str, Any]:
        if self.count == 0 or self.count != target_statistics.count:
            raise ValueError(
                "Prediction and target statistics must contain the same nonzero count."
            )
        target_centered_ss = target_statistics.centered_square_sum()
        per_dim_mse = self.square_error_sum / self.count
        per_dim_baseline = target_centered_ss / self.count
        per_dimension = []
        for index in range(self.action_dim):
            target_ss = float(target_centered_ss[index])
            error_ss = float(self.square_error_sum[index])
            per_dimension.append(
                {
                    "name": (
                        ACTION_NAMES[index]
                        if self.action_dim == len(ACTION_NAMES)
                        else str(index)
                    ),
                    "target_mean": float(
                        target_statistics.value_sum[index] / target_statistics.count
                    ),
                    "target_min": float(target_statistics.minimum[index]),
                    "target_max": float(target_statistics.maximum[index]),
                    "mae": float(self.absolute_error_sum[index] / self.count),
                    "mse": float(per_dim_mse[index]),
                    "rmse": math.sqrt(float(per_dim_mse[index])),
                    "mean_baseline_mse": float(per_dim_baseline[index]),
                    "r2": self._r2(error_ss, target_ss),
                }
            )

        summary: dict[str, Any] = {
            "overall": self._group_summary(
                slice(0, self.action_dim), target_centered_ss
            ),
            "per_dimension": per_dimension,
        }
        if self.action_dim == len(ACTION_NAMES):
            summary["pose_6d"] = self._group_summary(slice(0, 6), target_centered_ss)
            summary["gripper"] = {
                **self._group_summary(slice(6, 7), target_centered_ss),
                "sign_accuracy": self.gripper_sign_correct / self.count,
            }
        return summary


@dataclass
class SampleDiagnostics:
    """Accumulate uncertainty and target coverage across random Flow samples."""

    action_dim: int
    count: int = 0
    standard_deviation_sum: torch.Tensor = field(init=False)
    envelope_covered: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.standard_deviation_sum = torch.zeros(self.action_dim, dtype=torch.float64)
        self.envelope_covered = torch.zeros(self.action_dim, dtype=torch.int64)

    def update(self, samples: torch.Tensor, target: torch.Tensor) -> None:
        samples = samples.detach().to(device="cpu", dtype=torch.float64)
        target = target.detach().to(device="cpu", dtype=torch.float64)
        sample_target_shape = (
            (samples.shape[0], samples.shape[2]) if samples.ndim == 3 else None
        )
        if sample_target_shape != tuple(target.shape):
            raise ValueError(
                "samples must be [B, K, D] and target [B, D], got "
                f"{tuple(samples.shape)} and {tuple(target.shape)}."
            )
        self.count += target.shape[0]
        self.standard_deviation_sum += samples.std(dim=1, unbiased=False).sum(dim=0)
        self.envelope_covered += (
            (target >= samples.amin(dim=1)) & (target <= samples.amax(dim=1))
        ).sum(dim=0)

    def summarize(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("Cannot summarize empty sample diagnostics.")
        mean_std = self.standard_deviation_sum / self.count
        coverage = self.envelope_covered.to(dtype=torch.float64) / self.count
        return {
            "mean_predictive_std": float(mean_std.mean()),
            "mean_envelope_coverage": float(coverage.mean()),
            "per_dimension_predictive_std": mean_std.tolist(),
            "per_dimension_envelope_coverage": coverage.tolist(),
        }


def _sample_actions(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    random_samples: int,
    num_steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-noise and random ODE actions without re-encoding each sample."""
    obs = model.preprocess_env_obs(batch["obs"], images_preprocessed=True)
    full_feature, _ = model.get_feature(obs)
    mix_feature = model.mix_proj(full_feature)
    condition = model.flow_actor.encode_condition(mix_feature, update_stats=False)
    batch_size = condition.shape[0]
    action_dim = model.flow_actor.action_dim

    initial_random_noise = torch.randn(
        (batch_size, random_samples, action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=condition.device, dtype=condition.dtype)
    initial_noise = torch.cat(
        (
            torch.zeros(
                (batch_size, 1, action_dim),
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
    sample = model.flow_actor.sample_path(
        repeated_condition,
        initial_noise=initial_noise.reshape(batch_size * sample_count, action_dim),
        sampler_method="flow_ode",
        num_steps=num_steps,
        train=False,
    )
    actions = sample.action.reshape(batch_size, sample_count, action_dim)
    return actions[:, 0], actions[:, 1:]


def _build_verdict(
    sample_mean_metrics: Mapping[str, Any],
    *,
    minimum_r2: float,
    minimum_pose_r2: float,
    maximum_pose_rmse: float,
    minimum_gripper_accuracy: float,
) -> dict[str, Any]:
    overall_r2 = sample_mean_metrics["overall"]["r2"]
    pose_r2 = sample_mean_metrics["pose_6d"]["r2"]
    checks = {
        "overall_r2": overall_r2 is not None and overall_r2 >= minimum_r2,
        "pose_6d_r2": pose_r2 is not None and pose_r2 >= minimum_pose_r2,
        "pose_6d_rmse": (sample_mean_metrics["pose_6d"]["rmse"] <= maximum_pose_rmse),
        "gripper_sign_accuracy": (
            sample_mean_metrics["gripper"]["sign_accuracy"] >= minimum_gripper_accuracy
        ),
    }
    return {
        "well_fitted": all(checks.values()),
        "prediction": "sample_mean",
        "checks": checks,
        "thresholds": {
            "minimum_overall_r2": minimum_r2,
            "minimum_pose_6d_r2": minimum_pose_r2,
            "maximum_pose_6d_rmse": maximum_pose_rmse,
            "minimum_gripper_sign_accuracy": minimum_gripper_accuracy,
        },
    }


def _build_dataset(cfg: Any, dataset_path: Path) -> FrankaGelloFlowDataset:
    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    return FrankaGelloFlowDataset(
        dataset_path,
        state_dim=int(model_cfg.state_dim),
        action_dim=int(model_cfg.action_dim),
        image_keys=tuple(data_cfg.get("image_keys", ["image"])),
        image_shape=tuple(model_cfg.image_size),
        state_key=str(data_cfg.get("state_key", "state")),
        action_key=str(data_cfg.get("action_key", "actions")),
        image_value_range=str(data_cfg.get("image_value_range", "zero_one")),
        fps=int(data_cfg.get("fps", 10)),
        min_frames=int(data_cfg.get("min_frames", 1)),
        load_workers=int(data_cfg.get("load_workers", 0)),
    )


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
    dataset_path = _resolve_dataset_path(args.data_root)
    checkpoint_path = _resolve_portable_checkpoint(args.checkpoint)
    device = _resolve_device(args.device)
    cfg = OmegaConf.load(config_path)
    _validate_flow_policy_v2_cfg(
        cfg.actor.model,
        task_type="sft",
        data_cfg=cfg.data,
        resume_dir=None,
    )
    if str(cfg.actor.model.get("flow_matching", {}).get("implementation", "")) != (
        "pytorch_flow_t_v2"
    ):
        raise ValueError("Static Flow BC evaluation requires pytorch_flow_t_v2.")

    LOGGER.info("Loading training dataset from %s", dataset_path)
    dataset_started_at = time.time()
    dataset = _build_dataset(cfg, dataset_path)
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
    action_min, action_max = (float(value) for value in metadata["action_range"])
    num_steps = args.num_steps or int(cfg.actor.model.denoising_steps)
    target_statistics = TargetStatistics(action_dim)
    raw_target_statistics = TargetStatistics(action_dim)
    accumulators = {
        name: RegressionAccumulator(action_dim)
        for name in ("zero_noise", "single_sample", "sample_mean", "best_of_k")
    }
    raw_sample_mean = RegressionAccumulator(action_dim)
    sample_diagnostics = SampleDiagnostics(action_dim)
    objective_sums: dict[str, float] = {}
    objective_sample_count = 0
    raw_outside_count = 0
    raw_value_count = 0
    raw_max_overshoot = 0.0
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(args.seed)
    processed = 0
    inference_started_at = time.time()

    for batch_index, cpu_batch in enumerate(loader):
        remaining = evaluated_samples - processed
        if remaining <= 0:
            break
        batch_size = int(cpu_batch["actions"].shape[0])
        if batch_size > remaining:
            cpu_batch = _slice_batch(cpu_batch, remaining)
            batch_size = remaining
        batch = _move_to_device(cpu_batch, device)
        raw_target = batch["actions"].float()
        policy_target = raw_target.clamp(action_min, action_max)
        outside = (raw_target < action_min) | (raw_target > action_max)
        raw_outside_count += int(outside.sum().detach().cpu())
        raw_value_count += raw_target.numel()
        raw_max_overshoot = max(
            raw_max_overshoot,
            float(
                torch.maximum(
                    (action_min - raw_target).clamp_min(0.0),
                    (raw_target - action_max).clamp_min(0.0),
                )
                .max()
                .detach()
                .cpu()
            ),
        )

        with torch.inference_mode():
            zero_noise, random_actions = _sample_actions(
                model,
                batch,
                random_samples=args.samples_per_observation,
                num_steps=num_steps,
                generator=cpu_generator,
            )
        single_sample = random_actions[:, 0]
        sample_mean = random_actions.mean(dim=1)
        sample_errors = (random_actions - policy_target[:, None]).square().mean(dim=-1)
        best_indices = sample_errors.argmin(dim=1)
        best_of_k = random_actions[
            torch.arange(batch_size, device=random_actions.device), best_indices
        ]

        target_statistics.update(policy_target)
        raw_target_statistics.update(raw_target)
        accumulators["zero_noise"].update(zero_noise, policy_target)
        accumulators["single_sample"].update(single_sample, policy_target)
        accumulators["sample_mean"].update(sample_mean, policy_target)
        accumulators["best_of_k"].update(best_of_k, policy_target)
        raw_sample_mean.update(sample_mean, raw_target)
        sample_diagnostics.update(random_actions, policy_target)

        objective = _objective_metrics(
            model,
            batch,
            draws=args.objective_draws,
            seed=args.seed + 1_000_000 + batch_index * max(args.objective_draws, 1),
        )
        for key, value in objective.items():
            objective_sums[key] = objective_sums.get(key, 0.0) + value * batch_size
        objective_sample_count += batch_size if objective else 0
        processed += batch_size
        if batch_index == 0 or processed == evaluated_samples or batch_index % 10 == 0:
            LOGGER.info("Evaluated %d/%d frames", processed, evaluated_samples)

    if processed != evaluated_samples:
        raise RuntimeError(
            f"Expected {evaluated_samples} frames, evaluated {processed}."
        )
    inference_seconds = time.time() - inference_started_at
    regression_metrics = {
        name: accumulator.summarize(target_statistics)
        for name, accumulator in accumulators.items()
    }
    verdict = _build_verdict(
        regression_metrics["sample_mean"],
        minimum_r2=args.good_fit_r2,
        minimum_pose_r2=args.good_fit_pose_r2,
        maximum_pose_rmse=args.good_fit_pose_rmse,
        minimum_gripper_accuracy=args.good_fit_gripper_accuracy,
    )
    weights_path = load_report.checkpoint.weights_path
    summary = {
        "verdict": verdict,
        "metrics": regression_metrics,
        "raw_target_sample_mean": raw_sample_mean.summarize(raw_target_statistics),
        "sampling_diagnostics": sample_diagnostics.summarize(),
        "objective_metrics": (
            {
                key: value / objective_sample_count
                for key, value in sorted(objective_sums.items())
            }
            if objective_sample_count
            else {}
        ),
        "evaluation": {
            "static": True,
            "sampler": "flow_ode",
            "num_steps": num_steps,
            "samples_per_observation": args.samples_per_observation,
            "objective_draws": args.objective_draws,
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "evaluated_samples": evaluated_samples,
            "complete_dataset": evaluated_samples == len(dataset),
            "dataset_load_seconds": dataset_load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.time() - started_at,
        },
        "data": {
            "configured_root": str(args.data_root.expanduser().resolve()),
            "resolved_lerobot_path": str(dataset_path),
            "dataset_samples": len(dataset),
            "action_dim": action_dim,
            "action_names": list(ACTION_NAMES),
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
            "sample_mean": "Mean of K random Flow ODE actions; used for the verdict.",
            "zero_noise": "Deterministic ODE action starting from zero latent noise.",
            "best_of_k": (
                "Diagnostic oracle selected using ground truth; not deployable performance."
            ),
            "mean_baseline": "Constant per-dimension mean of this evaluated dataset.",
        },
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")

    headline = regression_metrics["sample_mean"]
    LOGGER.info(
        "sample_mean: RMSE=%.6f R2=%s pose_RMSE=%.6f pose_R2=%s gripper_acc=%.4f",
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
