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

"""Run a bounded, single-GPU Flow BC optimization verification.

This script deliberately avoids Ray/FSDP so a Flow BC data/model/objective seam
can be checked without reserving any accelerator except the one exposed through
``CUDA_VISIBLE_DEVICES``. It is an optimization smoke test, not a replacement
for a production SFT run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.config import _validate_flow_policy_v2_cfg
from rlinf.data.datasets.flow import build_franka_gello_flow_dataloader
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.flow_policy import get_model
from rlinf.utils.flow_actor_checkpoint import (
    build_flow_actor_metadata_from_config,
    export_flow_actor_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples/sft/config/franka_gello_flow_bc.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/flow_bc/verification_gpu6"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--probe-draws", type=int, default=8)
    parser.add_argument(
        "--batch-mode",
        choices=("stream", "fixed"),
        default="stream",
        help="Use sampler batches or repeatedly overfit the first real batch.",
    )
    parser.add_argument(
        "--expected-visible-device",
        default="6",
        help="Exact CUDA_VISIBLE_DEVICES value required before CUDA is initialized.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.1,
        help="Maximum fraction of the visible GPU memory available to this process.",
    )
    return parser.parse_args()


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvidia_smi_snapshot() -> dict[str, Any]:
    query = (
        "index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    fields = tuple(part.strip() for part in query.split(","))
    devices = []
    for line in completed.stdout.splitlines():
        values = tuple(value.strip() for value in line.split(","))
        if len(values) == len(fields):
            devices.append(dict(zip(fields, values, strict=True)))
    return {"devices": devices}


def _assert_single_gpu(expected_visible_device: str, memory_fraction: float) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != expected_visible_device:
        raise RuntimeError(
            "This verifier requires an explicit, exclusive CUDA mapping: "
            f"expected CUDA_VISIBLE_DEVICES={expected_visible_device!r}, got {visible!r}."
        )
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError(f"memory_fraction must lie in (0, 1], got {memory_fraction}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"The verifier must see exactly one GPU, got {torch.cuda.device_count()}."
        )
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    torch.cuda.set_device(0)
    return torch.cuda.get_device_name(0)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _scalar_metrics(output: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in output.items():
        if torch.is_tensor(value) and value.numel() == 1:
            metrics[key] = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            metrics[key] = float(value)
    return metrics


def _mean(records: Sequence[Mapping[str, float]], key: str) -> float:
    values = [float(record[key]) for record in records]
    return sum(values) / len(values)


def _relative_change(before: float, after: float) -> float:
    return (after - before) / max(abs(before), 1e-12)


def _evaluate_probe(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    seed: int,
    draws: int,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float]] = []
    for draw in range(draws):
        _set_seed(seed + draw)
        with torch.enable_grad():
            output = model(forward_type=ForwardType.SFT, data=batch)
        record = _scalar_metrics(output)
        if not all(math.isfinite(value) for value in record.values()):
            raise RuntimeError(f"Probe produced non-finite metrics: {record}")
        records.append(record)
        del output
    model.zero_grad(set_to_none=True)
    return {key: _mean(records, key) for key in records[0]}


def _build_optimizer(model: torch.nn.Module, cfg: Any) -> tuple[Any, bool]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    kwargs = {
        "lr": float(cfg.lr),
        "betas": (float(cfg.adam_beta1), float(cfg.adam_beta2)),
        "eps": float(cfg.adam_eps),
        "weight_decay": float(cfg.weight_decay),
    }
    try:
        return torch.optim.AdamW(parameters, fused=True, **kwargs), True
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(parameters, **kwargs), False


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> None:
    args = _parse_args()
    _require_positive(args.steps, "steps")
    _require_positive(args.batch_size, "batch_size")
    _require_positive(args.probe_draws, "probe_draws")

    device_name = _assert_single_gpu(args.expected_visible_device, args.memory_fraction)
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _set_seed(args.seed)

    cfg = OmegaConf.load(args.config.resolve())
    cfg.actor.micro_batch_size = args.batch_size
    cfg.actor.global_batch_size = args.batch_size
    cfg.runner.max_steps = args.steps
    placement = str(cfg.cluster.component_placement.actor)
    if placement != f"{args.expected_visible_device}-{args.expected_visible_device}":
        raise RuntimeError(
            "Config placement does not match the requested physical GPU: "
            f"actor={placement!r}."
        )
    _validate_flow_policy_v2_cfg(
        cfg.actor.model,
        task_type="sft",
        data_cfg=cfg.data,
        resume_dir=cfg.runner.resume_dir,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.jsonl"
    started_at = time.time()
    initial_gpu = _nvidia_smi_snapshot()

    data_started_at = time.time()
    data_loader, data_config = build_franka_gello_flow_dataloader(
        cfg,
        world_size=1,
        rank=0,
        data_paths=cfg.data.train_data_paths,
    )
    data_load_seconds = time.time() - data_started_at
    data_iter = iter(data_loader)
    probe_batch = _move_to_device(next(data_iter), device)
    fixed_training_batch = probe_batch if args.batch_mode == "fixed" else None

    model = get_model(cfg.actor.model, torch_dtype=torch.float32).to(device)
    optimizer, fused_optimizer = _build_optimizer(model, cfg.actor.optim)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parameter_probe = trainable_parameters[0].detach().clone()
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )

    before_probe = _evaluate_probe(
        model,
        probe_batch,
        seed=args.seed + 100_000,
        draws=args.probe_draws,
    )
    _set_seed(args.seed + 1)
    torch.cuda.reset_peak_memory_stats(device)
    training_started_at = time.time()
    records: list[dict[str, float]] = []

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            if fixed_training_batch is None:
                try:
                    cpu_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    cpu_batch = next(data_iter)
                batch = _move_to_device(cpu_batch, device)
            else:
                batch = fixed_training_batch

            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(forward_type=ForwardType.SFT, data=batch)
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Step {step} produced non-finite loss {loss}.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=float(cfg.actor.optim.clip_grad),
                error_if_nonfinite=True,
            )
            optimizer.step()

            record = _scalar_metrics(output)
            record.update(
                {
                    "step": float(step),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if not all(math.isfinite(value) for value in record.values()):
                raise RuntimeError(f"Step {step} produced non-finite metrics: {record}")
            records.append(record)
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(
                    f"step={step:03d} loss={record['loss']:.6f} "
                    f"field_mse={record['field_mse']:.6f} "
                    f"grad_norm={record['grad_norm']:.6f}",
                    flush=True,
                )

    torch.cuda.synchronize(device)
    training_seconds = time.time() - training_started_at
    after_probe = _evaluate_probe(
        model,
        probe_batch,
        seed=args.seed + 100_000,
        draws=args.probe_draws,
    )
    parameter_delta_max = float(
        (trainable_parameters[0].detach() - parameter_probe).abs().max().cpu()
    )
    window = min(10, len(records))
    first_window = records[:window]
    last_window = records[-window:]
    first_loss = _mean(first_window, "loss")
    last_loss = _mean(last_window, "loss")
    first_field_mse = _mean(first_window, "field_mse")
    last_field_mse = _mean(last_window, "field_mse")
    probe_loss_change = _relative_change(before_probe["loss"], after_probe["loss"])
    probe_field_change = _relative_change(
        before_probe["field_mse"], after_probe["field_mse"]
    )
    checks = {
        "completed_exact_step_count": len(records) == args.steps,
        "all_step_metrics_finite": all(
            math.isfinite(value) for record in records for value in record.values()
        ),
        "parameters_updated": parameter_delta_max > 0.0,
        "train_loss_window_decreased": last_loss < first_loss,
        "train_field_mse_window_decreased": last_field_mse < first_field_mse,
        "probe_loss_decreased_by_at_least_1_percent": probe_loss_change <= -0.01,
        "probe_field_mse_decreased_by_at_least_20_percent": (
            probe_field_change <= -0.20
        ),
    }
    passed = all(checks.values())

    encoder_path = Path(str(cfg.actor.model.model_path)) / str(
        cfg.actor.model.encoder_config.ckpt_name
    )
    artifact = export_flow_actor_checkpoint(
        model,
        run_dir / "portable_actor_smoke",
        metadata=build_flow_actor_metadata_from_config(cfg.actor.model),
    )
    summary = {
        "passed": passed,
        "checks": checks,
        "purpose": "single-GPU optimization smoke test; not production SFT",
        "config_path": str(args.config.resolve()),
        "data_path": str(cfg.data.train_data_paths),
        "data_config": data_config,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "batch_mode": args.batch_mode,
        "seed": args.seed,
        "probe_draws": args.probe_draws,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_cuda_device": 0,
        "device_name": device_name,
        "config_actor_placement": placement,
        "memory_fraction_limit": args.memory_fraction,
        "initial_gpu_snapshot": initial_gpu,
        "final_gpu_snapshot": _nvidia_smi_snapshot(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "data_load_seconds": data_load_seconds,
        "training_seconds": training_seconds,
        "total_seconds": time.time() - started_at,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "optimizer": "AdamW",
        "fused_optimizer": fused_optimizer,
        "learning_rate": float(cfg.actor.optim.lr),
        "clip_grad": float(cfg.actor.optim.clip_grad),
        "first_window_loss": first_loss,
        "last_window_loss": last_loss,
        "train_loss_relative_change": _relative_change(first_loss, last_loss),
        "first_window_field_mse": first_field_mse,
        "last_window_field_mse": last_field_mse,
        "train_field_mse_relative_change": _relative_change(
            first_field_mse, last_field_mse
        ),
        "probe_before": before_probe,
        "probe_after": after_probe,
        "probe_loss_relative_change": probe_loss_change,
        "probe_field_mse_relative_change": probe_field_change,
        "parameter_delta_max": parameter_delta_max,
        "encoder_checkpoint": str(encoder_path),
        "encoder_checkpoint_sha256": _sha256(encoder_path),
        "portable_actor_checkpoint": str(artifact.checkpoint_dir),
        "metrics_path": str(metrics_path),
    }
    _write_json(run_dir / "summary.json", summary)
    (args.output_root.resolve() / "latest.txt").write_text(
        f"{run_dir.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Flow BC verification failed checks: {failed}")


if __name__ == "__main__":
    main()
