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

"""Franka + GELLO LeRobot adapter for Flow-T behavior cloning.

The real-world collector writes finalized LeRobot repositories below
``collected_data/rank_N/id_M``. This module reuses
:class:`RollingLeRobotDataset` for decoding and in-memory storage and only
adapts a decoded frame to the observation contract consumed by FlowPolicy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DistributedSampler

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root

logger = logging.getLogger(__name__)

FRANKA_GELLO_ACTION_DIM = 7
_IMAGE_CHANNEL_COUNTS = (1, 3, 4)


def _rolling_dataset_cls():
    """Import the existing LeRobot implementation only when data is opened."""
    from rlinf.data.datasets.dagger.dataset import RollingLeRobotDataset

    return RollingLeRobotDataset


def _as_path_list(data_paths: Any) -> list[str]:
    """Normalize the SFT ``data_paths`` shapes used by existing YAML files."""
    if isinstance(data_paths, (str, Path)):
        return [str(data_paths)]
    if isinstance(data_paths, Mapping):
        path = data_paths.get("dataset_path", data_paths.get("data_path"))
        if path is None:
            raise ValueError(
                "Flow BC dataset entries must define 'dataset_path' or 'data_path'."
            )
        return [str(path)]
    if isinstance(data_paths, Sequence):
        paths: list[str] = []
        for entry in data_paths:
            paths.extend(_as_path_list(entry))
        if paths:
            return paths
    raise ValueError("Flow BC requires at least one LeRobot dataset path.")


def discover_lerobot_shards(data_paths: Any) -> list[Path]:
    """Find finalized collector shards or ordinary LeRobot repositories.

    A configured path can point to one LeRobot repository, the collector's
    ``collected_data`` root, or one ``rank_N`` directory. Partially written
    shards are excluded through the validation already implemented by
    :meth:`RollingLeRobotDataset.archived_shard_info`.
    """
    rolling_dataset_cls = _rolling_dataset_cls()
    shards: list[Path] = []
    for configured_path in _as_path_list(data_paths):
        root = resolve_lerobot_dataset_root(configured_path)
        if rolling_dataset_cls.archived_shard_info(root) is not None:
            shards.append(root)
            continue
        if not root.is_dir():
            raise FileNotFoundError(f"Flow BC dataset path does not exist: {root}")

        info_paths = {
            *root.glob("rank_*/id_*/meta/info.json"),
            *root.glob("id_*/meta/info.json"),
        }
        candidates = sorted(
            {info_path.parent.parent for info_path in info_paths}, key=str
        )
        valid = [
            path
            for path in candidates
            if rolling_dataset_cls.archived_shard_info(path) is not None
        ]
        if not valid:
            raise ValueError(
                f"No finalized LeRobot rank_N/id_M shards were found under {root}."
            )
        shards.extend(valid)

    # Preserve deterministic ordering while removing duplicated configured roots.
    return list(dict.fromkeys(path.resolve() for path in shards))


def _normalize_image_shape(image_shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if image_shape is None:
        return None
    normalized = tuple(int(dimension) for dimension in image_shape)
    if len(normalized) != 3 or any(dimension <= 0 for dimension in normalized):
        raise ValueError(
            f"image_shape must be positive CHW dimensions, got {normalized}."
        )
    return normalized


def _image_to_unit_chw(
    image: Any,
    value_range: str,
    key: str,
    expected_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    """Convert one image to contiguous float32 CHW in ``[0, 1]``."""
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Image {key!r} must have 3 dimensions, got {tensor.shape}.")

    normalized_shape = _normalize_image_shape(expected_shape)
    raw_shape = tuple(tensor.shape)
    if normalized_shape is not None:
        expected_hwc = (
            normalized_shape[1],
            normalized_shape[2],
            normalized_shape[0],
        )
        matches_chw = raw_shape == normalized_shape
        matches_hwc = raw_shape == expected_hwc
        if matches_chw and matches_hwc:
            raise ValueError(
                f"Image {key!r} shape {raw_shape} is ambiguous between CHW and HWC."
            )
        if matches_hwc:
            tensor = tensor.permute(2, 0, 1)
        elif not matches_chw:
            raise ValueError(
                f"Image {key!r} must match FlowPolicy image_size={normalized_shape} "
                f"as CHW or HWC, got {raw_shape}."
            )
    else:
        leading_is_channel = tensor.shape[0] in _IMAGE_CHANNEL_COUNTS
        trailing_is_channel = tensor.shape[-1] in _IMAGE_CHANNEL_COUNTS
        if leading_is_channel and trailing_is_channel:
            raise ValueError(
                f"Image {key!r} shape {raw_shape} is ambiguous between CHW and HWC; "
                "provide image_shape."
            )
        if not leading_is_channel and not trailing_is_channel:
            raise ValueError(
                f"Image {key!r} must be CHW or HWC with 1/3/4 channels, "
                f"got {raw_shape}."
            )
        if trailing_is_channel:
            tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] not in _IMAGE_CHANNEL_COUNTS:
        raise ValueError(
            f"Image {key!r} must be CHW or HWC with 1/3/4 channels, "
            f"got {tuple(tensor.shape)}."
        )

    if normalized_shape is not None and tuple(tensor.shape) != normalized_shape:
        raise ValueError(
            f"Image {key!r} must match FlowPolicy image_size={normalized_shape}, "
            f"got CHW shape {tuple(tensor.shape)}."
        )

    was_floating = tensor.is_floating_point()
    tensor = tensor.to(dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError(f"Image {key!r} must not be empty.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Image {key!r} contains non-finite values.")
    minimum = float(tensor.min())
    maximum = float(tensor.max())

    if value_range not in {"auto", "zero_one", "zero_255"}:
        raise ValueError(
            "image_value_range must be one of 'auto', 'zero_one', or 'zero_255'."
        )
    if value_range == "zero_one":
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError(
                f"Image {key!r} was declared [0,1], observed [{minimum}, {maximum}]."
            )
    elif value_range == "zero_255":
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(
                f"Image {key!r} was declared [0,255], observed [{minimum}, {maximum}]."
            )
        tensor = tensor / 255.0
    elif not was_floating or maximum > 1.0 + 1e-6:
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(
                f"Cannot infer image range for {key!r}: [{minimum}, {maximum}]."
            )
        tensor = tensor / 255.0
    elif minimum < -1e-6:
        raise ValueError(
            f"Float image {key!r} must be non-negative, observed minimum {minimum}."
        )

    return tensor.contiguous().clamp(0.0, 1.0)


def adapt_franka_gello_sample(
    sample: Mapping[str, Any],
    *,
    state_key: str = "state",
    action_key: str = "actions",
    image_keys: Sequence[str] = ("image",),
    state_dim: int | None = None,
    action_dim: int = FRANKA_GELLO_ACTION_DIM,
    image_shape: Sequence[int] | None = None,
    image_value_range: str = "auto",
) -> dict[str, Any]:
    """Adapt one decoded collector frame to the FlowPolicy SFT schema.

    This adapter intentionally supports only the single-arm Franka + GELLO
    contract: one 7-D action (six Cartesian deltas plus gripper), one state
    vector, and one or more images. No action slicing or implicit camera resize
    is performed.
    """
    if int(action_dim) != FRANKA_GELLO_ACTION_DIM:
        raise ValueError(
            "Franka + GELLO Flow BC supports exactly action_dim=7; "
            f"configured action_dim={action_dim}. Action slicing is unsupported."
        )
    if not image_keys:
        raise ValueError("Flow BC requires at least one image key.")
    if any(not isinstance(key, str) or not key for key in image_keys):
        raise ValueError("Flow BC image_keys must be non-empty strings.")
    if len(set(image_keys)) != len(image_keys):
        raise ValueError(f"Flow BC image_keys must be unique, got {list(image_keys)}.")

    missing = [key for key in (state_key, action_key, *image_keys) if key not in sample]
    if missing:
        raise KeyError(f"Flow BC sample is missing required keys: {missing}.")

    states = torch.as_tensor(sample[state_key], dtype=torch.float32)
    if states.ndim == 2 and states.shape[0] == 1:
        states = states[0]
    if states.ndim != 1:
        raise ValueError(
            f"Franka state must have shape [state_dim], got {tuple(states.shape)}."
        )
    if state_dim is not None and states.numel() != int(state_dim):
        raise ValueError(
            f"Expected Franka state_dim={state_dim}, got {states.numel()}."
        )

    actions = torch.as_tensor(sample[action_key], dtype=torch.float32)
    if actions.ndim == 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 1 or actions.numel() != FRANKA_GELLO_ACTION_DIM:
        raise ValueError(
            f"Expected one GELLO action with shape [7], got {tuple(actions.shape)}; "
            "action slicing is intentionally unsupported."
        )
    if not torch.isfinite(states).all():
        raise ValueError("Franka state contains non-finite values.")
    if not torch.isfinite(actions).all():
        raise ValueError("GELLO action contains non-finite values.")

    images = [
        _image_to_unit_chw(
            sample[key],
            image_value_range,
            key,
            expected_shape=image_shape,
        )
        for key in image_keys
    ]
    obs: dict[str, torch.Tensor] = {
        "states": states.contiguous(),
        "main_images": images[0],
    }
    if len(images) > 1:
        obs["extra_view_images"] = torch.stack(images[1:], dim=0)

    return {
        "obs": obs,
        "actions": actions.contiguous(),
        # The FlowPolicy SFT seam must honor this and skip preprocess_env_obs's
        # second /255 conversion.
        "images_preprocessed": torch.tensor(True),
    }


class FrankaGelloFlowDataset(Dataset):
    """Static BC view over finalized Franka collector LeRobot shards."""

    def __init__(
        self,
        data_paths: Any,
        *,
        state_dim: int,
        action_dim: int = FRANKA_GELLO_ACTION_DIM,
        image_keys: Sequence[str] = ("image",),
        image_shape: Sequence[int] | None = None,
        state_key: str = "state",
        action_key: str = "actions",
        image_value_range: str = "zero_one",
        fps: int = 10,
        min_frames: int = 1,
        load_workers: int = 0,
    ) -> None:
        super().__init__()
        if int(action_dim) != FRANKA_GELLO_ACTION_DIM:
            raise ValueError(
                f"Franka + GELLO Flow BC requires model.action_dim=7, got {action_dim}."
            )
        if int(state_dim) <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}.")
        if int(fps) <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if int(min_frames) <= 0:
            raise ValueError(f"min_frames must be positive, got {min_frames}.")
        if int(load_workers) < 0:
            raise ValueError(f"load_workers must be non-negative, got {load_workers}.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.image_keys = tuple(image_keys)
        self.image_shape = _normalize_image_shape(image_shape)
        self.state_key = state_key
        self.action_key = action_key
        self.image_value_range = image_value_range

        shards = discover_lerobot_shards(data_paths)
        rolling_dataset_cls = _rolling_dataset_cls()
        self._base = rolling_dataset_cls(
            root_dir=Path(shards[0]).parent,
            chunk_size=1,
            keys=[state_key, action_key, *self.image_keys],
            min_frames=int(min_frames),
            action_sequence_keys=[action_key],
            in_memory_mode=True,
            fps=int(fps),
        )
        staged = self._base.load_archived_shards_staged(
            shards, num_workers=int(load_workers)
        )
        episodes, frames = self._base.publish_staged_resume_shards(staged)
        if frames < int(min_frames):
            raise ValueError(
                f"Flow BC dataset has {frames} usable frames, fewer than "
                f"data.min_frames={min_frames}."
            )
        logger.info(
            "Loaded Franka GELLO Flow BC data: shards=%d episodes=%d frames=%d",
            len(shards),
            episodes,
            frames,
        )

        # Fail before distributed training starts if the stored schema differs
        # from the policy contract.
        self[0]

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return adapt_franka_gello_sample(
            self._base[index],
            state_key=self.state_key,
            action_key=self.action_key,
            image_keys=self.image_keys,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            image_shape=self.image_shape,
            image_value_range=self.image_value_range,
        )


def _validate_flow_bc_model_contract(model_cfg: Any, image_keys: Sequence[str]) -> None:
    actor_type = str(model_cfg.get("flow_actor_type", ""))
    if actor_type != "FlowTActor":
        raise ValueError(
            "Flow BC supports only the PyTorch FlowTActor; "
            f"configured flow_actor_type={actor_type!r}."
        )
    if str(model_cfg.get("input_type", "mixed")) != "mixed":
        raise ValueError(
            "Franka + GELLO image Flow BC requires model.input_type='mixed'."
        )
    if int(model_cfg.get("action_dim", -1)) != FRANKA_GELLO_ACTION_DIM:
        raise ValueError(
            "Franka + GELLO Flow BC requires model.action_dim=7; "
            f"got {model_cfg.get('action_dim', None)!r}."
        )
    if int(model_cfg.get("num_action_chunks", 1)) != 1:
        raise ValueError(
            "Franka + GELLO Flow BC currently supports num_action_chunks=1 only."
        )
    image_num = int(model_cfg.get("image_num", 1))
    if image_num != len(image_keys):
        raise ValueError(
            f"model.image_num={image_num} but data.image_keys has "
            f"{len(image_keys)} entries."
        )
    image_size = model_cfg.get("image_size", None)
    if image_size is None:
        raise ValueError("Franka + GELLO image Flow BC requires model.image_size.")
    _normalize_image_shape(image_size)


def build_franka_gello_flow_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build the distributed stateful dataloader used by ``SFTRunner``."""
    if int(world_size) <= 0 or int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(
            f"Invalid rank/world_size: rank={rank}, world_size={world_size}."
        )

    model_cfg = cfg.actor.model
    data_cfg = cfg.data
    image_value_range = str(data_cfg.get("image_value_range", "zero_one"))
    if image_value_range not in {"zero_one", "auto"}:
        raise ValueError(
            "RollingLeRobotDataset decodes Franka images as float [0,1]; "
            "data.image_value_range must be 'zero_one' or 'auto'."
        )
    image_keys = tuple(data_cfg.get("image_keys", ["image"]))
    _validate_flow_bc_model_contract(model_cfg, image_keys)

    dataset = FrankaGelloFlowDataset(
        data_paths,
        state_dim=int(model_cfg.state_dim),
        action_dim=int(model_cfg.action_dim),
        image_keys=image_keys,
        image_shape=tuple(model_cfg.image_size),
        state_key=str(data_cfg.get("state_key", "state")),
        action_key=str(data_cfg.get("action_key", "actions")),
        # RollingLeRobotDataset decodes images as float CHW in [0,1]. Keeping
        # this explicit detects accidental uint8/double-normalized paths.
        image_value_range=image_value_range,
        fps=int(data_cfg.get("fps", 10)),
        min_frames=int(data_cfg.get("min_frames", 1)),
        load_workers=int(data_cfg.get("load_workers", 0)),
    )
    use_random_replacement = (
        bool(data_cfg.get("use_random_replacement", False)) and not eval_dataset
    )
    configured_samples = data_cfg.get("num_samples_per_epoch", None)
    num_samples_per_epoch = (
        None if configured_samples is None else int(configured_samples)
    )
    if num_samples_per_epoch is not None and num_samples_per_epoch <= 0:
        raise ValueError(
            "data.num_samples_per_epoch must be positive when configured, got "
            f"{num_samples_per_epoch}."
        )
    seed = int(cfg.actor.get("seed", 0))
    if use_random_replacement:
        # Reuse DAgger's tested replacement samplers while retaining a
        # StatefulDataLoader so SFT checkpoint/resume keeps dataloader state.
        from rlinf.data.datasets.dagger.dataloader import (
            DistributedRandomReplacementSampler,
            RandomReplacementSampler,
        )

        if int(world_size) > 1:
            sampler = DistributedRandomReplacementSampler(
                dataset,
                num_samples=num_samples_per_epoch,
                num_replicas=int(world_size),
                rank=int(rank),
                seed=seed,
            )
        else:
            sampler = RandomReplacementSampler(
                dataset,
                num_samples=num_samples_per_epoch,
                seed=seed,
            )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=not eval_dataset,
            drop_last=not eval_dataset,
            seed=seed,
        )
    num_workers = int(data_cfg.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError(f"data.num_workers must be non-negative, got {num_workers}.")
    batch_size = int(cfg.actor.micro_batch_size)
    if batch_size <= 0:
        raise ValueError(f"actor.micro_batch_size must be positive, got {batch_size}.")
    if not eval_dataset and len(sampler) < batch_size:
        raise ValueError(
            f"Flow BC sampler provides {len(sampler)} samples per rank, fewer than "
            f"micro_batch_size={batch_size} with drop_last=True. Enable "
            "data.use_random_replacement and increase data.num_samples_per_epoch."
        )

    # Imported here so the schema adapter remains unit-testable in lightweight
    # environments that do not install the SFT runtime extras.
    from torchdata.stateful_dataloader import StatefulDataLoader

    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=(
            int(data_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
        ),
    )
    return loader, {
        "num_samples": len(dataset),
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
        "image_keys": list(dataset.image_keys),
        "image_shape": list(dataset.image_shape) if dataset.image_shape else None,
        "image_layout": "CHW",
        "image_value_range": "zero_one",
        "images_preprocessed": True,
        "use_random_replacement": use_random_replacement,
        "num_samples_per_epoch": num_samples_per_epoch,
    }
