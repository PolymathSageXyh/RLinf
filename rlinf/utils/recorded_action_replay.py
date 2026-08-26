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

"""Load recorded real-world actions without importing training datasets."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import torch

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root

logger = logging.getLogger(__name__)

RECORDED_ACTIONS_FORMAT = "rlinf.recorded_actions/v1"


def _validate_recorded_actions(
    actions,
    *,
    action_dim: int,
    source: str | Path,
) -> torch.Tensor:
    """Convert recorded actions to a finite ``[T, action_dim]`` tensor."""
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[1] != int(action_dim):
        raise ValueError(
            f"Replay actions from {source} must have shape [T, {action_dim}], got "
            f"{tuple(actions.shape)}."
        )
    if actions.shape[0] == 0:
        raise ValueError(f"Replay action source {source} contains no actions.")
    if not torch.isfinite(actions).all():
        raise ValueError(f"Replay action source {source} contains non-finite actions.")
    if torch.any(actions < -1.0) or torch.any(actions > 1.0):
        logger.warning(
            "Recorded actions from %s extend outside [-1, 1] (min=%f, max=%f). "
            "Values are preserved because frame conversion can expand the "
            "recorded policy-frame representation.",
            source,
            float(actions.min()),
            float(actions.max()),
        )
    return actions.contiguous()


def load_recorded_actions_file(
    actions_path: str | Path,
    *,
    action_dim: int,
) -> torch.Tensor:
    """Load a portable recorded-action JSON trajectory."""
    actions_path = Path(actions_path).expanduser().resolve()
    if not actions_path.is_file():
        raise FileNotFoundError(f"Recorded-action JSON does not exist: {actions_path}")
    with actions_path.open(encoding="utf-8") as actions_file:
        payload = json.load(actions_file)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Recorded-action JSON must contain an object: {actions_path}")
    if payload.get("format") != RECORDED_ACTIONS_FORMAT:
        raise ValueError(
            f"Recorded-action JSON format must be {RECORDED_ACTIONS_FORMAT!r}, "
            f"got {payload.get('format')!r} in {actions_path}."
        )
    if "actions" not in payload:
        raise ValueError(f"Recorded-action JSON has no 'actions' field: {actions_path}")
    actions = _validate_recorded_actions(
        payload["actions"],
        action_dim=action_dim,
        source=actions_path,
    )
    logger.info(
        "Loaded portable recorded-action trajectory from %s (%d actions).",
        actions_path,
        actions.shape[0],
    )
    return actions


def _resolve_replay_shard(
    data_path: str | Path,
    *,
    rank: int,
    shard_id: int,
) -> Path:
    """Resolve one finalized collector shard for real-world replay."""
    root = resolve_lerobot_dataset_root(str(data_path))
    candidates = [
        root,
        root / f"rank_{int(rank)}" / f"id_{int(shard_id)}",
        root / f"id_{int(shard_id)}",
    ]
    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file() and (
            candidate / "meta" / "episodes.jsonl"
        ).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No finalized LeRobot shard found for replay at "
        f"data_path={data_path!s}, rank={rank}, shard_id={shard_id}."
    )


def _episode_parquet_path(
    shard: Path,
    info: Mapping,
    episode_index: int,
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return shard / str(data_path).format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def load_franka_gello_episode_actions(
    data_path: str | Path,
    episode_index: int,
    *,
    rank: int = 0,
    shard_id: int = 0,
    action_dim: int,
) -> torch.Tensor:
    """Load one collector trajectory directly from a finalized LeRobot shard."""
    if int(episode_index) < 0:
        raise ValueError(f"episode_index must be non-negative, got {episode_index}.")
    shard = _resolve_replay_shard(data_path, rank=rank, shard_id=shard_id)
    with (shard / "meta" / "info.json").open(encoding="utf-8") as info_file:
        info = json.load(info_file)
    with (shard / "meta" / "episodes.jsonl").open(encoding="utf-8") as episodes_file:
        episodes = [json.loads(line) for line in episodes_file if line.strip()]
    episode = next(
        (
            record
            for record in episodes
            if int(record.get("episode_index", -1)) == int(episode_index)
        ),
        None,
    )
    if episode is None:
        available = [int(record["episode_index"]) for record in episodes]
        raise ValueError(
            f"Episode {episode_index} is not present in {shard}; "
            f"available episodes={available}."
        )

    parquet_path = _episode_parquet_path(shard, info, int(episode_index))
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Replay episode parquet does not exist: {parquet_path}"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Direct LeRobot replay requires pyarrow in the rollout runtime. "
            "Use recorded_action_replay.actions_path for a portable JSON replay."
        ) from exc
    try:
        table = pq.read_table(parquet_path, columns=["actions"])
    except Exception as exc:
        raise RuntimeError(
            "Failed to read recorded actions from "
            f"{parquet_path}. Use a pyarrow version compatible with the "
            "LeRobot writer, or export a portable recorded-action JSON."
        ) from exc
    actions = _validate_recorded_actions(
        table["actions"].to_pylist(),
        action_dim=action_dim,
        source=parquet_path,
    )
    logger.info(
        "Loaded recorded-action replay episode=%d from %s (%d actions).",
        int(episode_index),
        parquet_path,
        actions.shape[0],
    )
    return actions
