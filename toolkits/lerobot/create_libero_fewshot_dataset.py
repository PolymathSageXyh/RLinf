#!/usr/bin/env python3
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

"""Create a small LeRobot LIBERO subset for few-shot SFT before RL.

The source dataset is expected to be the combined OpenPI/RLinf LIBERO v2
LeRobot layout:

    libero_10/long: task_index 0..9
    libero_goal:    task_index 10..19
    libero_object:  task_index 20..29
    libero_spatial: task_index 30..39

Example:

    python toolkits/lerobot/create_libero_fewshot_dataset.py \
        --total-episodes 40 \
        --suite-ratios spatial=1,object=1,goal=1,long=1 \
        --seed 0

The output is a complete LeRobot dataset directory. Parquet files are rewritten
with compact episode_index/index/task_index values, while a manifest records the
original episode ids for reproducibility.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE_DIR = Path("/data0/vla_dataset/libero/libero-v2/libero")
DEFAULT_OUTPUT_ROOT = Path("/data0/vla_dataset/libero/libero-v2")

SUITE_TASK_RANGES: dict[str, range] = {
    "long": range(0, 10),
    "goal": range(10, 20),
    "object": range(20, 30),
    "spatial": range(30, 40),
}

SUITE_ALIASES = {
    "libero_10": "long",
    "libero_long": "long",
    "libero-goal": "goal",
    "libero_goal": "goal",
    "libero-object": "object",
    "libero_object": "object",
    "libero-spatial": "spatial",
    "libero_spatial": "spatial",
}

NUMERIC_STAT_COLUMNS = (
    "state",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)

REQUIRED_PARQUET_COLUMNS = {
    "episode_index",
    "index",
    "frame_index",
    "task_index",
}


def _canonical_suite(name: str) -> str:
    key = name.strip().lower()
    key = SUITE_ALIASES.get(key, key)
    if key not in SUITE_TASK_RANGES:
        valid = ", ".join(sorted(SUITE_TASK_RANGES))
        raise ValueError(f"Unknown suite {name!r}. Expected one of: {valid}.")
    return key


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, value: Any, *, indent: int = 4) -> None:
    with open(path, "w") as f:
        json.dump(value, f, indent=indent)
        f.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_suite_ratios(args: argparse.Namespace) -> dict[str, float]:
    explicit_values = {
        "spatial": args.spatial_ratio,
        "object": args.object_ratio,
        "goal": args.goal_ratio,
        "long": args.long_ratio,
    }
    has_explicit = any(value is not None for value in explicit_values.values())

    if args.suite_ratios and has_explicit:
        raise ValueError(
            "Use either --suite-ratios or the individual --*-ratio flags, not both."
        )

    if args.suite_ratios:
        ratios = dict.fromkeys(SUITE_TASK_RANGES, 0.0)
        seen_suites: set[str] = set()
        for part in args.suite_ratios.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"Invalid suite ratio item {part!r}; expected suite=value."
                )
            name, raw_value = part.split("=", 1)
            suite = _canonical_suite(name)
            if suite in seen_suites:
                raise ValueError(f"Suite {suite!r} is specified more than once.")
            seen_suites.add(suite)
            ratios[suite] = float(raw_value)
    elif has_explicit:
        ratios = {
            suite: float(value) if value is not None else 0.0
            for suite, value in explicit_values.items()
        }
    else:
        ratios = dict.fromkeys(SUITE_TASK_RANGES, 1.0)

    for suite, value in ratios.items():
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"Ratio for suite {suite!r} must be finite and >= 0.")
    if sum(ratios.values()) <= 0:
        raise ValueError("At least one suite ratio must be > 0.")
    return {suite: ratios[suite] for suite in SUITE_TASK_RANGES}


def _largest_remainder_with_caps(
    total: int,
    weights: dict[int | str, float],
    caps: dict[int | str, int],
    *,
    mins: dict[int | str, int] | None = None,
) -> dict[int | str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")

    mins = mins or dict.fromkeys(weights, 0)
    keys = list(weights)
    for key in keys:
        if caps.get(key, 0) < mins.get(key, 0):
            raise ValueError(f"Minimum allocation exceeds cap for {key!r}.")

    min_total = sum(mins.get(key, 0) for key in keys)
    cap_total = sum(caps.get(key, 0) for key in keys)
    if total < min_total:
        raise ValueError(f"Cannot allocate {total}; minimum required is {min_total}.")
    if total > cap_total:
        raise ValueError(f"Cannot allocate {total}; only {cap_total} items available.")

    counts = {key: mins.get(key, 0) for key in keys}
    remaining = total - min_total
    if remaining == 0:
        return counts

    active = [key for key in keys if caps.get(key, 0) > counts[key]]
    while remaining > 0:
        active = [key for key in active if caps[key] > counts[key]]
        if not active:
            raise ValueError("Internal allocation error: no capacity left.")

        total_weight = sum(max(weights.get(key, 0.0), 0.0) for key in active)
        if total_weight <= 0:
            total_weight = float(len(active))
            effective_weights = dict.fromkeys(active, 1.0)
        else:
            effective_weights = {key: max(weights.get(key, 0.0), 0.0) for key in active}

        raw = {key: remaining * effective_weights[key] / total_weight for key in active}
        floors = {
            key: min(int(math.floor(raw[key])), caps[key] - counts[key])
            for key in active
        }
        added = sum(floors.values())
        for key, value in floors.items():
            counts[key] += value
        remaining -= added

        if remaining <= 0:
            break

        order = sorted(
            active,
            key=lambda key: (
                raw[key] - math.floor(raw[key]),
                effective_weights[key],
            ),
            reverse=True,
        )
        made_progress = False
        for key in order:
            if remaining == 0:
                break
            if counts[key] >= caps[key]:
                continue
            counts[key] += 1
            remaining -= 1
            made_progress = True
        if not made_progress:
            raise ValueError("Internal allocation error: largest remainder stalled.")

    return counts


def _allocate_suite_counts(
    total: int,
    ratios: dict[str, float],
    suite_caps: dict[str, int],
    *,
    ensure_positive_suites: bool,
) -> dict[str, int]:
    active_suites = [suite for suite, ratio in ratios.items() if ratio > 0]
    if ensure_positive_suites and total >= len(active_suites):
        empty = [suite for suite in active_suites if suite_caps.get(suite, 0) == 0]
        if empty:
            raise ValueError(
                "Cannot give every positive-ratio suite an episode; no episodes "
                f"are available for: {', '.join(empty)}."
            )
    counts = {
        str(k): v
        for k, v in _largest_remainder_with_caps(
            total,
            weights=ratios,
            caps=suite_caps,
        ).items()
    }
    if not ensure_positive_suites or total < len(active_suites):
        return counts

    for suite in active_suites:
        if counts[suite] > 0:
            continue
        donors = [candidate for candidate in active_suites if counts[candidate] > 1]
        if not donors:
            return counts
        donor = max(
            donors,
            key=lambda candidate: (
                counts[candidate],
                ratios[candidate],
                candidate,
            ),
        )
        if suite_caps[suite] <= 0:
            raise ValueError(f"Cannot allocate an episode to empty suite {suite!r}.")
        counts[donor] -= 1
        counts[suite] += 1
    return counts


def _allocate_task_counts(
    suite: str,
    suite_total: int,
    episodes_by_task: dict[int, list[dict[str, Any]]],
    *,
    min_per_task: int,
) -> dict[int, int]:
    task_ids = list(SUITE_TASK_RANGES[suite])
    caps = {task_id: len(episodes_by_task.get(task_id, [])) for task_id in task_ids}
    weights = dict.fromkeys(task_ids, 1.0)

    if min_per_task > 0 and suite_total >= min_per_task * len(task_ids):
        mins = {task_id: min(min_per_task, caps[task_id]) for task_id in task_ids}
    else:
        mins = dict.fromkeys(task_ids, 0)

    return {
        int(k): v
        for k, v in _largest_remainder_with_caps(
            suite_total,
            weights=weights,
            caps=caps,
            mins=mins,
        ).items()
    }


def _source_episode_path(
    source_dir: Path, info: dict[str, Any], episode_index: int
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel_path = data_path.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return source_dir / rel_path


def _replace_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Required Parquet column is missing: {name!r}.")
    field = table.schema.field(index)
    array = pa.array(values, type=field.type)
    return table.set_column(index, field, array)


def _column_to_numpy(table: pa.Table, name: str) -> np.ndarray | None:
    if name not in table.column_names:
        return None
    values = table[name].combine_chunks().to_pylist()
    if not values:
        return None
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float64, copy=False)


def _validate_source_metadata(
    info: dict[str, Any],
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> None:
    expected_task_ids = set(range(40))
    task_ids = [int(record["task_index"]) for record in tasks]
    task_texts = [record["task"] for record in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Source tasks.jsonl contains duplicate task_index values.")
    if len(task_texts) != len(set(task_texts)):
        raise ValueError("Source tasks.jsonl contains duplicate task strings.")
    if set(task_ids) != expected_task_ids:
        raise ValueError(
            "Expected exactly LIBERO task_index 0..39; got "
            f"{sorted(task_ids)}."
        )
    if int(info.get("total_tasks", -1)) != len(tasks):
        raise ValueError("info.json total_tasks does not match tasks.jsonl.")

    episode_ids = [int(record["episode_index"]) for record in episodes]
    if episode_ids != list(range(len(episodes))):
        raise ValueError(
            "Source episodes.jsonl must be ordered with contiguous episode_index "
            "values 0..N-1."
        )
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json total_episodes does not match episodes.jsonl.")
    for episode in episodes:
        episode_tasks = episode.get("tasks")
        if not isinstance(episode_tasks, list) or len(episode_tasks) != 1:
            raise ValueError(
                f"Episode {episode['episode_index']} must reference exactly one task."
            )
        length = episode.get("length")
        if not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"Episode {episode['episode_index']} has invalid length {length!r}."
            )
    total_frames = sum(int(record["length"]) for record in episodes)
    if int(info.get("total_frames", -1)) != total_frames:
        raise ValueError(
            f"info.json total_frames={info.get('total_frames')} but "
            f"episodes.jsonl sums to {total_frames}."
        )
    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("info.json chunks_size must be a positive integer.")
    expected_chunks = max(1, math.ceil(len(episodes) / chunks_size))
    if int(info.get("total_chunks", -1)) != expected_chunks:
        raise ValueError(
            f"info.json total_chunks={info.get('total_chunks')} but expected "
            f"{expected_chunks}."
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json features must be an object.")
    missing = REQUIRED_PARQUET_COLUMNS - set(features)
    if missing:
        raise ValueError(f"info.json features is missing: {sorted(missing)}.")
    if int(info.get("total_videos", 0)) != 0:
        raise ValueError(
            "This subset writer supports inline Parquet images only; source "
            "info.json declares external videos."
        )


def _image_stat_buffers(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "count": 0,
            "sum": None,
            "sum_sq": None,
            "min": None,
            "max": None,
        }
        for name, feature in info["features"].items()
        if feature.get("dtype") == "image"
    }


def _update_image_stats(
    table: pa.Table,
    buffers: dict[str, dict[str, Any]],
) -> None:
    if not buffers:
        return
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to recompute exact subset image statistics."
        ) from exc

    for name, stats in buffers.items():
        if name not in table.column_names:
            raise ValueError(f"Image feature {name!r} is missing from Parquet.")
        for frame_index, value in enumerate(table[name].combine_chunks().to_pylist()):
            payload: bytes | None
            if isinstance(value, (bytes, bytearray)):
                payload = bytes(value)
            elif isinstance(value, dict):
                payload = value.get("bytes")
                if payload is None and value.get("path"):
                    raise ValueError(
                        f"Image {name!r} frame {frame_index} stores only a path. "
                        "The output would not be self-contained."
                    )
            else:
                payload = None
            if payload is None:
                raise ValueError(
                    f"Image {name!r} frame {frame_index} has no embedded bytes."
                )
            with Image.open(io.BytesIO(payload)) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
            flat = pixels.reshape(-1, pixels.shape[-1])
            channel_sum = flat.sum(axis=0)
            channel_sum_sq = np.square(flat).sum(axis=0)
            channel_min = flat.min(axis=0)
            channel_max = flat.max(axis=0)
            stats["count"] += flat.shape[0]
            stats["sum"] = (
                channel_sum if stats["sum"] is None else stats["sum"] + channel_sum
            )
            stats["sum_sq"] = (
                channel_sum_sq
                if stats["sum_sq"] is None
                else stats["sum_sq"] + channel_sum_sq
            )
            stats["min"] = (
                channel_min
                if stats["min"] is None
                else np.minimum(stats["min"], channel_min)
            )
            stats["max"] = (
                channel_max
                if stats["max"] is None
                else np.maximum(stats["max"], channel_max)
            )


def _finalize_image_stats(
    buffers: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[Any]]]:
    result: dict[str, dict[str, list[Any]]] = {}
    for name, stats in buffers.items():
        count = int(stats["count"])
        if count == 0:
            raise ValueError(f"No pixels were collected for image feature {name!r}.")
        mean = stats["sum"] / count
        variance = np.maximum(stats["sum_sq"] / count - np.square(mean), 0.0)
        result[name] = {
            "mean": mean.reshape(-1, 1, 1).tolist(),
            "std": np.sqrt(variance).reshape(-1, 1, 1).tolist(),
            "max": stats["max"].reshape(-1, 1, 1).tolist(),
            "min": stats["min"].reshape(-1, 1, 1).tolist(),
        }
    return result


def _validate_source_episode_table(
    table: pa.Table,
    *,
    episode: dict[str, Any],
    expected_task_index: int,
    info: dict[str, Any],
    path: Path,
) -> None:
    expected_columns = set(info["features"])
    actual_columns = set(table.column_names)
    if actual_columns != expected_columns:
        raise ValueError(
            f"{path}: Parquet columns differ from info.json features; "
            f"missing={sorted(expected_columns - actual_columns)}, "
            f"extra={sorted(actual_columns - expected_columns)}."
        )
    if table.num_rows != int(episode["length"]):
        raise ValueError(
            f"{path}: has {table.num_rows} rows but episodes.jsonl says "
            f"{episode['length']}."
        )
    episode_index = int(episode["episode_index"])
    checks = {
        "episode_index": [episode_index] * table.num_rows,
        "frame_index": list(range(table.num_rows)),
        "task_index": [expected_task_index] * table.num_rows,
    }
    for name, expected in checks.items():
        actual = table[name].combine_chunks().to_pylist()
        if actual != expected:
            raise ValueError(f"{path}: column {name!r} is inconsistent with metadata.")


def _validate_output_dataset(
    output_dir: Path,
    *,
    expected_info: dict[str, Any],
    expected_tasks: list[dict[str, Any]],
    expected_episodes: list[dict[str, Any]],
    expected_stats: dict[str, Any],
    expected_manifest: dict[str, Any],
) -> None:
    info = _read_json(output_dir / "meta" / "info.json")
    tasks = _read_jsonl(output_dir / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(output_dir / "meta" / "episodes.jsonl")
    stats = _read_json(output_dir / "meta" / "stats.json")
    manifest = _read_json(output_dir / "meta" / "fewshot_selection.json")
    if info != expected_info:
        raise ValueError("Written info.json does not match the generated metadata.")
    if tasks != expected_tasks:
        raise ValueError("Written tasks.jsonl does not match the task mapping.")
    if episodes != expected_episodes:
        raise ValueError("Written episodes.jsonl does not match selected episodes.")
    if stats != expected_stats:
        raise ValueError("Written stats.json does not match computed subset statistics.")
    if manifest != expected_manifest:
        raise ValueError("Written fewshot_selection.json does not match selection.")
    if info["total_tasks"] != len(tasks):
        raise ValueError("Output total_tasks does not match tasks.jsonl.")
    if info["total_episodes"] != len(episodes):
        raise ValueError("Output total_episodes does not match episodes.jsonl.")
    if sum(record["length"] for record in episodes) != info["total_frames"]:
        raise ValueError("Output total_frames does not match episodes.jsonl.")

    task_by_text = {record["task"]: int(record["task_index"]) for record in tasks}
    if len(task_by_text) != len(tasks):
        raise ValueError("Output tasks.jsonl contains duplicate task strings.")
    expected_paths: set[Path] = set()
    global_frame_index = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index < 0 or episode_index >= len(episodes):
            raise ValueError("Output episode_index is outside the compact range.")
        chunk_index = episode_index // int(info["chunks_size"])
        path = output_dir / info["data_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        expected_paths.add(path.resolve())
        if not path.is_file():
            raise ValueError(f"Output Parquet is missing: {path}.")
        table = pq.read_table(path)
        n_frames = int(episode["length"])
        task = episode["tasks"][0]
        checks = {
            "episode_index": [episode_index] * n_frames,
            "frame_index": list(range(n_frames)),
            "index": list(range(global_frame_index, global_frame_index + n_frames)),
            "task_index": [task_by_text[task]] * n_frames,
        }
        if table.num_rows != n_frames:
            raise ValueError(f"{path}: row count does not match episodes.jsonl.")
        if set(table.column_names) != set(info["features"]):
            raise ValueError(f"{path}: columns do not match info.json features.")
        for name, expected in checks.items():
            if table[name].combine_chunks().to_pylist() != expected:
                raise ValueError(f"{path}: output column {name!r} is inconsistent.")
        global_frame_index += n_frames
    actual_paths = {path.resolve() for path in (output_dir / "data").rglob("*.parquet")}
    if actual_paths != expected_paths:
        raise ValueError("Output data/ contains missing or unexpected Parquet files.")
    if global_frame_index != info["total_frames"]:
        raise ValueError("Output Parquet rows do not match total_frames.")


def _summarize_array(arr: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "min": arr.min(axis=0).astype(float).tolist(),
    }


def _norm_stats_for_array(arr: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
    }


def _compute_stats(
    buffers: dict[str, list[np.ndarray]],
    image_buffers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stats: dict[str, Any] = _finalize_image_stats(image_buffers)
    for name, parts in buffers.items():
        if not parts:
            continue
        arr = np.concatenate(parts, axis=0)
        stats[name] = _summarize_array(arr)
    return stats


def _compute_norm_stats(
    buffers: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    norm_stats: dict[str, Any] = {}
    for name in ("state", "actions"):
        if not buffers.get(name):
            continue
        arr = np.concatenate(buffers[name], axis=0)
        norm_stats[name] = _norm_stats_for_array(arr)
    return {"norm_stats": norm_stats}


def _format_ratio_for_name(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return text.replace("-", "m")


def _default_output_name(total: int, ratios: dict[str, float], seed: int) -> str:
    parts = [
        f"sp{_format_ratio_for_name(ratios['spatial'])}",
        f"ob{_format_ratio_for_name(ratios['object'])}",
        f"go{_format_ratio_for_name(ratios['goal'])}",
        f"lo{_format_ratio_for_name(ratios['long'])}",
    ]
    return f"libero_fewshot_{total}_{'_'.join(parts)}_seed{seed}"


def _prepare_output_dir(output_dir: Path, *, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True)
    (output_dir / "data").mkdir()


def _copy_optional_root_files(source_dir: Path, output_dir: Path) -> None:
    for filename in (".gitattributes", "README.md"):
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)


def create_subset(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ratios = _parse_suite_ratios(args)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else output_root
        / (
            args.output_name
            or _default_output_name(args.total_episodes, ratios, args.seed)
        )
    )

    if args.total_episodes <= 0:
        raise ValueError("--total-episodes must be > 0.")
    if args.min_per_task < 0:
        raise ValueError("--min-per-task must be >= 0.")
    if args.output_chunks_size <= 0:
        raise ValueError("--output-chunks-size must be > 0.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0.")
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("--output-dir must not equal or be inside --source-dir.")
    if not (source_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing LeRobot info.json under {source_dir}")

    info = _read_json(source_dir / "meta" / "info.json")
    tasks = _read_jsonl(source_dir / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(source_dir / "meta" / "episodes.jsonl")
    _validate_source_metadata(info, tasks, episodes)

    task_text_to_original_index = {
        record["task"]: int(record["task_index"]) for record in tasks
    }
    original_task_to_text = {
        int(record["task_index"]): record["task"] for record in tasks
    }

    episodes_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        episode_tasks = episode.get("tasks") or []
        if not episode_tasks:
            continue
        task = episode_tasks[0]
        if task not in task_text_to_original_index:
            raise ValueError(f"Episode references unknown task: {task!r}")
        original_task_index = task_text_to_original_index[task]
        enriched = dict(episode)
        enriched["original_task_index"] = original_task_index
        enriched["suite"] = next(
            (
                suite
                for suite, task_range in SUITE_TASK_RANGES.items()
                if original_task_index in task_range
            ),
            None,
        )
        if enriched["suite"] is None:
            raise ValueError(
                f"Task index {original_task_index} is outside LIBERO ranges."
            )
        episodes_by_task[original_task_index].append(enriched)

    suite_caps = {
        suite: sum(len(episodes_by_task[task_id]) for task_id in task_range)
        for suite, task_range in SUITE_TASK_RANGES.items()
    }
    if args.total_episodes > sum(
        cap for suite, cap in suite_caps.items() if ratios[suite] > 0
    ):
        raise ValueError(
            f"Requested {args.total_episodes} episodes, but selected suites only "
            f"contain {sum(cap for suite, cap in suite_caps.items() if ratios[suite] > 0)}."
        )

    suite_counts = _allocate_suite_counts(
        args.total_episodes,
        ratios,
        suite_caps,
        ensure_positive_suites=args.ensure_positive_suites,
    )
    task_counts: dict[int, int] = {}
    for suite, suite_total in suite_counts.items():
        if suite_total == 0:
            continue
        task_counts.update(
            _allocate_task_counts(
                suite,
                suite_total,
                episodes_by_task,
                min_per_task=args.min_per_task,
            )
        )

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for task_id in sorted(task_counts):
        count = task_counts[task_id]
        if count == 0:
            continue
        candidates = list(episodes_by_task[task_id])
        if args.selection == "random":
            selected.extend(rng.sample(candidates, count))
        else:
            selected.extend(
                sorted(candidates, key=lambda item: item["episode_index"])[:count]
            )

    if args.shuffle_output:
        rng.shuffle(selected)
    else:
        suite_order = {"long": 0, "goal": 1, "object": 2, "spatial": 3}
        selected.sort(
            key=lambda item: (
                suite_order[item["suite"]],
                item["original_task_index"],
                item["episode_index"],
            )
        )

    if args.compact_task_indices:
        selected_task_texts: list[str] = []
        for episode in selected:
            task = episode["tasks"][0]
            if task not in selected_task_texts:
                selected_task_texts.append(task)
        task_index_for_task = {
            task: idx for idx, task in enumerate(selected_task_texts)
        }
        output_tasks = [
            {"task_index": idx, "task": task}
            for task, idx in sorted(
                task_index_for_task.items(), key=lambda item: item[1]
            )
        ]
    else:
        task_index_for_task = {
            record["task"]: int(record["task_index"]) for record in tasks
        }
        output_tasks = [
            {"task_index": int(record["task_index"]), "task": record["task"]}
            for record in tasks
        ]

    print("[fewshot] Source:", source_dir)
    print("[fewshot] Output:", output_dir)
    print("[fewshot] Requested episodes:", args.total_episodes)
    print("[fewshot] Suite ratios:", ratios)
    print("[fewshot] Suite counts:", suite_counts)
    print(
        "[fewshot] Covered tasks:",
        f"{sum(1 for count in task_counts.values() if count > 0)}/{len(tasks)}",
    )
    if args.dry_run:
        for suite in ("spatial", "object", "goal", "long"):
            counts = [
                task_counts.get(task_id, 0) for task_id in SUITE_TASK_RANGES[suite]
            ]
            print(f"[fewshot] {suite:7s} task counts:", counts)
        print("[fewshot] Dry run only; no files written.")
        return output_dir

    _prepare_output_dir(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)
    _copy_optional_root_files(source_dir, output_dir)

    output_chunks_size = int(args.output_chunks_size)
    global_frame_index = 0
    output_episodes: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    stats_buffers: dict[str, list[np.ndarray]] = {
        name: [] for name in NUMERIC_STAT_COLUMNS
    }
    image_buffers = _image_stat_buffers(info)
    unsupported_stats = set(info["features"]) - set(stats_buffers) - set(image_buffers)
    if unsupported_stats:
        raise ValueError(
            "Cannot compute exact stats for unsupported features: "
            f"{sorted(unsupported_stats)}."
        )

    for new_episode_index, episode in enumerate(selected):
        old_episode_index = int(episode["episode_index"])
        source_parquet = _source_episode_path(source_dir, info, old_episode_index)
        if not source_parquet.is_file():
            raise FileNotFoundError(f"Missing source parquet: {source_parquet}")

        table = pq.read_table(source_parquet)
        task = episode["tasks"][0]
        _validate_source_episode_table(
            table,
            episode=episode,
            expected_task_index=int(episode["original_task_index"]),
            info=info,
            path=source_parquet,
        )
        n_frames = table.num_rows
        new_task_index = task_index_for_task[task]

        table = _replace_column(table, "episode_index", [new_episode_index] * n_frames)
        table = _replace_column(
            table,
            "index",
            range(global_frame_index, global_frame_index + n_frames),
        )
        table = _replace_column(table, "frame_index", range(n_frames))
        table = _replace_column(table, "task_index", [new_task_index] * n_frames)

        for name in NUMERIC_STAT_COLUMNS:
            arr = _column_to_numpy(table, name)
            if arr is not None:
                stats_buffers[name].append(arr)
        _update_image_stats(table, image_buffers)

        chunk_index = new_episode_index // output_chunks_size
        chunk_dir = output_dir / "data" / f"chunk-{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = chunk_dir / f"episode_{new_episode_index:06d}.parquet"
        pq.write_table(table, out_parquet)

        output_episode = {
            "episode_index": new_episode_index,
            "tasks": [task],
            "length": n_frames,
        }
        for key, value in episode.items():
            if key not in {
                "episode_index",
                "tasks",
                "length",
                "original_task_index",
                "suite",
            }:
                output_episode[key] = value
        output_episodes.append(output_episode)

        manifest_records.append(
            {
                "new_episode_index": new_episode_index,
                "original_episode_index": old_episode_index,
                "original_task_index": int(episode["original_task_index"]),
                "new_task_index": new_task_index,
                "suite": episode["suite"],
                "task": task,
                "length": n_frames,
            }
        )

        global_frame_index += n_frames
        if (new_episode_index + 1) % args.log_every == 0 or (
            new_episode_index + 1
        ) == len(selected):
            print(
                f"[fewshot] Wrote {new_episode_index + 1}/{len(selected)} episodes "
                f"({global_frame_index} frames)"
            )

    info_out = dict(info)
    total_chunks = (len(selected) + output_chunks_size - 1) // output_chunks_size
    info_out.update(
        {
            "total_episodes": len(selected),
            "total_frames": global_frame_index,
            "total_tasks": len(output_tasks),
            "total_videos": 0,
            "total_chunks": max(1, total_chunks),
            "chunks_size": output_chunks_size,
            "splits": {"train": f"0:{len(selected)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        }
    )
    if not (output_dir / "videos").exists():
        info_out.pop("video_path", None)

    _write_json(output_dir / "meta" / "info.json", info_out)
    _write_jsonl(output_dir / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(output_dir / "meta" / "tasks.jsonl", output_tasks)

    stats = _compute_stats(stats_buffers, image_buffers)
    if set(stats) != set(info_out["features"]):
        raise ValueError(
            "Computed stats keys do not exactly match info.json features: "
            f"stats={sorted(stats)}, features={sorted(info_out['features'])}."
        )
    _write_json(output_dir / "meta" / "stats.json", stats)

    norm_stats_mode = args.norm_stats
    source_norm_stats = source_dir / "norm_stats.json"
    if norm_stats_mode == "copy":
        if source_norm_stats.is_file():
            shutil.copy2(source_norm_stats, output_dir / "norm_stats.json")
        else:
            print("[fewshot] WARNING: source norm_stats.json missing; skipping copy.")
    elif norm_stats_mode == "compute":
        _write_json(
            output_dir / "norm_stats.json", _compute_norm_stats(stats_buffers), indent=2
        )

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "total_episodes": len(selected),
        "total_frames": global_frame_index,
        "seed": args.seed,
        "selection": args.selection,
        "shuffle_output": args.shuffle_output,
        "compact_task_indices": args.compact_task_indices,
        "suite_ratios": ratios,
        "suite_counts": suite_counts,
        "task_counts": {
            str(task_id): count
            for task_id, count in sorted(task_counts.items())
            if count > 0
        },
        "suite_task_ranges": {
            suite: [task_range.start, task_range.stop - 1]
            for suite, task_range in SUITE_TASK_RANGES.items()
        },
        "original_task_names": {
            str(task_id): original_task_to_text[task_id]
            for task_id in sorted(original_task_to_text)
        },
        "records": manifest_records,
        "notes": [
            (
                "Parquet episode_index/index values were compacted; "
                "task_index values were also compacted."
                if args.compact_task_indices
                else (
                    "Parquet episode_index/index values were compacted; "
                    "task_index values preserve the source LIBERO suite mapping."
                )
            ),
            "meta/stats.json exactly recomputes numeric and image fields "
            "from the selected subset.",
            f"norm_stats mode: {norm_stats_mode}.",
        ],
    }
    _write_json(output_dir / "meta" / "fewshot_selection.json", manifest, indent=2)

    _validate_output_dataset(
        output_dir,
        expected_info=info_out,
        expected_tasks=output_tasks,
        expected_episodes=output_episodes,
        expected_stats=stats,
        expected_manifest=manifest,
    )
    print("[fewshot] Validation passed: all metadata and Parquet indices agree.")
    print("[fewshot] Done:", output_dir)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a small LeRobot LIBERO dataset by suite proportions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. Overrides --output-root/--output-name.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Directory name under --output-root. Auto-generated if omitted.",
    )
    parser.add_argument("--total-episodes", type=int, required=True)
    parser.add_argument(
        "--suite-ratios",
        default=None,
        help=(
            "Comma-separated ratios, e.g. spatial=1,object=1,goal=1,long=1. "
            "Values are normalized, so percentages and weights both work."
        ),
    )
    parser.add_argument("--spatial-ratio", type=float, default=None)
    parser.add_argument("--object-ratio", type=float, default=None)
    parser.add_argument("--goal-ratio", type=float, default=None)
    parser.add_argument("--long-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("random", "first"),
        default="random",
        help="How to select episodes inside each task bucket.",
    )
    parser.add_argument(
        "--min-per-task",
        type=int,
        default=1,
        help=(
            "Minimum per task when a suite allocation is large enough to cover "
            "all ten tasks in that suite."
        ),
    )
    parser.add_argument(
        "--no-ensure-positive-suites",
        action="store_false",
        dest="ensure_positive_suites",
        help="Allow a positive-ratio suite to receive zero episodes when total is small.",
    )
    parser.set_defaults(ensure_positive_suites=True)
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle selected episodes before assigning new episode indices.",
    )
    parser.add_argument(
        "--compact-task-indices",
        action="store_true",
        help=(
            "Rewrite task_index to a compact 0..N-1 range. By default, the "
            "source LIBERO task indices are preserved so suite ranges stay "
            "long=0..9, goal=10..19, object=20..29, spatial=30..39."
        ),
    )
    parser.add_argument(
        "--output-chunks-size",
        type=int,
        default=1000,
        help="Number of episodes per output data/chunk-* directory.",
    )
    parser.add_argument(
        "--norm-stats",
        choices=("copy", "compute", "skip"),
        default="copy",
        help=(
            "Write root norm_stats.json by copying the source, computing subset "
            "state/actions stats, or skipping it."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        create_subset(args)
    except Exception as exc:
        print(f"[fewshot] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
