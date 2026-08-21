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

"""Run structural, signal, image, and log checks on a LeRobot v2 dataset.

The checker intentionally reads parquet files directly. It therefore does not
need LeRobot or torch and can be run in a lightweight data-audit environment.
It writes a machine-readable JSON report, a Markdown report, per-episode CSV,
and deterministic sample visualizations.

Example:
    python toolkits/lerobot/quality_check_lerobot_dataset.py \
        --dataset-path logs/20260813-09:10:28 \
        --output-dir logs/20260813-09:10:28/data_quality_report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

Severity = Literal["error", "warning", "info"]

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.parquet$")
LOCAL_TIME_RE = re.compile(r"\[(?:INFO|WARNING|ERROR)\s+(\d{2}:\d{2}:\d{2})")
EPOCH_TIME_RE = re.compile(r"\[ERROR\]\s+\[([0-9]{10}(?:\.[0-9]+)?)\]")


@dataclass
class Issue:
    """One quality-check finding."""

    severity: Severity
    code: str
    message: str
    episode_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Thresholds:
    """Configurable quality thresholds."""

    action_limit: float = 1.0
    action_tolerance: float = 1e-4
    action_jump_limit: float = 0.75
    state_jump_robust_ratio: float = 1.5
    timestamp_relative_tolerance: float = 0.02
    black_mean: float = 5.0
    white_mean: float = 250.0
    low_contrast_std: float = 2.0
    blur_gradient: float = 1.0
    frozen_mean_abs_diff: float = 0.1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a local LeRobot parquet dataset and generate sample "
            "visualizations."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help=(
            "LeRobot dataset root, collected_data directory, run directory, "
            "or one episode parquet file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory (default: <dataset-root>/data_quality_report).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Collection log to correlate with episodes (auto-detected by default).",
    )
    parser.add_argument(
        "--no-log-check",
        action="store_true",
        help="Do not inspect the collection log.",
    )
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=4,
        help="Maximum number of episodes to visualize (default: 4).",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=5,
        help="Number of evenly spaced frames per sample episode (default: 5).",
    )
    parser.add_argument(
        "--action-limit",
        type=float,
        default=1.0,
        help="Expected absolute action bound (default: 1.0).",
    )
    parser.add_argument(
        "--action-tolerance",
        type=float,
        default=1e-4,
        help="Numerical tolerance around the action bound (default: 1e-4).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 when the result is REVIEW or FAIL.",
    )
    return parser


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it with `pip install pyarrow`."
        ) from exc
    return pq


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: Pillow. Install it with `pip install Pillow`."
        ) from exc
    return Image


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object at {path}:{line_number}, got {type(row)}"
                )
            rows.append(row)
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve_dataset_root(dataset_path: Path) -> Path:
    path = dataset_path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Dataset path does not exist: {path}")

    if path.is_file():
        if path.suffix != ".parquet":
            raise SystemExit(f"Expected a parquet file, got: {path}")
        for parent in path.parents:
            if (parent / "meta" / "info.json").exists():
                return parent
        raise SystemExit(f"Could not find meta/info.json above {path}")

    if (path / "meta" / "info.json").exists():
        return path

    candidates = sorted(
        info_path.parent.parent
        for info_path in path.rglob("meta/info.json")
        if (info_path.parent.parent / "data").exists()
    )
    if not candidates:
        raise SystemExit(f"No LeRobot dataset found below: {path}")
    if len(candidates) > 1:
        rendered = "\n  ".join(str(candidate) for candidate in candidates)
        raise SystemExit(
            "Multiple LeRobot datasets were found. Point --dataset-path to one "
            f"dataset root:\n  {rendered}"
        )
    return candidates[0]


def _resolve_parquet_files(dataset_root: Path) -> list[Path]:
    files = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        files = sorted(dataset_root.glob("data/**/episode_*.parquet"))
    if not files:
        raise SystemExit(f"No episode parquet files found in {dataset_root / 'data'}")
    return files


def _infer_run_root(dataset_root: Path) -> Path | None:
    for candidate in (dataset_root, *dataset_root.parents):
        if (candidate / "run_embodiment.log").exists() or (
            candidate / "demos" / "metadata.json"
        ).exists():
            return candidate
    return None


def _infer_log_path(dataset_root: Path) -> Path | None:
    run_root = _infer_run_root(dataset_root)
    if run_root is None:
        return None
    candidate = run_root / "run_embodiment.log"
    return candidate if candidate.exists() else None


def _line_time_seconds(line: str) -> int | None:
    match = LOCAL_TIME_RE.search(line)
    if match:
        hour, minute, second = (int(part) for part in match.group(1).split(":"))
        return hour * 3600 + minute * 60 + second
    match = EPOCH_TIME_RE.search(line)
    if match:
        timestamp = datetime.fromtimestamp(float(match.group(1)))
        return timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    return None


def _compact_log_message(line: str, max_length: int = 300) -> str:
    clean = ANSI_ESCAPE_RE.sub("", line).strip()
    clean = re.sub(r"^.*?\)\s*", "", clean)
    return clean if len(clean) <= max_length else clean[: max_length - 3] + "..."


def parse_collection_log(log_path: Path) -> dict[str, Any]:
    """Parse recording attempts and associate saved episodes with log events."""
    successful_attempts: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    global_events: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = re.split(r"[\r\n]+", text)
    for line_number, raw_line in enumerate(lines, start=1):
        line = ANSI_ESCAPE_RE.sub("", raw_line)
        if not line.strip():
            continue
        seconds = _line_time_seconds(line)

        if "[keyboard] start" in line:
            active = {
                "start_seconds": seconds,
                "end_seconds": None,
                "events": [],
            }
            continue

        event: dict[str, Any] | None = None
        if "cartesian_reflex" in line:
            event = {
                "kind": "cartesian_reflex",
                "severity": "error",
                "line_number": line_number,
                "time_seconds": seconds,
                "message": _compact_log_message(line),
            }
        elif "[WARNING" in line:
            event = {
                "kind": "runtime_warning",
                "severity": "warning",
                "line_number": line_number,
                "time_seconds": seconds,
                "message": _compact_log_message(line),
            }
        elif "[ERROR" in line:
            event = {
                "kind": "runtime_error",
                "severity": "error",
                "line_number": line_number,
                "time_seconds": seconds,
                "message": _compact_log_message(line),
            }

        if event is not None:
            global_events.append(event)
            if active is not None:
                active["events"].append(event)

        outcome = None
        if "[keyboard] end_success" in line:
            outcome = "success"
        elif "[keyboard] end_failure" in line:
            outcome = "failure"
        if outcome is not None:
            if active is None:
                active = {
                    "start_seconds": None,
                    "end_seconds": seconds,
                    "events": [],
                }
            active["end_seconds"] = seconds
            active["outcome"] = outcome
            if outcome == "success":
                active["episode_index"] = len(successful_attempts)
                successful_attempts.append(active)
            else:
                failed_attempts.append(active)
            active = None

    return {
        "path": str(log_path),
        "successful_attempts": successful_attempts,
        "failed_attempts": failed_attempts,
        "unfinished_attempt": active,
        "events": global_events,
        "counts": {
            "successful_attempts": len(successful_attempts),
            "failed_attempts": len(failed_attempts),
            "cartesian_reflex": sum(
                event["kind"] == "cartesian_reflex" for event in global_events
            ),
            "warnings": sum(event["severity"] == "warning" for event in global_events),
            "errors": sum(event["severity"] == "error" for event in global_events),
        },
    }


def _feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...] | None:
    feature = info.get("features", {}).get(key, {})
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if not isinstance(shape, list):
        return None
    return tuple(int(value) for value in shape)


def _image_keys(info: dict[str, Any], columns: list[str]) -> list[str]:
    keys = []
    for key, feature in info.get("features", {}).items():
        if isinstance(feature, dict) and feature.get("dtype") == "image":
            keys.append(key)
    if keys:
        return sorted(keys)
    return [key for key in columns if "image" in key.lower()]


def _as_numeric_matrix(values: list[Any], key: str, path: Path) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {key} in {path} to a numeric matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"Expected {key} in {path} to be 2-D, got {matrix.shape}")
    return matrix


def _image_metrics(
    image_structs: list[Any],
    expected_shape: tuple[int, ...] | None,
    thresholds: Thresholds,
    image_cls: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    brightness: list[float] = []
    contrast: list[float] = []
    sharpness: list[float] = []
    frame_change: list[float] = []
    errors: list[dict[str, Any]] = []
    hashes: list[str] = []
    previous: np.ndarray | None = None
    observed_shapes: set[tuple[int, ...]] = set()

    for frame_index, value in enumerate(image_structs):
        if not isinstance(value, dict) or not value.get("bytes"):
            errors.append({"frame_index": frame_index, "reason": "missing image bytes"})
            continue
        raw = value["bytes"]
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        hashes.append(hashlib.sha256(raw).hexdigest())
        try:
            with image_cls.open(io.BytesIO(raw)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        except Exception as exc:
            errors.append(
                {
                    "frame_index": frame_index,
                    "reason": f"decode failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        observed_shapes.add(tuple(int(value) for value in rgb.shape))
        gray = rgb.mean(axis=2)
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        gradient = (
            float(np.abs(np.diff(gray, axis=0)).mean())
            + float(np.abs(np.diff(gray, axis=1)).mean())
        ) / 2.0
        sharpness.append(gradient)
        if previous is not None:
            frame_change.append(float(np.abs(rgb - previous).mean()))
        previous = rgb

    brightness_array = np.asarray(brightness)
    contrast_array = np.asarray(contrast)
    sharpness_array = np.asarray(sharpness)
    change_array = np.asarray(frame_change)
    consecutive_duplicates = sum(
        left == right for left, right in zip(hashes, hashes[1:], strict=False)
    )

    def summary(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"min": None, "median": None, "max": None, "mean": None}
        return {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    metrics = {
        "decoded_frames": len(brightness),
        "observed_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "expected_shape": list(expected_shape) if expected_shape else None,
        "shape_mismatch": bool(
            expected_shape and any(shape != expected_shape for shape in observed_shapes)
        ),
        "brightness": summary(brightness_array),
        "contrast": summary(contrast_array),
        "sharpness": summary(sharpness_array),
        "frame_change": summary(change_array),
        "black_frames": int((brightness_array < thresholds.black_mean).sum()),
        "white_frames": int((brightness_array > thresholds.white_mean).sum()),
        "low_contrast_frames": int(
            (contrast_array < thresholds.low_contrast_std).sum()
        ),
        "blurred_frames": int((sharpness_array < thresholds.blur_gradient).sum()),
        "frozen_transitions": int(
            (change_array < thresholds.frozen_mean_abs_diff).sum()
        ),
        "exact_consecutive_duplicates": int(consecutive_duplicates),
        "unique_encoded_frames": len(set(hashes)),
    }
    return metrics, errors


def _metadata_stats_match(
    episode_stats: dict[str, Any],
    key: str,
    values: np.ndarray,
) -> bool | None:
    stats = episode_stats.get("stats", {}).get(key)
    if not isinstance(stats, dict):
        return None
    expected_count = stats.get("count")
    if isinstance(expected_count, list) and expected_count:
        if int(expected_count[0]) != int(values.shape[0]):
            return False
    for stat_key, actual in (
        ("min", values.min(axis=0)),
        ("max", values.max(axis=0)),
        ("mean", values.mean(axis=0)),
        ("std", values.std(axis=0)),
    ):
        expected = stats.get(stat_key)
        if expected is None:
            return False
        if not np.allclose(
            np.asarray(expected, dtype=np.float64), actual, rtol=1e-5, atol=1e-6
        ):
            return False
    return True


def _analyse_episode(
    parquet_path: Path,
    info: dict[str, Any],
    episode_meta: dict[str, Any] | None,
    episode_stats: dict[str, Any] | None,
    task_indices: set[int],
    thresholds: Thresholds,
    pq: Any,
    image_cls: Any,
) -> tuple[dict[str, Any], list[Issue], np.ndarray, np.ndarray]:
    issues: list[Issue] = []
    table = pq.read_table(parquet_path)
    rows = table.to_pydict()
    columns = table.column_names
    match = EPISODE_FILE_RE.search(parquet_path.name)
    file_episode_index = int(match.group(1)) if match else -1
    frame_count = table.num_rows

    required = {
        "state",
        "actions",
        "done",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    required.update(info.get("features", {}).keys())
    missing_columns = sorted(required - set(columns))
    if missing_columns:
        issues.append(
            Issue(
                "error",
                "missing_columns",
                f"Missing required columns: {', '.join(missing_columns)}",
                file_episode_index,
            )
        )
        return (
            {
                "episode_index": file_episode_index,
                "path": str(parquet_path),
                "frame_count": frame_count,
                "missing_columns": missing_columns,
            },
            issues,
            np.empty((0, 0)),
            np.empty((0, 0)),
        )

    null_counts = {
        key: int(table[key].null_count) for key in columns if table[key].null_count
    }
    if null_counts:
        issues.append(
            Issue(
                "error",
                "null_values",
                f"Top-level null values found: {null_counts}",
                file_episode_index,
            )
        )

    states = _as_numeric_matrix(rows["state"], "state", parquet_path)
    actions = _as_numeric_matrix(rows["actions"], "actions", parquet_path)
    state_shape = _feature_shape(info, "state")
    action_shape = _feature_shape(info, "actions")
    for key, values, expected in (
        ("state", states, state_shape),
        ("actions", actions, action_shape),
    ):
        if expected and tuple(values.shape[1:]) != expected:
            issues.append(
                Issue(
                    "error",
                    "feature_shape_mismatch",
                    f"{key} shape is {values.shape[1:]}, expected {expected}",
                    file_episode_index,
                )
            )
        non_finite = int((~np.isfinite(values)).sum())
        if non_finite:
            issues.append(
                Issue(
                    "error",
                    "non_finite_values",
                    f"{key} contains {non_finite} NaN/Inf values",
                    file_episode_index,
                    {"feature": key, "count": non_finite},
                )
            )

    episode_values = np.asarray(rows["episode_index"], dtype=np.int64)
    frame_values = np.asarray(rows["frame_index"], dtype=np.int64)
    global_indices = np.asarray(rows["index"], dtype=np.int64)
    task_values = np.asarray(rows["task_index"], dtype=np.int64)
    timestamp_values = np.asarray(rows["timestamp"], dtype=np.float64)
    done_values = np.asarray(rows["done"], dtype=bool)

    index_checks = {
        "episode_index_constant": bool(
            np.array_equal(episode_values, np.full(frame_count, file_episode_index))
        ),
        "frame_index_contiguous": bool(
            np.array_equal(frame_values, np.arange(frame_count))
        ),
        "global_index_contiguous_within_episode": bool(
            frame_count <= 1 or np.all(np.diff(global_indices) == 1)
        ),
        "task_indices_known": bool(set(task_values.tolist()) <= task_indices),
        "done_only_on_last_frame": bool(
            frame_count > 0 and done_values[-1] and int(done_values.sum()) == 1
        ),
    }
    for check, passed in index_checks.items():
        if not passed:
            issues.append(
                Issue(
                    "error",
                    check,
                    f"Episode invariant failed: {check}",
                    file_episode_index,
                )
            )

    fps = float(info.get("fps", 0) or 0)
    expected_period = 1.0 / fps if fps > 0 else None
    timestamp_diffs = np.diff(timestamp_values)
    timestamp_ok = bool(
        np.isfinite(timestamp_values).all()
        and (frame_count == 0 or abs(timestamp_values[0]) <= 1e-5)
        and (
            expected_period is None
            or timestamp_diffs.size == 0
            or np.allclose(
                timestamp_diffs,
                expected_period,
                rtol=thresholds.timestamp_relative_tolerance,
                atol=1e-5,
            )
        )
    )
    if not timestamp_ok:
        issues.append(
            Issue(
                "error",
                "timestamp_cadence",
                "Timestamps do not start at zero or do not match the declared FPS",
                file_episode_index,
            )
        )

    if episode_meta and int(episode_meta.get("length", -1)) != frame_count:
        issues.append(
            Issue(
                "error",
                "episode_length_metadata",
                "episodes.jsonl length does not match parquet rows",
                file_episode_index,
                {
                    "metadata": episode_meta.get("length"),
                    "parquet": frame_count,
                },
            )
        )

    stats_match = {
        "state": _metadata_stats_match(episode_stats or {}, "state", states),
        "actions": _metadata_stats_match(episode_stats or {}, "actions", actions),
    }
    if any(value is False for value in stats_match.values()):
        issues.append(
            Issue(
                "warning",
                "stale_episode_stats",
                "episodes_stats.jsonl does not match the parquet values",
                file_episode_index,
                stats_match,
            )
        )

    out_of_bounds = np.abs(actions) > (
        thresholds.action_limit + thresholds.action_tolerance
    )
    action_metrics = {
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "max_abs": float(np.abs(actions).max()),
        "out_of_bounds_values": int(out_of_bounds.sum()),
        "out_of_bounds_frames": int(out_of_bounds.any(axis=1).sum()),
        "out_of_bounds_by_dimension": out_of_bounds.sum(axis=0).tolist(),
        "out_of_bounds_frame_indices": np.flatnonzero(
            out_of_bounds.any(axis=1)
        ).tolist(),
        "max_abs_step": (
            np.abs(np.diff(actions, axis=0)).max(axis=0).tolist()
            if frame_count > 1
            else [0.0] * actions.shape[1]
        ),
    }
    state_metrics = {
        "min": states.min(axis=0).tolist(),
        "max": states.max(axis=0).tolist(),
        "mean": states.mean(axis=0).tolist(),
        "std": states.std(axis=0).tolist(),
        "max_abs_step": (
            np.abs(np.diff(states, axis=0)).max(axis=0).tolist()
            if frame_count > 1
            else [0.0] * states.shape[1]
        ),
    }

    per_image_metrics = {}
    for key in _image_keys(info, columns):
        metrics, image_errors = _image_metrics(
            rows[key], _feature_shape(info, key), thresholds, image_cls
        )
        per_image_metrics[key] = metrics
        if image_errors:
            issues.append(
                Issue(
                    "error",
                    "image_decode",
                    f"{key} has {len(image_errors)} missing or corrupt frames",
                    file_episode_index,
                    {"image_key": key, "examples": image_errors[:10]},
                )
            )
        if metrics["shape_mismatch"]:
            issues.append(
                Issue(
                    "error",
                    "image_shape",
                    f"{key} image shape does not match info.json",
                    file_episode_index,
                    metrics,
                )
            )

    is_success_values = (
        np.asarray(rows["is_success"], dtype=bool)
        if "is_success" in rows
        else np.zeros(frame_count, dtype=bool)
    )
    intervene_values = (
        np.asarray(rows["intervene_flag"], dtype=bool)
        if "intervene_flag" in rows
        else np.zeros(frame_count, dtype=bool)
    )
    duration = float(timestamp_values[-1]) if frame_count else 0.0
    episode_result = {
        "episode_index": file_episode_index,
        "path": str(parquet_path),
        "file_size_bytes": parquet_path.stat().st_size,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "columns": columns,
        "null_counts": null_counts,
        "index_checks": index_checks,
        "timestamp_ok": timestamp_ok,
        "timestamp_period": (
            {
                "min": float(timestamp_diffs.min()),
                "median": float(np.median(timestamp_diffs)),
                "max": float(timestamp_diffs.max()),
            }
            if timestamp_diffs.size
            else None
        ),
        "task_indices": sorted(set(task_values.tolist())),
        "success_frames": int(is_success_values.sum()),
        "intervened_frames": int(intervene_values.sum()),
        "stats_metadata_match": stats_match,
        "state": state_metrics,
        "actions": action_metrics,
        "images": per_image_metrics,
        "global_index_start": int(global_indices[0]) if frame_count else None,
        "global_index_end": int(global_indices[-1]) if frame_count else None,
    }
    return episode_result, issues, states, actions


def _check_dataset_metadata(
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> list[Issue]:
    issues: list[Issue] = []
    actual_episode_indices = [episode["episode_index"] for episode in episodes]
    expected_episode_indices = list(range(len(episodes)))
    if actual_episode_indices != expected_episode_indices:
        issues.append(
            Issue(
                "error",
                "episode_index_sequence",
                "Parquet episode indices are not contiguous from zero",
                details={"actual": actual_episode_indices},
            )
        )

    expected_counts = {
        "info.total_episodes": info.get("total_episodes"),
        "episodes.jsonl": len(episode_rows),
        "episodes_stats.jsonl": len(stats_rows),
    }
    if len(set(expected_counts.values())) != 1 or int(
        info.get("total_episodes", -1)
    ) != len(episodes):
        issues.append(
            Issue(
                "error",
                "episode_count_metadata",
                "Episode counts disagree between metadata and parquet files",
                details={**expected_counts, "parquet_files": len(episodes)},
            )
        )

    total_frames = sum(episode["frame_count"] for episode in episodes)
    if int(info.get("total_frames", -1)) != total_frames:
        issues.append(
            Issue(
                "error",
                "frame_count_metadata",
                "info.json total_frames does not match parquet rows",
                details={
                    "info.total_frames": info.get("total_frames"),
                    "parquet_rows": total_frames,
                },
            )
        )

    if int(info.get("total_tasks", -1)) != len(task_rows):
        issues.append(
            Issue(
                "error",
                "task_count_metadata",
                "info.json total_tasks does not match tasks.jsonl",
            )
        )

    global_ranges = [
        (episode["global_index_start"], episode["global_index_end"])
        for episode in episodes
        if episode.get("global_index_start") is not None
    ]
    if global_ranges:
        expected_start = 0
        for start, end in global_ranges:
            if start != expected_start:
                issues.append(
                    Issue(
                        "error",
                        "global_index_sequence",
                        "Global frame indices are not contiguous across episodes",
                        details={"expected_start": expected_start, "actual": start},
                    )
                )
                break
            expected_start = end + 1
    return issues


def _add_temporal_signal_issues(
    episodes: list[dict[str, Any]],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    thresholds: Thresholds,
) -> list[Issue]:
    """Annotate and flag gross state/action discontinuities."""
    if (
        not states
        or not actions
        or len(states) != len(episodes)
        or len(actions) != len(episodes)
    ):
        return []
    all_states = np.concatenate(states, axis=0)
    state_low = np.percentile(all_states, 1, axis=0)
    state_high = np.percentile(all_states, 99, axis=0)
    robust_range = state_high - state_low
    safe_range = np.where(robust_range > 1e-8, robust_range, 1.0)
    state_affected = []
    action_affected = []

    for episode, episode_states in zip(episodes, states, strict=True):
        state_step = (
            np.abs(np.diff(episode_states, axis=0)).max(axis=0)
            if len(episode_states) > 1
            else np.zeros(episode_states.shape[1])
        )
        state_ratio = state_step / safe_range
        episode["state"]["global_robust_range_1_99"] = robust_range.tolist()
        episode["state"]["max_step_over_global_robust_range"] = state_ratio.tolist()
        episode["state"]["max_step_robust_ratio"] = float(state_ratio.max())
        episode["state"]["max_step_robust_ratio_dimension"] = int(state_ratio.argmax())
        if np.any(state_ratio > thresholds.state_jump_robust_ratio):
            state_affected.append(episode["episode_index"])

        action_step = np.asarray(episode["actions"]["max_abs_step"])
        if np.any(action_step > thresholds.action_jump_limit):
            action_affected.append(episode["episode_index"])

    issues = []
    if state_affected:
        issues.append(
            Issue(
                "warning",
                "state_discontinuity",
                (
                    "A one-frame state jump exceeds the configured fraction of "
                    "the dataset's 1st-to-99th percentile range"
                ),
                details={
                    "episodes": state_affected,
                    "threshold": thresholds.state_jump_robust_ratio,
                },
            )
        )
    if action_affected:
        issues.append(
            Issue(
                "warning",
                "action_discontinuity",
                "A one-frame action jump exceeds the configured threshold",
                details={
                    "episodes": action_affected,
                    "threshold": thresholds.action_jump_limit,
                },
            )
        )
    return issues


def _check_cross_format(
    run_root: Path | None, episodes: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[Issue]]:
    if run_root is None:
        return {"available": False}, []
    demos_root = run_root / "demos"
    metadata_path = demos_root / "metadata.json"
    index_path = demos_root / "trajectory_index.json"
    if not metadata_path.exists() or not index_path.exists():
        return {"available": False, "path": str(demos_root)}, []

    metadata = _load_json(metadata_path)
    index = _load_json(index_path)
    trajectory_index = index.get("trajectory_index", {})
    pt_files = sorted(demos_root.glob("trajectory_*.pt"))
    parquet_lengths = {
        str(episode["episode_index"]): episode["frame_count"] for episode in episodes
    }
    pt_lengths = {
        str(key): int(value.get("num_samples", -1))
        for key, value in trajectory_index.items()
    }
    result = {
        "available": True,
        "path": str(demos_root),
        "metadata": metadata,
        "pt_file_count": len(pt_files),
        "pt_index_lengths": pt_lengths,
        "parquet_lengths": parquet_lengths,
        "episode_count_match": (
            int(metadata.get("size", -1)) == len(episodes) == len(pt_files)
        ),
        "total_frames_match": int(metadata.get("total_samples", -1))
        == sum(episode["frame_count"] for episode in episodes),
        "per_episode_lengths_match": pt_lengths == parquet_lengths,
        "all_pt_files_nonempty": all(path.stat().st_size > 0 for path in pt_files),
    }
    issues = []
    if not all(
        result[key]
        for key in (
            "episode_count_match",
            "total_frames_match",
            "per_episode_lengths_match",
            "all_pt_files_nonempty",
        )
    ):
        issues.append(
            Issue(
                "error",
                "cross_format_mismatch",
                "Replay-buffer .pt index and LeRobot parquet metadata disagree",
                details=result,
            )
        )
    return result, issues


def _add_aggregate_signal_issues(
    episodes: list[dict[str, Any]],
    log_analysis: dict[str, Any] | None,
    thresholds: Thresholds,
) -> list[Issue]:
    issues: list[Issue] = []
    out_of_bounds = [
        episode
        for episode in episodes
        if episode.get("actions", {}).get("out_of_bounds_values", 0)
    ]
    if out_of_bounds:
        total_values = sum(
            episode["actions"]["out_of_bounds_values"] for episode in out_of_bounds
        )
        total_frames = sum(
            episode["actions"]["out_of_bounds_frames"] for episode in out_of_bounds
        )
        maximum = max(episode["actions"]["max_abs"] for episode in out_of_bounds)
        dimensions = np.sum(
            [
                episode["actions"]["out_of_bounds_by_dimension"]
                for episode in out_of_bounds
            ],
            axis=0,
        ).astype(int)
        issues.append(
            Issue(
                "warning",
                "action_out_of_bounds",
                (
                    f"{total_values} action values in {total_frames} frames exceed "
                    f"±{thresholds.action_limit:g}; max |action|={maximum:.6f}"
                ),
                details={
                    "episodes": [episode["episode_index"] for episode in out_of_bounds],
                    "value_count": total_values,
                    "frame_count": total_frames,
                    "max_abs": maximum,
                    "value_count_by_dimension": dimensions.tolist(),
                },
            )
        )

    image_problem_keys = (
        "black_frames",
        "white_frames",
        "low_contrast_frames",
        "blurred_frames",
        "frozen_transitions",
        "exact_consecutive_duplicates",
    )
    image_problems: dict[str, int] = {key: 0 for key in image_problem_keys}
    affected_episodes: set[int] = set()
    for episode in episodes:
        for image_metrics in episode.get("images", {}).values():
            for key in image_problem_keys:
                count = int(image_metrics.get(key, 0))
                image_problems[key] += count
                if count:
                    affected_episodes.add(episode["episode_index"])
    if any(image_problems.values()):
        issues.append(
            Issue(
                "warning",
                "image_quality_threshold",
                "One or more frames crossed an image quality threshold",
                details={
                    **image_problems,
                    "episodes": sorted(affected_episodes),
                },
            )
        )

    if log_analysis:
        attempts = log_analysis["successful_attempts"]
        for attempt in attempts:
            episode_index = int(attempt["episode_index"])
            reflexes = [
                event
                for event in attempt["events"]
                if event["kind"] == "cartesian_reflex"
            ]
            if reflexes:
                issues.append(
                    Issue(
                        "warning",
                        "cartesian_reflex_during_saved_episode",
                        (
                            f"Collection log contains {len(reflexes)} cartesian "
                            "reflex event(s) during this saved episode"
                        ),
                        episode_index,
                        {"events": reflexes},
                    )
                )

        network_warnings = [
            event
            for event in log_analysis["events"]
            if "USB Ethernet" in event["message"] or "pause frames" in event["message"]
        ]
        if network_warnings:
            issues.append(
                Issue(
                    "warning",
                    "robot_network_warning",
                    (
                        "Collection log reports a robot-network reliability warning; "
                        "review the interface before future collection"
                    ),
                    details={"events": network_warnings},
                )
            )
    return issues


def summarize_status(issues: list[Issue]) -> str:
    """Return FAIL for integrity errors, REVIEW for warnings, otherwise PASS."""
    if any(issue.severity == "error" for issue in issues):
        return "FAIL"
    if any(issue.severity == "warning" for issue in issues):
        return "REVIEW"
    return "PASS"


def _select_sample_episode_indices(
    episode_indices: list[int], flagged_indices: list[int], count: int
) -> list[int]:
    if count <= 0 or not episode_indices:
        return []
    count = min(count, len(episode_indices))
    selected: list[int] = []
    for value in flagged_indices:
        if value in episode_indices and value not in selected:
            selected.append(value)
            if len(selected) == count:
                return sorted(selected)
    positions = [0, len(episode_indices) - 1]
    positions.extend(
        np.linspace(0, len(episode_indices) - 1, count, dtype=int).tolist()
    )
    for position in positions:
        value = episode_indices[int(position)]
        if value not in selected:
            selected.append(value)
        if len(selected) == count:
            break
    for value in episode_indices:
        if len(selected) == count:
            break
        if value not in selected:
            selected.append(value)
    return sorted(selected)


def _episode_issue_counts(issues: list[Issue]) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {}
    for issue in issues:
        affected = []
        if issue.episode_index is not None:
            affected.append(issue.episode_index)
        detail_episodes = issue.details.get("episodes")
        if isinstance(detail_episodes, list):
            affected.extend(
                int(value) for value in detail_episodes if isinstance(value, int)
            )
        for episode_index in set(affected):
            bucket = counts.setdefault(episode_index, {"error": 0, "warning": 0})
            if issue.severity in bucket:
                bucket[issue.severity] += 1
    return counts


def _plot_overview(
    episodes: list[dict[str, Any]],
    issues: list[Issue],
    thresholds: Thresholds,
    output_path: Path,
    plt: Any,
) -> None:
    indices = [episode["episode_index"] for episode in episodes]
    issue_counts = _episode_issue_counts(issues)
    colors = []
    for index in indices:
        count = issue_counts.get(index, {})
        if count.get("error"):
            colors.append("#d62728")
        elif count.get("warning"):
            colors.append("#ffbf00")
        else:
            colors.append("#4c78a8")

    lengths = [episode["frame_count"] for episode in episodes]
    max_actions = [episode["actions"]["max_abs"] for episode in episodes]
    action_oob = [episode["actions"]["out_of_bounds_values"] for episode in episodes]
    image_brightness = []
    image_contrast = []
    for episode in episodes:
        first_image = next(iter(episode["images"].values()), {})
        image_brightness.append(first_image.get("brightness", {}).get("mean", np.nan))
        image_contrast.append(first_image.get("contrast", {}).get("mean", np.nan))

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].bar(indices, lengths, color=colors)
    axes[0, 0].axhline(np.mean(lengths), color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Episode length", xlabel="Episode", ylabel="Frames")

    axes[0, 1].plot(indices, max_actions, marker="o", color="#e45756")
    axes[0, 1].axhline(
        thresholds.action_limit, color="black", linestyle="--", label="bound"
    )
    axes[0, 1].set(
        title="Maximum absolute action", xlabel="Episode", ylabel="max |action|"
    )
    axes[0, 1].legend()

    axes[1, 0].bar(indices, action_oob, color="#f58518")
    axes[1, 0].set(
        title="Action values outside expected bound",
        xlabel="Episode",
        ylabel="Value count",
    )

    axes[1, 1].plot(indices, image_brightness, marker="o", label="brightness mean")
    axes[1, 1].plot(indices, image_contrast, marker="s", label="contrast mean")
    axes[1, 1].set(
        title="Image luminance statistics", xlabel="Episode", ylabel="8-bit value"
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xticks(indices)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    low = np.percentile(values, 1, axis=0)
    high = np.percentile(values, 99, axis=0)
    scale = np.where(high > low, high - low, 1.0)
    normalized = 2.0 * (values - low) / scale - 1.0
    return np.clip(normalized, -1.5, 1.5)


def _plot_signal_heatmap(
    episodes: list[dict[str, Any]],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    output_path: Path,
    plt: Any,
) -> None:
    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0)
    normalized_states = _robust_normalize(all_states)
    boundaries = np.cumsum([episode["frame_count"] for episode in episodes])[:-1]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
        constrained_layout=True,
    )
    state_image = axes[0].imshow(
        normalized_states.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
    )
    axes[0].set(title="State (robust normalized)", ylabel="State dimension")
    figure.colorbar(state_image, ax=axes[0], fraction=0.02, pad=0.01)
    action_image = axes[1].imshow(
        all_actions.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.1,
        vmax=1.1,
    )
    axes[1].set(title="Action", xlabel="Global frame", ylabel="Action dimension")
    figure.colorbar(action_image, ax=axes[1], fraction=0.02, pad=0.01)
    for boundary in boundaries:
        for axis in axes:
            axis.axvline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _read_image_array(value: Any, image_cls: Any) -> np.ndarray:
    raw = value["bytes"]
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    with image_cls.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"))


def _plot_episode_sample(
    episode: dict[str, Any],
    parquet_path: Path,
    info: dict[str, Any],
    output_dir: Path,
    sample_frames: int,
    pq: Any,
    image_cls: Any,
    plt: Any,
) -> dict[str, Any]:
    table = pq.read_table(parquet_path)
    rows = table.to_pydict()
    frame_count = table.num_rows
    frame_indices = sorted(
        set(np.linspace(0, frame_count - 1, min(sample_frames, frame_count), dtype=int))
    )
    image_keys = _image_keys(info, table.column_names)
    contact_path = output_dir / f"episode_{episode['episode_index']:06d}_frames.png"
    signal_path = output_dir / f"episode_{episode['episode_index']:06d}_signals.png"

    figure, axes = plt.subplots(
        len(image_keys),
        len(frame_indices),
        figsize=(3 * len(frame_indices), 3 * len(image_keys)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, image_key in enumerate(image_keys):
        for column_index, frame_index in enumerate(frame_indices):
            axis = axes[row_index, column_index]
            axis.imshow(_read_image_array(rows[image_key][frame_index], image_cls))
            timestamp = float(rows["timestamp"][frame_index])
            axis.set_title(f"f={frame_index}, t={timestamp:.1f}s", fontsize=9)
            if column_index == 0:
                axis.set_ylabel(image_key)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"Episode {episode['episode_index']} sampled frames")
    figure.savefig(contact_path, dpi=180)
    plt.close(figure)

    states = np.asarray(rows["state"], dtype=np.float64)
    actions = np.asarray(rows["actions"], dtype=np.float64)
    timestamps = np.asarray(rows["timestamp"], dtype=np.float64)
    normalized_states = _robust_normalize(states)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.5, 1]},
        constrained_layout=True,
    )
    axes[0].imshow(
        normalized_states.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
        extent=[timestamps[0], timestamps[-1], states.shape[1] - 0.5, -0.5],
    )
    axes[0].set(title="State (within-episode robust normalized)", ylabel="Dimension")
    for dimension in range(actions.shape[1]):
        axes[1].plot(timestamps, actions[:, dimension], label=f"a{dimension}")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].axhline(-1.0, color="black", linestyle="--", linewidth=0.7)
    axes[1].set(title="Actions", ylabel="Value")
    axes[1].legend(ncol=min(actions.shape[1], 7), fontsize=8)

    image_key = image_keys[0]
    brightness = []
    frame_change = [np.nan]
    previous = None
    for value in rows[image_key]:
        image = _read_image_array(value, image_cls).astype(np.float32)
        brightness.append(float(image.mean()))
        if previous is not None:
            frame_change.append(float(np.abs(image - previous).mean()))
        previous = image
    axes[2].plot(timestamps, brightness, label="brightness")
    change_axis = axes[2].twinx()
    change_axis.plot(timestamps, frame_change, color="#e45756", label="frame change")
    axes[2].set(title=f"Image signal ({image_key})", xlabel="Time (s)", ylabel="Mean")
    change_axis.set_ylabel("Mean absolute frame difference")
    axes[2].grid(alpha=0.2)
    figure.suptitle(f"Episode {episode['episode_index']} signal overview")
    figure.savefig(signal_path, dpi=180)
    plt.close(figure)
    return {
        "episode_index": episode["episode_index"],
        "frame_indices": frame_indices,
        "contact_sheet": str(contact_path),
        "signals": str(signal_path),
    }


def _write_episode_csv(
    output_path: Path, episodes: list[dict[str, Any]], issues: list[Issue]
) -> None:
    counts = _episode_issue_counts(issues)
    fields = [
        "episode_index",
        "frame_count",
        "duration_seconds",
        "action_max_abs",
        "action_out_of_bounds_values",
        "action_out_of_bounds_frames",
        "image_brightness_mean",
        "image_contrast_mean",
        "image_frame_change_median",
        "error_count",
        "warning_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            image = next(iter(episode["images"].values()), {})
            issue_count = counts.get(episode["episode_index"], {})
            writer.writerow(
                {
                    "episode_index": episode["episode_index"],
                    "frame_count": episode["frame_count"],
                    "duration_seconds": f"{episode['duration_seconds']:.3f}",
                    "action_max_abs": f"{episode['actions']['max_abs']:.6f}",
                    "action_out_of_bounds_values": episode["actions"][
                        "out_of_bounds_values"
                    ],
                    "action_out_of_bounds_frames": episode["actions"][
                        "out_of_bounds_frames"
                    ],
                    "image_brightness_mean": image.get("brightness", {}).get("mean"),
                    "image_contrast_mean": image.get("contrast", {}).get("mean"),
                    "image_frame_change_median": image.get("frame_change", {}).get(
                        "median"
                    ),
                    "error_count": issue_count.get("error", 0),
                    "warning_count": issue_count.get("warning", 0),
                }
            )


def _markdown_issue(issue: Issue) -> str:
    episode = (
        f"（episode {issue.episode_index}）" if issue.episode_index is not None else ""
    )
    return f"- `{issue.severity.upper()}` `{issue.code}`{episode}：{issue.message}"


def _write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    issues = [Issue(**issue) for issue in report["issues"]]
    status_text = {
        "PASS": "通过：未发现超过阈值的问题。",
        "REVIEW": "需复核：结构完整，但存在需要人工确认的告警。",
        "FAIL": "未通过：存在结构或数据完整性错误。",
    }[report["status"]]
    lines = [
        "# LeRobot 数据质检报告",
        "",
        f"**结论：{report['status']} — {status_text}**",
        "",
        f"- 数据集：`{report['dataset_root']}`",
        f"- Episode：{summary['episodes']} 条",
        f"- 总帧数：{summary['frames']} 帧",
        f"- 时长：{summary['duration_seconds']:.1f} 秒（按 episode 求和）",
        f"- 采样频率：{summary['fps']:g} Hz",
        f"- 任务：{', '.join(summary['tasks']) or '未记录'}",
        f"- 图像字段：{', '.join(summary['image_keys']) or '无'}",
        "",
        "## 关键发现",
        "",
    ]
    actionable = [issue for issue in issues if issue.severity != "info"]
    if actionable:
        lines.extend(_markdown_issue(issue) for issue in actionable)
    else:
        lines.append("- 未发现错误或告警。")

    log_summary = report.get("collection_log")
    if log_summary:
        counts = log_summary["counts"]
        attempts = counts["successful_attempts"] + counts["failed_attempts"]
        success_rate = (
            100.0 * counts["successful_attempts"] / attempts if attempts else 0.0
        )
        lines.extend(
            [
                "",
                "## 采集日志关联",
                "",
                f"- 采集尝试：{attempts} 次；成功 {counts['successful_attempts']} 次，"
                f"失败并丢弃 {counts['failed_attempts']} 次，成功率 {success_rate:.1f}% 。",
                f"- `cartesian_reflex`：共 {counts['cartesian_reflex']} 次。",
            ]
        )
        attached = [
            issue
            for issue in issues
            if issue.code == "cartesian_reflex_during_saved_episode"
        ]
        if attached:
            episode_list = ", ".join(str(issue.episode_index) for issue in attached)
            lines.append(f"- 其中保存数据内受影响的 episode：{episode_list}。")
        discarded_reflexes = counts["cartesian_reflex"] - len(attached)
        if discarded_reflexes:
            lines.append(
                f"- 其余 {discarded_reflexes} 次出现在失败且已丢弃的采集尝试中。"
            )

    counts = _episode_issue_counts(issues)
    lines.extend(
        [
            "",
            "## Episode 明细",
            "",
            "| Episode | 帧数 | 时长(s) | max|action| | 越界值 | 图像均值 | "
            "帧差中位数 | 结果 |",
            "|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
    )
    for episode in report["episodes"]:
        image = next(iter(episode["images"].values()), {})
        episode_counts = counts.get(episode["episode_index"], {})
        if episode_counts.get("error"):
            result = "FAIL"
        elif episode_counts.get("warning"):
            result = "REVIEW"
        else:
            result = "PASS"
        lines.append(
            f"| {episode['episode_index']} | {episode['frame_count']} | "
            f"{episode['duration_seconds']:.1f} | "
            f"{episode['actions']['max_abs']:.4f} | "
            f"{episode['actions']['out_of_bounds_values']} | "
            f"{image.get('brightness', {}).get('mean', float('nan')):.2f} | "
            f"{image.get('frame_change', {}).get('median', float('nan')):.3f} | "
            f"{result} |"
        )

    lines.extend(
        [
            "",
            "## 已完成的自动检查",
            "",
            "- Parquet 可读性、必需字段、空值、NaN/Inf 与字段 shape。",
            "- info/episodes/tasks/stats 元数据、全局帧索引及双格式轨迹长度一致性。",
            "- 时间戳起点、声明 FPS、frame/episode/task 索引与末帧 done 语义。",
            "- 动作范围、逐维统计，以及状态/动作的相邻帧跳变。",
            "- 全量图像解码、尺寸、黑/白屏、低对比、模糊、冻结与连续重复帧。",
            "- 采集日志中的失败尝试、硬件 reflex 与网络告警，并关联到保存 episode。",
            "",
            "## 可视化",
            "",
            "- [数据概览](overview.png)",
            "- [全量状态/动作热图](signal_heatmap.png)",
        ]
    )
    for sample in report["samples"]:
        episode_index = sample["episode_index"]
        contact = Path(sample["contact_sheet"]).relative_to(output_path.parent)
        signals = Path(sample["signals"]).relative_to(output_path.parent)
        lines.append(
            f"- Episode {episode_index}：[关键帧]({contact.as_posix()}) · "
            f"[信号]({signals.as_posix()})"
        )

    threshold = report["thresholds"]
    lines.extend(
        [
            "",
            "## 阈值说明",
            "",
            f"- 动作范围：±{threshold['action_limit']}，容差 "
            f"{threshold['action_tolerance']}。",
            f"- 动作单帧跳变阈值：>{threshold['action_jump_limit']}。",
            f"- 状态单帧跳变阈值：>全量 1%–99% 稳健范围的 "
            f"{threshold['state_jump_robust_ratio']} 倍。",
            f"- 黑/白屏均值阈值：<{threshold['black_mean']} / "
            f">{threshold['white_mean']}（8-bit RGB）。",
            f"- 低对比阈值：灰度标准差 <{threshold['low_contrast_std']}。",
            f"- 模糊阈值：平均一阶梯度 <{threshold['blur_gradient']}。",
            f"- 冻结阈值：相邻帧平均绝对差 <" f"{threshold['frozen_mean_abs_diff']}。",
            "",
            "自动质检不能替代任务语义复核；请重点查看接触过程、插入是否完整、"
            "动作是否平顺，以及告警 episode 的关键帧/信号图。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_quality_check(
    dataset_path: Path,
    output_dir: Path | None = None,
    log_path: Path | None = None,
    check_log: bool = True,
    sample_episodes: int = 4,
    sample_frames: int = 5,
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """Run the complete audit and return the generated report dictionary."""
    thresholds = thresholds or Thresholds()
    dataset_root = _resolve_dataset_root(dataset_path)
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else dataset_root / "data_quality_report"
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SystemExit(
            f"Output directory is not writable: {output_dir}. Choose a writable "
            "path with --output-dir."
        ) from exc
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    plot_cache = Path(tempfile.gettempdir()) / "rlinf-quality-check-cache"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache / "xdg"))

    pq = _require_pyarrow()
    image_cls = _require_pillow()
    plt = _require_matplotlib()
    info = _load_json(dataset_root / "meta" / "info.json")
    episode_rows = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    stats_rows = _load_jsonl(dataset_root / "meta" / "episodes_stats.jsonl")
    task_rows = _load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    parquet_files = _resolve_parquet_files(dataset_root)
    episode_meta_map = {
        int(row["episode_index"]): row for row in episode_rows if "episode_index" in row
    }
    episode_stats_map = {
        int(row["episode_index"]): row for row in stats_rows if "episode_index" in row
    }
    task_indices = {int(row["task_index"]) for row in task_rows if "task_index" in row}

    episodes = []
    all_states = []
    all_actions = []
    issues: list[Issue] = []
    for parquet_path in parquet_files:
        match = EPISODE_FILE_RE.search(parquet_path.name)
        file_index = int(match.group(1)) if match else -1
        episode, episode_issues, states, actions = _analyse_episode(
            parquet_path,
            info,
            episode_meta_map.get(file_index),
            episode_stats_map.get(file_index),
            task_indices,
            thresholds,
            pq,
            image_cls,
        )
        episodes.append(episode)
        issues.extend(episode_issues)
        if states.size and actions.size:
            all_states.append(states)
            all_actions.append(actions)

    issues.extend(
        _check_dataset_metadata(info, episode_rows, stats_rows, task_rows, episodes)
    )
    issues.extend(
        _add_temporal_signal_issues(episodes, all_states, all_actions, thresholds)
    )
    run_root = _infer_run_root(dataset_root)
    cross_format, cross_issues = _check_cross_format(run_root, episodes)
    issues.extend(cross_issues)

    log_analysis = None
    if check_log:
        selected_log = (
            log_path.expanduser().resolve()
            if log_path
            else _infer_log_path(dataset_root)
        )
        if selected_log and selected_log.exists():
            log_analysis = parse_collection_log(selected_log)
            if len(log_analysis["successful_attempts"]) != len(episodes):
                issues.append(
                    Issue(
                        "warning",
                        "log_episode_count",
                        "Successful recording attempts in log do not match saved episodes",
                        details={
                            "log": len(log_analysis["successful_attempts"]),
                            "parquet": len(episodes),
                        },
                    )
                )
        elif log_path is not None:
            issues.append(
                Issue(
                    "warning",
                    "collection_log_missing",
                    f"Requested collection log does not exist: {selected_log}",
                )
            )

    issues.extend(_add_aggregate_signal_issues(episodes, log_analysis, thresholds))
    status = summarize_status(issues)
    flagged_indices = sorted(
        {
            issue.episode_index
            for issue in issues
            if issue.episode_index is not None and issue.severity != "info"
        }
    )
    sample_indices = _select_sample_episode_indices(
        [episode["episode_index"] for episode in episodes],
        flagged_indices,
        sample_episodes,
    )

    _plot_overview(episodes, issues, thresholds, output_dir / "overview.png", plt)
    if all_states and all_actions:
        _plot_signal_heatmap(
            episodes,
            all_states,
            all_actions,
            output_dir / "signal_heatmap.png",
            plt,
        )
    parquet_by_index = {
        episode["episode_index"]: Path(episode["path"]) for episode in episodes
    }
    episode_by_index = {episode["episode_index"]: episode for episode in episodes}
    samples = [
        _plot_episode_sample(
            episode_by_index[index],
            parquet_by_index[index],
            info,
            sample_dir,
            sample_frames,
            pq,
            image_cls,
            plt,
        )
        for index in sample_indices
    ]

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "summary": {
            "episodes": len(episodes),
            "frames": sum(episode["frame_count"] for episode in episodes),
            "duration_seconds": sum(
                episode["duration_seconds"] for episode in episodes
            ),
            "fps": float(info.get("fps", 0) or 0),
            "tasks": [row.get("task", "") for row in task_rows],
            "image_keys": _image_keys(
                info, episodes[0].get("columns", []) if episodes else []
            ),
            "errors": sum(issue.severity == "error" for issue in issues),
            "warnings": sum(issue.severity == "warning" for issue in issues),
        },
        "thresholds": asdict(thresholds),
        "metadata": info,
        "cross_format": cross_format,
        "collection_log": log_analysis,
        "issues": [asdict(issue) for issue in issues],
        "episodes": episodes,
        "samples": samples,
    }
    report = _json_safe(report)
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "sample_manifest.json").write_text(
        json.dumps(_json_safe(samples), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_episode_csv(output_dir / "episode_metrics.csv", episodes, issues)
    _write_markdown_report(report, output_dir / "quality_report.md")
    return report


def main() -> int:
    args = _build_arg_parser().parse_args()
    thresholds = Thresholds(
        action_limit=args.action_limit,
        action_tolerance=args.action_tolerance,
    )
    report = run_quality_check(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        log_path=args.log_file,
        check_log=not args.no_log_check,
        sample_episodes=args.sample_episodes,
        sample_frames=args.sample_frames,
        thresholds=thresholds,
    )
    print(
        f"Quality check: {report['status']} | "
        f"episodes={report['summary']['episodes']} | "
        f"frames={report['summary']['frames']} | "
        f"errors={report['summary']['errors']} | "
        f"warnings={report['summary']['warnings']}"
    )
    print(f"Report: {Path(report['output_dir']) / 'quality_report.md'}")
    if args.strict and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
