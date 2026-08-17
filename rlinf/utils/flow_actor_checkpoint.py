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

"""Portable actor-only checkpoints for PyTorch Flow-T policies.

RLinf's normal FSDP checkpoint contains the complete policy, including SAC
critic heads. This module selects only actor/observation-encoder tensors and
records a semantic manifest so BC and online SAC cannot silently disagree on
the flow objective or time direction.

The module intentionally does not import OpenPI, JAX, or either Flow actor
implementation. Artifact provenance is carried by the manifest and, when a
live module is supplied, checked by its class name.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

FLOW_ACTOR_CHECKPOINT_FORMAT = "rlinf.flow_actor"
FLOW_ACTOR_CHECKPOINT_VERSION = 1
FLOW_ACTOR_MANIFEST_FILENAME = "flow_actor_manifest.json"
FULL_WEIGHTS_FILENAME = "full_weights.pt"
FLOW_ACTOR_FRAMEWORK = "pytorch"
FLOW_ACTOR_TYPE = "FlowTActor"

RECTIFIED_FLOW_OBJECTIVE = "rectified_flow"
IMPROVED_MEANFLOW_OBJECTIVE = "improved_meanflow"

# These scopes cover the actor side of FlowPolicy and FlowStatePolicy. Critic
# heads are excluded even if a caller supplies a broader custom scope.
DEFAULT_ACTOR_SCOPES = (
    "backbone",
    "encoders",
    "state_proj",
    "mix_proj",
    "flow_actor",
    "action_scale",
    "action_bias",
)
DEFAULT_EXCLUDED_PREFIXES = ("q_head", "value_head")

_SEMANTIC_MANIFEST_FIELDS = (
    "framework",
    "actor_class",
    "objective",
    "field_kind",
    "integration_direction",
    "endpoint_t0",
    "endpoint_t1",
    "time_inputs",
    "time_fusion",
    "action_dim",
    "action_range",
    "action_transform",
    "load_mode",
    "strict_actor",
    "require_manifest",
)

_OBJECTIVE_ENDPOINTS = {
    RECTIFIED_FLOW_OBJECTIVE: {
        "field_kind": "instantaneous_velocity",
        "integration_direction": "t0_to_t1",
        "endpoint_t0": "standard_normal",
        "endpoint_t1": "action",
        "time_inputs": ("t_from",),
        "time_encoding": "single",
        "time_fusion": "none",
    },
    IMPROVED_MEANFLOW_OBJECTIVE: {
        "field_kind": "interval_average",
        "integration_direction": "t1_to_t0",
        "endpoint_t0": "action",
        "endpoint_t1": "standard_normal",
        "time_inputs": ("t_from", "t_to"),
        "time_encoding": "dual",
        # Improved MeanFlow requires a real fusion operation. The concrete
        # implementation name is stored in each artifact and may evolve.
        "time_fusion": None,
    },
}


class FlowActorCheckpointError(ValueError):
    """Raised when a Flow actor checkpoint is invalid or incompatible."""


@dataclass(frozen=True)
class ResolvedFlowActorCheckpoint:
    """Canonical files belonging to a resolved checkpoint."""

    checkpoint_dir: Path
    weights_path: Path
    manifest_path: Path | None


@dataclass(frozen=True)
class FlowActorLoadReport:
    """Summary of a successful actor-scoped load."""

    checkpoint: ResolvedFlowActorCheckpoint
    loaded_keys: tuple[str, ...]
    expected_missing_keys: tuple[str, ...]
    ignored_checkpoint_keys: tuple[str, ...]
    metadata: dict[str, Any] | None


def _normalize_prefixes(
    prefixes: Sequence[str],
    *,
    name: str,
    require_nonempty: bool,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for prefix in prefixes:
        if not isinstance(prefix, str):
            raise TypeError(f"{name} entries must be strings, got {type(prefix)!r}.")
        prefix = prefix.rstrip(".")
        if not prefix:
            raise ValueError(f"{name} entries must not be empty.")
        if prefix not in normalized:
            normalized.append(prefix)
    if require_nonempty and not normalized:
        raise ValueError(f"{name} must contain at least one prefix.")
    return tuple(normalized)


def _matches_prefix(key: str, prefixes: Sequence[str]) -> bool:
    return any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise FlowActorCheckpointError(
            f"{name} must be a positive integer, got {value!r}."
        )
    return value


def build_flow_actor_metadata(
    *,
    objective: str,
    action_dim: int,
    state_dim: int | None = None,
    image_size: Sequence[int] | None = None,
    image_num: int = 0,
    time_fusion: str | None = None,
    action_range: Sequence[float] = (-1.0, 1.0),
    action_transform: str = "tanh_latent",
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical FlowTActor artifact metadata.

    Rectified Flow is fixed to noise ``t=0`` -> action ``t=1``. Improved
    MeanFlow is fixed to noise ``t=1`` -> action ``t=0`` and records separate
    ``t_from``/``t_to`` encodings plus their fusion. The resulting mapping is
    accepted by :func:`export_flow_actor_checkpoint` and can be passed back as
    ``expected_metadata`` during online initialization.
    """
    if objective not in _OBJECTIVE_ENDPOINTS:
        raise FlowActorCheckpointError(
            f"Unsupported Flow BC objective {objective!r}; expected one of "
            f"{sorted(_OBJECTIVE_ENDPOINTS)}."
        )
    action_dim = _require_positive_int(action_dim, "action_dim")
    if action_dim != 7:
        raise FlowActorCheckpointError(
            f"Franka + GELLO FlowTActor artifacts require action_dim=7, got {action_dim}."
        )
    if action_transform != "tanh_latent":
        raise FlowActorCheckpointError(
            "FlowTActor artifacts currently require action_transform='tanh_latent', "
            f"got {action_transform!r}."
        )
    if len(action_range) != 2:
        raise FlowActorCheckpointError(
            "action_range must contain exactly [minimum, maximum]."
        )
    action_min, action_max = (float(value) for value in action_range)
    if not (
        math.isfinite(action_min)
        and math.isfinite(action_max)
        and action_min < action_max
    ):
        raise FlowActorCheckpointError(
            "action_range must contain finite values with minimum < maximum, "
            f"got {list(action_range)!r}."
        )
    if state_dim is not None:
        state_dim = _require_positive_int(state_dim, "state_dim")
    if type(image_num) is not int or image_num < 0:
        raise FlowActorCheckpointError(
            f"image_num must be a non-negative integer, got {image_num!r}."
        )
    normalized_image_size: list[int] | None = None
    if image_size is not None:
        normalized_image_size = [int(dimension) for dimension in image_size]
        if len(normalized_image_size) != 3 or any(
            dimension <= 0 for dimension in normalized_image_size
        ):
            raise FlowActorCheckpointError(
                f"image_size must be positive CHW dimensions, got {image_size!r}."
            )
    if image_num > 0 and normalized_image_size is None:
        raise FlowActorCheckpointError(
            "image_size is required when image_num is greater than zero."
        )
    if image_num == 0 and normalized_image_size is not None:
        raise FlowActorCheckpointError(
            "image_num must be greater than zero when image_size is provided."
        )

    contract = _OBJECTIVE_ENDPOINTS[objective]
    if objective == RECTIFIED_FLOW_OBJECTIVE:
        if time_fusion not in (None, "none"):
            raise FlowActorCheckpointError(
                "Rectified Flow uses one time input and time_fusion='none'."
            )
        resolved_fusion = "none"
    else:
        resolved_fusion = "concat_mlp" if time_fusion is None else time_fusion
        if not isinstance(resolved_fusion, str) or not resolved_fusion.strip():
            raise FlowActorCheckpointError(
                "Improved MeanFlow requires a non-empty dual-time fusion name."
            )
        if resolved_fusion == "none":
            raise FlowActorCheckpointError(
                "Improved MeanFlow requires dual-time fusion; 'none' is invalid."
            )

    metadata: dict[str, Any] = {
        "framework": FLOW_ACTOR_FRAMEWORK,
        "actor_class": FLOW_ACTOR_TYPE,
        # Kept as an explicit compatibility alias for the existing FlowConfig.
        "flow_actor_type": FLOW_ACTOR_TYPE,
        "objective": objective,
        "field_kind": contract["field_kind"],
        "integration_direction": contract["integration_direction"],
        "endpoint_t0": contract["endpoint_t0"],
        "endpoint_t1": contract["endpoint_t1"],
        "time_inputs": list(contract["time_inputs"]),
        "time_encoding": contract["time_encoding"],
        "time_fusion": resolved_fusion,
        "action_dim": action_dim,
        "action_range": [action_min, action_max],
        "action_transform": action_transform,
        "state_dim": state_dim,
        "image_size": normalized_image_size,
        "image_num": image_num,
        "image_layout": "CHW" if image_num else None,
        "image_value_range": "zero_one" if image_num else None,
        "load_mode": "actor_only",
        "strict_actor": True,
        "require_manifest": True,
    }
    if extra_metadata is not None:
        collisions = sorted(set(extra_metadata) & set(metadata))
        if collisions:
            raise FlowActorCheckpointError(
                f"extra_metadata cannot override reserved fields: {collisions}."
            )
        metadata.update(dict(extra_metadata))
    return validate_flow_actor_metadata(metadata)


def validate_flow_actor_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-normalized FlowTActor metadata mapping."""
    if not isinstance(metadata, Mapping):
        raise FlowActorCheckpointError(
            f"Flow actor metadata must be a mapping, got {type(metadata)!r}."
        )
    try:
        normalized = json.loads(json.dumps(dict(metadata)))
    except (TypeError, ValueError) as exc:
        raise FlowActorCheckpointError(
            f"Flow actor metadata must be JSON serializable: {exc}"
        ) from exc

    if normalized.get("framework") != FLOW_ACTOR_FRAMEWORK:
        raise FlowActorCheckpointError(
            "Only PyTorch Flow actor artifacts are supported; metadata.framework "
            f"must be {FLOW_ACTOR_FRAMEWORK!r}."
        )
    if normalized.get("actor_class") != FLOW_ACTOR_TYPE:
        raise FlowActorCheckpointError(
            "Only FlowTActor artifacts are supported; metadata.actor_class "
            f"must be {FLOW_ACTOR_TYPE!r}."
        )
    if normalized.get("flow_actor_type") != FLOW_ACTOR_TYPE:
        raise FlowActorCheckpointError(
            "metadata.flow_actor_type must match actor_class='FlowTActor'."
        )
    if normalized.get("load_mode") != "actor_only":
        raise FlowActorCheckpointError(
            "FlowTActor artifacts require metadata.load_mode='actor_only'."
        )
    if normalized.get("strict_actor") is not True:
        raise FlowActorCheckpointError(
            "FlowTActor artifacts require metadata.strict_actor=true."
        )
    if normalized.get("require_manifest") is not True:
        raise FlowActorCheckpointError(
            "FlowTActor artifacts require metadata.require_manifest=true."
        )

    objective_name = normalized.get("objective")
    if objective_name not in _OBJECTIVE_ENDPOINTS:
        raise FlowActorCheckpointError(
            f"Unsupported metadata.objective={objective_name!r}; expected one "
            f"of {sorted(_OBJECTIVE_ENDPOINTS)}."
        )
    expected = _OBJECTIVE_ENDPOINTS[objective_name]
    for field in (
        "field_kind",
        "integration_direction",
        "endpoint_t0",
        "endpoint_t1",
        "time_inputs",
        "time_encoding",
    ):
        expected_value = (
            list(expected[field]) if field == "time_inputs" else expected[field]
        )
        if normalized.get(field) != expected_value:
            raise FlowActorCheckpointError(
                f"Invalid {objective_name} objective contract: metadata.{field}="
                f"{normalized.get(field)!r}, expected {expected_value!r}."
            )
    fusion = normalized.get("time_fusion")
    if objective_name == RECTIFIED_FLOW_OBJECTIVE and fusion != "none":
        raise FlowActorCheckpointError(
            "Rectified Flow metadata requires time_fusion='none'."
        )
    if objective_name == IMPROVED_MEANFLOW_OBJECTIVE and (
        not isinstance(fusion, str) or not fusion.strip() or fusion == "none"
    ):
        raise FlowActorCheckpointError(
            "Improved MeanFlow metadata requires a non-empty dual-time fusion."
        )

    action_dim = _require_positive_int(normalized.get("action_dim"), "action_dim")
    if action_dim != 7:
        raise FlowActorCheckpointError(
            f"Franka + GELLO FlowTActor artifacts require action_dim=7, got {action_dim}."
        )
    if normalized.get("action_transform") != "tanh_latent":
        raise FlowActorCheckpointError(
            "FlowTActor metadata requires action_transform='tanh_latent'."
        )
    action_range = normalized.get("action_range")
    if not isinstance(action_range, list) or len(action_range) != 2:
        raise FlowActorCheckpointError(
            "FlowTActor metadata requires action_range=[minimum, maximum]."
        )
    action_min, action_max = (float(value) for value in action_range)
    if not (
        math.isfinite(action_min)
        and math.isfinite(action_max)
        and action_min < action_max
    ):
        raise FlowActorCheckpointError(
            "FlowTActor metadata action_range must be finite with minimum < maximum."
        )
    state_dim = normalized.get("state_dim")
    if state_dim is not None:
        _require_positive_int(state_dim, "state_dim")
    image_num = normalized.get("image_num")
    if type(image_num) is not int or image_num < 0:
        raise FlowActorCheckpointError("image_num must be a non-negative integer.")
    image_size = normalized.get("image_size")
    if image_num:
        if (
            not isinstance(image_size, list)
            or len(image_size) != 3
            or any(type(value) is not int or value <= 0 for value in image_size)
        ):
            raise FlowActorCheckpointError(
                "Image FlowTActor metadata requires a positive CHW image_size."
            )
        if normalized.get("image_layout") != "CHW":
            raise FlowActorCheckpointError(
                "Image FlowTActor metadata requires image_layout='CHW'."
            )
        if normalized.get("image_value_range") != "zero_one":
            raise FlowActorCheckpointError(
                "Image FlowTActor metadata requires image_value_range='zero_one'."
            )
    elif any(
        normalized.get(field) is not None
        for field in ("image_size", "image_layout", "image_value_range")
    ):
        raise FlowActorCheckpointError(
            "State-only FlowTActor metadata must not declare image fields."
        )
    return normalized


def build_flow_actor_metadata_from_config(model_cfg: Any) -> dict[str, Any]:
    """Build canonical artifact metadata from a FlowPolicy model config."""
    if str(model_cfg.get("flow_actor_type", "")) != FLOW_ACTOR_TYPE:
        raise FlowActorCheckpointError(
            "Only model.flow_actor_type='FlowTActor' can produce this artifact."
        )
    flow_matching = model_cfg.get("flow_matching", None)
    if flow_matching is None:
        raise FlowActorCheckpointError(
            "model.flow_matching is required to record checkpoint semantics."
        )
    objective = str(flow_matching.get("objective", ""))
    time_fusion = None
    if objective == IMPROVED_MEANFLOW_OBJECTIVE:
        improved = flow_matching.get("improved_meanflow", None)
        time_conditioning = (
            improved.get("time_conditioning", None) if improved is not None else None
        )
        if time_conditioning is None or not time_conditioning.get("fusion", None):
            raise FlowActorCheckpointError(
                "Improved MeanFlow config requires "
                "model.flow_matching.improved_meanflow.time_conditioning.fusion."
            )
        time_fusion = str(time_conditioning.get("fusion"))

    extra_metadata: dict[str, Any] = {}
    implementation = flow_matching.get("implementation", None)
    if implementation is not None:
        extra_metadata["implementation"] = str(implementation)
    return build_flow_actor_metadata(
        objective=objective,
        action_dim=int(model_cfg.get("action_dim", -1)),
        state_dim=int(model_cfg.state_dim),
        image_size=list(model_cfg.image_size),
        image_num=int(model_cfg.get("image_num", 1)),
        time_fusion=time_fusion,
        action_range=list(model_cfg.get("action_scale", [-1.0, 1.0]) or [-1.0, 1.0]),
        action_transform=str(flow_matching.get("action_transform", "")),
        extra_metadata=extra_metadata,
    )


def _checkpoint_candidates(path: Path) -> list[Path]:
    return [
        path / FULL_WEIGHTS_FILENAME,
        path / "model_state_dict" / FULL_WEIGHTS_FILENAME,
        path / "actor" / "model_state_dict" / FULL_WEIGHTS_FILENAME,
    ]


def resolve_flow_actor_checkpoint(
    checkpoint: str | Path,
) -> ResolvedFlowActorCheckpoint:
    """Resolve a portable actor directory or its ``full_weights.pt`` file."""
    path = Path(checkpoint).expanduser().resolve()
    if path.is_file():
        if path.name != FULL_WEIGHTS_FILENAME:
            raise FlowActorCheckpointError(
                f"Expected a file named {FULL_WEIGHTS_FILENAME!r}, got {path}."
            )
        weights_path = path
    elif path.is_dir():
        candidates = [
            candidate
            for candidate in _checkpoint_candidates(path)
            if candidate.is_file()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find {FULL_WEIGHTS_FILENAME} under {path}. Tried: "
                f"{[str(candidate) for candidate in _checkpoint_candidates(path)]}"
            )
        if len(candidates) > 1:
            raise FlowActorCheckpointError(
                f"Ambiguous checkpoint directory {path}; found multiple weight "
                f"files: {[str(candidate) for candidate in candidates]}"
            )
        weights_path = candidates[0]
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    checkpoint_dir = (
        weights_path.parent.parent
        if weights_path.parent.name == "model_state_dict"
        else weights_path.parent
    )
    manifest_candidates = [
        checkpoint_dir / FLOW_ACTOR_MANIFEST_FILENAME,
        weights_path.parent / FLOW_ACTOR_MANIFEST_FILENAME,
    ]
    manifests = list(
        dict.fromkeys(path for path in manifest_candidates if path.is_file())
    )
    if len(manifests) > 1:
        raise FlowActorCheckpointError(
            f"Ambiguous manifests for {weights_path}: "
            f"{[str(candidate) for candidate in manifests]}"
        )
    return ResolvedFlowActorCheckpoint(
        checkpoint_dir=checkpoint_dir,
        weights_path=weights_path,
        manifest_path=manifests[0] if manifests else None,
    )


def _validate_state_dict(
    state_dict: Mapping[str, Any], *, source_name: str
) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping):
        raise FlowActorCheckpointError(
            f"{source_name} must contain a state-dict mapping, got "
            f"{type(state_dict)!r}."
        )
    validated: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise FlowActorCheckpointError(
                f"{source_name} contains a non-string key: {key!r}."
            )
        if not isinstance(value, torch.Tensor):
            raise FlowActorCheckpointError(
                f"{source_name} value for {key!r} must be a tensor, got "
                f"{type(value)!r}."
            )
        validated[key] = value
    return validated


def _load_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    return _validate_state_dict(state_dict, source_name=f"Checkpoint {weights_path}")


def _unwrap_module(module: nn.Module) -> nn.Module:
    for attribute in ("module", "_fsdp_wrapped_module"):
        wrapped = getattr(module, attribute, None)
        if isinstance(wrapped, nn.Module) and wrapped is not module:
            return _unwrap_module(wrapped)
    return module


def _assert_flow_t_actor_module(model: nn.Module, *, name: str) -> None:
    unwrapped = _unwrap_module(model)
    actor = getattr(unwrapped, "flow_actor", None)
    if actor is None and unwrapped.__class__.__name__ == FLOW_ACTOR_TYPE:
        actor = unwrapped
    if not isinstance(actor, nn.Module) or actor.__class__.__name__ != FLOW_ACTOR_TYPE:
        observed = actor.__class__.__name__ if isinstance(actor, nn.Module) else None
        raise FlowActorCheckpointError(
            f"{name} must contain a PyTorch {FLOW_ACTOR_TYPE}; observed "
            f"flow_actor class {observed!r}."
        )


def select_flow_actor_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    actor_scopes: Sequence[str] = DEFAULT_ACTOR_SCOPES,
    excluded_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
) -> dict[str, torch.Tensor]:
    """Select actor/encoder tensors from a complete Flow policy state dict."""
    validated = _validate_state_dict(state_dict, source_name="state_dict")
    scopes = _normalize_prefixes(
        actor_scopes, name="actor_scopes", require_nonempty=True
    )
    exclusions = _normalize_prefixes(
        excluded_prefixes, name="excluded_prefixes", require_nonempty=False
    )
    selected = {
        key: value
        for key, value in validated.items()
        if _matches_prefix(key, scopes) and not _matches_prefix(key, exclusions)
    }
    if not selected:
        raise FlowActorCheckpointError(
            "Checkpoint contains zero tensors in the requested actor scopes "
            f"{list(scopes)} after exclusions {list(exclusions)}."
        )
    return selected


def _coerce_source_state_dict(
    source: str | Path | nn.Module | Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if isinstance(source, (str, Path)):
        resolved = resolve_flow_actor_checkpoint(source)
        return _load_state_dict(resolved.weights_path)
    if isinstance(source, nn.Module):
        _assert_flow_t_actor_module(source, name="source model")
        return _validate_state_dict(source.state_dict(), source_name="source model")
    return _validate_state_dict(source, source_name="source state_dict")


def _tensor_schema(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
        for key, value in sorted(state_dict.items())
    }


def export_flow_actor_checkpoint(
    source: str | Path | nn.Module | Mapping[str, torch.Tensor],
    output: str | Path,
    *,
    metadata: Mapping[str, Any],
    actor_scopes: Sequence[str] = DEFAULT_ACTOR_SCOPES,
    excluded_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
    overwrite: bool = False,
) -> ResolvedFlowActorCheckpoint:
    """Export actor-only weights plus a strict FlowTActor manifest."""
    scopes = _normalize_prefixes(
        actor_scopes, name="actor_scopes", require_nonempty=True
    )
    exclusions = _normalize_prefixes(
        excluded_prefixes, name="excluded_prefixes", require_nonempty=False
    )
    serialized_metadata = validate_flow_actor_metadata(metadata)
    actor_state = select_flow_actor_state_dict(
        _coerce_source_state_dict(source),
        actor_scopes=scopes,
        excluded_prefixes=exclusions,
    )
    actor_state = {
        key: value.detach().cpu() for key, value in sorted(actor_state.items())
    }

    output_path = Path(output).expanduser().resolve()
    if output_path.name == FULL_WEIGHTS_FILENAME:
        weights_path = output_path
        checkpoint_dir = output_path.parent
    else:
        if output_path.exists() and not output_path.is_dir():
            raise FileExistsError(f"Output is not a directory: {output_path}")
        checkpoint_dir = output_path
        weights_path = checkpoint_dir / "model_state_dict" / FULL_WEIGHTS_FILENAME
    manifest_path = checkpoint_dir / FLOW_ACTOR_MANIFEST_FILENAME
    existing = [path for path in (weights_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Flow actor checkpoint output already exists: "
            f"{[str(path) for path in existing]}. Pass overwrite=True to replace it."
        )

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor_state, weights_path)
    manifest = {
        "format": FLOW_ACTOR_CHECKPOINT_FORMAT,
        "version": FLOW_ACTOR_CHECKPOINT_VERSION,
        "weights_file": str(weights_path.relative_to(checkpoint_dir)),
        "actor_scopes": list(scopes),
        "excluded_prefixes": list(exclusions),
        "num_tensors": len(actor_state),
        "tensors": _tensor_schema(actor_state),
        "metadata": serialized_metadata,
        **{field: serialized_metadata[field] for field in _SEMANTIC_MANIFEST_FIELDS},
    }
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    return ResolvedFlowActorCheckpoint(
        checkpoint_dir=checkpoint_dir,
        weights_path=weights_path,
        manifest_path=manifest_path,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (json.JSONDecodeError, OSError) as exc:
        raise FlowActorCheckpointError(
            f"Could not read Flow actor manifest {path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise FlowActorCheckpointError(
            f"Flow actor manifest {path} must contain a JSON object."
        )
    return manifest


def _validate_manifest(
    manifest: Mapping[str, Any],
    resolved: ResolvedFlowActorCheckpoint,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    required = {
        "format",
        "version",
        "weights_file",
        "actor_scopes",
        "excluded_prefixes",
        "num_tensors",
        "tensors",
        "metadata",
        *_SEMANTIC_MANIFEST_FIELDS,
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise FlowActorCheckpointError(
            f"Manifest {resolved.manifest_path} is missing required fields: {missing}."
        )
    if manifest["format"] != FLOW_ACTOR_CHECKPOINT_FORMAT:
        raise FlowActorCheckpointError(
            f"Unsupported checkpoint format {manifest['format']!r}; expected "
            f"{FLOW_ACTOR_CHECKPOINT_FORMAT!r}."
        )
    if type(manifest["version"]) is not int or (
        manifest["version"] != FLOW_ACTOR_CHECKPOINT_VERSION
    ):
        raise FlowActorCheckpointError(
            f"Unsupported Flow actor checkpoint version {manifest['version']!r}; "
            f"expected {FLOW_ACTOR_CHECKPOINT_VERSION}."
        )
    relative_weights = manifest["weights_file"]
    if not isinstance(relative_weights, str) or Path(relative_weights).is_absolute():
        raise FlowActorCheckpointError(
            "Manifest weights_file must be a relative string path."
        )
    manifest_weights_path = (resolved.manifest_path.parent / relative_weights).resolve()
    if manifest_weights_path != resolved.weights_path:
        raise FlowActorCheckpointError(
            f"Manifest points to {manifest_weights_path}, but resolved weights are "
            f"{resolved.weights_path}."
        )
    if not isinstance(manifest["actor_scopes"], list):
        raise FlowActorCheckpointError("Manifest actor_scopes must be a list.")
    if not isinstance(manifest["excluded_prefixes"], list):
        raise FlowActorCheckpointError("Manifest excluded_prefixes must be a list.")
    scopes = _normalize_prefixes(
        manifest["actor_scopes"],
        name="manifest actor_scopes",
        require_nonempty=True,
    )
    exclusions = _normalize_prefixes(
        manifest["excluded_prefixes"],
        name="manifest excluded_prefixes",
        require_nonempty=False,
    )

    schema = manifest["tensors"]
    if type(manifest["num_tensors"]) is not int or manifest["num_tensors"] <= 0:
        raise FlowActorCheckpointError("Manifest num_tensors must be positive.")
    if not isinstance(schema, dict):
        raise FlowActorCheckpointError("Manifest tensors must be an object.")
    if manifest["num_tensors"] != len(state_dict):
        raise FlowActorCheckpointError(
            "Manifest tensor count does not match checkpoint: "
            f"manifest={manifest['num_tensors']} checkpoint={len(state_dict)}."
        )
    if set(schema) != set(state_dict):
        raise FlowActorCheckpointError(
            "Manifest tensor keys do not match checkpoint: "
            f"only_manifest={sorted(set(schema) - set(state_dict))[:10]}, "
            f"only_checkpoint={sorted(set(state_dict) - set(schema))[:10]}."
        )
    mismatches: list[str] = []
    for key, tensor in state_dict.items():
        entry = schema[key]
        expected_entry = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        if entry != expected_entry:
            mismatches.append(
                f"{key}: manifest={entry!r}, checkpoint={expected_entry!r}"
            )
    if mismatches:
        raise FlowActorCheckpointError(
            "Manifest tensor schema does not match checkpoint: "
            + "; ".join(mismatches[:10])
        )
    metadata = validate_flow_actor_metadata(manifest["metadata"])
    inconsistent_semantics = [
        field
        for field in _SEMANTIC_MANIFEST_FIELDS
        if manifest[field] != metadata[field]
    ]
    if inconsistent_semantics:
        raise FlowActorCheckpointError(
            "Top-level manifest semantics do not match metadata for fields: "
            f"{inconsistent_semantics}."
        )
    return scopes, exclusions, metadata


def _validate_expected_metadata(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    path: str = "metadata",
) -> None:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        field_path = f"{path}.{key}"
        if key not in actual:
            mismatches.append(f"{field_path}: missing (expected {expected_value!r})")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                mismatches.append(
                    f"{field_path}: checkpoint={actual_value!r}, expected object"
                )
                continue
            try:
                _validate_expected_metadata(
                    actual_value, expected_value, path=field_path
                )
            except FlowActorCheckpointError as exc:
                mismatches.append(str(exc))
        elif actual_value != expected_value:
            mismatches.append(
                f"{field_path}: checkpoint={actual_value!r}, "
                f"expected={expected_value!r}"
            )
    if mismatches:
        raise FlowActorCheckpointError(
            "Flow actor checkpoint metadata mismatch: " + "; ".join(mismatches)
        )


def load_flow_actor_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    load_mode: str = "actor_only",
    strict_actor: bool = True,
    actor_scopes: Sequence[str] | None = None,
    excluded_prefixes: Sequence[str] | None = None,
    expected_missing_prefixes: Sequence[str] = (),
    expected_metadata: Mapping[str, Any] | None = None,
    require_manifest: bool = True,
) -> FlowActorLoadReport:
    """Strictly load actor weights while leaving critic state untouched.

    Pass the online policy's canonical metadata (or a required nested subset)
    through ``expected_metadata`` to prevent loading a checkpoint with another
    objective, direction, action schema, or dual-time fusion.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}.")
    if load_mode != "actor_only":
        raise FlowActorCheckpointError(
            f"Flow actor checkpoints support load_mode='actor_only', got {load_mode!r}."
        )
    if strict_actor is not True:
        raise FlowActorCheckpointError(
            "Flow actor checkpoints require strict_actor=true. Use "
            "expected_missing_prefixes for explicitly initialized online-only heads."
        )
    if require_manifest is not True:
        raise FlowActorCheckpointError(
            "FlowTActor artifacts require require_manifest=true; convert legacy "
            "weights with export_flow_actor_checkpoint first."
        )
    _assert_flow_t_actor_module(model, name="target model")
    resolved = resolve_flow_actor_checkpoint(checkpoint)
    checkpoint_state = _load_state_dict(resolved.weights_path)

    metadata: dict[str, Any] | None = None
    manifest_scopes: tuple[str, ...] | None = None
    manifest_exclusions: tuple[str, ...] | None = None
    if resolved.manifest_path is None:
        raise FlowActorCheckpointError(
            f"{FLOW_ACTOR_MANIFEST_FILENAME} is required for {resolved.weights_path}."
        )
    else:
        manifest = _read_manifest(resolved.manifest_path)
        manifest_scopes, manifest_exclusions, metadata = _validate_manifest(
            manifest, resolved, checkpoint_state
        )
        if expected_metadata is not None:
            _validate_expected_metadata(metadata, expected_metadata)

    scopes = _normalize_prefixes(
        actor_scopes
        if actor_scopes is not None
        else (manifest_scopes or DEFAULT_ACTOR_SCOPES),
        name="actor_scopes",
        require_nonempty=True,
    )
    exclusions = _normalize_prefixes(
        excluded_prefixes
        if excluded_prefixes is not None
        else (manifest_exclusions or DEFAULT_EXCLUDED_PREFIXES),
        name="excluded_prefixes",
        require_nonempty=False,
    )
    if manifest_scopes is not None and scopes != manifest_scopes:
        raise FlowActorCheckpointError(
            "Requested actor_scopes do not match the checkpoint manifest: "
            f"requested={list(scopes)} manifest={list(manifest_scopes)}."
        )
    if manifest_exclusions is not None and exclusions != manifest_exclusions:
        raise FlowActorCheckpointError(
            "Requested excluded_prefixes do not match the checkpoint manifest: "
            f"requested={list(exclusions)} manifest={list(manifest_exclusions)}."
        )

    allowed_missing = _normalize_prefixes(
        expected_missing_prefixes,
        name="expected_missing_prefixes",
        require_nonempty=False,
    )
    scoped_checkpoint = select_flow_actor_state_dict(
        checkpoint_state,
        actor_scopes=scopes,
        excluded_prefixes=exclusions,
    )
    ignored_keys = tuple(sorted(set(checkpoint_state) - set(scoped_checkpoint)))
    target_state = model.state_dict()
    scoped_target = {
        key: value
        for key, value in target_state.items()
        if _matches_prefix(key, scopes) and not _matches_prefix(key, exclusions)
    }
    if not scoped_target:
        raise FlowActorCheckpointError(
            "Target model contains zero tensors in the requested actor scopes "
            f"{list(scopes)}."
        )

    unexpected_keys = sorted(set(scoped_checkpoint) - set(scoped_target))
    shape_mismatches: list[str] = []
    compatible_state: dict[str, torch.Tensor] = {}
    for key in sorted(set(scoped_checkpoint) & set(scoped_target)):
        checkpoint_value = scoped_checkpoint[key]
        target_value = scoped_target[key]
        if tuple(checkpoint_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(checkpoint_value.shape)} "
                f"model={tuple(target_value.shape)}"
            )
        else:
            compatible_state[key] = checkpoint_value

    missing_keys = sorted(set(scoped_target) - set(scoped_checkpoint))
    expected_missing_keys = [
        key for key in missing_keys if _matches_prefix(key, allowed_missing)
    ]
    unexpected_missing_keys = sorted(set(missing_keys) - set(expected_missing_keys))
    problems: list[str] = []
    if not compatible_state:
        problems.append("zero compatible actor tensors")
    if unexpected_keys:
        problems.append(f"unexpected actor keys={unexpected_keys[:20]}")
    if shape_mismatches:
        problems.append(f"shape mismatches={shape_mismatches[:20]}")
    if unexpected_missing_keys:
        problems.append(f"missing actor keys={unexpected_missing_keys[:20]}")
    if problems:
        raise FlowActorCheckpointError(
            f"Actor-scoped checkpoint {resolved.weights_path} is incompatible: "
            + "; ".join(problems)
        )

    # No mutation occurs before all compatibility and semantic checks pass.
    model.load_state_dict(compatible_state, strict=False)
    return FlowActorLoadReport(
        checkpoint=resolved,
        loaded_keys=tuple(sorted(compatible_state)),
        expected_missing_keys=tuple(expected_missing_keys),
        ignored_checkpoint_keys=ignored_keys,
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_ACTOR_SCOPES",
    "DEFAULT_EXCLUDED_PREFIXES",
    "FLOW_ACTOR_CHECKPOINT_FORMAT",
    "FLOW_ACTOR_CHECKPOINT_VERSION",
    "FLOW_ACTOR_FRAMEWORK",
    "FLOW_ACTOR_MANIFEST_FILENAME",
    "FLOW_ACTOR_TYPE",
    "FULL_WEIGHTS_FILENAME",
    "IMPROVED_MEANFLOW_OBJECTIVE",
    "RECTIFIED_FLOW_OBJECTIVE",
    "FlowActorCheckpointError",
    "FlowActorLoadReport",
    "ResolvedFlowActorCheckpoint",
    "build_flow_actor_metadata",
    "build_flow_actor_metadata_from_config",
    "export_flow_actor_checkpoint",
    "load_flow_actor_checkpoint",
    "resolve_flow_actor_checkpoint",
    "select_flow_actor_state_dict",
    "validate_flow_actor_metadata",
]
