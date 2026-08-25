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

"""Helpers for model configuration contracts shared by runtime workers."""

from numbers import Integral
from typing import Any

_MISSING = object()


def _config_value(config: Any, name: str, default: Any = _MISSING) -> Any:
    if hasattr(config, "get"):
        value = config.get(name, _MISSING)
    else:
        value = getattr(config, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ValueError(f"model config requires {name}")
        return default
    return value


def _has_config_field(config: Any, name: str) -> bool:
    if hasattr(config, "keys"):
        return name in config
    return hasattr(config, name)


def resolve_model_action_horizon(model_cfg: Any) -> int:
    """Resolve the environment-facing action horizon for one model config.

    ``flow_policy`` owns the versionless ``action_horizon`` field. Other
    embodied models retain the existing ``num_action_chunks`` contract.

    Args:
        model_cfg: Mapping-like or attribute-based model configuration.

    Returns:
        The positive action horizon.

    Raises:
        ValueError: If the required field is absent, legacy FlowPolicy fields
            remain, or the configured value is not a positive integer.
    """
    model_type = _config_value(model_cfg, "model_type")
    model_type = getattr(model_type, "value", model_type)
    if model_type == "flow_policy":
        if _has_config_field(model_cfg, "num_action_chunks"):
            raise ValueError(
                "flow_policy does not support num_action_chunks; use action_horizon"
            )
        field_name = "action_horizon"
    else:
        field_name = "num_action_chunks"

    horizon = _config_value(model_cfg, field_name)
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
        raise ValueError(f"model.{field_name} must be a positive integer")
    return int(horizon)
