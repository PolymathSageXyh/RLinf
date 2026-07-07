# Copyright 2025 The RLinf Authors.
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
# openpi model configs

import glob
import logging
import os

import torch
from openpi.shared import normalize as _normalize
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
_MEANFLOW_INIT_SCOPES = {"all_compatible", "vlm_only"}


def _load_openpi_state_dict(checkpoint_dir: str):
    import safetensors

    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    if os.path.exists(full_weights_path):
        return torch.load(full_weights_path, map_location="cpu"), full_weights_path
    if os.path.exists(actor_full_weights_path):
        return (
            torch.load(actor_full_weights_path, map_location="cpu"),
            actor_full_weights_path,
        )

    weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

    all_state_dict = {}
    for weight_path in weight_paths:
        state_dict = safetensors.torch.load_file(weight_path, device="cpu")
        all_state_dict.update(state_dict)
    return all_state_dict, ",".join(weight_paths)


def _log_key_sample(label: str, keys: list[str], limit: int = 20) -> None:
    if keys:
        logger.info("%s (%d): %s", label, len(keys), keys[:limit])


def _load_meanflow_state_dict(
    model: torch.nn.Module, state_dict: dict, source: str, init_scope: str
) -> None:
    if init_scope not in _MEANFLOW_INIT_SCOPES:
        raise ValueError(
            "actor.model.openpi.meanflow_init_scope must be one of "
            f"{sorted(_MEANFLOW_INIT_SCOPES)}, got {init_scope!r}."
        )

    target_state = model.state_dict()
    compatible_state = {}
    unexpected_keys = []
    shape_mismatch_keys = []
    skipped_by_scope_keys = []

    for key, value in state_dict.items():
        if init_scope == "vlm_only" and not key.startswith(
            "paligemma_with_expert.paligemma."
        ):
            skipped_by_scope_keys.append(key)
            continue
        if key not in target_state:
            unexpected_keys.append(key)
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            shape_mismatch_keys.append(
                f"{key}: checkpoint={tuple(value.shape)} "
                f"model={tuple(target_state[key].shape)}"
            )
            continue
        compatible_state[key] = value

    missing_keys = [key for key in target_state if key not in compatible_state]
    model.load_state_dict(compatible_state, strict=False)

    logger.info(
        "Loaded OpenPI MeanFlow weights from %s with init_scope=%s: "
        "compatible=%d missing_or_random_init=%d unexpected=%d "
        "shape_mismatch=%d skipped_by_scope=%d",
        source,
        init_scope,
        len(compatible_state),
        len(missing_keys),
        len(unexpected_keys),
        len(shape_mismatch_keys),
        len(skipped_by_scope_keys),
    )
    _log_key_sample("MeanFlow random-init/missing keys", missing_keys)
    _log_key_sample("MeanFlow unexpected checkpoint keys", unexpected_keys)
    _log_key_sample("MeanFlow shape-mismatch checkpoint keys", shape_mismatch_keys)
    _log_key_sample("MeanFlow skipped-by-scope checkpoint keys", skipped_by_scope_keys)


def get_model(cfg: DictConfig, torch_dtype=None):
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    try:
        from openpi.models.pi0_meanflow import Pi0MeanflowConfig
    except ImportError:
        Pi0MeanflowConfig = None

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
        OpenPi0MeanFlowConfig,
        OpenPi0MeanFlowForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    is_meanflow_config = (
        Pi0MeanflowConfig is not None
        and isinstance(actor_model_config, Pi0MeanflowConfig)
    )
    if is_meanflow_config:
        actor_model_config = OpenPi0MeanFlowConfig(**actor_model_config.__dict__)
    else:
        actor_model_config = OpenPi0Config(**actor_model_config.__dict__)

    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    model_state_dict, state_source = _load_openpi_state_dict(checkpoint_dir)

    if isinstance(actor_model_config, OpenPi0MeanFlowConfig):
        model = OpenPi0MeanFlowForRLActionPrediction(actor_model_config)
        if actor_model_config.train_expert_only:
            model.freeze_vlm()
        init_scope = str(actor_model_config.meanflow_init_scope)
        _load_meanflow_state_dict(model, model_state_dict, state_source, init_scope)
        model._rlinf_full_weight_fallback_state_source = state_source
    else:
        model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
            actor_model_config
        )
        if actor_model_config.train_expert_only:
            model.freeze_vlm()
        model.load_state_dict(model_state_dict, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats = getattr(data_config, "norm_stats", None)
    if norm_stats is None:
        # Prefer norm stats saved with RLinf checkpoints, then fall back to the
        # source checkpoint layout used by OpenPI.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        saved_norm_stats_dir = os.path.join(checkpoint_dir, data_config.asset_id)
        saved_norm_stats_path = os.path.join(saved_norm_stats_dir, "norm_stats.json")
        if os.path.exists(saved_norm_stats_path):
            norm_stats = _normalize.load(saved_norm_stats_dir)
        else:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    model._rlinf_norm_stats = norm_stats
    model._rlinf_norm_stats_asset_id = data_config.asset_id
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
