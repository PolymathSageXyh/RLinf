import dataclasses
from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("openpi.models.pi0_meanflow")

from rlinf.models.embodiment.base_policy import ForwardType  # noqa: E402
from rlinf.models.embodiment.modules.explore_noise_net import (  # noqa: E402
    ExploreNoiseNet,
)
from rlinf.models.embodiment.openpi import openpi_action_model  # noqa: E402
from rlinf.models.embodiment.openpi.openpi_action_model import (  # noqa: E402
    OpenPi0MeanFlowConfig,
    OpenPi0MeanFlowForRLActionPrediction,
)


def _make_policy(
    noise_method: str,
    *,
    joint_logprob: bool = False,
    noise_level: float = 0.5,
    noise_logvar_range: tuple[float, float] = (0.08, 0.16),
):
    policy = object.__new__(OpenPi0MeanFlowForRLActionPrediction)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_chunk=2,
        action_dim=2,
        action_env_dim=2,
        action_horizon=2,
        add_value_head=False,
        chunk_critic_input=False,
        config_name="pi05_libero_meanflow",
        detach_critic_input=False,
        ignore_last=False,
        joint_logprob=joint_logprob,
        noise_level=noise_level,
        noise_logvar_range=list(noise_logvar_range),
        noise_method=noise_method,
        num_steps=2,
        safe_get_logprob=False,
        train_expert_only=True,
        value_after_vlm=False,
    )
    policy.use_vlm_value = False
    policy.velocity_scale = torch.nn.Parameter(torch.tensor(0.25))
    policy.noise_head = ExploreNoiseNet(
        in_dim=4,
        out_dim=2,
        hidden_dims=[8],
        activation_type="tanh",
        noise_logvar_range=list(noise_logvar_range),
        noise_scheduler_type="learn",
    )

    def get_velocity(
        self,
        state,
        x_t,
        timestep,
        r_timestep,
        prefix_pad_masks,
        past_key_values,
    ):
        del state, timestep, r_timestep, prefix_pad_masks, past_key_values
        velocity = self.velocity_scale * torch.ones_like(x_t)
        suffix_out = torch.cat([x_t, x_t], dim=-1)
        return velocity, suffix_out

    def build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks):
        del images, img_masks, lang_tokens, lang_masks
        return torch.empty(2, 0, 1), None, None

    policy.get_velocity = MethodType(get_velocity, policy)
    policy._build_prefix_cache = MethodType(build_prefix_cache, policy)
    return policy


def test_meanflow_config_uses_noise_method_instead_of_add_noise_head():
    config_fields = {field.name for field in dataclasses.fields(OpenPi0MeanFlowConfig)}
    assert "add_noise_head" not in config_fields
    assert OpenPi0MeanFlowConfig().noise_method == "flow_ode"
    assert OpenPi0MeanFlowConfig().noise_level == 0.5
    assert OpenPi0MeanFlowConfig().joint_logprob is True


def test_noise_head_construction_and_fsdp_registration_follow_noise_method(
    monkeypatch,
):
    def fake_base_init(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.action_out_proj = torch.nn.Linear(4, config.action_dim)

    monkeypatch.setattr(
        openpi_action_model._PI0PytorchMeanflowBase,
        "__init__",
        fake_base_init,
    )

    common_config = {
        "action_dim": 2,
        "add_value_head": False,
        "double_layer": False,
        "joint_logprob": True,
        "noise_logvar_range": [0.08, 0.16],
        "train_expert_only": True,
        "value_after_vlm": False,
    }
    noise_policy = OpenPi0MeanFlowForRLActionPrediction(
        SimpleNamespace(noise_method="flow_noise", **common_config)
    )
    ode_policy = OpenPi0MeanFlowForRLActionPrediction(
        SimpleNamespace(noise_method="flow_ode", **common_config)
    )
    sde_policy = OpenPi0MeanFlowForRLActionPrediction(
        SimpleNamespace(noise_method="flow_sde", noise_level=0.5, **common_config)
    )

    assert hasattr(noise_policy, "noise_head")
    assert "ExploreNoiseNet" in noise_policy._no_split_modules
    assert not hasattr(ode_policy, "noise_head")
    assert "ExploreNoiseNet" not in ode_policy._no_split_modules
    assert not hasattr(sde_policy, "noise_head")
    assert "ExploreNoiseNet" not in sde_policy._no_split_modules


def test_flow_ode_is_deterministic_with_zero_logprob_and_entropy():
    policy = _make_policy("flow_ode")
    x_t = torch.ones(2, 2, 2)
    state = torch.zeros(2, 2)

    x_t_mean, x_t_std, _, _ = policy.sample_mean_var_val(
        x_t, 0, state, None, None, "flow_ode", 2
    )
    torch.testing.assert_close(x_t_std, torch.zeros_like(x_t_std))

    chains = torch.stack([x_t, x_t_mean.detach(), x_t], dim=1)
    denoise_inds = torch.zeros(2, 2, dtype=torch.long)
    log_probs, _, entropy = policy.get_log_prob_value(
        None, None, None, None, state, chains, denoise_inds
    )

    torch.testing.assert_close(log_probs, torch.zeros_like(log_probs))
    torch.testing.assert_close(entropy, torch.zeros_like(entropy))
    assert log_probs.requires_grad


def test_flow_noise_rollout_recompute_matches_and_backpropagates():
    policy = _make_policy("flow_noise")
    x_t = torch.ones(2, 2, 2)
    state = torch.zeros(2, 2)
    epsilon = torch.full_like(x_t, 0.5)

    old_mean, old_std, _, _ = policy.sample_mean_var_val(
        x_t, 0, state, None, None, "flow_noise", 2
    )
    assert torch.all(old_std > 0)
    sampled_next = (old_mean + epsilon * old_std).detach()
    expected_log_probs = policy.get_logprob_norm(
        sampled_next, old_mean.detach(), old_std.detach()
    )

    chains = torch.stack([x_t, sampled_next, x_t], dim=1)
    denoise_inds = torch.zeros(2, 2, dtype=torch.long)
    log_probs, _, entropy = policy.get_log_prob_value(
        None, None, None, None, state, chains, denoise_inds
    )

    torch.testing.assert_close(log_probs[:, 0], expected_log_probs)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    assert torch.count_nonzero(entropy) > 0

    (-log_probs.mean()).backward()
    assert policy.velocity_scale.grad is not None
    assert torch.count_nonzero(policy.velocity_scale.grad) > 0
    noise_grads = [
        parameter.grad
        for parameter in policy.noise_head.parameters()
        if parameter.grad is not None
    ]
    assert noise_grads
    assert any(torch.count_nonzero(grad) > 0 for grad in noise_grads)


def test_flow_sde_matches_reference_formula_and_flow_noise_shapes():
    policy = _make_policy(
        "flow_sde",
        noise_level=0.5,
        noise_logvar_range=(0.01, 2.0),
    )
    x_t = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) / 4.0
    state = torch.zeros(2, 2)
    denoise_inds = torch.tensor([0, 1])

    x_t_mean, x_t_std, _, v_t = policy.sample_mean_var_val(
        x_t, denoise_inds, state, None, None, "flow_sde", 2
    )
    flow_noise_mean, flow_noise_std, _, _ = policy.sample_mean_var_val(
        x_t, denoise_inds, state, None, None, "flow_noise", 2
    )

    timestep = torch.tensor([1.0, 0.5])
    r_timestep = torch.tensor([0.5, 0.0])
    safe_timestep = torch.tensor([0.99, 0.5])
    midpoint = (timestep + r_timestep) / 2.0
    log1p_diff = torch.log1p(-r_timestep) - torch.log1p(-safe_timestep)
    log1p_diff_mid = torch.log1p(-r_timestep) - torch.log1p(-midpoint)
    time_diff = timestep - r_timestep
    noise_level_sq = torch.tensor(0.5).square()

    expected_mean = x_t * (
        1.0 - noise_level_sq / 2.0 * log1p_diff[:, None, None]
    ) - time_diff[:, None, None] * v_t * (
        1.0 + noise_level_sq / 2.0 * (1.0 - log1p_diff_mid[:, None, None])
    )
    expected_std = torch.sqrt(
        (noise_level_sq * (log1p_diff - time_diff)).clamp_min(0.0)
    )[:, None, None].expand_as(x_t)

    assert x_t_mean.shape == flow_noise_mean.shape == x_t.shape
    assert x_t_std.shape == flow_noise_std.shape == x_t.shape
    assert x_t_mean.dtype == x_t.dtype
    assert x_t_std.dtype == x_t.dtype
    torch.testing.assert_close(x_t_mean, expected_mean)
    torch.testing.assert_close(x_t_std, expected_std)
    assert torch.isfinite(x_t_mean).all()
    assert torch.isfinite(x_t_std).all()


def test_flow_sde_std_is_clamped_to_configured_bounds():
    x_t = torch.ones(2, 2, 2)
    state = torch.zeros(2, 2)

    low_policy = _make_policy("flow_sde", noise_level=0.0)
    _, low_std, _, _ = low_policy.sample_mean_var_val(
        x_t, 0, state, None, None, "flow_sde", 2
    )
    torch.testing.assert_close(low_std, torch.full_like(x_t, 0.08))

    high_policy = _make_policy("flow_sde", noise_level=1.0)
    _, high_std, _, _ = high_policy.sample_mean_var_val(
        x_t, 0, state, None, None, "flow_sde", 2
    )
    torch.testing.assert_close(high_std, torch.full_like(x_t, 0.16))


@pytest.mark.parametrize(
    ("noise_level", "noise_range", "error"),
    [
        (-0.1, (0.08, 0.16), "noise_level"),
        (0.5, (0.0, 0.16), "noise_logvar_range"),
        (0.5, (0.2, 0.1), "noise_logvar_range"),
    ],
)
def test_flow_sde_rejects_invalid_noise_config(noise_level, noise_range, error):
    policy = _make_policy(
        "flow_sde",
        noise_level=noise_level,
        noise_logvar_range=noise_range,
    )
    with pytest.raises(ValueError, match=error):
        policy._validate_flow_sde_config()


def test_joint_flow_noise_recomputes_full_chain_and_backpropagates():
    policy = _make_policy("flow_noise", joint_logprob=True)
    state = torch.zeros(2, 2)
    chain_states = [torch.ones(2, 2, 2)]
    expected_log_probs = [
        policy.get_logprob_norm(
            chain_states[0],
            torch.zeros_like(chain_states[0]),
            torch.ones_like(chain_states[0]),
        )
    ]

    for idx in range(policy.config.num_steps):
        mean, std, _, _ = policy.sample_mean_var_val(
            chain_states[-1], idx, state, None, None, "flow_noise", 2
        )
        sampled_next = (mean + 0.5 * std).detach()
        expected_log_probs.append(
            policy.get_logprob_norm(sampled_next, mean.detach(), std.detach())
        )
        chain_states.append(sampled_next)

    chains = torch.stack(chain_states, dim=1)
    denoise_inds = torch.arange(2).repeat(2, 1)
    log_probs, values, entropy = policy.get_log_prob_value(
        None, None, None, None, state, chains, denoise_inds
    )

    expected_log_probs = torch.stack(expected_log_probs, dim=1)
    torch.testing.assert_close(log_probs, expected_log_probs)
    assert log_probs.shape == (2, 3, 2, 2)
    assert values.shape == (2, 2)
    assert entropy.shape == log_probs.shape
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()

    (-log_probs.mean()).backward()
    assert policy.velocity_scale.grad is not None
    assert torch.count_nonzero(policy.velocity_scale.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.noise_head.parameters()
    )


def test_joint_flow_sde_recomputes_full_chain_with_zero_entropy():
    policy = _make_policy(
        "flow_sde",
        joint_logprob=True,
        noise_logvar_range=(0.01, 2.0),
    )
    state = torch.zeros(2, 2)
    chain_states = [torch.ones(2, 2, 2)]
    expected_log_probs = [
        policy.get_logprob_norm(
            chain_states[0],
            torch.zeros_like(chain_states[0]),
            torch.ones_like(chain_states[0]),
        )
    ]

    for idx in range(policy.config.num_steps):
        mean, std, _, _ = policy.sample_mean_var_val(
            chain_states[-1], idx, state, None, None, "flow_sde", 2
        )
        sampled_next = (mean + 0.5 * std).detach()
        expected_log_probs.append(
            policy.get_logprob_norm(sampled_next, mean.detach(), std.detach())
        )
        chain_states.append(sampled_next)

    log_probs, values, entropy = policy.get_log_prob_value(
        None,
        None,
        None,
        None,
        state,
        torch.stack(chain_states, dim=1),
        torch.arange(2).repeat(2, 1),
    )

    torch.testing.assert_close(log_probs, torch.stack(expected_log_probs, dim=1))
    assert log_probs.shape == entropy.shape == (2, 3, 2, 2)
    assert values.shape == (2, 2)
    torch.testing.assert_close(entropy, torch.zeros_like(log_probs))
    assert torch.isfinite(log_probs).all()

    (-log_probs.mean()).backward()
    assert policy.velocity_scale.grad is not None
    assert torch.count_nonzero(policy.velocity_scale.grad) > 0


def test_joint_flow_ode_keeps_only_initial_prior_logprob():
    policy = _make_policy("flow_ode", joint_logprob=True)
    state = torch.zeros(2, 2)
    chain_states = [torch.ones(2, 2, 2)]
    for idx in range(policy.config.num_steps):
        mean, std, _, _ = policy.sample_mean_var_val(
            chain_states[-1], idx, state, None, None, "flow_ode", 2
        )
        torch.testing.assert_close(std, torch.zeros_like(std))
        chain_states.append(mean.detach())

    log_probs, _, entropy = policy.get_log_prob_value(
        None,
        None,
        None,
        None,
        state,
        torch.stack(chain_states, dim=1),
        torch.arange(2).repeat(2, 1),
    )

    expected_initial = policy.get_logprob_norm(
        chain_states[0],
        torch.zeros_like(chain_states[0]),
        torch.ones_like(chain_states[0]),
    )
    torch.testing.assert_close(log_probs[:, 0], expected_initial)
    torch.testing.assert_close(log_probs[:, 1:], torch.zeros_like(log_probs[:, 1:]))
    torch.testing.assert_close(entropy, torch.zeros_like(entropy))


def _install_rollout_stubs(policy, observed_methods):
    def preprocess_observation(self, observation, train):
        del self, train
        return [], [], None, None, observation.state

    def sample_noise(self, shape, device):
        del self
        return torch.ones(shape, device=device)

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        denoise_steps,
        compute_values=True,
    ):
        del (
            self,
            idx,
            state,
            prefix_pad_masks,
            past_key_values,
            denoise_steps,
            compute_values,
        )
        observed_methods.append(sample_method)
        if sample_method in {"flow_noise", "flow_sde"}:
            std = torch.full_like(x_t, 0.25)
        else:
            std = torch.zeros_like(x_t)
        return x_t + 1.0, std, torch.zeros(x_t.shape[0]), torch.zeros_like(x_t)

    policy._preprocess_observation = MethodType(preprocess_observation, policy)
    policy.sample_noise = MethodType(sample_noise, policy)
    policy.sample_mean_var_val = MethodType(sample_mean_var_val, policy)


def test_non_joint_rollout_uses_configured_noise_once_and_eval_is_ode(monkeypatch):
    policy = _make_policy("flow_noise")
    policy.config.num_steps = 3
    observed_methods = []
    _install_rollout_stubs(policy, observed_methods)
    monkeypatch.setattr(openpi_action_model.random, "randint", lambda start, end: 1)
    observation = SimpleNamespace(state=torch.zeros(2, 2))

    train_outputs = policy.sample_actions(observation, mode="train")
    assert observed_methods == ["flow_ode", "flow_noise", "flow_ode"]
    assert torch.equal(
        train_outputs["denoise_inds"], torch.ones(2, 3, dtype=torch.long)
    )
    assert torch.isfinite(train_outputs["prev_logprobs"]).all()

    observed_methods.clear()
    eval_outputs = policy.sample_actions(observation, mode="eval")
    assert observed_methods == ["flow_ode", "flow_ode", "flow_ode"]
    torch.testing.assert_close(
        eval_outputs["prev_logprobs"],
        torch.zeros_like(eval_outputs["prev_logprobs"]),
    )


@pytest.mark.parametrize("noise_method", ["flow_noise", "flow_sde"])
def test_joint_rollout_uses_noise_at_every_step_and_eval_keeps_prior(noise_method):
    policy = _make_policy(noise_method, joint_logprob=True)
    policy.config.num_steps = 3
    observed_methods = []
    _install_rollout_stubs(policy, observed_methods)
    observation = SimpleNamespace(state=torch.zeros(2, 2))

    train_outputs = policy.sample_actions(observation, mode="train")
    assert observed_methods == [noise_method, noise_method, noise_method]
    assert torch.equal(train_outputs["denoise_inds"], torch.arange(3).repeat(2, 1))
    assert train_outputs["chains"].shape == (2, 4, 2, 2)
    assert torch.isfinite(train_outputs["prev_logprobs"]).all()

    observed_methods.clear()
    eval_outputs = policy.sample_actions(observation, mode="eval")
    assert observed_methods == ["flow_ode", "flow_ode", "flow_ode"]
    assert torch.equal(
        eval_outputs["denoise_inds"], -torch.ones(2, 3, dtype=torch.long)
    )
    expected_prior = policy.get_logprob_norm(
        eval_outputs["chains"][:, 0],
        torch.zeros_like(eval_outputs["chains"][:, 0]),
        torch.ones_like(eval_outputs["chains"][:, 0]),
    )
    torch.testing.assert_close(eval_outputs["prev_logprobs"], expected_prior / 4)


def test_invalid_denoise_index_raises():
    policy = _make_policy("flow_noise")
    with pytest.raises(ValueError, match="denoise_inds must be"):
        policy.get_log_prob_value(
            None,
            None,
            None,
            None,
            torch.zeros(2, 2),
            torch.ones(2, 3, 2, 2),
            -torch.ones(2, 2, dtype=torch.long),
        )


def test_forward_dispatches_only_default_and_sft():
    policy = _make_policy("flow_ode")
    policy.default_forward = MethodType(
        lambda self, **kwargs: ("default", kwargs), policy
    )
    policy.sft_forward = MethodType(lambda self, **kwargs: ("sft", kwargs), policy)

    assert policy.forward(ForwardType.DEFAULT, value=1) == ("default", {"value": 1})
    assert policy.forward(ForwardType.SFT, value=2) == ("sft", {"value": 2})
    with pytest.raises(NotImplementedError, match="only ForwardType.SFT"):
        policy.forward(ForwardType.NFT)


def test_sft_forward_returns_scalar_and_rejects_chunk_loss(monkeypatch):
    policy = _make_policy("flow_ode")
    policy.global_step = 7
    policy.gradient_checkpointing_disable = MethodType(lambda self: None, policy)
    captured = {}

    def fake_forward(
        self,
        observation,
        actions,
        global_step=None,
        num_train_steps=None,
    ):
        captured.update(
            observation=observation,
            actions=actions,
            global_step=global_step,
            num_train_steps=num_train_steps,
        )
        return self.velocity_scale * torch.ones(2, device=actions.device)

    monkeypatch.setattr(
        openpi_action_model._PI0PytorchMeanflowBase,
        "forward",
        fake_forward,
    )
    observation = {"state": torch.zeros(2, 2)}
    actions = torch.zeros(2, 2, 2)

    loss = policy.sft_forward(
        (observation, actions),
        num_train_steps=11,
    )
    assert loss.ndim == 0
    assert captured["global_step"] == 7
    assert captured["num_train_steps"] == 11
    assert captured["actions"].dtype == torch.float32

    with pytest.raises(ValueError, match="does not support"):
        policy.sft_forward(
            (observation, actions),
            use_action_chunk_loss=True,
        )


def test_unsupported_meanflow_noise_method_raises():
    policy = _make_policy("invalid")
    with pytest.raises(ValueError, match="flow_ode.*flow_sde.*flow_noise"):
        policy.sample_mean_var_val(
            torch.ones(2, 2, 2),
            0,
            torch.zeros(2, 2),
            None,
            None,
            "invalid",
            2,
        )
