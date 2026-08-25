# Franka + GELLO Flow-T BC 开发与操作指南

本文说明 PyTorch Flow-T 行为克隆的统一实现。`H=1` 与 `H>1` 使用同一数据、模型、
objective、sampler 和 checkpoint 合同；`H` 只改变动作 horizon 与内部 flow state 宽度。

## 1. 实现边界

- 训练与静态评估支持 Rectified Flow（RF）和 Improved MeanFlow（iMF）。
- 静态采样支持 `flow_ode` 和 `flow_sde`，默认使用配置中的 evaluation sampler。
- online SACFlow 暂时只支持 `action_horizon: 1`。`H>1` 静态 SDE 可用并不代表
  rollout、replay、discount 或真机 chunk execution 已支持。
- 只支持 `flow_matching.implementation: pytorch_flow_t`。旧配置字段和旧 portable
  manifest 会直接失败，不提供转换或跨 horizon warm-start。

核心代码入口：

| 模块 | 职责 |
| --- | --- |
| `rlinf/utils/flow_bc_contract.py` | 解析唯一的 `FlowBCSpec`，校验配置与 sampler |
| `rlinf/data/datasets/flow/franka_gello.py` | 构造同 episode action chunk、mask 和零 padding |
| `rlinf/models/embodiment/modules/flow_objectives.py` | RF/iMF target、JVP 与 masked loss |
| `rlinf/models/embodiment/modules/flow_actor.py` | 联合 field 网络和 ODE/SDE path sampling |
| `rlinf/utils/flow_actor_checkpoint.py` | 严格 portable actor manifest 与 actor-only load |
| `examples/sft/evaluate_franka_gello_flow_bc.py` | 不启动 Ray 的静态 ODE/SDE 评估 |

## 2. 数据合同

dataset 的每个样本固定返回：

```text
action:            float[H, 7]
action_valid_mask: bool[H]
```

DataLoader 后分别为 `[B,H,7]` 与 `[B,H]`。`H=1` 也保留 horizon 维，mask 恒为
`[True]`；代码不再接受 `[B,7]` 作为 BC action。

future window 只能位于同一 episode。episode 尾部不足 `H` 时：

1. 先根据 padding 标记生成 prefix-valid mask；
2. 验证所有 valid action 均为 finite；
3. 将所有 invalid action coordinate 无条件覆盖为 `0`。

mask 不能由 action 是否为零推断，因为全零可能是合法控制量。zero padding 只是固定
shape 的占位值，不能进入监督或指标分母。

## 3. 模型与 loss

公共 action、initial noise 和 sample 始终使用 `[B,H,A]`。`FlowTActor` 仅在内部将其
reshape 为 chunk-major `[B,H*A]`，field 输出后再恢复为 `[B,H,A]`。

所有 horizon 使用相同 transform：

```text
environment action -> affine inverse -> clamp -> atanh -> identity latent
```

`action_valid_mask` 只进入 objective 的 loss/metric reduction，不进入 observation
condition、field 网络、JVP 输入或 sampler。这样静态评估不会把目标 episode 还剩多少步
泄露给模型。

对每个样本，设有效 coordinate 数 `n_i = A * sum(mask_i)`，masked squared error 的
样本均值为 `m_i`。adaptive loss 使用：

```text
d_i = stopgrad((m_i + epsilon) ** power)
loss = sum(n_i * m_i / d_i) / sum(n_i)
```

invalid error 在求和前通过 `torch.where` 置零。禁止直接对 `[B,H,A]` 调用 `mean()`，
否则短尾样本会被 zero padding 稀释。`field_mse`、norm 与逐 horizon metric 同样使用
有效 coordinate count。H=1 全真 mask 应与普通 unmasked reduction 数值一致。

注意：target 的 padded endpoint 是零，但 flow noise 仍覆盖整个 `[B,H,A]`，sampler 也
始终生成完整 chunk。mask 不用于清零 noise、flow state、prediction 或 JVP。

## 4. 配置

最小 iMF BC 配置如下：

```yaml
actor:
  model:
    flow_actor_type: FlowTActor
    action_dim: 7
    action_horizon: 8
    d_model: 256
    use_batch_norm: false
    flow_matching:
      implementation: pytorch_flow_t
      objective: improved_meanflow
      action_transform: tanh_latent
      action_chunking:
        action_layout: chunk_major
        latent_normalization: identity
      improved_meanflow:
        boundary_pair_probability: 0.5
        logit_mean: -0.4
        logit_std: 1.0
        adaptive_power: 0.5
        adaptive_epsilon: 0.001
        time_conditioning:
          from_encoder: independent_mlp
          to_encoder: independent_mlp
          fusion: concat_linear_2d_to_d
    flow_sampling:
      evaluation:
        method: flow_ode
        num_steps: 10
```

`action_horizon` 必须为正整数，`d_model >= action_horizon * action_dim`，采样步数必须
位于 `[1,99]`。BC 只接受 `flow_sampling.evaluation`，method 为 `flow_ode` 或
`flow_sde`。

使用 iMF SDE 时增加：

```yaml
flow_sampling:
  evaluation:
    method: flow_sde
    num_steps: 10
  flow_sde:
    noise_level: 0.1
    noise_std_range: [0.005, 0.05]
    safe_initial_time: 0.99
    joint_path_logprob: true
```

iMF 要求 `0 < std_min <= std_max`、`0 < safe_initial_time < 1`，且
`safe_initial_time > 1 - 1 / num_steps`。RF SDE 只配置正数 `noise_level`；其 std 由
corrected-drift kernel 推导，必须删除 iMF 专用的 range 与 safe-time 字段。

## 5. ODE/SDE 采样

RF 保持 `t=0` noise 到 `t=1` action；iMF 保持 `t=1` noise 到 `t=0` action。两者直接
复用 `flow_transition.py` 中现有 kernel。

公共随机量与 trace shape：

```text
initial_noise: [B,H,A]
step_noises:   [B,N,H,A]
states:        [B,N+1,H,A]
means/stds:    [B,N,H,A]
step_log_prob: [B,N,1]
path_log_prob: [B,1]
```

给定相同 condition、initial noise、step noises 与配置，SDE path 必须完全可复现。
target mask 从不传给 ODE/SDE；只在输出 action 与 target 比较时过滤 padded tail。

## 6. 训练与静态评估

确认数据路径、encoder checkpoint 和 placement 后启动：

```bash
bash examples/sft/run_vla_sft.sh franka_gello_flow_bc
```

单 GPU 100-step smoke：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
  examples/sft/verify_franka_gello_flow_bc.py \
  --steps 100 --batch-size 8
```

静态 ODE 评估：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
  examples/sft/evaluate_franka_gello_flow_bc.py \
  --checkpoint /path/to/global_step_N \
  --data-root /path/to/collected_data \
  --split val \
  --sampler-method flow_ode \
  --num-steps 10
```

同一 checkpoint 切换到 SDE 不需要改权重：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
  examples/sft/evaluate_franka_gello_flow_bc.py \
  --checkpoint /path/to/global_step_N \
  --data-root /path/to/collected_data \
  --split val \
  --sampler-method flow_sde \
  --num-steps 10 \
  --sde-noise-level 0.1 \
  --sde-noise-std-range 0.005 0.05 \
  --sde-safe-initial-time 0.99 \
  --initial-noise-seed 1234 \
  --step-noise-seed 1235
```

报告记录实际 sampler、步数、SDE 参数、initial-noise seed、step-noise seed 和 objective
seed base。aggregate、all-valid full-chunk、per-dimension、per-horizon 与 best-of-K
均忽略 padded tail；all-valid full-chunk 指标只统计 `mask.all(dim=1)` 的样本，
best-of-K 每个 observation 只能选择一条完整 sample，不能跨 horizon 拼接。

## 7. Portable checkpoint

每个 portable artifact 包含：

```text
flow_actor/
├── flow_actor_manifest.json
└── model_state_dict/full_weights.pt
```

manifest 只接受 `schema: unified_flow_t_bc`，不使用数字版本分派。它记录 objective/time
方向、H、A、`flow_state_dim=H*A`、`chunk_major`、identity normalization、
`padding=zero`、`mask_role=loss_only` 与有效 coordinate loss reduction。

ODE/SDE method 和随机 seed 是运行时选项，不属于权重兼容条件。同一 H 的 portable actor
严格加载；跨 H、旧 manifest、缺少 manifest、tensor key/shape/dtype 不一致都会在修改
目标模型前失败。critic/value head 不写入 portable artifact。

完整 FSDP/optimizer resume 与 portable actor load 是两条不同路径。新完整 checkpoint 只
保证同一合同 resume；没有 H1→H>1 inflation 或旧 artifact 转换。

## 8. 验证清单

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/unit_tests/test_franka_gello_flow_dataset.py \
  tests/unit_tests/test_flow_objectives.py \
  tests/unit_tests/test_flow_t_actor.py \
  tests/unit_tests/test_flow_actor_checkpoint.py \
  tests/unit_tests/test_flow_bc_static_evaluation.py

.venv/bin/ruff check rlinf examples/sft tests/unit_tests
.venv/bin/ruff format --check rlinf examples/sft tests/unit_tests
git diff --check
```

最低验收包括 `H={1,2,8}`、短 episode、跨 episode 边界、zero tail、RF/iMF、
ODE/SDE、固定随机量复现、跨 H checkpoint 拒绝，以及 H>1 online 配置 fail-fast。
