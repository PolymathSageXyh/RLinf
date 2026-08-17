# Franka + GELLO Flow BC 开发与操作指南

本文帮助你快速读懂并运行 RLinf 当前的 PyTorch Flow-T 行为克隆（Behavior Cloning，BC）路径。阅读完成后，你应该能回答三个问题：数据从哪里来、RF/iMF loss 在哪里计算、训练产物如何交给 online SACFlow。

除非特别说明，本文命令都从 RLinf 仓库根目录执行。

> 当前实现只支持单臂 Franka + GELLO、`FlowPolicy(input_type="mixed")`、`FlowTActor`、7 维动作和单动作块。`JaxFlowTActor`、JAX/Flax 和 OpenPI 不在这条 BC 路径中。

本机真实数据与 GPU6 的 100 步验收记录见
[`flow_bc_progress.md`](flow_bc_progress.md)。

## 1. 先看结论

BC 复用 RLinf 现有 SFT/FSDP 训练框架，并没有另建 trainer：

```text
GELLO 采集
  └─ collected_data/rank_N/id_M   LeRobot shards
       ↓
FrankaGelloFlowDataset
       ↓
StatefulDataLoader
       ↓
FSDPVlaSftWorker
       ↓ ForwardType.SFT
FlowPolicy
  ├─ ResNetEncoder + state_proj + mix_proj
  └─ FlowTActor
       ↓
RectifiedFlowObjective 或 ImprovedMeanFlowObjective
       ↓
FSDP checkpoint + portable actor-only checkpoint
```

默认示例训练 Improved MeanFlow（iMF）。如果只想先跑通链路，建议先保持默认配置；RF 只需要切换 objective。

## 2. 推荐代码阅读顺序

不要从 `FlowTActor` 的数学实现直接开始。按下面顺序读，能先建立“入口—数据—模型—产物”的完整地图。

| 顺序 | 文件与关键符号 | 重点看什么 |
|---|---|---|
| 1 | [`examples/sft/config/franka_gello_flow_bc.yaml`](../../examples/sft/config/franka_gello_flow_bc.yaml) | 当前可运行配置、7D/19D 数据契约、默认 iMF 参数 |
| 2 | [`rlinf/config.py::_validate_flow_policy_v2_cfg`](../../rlinf/config.py) | 哪些配置是硬约束，哪些拼写错误会 fail-fast |
| 3 | [`examples/sft/train_vla_sft.py::main`](../../examples/sft/train_vla_sft.py) | Hydra 校验、Ray cluster、worker group 和 runner 如何创建 |
| 4 | [`rlinf/runners/sft_runner.py::SFTRunner`](../../rlinf/runners/sft_runner.py) | 训练 step、日志、保存频率和 `runner.resume_dir` |
| 5 | [`rlinf/workers/sft/fsdp_vla_sft_worker.py::FSDPVlaSftWorker`](../../rlinf/workers/sft/fsdp_vla_sft_worker.py) | Flow dataloader 分支、`ForwardType.SFT`、FSDP save/resume 和 actor-only 导出 |
| 6 | [`rlinf/data/datasets/flow/franka_gello.py`](../../rlinf/data/datasets/flow/franka_gello.py) | LeRobot shard 发现、样本适配、图像范围和 7D 动作检查 |
| 7 | [`rlinf/models/embodiment/flow_policy/flow_policy.py::FlowPolicy`](../../rlinf/models/embodiment/flow_policy/flow_policy.py) | observation encoder 如何与同一个 `FlowTActor` 组合；`sft_forward()` 如何进入 objective |
| 8 | [`rlinf/models/embodiment/modules/flow_actor.py::FlowTActor`](../../rlinf/models/embodiment/modules/flow_actor.py) | `encode_condition()`、`predict_field()`、RF 单时间和 iMF 双时间条件 |
| 9 | [`rlinf/models/embodiment/modules/flow_objectives.py`](../../rlinf/models/embodiment/modules/flow_objectives.py) | RF/iMF 的插值、target、JVP、stop-gradient 和日志指标 |
| 10 | [`rlinf/utils/flow_actor_checkpoint.py`](../../rlinf/utils/flow_actor_checkpoint.py) | portable checkpoint 的 manifest、actor scope 和 online 兼容校验 |
| 11 | [`tests/unit_tests/test_flow_objectives.py`](../../tests/unit_tests/test_flow_objectives.py)、[`test_flow_policy_v2.py`](../../tests/unit_tests/test_flow_policy_v2.py)、[`test_flow_actor_checkpoint.py`](../../tests/unit_tests/test_flow_actor_checkpoint.py) | 代码契约的最小、可执行示例 |

如果你还需要理解 demonstration 是如何生成的，再回头读：

1. [`examples/embodiment/config/realworld_collect_data_gello.yaml`](../../examples/embodiment/config/realworld_collect_data_gello.yaml)
2. [`examples/embodiment/collect_real_data.py::DataCollector`](../../examples/embodiment/collect_real_data.py)
3. [`rlinf/envs/wrappers/collect_episode.py::CollectEpisode`](../../rlinf/envs/wrappers/collect_episode.py)
4. [`rlinf/data/lerobot_writer.py::LeRobotDatasetWriter`](../../rlinf/data/lerobot_writer.py)

## 3. 数据从采集到训练

### 3.1 采集会生成两份互补数据

运行 GELLO 采集后，同一个日志目录下会有：

```text
<collection_log>/
├─ collected_data/     # LeRobot；给 BC 使用
│  └─ rank_0/id_*/
└─ demos/              # TrajectoryReplayBuffer；给 online SAC 使用
```

不要用 `collected_data/` 重建 online transition。LeRobot 数据包含状态、动作、图像和 episode 标志，但不完整保存 SAC 所需的 reward、`next_obs`、termination/truncation 语义；`demos/` 已经保存这些 transition 字段。

采集命令：

```bash
bash examples/embodiment/collect_data.sh realworld_collect_data_gello
```

运行前至少修改采集 YAML 中的：

- `cluster.node_groups[].hardware.configs[].robot_ip`
- `env.eval.gello_port`
- `env.eval.override_cfg.target_ee_pose`
- `runner.num_data_episodes`

`collect_data.sh` 会覆盖 `runner.logger.log_path` 为新的时间戳目录。完成后从日志中确认实际路径，再把 `<collection_log>/collected_data` 写入 BC YAML，把 `<collection_log>/demos` 留给 online YAML。

如需在同一采集目录续采，给 `env.eval.data_collection` 增加 `resume: true`，并显式复用原来的 `runner.logger.log_path`。`collect_data.sh` 每次都会生成新路径，因此续采时应直接调用入口：

```bash
export EMBODIED_PATH="$(pwd)/examples/embodiment"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python examples/embodiment/collect_real_data.py \
  --config-path examples/embodiment/config \
  --config-name realworld_collect_data_gello \
  runner.logger.log_path=/data/franka_collection_A \
  +env.eval.data_collection.resume=true
```

恢复逻辑要求 `demos/metadata.json` 与 `demos/trajectory_index.json` 同时存在；只存在一个会直接报错，防止覆盖旧数据。

### 3.2 BC dataset 的实际输出

`build_franka_gello_flow_dataloader()` 复用 `RollingLeRobotDataset` 解码 shard，再由 `adapt_franka_gello_sample()` 输出：

```python
{
    "obs": {
        "states": float32[B, 19],
        "main_images": float32[B, 3, 128, 128],
        # 多相机时才有：
        "extra_view_images": float32[B, N - 1, 3, 128, 128],
    },
    "actions": float32[B, 7],
    "images_preprocessed": bool[B],  # 全 True
}
```

这里有四个有意设置的硬边界：

- 动作必须是 7D：6D 末端增量 + gripper，不允许静默切片。
- 状态维数必须和 `actor.model.state_dim` 一致，当前 Franka 合同为 19。
- 图像输出是 CHW、float32、`[0,1]`。
- dataset 不负责 resize；图像尺寸必须直接匹配 `actor.model.image_size`。

`images_preprocessed=True` 会让 `FlowPolicy.preprocess_env_obs()` 跳过 `/255`。不要把这里改回原始 env 的 uint8 处理方式，否则会发生二次归一化。

## 4. 关键模块与设计思路

### 4.1 `FSDPVlaSftWorker`：复用统一 SFT 基础设施

`FSDPVlaSftWorker.build_dataloader()` 根据 `model_type=flow_policy` 选择 Franka GELLO adapter。基础类继续负责：

- FSDP/FSDP2 包装；
- AMP 与 gradient accumulation；
- optimizer、scheduler 和 grad clipping；
- distributed metric reduce；
- model、optimizer、scheduler 的完整断点；
- `StatefulDataLoader` 和各 rank RNG 状态的保存恢复。

`get_train_model_output()` 调用：

```python
self.model(forward_type=ForwardType.SFT, data=batch)
```

返回字典中的 `loss` 保持计算图，其余 objective metrics 被 detach 后记录。

### 4.2 `FlowPolicy`：BC 与 online 共用的完整 actor

对 mixed observation，`FlowPolicy` 的共享路径是：

```text
main/extra images → encoders ┐
                             ├→ full_feature → mix_proj → mix_feature
states → state_proj ─────────┘
                                              ↓
                                   FlowTActor.encode_condition()
```

`sft_forward()` 随后把 demonstration action 从环境空间反变换到 flow latent：

```text
action → (action - bias) / scale → clamp → atanh → latent action
```

默认 `action_scale: [-1, 1]`，因此 latent action 是 `atanh(clamp(action))`。GELLO
先在基坐标系裁剪动作，随后 `RelativeFrame` 的逆变换可能让记录在策略坐标系中
超出范围；BC 会按约定直接裁剪到 `[-1+1e-4, 1-1e-4]`，并通过
`action_clamp_fraction` 和 `action_clamp_abs_max` 指标暴露严格超出 `[-1, 1]` 的比例和
最大超幅。合法的 `+/-1` 也会为 `atanh` 的数值稳定性替换成 `+/- (1-1e-4)`，但不计入
这两个越界指标。

BC checkpoint 导出的不只是 `flow_actor.*`，还包含 observation-to-action 所需的：

- `encoders.*`
- `state_proj.*`
- `mix_proj.*`
- `flow_actor.*`
- action scale/bias buffers

它明确排除 `q_head.*` 和 `value_head.*`。这就是 BC 和 online 能共享完整策略表征、同时让 online critic 保持新初始化的原因。

### 4.3 `FlowTActor`：legacy 与 v2 显式隔离

只有配置：

```yaml
flow_matching:
  implementation: pytorch_flow_t_v2
```

才启用新接口。没有该块的旧 YAML 继续走 legacy `forward()`；`JaxFlowTActor` 不受影响。

v2 的核心接口是：

- `encode_condition(obs, update_stats=...)`：编码一次 observation condition。
- `predict_field(condition, z, t_from=..., t_to=...)`：纯确定性 field 预测。
- `sample_path(...)`：online/rollout 才使用的 ODE、flow-noise、flow-SDE 路径采样。

RF 与 iMF 复用 action projection、Transformer cross-attention 和输出 head，但时间语义不同：

| Objective | 时间方向 | 网络输出 | 时间条件 |
|---|---|---|---|
| Rectified Flow | `t=0 noise → t=1 action` | 瞬时速度 `v(z,t)` | `t_from` 单编码 |
| Improved MeanFlow | `t=1 noise → t=0 action` | 区间平均场 `U(z,t_from,t_to)` | 独立 from/to 编码，concat 后 `2d→d` fusion |

iMF 专用的 `time_to_embedding` 和 `time_fusion` 只在 iMF v2 中创建，因此 RF 与 iMF checkpoint 不能互载。

### 4.4 `flow_objectives.py`：field 到底从哪里预测

先记住最重要的一句话： **`field` 不是 dataset 里的字段，也不是 `flow_objectives.py` 自己计算出来的标签；它是 `FlowTActor` 根据 condition、当前 flow state 和时间预测出的网络输出。**

生产训练中的完整调用链是：

```text
FlowPolicy.sft_forward(data)
  │
  ├─ get_feature(obs) + mix_proj
  │    └─ mix_feature: [B, 256]
  │
  ├─ FlowTActor.encode_condition(mix_feature)
  │    └─ condition: [B, 1, d_model]
  │
  ├─ environment action → atanh → latent_action: [B, action_dim]
  │
  └─ objective(FlowTActor, condition, latent_action)
       │
       ├─ 构造 flow_state z_t 和监督 target
       ├─ _resolve_predict_field(FlowTActor, condition)
       │    └─ 返回捕获 condition 的 predict_from_condition 闭包
       ├─ _call_predict_field(closure, z_t, t_from, t_to)
       │    └─ closure → FlowTActor.predict_field(condition, z_t, ...)
       │         ├─ action_proj(z_t)
       │         ├─ time embedding（iMF 还会做双时间 fusion）
       │         ├─ Transformer decoder，以 condition 为 cross-attention memory
       │         └─ velocity_mean_head(hidden)
       │              └─ FlowFieldOutput.value: [B, action_dim]
       └─ predicted field 与 target 计算 loss
```

所以，`flow_objectives.py` 的职责是 **构造监督问题并调用 field predictor**；`FlowTActor` 才是 **实际预测 field 的神经网络**。

#### 4.4.1 `FlowFieldOutput.value` 的生成路径

`FlowTActor.predict_field()` 接收：

```python
condition:  [B, 1, d_model]
flow_state: [B, action_dim]
t_from:     scalar 或 [B, 1]
t_to:       iMF 为 scalar/[B, 1]；RF 必须为 None
```

内部先执行 `_embed_action_time()`：

```text
flow_state
  → optional BatchRenorm
  → action_proj
  → action_embedding [B, 1, d_model]

t_from
  → time_embedding
  → from_embedding [B, 1, d_model]

iMF only:
t_to
  → time_to_embedding
  → to_embedding [B, 1, d_model]

[from_embedding, to_embedding]
  → concat [B, 1, 2*d_model]
  → time_fusion
  → time_embedding [B, 1, d_model]

token = action_embedding + time_embedding
```

然后 `_decode_field_output(token, condition)` 把 token 依次送入 Transformer decoder layers。这里：

- `token` 是 decoder target/query；
- `condition` 是 observation memory；
- cross-attention 让每个 action-time token读取图像和状态编码；
- 最终 `hidden.squeeze(1)` 的 shape 为 `[B,d_model]`；
- `velocity_mean_head(hidden)` 投影到 `[B,action_dim]`。

返回值是：

```python
FlowFieldOutput(
    value=velocity_mean_head(hidden),
    hidden=hidden,
    field_kind=(
        "instantaneous_velocity"   # RF
        or "interval_average"      # iMF
    ),
)
```

虽然最后一层历史命名为 `velocity_mean_head`，其数学语义由 objective 决定：RF 的 `value` 是瞬时速度 `v(z,t)`；iMF 的 `value` 是区间平均场 `U(z,t_from,t_to)`。

#### 4.4.2 `_resolve_predict_field()`：把 actor 统一成一个纯调用接口

函数签名是：

```python
def _resolve_predict_field(
    actor_or_predict_field,
    obs,
    *,
    update_stats,
) -> PredictField
```

这里的 `actor_or_predict_field` 支持两种输入：

1. 生产代码中的 `FlowTActor`，它同时提供 `encode_condition()` 和 `predict_field()`；
2. 单元测试中的解析函数/test double，它本身就是 callable。

`PredictField` 类型统一为：

```python
(flow_state, t_from, t_to) -> Tensor 或带 value 的输出对象
```

对于真正的 actor，函数先用 `getattr()` 取出两个方法，并确认都可调用：

```python
encode_condition = actor.encode_condition
predict_field = actor.predict_field
```

接下来它判断第二个参数 `obs` 是否已经是 condition：

```python
is Tensor
and obs.ndim == 3
and obs.shape[1] == 1
and obs.shape[-1] == actor.d_model
```

满足时直接执行：

```python
condition = obs
```

否则才执行：

```python
condition = actor.encode_condition(obs, update_stats=update_stats)
```

这解释了为什么参数名虽然叫 `obs`，生产 BC 路径里实际传入的是 `condition`。`FlowPolicy.sft_forward()` 已经提前调用过：

```python
condition = self.flow_actor.encode_condition(
    mix_feature,
    update_stats=self.training,
)

objective_output = self.flow_bc_objective(
    self.flow_actor,
    condition,
    latent_actions,
)
```

因此 objective 不会第二次编码 observation。

最后 `_resolve_predict_field()` 定义并返回闭包：

```python
def predict_from_condition(flow_state, t_from, t_to):
    return actor.predict_field(
        condition,
        flow_state,
        t_from=t_from,
        t_to=t_to,
    )
```

闭包把固定的 `condition` 捕获起来，外部只需要反复传 `z/t/r`。这样设计主要服务 iMF JVP：

- observation encoder 只运行一次，避免重复计算；
- `torch.autograd.functional.jvp` 的 closure 只包含 field 对 `z/t/r` 的函数；
- BatchRenorm running stats 不会在一次 objective/JVP 内被多次修改；
- condition 仍保留正常 autograd 图，所以主 loss 可以更新 observation encoder。

如果第一个参数没有 actor 接口但本身可调用，函数会直接返回它。这个分支主要让测试传入简单解析场，例如：

```python
def affine_field(flow_state, t_from, t_to):
    return 2 * flow_state + t_from
```

若两种形式都不满足，则抛出 `TypeError`，不会猜测未知模型接口。

#### 4.4.3 `_extract_field_value()`：只解包，不做预测

`_extract_field_value(output)` 解决的是“不同 predictor 返回格式不同”的兼容问题。它按以下顺序取出 Tensor：

| predictor 返回值 | 提取方式 | 典型用途 |
|---|---|---|
| `Tensor` | 直接返回 | 解析测试函数 |
| `Mapping` | 依次查 `value`、`field`、`mean` | dict 风格 test double/兼容接口 |
| object/dataclass | 依次取 `.value`、`.field`、`.mean` | 生产代码的 `FlowFieldOutput.value` |

生产 `FlowTActor.predict_field()` 返回 `FlowFieldOutput`，所以实际走的是：

```python
value = output.value
```

该函数不会调用网络、不会复制 Tensor、不会 detach，也不会改变梯度。它只把不同外壳规范化成一个 Tensor。如果最终找不到 Tensor，就立即抛出 `TypeError`。

`hidden` 和 `field_kind` 不参与 BC objective 的数值计算：

- `hidden` 供 online `flow_noise_head` 使用；
- `field_kind` 记录输出语义，便于 sampler/checkpoint 契约表达；
- BC loss这里只需要 `value`。

#### 4.4.4 `_call_predict_field()`：执行预测并检查 shape

该函数只有两步：

```python
raw_output = predict_field(flow_state, t_from, t_to)
value = _extract_field_value(raw_output)
```

第一行才会沿着闭包真正调用 `FlowTActor.predict_field()`。第二行把 `FlowFieldOutput` 解包成 field Tensor。

随后它强制：

```python
value.shape == flow_state.shape
```

当前 Franka 场景中两者都应为 `[B,7]`。这是必要的，因为 flow field 是 action latent 空间中的向量：每个 action 维度都必须有一个对应的速度或区间平均分量。若模型意外返回 `[B,1]`、`[B,d_model]` 或错误 action chunk shape，会在 loss/JVP 前直接报错。

同样，该函数不 detach field。RF 的 `prediction` 和 iMF 的 `average_field` 都通过它保留对 `velocity_mean_head`、Transformer、time encoder、condition encoder 的梯度。

#### 4.4.5 三个 helper 在 RF 中如何串起来

RF forward 的核心可以展开为：

```python
# 1. objective 构造监督输入
flow_state = (1 - t) * noise + t * latent_action
target = latent_action - noise

# 2. 把 FlowTActor + condition 解析成统一 callable
predict = _resolve_predict_field(
    actor_or_predict_field=flow_actor,
    obs=condition,
    update_stats=False,
)

# 3. 真正调用网络并得到 [B, 7] field
prediction = _call_predict_field(
    predict,
    flow_state,
    t,
    None,
)

# 等价的实际网络调用
prediction = flow_actor.predict_field(
    condition,
    flow_state,
    t_from=t,
    t_to=None,
).value

# 4. 监督瞬时速度
loss = MSE(prediction, target)
```

`_resolve_predict_field()` 和 `_call_predict_field()` 是接口适配层；真正可训练的 field 参数仍全部位于 `FlowTActor` 及其上游 condition encoder 中。

#### 4.4.6 三个 helper 在 iMF/JVP 中如何串起来

iMF 会先解析一次 predictor：

```python
predict_field = _resolve_predict_field(flow_actor, condition, update_stats=False)
```

然后第一次调用对角 field：

```python
boundary_field = _call_predict_field(
    predict_field,
    flow_state,
    t_from,
    t_from,
)
```

这等价于 `U(z,t,t)`，只用于 JVP 的 state tangent。

接着定义 JVP closure。注意函数参数顺序是 `(state_arg, r_arg, time_arg)`，但 actor API 是 `t_from=time_arg, t_to=r_arg`：

```python
def interval_field(state_arg, r_arg, time_arg):
    return _call_predict_field(
        predict_field,
        state_arg,
        time_arg,  # t_from
        r_arg,     # t_to
    )
```

JVP 调用：

```python
average_field, material_derivative = torch.autograd.functional.jvp(
    interval_field,
    (flow_state, t_to, t_from),
    (
        boundary_field.detach(),
        zeros_like(t_to),
        ones_like(t_from),
    ),
    create_graph=True,
    strict=False,
)
```

其方向导数语义是：

```text
material_derivative
  = (∂U/∂z) · stop_grad(U(z,t,t))
  + (∂U/∂t_from) · 1
  + (∂U/∂t_to) · 0
```

也就是说，`t_to` 在这次导数中保持不动。`average_field` 是 closure 在 primal `(z,t_to,t_from)` 上的普通网络输出，即 `U(z,t_from,t_to)`。

`torch.autograd.functional.jvp` 使用 backward-of-backward 方式计算 JVP。代码显式设置：

- `create_graph=True`：保留 primal `average_field` 的计算图，使主 loss 可以更新 actor；若使用默认 `False`，primal 和 JVP 都会失去 `grad_fn`；
- `strict=False`：当 field 与某个输入端点无关时，将该方向导数视为数学上的 0，而不是报错。

代码还使用 `sdpa_kernel(SDPBackend.MATH)` 强制 attention 的 math backend，使 Transformer attention 支持该 double-backward JVP 路径。虽然 `create_graph=True` 也会让 `material_derivative` 暂时带图，但下一行会将它 detach；保留图的必要对象是 `average_field`。

最后：

```python
regression_field = (
    average_field
    + (t_from - t_to) * material_derivative.detach()
)
```

这里的梯度边界很重要：

- `average_field` 不 detach，主 loss 通过它更新 actor；
- `material_derivative` detach，避免主 loss对 JVP 再求高阶参数导数；
- 作为 tangent 的 `boundary_field` 也 detach；
- 当 `t_from == t_to` 时，第二项为 0，`average_field=U(z,t,t)` 得到直接监督。

#### 4.4.7 其余 helper 的职责

| helper | 作用 | 关键约束 |
|---|---|---|
| `FlowObjectiveOutput` | 返回可反传 scalar `loss` 和 detached metrics | worker 只对 `loss` backward |
| `_adaptive_weighted_mse()` | 先算每样本 MSE，再除以 detached `(mse+epsilon)^power` | RF 默认 `power=0` 等价普通 MSE；iMF 默认 `power=1` |
| `_make_generator()` | 统一显式 generator 或 seed | 两者不能同时传；seed 会创建同 device generator |
| `_sample_like()` | 采与 action 同 shape/device/dtype 的高斯噪声 | 当前为 `[B,7]` |
| `_sample_batch_uniform()` | 为 RF 采每样本时间 | 输出 `[B,1]` |
| `_normalize_time()` | 把 scalar/单元素/每样本时间转成 `[B,1]` | 要求 finite 且位于 `[0,1]` |
| `_time_for_flow_state()` | 把 `[B,1]` reshape 成可与任意 flow-state rank 广播的 shape | 当前 action `[B,A]` 时仍为 `[B,1]` |
| `_validate_action()` | 检查至少含 batch/action 两维且为浮点 Tensor | objective 输入已是 latent action |
| `_prepare_noise()` | 使用显式 noise 或按 action shape采样 | 显式 noise 必须严格同 shape |
| `build_flow_objective()` | 把 YAML-facing 名称映射到 objective class | 支持 `rectified_flow/rf` 与 `improved_meanflow/imf` aliases |

`_adaptive_weighted_mse()` 中 denominator 被 detach，因此它只改变每个样本 loss 的权重，不让模型通过改变 denominator 走捷径。返回的 `field_mse` 是未加权均值，适合判断真实回归误差是否下降。

### 4.5 RF objective

设 demonstration latent action 为 `y`，标准高斯噪声为 `ε`：

```text
z_t = (1 - t) ε + t y
target = y - ε
loss = MSE(vθ(z_t, t), target)
```

`t` 默认按 batch 独立从均匀分布采样。当前 YAML 不需要 `flow_matching.rectified_flow` 子参数。

### 4.6 Improved MeanFlow objective

iMF 保持 native 时间方向：

```text
z_t = (1 - t) y + t ε
target = ε - y
t_from >= t_to
```

两个时间先各自从 Logit-Normal 采样，然后取 max/min。`boundary_pair_probability` 比例的样本会令 `t_to=t_from`，直接监督对角 field。

objective 内部只编码一次 condition。随后：

1. 用 `U(z,t,t)` 得到 JVP 的 state tangent，并 detach；
2. 用 `torch.autograd.functional.jvp(create_graph=True, strict=False)` 计算 interval field 的 material derivative；
3. 只 detach material derivative，不 detach主 field；
4. 对回归误差应用 adaptive weighting。

注意：`U(z,t,t)` 只服务于 BC objective 的 JVP。iMF 的 ODE/noise/SDE sampler 始终查询相邻区间的 `U(z,t_from,t_to)`。

## 5. 你需要修改的配置

以 [`franka_gello_flow_bc.yaml`](../../examples/sft/config/franka_gello_flow_bc.yaml) 为唯一模板。下面按“必须修改”“通常调参”“兼容性合同”分类。

### 5.1 每次运行必须确认

| 配置键 | 当前示例 | 作用 |
|---|---:|---|
| `data.train_data_paths` | `/nas/xyh/data/data_pick_cube/collected_data` | 采集根目录、某个 `rank_N` 或单个 finalized LeRobot shard |
| `actor.model.model_path` | `/nas/xyh/RLinf/.cache/flow_bc/RLinf-ResNet10-pretrained` | ResNet 权重所在目录 |
| `actor.model.encoder_config.ckpt_name` | `resnet10_pretrained.pt` | 相对 `model_path` 的 encoder 权重文件名 |
| `runner.logger.log_path` | `/nas/xyh/RLinf/results/flow_bc` | 直接运行 Python 时的输出根；shell wrapper 会覆盖为时间戳日志目录 |
| `runner.logger.experiment_name` | `franka_gello_flow_bc` | checkpoint 的实验子目录 |
| `cluster.num_nodes` / placement | `1` / `actor: 6-6` | 单节点、宿主机物理 GPU 6 上的 actor placement |

### 5.2 数据与 batch 参数

| 配置键 | 默认值 | 如何理解 |
|---|---:|---|
| `data.format` | `lerobot` | 当前 adapter 的唯一格式 |
| `data.state_key` | `state` | LeRobot frame 中的状态字段 |
| `data.action_key` | `actions` | LeRobot frame 中的动作字段 |
| `data.image_keys` | `[image]` | 第一项映射到 `main_images`，其余映射到 `extra_view_images` |
| `data.image_value_range` | `zero_one` | `RollingLeRobotDataset` 解码后的范围；建议保持此值，亦允许 `auto` |
| `data.fps` | `10` | 传给 rolling dataset 的数据帧率合同，应与采集一致 |
| `data.min_frames` | `1` | 可用 frame 少于此值时启动失败 |
| `data.load_workers` | 未设置，等价 `0` | 并行载入 archived shards 的 worker 数 |
| `data.num_workers` | `0` | PyTorch dataloader worker 数；真机数据先从 0 开始验证 |
| `data.use_random_replacement` | `true` | 小数据集按有放回方式定义每个 epoch，避免 batch 不足 |
| `data.num_samples_per_epoch` | `3200` | replacement sampler 每个 epoch 抽取的 sample 数；不等于真实 frame 数 |
| `actor.micro_batch_size` | `8` | 每 rank 单次 forward 的样本数；可按 GPU 余量调整 |
| `actor.global_batch_size` | `8` | 当前单 actor rank 的有效 batch；多 rank 时需按约束同步调整 |

必须满足：

```text
global_batch_size % (micro_batch_size × actor_world_size) == 0
```

关闭 replacement sampling 时，每个 rank 的 sampler 长度还必须不小于 `micro_batch_size`。

### 5.3 模型合同

下列值当前不是普通超参数，而是 Franka GELLO v2 的兼容边界：

```yaml
actor:
  model:
    model_type: flow_policy
    input_type: mixed
    flow_actor_type: FlowTActor
    state_dim: 19
    action_dim: 7
    num_action_chunks: 1
    image_size: [3, 128, 128]
    image_num: 1
    action_scale: [-1.0, 1.0]
    add_q_head: false
```

如果增加相机，需同时调整：

- `data.image_keys`
- `actor.model.image_num`
- 对应 observation encoder 的显存预算

不要只改 `image_num`。

模型容量参数：

| 配置键 | 默认值 | 作用与代价 |
|---|---:|---|
| `denoising_steps` | `4` | 默认 flow 步数；BC loss 本身随机采一个时间点，不做 K 步 rollout |
| `d_model` | `256` | action/time token 和 Transformer hidden 维度；增大会提高显存与计算量 |
| `n_head` | `4` | cross-attention 头数，必须整除 `d_model` |
| `n_layers` | `2` | Flow-T Transformer 层数 |
| `use_batch_norm` | `false` | 是否对 condition/action 使用 BatchRenorm 路径；默认关闭更易保持 BC/online 一致 |
| `batch_norm_momentum` | `0.99` | 仅在启用 normalization 时生效 |

上述架构字段会进入 checkpoint tensor schema或影响 key/shape。BC 与 online 必须保持一致。

### 5.4 iMF 参数

```yaml
flow_matching:
  implementation: pytorch_flow_t_v2
  objective: improved_meanflow
  action_transform: tanh_latent
  improved_meanflow:
    boundary_pair_probability: 0.5
    logit_mean: -0.4
    logit_std: 1.0
    adaptive_power: 1.0
    adaptive_epsilon: 0.01
    time_conditioning:
      from_encoder: independent_mlp
      to_encoder: independent_mlp
      fusion: concat_linear_2d_to_d
```

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `boundary_pair_probability` | 令 `t_to=t_from` 的 batch 比例；直接约束对角 field | 默认 `0.5`；降低前先检查对角 field 质量 |
| `logit_mean` | 控制时间采样集中位置 | 默认 `-0.4`，修改后观察 `t_mean/r_mean/time_gap_mean` |
| `logit_std` | 控制时间覆盖宽度，必须大于 0 | 太小会缩窄训练时间区域 |
| `adaptive_power` | adaptive loss denominator 的幂 | `0` 退化为普通 MSE；默认 `1` |
| `adaptive_epsilon` | 防止小误差权重发散，必须大于 0 | 默认 `0.01` |
| `time_conditioning.*` | 双时间架构标识 | 当前必须保持示例中的三个固定值 |

### 5.5 切换到 Rectified Flow

BC 只需把：

```yaml
actor:
  model:
    flow_matching:
      objective: rectified_flow
```

其余 actor 结构可以不变。建议删除不再生效的 `improved_meanflow` 子块，避免阅读配置时误判；online 配置还必须同步切换 profile，详见 online 指南。

### 5.6 训练与保存参数

| 配置键 | 默认值 | 作用 |
|---|---:|---|
| `runner.max_steps` | `100` | 当前验收配置的 optimizer step 上限 |
| `runner.max_epochs` | `-1` | epoch 上限；若两个上限都为正，先到者停止 |
| `runner.save_interval` | `100` | 当前验收配置每多少个 global step 保存一次 |
| `runner.val_check_interval` | `-1` | 当前 embodied SFT eval 未实现，保持关闭 |
| `actor.optim.lr` | `3e-4` | encoder、projection 和 Flow-T 的共享学习率 |
| `actor.optim.weight_decay` | `0` | Adam weight decay |
| `actor.optim.clip_grad` | `1.0` | FSDP 梯度裁剪阈值 |
| `actor.fsdp_config.save_full_model_weights` | `true` | 必须为 true，才能导出 portable actor checkpoint |

## 6. 如何运行、查看和恢复

### 6.1 启动

确认 Ray 已在完整节点资源视图中就绪，并使用仓库 `.venv` 后运行。示例 YAML 已将
`cluster.component_placement.actor` 固定为 `6-6`，因此不要在正式 Ray 入口前设置
`CUDA_VISIBLE_DEVICES=6` 来重新映射 rank；这会让 placement 语义与 Ray 的硬件 rank
不一致。

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python examples/sft/train_vla_sft.py \
  --config-path examples/sft/config \
  --config-name franka_gello_flow_bc \
  cluster.component_placement.actor=6-6
```

如果要在命令行临时覆盖配置，可以直接传 Hydra override：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python examples/sft/train_vla_sft.py \
  --config-path examples/sft/config \
  --config-name franka_gello_flow_bc \
  data.train_data_paths=/nas/xyh/data/data_pick_cube/collected_data \
  actor.model.model_path=/nas/xyh/RLinf/.cache/flow_bc/RLinf-ResNet10-pretrained \
  cluster.component_placement.actor=6-6 \
  runner.logger.log_path=/results
```

只做单进程 GPU 验收时，使用本页配套的
`examples/sft/verify_franka_gello_flow_bc.py`，并将 `CUDA_VISIBLE_DEVICES=6` 与
`cluster.component_placement.actor=6-6` 分开使用；前者只适用于该单进程脚本。

### 6.2 重点观察的指标

`SFTRunner` 会给 objective metrics 添加 `train/` 前缀：

| 指标 | 含义 |
|---|---|
| `train/loss` | 实际反向传播的 objective loss |
| `train/field_mse` | 未经 adaptive denominator 缩放的 field MSE |
| `train/t_mean` | 当前 batch 的 `t_from` 均值 |
| `train/r_mean` | iMF 的 `t_to` 均值 |
| `train/time_gap_mean` | iMF 区间长度均值 |
| `train/field_norm` | 回归 field 的平均范数 |
| `train/boundary_field_norm` | iMF 对角 field 范数 |
| `train/material_derivative_norm` | iMF JVP material derivative 范数 |
| `train/target_norm` | target field 范数 |
| `train/action_clamp_fraction` | 当前 batch 中严格超出 `[-1,1]` 的动作标量比例；合法端点不计入 |
| `train/action_clamp_abs_max` | 当前 batch 最大归一化超幅 `max(abs(action)-1, 0)` |

优先排查非有限值、field/target 范数数量级明显不一致，以及 `time_gap_mean` 长期接近 0。

### 6.3 checkpoint 目录

假设最终 `runner.logger.log_path=/results/run_A`，checkpoint 位于：

```text
/results/run_A/franka_gello_flow_bc/checkpoints/global_step_N/actor/
├─ model_state_dict/full_weights.pt   # 完整 SFT policy 权重
├─ data.pt                            # 各 rank StatefulDataLoader 状态
├─ rng.pt                             # 各 rank RNG 状态
└─ flow_actor/
   ├─ flow_actor_manifest.json
   └─ model_state_dict/full_weights.pt
```

online 配置的 `actor.model.pretrained_actor.path` 应指向最后的 `actor/flow_actor`，而不是完整 SFT actor 目录。

### 6.4 恢复 BC

BC 的恢复入口是完整 checkpoint：

```yaml
runner:
  resume_dir: /results/run_A/franka_gello_flow_bc/checkpoints/global_step_N
```

恢复时会加载 model、optimizer、scheduler、dataloader state 和 RNG。不要在 BC YAML 中配置 `actor.model.pretrained_actor`；该字段只用于 fresh online 初始化。

### 6.5 静态检查训练集拟合

训练结束后，可在不启动 Ray、不构造 optimizer 的情况下遍历训练集，检查 portable actor
对 demonstration action 的拟合程度：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python \
  examples/sft/evaluate_franka_gello_flow_bc.py \
  --checkpoint results/flow_bc/franka_gello_flow_bc/checkpoints/global_step_12000 \
  --data-root /nas/xyh/data/data_pick_cube
```

`--data-root` 可以指向采集根目录或其中的 `collected_data`。脚本会严格校验 portable
checkpoint manifest，并默认评估全部 frame。Flow 策略的输出依赖初始噪声，报告因此同时
包含单次随机采样、8 次采样均值、zero-noise 预测和 best-of-8；只有采样均值用于最终
`well_fitted` 判定，best-of-8 只是使用真值选择样本的诊断上界，不能当作部署指标。

默认判定要求采样均值满足：整体 R² 不低于 0.5、前 6 维 pose R² 不低于 0.5、pose
RMSE 不高于 0.1、gripper 符号准确率不低于 95%。阈值均可通过命令行修改。完整结果写入
`results/flow_bc/franka_gello_flow_bc/evaluations/global_step_12000_train_fit.json`；加上
`--require-good-fit` 可让未通过判定的运行返回非零状态。

## 7. Portable actor checkpoint 为什么严格

`flow_actor_manifest.json` 记录并校验：

- `framework=pytorch`
- `actor_class=FlowTActor`
- RF/iMF objective、field kind、时间方向和端点
- 单时间或双时间/fusion 合同
- action dim、action range 和 `tanh_latent`
- state/image 维数与图像范围
- 权重 key、shape 和 dtype schema

这样可以在加载权重之前拒绝以下错误：

- RF checkpoint 加载到 iMF online 配置；
- 6D 与 7D 动作混用；
- action range 不一致；
- iMF checkpoint 缺双时间参数；
- `JaxFlowTActor` checkpoint 进入 PyTorch v2；
- BC/online encoder 或 Transformer 结构不一致。

online-only 的 `flow_noise_head` 可以作为显式 allowed-missing scope 初始化；Q head 永远不会从 BC artifact 加载。

## 8. 常见问题

### 启动时报找不到 encoder checkpoint

检查：

```text
actor.model.model_path / actor.model.encoder_config.ckpt_name
```

两者拼接后必须是实际文件。

### 找不到 finalized LeRobot shard

`data.train_data_paths` 可以指向 `collected_data`、`rank_N` 或单个 `id_M`，但 shard 必须已经 finalized 并包含 `meta/info.json`。采集中断但未 finalize 的目录会被排除。

### 图像过暗或 loss 异常

保持 `data.image_value_range: zero_one`。当前 LeRobot decoder 已输出 `[0,1]` float；不要设为 `zero_255`。

### action 超出范围

确认采集和训练都使用 `no_gripper: false`。当前 BC 会按配置范围直接裁剪后再做
`atanh`；观察 `train/action_clamp_fraction` 和 `train/action_clamp_abs_max`，区分
`RelativeFrame` 造成的坐标系越界与采集异常。不要通过改大 `action_scale` 绕过问题，
因为该范围也是 online checkpoint 的兼容语义。

### 数据太少，dataloader 无法组成 batch

保持 `data.use_random_replacement: true`，并令 `data.num_samples_per_epoch` 至少覆盖每 rank 的 `micro_batch_size`。

### 想加载普通 legacy Flow 权重继续 BC

当前 v2 要求带 manifest 的 PyTorch `FlowTActor` artifact。不要使用 `strict=False` 猜测兼容 key；先显式转换并补全正确语义，或重新训练 v2 BC。

## 9. 修改代码时的最小验证集

修改 dataset、objective、FlowPolicy 或 checkpoint seam 后，至少运行：

```bash
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_franka_gello_flow_dataset.py \
  tests/unit_tests/test_flow_objectives.py \
  tests/unit_tests/test_flow_policy_v2.py \
  tests/unit_tests/test_flow_actor_checkpoint.py \
  tests/unit_tests/test_flow_config_v2.py
```

代码职责边界建议保持不变：

- 数据格式适配放在 `data/datasets/flow/`；
- RF/iMF 数学 loss 放在 `flow_objectives.py`；
- field/网络结构放在 `FlowTActor`；
- observation 与 action transform 放在 `FlowPolicy`；
- checkpoint 语义放在 `flow_actor_checkpoint.py`；
- 不复制 SFT runner、FSDP 或 logging 逻辑。

下一步请接着读 [`sacflow_online_finetune.md`](sacflow_online_finetune.md)，了解该 actor artifact 如何初始化 critic、anchor、alpha 和 online replay。
