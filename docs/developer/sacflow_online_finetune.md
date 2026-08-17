# Pretrained Flow Actor → Online SACFlow 开发与操作指南

本文帮助你快速读懂并运行 RLinf 当前的 PyTorch pretrained Flow-T 在线微调路径。它覆盖 actor-only 初始化、frozen anchor、critic warm-up、online SACFlow loss、replay/demo 混采、随机 flow sampler 和完整断点恢复。

除非特别说明，本文命令都从 RLinf 仓库根目录执行。

> 这是 SACFlow 附录 F 思路上的工程变体：不使用 Flow-T adapter，而是保留一个 frozen full-policy anchor，并直接微调 live actor。iMF online 和 Franka 真机训练仍应视为 experimental；先完成 dummy、硬件在环和低风险真机检查。

## 1. 先看结论

online 初始化和训练状态如下：

```text
BC portable actor checkpoint
  ├─ strict actor-only load → live FlowPolicy + fresh q_head
  ├─ exact deepcopy        → frozen anchor policy
  └─ live state hard-copy  → target policy

fresh alpha = 0.2
demos/ checkpoint → demo buffer
empty buffer       → online replay

前 10,000 个 online transitions：WARMUP
  critic/target 正常更新
  actor 只优化与 frozen anchor 的动作距离
  alpha 固定

之后：ONLINE
  critic/target 正常更新
  actor 优化 alpha * path_log_prob - Q + beta * anchor_loss
  alpha 开始更新
```

live actor 与 anchor 使用相同 observation 和同一组 initial/step noise。这样动作差异主要反映参数漂移，而不是两次随机采样的差异。

## 2. 推荐代码阅读顺序

按“配置—运行循环—初始化—loss—sampler—buffer—恢复”阅读，不要先陷入 SDE 公式。

| 顺序 | 文件与关键符号 | 重点看什么 |
|---|---|---|
| 1 | [`examples/embodiment/config/franka_sacflow_online_finetune.yaml`](../../examples/embodiment/config/franka_sacflow_online_finetune.yaml) | 当前完整配置、路径占位符、默认 iMF 和 2-node placement |
| 2 | [`rlinf/config.py::_validate_flow_policy_v2_cfg`](../../rlinf/config.py) | objective/profile/sampler/pretrained/Franka 合同如何 fail-fast |
| 3 | [`examples/embodiment/train_embodied_agent.py::main`](../../examples/embodiment/train_embodied_agent.py) | `loss_type=embodied_sac` 如何选择 `EmbodiedSACFSDPPolicy` |
| 4 | [`rlinf/runners/embodied_runner.py::EmbodiedRunner`](../../rlinf/runners/embodied_runner.py) | 权重同步、env/rollout/actor channel、训练、评估和保存顺序 |
| 5 | [`rlinf/workers/actor/fsdp_sac_policy_worker.py::EmbodiedSACFSDPPolicy`](../../rlinf/workers/actor/fsdp_sac_policy_worker.py) | actor-only load、anchor/target/alpha/replay 初始化和三类 loss |
| 6 | [`rlinf/workers/actor/sacflow_finetune.py`](../../rlinf/workers/actor/sacflow_finetune.py) | warm-up/online phase、UTD credit、anchor loss 和 common noise |
| 7 | [`rlinf/models/embodiment/flow_policy/flow_policy.py::FlowPolicy`](../../rlinf/models/embodiment/flow_policy/flow_policy.py) | `SAC`/`SAC_Q` dispatch、rollout/eval sampler 选择、三种 log-prob 合同 |
| 8 | [`rlinf/models/embodiment/modules/flow_actor.py::FlowTActor`](../../rlinf/models/embodiment/modules/flow_actor.py) | `sample_path()`、固定 trace 重算、RF/iMF field 调用 |
| 9 | [`rlinf/models/embodiment/modules/flow_sampler.py`](../../rlinf/models/embodiment/modules/flow_sampler.py) | time grid、common random numbers、joint path sum 和 tanh Jacobian |
| 10 | [`rlinf/models/embodiment/modules/flow_transition.py`](../../rlinf/models/embodiment/modules/flow_transition.py) | RF corrected SDE 与 iMF interval-SDE 的 transition moments |
| 11 | [`rlinf/data/embodied_buffer_dataset.py::ReplayBufferDataset`](../../rlinf/data/embodied_buffer_dataset.py) | online/demo batch 混采比例 |
| 12 | [`rlinf/data/replay_buffer.py::TrajectoryReplayBuffer`](../../rlinf/data/replay_buffer.py) | trajectory 存储、frame sampling、checkpoint/resume |
| 13 | [`rlinf/utils/flow_actor_checkpoint.py::load_flow_actor_checkpoint`](../../rlinf/utils/flow_actor_checkpoint.py) | 为什么 BC/online objective、动作和模型结构必须一致 |
| 14 | [`tests/unit_tests/test_sacflow_frozen_anchor_support.py`](../../tests/unit_tests/test_sacflow_frozen_anchor_support.py)、[`test_flow_t_actor_v2.py`](../../tests/unit_tests/test_flow_t_actor_v2.py) | phase、loss、共同噪声、SDE 和梯度合同的可执行示例 |

异步路径只需要在理解同步 worker 后再读：

- [`examples/embodiment/train_async.py`](../../examples/embodiment/train_async.py)
- [`rlinf/workers/actor/async_fsdp_sac_policy_worker.py::AsyncEmbodiedSACFSDPPolicy`](../../rlinf/workers/actor/async_fsdp_sac_policy_worker.py)
- [`rlinf/runners/async_embodied_runner.py`](../../rlinf/runners/async_embodied_runner.py)

异步 worker 复用同一个 `SACFlowFinetuneController` 和 `update_one_epoch()`，不会另定义一套 warm-up 规则。

## 3. 完整运行调用链

同步入口的单个 global step 是：

```text
train_embodied_agent.py
  → validate_cfg()
  → actor = EmbodiedSACFSDPPolicy
  → rollout = MultiStepRolloutWorker
  → env = EnvWorker
  → EmbodiedRunner.init_workers()
       ├─ rollout/env init
       └─ actor.init_worker()
            ├─ setup_model_and_optimizer(initialize_target=True)
            ├─ setup_sac_components()
            └─ hard-copy target (tau=1)

EmbodiedRunner.run()
  → sync live actor weights to rollout
  → env.interact() ↔ rollout.generate()
  → actor.recv_rollout_trajectories()
  → actor.run_training()
       ├─ claim UTD phases
       └─ update_one_epoch(phase)
            ├─ forward_critic()
            ├─ forward_actor()
            ├─ forward_alpha()       # warm-up 时跳过
            └─ soft_update_target_model()
  → optional evaluation/checkpoint
```

rollout worker 调用 `FlowPolicy.predict_action_batch(mode="train")`，因此使用 `flow_sampling.online_rollout`。评估传 `mode="eval"`，因此使用 `flow_sampling.evaluation`。

## 4. 初始化阶段的模块职责

### 4.1 actor-only checkpoint load

fresh online run 中，`_setup_sacflow_finetune_modules()` 在 FSDP wrap 之前执行：

1. 构造含 `q_head` 的完整 online `FlowPolicy`；
2. 从 online config 生成 expected manifest metadata；
3. 用 `load_flow_actor_checkpoint()` 只加载 encoder/projection/FlowTActor/action buffers；
4. 保持 `q_head` 的随机初始化；
5. 若使用 `flow_noise`，允许 BC artifact 缺少 online-only `flow_noise_head`；
6. 把 live module 完整复制给 target module；
7. deepcopy live module 得到 frozen full-policy anchor。

这意味着初始化完成时：

- live actor 和 anchor 的 action path 完全一致；
- live Q 与 target Q 完全一致，但来自 fresh random initialization；
- anchor 虽然是 full policy 副本，训练时只使用其 action 输出；
- anchor 不进入 FSDP、不进入 optimizer、不参与 target EMA；
- full online resume 时不再执行 actor-only 初始化。

### 4.2 optimizer ownership

frozen-anchor 模式将参数明确分成两组：

```text
actor optimizer:
  encoders + state_proj + mix_proj + flow_actor

critic optimizer:
  q_head only
```

critic forward 设置 `detach_encoder=True`。这避免 actor observation encoder 同时被两个 optimizer 拥有，也避免 critic loss 在 warm-up 中破坏 pretrained actor representation。

配置必须使用：

```yaml
actor:
  fsdp_config:
    use_orig_params: true
```

worker 会检查参数遗漏、optimizer 重叠、actor 中误含 Q 参数或 critic 中误含非 Q 参数。

### 4.3 critic、target 和 alpha

默认初始化：

- critic：fresh `q_head`，`num_q_heads=10`；
- target：live policy 的 hard copy；
- target 后续：每 `target_update_freq` 次 update 做 `tau` EMA；
- alpha：`EntropyTemperature(initial_alpha=0.2, alpha_type="softplus")`；
- warm-up：不执行 alpha optimizer step；
- online：alpha 根据 detached temperature log-prob 更新。

`target_entropy=0` 针对的是 augmented path density，不是传统单步高斯动作分布的 `-action_dim` 默认值。

## 5. Replay、demo 与 phase 时钟

### 5.1 两个 buffer

online 配置使用：

```text
online replay
  - fresh empty buffer
  - 接收每次真实环境 rollout
  - auto_save=true

demo buffer
  - 从 <collection_log>/demos 加载
  - auto_save=false
  - 在线有人为 intervention 时也可追加 intervention trajectory
```

`ReplayBufferDataset` 每个 batch 同时从两个 buffer 采样。默认：

```text
demo_fraction = 0.5
global batch = 50% online + 50% demo
```

`demo_fraction` 是 batch 中 demonstration 样本比例，不是“以该概率选择整个 buffer”。当前 global batch 为 256，因此得到精确的 128/128。若 batch size 为奇数且比例恰为 0.5，为兼容旧行为会从两侧各取 `batch_size // 2`，总数少 1。

### 5.2 `min_buffer_size` 的单位

`TrajectoryReplayBuffer.is_ready()` 当前比较的是 trajectory 数量，即 `len(buffer)`，不是 transition/frame 数量。

因此：

```yaml
replay_buffer:
  min_buffer_size: 2
demo_buffer:
  min_buffer_size: 1
```

分别表示至少 2 条 online trajectory 和 1 条 demo trajectory。不要把这里误解为 2 个 transition。

`sample_window_size` 控制从最近多少条 trajectory 的 frame 中采样；`0` 表示 demo buffer 使用全部历史。`cache_size` 控制内存缓存容量，已经与 `sample_window_size` 解耦。

### 5.3 phase 和 UTD

`SACFlowFinetuneController` 只根据新收到的 online transition 数推进时钟。demo frame 不计入 `online_transitions`。

每收到 `N` 个 online transitions，会增加：

```text
N × updates_per_transition
```

个 update credit。sync/async worker 都从该 credit 队列领取 update phase。即使一次接收跨过 10k 阈值，阈值前对应的 optimizer update 仍被标记为 warm-up，不会整批跳到 online。

当前 config validator 固定要求：

```yaml
updates_per_transition: 1
critic_actor_ratio: 1
```

即每个 transition 对应一次 critic update和一次 actor update。

## 6. Warm-up 与 Online Loss

### 6.1 共同噪声动作正则

每个 actor update 都创建：

- `initial_noise: [B, action_dim]`
- `step_noises`: K 个 `[B, action_dim]` tensor

live actor 和 frozen anchor 复用同一对象、同一 sampler method 和同一 K。行为正则是：

```text
L_anchor = MSE(a_live(s, noise), stop_grad(a_anchor(s, noise)))
```

它约束最终环境动作，而不是 field 参数或 replay action。`common_noise=true` 是硬要求。

### 6.2 warm-up

当 online transition 数小于 `warmup_transitions`：

```text
L_actor = warmup_behavior_beta × L_anchor
```

同时：

- critic 继续做标准 SAC Bellman update；
- target critic 继续 EMA；
- actor optimizer 更新整个 live observation-to-action path；
- actor loss不含 Q、不含 entropy；
- alpha 固定为初始化值。

这不是 actor freeze。actor 会被 anchor loss 更新，只是被限制在 pretrained behavior 附近。

### 6.3 online

阈值之后：

```text
L_sac = mean(alpha * log_p_path - aggregate(Q(s, a_live)))
L_actor = L_sac + behavior_beta * L_anchor
```

默认 `actor_agg_q=mean`，即 actor 对 10 个 Q head 取平均；critic target 默认 `agg_q=min`，并可由 `critic_subsample_size=2` 随机选择部分 head 后取 min。

critic target：

```text
y = reward + gamma * (min/mean target_Q(s', a') - alpha * log_p_path(a'|s'))
```

遇到 termination 时按 `bootstrap_type=standard` 停止 bootstrap；truncation 与 termination 的区分来自 `demos/` 和 online trajectory。

### 6.4 alpha loss

online phase 使用：

```text
L_alpha = -alpha * (mean(detached_log_p_path) + target_entropy)
```

当前 `target_entropy=0`。warm-up 不调用 alpha optimizer。

## 7. Flow sampling 与 log-prob 设计

### 7.1 三个使用场景

配置把 sampler 分成三个 section：

| section | 调用者 | 默认 | 要求 |
|---|---|---|---|
| `actor_update` | critic bootstrap、actor、alpha | `flow_sde` | 必须随机，支持 path log-prob |
| `online_rollout` | 真机数据采集 | `flow_sde` | 必须随机，提供探索 |
| `evaluation` | validation/eval | `flow_ode` | 必须确定性 |

`actor_update` 与 `online_rollout` 可以配置不同 K 或随机方法，但 live/anchor 比较始终共享 actor-update section 的 method、K 和 noise。

### 7.2 RF 与 iMF 时间方向

| Objective | schedule | 每步 field |
|---|---|---|
| RF | `0 → 1` | 瞬时速度 `v(z,t_from)` |
| iMF | `1 → 0` | 区间平均场 `U(z,t_from,t_to)` |

iMF 的 ODE、flow-noise 和 flow-SDE 每步只调用一次相邻区间 `U`，不会在 sampler 中调用 `U(z,t,t)`。

### 7.3 iMF interval-SDE

`improved_meanflow_sde_moments()` 在本地 PyTorch 模块中复写 OpenPI MeanFlow transition 公式，但没有 runtime import 或代码共享。它使用：

- interval field `U(z,t_from,t_to)`；
- `noise_level` 控制 drift/variance 强度；
- `safe_initial_time` 处理第一步 `t=1` 的对数奇点；
- `noise_std_range` 对 next-state std 做上下界裁剪；
- FP32 执行 log calculation，再转回模型 dtype。

为保证第一步区间合法，必须满足：

```text
safe_initial_time > 1 - 1 / num_steps
```

该条件会对每个使用 flow-SDE 的 section 单独校验。

### 7.4 joint path log-prob

随机路径的密度为：

```text
log p(path)
  = log p(initial_noise)
  + sum_k log p(z_{k+1} | z_k)
  - log |det J_tanh,scale|
```

这是 path 维度的 sum，不是 OpenPI PPO 路径中的 mean。它是 augmented Markov path density，不应表述为解析的 marginal action density。

### 7.5 三种 log-prob 模式

`FlowPolicy` 明确区分：

| mode | 用途 | 梯度语义 |
|---|---|---|
| `path` | 数值/调试 | 当前 reparameterized path 的联合密度 |
| `temperature` | critic target、alpha、anchor/warm-up | detached 数值，避免无意义的额外 field forward |
| `actor_surrogate` | online actor entropy term | detach sampled trace 后重算 transition density，给 drift/field 提供 score gradient |

actor 的 Q gradient仍穿过完整 reparameterized K-step sample path。固定 trace 的 entropy surrogate只为 `alpha * log_p_path` 提供正确的 score-gradient 方向。

## 8. 你需要修改的配置

以 [`franka_sacflow_online_finetune.yaml`](../../examples/embodiment/config/franka_sacflow_online_finetune.yaml) 为模板。

### 8.1 真正必须替换的占位符

| 配置键 | 必须填写什么 |
|---|---|
| `cluster.node_groups[].hardware.configs[].robot_ip` | Franka 控制 IP |
| `env.train.override_cfg.target_ee_pose` | 任务目标末端位姿 |
| `actor.model.model_path` | 与 BC 相同的 ResNet checkpoint 目录 |
| `rollout.model.model_path` | 通常与 actor 的 `model_path` 相同 |
| `actor.model.pretrained_actor.path` | BC 输出的 `global_step_N/actor/flow_actor` |
| `algorithm.demo_buffer.load_path` | 同一次采集产生的 `<collection_log>/demos` |
| `cluster.node_groups` / placement | 实际 GPU 主机和 Franka 控制机拓扑 |

同时确认：

```yaml
env:
  train:
    no_gripper: false
  eval:
    no_gripper: false
```

当前 v2 是 7D gripper-inclusive 合同，任一 split 配置 `no_gripper: true` 都会在 config validation 阶段失败。

### 8.2 pretrained checkpoint 与 resume

fresh run：

```yaml
runner:
  resume_dir: null
actor:
  model:
    pretrained_actor:
      path: /results/.../global_step_N/actor/flow_actor
      load_mode: actor_only
      require_manifest: true
      strict_actor: true
```

完整恢复：

```yaml
runner:
  resume_dir: /results/.../checkpoints/global_step_M
```

`resume_dir` 优先。它恢复 live policy、actor/critic optimizer、scheduler、target、alpha、online replay、frozen anchor、phase controller和 `update_step`。不要把 resume 当作重新读取 BC artifact；恢复时保存的 anchor 和 phase 是训练语义的一部分。

### 8.3 SAC 主参数

| 配置键 | 默认值 | 作用 |
|---|---:|---|
| `algorithm.gamma` | `0.99` | Bellman discount |
| `algorithm.tau` | `0.005` | target EMA 系数 |
| `algorithm.agg_q` | `min` | critic bootstrap 时如何聚合 Q heads |
| `algorithm.actor_agg_q` | `mean` | actor loss 中如何聚合 Q heads |
| `algorithm.critic_subsample_size` | `2` | bootstrap 前随机选择多少个 Q heads；小于等于 0 表示全用 |
| `algorithm.critic_actor_ratio` | `1` | 当前 frozen-anchor 路径固定要求 1 |
| `algorithm.backup_entropy` | `true` | critic target 是否减去 `alpha*log_p_path` |
| `algorithm.target_update_freq` | `1` | 每多少 optimizer update做一次 target EMA |
| `entropy_tuning.initial_alpha` | `0.2` | fresh alpha 初值 |
| `entropy_tuning.target_entropy` | `0` | augmented path entropy target |
| `entropy_tuning.optim.lr` | `3e-4` | alpha optimizer 学习率 |

### 8.4 fine-tuning phase 参数

```yaml
algorithm:
  sacflow_finetune:
    enabled: true
    warmup_transitions: 10000
    warmup_behavior_beta: 1000.0
    behavior_beta: 1000.0
    demo_fraction: 0.5
    updates_per_transition: 1
    common_noise: true
```

| 参数 | 作用 | 调参风险 |
|---|---|---|
| `warmup_transitions` | warm-up 使用的在线 transition 数，不含 demo | 太小会让未校准 critic 很快影响 actor；真机首轮不要激进缩短 |
| `warmup_behavior_beta` | warm-up anchor MSE 权重 | 太小会让 BC policy提前漂移；太大可能几乎无有效更新 |
| `behavior_beta` | online 阶段持续使用的 anchor MSE 权重 | 默认 1000；降低前监控 success、anchor loss 和动作偏移 |
| `demo_fraction` | 每个训练 batch 中 demo 的比例 | `0.5` 是默认安全基线；提高会降低最新 online data 的占比 |
| `updates_per_transition` | UTD | 当前必须为 1 |
| `common_noise` | live/anchor 是否共享完整随机路径 | 当前必须为 true |

### 8.5 buffer 参数

| 配置键 | 默认值 | 说明 |
|---|---:|---|
| `replay_buffer.cache_size` | `2000` | online trajectory 内存缓存容量 |
| `replay_buffer.min_buffer_size` | `2` | 启动训练前所需 online trajectory 数 |
| `replay_buffer.sample_window_size` | `2000` | 只从最近这些 trajectory 的 frame 中采样 |
| `replay_buffer.auto_save` | `true` | online trajectory 异步落盘 |
| `demo_buffer.cache_size` | `2000` | demo trajectory cache 容量，必须大于 0 |
| `demo_buffer.min_buffer_size` | `1` | 所需 demo trajectory 数 |
| `demo_buffer.sample_window_size` | `0` | 0 表示从所有 demo 历史采样 |
| `demo_buffer.auto_save` | `false` | 静态 demo 不重复写到新的 auto-save 目录 |

多 rank 加载会按 trajectory 分配 demo。demo trajectory 数应至少覆盖 actor world size，否则某些 rank 可能没有可采数据；启动真机前先检查每个 rank 的 demo buffer stats。

### 8.6 sampler 参数

默认 iMF：

```yaml
flow_sampling:
  implementation: pytorch_flow_t_v2
  profile: imf_openpi_interval_sde_v1
  actor_update:
    method: flow_sde
    num_steps: 4
  online_rollout:
    method: flow_sde
    num_steps: 4
  evaluation:
    method: flow_ode
    num_steps: 4
  flow_sde:
    field_source: interval_average
    noise_level: 0.10
    noise_std_range: [0.005, 0.05]
    safe_initial_time: 0.99
    joint_path_logprob: true
```

| 参数 | 作用 | 建议 |
|---|---|---|
| `num_steps` | flow transition 数 K | 增大可提高积分分辨率，但 rollout/actor 计算近似线性增加 |
| `noise_level` | SDE drift correction 与原始方差强度 | 真机从小值开始；不是最终 std 的唯一决定因素 |
| `noise_std_range[0]` | next-state std 下界 | 必须大于 0，过大会导致策略无法低噪声运行 |
| `noise_std_range[1]` | next-state std 上界 | 真机主要安全旋钮之一 |
| `safe_initial_time` | 第一段 `t=1` 的安全替代值 | 默认 0.99；必须满足和 K 相关的动态约束 |
| `joint_path_logprob` | 是否使用 prior + 所有 transition + Jacobian | 当前必须为 true |

若使用 `flow_noise`，需把对应 section 的 `method` 改为 `flow_noise`，并提供：

```yaml
flow_noise:
  noise_std_range: [0.005, 0.05]
```

`flow_noise` 的 std 来自 online-only `ExploreNoiseNet`；BC checkpoint 不含该 head。它会被零初始化为跨 rank 一致的确定初始状态。

### 8.7 模型与 optimizer 参数

BC 与 online 必须完全一致：

- `flow_actor_type`
- `flow_matching` objective 和双时间合同
- `image_size/image_num/state_dim/action_dim`
- `d_model/n_head/n_layers`
- `action_scale`
- encoder checkpoint 架构

online 特有：

| 配置键 | 默认值 | 作用 |
|---|---:|---|
| `add_q_head` | `true` | 构造 critic |
| `num_q_heads` | `10` | Q ensemble 数 |
| `actor.optim.lr` | `3e-4` | live encoder + Flow actor 学习率 |
| `actor.critic_optim.lr` | `3e-4` | q_head-only 学习率 |
| 两个 `clip_grad` | `1.0` | actor/critic 各自裁剪 |

## 9. 切换 RF 与其他 sampler

### 9.1 从默认 iMF 切到 RF

BC 和 online 的 objective 必须一起改。online 配置应使用完整 RF block：

```yaml
flow_matching:
  implementation: pytorch_flow_t_v2
  objective: rectified_flow
  action_transform: tanh_latent

flow_sampling:
  implementation: pytorch_flow_t_v2
  profile: rf_sacflow_sde_v1
  actor_update:
    method: flow_sde
    num_steps: 4
  online_rollout:
    method: flow_sde
    num_steps: 4
  evaluation:
    method: flow_ode
    num_steps: 4
  flow_sde:
    field_source: instantaneous_velocity
    noise_level: 0.10
    joint_path_logprob: true
```

RF corrected SDE 的 std 直接由 `noise_level * sqrt(dt)` 产生，因此必须删除 iMF-only：

- `flow_sde.noise_std_range`
- `flow_sde.safe_initial_time`

并把 `pretrained_actor.path` 改为 RF BC artifact。iMF checkpoint 不能加载到 RF。

### 9.2 为什么 evaluation 不能改成随机 sampler

validator 固定要求 `evaluation.method=flow_ode`，以便：

- checkpoint 前后做确定性动作对比；
- 把策略质量变化与探索噪声分离；
- 真机 validation 不意外注入 online SDE 噪声。

### 9.3 为什么 online rollout 不能用 ODE

online SAC 需要随机交互。validator 会拒绝 `online_rollout.method=flow_ode` 和 `actor_update.method=flow_ode`，因为确定性 transition 没有可用于 SAC entropy 的正规 Gaussian path density。

## 10. 如何启动

### 10.1 两节点真机准备

示例把 actor/rollout 放在 GPU 节点 0，把 Franka env 放在控制节点 1。必须在每个节点启动 Ray **之前** 设置不同的：

```bash
export RLINF_NODE_RANK=0  # GPU/head
# 另一台控制机使用 1
```

然后启动 head/worker Ray，并只在 head 节点启动训练。确保 `cluster.num_nodes`、`node_ranks`、robot `node_rank` 与实际拓扑一致。

### 10.2 同步训练

示例配置默认对应同步入口：

```bash
bash examples/embodiment/run_embodiment.sh franka_sacflow_online_finetune
```

该脚本会创建时间戳日志目录，并覆盖 `runner.logger.log_path`。

### 10.3 异步训练

同一个 fine-tune controller 已接入 async worker。如果你的真机部署已经使用异步 runner，可运行：

```bash
export EMBODIED_PATH="$(pwd)/examples/embodiment"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
bash examples/embodiment/run_realworld_async.sh franka_sacflow_online_finetune
```

异步模式会改变 env/rollout/actor 的调度与数据到达时机，但不会改变 phase、loss 和 UTD 语义。第一次真机联调建议先用同步模式减少并发变量。

## 11. 日志、checkpoint 与恢复

### 11.1 重点监控指标

| 指标 | 含义 |
|---|---|
| `train/finetune/phase` | `0=warmup`，`1=online` |
| `train/finetune/online_transitions` | phase 时钟，只累计 online transitions |
| `train/finetune/update_credit` | 尚未消费的 optimizer update credit |
| `train/finetune/completed_updates` | 已完成的 transition-budgeted update 数 |
| `train/actor/anchor_loss` | live/anchor 动作 MSE |
| `train/actor/sac_loss` | warm-up 为 0，online 为 entropy-Q 部分 |
| `train/sac/actor_loss` | 加权后的总 actor loss |
| `train/sac/critic_loss` | Q Bellman MSE |
| `train/sac/alpha` | 当前 entropy temperature |
| `train/actor/entropy` | `-log_p_path` 的 batch 均值 |
| `train/actor/q_pi` | 当前 sampled action 的聚合 Q |
| `train/replay_buffer/total_samples` | online replay transition 数 |
| `train/demo_buffer/total_samples` | demo transition 数 |
| `env/success_once` | episode 是否至少成功一次 |

真机上不要只看 reward。同步监控动作幅值、anchor loss、SDE std clamp、控制频率、termination 原因和机器人安全状态。

### 11.2 online checkpoint 内容

保存目录：

```text
<log_path>/<experiment_name>/checkpoints/global_step_M/actor/
├─ model / optimizer / scheduler shards
└─ sac_components/
   ├─ alpha/
   ├─ target_model/checkpoint_rank_N.pt
   ├─ replay_buffer/rank_N/
   ├─ anchor/checkpoint_rank_N.pt
   └─ sacflow_finetune/state_rank_N.pt
```

`sacflow_finetune/state_rank_N.pt` 保存 `update_step` 和 controller state。恢复时会检查当前 `warmup_transitions`、UTD 和已保存 state 一致；不要在 resume 时静默更改 phase schedule。

demo buffer 默认不复制进 online checkpoint。恢复时仍按 `algorithm.demo_buffer.load_path` 重新加载静态 demo；确保该目录持久可用且内容没有被替换。

## 12. 真机操作前的安全门槛

建议按下面顺序逐级放行：

1. **配置 dry-run**：Hydra composition 和 `validate_cfg()` 通过，manifest 加载成功。
2. **checkpoint 一致性**：固定 observation、initial noise 和 step noise，验证 BC actor、live actor、frozen anchor 初始动作一致。
3. **dataset/replay dry-run**：从 `demos/` 分布式加载并采样，检查 7D action、reward、termination、next observation。
4. **dummy env**：确认 online rollout 随机、evaluation ODE 确定、warm-up alpha 不更新。
5. **硬件在环**：断开真实运动输出，记录 action、SDE std、控制周期和限幅。
6. **低风险真机 smoke**：限制 workspace/速度/力矩，保留急停，使用小 `noise_level` 和较低 std 上界。
7. **短 warm-up 检查**：确认 anchor loss、动作偏移、Q target 和 replay 增长合理后，再延长运行。

不要把论文附录中的仿真超参数直接视为真机已经验证的安全参数。

## 13. 常见问题

### checkpoint metadata mismatch

逐项比较 BC 与 online：objective、action range、7D/19D、image size、camera count、`d_model/n_head/n_layers` 和 iMF time fusion。不要使用 `strict_actor=false`；validator 和 loader都会拒绝。

### online 一直不训练

依次检查：

1. online replay 是否已有 `min_buffer_size` 条 trajectory；
2. demo buffer 是否已有 `min_demo_buffer_size` 条 trajectory；
3. `finetune/update_credit` 是否大于等于 1；
4. 多 rank 中最慢 rank 是否没有收到 trajectory。

### phase 看起来跨得太早或太晚

phase 按 transition 数，不按 episode、runner global step 或 demo 数。查看 `finetune/online_transitions`，不要用 `global_step` 推断。

### anchor loss 很大

首先验证 `common_noise=true`，然后固定一组 noise 对比初始 live/anchor。若训练开始前已不一致，通常是 checkpoint、image normalization、sampler K/method 或 action scale不一致，而不是 beta 太小。

### rollout 没有随机性

确认 `flow_sampling.online_rollout.method` 是 `flow_sde` 或 `flow_noise`，并检查 rollout 调用传入 `mode="train"`。evaluation 的 ODE 确定性是预期行为。

### iMF-SDE 出现 NaN 或 std 总在边界

检查：

- `safe_initial_time > 1 - 1/num_steps`
- `noise_level > 0`
- `0 < std_min <= std_max`
- 输入 action/observation finite
- FP16/BF16 下是否仍由 transition kernel 的 FP32 log 路径处理

### resume 缺 anchor 或 phase state

frozen-anchor resume 必须同时存在 `sac_components/anchor` 和 `sac_components/sacflow_finetune`。不要用旧的普通 SAC checkpoint伪装成该模式的完整 resume。

## 14. 修改代码时的最小验证集

修改 worker、sampler、buffer、checkpoint 或 config 后，至少运行：

```bash
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_flow_config_v2.py \
  tests/unit_tests/test_flow_actor_checkpoint.py \
  tests/unit_tests/test_flow_policy_v2.py \
  tests/unit_tests/test_flow_t_actor_v2.py \
  tests/unit_tests/test_flow_replay_checkpoint.py \
  tests/unit_tests/test_sacflow_frozen_anchor_support.py
```

并执行范围保护：

```bash
git diff --exit-code -- \
  rlinf/models/embodiment/openpi/openpi_action_model.py
```

设计职责建议保持：

- phase/UTD/common-noise/anchor loss：`sacflow_finetune.py`；
- SAC optimizer/update/checkpoint 编排：`fsdp_sac_policy_worker.py`；
- policy forward mode 与 sampler section：`flow_policy.py`；
- field 网络：`FlowTActor`；
- schedule、trace 与 path density：`flow_sampler.py`；
- 纯 transition moments：`flow_transition.py`；
- replay/demo 混采：`embodied_buffer_dataset.py`；
- 不修改或 runtime import `openpi_action_model.py`；
- 不给 `JaxFlowTActor` 增加 v2 分支。

如果你还没有生成 portable actor checkpoint，请先读 [`flow_bc.md`](flow_bc.md)。
