# OpenPI RLT：带 Action Chunking 的 Franka 真机强化微调详解

本文基于当前 RLinf 实现，说明 OpenPI RLT 如何把视觉语言动作模型、action
chunking、Franka 真机执行、人工接管、trajectory replay buffer 和 off-policy
actor-critic 串成一条在线强化学习链路。

本文重点回答四个问题：

1. Stage 1 的 OpenPI 特征模型与 Stage 2 的强化学习策略分别训练什么？
2. 20 步 reference chunk、10 步 actor chunk 和单步 Franka 控制如何对齐？
3. buffer 如何保存完整 action chunk、逐步 reward/done 和实际人工接管动作？
4. Franka 没有 `multistep()` API 时，环境如何执行、聚合和对齐多步 transition？

除非特别说明，shape 和配置值均来自当前示例
[rlt_stage2_ac_mlp.yaml](../../examples/embodiment/config/rlt_stage2_ac_mlp.yaml)。
文中的“真机强化微调”是指冻结 Stage 1 OpenPI + RLT 特征模型、在线训练独立的
Stage 2 MLP actor 和 twin-Q critic，不是用 RL loss 更新 OpenPI 主干参数。

> 真机语义提示：当前实现以完整 10 步 chunk 为一个宏观 transition。chunk
> 内首次出现 termination 或 truncation 后不会提前停止，剩余动作仍会发给真机。
> 这是算法语义和安全边界的一部分，不能按普通单步 Gym 环境理解。

## 1. 核心结论

整条链路可以压缩为：

```text
Stage 1 demonstrations
  -> OpenPI action SFT
  -> detached OpenPI prefix hidden states
  -> RLT token autoencoder
  -> Stage 1 checkpoint

Stage 2 raw observation s_t
  -> frozen Stage 1 model
     -> z_rl + proprio + OpenPI reference chunk
  -> Stage 2 MLP actor
     -> actor action chunk
  -> choose reference chunk or actor chunk
  -> optional substep-level SpaceMouse override
  -> Franka executes H primitive actions
  -> one chunk reward/done + next boundary observation
  -> one macro-transition in trajectory replay
  -> twin-Q TD update + Q/BC actor update
  -> sync updated Stage 2 actor to rollout worker
```

关键设计点如下：

- 冻结的 Stage 1 模型把原始 observation 转换为 `z_rl`、`proprio` 和
  reference action，Stage 2 不直接从 replay 中重新编码图像。
- Stage 2 将 `H=10` 个 7D 动作视作一个 70D 宏观动作。critic 对整个 70D
  chunk 给出一对标量 Q 值。
- 环境依然保留 chunk 内每个物理子步的 reward、termination 和 truncation，
  训练时先做 chunk 内折扣求和，再用 `gamma ** H` bootstrap。
- policy switch 是整 chunk 生效；SpaceMouse intervention 是物理子步级生效。
- replay 中保存的是选择和人工接管之后实际送入环境的动作。
- buffer 以 trajectory 为存储和窗口单位，以宏观 transition 为训练 sample。

## 2. 两阶段训练边界

### 2.1 Stage 1：OpenPI SFT 与 RLT 表征学习

Stage 1 使用
[rlt_stage1_sft_openpi_pi05.yaml](../../examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml)
在同一批示范数据上计算两个目标：

```text
L_stage1 = L_rlt + rlt_alpha * L_vla
```

- `L_vla` 是 OpenPI flow-matching action loss，更新 OpenPI 模型。
- `L_rlt` 是 RLT token transformer 对 OpenPI prefix hidden states 的
  mask-aware reconstruction loss，更新 RLT encoder/decoder。

这里有一个容易忽略的梯度边界：OpenPI prefix hidden states 在送入 RLT module
前会 `detach()`，reconstruction target 也会 detach。因此 `L_rlt` 不会反向
更新 OpenPI 主干。所谓“联合训练”是同一 Stage、同一 batch 同时优化两组目标，
不是两个 loss 都作用到所有参数。

当前 RLT token transformer 将 `[B, S, 2048]` 的 prefix states 压缩成一个
2048D RL token：

```text
prefix hidden states [B, S, 2048]
  -> RLT encoder
  -> one RL token [B, 1, 2048]
  -> encode_flat()
  -> z_rl [B, 2048]
```

`rlt_prefix_seq_len: 1024` 是允许的最大 prefix 长度，不代表每个 observation
固定产生 1024 个 token。`rlt_image_only: False` 时保留图像和语言 prefix，
`rlt_use_mask: True` 时 reconstruction 与编码都使用 padding mask。

### 2.2 Stage 2：冻结特征模型，训练 Chunk Actor-Critic

rollout worker 会单独构造 `rollout.rlt_feature_model`，随后调用 `eval()` 并将
所有参数设为 `requires_grad_(False)`。每个 chunk 边界调用
`extract_rlt_obs()`，得到：

```text
rlt_obs = {
  z_rl:      learned state representation,
  proprio:   selected raw robot state,
  ref_chunk: OpenPI reference actions,
}
```

Stage 2 只更新 `rlt_mlp_policy` 的 actor、twin-Q heads 及其 target network。
OpenPI、RLT token transformer 和 OpenPI normalization stats 在在线训练期间保持
不变。

Stage 1 checkpoint 推荐配置为：

```yaml
rollout:
  rlt_feature_model:
    model_path: /path/to/global_step_N/actor
```

使用 `global_step_N/actor` 而不是只写 `global_step_N` 更稳妥：权重 loader
虽然可以从 checkpoint 根目录继续查找 actor 权重，但 OpenPI normalization
stats 默认保存在 actor 子目录的 asset 路径下。Stage 2 的
`rollout.model.model_path` 和 `actor.model` 都不是 Stage 1 加载入口。

## 3. 三个时间尺度

当前配置同时存在三个不同的 horizon：

| 时间尺度 | 默认值 | 配置/接口 | 作用 |
|---|---:|---|---|
| OpenPI reference horizon `H_ref` | 20 | `ref_num_action_chunks`、`openpi.action_chunk` | 每个边界生成的 reference action 数 |
| Stage 2 execution horizon `H` | 10 | `actor.model.num_action_chunks` | actor 输出和环境实际执行的 chunk 长度 |
| Franka primitive step | 1 | `FrankaEnv.step(action)` | 一次接收一个 7D action |

单步 7D action 为：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

`H_ref=20` 大于 `H=10` 是有意的。完整的 20 步 `ref_chunk` 会进入
RLT observation 并写入 buffer；actor 只消费前 10 步，切换前也只执行前 10
步。额外的 reference actions 不会在同一次环境交互中自动执行。

`EnvWorker` 通过模型配置解析 `H`，并要求：

```text
max_steps_per_rollout_epoch % H == 0
```

当前训练配置 `max_steps_per_rollout_epoch=300`，所以一个 rollout epoch
包含：

```text
300 physical steps / 10 steps per chunk = 30 macro-transitions
```

这里的 300 是物理子步数，不是 300 个 action chunks。

## 4. Tensor Shape 与模型输入

定义：

- `B`：并行环境数，当前每个 RealWorldEnv worker 只支持 `B=1`。
- `T`：一个 rollout trajectory 容器中的宏观 chunk transition 数，当前通常为
  30。它不等同于一个 environment episode。
- `N`：一次 replay sampling 返回的训练 batch size。
- `A`：单步 action dimension，当前为 7。

主要 shape 如下：

| 数据 | rollout 边界 shape | trajectory shape | sampled shape |
|---|---|---|---|
| `z_rl` | `[B, 2048]` | `[T, B, 2048]` | `[N, 2048]` |
| `proprio` | `[B, 19]` | `[T, B, 19]` | `[N, 19]` |
| `ref_chunk` | `[B, 20, 7]` | `[T, B, 20, 7]` | `[N, 20, 7]` |
| selected/executed chunk | `[B, 10, 7]` | 展平为 `[T, B, 70]` | `[N, 70]` |
| `intervene_flags` | 原始 `[B, 10]` | 扩到 action coordinate 后为 `[T, B, 70]` | `[N, 70]` |
| rewards | `[B, 10]` | `[T, B, 10]` | `[N, 10]` |
| terminations/truncations | `[B, 10]` | 对齐后 `[T, B, 10]` | `[N, 10]` |

当前 19D proprio 来自 wrapper 处理后的完整 Franka state。base Franka state
使用 7D quaternion pose，总计 20D；`RelativeFrame` 和 `Quat2EulerWrapper`
处理后 pose 变为 6D，最终 state 为 19D。配置中的
`state_indices: []` 表示不做子集选择；如果修改 state layout 或
`state_indices`，必须同步修改 `actor.model.proprio_dim`。

当前 wrapper 后的 19D state 由 1D gripper、3D TCP force、6D TCP pose
（位置 + Euler rotation）、3D TCP torque 和 6D TCP velocity 拼接而成。

### 4.1 Actor 输入

`RLTMLPPolicy` 取 reference 的前 10 步并展平：

```text
actor state =
  flatten(ref_chunk[:, :10])  # 10 * 7 = 70
  + z_rl                      # 2048
  + proprio                   # 19

actor input dimension = 70 + 2048 + 19 = 2137
actor output dimension = 10 * 7 = 70
```

基类内部将它视作一个 70D action，而不是 10 个独立 policy calls。输出最终
reshape 为 `[B, 10, 7]` 交给环境。

训练时 actor 是 fixed-std pre-tanh Normal，默认 `fixed_std=0.002`；训练 rollout
使用 reparameterized sample，eval 使用 mean，二者最后都经过 `tanh`。

### 4.2 Critic 输入

critic state 不包含 reference chunk：

```text
critic state = z_rl + proprio             # 2048 + 19 = 2067
critic action = flatten(action_chunk)      # 70
Q(state, action_chunk) -> [Q1, Q2]
```

因此 reference 只负责引导 actor 和提供 BC target，不作为 critic 估值的显式状态
字段。critic 为一个完整 chunk 输出两个标量 Q 值。

## 5. Rollout：如何选择下一整个 Action Chunk

Stage 2 rollout 先同时得到：

1. 冻结 OpenPI 生成的 `ref_chunk [B, 20, 7]`；
2. RLT MLP actor 生成的 `actor_chunk [B, 10, 7]`。

`predict_rlt_actions()` 根据 policy switch flag 一次性选择下一整个 chunk：

```text
switch == false -> ref_chunk[:, :10]
switch == true  -> actor_chunk
```

选中的结果会同时：

- 以 `[B, 10, 7]` 返回给 `EnvWorker`；
- 展平为 `[B, 70]` 覆盖 `forward_inputs["action"]`；
- 后续作为该宏观 transition 的 action 写入 replay。

这一步覆盖很重要。即使 actor 在 reference 阶段也计算了 proposal，buffer 保存的
仍是实际选择的 reference chunk，而不是未执行的 actor proposal。

### 5.1 键盘 Policy Switch 的时序

`keyboard_reward_wrapper: rlt_policy_switch` 使用 `b` 键将控制从 reference
policy 单向切换到 Stage 2 actor。它的粒度是完整 chunk：

1. keyboard wrapper 在每个 `env.step()` 完成后读取按键并记录 flag；
2. `RealWorldEnv.chunk_step()` 聚合当前 10 个子步的 flags；
3. 下一次 rollout 只取上一 chunk 的最后一个 flag；
4. 该 flag 被扩展到下一 chunk 的全部 10 个 action。

因此，在当前 chunk 的第 4 步按下 `b` 时：

```text
current chunk: steps 1..10 continue with the already selected reference chunk
next chunk:    all steps 1..10 use the Stage 2 actor
```

它不会从当前 chunk 的第 5 步开始切换。初始缺少 flag 时默认执行 reference；
episode reset 后 wrapper 恢复 reference 状态。

`rlt_switch_flags` 本身不写入 replay。离线查看 buffer 时可以看到最终 executed
actions 和 intervention mask，但不能直接从一个持久化的 switch 字段还原切换时刻。
切换前的 reference chunks 和切换后的 actor chunks 都会进入同一个 replay
buffer；policy switch 不会过滤 transition，只改变其中记录和执行的 action。

### 5.2 SpaceMouse Intervention 的时序

SpaceMouse 与 `b` 键不同，它在每个物理子步执行前读取输入，可以立即覆盖该子步
action。chunk 完成后环境返回：

```text
intervene_action: [B, H * A] = [B, 70]
intervene_flag:   [B, H]     = [B, 10]
```

干预不是只覆盖检测到非零输入的瞬间。wrapper 在最后一次非零 SpaceMouse 输入或
gripper 按键后的 0.5 秒内持续输出 expert action，并持续标记 intervention。默认
10 Hz 控制下，一次短输入通常会覆盖约 5 个连续子步，实际数量还受单步延迟影响。

由于模型 action 在执行前已经追加进 `EmbodiedRolloutResult`，人工动作需要在
下一轮环境循环开始时回写上一 chunk：

```text
last_full_action[i] =
  human_action[i] if intervene_flag[i]
  else selected_policy_action[i]
```

`update_last_actions()` 同时更新：

- trajectory 顶层 `actions`；
- 展平到 action coordinate 的 `intervene_flags`；
- `forward_inputs["action"]`；
- 并移除不再代表真实执行结果的 `model_action`。

因此 critic 始终用环境实际执行的 action 学习，BC loss 也可以逐子步区分
reference target 和 human target。

严格来说，buffer 保存的是与实际执行对应的 policy-frame action 或 intervention
action，不是底层 controller 的最终 absolute pose 命令；Franka 后续仍会做
clip、尺度映射和安全箱限制。

## 6. Franka 的 Multistep 适配

### 6.1 没有 `multistep()`，由 `chunk_step()` 顺序执行

Franka base env 只实现单步 `step(action)`。action chunk 由通用
`RealWorldEnv.chunk_step()` 拆分：

```text
input chunk_actions: [B, H, A]

for i in range(H):
    action_i = chunk_actions[:, i]       # [B, A]
    obs_i, reward_i, terminated_i, truncated_i, info_i =
        self.step(action_i, auto_reset=False)

stack all substep outputs
```

`EnvWorker.env_interact_step()` 在调用它之前会经过 `prepare_actions()`。
`env_type: realworld` 分支只把 tensor 移到 CPU/NumPy 后原样返回，不做 gripper
符号、shape、尺度或 frame 转换。因此必须保证以下合同一致：

- OpenPI output transform 已完成正确的反归一化；
- 末维顺序为 Franka 期望的 7D 顺序；
- action 位于预期范围；
- RelativeFrame wrapper 与训练数据的坐标系约定一致。

### 6.2 一个 Franka Primitive Step 做什么

每个 `FrankaEnv.step()` 执行：

1. 将 action clip 到 action space 的 `[-1, 1]`。
2. 从当前 TCP pose 出发，应用位置和 Euler rotation delta。
3. 处理 gripper action。
4. 将目标 TCP pose clip 到任务定义的 safety box。
5. 向低层 Cartesian impedance controller 发布目标 pose。
6. 按 `step_frequency` 做节流，然后读取 controller state 和相机观测。
7. 计算 reward、termination 和 truncation。

Peg insertion task 的默认 action scale 为：

```text
position delta scale: 0.02 m
rotation delta scale: 0.1 rad
gripper scale:        1
```

Peg insertion 的 position safety box 以 target 为中心：x/y 各限制在正负 5 cm，
z 限制在 target 到 target 上方 10 cm；orientation 对 roll/pitch 限制约
正负 0.01 rad，对 yaw 限制约正负 `pi/6`。这些边界限制的是每步发布的目标 pose，
不是完整轨迹的碰撞检测。

每个 primitive step 都重新读取机器人状态，并相对新状态累加下一个 delta；但这些
中间 observation 不会返回 policy 重规划。对上层 RLT policy 而言，它仍是一段
约 1 秒的开环高层 setpoint 序列。

默认 `step_frequency=10` Hz 只表示最大频率，不是硬实时保证。RPC、相机读取、
错误恢复以及夹爪开关都可能让单步超过 0.1 秒。controller RPC 返回也只表示命令
已经发布，不表示机械臂已到达目标 pose。

### 6.3 Chunk 输出聚合

`chunk_step()` 返回：

| 字段 | 形态 | 后续用途 |
|---|---|---|
| `obs_list` | 长度 `H`，每项为 substep 后 observation | policy 通常只取最后一项 |
| `chunk_rewards` | `[B, H]` | chunk TD reward |
| raw termination/truncation | 各 `[B, H]` | 计算 chunk 是否结束 |
| `infos_list` | 长度 `H` | intervention、switch、video 等 |
| intervention actions | `[B, H*A]` | 回写实际执行 action |
| intervention/switch flags | `[B, H]` | BC mask 与下一 chunk switch |

普通 RLT rollout 不在 chunk 内消费 `obs_list[0:H-1]`，只对 chunk 边界的最后
observation 做下一次 OpenPI/RLT feature extraction。

### 6.4 提前 Done 的准确语义

当前 `chunk_step()` 循环没有 done mask、`break` 或后续 no-op。假设 `H=10`
且第 4 个子步首次成功：

```text
substep:       1  2  3  4  5  6  7  8  9  10
raw terminal:  0  0  0  1  ?  ?  ?  ?  ?   ?
hardware:      execute every substep, including 5..10
auto-reset:    only after substep 10
```

完整 chunk 执行后，环境用 `any(dim=1)` 聚合是否曾 termination/truncation。
当 `auto_reset: True` 时，返回给算法的 done 位会全部清零，只在最后一列写入
聚合结果：

```text
returned termination: [0, 0, 0, 0, 0, 0, 0, 0, 0, any(raw termination)]
returned truncation:   [0, 0, 0, 0, 0, 0, 0, 0, 0, any(raw truncation)]
```

reward 不会移动到最后一列，仍留在真实产生它的物理子步。宏观 transition
因此包含完整 10 步执行产生的 reward 序列。

### 6.5 Auto-Reset 与 `final_observation`

auto-reset 发生在所有 `H` 个动作执行完成后。环境保留两种 observation：

- `obs`：reset 后的新 episode observation，供下一次 policy inference 使用；
- `final_observation`：chunk 最后一个动作之后、reset 之前的 observation，
  供上一条 terminal transition 的 `next_obs` 使用。

如果首次 done 出现在第 4 步，`final_observation` 对应第 10 步后的状态，不是
第 4 步的 terminal frame。这一点直接影响终止 transition 的定义和真机安全分析。

## 7. RLT Transition 的跨 Chunk 对齐

环境只有执行 `a_t` 后才能得到 `r_t` 和 `s_(t+1)`；RLT 特征又由 rollout
worker 异步计算。`EnvWorker` 因此为每个 pipeline stage 保存一个
`pending_obs`，让下一次 rollout 闭合上一条 transition。

精确时序如下：

```text
initial rollout(s0)
  -> choose and store a0
  -> cache RLT(s0) as pending_obs

execute a0
  -> environment returns reward chunk r0 and raw s1

rollout(s1)
  -> extract RLT(s1)
  -> append transition (RLT(s0), a0, r0, RLT(s1))
  -> choose and store a1
  -> cache RLT(s1)

execute a1
  -> environment returns r1 and raw s2

...

final rollout(sN)
  -> close transition N-1
  -> do not cache another state
  -> do not execute another action
```

初始 bootstrap inference 提供 `s0/a0`，末尾 final inference 提供最后一条
transition 的 `next_obs`。所以 action、reward 和 state pair 虽然在循环中的追加
时刻错开，trajectory 完成时仍一一对齐。

### 7.1 Auto-Reset 时的双 Observation 特征

terminal chunk 结束后，rollout 同时收到 post-reset `env_obs` 和 pre-reset
`final_obs`。`predict_rlt_actions()` 会提取两组 RLT features：

```text
plain keys:
  z_rl, proprio, ref_chunk
  <- extracted from post-reset env_obs
  <- used to select the new episode action

transition keys:
  rlt_transition_z_rl
  rlt_transition_proprio
  rlt_transition_ref_chunk
  <- extracted from pre-reset final_obs
  <- used only as terminal next_obs
```

`update_rlt_transitions()` 只用带 `rlt_transition_` 前缀的字段闭合上一条
transition，再将普通字段缓存为新 episode 的 `curr_obs`。这避免 terminal
`next_obs` 被 reset observation 污染。

## 8. Replay Buffer 如何存 Action Chunk

### 8.1 存的是宏观 Transition，不是 10 条单步 Transition

每个 replay sample 的逻辑结构为：

```text
transition_t = {
  curr_obs: {
    z_rl_t,
    proprio_t,
    ref_chunk_t,
  },
  actions: flatten(executed_action_repr_t),      # [70]
  intervene_flags: per-action-coordinate mask,  # [70]
  rewards: [r[t,0], ..., r[t,9]],                # [10]
  terminations: [d[t,0], ..., d[t,9]],           # [10]
  truncations: [u[t,0], ..., u[t,9]],            # [10]
  next_obs: {
    z_rl_t+1,
    proprio_t+1,
    ref_chunk_t+1,
  },
}
```

不会把它拆成十条 `(s_i, a_i, r_i, s_i+1)`，因为 policy 没有为 chunk 内中间
observation 计算 `z_rl/ref_chunk`，critic 也被定义为评估完整 70D chunk。

原始相机图像不进入 `curr_obs` 和 `next_obs`。这样 online replay 更小，但也意味着：

- Stage 2 不能从历史 buffer 重新计算 OpenPI/RLT feature；
- feature representation 必须保持冻结；
- 更换 Stage 1 checkpoint 后，旧 buffer 与新 feature space 不再天然兼容。

trajectory 还会保留 `forward_inputs` 中的 action 与 RLT feature 辅助副本；
RealWorld 路径每个 chunk 回写 action 时会移除 `model_action`。RLT
actor-critic loss 读取的是上面列出的顶层 `actions`、`intervene_flags`、
`curr_obs` 和 `next_obs` 等 canonical 字段。

### 8.2 实际动作回写

action 的三个版本要严格区分：

| 名称 | 含义 | 是否用于训练 |
|---|---|---|
| actor proposal | Stage 2 actor 原始生成的 chunk | 只有被选中且未干预的位置才执行 |
| selected policy action | reference 或 actor switch 后选中的 chunk | intervention 前的候选动作 |
| executed-action representation | SpaceMouse 覆盖后、与真实执行对应的 policy-frame action | 写入顶层 `actions`，供 critic 与 human BC 使用 |

policy switch 阶段，`predict_rlt_actions()` 先把 selected action 写入
`forward_inputs["action"]`。执行后如果有 intervention，`update_last_actions()`
再用逐步 mask 覆盖对应位置。最终 trajectory 的 `actions [T,B,70]` 与
`intervene_flags [T,B,70]` 对齐。

actor loss 会把 mask reshape 为 `[N, H, A]`，再对 action dimension 做 `any`，
恢复成 `[N, H]` 的逐物理子步 human mask。

### 8.3 为什么 Done 暂时有 `T+1`

`EmbodiedRolloutResult` 按时间把结果组装成 `[T, B, ...]`。每个 rollout epoch
还包含一个用于 bootstrap 的初始 done/termination/truncation 项，因此当前
`rollout_epoch=1` 时，这三个字段在组装阶段暂时为：

```text
actions/rewards/curr_obs/next_obs: [T,   B, ...]
dones/terminations/truncations:    [T+1, B, ...]
```

`TrajectoryReplayBuffer._flatten_trajectory()` 会识别每个 epoch 的额外 bootstrap
项，删除每段开头的第一项，再展平 `T` 和 `B`：

```text
[T, B, ...] -> [T * B, ...]
```

所以 sampled action、reward、done、`curr_obs` 和 `next_obs` 最终仍按同一个宏观
transition 对齐。

### 8.4 Trajectory、Sample 和 Cache 的计数单位

`TrajectoryReplayBuffer` 的几个配置单位不同：

| 配置/指标 | 实际单位 | 当前值/含义 |
|---|---|---|
| `min_buffer_size` | rollout trajectory 对象数 | 2；至少收到两个 trajectory 容器才开始训练 |
| `sample_window_size` | 最近 trajectory 数 | 200；只从最近 200 条 trajectory 的 samples 中抽取 |
| `cache_size` | 内存中的展平 trajectory 数 | 200 |
| `num_trajectories` | buffer 索引中的 trajectory 总数 | 每次 `add_trajectories()` 增加 |
| `total_samples` | 所有已索引 trajectory 的 `T*B` 宏观 transition 总数 | 当前单条通常增加 30 |

这里的 storage trajectory 是一次 rollout epoch 组装并发送给 actor buffer 的
容器，不是机器人 episode。当前 `auto_reset: True`，所以一个包含 30 个 chunks
的 trajectory 可以跨越多个成功、终止或 reset 的环境 episodes。

`sample_window_size` 不是 transition window。sample 流程是：

1. 取最近 `sample_window_size` 条 trajectory id；值为 0 时使用全部历史。
2. 累加这些 trajectory 各自的 `T*B` sample 数。
3. 在窗口内的全部宏观 samples 上有放回地生成随机全局索引。
4. 将索引映射回 trajectory 和局部 transition，再组成 `[N, ...]` batch。

因此长度更长的 trajectory 在窗口中自然拥有更多可抽取 samples。若请求 256 个
samples 但窗口内总共只有 60 个，buffer 会先把请求量缩到 60；不会靠重复采样补回
256。指标 `total_samples` 统计全部已索引 samples；开启 recent window 后，其中
较老的 samples 不一定仍参与当前采样。

当前 `auto_save: False`，`cache_size=sample_window_size=200`。这保证近期采样
窗口内的 trajectory 都能留在内存。若关闭磁盘保存又把 cache 配得小于 sampling
window，cache miss 将没有可靠的磁盘 trajectory 可加载，应避免这种组合。

## 9. Chunk 级 Actor-Critic Loss

### 9.1 Critic Target

critic 不把 10 个 reward 简单相加，而是按物理子步折扣：

```text
R_chunk = sum_{i=0}^{H-1} gamma^i * reward_i
```

下一状态由 online actor 生成 next action，target twin-Q 取较小值。默认
`bootstrap_type: standard`：

```text
not_done = not any(termination_i for i in [0, H))

y = R_chunk
    + not_done * gamma^H
    * min(Q1_target(next_obs, next_action),
          Q2_target(next_obs, next_action))
```

两个 current Q heads 都对同一个 `y` 做 MSE。

这里有两个细节：

- `H` 从 sampled reward tensor 的实际长度得到，当前为 10。
- `not_done` 只检查 termination，不检查 truncation，所以 truncation 仍会
  bootstrap。auto-reset 把 done 压到 chunk 最后一列不会改变 `any()` 的结果。

这是一种固定时长 macro-action 或 SMDP 风格的 target。当前实现始终使用
`gamma ** H`；它没有为中途 done 记录一个更短的有效 horizon。

### 9.2 Actor Target

actor 目标为：

```text
L_actor =
  -q_weight * mean(Q1(state, actor_chunk))
  +bc_weight * mean(MSE(actor_chunk, bc_target_chunk))
```

当前配置：

```yaml
q_weight: 0.1
bc_weight: 5
reference_dropout_prob: 0.5
```

BC target 按物理子步选择：

```text
bc_target[i] =
  actual human action[i], if intervene_flag[i]
  OpenPI reference[i],    otherwise
```

BC MSE 先在 7 个 action dimensions 上求均值，再对 batch 和 10 个子步求均值。
`bc_ref_loss` 与 `bc_human_loss` 分别只在各自 mask 上统计。

actor 的 Q 项明确使用 Q1，不使用 twin-Q minimum。`reference_dropout_prob=0.5`
会为每个 sampled transition 生成一个 `[N,1]` mask，将该样本的完整 10 步
reference 输入一起置零，而不是随机丢弃某几个子步。

### 9.3 它不是标准 Maximum-Entropy SAC

当前配置固定：

```yaml
entropy_tuning:
  alpha_type: fixed_alpha
  initial_alpha: 0.0
backup_entropy: false
```

RLT worker 禁止 alpha training，log-prob/entropy 不进入 actor target。因此该实现
复用了 SAC worker 的 replay、optimizer、target network 和 update scheduling，
但优化目标是 twin-Q critic 加 `-Q1 + BC` actor，不是 maximum-entropy SAC。

默认更新调度为：

- `update_epoch=8`：每轮做 8 次 critic update；
- `critic_actor_ratio=4`：通常在第 0、4 次同时更新 actor，共 2 次；
- `target_update_freq=1`、`tau=0.005`：每次 update 后做 target EMA；
- `min_buffer_size=2`：当前异步入口至少收到两个 rollout trajectory 对象后开始
  critic 和 actor update。

`train_actor_steps=2` 只在同步 SAC/RLT worker 中提供额外的 actor 启动门槛；
本文使用的 `train_async.py` 路径不读取该字段。当前两个值恰好都是 2，所以默认
现象相同，但单独增大 `train_actor_steps` 不会延迟异步 actor。

## 10. 一次完整在线训练循环

将前面的局部行为合起来，一个 Stage 2 iteration 是：

```text
1. Actor worker supplies current Stage 2 MLP weights
2. Weight syncer copies MLP weights to rollout worker
3. Env worker sends raw boundary observation
4. Frozen OpenPI:
     raw obs -> z_rl, proprio, ref_chunk[20,7]
5. Stage 2 MLP:
     RLT obs -> actor_chunk[10,7]
6. RLT switch:
     choose ref[:10] or actor chunk
7. RealWorldEnv:
     execute 10 Franka primitive steps
     SpaceMouse may replace individual steps
8. Env worker:
     write intervention actions back
     align pending curr_obs with next boundary feature
9. EmbodiedRolloutResult:
     build one [T,B,...] trajectory
10. Actor worker:
     add trajectory to replay
     sample macro-transitions
     update twin-Q, actor and target
11. Repeat from weight sync
```

异步入口使用
[run_realworld_async.sh](../../examples/embodiment/run_realworld_async.sh)
和 [train_async.py](../../examples/embodiment/train_async.py)。它选择异步 RLT actor
worker，但 feature extraction、action selection、Franka chunk execution、
transition layout 和 loss 定义与上文相同。

## 11. 关键配置合同

### 11.1 Stage 1 与 Stage 2 必须一致

以下字段必须与示范数据和 normalization stats 对齐：

| 合同 | 当前值 |
|---|---|
| OpenPI data/config | `repo_id=realworld_peg_insertion_rlt_stage1`、`config_name=pi05_franka_state` |
| 单步 action dimension | 7 |
| reference horizon | 20 |
| Stage 2 execution horizon | 10，且不大于 reference horizon |
| RLT feature dimension | `z_dim=2048` |
| Franka state dimension | wrapper 后 `proprio_dim=19` |
| state selection | `state_indices=[]`，保留 wrapper 输出的完整 state |

若其中任何一项变化，至少同时检查 Stage 1 data transform、OpenPI output transform、
RLT MLP 维度、env action contract 和旧 replay 的兼容性。

### 11.2 保存与恢复

当前 Stage 2 示例设置 `runner.save_interval: -1`，默认不会保存在线训练 checkpoint。
如果计划使用 `runner.resume_dir`，必须先把保存间隔改为正值并验证 checkpoint
中包含 Stage 2 actor/critic、target、optimizer 和 replay 状态。

路径职责如下：

| 配置 | 应指向什么 |
|---|---|
| `rollout.rlt_feature_model.model_path` | Stage 1 `global_step_N/actor` |
| `rollout.model.model_path` | fresh Stage 2 通常为 null；不要填 Stage 1 |
| `actor.model` | Stage 2 MLP 结构，不是 Stage 1 loader |
| `runner.resume_dir` | 完整 Stage 2 RLinf checkpoint |
| `runner.ckpt_path` | 可选的单个 Stage 2 权重文件 |

### 11.3 真机与集群

当前配置将 actor/rollout 放在 GPU node，将 env 放在 Franka node。多机启动时必须在
每台机器执行 `ray start` 之前设置唯一的 `RLINF_NODE_RANK`，并保证 hardware
配置中的 robot IP、controller node rank 和真实拓扑一致。

## 12. 安全边界与当前限制

在真实机器人上运行前，应明确以下当前代码事实：

1. **chunk 内不重规划。** 中间 camera/robot observations 不会送回 policy。
2. **done 不提前停止。** 首次成功或超时后，剩余 primitive actions 仍会执行。
3. **terminal state 是 chunk 尾。** `final_observation` 不是首次 done 帧。
4. **switch 下一 chunk 生效。** `b` 键不是 substep interrupt；SpaceMouse 才是
   substep-level override，且默认会在最后输入后 latch 0.5 秒。
5. **没有 force/torque chunk abort。** force/torque 当前是 observation，不会自动
   取消剩余 chunk。软件 action clip 和 pose safety box 不能保证轨迹无碰撞。
6. **只有 `prepare_actions()` 的 RealWorld 分支原样透传。** 下游
   `RelativeFrame` 仍会变换坐标，Franka 仍会 clip、scale 并应用 safety box；
   normalization、7D 顺序、gripper convention 和 frame 错配仍会作用到硬件。
7. **每个 RealWorldEnv worker 当前只支持一台真机。** 不能把模拟环境的任意 `B`
   假设直接搬到 Franka。
8. **Stage 1 feature 已物化进 replay。** 换 Stage 1 checkpoint 后不应无条件复用
   旧 buffer。
9. **switch flag 不持久化。** buffer 保存最终动作，不保存显式 reference/actor
   阶段标签。
10. **固定 H 是硬合同。** 当前没有 action-valid mask、可变 chunk 长度或 done
    后 padding；`intervene_flags` 只是 human/BC mask，`rlt_use_mask` 只是
    OpenPI prefix token mask。
11. **直接真机测试覆盖有限。** 当前相关单测没有覆盖 chunk 中途 done、
    `pending_obs` 全链路、真实 intervention 回写和 Franka multistep。

对于 peg insertion，建议先把 `H`、action scale 和安全箱调到保守值，在 dummy
环境验证 shape 与 transition，再做单 chunk 硬件在环测试，最后才启用持续在线更新。

如果未来要安全支持可变长 chunk，不能只在 done 后停止循环。至少还需要：

- 首次 done 后立即停止硬件执行；
- 保存首次 terminal observation；
- 用 `action_valid_mask` 或实际长度 `L` 标记 padding；
- 对 reward、BC 和 intervention mask 做有效步 reduction；
- 用 `gamma ** L` 而不是固定 `gamma ** H` bootstrap；
- 明确 replay 中 padded action 对 critic 的语义。

## 13. 监控与排错

### 13.1 Replay 指标

使用以下真实指标名：

| 指标 | 含义 |
|---|---|
| `train/replay_buffer/num_trajectories` | 已索引 trajectory 数 |
| `train/replay_buffer/total_samples` | 所有已索引宏观 transitions 数 |
| `train/replay_buffer/cache_size` | 当前缓存的展平 trajectories 数 |

当前 `B=1`、`T=30` 时，每收到一条完整 trajectory，`total_samples` 通常增加
30。不要使用 `train/replay_buffer/size` 判断 transition 数；当前 replay
实现没有这个统计键。

### 13.2 Actor-Critic 指标

| 指标 | 诊断方向 |
|---|---|
| `train/sac/critic_loss` | chunk TD error |
| `train/sac/actor_loss` | 加权 `-Q + BC` 总目标 |
| `train/actor/q_pi`、`train/actor/q_value_0`、`train/actor/q_value_1` | actor chunk 的 Q 与两个 heads |
| `train/actor/bc_loss` | 混合 reference/human target 的总 BC |
| `train/actor/bc_ref_loss` | 非干预子步相对 reference 的误差 |
| `train/actor/bc_human_loss` | 干预子步相对实际人类 action 的误差 |
| `train/actor/human_mask_ratio` | sampled substeps 中人工接管比例 |
| `env/success_once`、`env/episode_len` | 真机任务结果 |

常见异常判断：

- 当前单环境 rollout 中，`num_trajectories` 每增加 1 而 `total_samples`
  通常增加约 30；若比例明显不同，检查 `T`、`B` 和 actor-rank trajectory split。
- `human_mask_ratio` 始终为 0：检查 SpaceMouse wrapper、`intervene_action`
  聚合和上一 chunk 回写。
- 按 `b` 后当前 chunk 仍走 reference：这是预期时序；检查下一 chunk。
- terminal 后 next state 像 reset state：检查 `final_obs` 与
  `rlt_transition_*` 字段是否完整。
- buffer ready 比预期晚：`min_buffer_size` 按 trajectory，不按 transition。
- 一个 300-step rollout 只看到约 30 个 samples：这是 chunk 宏观 transition
  设计，不是漏存 270 个单步 transition。
- 请求 batch size 256 但早期只得到约 60 个 samples：window 内 sample 总数不足时，
  replay 会缩小请求量，不会用重复索引补足到 256。

## 14. 推荐代码阅读顺序

| 顺序 | 文件与关键符号 | 重点 |
|---:|---|---|
| 1 | [openpi_action_model.py：sft_forward/extract_rlt_obs](../../rlinf/models/embodiment/openpi/openpi_action_model.py) | Stage 1 loss、梯度边界、`z_rl` 与 reference chunk |
| 2 | [rlt_token_transformer.py](../../rlinf/models/embodiment/modules/rlt_token_transformer.py) | prefix 压缩、mask 和 reconstruction |
| 3 | [rlt_mlp_policy.py：RLTMLPPolicy](../../rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py) | actor/critic shape、reference dropout、chunk 输出 |
| 4 | [rlt/rollout.py：predict_rlt_actions](../../rlinf/algorithms/rlt/rollout.py) | reference/actor 选择与 current/final feature |
| 5 | [keyboard_rlt_policy_switch_wrapper.py](../../rlinf/envs/realworld/common/wrappers/keyboard_rlt_policy_switch_wrapper.py) | `b` 键的下一 chunk 时序 |
| 6 | [env_worker.py：_run_interact_once](../../rlinf/workers/env/env_worker.py) | 异步 env/rollout 循环、intervention 回写 |
| 7 | [realworld_env.py：chunk_step](../../rlinf/envs/realworld/realworld_env.py) | primitive step 循环、done 聚合和 auto-reset |
| 8 | [franka_env.py：FrankaEnv.step](../../rlinf/envs/realworld/franka/franka_env.py) | action clip/scale、安全箱、控制和 reward/done |
| 9 | [rlt/transition.py：update_rlt_transitions](../../rlinf/algorithms/rlt/transition.py) | `pending_obs` 与 terminal state 对齐 |
| 10 | [embodied_io_struct.py：EmbodiedRolloutResult](../../rlinf/data/embodied_io_struct.py) | action/intervention 回写和 `[T,B,...]` trajectory |
| 11 | [replay_buffer.py：TrajectoryReplayBuffer](../../rlinf/data/replay_buffer.py) | trajectory 计数、bootstrap done 移除和窗口采样 |
| 12 | [rlt_ac_policy_worker.py：RLTACLossMixin](../../rlinf/workers/actor/rlt_ac_policy_worker.py) | chunk return、twin-Q 与 `-Q1 + BC` |

## 15. 最小心智模型

阅读或修改这套实现时，可以始终用下面四句话检查设计是否仍一致：

1. OpenPI 在 chunk 边界产生 frozen feature 和 20 步 reference。
2. Stage 2 将前 10 步 reference 作为条件，一次预测并评价一个 70D 宏观动作。
3. RealWorldEnv 顺序执行 10 个 Franka primitive actions，并把环境输入、逐步
   reward/done 和 intervention 聚成一个 transition。
4. replay 保留逐步 reward/done，但 actor-critic 的状态转移和 Q 值都以完整 chunk
   为单位。

只要其中任一层改变 horizon、action layout、done 语义或 feature checkpoint，
其余三层都必须同步检查，不能只改一个 YAML 数字。
