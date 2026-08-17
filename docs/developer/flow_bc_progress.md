# Franka + GELLO Flow BC 检查进度

更新日期：2026-08-14（Asia/Shanghai）；GPU 验收执行于 2026-08-13 23:57。

## 结论

当前 Flow BC 已满足本轮停止条件：能够从
`/nas/xyh/data/data_pick_cube/collected_data` 加载真实 LeRobot 数据，在物理 GPU 6
上使用 Improved MeanFlow（iMF）完成 100 次完整的
`forward -> loss.backward() -> gradient clipping -> AdamW.step()`，全部指标 finite，
训练窗口和固定 probe 的 loss/field MSE 均下降。

本轮使用单进程验收脚本，未启动、连接或修改 Ray 集群。该脚本复用正式 YAML、
`FrankaGelloFlowDataset`、`FlowPolicy`、ResNet10、`FlowTActor` 和
`ImprovedMeanFlowObjective`，用于验证数据到优化器的完整链路；它不替代后续长时间的
FSDP 生产训练。

## 数据检查

训练输入必须是：

```text
/nas/xyh/data/data_pick_cube/collected_data
```

顶层目录中的 `demos/` 是 `TrajectoryReplayBuffer`，不是 Flow BC loader 的输入。

真实数据检查结果：

| 项目 | 结果 |
|---|---:|
| LeRobot 版本 | v2.1 |
| finalized shards | 1 |
| episodes | 34 |
| frames | 5875 |
| state | `float32[19]` |
| action | `float32[7]` |
| `image` | `uint8[128,128,3]`，loader 输出 `float32[3,128,128]`、`[0,1]` |
| `extra_view_image` | 存在，但当前单相机配置有意不使用 |
| fps | 10 |

LeRobot parquet 与 `demos/trajectory_*.pt` 中对应 episode 的 state/action 已逐元素
对齐，34/34 episode 的最大绝对差为 `0.0`，没有导出漂移或 frame/action 错位。

### 动作越界

5875 帧、41125 个动作标量中有 99 个严格超出 `[-1,1]`，分布在 95 帧、22 个
episode，比例为 `0.2407%`。越界只出现在旋转维：

```text
minimum = [-0.2418, -0.7873, -0.3304, -1.1133, -1.2824, -1.0935, -1.0000]
maximum = [ 0.7463,  0.4605,  0.2324,  1.1059,  1.1137,  1.1334,  1.0000]
outside = [0, 0, 0, 37, 32, 30, 0]
```

代码侧根因已经定位：

1. `gello_intervention.py` 先在 base frame 对 GELLO 动作逐维 clip。
2. wrapper 顺序使 `RelativeFrame` 随后对 `info["intervene_action"]` 做逆 Adjoint。
3. 逆坐标变换不保持逐维 `[-1,1]`，`collect_episode.py` 再将结果原样写盘。

按本轮要求，BC 不拒绝这些旧数据，而是在 `atanh` 前直接 clamp 到
`[-1+1e-4, 1-1e-4]`。同时：

- `action_clamp_fraction` 只统计严格超出 `[-1,1]` 的标量；
- `action_clamp_abs_max` 统计 `max(abs(unit_action)-1, 0)`；
- 合法端点 `+/-1` 虽为 `atanh` 数值稳定性变成 `+/- (1-1e-4)`，但不再误报为越界。

需要保留的风险：夹爪维有 4717/5875 个值恰为 `+/-1`，反变换后 latent 约为
`+/-4.9516`。当前数据的 latent 平方能量约 95.8% 来自夹爪维，iMF 的同权 MSE
可能优先拟合夹爪。此次没有修改 action scale 或按维权重，因为这会改变 BC 与 online
checkpoint 的动作语义。

## 配置与权重

[`franka_gello_flow_bc.yaml`](../../examples/sft/config/franka_gello_flow_bc.yaml)
已用于本轮验收，关键值为：

```yaml
cluster:
  component_placement:
    actor: 6-6

runner:
  max_steps: 100
  save_interval: 100

data:
  train_data_paths: /nas/xyh/data/data_pick_cube/collected_data
  num_samples_per_epoch: 3200

actor:
  micro_batch_size: 8
  global_batch_size: 8
  model:
    model_path: /nas/xyh/RLinf/.cache/flow_bc/RLinf-ResNet10-pretrained
```

ResNet10 权重：

```text
/nas/xyh/RLinf/.cache/flow_bc/RLinf-ResNet10-pretrained/resnet10_pretrained.pt
sha256=f21cf3b90a4febbea4042cc924c25c912a95b61d6542176de079ba6af7987a87
```

正式 Ray/FSDP 入口中的 `actor: 6-6` 是宿主机硬件 rank，不要在该入口外再用
`CUDA_VISIBLE_DEVICES=6` 把集群资源视图缩成一张卡。单进程验收脚本不经过 Ray，才使用
`CUDA_VISIBLE_DEVICES=6` 将物理 GPU 6 映射为逻辑 `cuda:0`。

## GPU6 100 步验收

运行命令：

```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/nas/xyh/RLinf \
MPLCONFIGDIR=/tmp/rlinf-flow-bc-mpl \
/nas/xyh/RLinf/.venv/bin/python \
  examples/sft/verify_franka_gello_flow_bc.py \
  --steps 100 \
  --batch-size 8 \
  --batch-mode stream \
  --memory-fraction 0.05
```

设备与资源：

| 项目 | 结果 |
|---|---:|
| physical GPU | 6 |
| UUID | `GPU-38ffa173-47d3-8e22-0491-7c6f71615be5` |
| model | NVIDIA H20-3e |
| process-visible GPUs | 1 (`cuda:0`) |
| memory fraction limit | 5% |
| PyTorch peak allocated | 0.158 GiB |
| PyTorch peak reserved | 0.166 GiB |
| data load | 97.24 s |
| 100 training steps | 5.46 s |
| total | 107.38 s |

训练结果：

| 指标 | 前 10 步均值 | 后 10 步均值 | 相对变化 |
|---|---:|---:|---:|
| adaptive `loss` | 0.994464 | 0.980512 | -1.40% |
| `field_mse` | 3.302294 | 1.901735 | -42.41% |

固定真实 batch、固定 8 组 iMF 噪声/时间的训练前后 probe：

| 指标 | 训练前 | 训练后 | 相对变化 |
|---|---:|---:|---:|
| adaptive `loss` | 0.997929 | 0.979593 | -1.84% |
| `field_mse` | 6.716301 | 1.629606 | -75.74% |

额外验证：

- 完成 step 数严格等于 100；
- 100 条逐步记录中的 loss、field MSE、grad norm 等全部 finite；
- 参数最大变化为 `0.011934`，确认 optimizer 确实更新了模型；
- stream batch 的 loss/field MSE 窗口下降；
- 固定 probe 的 loss 至少下降 1%，field MSE 至少下降 20%；
- 所有自动验收条件均为 `true`。

`adaptive_power=1.0` 时，反向传播的 loss 近似
`MSE / stopgrad(MSE + 0.01)`，数值自然接近 1，因此 `field_mse` 是更直观的拟合指标。
随机 stream batch 单步值会波动，窗口均值和固定 probe 同时下降才是本轮验收依据。

完整产物：

```text
results/flow_bc/verification_gpu6/20260813T155710Z/
├─ metrics.jsonl                         # 100 行逐步指标
├─ summary.json                          # 配置、设备、趋势与自动检查
└─ portable_actor_smoke/
   ├─ flow_actor_manifest.json
   └─ model_state_dict/full_weights.pt
```

## 修改与测试

本轮只修改/新增 Flow BC 范围内文件：

- `rlinf/models/embodiment/flow_policy/flow_policy.py`
- `tests/unit_tests/test_flow_policy_v2.py`
- `examples/sft/config/franka_gello_flow_bc.yaml`
- `examples/sft/verify_franka_gello_flow_bc.py`
- `docs/developer/flow_bc.md`
- `docs/developer/flow_bc_progress.md`

验证命令及结果：

```text
43 passed in 163.22s
ruff check: passed
ruff format --check: passed
git diff --check: passed
```

43 个测试覆盖 objective、FlowPolicy、FlowTActor、真实数据 adapter、portable checkpoint
和配置校验。新增回归测试同时覆盖真实越界 clamp 统计以及合法 `+/-1` 端点不应计为
异常。

## 后续注意事项

当前停止条件已完成，没有必须继续执行的验证。进入长时间生产训练前仍建议：

1. 保持监控 `train/field_mse`、`train/action_clamp_fraction` 和
   `train/action_clamp_abs_max`。
2. 单独记录各动作维的 field/error，确认夹爪 latent 没有掩盖 6D 位姿学习。
3. 若要解决采集源头越界，应在最外层策略坐标系写盘前显式应用 action space；旧数据仍按
   当前带指标的直接 clamp 兼容。
4. 正式 FSDP 启动前确认 Ray 对 GPU 6 的硬件 rank 视图正常，保持
   `cluster.component_placement.actor=6-6`。
