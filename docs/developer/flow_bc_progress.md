# Franka + GELLO Flow-T BC 合并检查记录

本文记录 `H=1` 与 `H>1` 单路径合并时使用的基线和验收条件。当前行为说明见
[`flow_bc.md`](flow_bc.md)。

## 合并前基线

- 真实数据目录：`/nas/xyh/data/data_pick_cube_new/collected_data`
- train split：26 episodes、3031 anchors
- `H=8` 完整 chunk：2849
- valid length 1 到 7：每个长度各 26 个 tail anchor
- padding：728 / 24248 action steps，约 3.0%
- 合并前 Flow 相关测试：115 passed

这些数字用于检测 split、anchor 与 episode-tail 行为是否意外变化。它们不是通用 dataset
常量；更换数据或 split 后应重新生成基线。

## 统一合同

```text
action:            [B,H,7]
action_valid_mask: [B,H]
flow state:        [B,H*7]（仅 actor 内部）
sample:            [B,H,7]
```

H=1 同样保留 horizon 维和全真 mask。episode 尾部 invalid action 必须为全零，mask 只
用于 loss 和指标。所有 horizon 使用 identity latent normalization。

静态 evaluator 支持 `flow_ode` 与 `flow_sde`，且不给 sampler 传 target mask。H>1 online
SACFlow 仍在配置阶段拒绝。

## Checkpoint 约束

portable actor 只接受 `schema: unified_flow_t_bc`。manifest 没有数字 version，不解析旧
字段，不提供 H1→H>1 inflation。ODE/SDE 是运行时 sampler 选择，不影响同一 checkpoint
的权重兼容性。

## 验收状态

- 数据：覆盖 `H={1,2,8}`、短 episode、相邻 episode、prefix mask 和 exact-zero tail。
- Objective：覆盖 RF/iMF masked reduction、H=1 parity、JVP、invalid coordinate gradient。
- Sampler：覆盖 RF/iMF × ODE/SDE × H，并验证显式 initial/step noise 可复现。
- Checkpoint：覆盖 H=1/H=8 round-trip、跨 H 拒绝、旧 manifest 拒绝和 actor-only load。
- Static eval：指标忽略 padded tail，报告 sampler、参数与随机 seed。
- GPU：H=1/H=8 的 100-step/FSDP save-resume 仍需在目标 GPU 环境执行并记录结果。
