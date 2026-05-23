# Eval 与 Export

本页介绍 GeneLab 的研究复现工具链 —— `genelab.rl.evaluator` / `eval_callback`
/ `exporter`，对应三条 CLI，把 **train → eval → export** 闭环串起来：

| 命令 | 用途 | 输出 |
|---|---|---|
| `genelab eval TASK CKPT` | Deterministic rollout，固定 seed，N episode | `eval.json` |
| `genelab train ... --eval-every K` | 训练期周期 eval + 保存 best model | `logs/.../best_model.<ext>` + `best_model_meta.json` |
| `genelab export TASK CKPT` | 不依赖后端的 TorchScript / ONNX policy | `policy.{ts,onnx}` + `<file>.metadata.json` |

三个命令都走同一个 backend 抽象（`InferenceSetup`，定义在
`genelab.rl.backends.base`），所以在 `rsl_rl` / `skrl` / `sb3` 上行为一致。

## `genelab eval`

跑 vectorized deterministic rollout，并按下面的 schema 写一份 JSON：

```bash
genelab eval GeneLab-Inverted-Pendulum-v0 logs/rsl_rl/exp1/.../model_500.pt \
    --num-envs 64 --episodes 100 --seed 0 \
    --deterministic --out eval.json
```

输出：

```json
{
  "task": "GeneLab-Inverted-Pendulum-v0",
  "checkpoint": "logs/.../model_500.pt",
  "num_episodes": 100,
  "metrics": {
    "return_mean": 487.3,
    "return_std": 22.1,
    "length_mean": 998.4,
    "success_rate": 0.96
  },
  "wall_clock_seconds": 18.2,
  "seed": 0,
  "deterministic": true,
  "evaluated_at": "2026-05-20T08:42:11+00:00"
}
```

### Success rate

`success_rate` 在任务通过 `ManagerBasedRlEnv.step` 的 `extras["is_success"]`
publish 一个 per-env bool tensor 时计算（gymnasium 约定）。任务在 termination
/ reward 项中设置 `self._extras["is_success"] = <(num_envs,) bool tensor>` —
通常是 manipulation 的目标姿态判定、locomotion 的速度命令跟踪判定等。

任务没暴露 `is_success` 时输出 `success_rate: null`；下游工具
（best-model 选择、reference-runs 表）必须容忍 `None`。

## `genelab train --eval-every`

设了 `--eval-every K` 后，训练按 `K` 个 iteration 分 chunk 跑。每 chunk 结束
后，加载最新 checkpoint 进同一 backend，跑一次 deterministic eval（默认 10
episode，`num_envs` 与训练相同）。当 `return_mean` 超过历史最佳，checkpoint
被复制到 `<log_dir>/best_model.<ext>`，同时 `best_model_meta.json` 写入 eval
payload。

```bash
genelab train GeneLab-Inverted-Pendulum-v0 \
    --max_iterations 1000 --num_envs 64 --seed 0 \
    --eval-every 100 --eval-episodes 16
```

注意：

- 每个 chunk 走 backend 的正常 train lifecycle，会关闭并重建 Genesis env。短
  任务把 `--eval-every` 设到 ≥ 50，让 Genesis 初始化时间被摊薄。
- Off-policy 算法（skrl / sb3 的 SAC / TD3 / DDPG）每 chunk 重载 checkpoint
  会丢 replay buffer，sample efficiency 下降但仍能收敛。
- `best_model.<ext>` 复用来源 backend 的 checkpoint 格式（`rsl_rl` / `skrl`
  用 `.pt`，`sb3` 用 `.zip`）。meta 文件记录来源 iter、eval seed、episode
  数、return 统计。

## `genelab export`

把 actor 子网络序列化为 **TorchScript** 或 **ONNX**，把每个 obs 项的
`scale` / `clip` 烘焙进单一 `forward(raw_obs) -> actions` 调用。部署侧只需
`torch`（TorchScript）或一个 ONNX runtime，**不需要** `rsl_rl` / `skrl`
/ `stable_baselines3`。

```bash
# TorchScript
genelab export Genelab-Velocity-Flat-Unitree-G1-v0 logs/.../model_30000.pt \
    --format torchscript --out policy.ts

# ONNX（默认 opset 17）
genelab export Genelab-Velocity-Flat-Unitree-G1-v0 logs/.../model_30000.pt \
    --format onnx --out policy.onnx --opset 17
```

> **注**：`GeneLab-Franka-Pick-And-Place-v0` 现在是 SAC+HER + goal-conditioned
> `Dict` 观测，命中下方的 limitations —— 对它跑 `genelab export` 会报清晰
> 错误。Locomotion 任务（cartpole / G1）走 flat-tensor obs，导出干净。

导出器会在旁边写一份 `<output>.metadata.json`，描述 obs schema：

```json
{
  "task": "Genelab-Velocity-Flat-Unitree-G1-v0",
  "checkpoint": "logs/.../model_30000.pt",
  "obs_groups": {
    "policy": {
      "dim": 23,
      "terms": [
        {"name": "joint_pos", "dim": 7, "start": 0, "scale": 1.0, "clip": null},
        {"name": "joint_vel", "dim": 7, "start": 7, "scale": 0.1, "clip": [-2, 2]}
      ]
    }
  },
  "action_dim": 7,
  "action_range": [-1.0, 1.0],
  "normalization_baked": true,
  "format": "torchscript",
  "exported_at": "2026-05-20T08:42:11+00:00",
  "torch_version": "2.4.0"
}
```

### 部署侧用法

```python
import torch
m = torch.jit.load("policy.ts")
m.eval()
# raw obs（按训练时的拼接顺序）；模型自己应用 scale/clip
actions = m(torch.tensor([[joint_pos_0, joint_pos_1, ..., joint_vel_0, ...]]))
```

ONNX：

```python
import onnxruntime as ort
sess = ort.InferenceSession("policy.onnx")
actions = sess.run(None, {"obs": raw_obs.astype("float32")})[0]
```

### 实际导出的内容

actor 通过 backend 各自的小 shim 取出来，包成统一的调用形态：

- `rsl_rl`：优先用 `runner.alg.actor_critic.actor`（当它是干净的 MLP 时）；
  否则 fallback 到 `act_inference`。
- `skrl`：包 `agent.policy.act`，对 `GaussianMixin` policy 返回 deterministic
  mean（`mean_actions` key）。
- `sb3`：包 `model.policy._predict(obs, deterministic=True)`，对 PPO / A2C /
  SAC / TD3 / DDPG 统一。

### 限制

- 字典 observation（SB3 + HER）暂不支持导出。仅支持单一 group 的 flat tensor
  obs。
- Recurrent policy（rsl_rl 设置 `rnn_type`）暂不支持 —— 导出的模型没有 hidden
  state 槽。
- 导出的模型不应用 `ObservationTermCfg.noise` —— noise 只在训练时启用。
