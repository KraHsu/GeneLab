# Unitree G1

Unitree G1 示例是倒立摆教程后的进阶机器人路径。它展示 GeneLab 如何扩展到人形机器人 locomotion 和动作模仿。

## 任务

| Task id | 展示内容 |
|---|---|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | 平地速度跟踪。 |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | 参考动作跟踪。 |

## 安装

```bash
uv pip install -e examples/unitree
genelab list tasks
```

## 速度跟踪

```bash
genelab play Genelab-Velocity-Flat-Unitree-G1-v0 --vis --steps 500
genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --max_iterations 1500
```

回放：

```bash
genelab play Genelab-Velocity-Flat-Unitree-G1-v0 \
  --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

## 动作模仿

```bash
python -m genelab_unitree.replay_motion
genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --max_iterations 30000
```

默认 clip 通过 asset zoo 获取并缓存到本地。

## 另见

- [运行 RL 实验](../best-practices/rl-experiments.md)
- [资产库](../concepts/asset_zoo.md)
