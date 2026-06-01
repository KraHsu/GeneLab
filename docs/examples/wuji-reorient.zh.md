# 舞肌手重定向

舞肌手的 SO(3) 手内 cube 重定向：一只固定基座、掌心朝上的灵巧手，需要把一个自由 cube 旋转到一连串随机朝向目标（在腕部 "tag" 系下表达），并在容差窗口内稳定握持而不掉落。本任务是 mjlab `reorient` 参考实现的 Genesis 适配移植，用 RSL-RL PPO 训练。

## 任务

```text
Genelab-Reorient-Wuji-Hand-v0
```

20-DoF 右手（5 指 × 4 关节）、一个 54 mm cube，以及带"保持-推进"成功循环的 SO(3) 目标命令。

## 运行

```bash
uv pip install -e examples/wuji
genelab train Genelab-Reorient-Wuji-Hand-v0 --num_envs 4096 --gpu
genelab play  Genelab-Reorient-Wuji-Hand-v0 --checkpoint logs/rsl_rl/wuji_reorient/<run>/model.pt --vis
```

## MDP 设计

- **动作** —— 带 EMA 平滑 + 启动 warmup 的关节位置偏移（`JointPositionOffsetEMAAction`），20 维，围绕 home 抓握关键帧缩放。
- **命令** —— `InHandReorientCommand`：在 tag 系下于 SO(3) 上均匀采样目标；APPROACHING → SUCCESS_WINDOW 状态机统计在容差内的步数，保持窗口后推进到新目标。
- **奖励** —— 朝向对齐（测地容差）、递增的保持奖励、掌心相对 AABB "cage" 逃逸惩罚、hand-pose / action-rate / torque 正则项，以及由自定义 `get_contacts` 手-cube 传感器驱动的接触项（指尖滑动、palm-detach、手指自碰撞）。
- **观测** —— policy：关节位置/速度、cube 在 tag 系下的位置、6D 目标朝向误差、上一步动作；critic 额外加入命令状态与 cage 计数进度。
- **终止** —— 超时，或 cube 离开掌心 cage 足够久时触发 `cage_drop`。

## 域随机化

训练随机化手部摩擦、link 质量/质心、PD 增益、编码器偏置，并周期性地给 cube 施加速度扰动；评估（`--play`）时剥离这些、跑标称物理。

!!! note "Genesis 与 MuJoCo 的接触 DR"
    mjlab 参考还会随机化 MuJoCo 专属的 `sol_params`（软垫柔顺度）、geom 尺寸和惯量张量。这些在 Genesis 无对应（求解器不同；不支持逐环境 geom 缩放或惯量设置），因此这里略去。

## 收敛情况

短时 GPU 冒烟（2048 环境、400 迭代、RTX 5060 Ti）—— 仅为学习信号验证，并非发布级策略：

- 朝向对齐奖励从约 5 升到约 9.7；多数回合活到超时而非掉落 cube。
- 100 回合确定性 eval（`genelab eval`）：**成功率约 0.33**（把 cube 重定向到至少一个"稳定保持"的 SO(3) 目标的回合比例——一个目标要求在整个保持窗口内都处于容差内），平均回报约 328，平均回合长度约 580 步。

在这个短训练预算下，严格的保持窗口是瓶颈：策略能把 cube 转到目标"附近"（朝向奖励高），但只有约三分之一的回合完成了"稳定保持"的成功。完整发布规模训练（8192 环境、数千迭代）留给使用者，应能把成功率大幅提升。

## 另见

- [舞肌手](wuji-hand.md)
- [资产库](../concepts/asset_zoo.md)
