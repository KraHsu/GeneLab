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
- **课程**（仅训练）—— 成功课程随策略稳定达成目标，把目标容差从松（0.8 rad）收紧到目标值（0.2 rad）；自适应回合课程随回合存活时长逐步加大 cube 速度扰动。

## 域随机化

训练随机化手部摩擦、link 质量/质心、PD 增益、编码器偏置，并周期性地给 cube 施加速度扰动；评估（`--play`）时剥离这些、跑标称物理。

!!! note "略去的接触随机化"
    MuJoCo 专属的接触 DR——`sol_params`（软垫柔顺度）、geom 尺寸、惯量张量——在 Genesis 无对应（求解器不同；不支持逐环境 geom 缩放或惯量设置），因此略去。

## 收敛情况

发布规模训练（8192 环境、5000 迭代、RTX 5060 Ti，约 5 小时）：

- 成功课程在约第 1000 迭代把容差收紧到目标 0.2 rad，之后策略在**满难度**下持续提升（训练末每回合约 6.7 个目标，抓握稳定，`cage_drop` ≈ 0.2）。
- 100 回合确定性 eval（`genelab eval`，0.2 阈值）：**成功率 0.99**（把 cube 重定向到至少一个"稳定保持"的 SO(3) 目标的回合比例），平均回报约 1060，平均回合长度约 591 步。

成功课程是必需的：其先松后紧的容差提供了早期奖励信号，让重正则化的策略学会主动重定向，而不是只稳握不动。

## 另见

- [舞肌手](wuji-hand.md)
- [资产库](../concepts/asset_zoo.md)
