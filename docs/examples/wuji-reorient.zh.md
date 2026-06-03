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

- **动作** —— 带 EMA 平滑 + 启动 warmup 的关节位置偏移（`JointPositionOffsetEMAAction`），20 维，围绕 home 抓握关键帧缩放；训练时逐步注入一个小幅的逐步动作噪声（见域随机化）。
- **命令** —— `InHandReorientCommand`：在 tag 系下于 SO(3) 上均匀采样目标；APPROACHING → SUCCESS_WINDOW 状态机统计在容差内的步数，保持窗口后推进到新目标。
- **奖励** —— 朝向对齐（测地容差）、递增的保持奖励、掌心相对 AABB "cage" 逃逸惩罚、hand-pose / action-rate / torque 正则项，以及由自定义 `get_contacts` 手-cube 传感器驱动的接触项（指尖滑动、palm-detach、手指自碰撞）。
- **观测** —— actor 看到一段 **3 步历史**（term-major，与 mjlab 参考一致）：相对 home 位姿的关节位置、关节速度、cube 在 tag 系下的位置、6D 的 cube 到目标朝向误差、上一步动作——每步 69 维，历史共 207 维。critic 在单步基础上额外加入命令状态与 cage 计数进度。
- **终止** —— 超时，或 cube 离开掌心 cage 足够久时触发 `cage_drop`。
- **课程**（仅训练）—— 成功课程随策略稳定达成目标，把目标容差从松（0.8 rad）收紧到目标值（0.2 rad）；自适应回合课程随回合存活时长逐步加大 cube 速度扰动。

## 域随机化

这里的随机化针对的是**闭环反馈步态的鲁棒性**，而非去匹配某个具体物理参数——原因见[跨模拟器迁移](#sim-to-sim-transfer)。训练时施加以下项（评估时全部剥离、跑标称物理）：

- 逐环境 startup 随机化：手部与 cube 摩擦、link 质量/质心、cube 质量、PD 增益，以及编码器偏置；
- 一个逐步动作噪声，以及更强的观测噪声（关节位置/速度、cube 位置、目标误差）；
- 一个频繁的线速度 + 角速度 cube 扰动。

!!! note "接触求解器随机化"
    Genesis 把 geom 求解器参数（`sol_params`）按全局而非逐环境存储，因此逐环境的接触柔顺度随机化——以及 MuJoCo 专属的 geom 尺寸/惯量随机化——在 Genesis 没有逐环境对应，故略去。况且迁移差距并不在这些参数上（见下文）。

## 收敛情况

发布规模训练（8192 环境、5000 迭代、RTX 5060 Ti，约 5 小时）：

- 成功课程在约第 1500 迭代把容差收紧到目标 0.2 rad，之后策略在**满难度**下持续提升（训练末在扰动下每回合约 5 个目标）。
- 256 回合确定性 eval（0.2 阈值、标称物理）：**成功率 ≈ 1.0**（把 cube 重定向到至少一个"稳定保持"的 SO(3) 目标的回合比例），每回合约 5 个目标。

成功课程是必需的：其先松后紧的容差提供了早期奖励信号，让重正则化的策略学会主动重定向，而不是只稳握不动。

## 跨模拟器迁移 { #sim-to-sim-transfer }

`sim2sim_mjlab` 直接在 mjlab 参考环境本身里评估训练好的策略——它的 `scene_builder`（手 + cube + 接触）、物理、动作管线、目标采样，以及掉落 + 保持/成功判据。只有策略和一个观测/动作适配器来自 GeneLab。两边的观测布局不同（mjlab：limit-normalized 关节角 + 关节位置目标误差 + tag 系目标误差；GeneLab：相对关节位置 + 关节速度 + 世界系目标误差，且都是 3 步历史），所以适配器从 mjlab 的场景状态、用 joint-major ↔ finger-major 重排和 3 步历史重建出 GeneLab 的 actor 观测。

在 mjlab 环境中跑 100 次试验 × 3 个随机种子（0.2 阈值）：**成功率 ≈ 0.65**（最佳 checkpoint ≈ 0.67）、**掉落率 ≈ 0**——抓握完全迁移,剩下的是没能在试验窗口内**保持**住目标。迁移率**对训练非单调**、在训练中段见顶（约 iter 3500–3800），所以迁移最优的 checkpoint 不一定是最后一个。`play_mjlab` 用同一套桥接驱动 mjlab 的原生 viewer（完整场景 + 目标可视化）以供观察。两者都在 wuji-mjlab 环境中运行，并把 GeneLab 挂到 `PYTHONPATH`。

!!! note "为什么随机化很关键"
    不带上述鲁棒性随机化训练出的策略，在 mjlab 下能握住 cube 却几乎无法重定向（成功率 ≈ 0）：再抓握步态过拟合到了 Genesis 精确的逐步接触响应。扫摩擦、质量、时间步、作动器增益和接触求解器都改变不了这一点，而把 Genesis 内取胜的手指目标序列回放到 mjlab 里确实能让 cube 转动——所以差距是闭环反馈的脆弱性，正是动作/观测噪声与角速度 cube 扰动所针对的。

## 另见

- [舞肌手](wuji-hand.md)
- [资产库](../concepts/asset_zoo.md)
