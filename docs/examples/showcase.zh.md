# 可视化 showcase

`examples/genelab_showcase/` 提供 7 个 play-only 任务，单条命令就能跑过 M1–M4 全部
building block。每个任务把真实机器人 (Franka 或 Unitree G1) 投入一个最小的
`ManagerBasedRlEnv`，跑一段脚本动作循环，把对应特征的证据写进
`logs/showcase/<slug>/`。目标是人眼 / 数值核对，不做训练。

## 共用 runner

7 个任务都继承自 `examples/genelab_showcase/src/genelab_showcase/runner.py` 中的
`ShowcaseRunner` 基类。它负责 env 生命周期、定长 scripted-action 循环、日志目录创建以及
viewer tick。子类只需重载两个 hook：`_scripted_action(env, step)` 返回每步动作张量，
`_dump(env, step, log_root)` 写各自的证据文件（PNG 帧、直方图、跟踪误差日志……）。
Recording showcase 是唯一不重载 `_dump` 的——它的实时图和文件写入全部跑在 Genesis 的
recorder 线程上，由 env-cfg 的 `recordings=(...)` 元组直接驱动。

## 任务列表

| Task id | Robot | 演示的功能 |
|---|---|---|
| `GeneLab-Sensors-Showcase-v0` | Franka | `CameraSensor` (RGB+depth PNG)、`IMUSensor`、`FrameTransformerSensor` |
| `GeneLab-RayCast-Showcase-v0` | Franka | `RayCastSensor` × 3 种 pattern (`GridPattern`、`RingPattern`、`HemispherePattern`) |
| `GeneLab-Contact-Showcase-v0` | Unitree G1 | `ContactSensor` 在双脚踝 link 上 `track_air_time=True` |
| `GeneLab-Terrain-Showcase-v0` | Unitree G1 | `TerrainGeneratorCfg` 1×5 行覆盖 5 种内置 sub-terrain |
| `GeneLab-Curriculum-Showcase-v0` | Unitree G1 | 5×5 RandomRough 网格上的 `terrain_levels_vel` 课程 |
| `GeneLab-Actuator-Showcase-v0` | Franka | `IdealPDActuator` 驱动机械臂 (force 通道)；记录跟踪误差 |
| `GeneLab-Recording-Showcase-v0` | Franka | `genelab.recording` —— 实时 PyQt + MPL 绘图，外加 NPZ + CSV 数据落盘 |

## 安装

```bash
uv sync --extra torch-cu128       # 选与本机匹配的 torch-* extra
uv pip install -e examples/genelab_showcase

uv run genelab list tasks | grep Showcase
```

装完后 showcase 扩展通过 `genelab.extensions` entry point 自动加载，不需要 `--import`。

## 传感器

```bash
uv run genelab play GeneLab-Sensors-Showcase-v0 --vis --steps 200
```

Joint 1 跟一段慢正弦 (±0.5 rad，4 秒周期) 摆动，机械臂其余关节保持 Menagerie home 姿态。
每 20 控制步 dump 一次：

- `logs/showcase/sensors/rgb_<step>.png` — 腕部相机 RGB，160×120。
- `logs/showcase/sensors/depth_<step>.png` — 腕部相机深度，16-bit 灰度，按相机
  [near, far] = [0.02, 2.5] m 的裁剪范围拉伸。
- `logs/showcase/sensors/frame.log` — 单行日志，记录 IMU 姿态、IMU 线/角加速度，以及
  末端和 link 7 在 base 坐标系下的位置。

!!! warning "BatchRenderer 是唯一支持并行 env 的渲染器"
    env cfg 已经设 `InteractiveSceneCfg.batch_render=True`。所以这个 showcase 需要
    Linux x86-64 + CUDA + 编译了 Madrona 的 Genesis。其它平台上 env 构造会在 Genesis
    分配 renderer 时立刻报错。

**See also**：[`CameraSensor` / `IMUSensor` / `FrameTransformerSensor`](../concepts/sensors.md)。

## Ray-cast 形态

```bash
uv run genelab play GeneLab-RayCast-Showcase-v0 --vis --steps 200
```

3 个 `RayCastSensor` 并列挂在 Franka 底座上——一个 `GridPattern` (81 条射线、0.8×0.8 m)、
一个 `RingPattern` (32 横 × 1 纵，±180° 满圈)、一个 `HemispherePattern` (128 条 Fibonacci
分布射线、70° 极轴朝下)。每 20 步往 `logs/showcase/raycast/distances.log` 追加三行：

```
step=0020
  raycast_grid: rays=81 min=… mean=… max=…
  raycast_ring: rays=32 min=… mean=… max=…
  raycast_hemi: rays=128 min=… mean=… max=…
```

`max` 顶到所配的 `max_distance`；没打到地面的射线 (极轴朝下时极少) 也会取这个上限。

**See also**：[Ray-cast 形态](../concepts/sensors.md#ray-cast-patterns)。

## 接触与离地时间

```bash
uv run genelab play GeneLab-Contact-Showcase-v0 --vis --steps 200
```

G1 在默认站立姿态稳定下来，双脚很快与地面接触。`logs/showcase/contact/air_time.log` 每
20 步记录：双脚的 `found` 旗、运行中的 `current_contact_time` / `current_air_time`，以及
最近一次完整的 `last_contact_time` / `last_air_time` 快照。

每 4 秒一次完整 reset (`episode_length_s=4`) 让 air-time 计数走过若干边界，snapshot
语义在日志里看得清。

**See also**：[`ContactSensor`](../concepts/sensors.md#contactsensor)。

## 地形

```bash
uv run genelab play GeneLab-Terrain-Showcase-v0 --vis --steps 200
```

一行 5 列的 sub-terrain 把 5 种内置地形铺到同一个场景里：

| 列 | Sub-terrain | 可见特征 |
|---|---|---|
| 0 | `FlatPatchCfg` | z=0 平面参考 |
| 1 | `PyramidStairsCfg(step_width=0.4, step_height=-0.08)` | 同心下降阶梯 |
| 2 | `RandomRoughCfg(min=-0.08, max=0.08)` | 均匀抖动 |
| 3 | `SlopeCfg(slope=-0.25)` | 线性斜坡 |
| 4 | `WaveCfg(num_waves=2, amplitude=0.08)` | 正弦起伏 |

G1 在 row 中点投放，保持默认姿态自由下落。一个朝下的 `GridPattern` ray-cast 挂在
pelvis 上，把局部 heightfield 响应写进 `logs/showcase/terrain/terrain.log`。

**See also**：[Terrains](../concepts/terrains.md)。

## 课程

```bash
uv run genelab play GeneLab-Curriculum-Showcase-v0 --vis --steps 400
```

16 个 G1 实例分布在一个 5×5 `RandomRoughCfg` 网格上，行号控制粗糙幅度。runner 每 30
控制步把一半 env 瞬移 1.5 m，触发 auto-reset 时 `walked > distance_threshold` 分支让它们
升级；剩下的 env 原地不动，逐步降级到 level 0。`logs/showcase/curriculum/levels.log`
每 30 步追加 per-env level 向量加一个行级直方图。

**See also**：[Manager 与 MDP term](../concepts/managers.md)、[Terrains](../concepts/terrains.md)。

## 执行器

```bash
uv run genelab play GeneLab-Actuator-Showcase-v0 --vis --steps 200
```

Franka 改用 `IdealPDActuator` 驱动 7 个机械臂关节——`stiffness=400` / `damping=80` 数值
与 asset zoo 默认相同，但走的是 `control_dofs_force` 力通道，不再走 Genesis 内建 PD。
对 joint 1 发 0.8 rad 正弦目标；`logs/showcase/actuators/tracking.log` 每 20 步记录目标 vs
实际位置和关节速度，可以和 sensors showcase (默认 ImplicitPDActuator) 直接对比跟踪质量。

**See also**：[Actuators](../concepts/actuators.md)。

## 数据录制

```bash
uv run genelab play GeneLab-Recording-Showcase-v0 --vis --steps 400
```

Franka 的 hand link 上挂一个 `IMUSensor`，env cfg 上配三个 `RecordingCfg` 数据流：

- IMU `lin_acc_b` → `PyQtPlotCfg` 实时窗口 **+** `CSVFileCfg` 写到
  `logs/showcase/recording/lin_acc.csv`。
- IMU `orientation` → `NPZFileCfg` 按 episode 落盘到
  `logs/showcase/recording/orientation_*.npz`（`save_on_reset=True`）。
- 一个自定义 callable，返回 `env.robot_state.joint_pos[:, 0]` → `MPLPlotCfg` 实时窗口。

脚本 runner 只做一件事：用 0.5 rad / 4 s 的正弦摆动 joint 1。recorder 由 Genesis 自己的
recorder 线程驱动，所以这个 showcase 不需要重载 `_dump`。

**See also**：[数据录制与绘图](../concepts/recording.md)。

!!! tip "Smoke-test 预算"
    单 env 默认 200 / 400 步，每个 showcase 都在 1 分钟内跑完。需要积累更长日志证据再
    加大 `--steps`；并行 env 数量对 showcase 没有好处。

## Log 输出

每个 showcase 写到 `logs/showcase/<slug>/`，slug 依次是 `sensors`、`raycast`、`contact`、
`terrain`、`curriculum`、`actuators`、`recording`。PNG / CSV / NPZ dump 和文本日志同目录，
比对运行时归档一个目录即可。

## See also

- [传感器](../concepts/sensors.md)
- [Terrains](../concepts/terrains.md)
- [Actuators](../concepts/actuators.md)
- [数据录制与绘图](../concepts/recording.md)
- [Manager 与 MDP term](../concepts/managers.md)
- [场景与实体](../concepts/scene.md)
