# 数据录制与实时绘图

`genelab.recording` 将 Genesis 的录制器 + 实时绘图栈封装为声明式 `RecordingCfg`。
一个场景可以声明多个录制项，每项把一个数据源（传感器、articulation 字段、或自定义
回调）连接到一个或多个输出（PyQt/Matplotlib 实时图、NPZ/CSV 文件、MP4/AVI 视频）。
真正的采样由 Genesis 的录制器线程完成 —— Genelab 只负责描述要录什么、把 env 的
生命周期接好。

## 快速开始

```python
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.recording import NPZFileCfg, PyQtPlotCfg, RecordingCfg
from genelab.sensor import IMUSensorCfg

scene = InteractiveSceneCfg(
    sensors=(IMUSensorCfg(name="cart_imu", link_name="cart"),),
    recordings=(
        RecordingCfg(
            name="cart_acc",
            source="cart_imu",
            field="lin_acc_b",
            outputs=(
                PyQtPlotCfg(title="cart linear acc", labels=("ax", "ay", "az")),
                NPZFileCfg(filename="logs/cart_acc.npz"),
            ),
        ),
    ),
)
```

每次 `env.step(...)` 就会按控制步采样 `cart_imu.data.lin_acc_b`，实时推送到 PyQt
窗口，并累积一个 `.npz` 文件在 `env.close()` 时写出。runner 里不需要手写任何数据
搬运代码。

## 心智模型

* **一个 `RecordingCfg` = 一个数据源 × N 个输出**。所有输出共享同一个数据 callable。
* **注册发生在 scene-build 时刻**。Genesis 的 `add_recorder` 标了 `@assert_unbuilt`，
  所以录制配置放在 `InteractiveSceneCfg`（紧挨 `sensors`），不在 env 层的
  `*_cfg` 字典里。小小的 `RecorderBridge` 把还未构造的 env 引用透传给回调。
* **采样在 `gs_scene.step()` 内部触发**。录制器跑在 Genesis 的线程上 —— PyQt 实时
  图不会阻塞 env 主循环。

## 数据源

`RecordingCfg.source` 可以是：

* **传感器名（字符串）**。配合 `field=...` 从该传感器的 `data` 里取一个张量
  （`"lin_acc_b"`、`"orientation"`、`"rgb"` 等）。如果省略 `field`，要求
  `sensor.data` 本身就是普通张量 —— 数据类负载会抛出清晰的错误。
* **可调用对象**。零参或单 `env` 参数；arity 在注册时检测一次。适合裸的
  articulation 状态或没有专门 sensor 包装的派生信号。

```python
# 传感器 + dotted field
RecordingCfg(name="imu_ori", source="imu", field="orientation", outputs=(...,))

# 零参 callable
RecordingCfg(name="wall_clock", source=time.monotonic, outputs=(...,))

# 带 env 的 callable
RecordingCfg(
    name="q0",
    source=lambda env: env.robot_state.joint_pos[:, 0],
    outputs=(...,),
)
```

`RecordingCfg.env_idx`（默认 0）会把多 env 的首维度切片到单 env 再交给下游 sink；
传 `env_idx=None` 可保留完整批次（适合把所有 env 一次性 dump 到 NPZ）。

## 输出参考

| Cfg | 后端 | 适用场景 |
|-----|------|---------|
| `PyQtPlotCfg` | `pyqtgraph`（线程化） | 实时绘图首选 —— Genesis 支持的所有平台都能用 |
| `MPLPlotCfg` | `matplotlib` | PyQt 不可用时的 fallback；macOS 下走主线程，可能卡顿 |
| `NPZFileCfg` | `numpy.savez_compressed` | 缓存到内存，`env.close()` 时一次写出（或开启 `save_on_reset=True` 按 episode 切文件）|
| `CSVFileCfg` | `csv.writer` | 行流式；`header=("a","b",...)` 设列名，`save_every_write=True` 每行 flush |
| `VideoFileCfg` | `cv2.VideoWriter` | 来自 `CameraSensor` 的 MP4/AVI；只能配相机源 |

所有五个 cfg 都接受 `RecorderOptions` 的公共字段：`hz`、`buffer_size`、
`buffer_full_wait_time`。显式设置 `hz` 总会覆盖 Genelab 的默认策略。

## 采样频率与传感器缓存

`Sensor.data` 在每个控制步缓存。如果录制器按物理 tick 采样，会得到 `decimation` 个
重复行。Genelab 自动处理这一点：

* **传感器源**默认 `hz = 1 / (sim.dt * decimation)`（即控制率）。
* **Callable 源**默认 `hz = None`（每个物理 tick 都采）—— 假设 callable 读裸状态，
  每个 tick 都会变。

显式重写时直接在输出 cfg 上设 `hz`：

```python
NPZFileCfg(filename="raw.npz", hz=200.0)  # 强制 200 Hz，无视 sim.dt
```

## Episode 边界与 `save_on_reset`

当至少有一个输出开了 `save_on_reset=True`，env 的 auto-reset 会调用
`gs_scene._recorder_manager.reset()`，flush 并轮换文件输出（NPZ 计数器自增、CSV
重写表头）。`ManagerBasedRlEnv.__init__` 里的首次 reset 会被跳过，所以文件计数从 0
开始，符合直觉（`pole_0.npz`、`pole_1.npz`、…）。最后没写完的缓存由 `env.close()`
flush 出去。

绘图器不受 reset 影响（历史是滚动窗口）。

## 多 env

绘图器永远只读一个 env。用 `env_idx` 选哪个 —— 默认 0，几乎总是 RL 调试时要的。

文件写出器可以一次记录所有 env：设 `env_idx=None`，就会存完整 `(num_envs, ...)`
张量。把绘图器和 `env_idx=None` 配在一起会在首次采样时报错 —— 拆成两个 recording。

## 相机与视频

一个 `CameraSensor` 源配单个 `VideoFileCfg` 就能写出单 env 的视频：

```python
RecordingCfg(
    name="wrist_video",
    source="wrist_cam",
    outputs=(VideoFileCfg(filename="logs/wrist.mp4", env_idx=0, fps=30),),
)
```

注意：

* 相机必须开 `render_rgb=True` —— 仅深度相机会在运行时报错。
* `BatchRenderer` 只支持 Linux x86-64 + CUDA。macOS 下默认 Rasterizer 仍会输出
  单 env 的 `(1, H, W, 3)` 张量，`env_idx=0` 正确切片。
* 绘图器配相机源会在注册阶段被拒绝 —— 绘图器吃不下 H×W×3 帧。

## 自定义 callable 和任意信号

任何 env 上能取到的量都能录。例子：

```python
# 关节扭矩（actuator 之后）
RecordingCfg(
    name="joint_effort",
    source=lambda env: env.robot_state.dof_force[:, env.joint_names.index("joint1")],
    outputs=(NPZFileCfg(filename="logs/effort.npz"),),
)

# 奖励项快照
RecordingCfg(
    name="track_reward",
    source=lambda env: env.reward_manager._term_sums["track_lin_vel"],
    outputs=(PyQtPlotCfg(title="track reward"),),
)
```

Callable 在每次录制 tick 都会跑（默认每个物理步）。保持轻量 —— 重计算应该放进
Genelab sensor。

## 可选依赖

实时绘图的后端都在 Genelab 的 `recording` extra 里：

```bash
pip install "genelab[recording]"   # 拉 pyqtgraph + PyQt5 + matplotlib
```

showcase 包通过 `genelab[recording]` 间接依赖，`uv pip install -e
examples/genelab_showcase` 已经会自动带上 —— 不需要手装。只写文件的录制
（NPZ / CSV / VideoFile）不需要额外依赖（numpy + PyAV 都是 Genesis 自带）。

## 另见

- [传感器](sensors.md)
- [Scene 与实体](scene.md)
- Examples → Showcase 中的 `Showcase-Recording` 任务。
