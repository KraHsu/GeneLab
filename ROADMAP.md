# GeneLab Development Roadmap

> 本文档面向 GeneLab 的维护者、贡献者与下游研究用户，描述未来 2–3 个 release 周期内的演进方向。
> 它**不是**承诺，而是当前的优先级共识；具体进度跟踪请看 GitHub Milestones / Issues。
>
> Last reviewed: 2026-05-20 · against `dev` @ `26c2da5`.

## 1. 定位与边界

GeneLab 的核心定位是：

> **Genesis 后端上的 manager-based 强化学习研究脚手架**，对标 Isaac Lab 的 API 形状，但**不复刻**它的全部能力。

设计上的边界（**非目标**）：

- **不重做仿真器**：物理 / 渲染 / 求解器全部由 Genesis 提供。GeneLab 的 `sim/` 层只暴露 Genesis 配置，不做物理层抽象。
- **不重做 RL 算法**：算法实现来自 `rsl_rl` / `skrl` / `stable-baselines3`。GeneLab 只提供**统一的 backend / vec-env / config / CLI 适配层**。
- **不做端到端真机部署框架**：sim2real 路径只保证「能把训好的 policy 导出成可独立运行的 nn.Module」，不绑定具体机器人 SDK。
- **不做大规模分布式**：单机多卡（torchrun）是必备；多节点 / federated / RLHF 风格的训练栈不在范围内。

GeneLab 的差异化价值：

1. **Genesis 高速并行**作为后端，零拷贝 GPU 多 env。
2. **Manager-based MDP**像 Isaac Lab 一样易写易扩，但更轻量（无 USD / Kit / NVIDIA-only 锁定）。
3. **多算法后端可插拔**（rsl_rl / skrl / SB3），统一 `genelab train / play` CLI。
4. **Locomotion + Manipulation 双线**：G1 / Anymal-C / Franka pick-and-place 都是一等公民。

---

## 2. 当前状态快照（dev @ 88ad2aa）

> 上一版快照写于 `26c2da5`（M1 之前）。本版反映 **M1 全部完成 + M2.2–M2.5 完成**
> 以及 R0–R7 架构重构（§9）全部落地后的真实状态。

✅ **已具备**
- 三 RL 后端 + 后端抽象层（`src/genelab/rl/backends/`）
- Action 项：JointPosition / DifferentialIK / Binary & ContinuousGripper
- 任务示例：Inverted Pendulum、G1 Velocity、Franka Pick-And-Place × 5 变种（含 SAC+HER）
- **可复现性闭环（M1 完成）**：`genelab eval`（`eval.json`）+ 训练期 `EvalCallback` + best-model；
  `genelab export`（TorchScript / ONNX，纯 `nn.Module`）；多 seed CLI（`--seeds`/`--parallel`）；
  reference-runs 文档；死代码（`resume`/`load_run`/`load_checkpoint`）已清理；`joint_acc_l2` 占位已明示
- **Sim2Real 硬约束（M2.3 / M2.4 完成）**：termination `joint_pos/vel_out_of_limit`、`contact_force_limit`；
  reward `lin_vel_z_l2` / `base_height_l2` / `alive_bonus` / `applied_torque_l2` / `joint_vel_limits`
- **学习型 actuator（M2.5 完成）**：`MlpResidualActuator`（DCMotor base + TorchScript MLP 残差）
- **DR（M2.1 / M2.2 完成）**：COM / mass / friction / encoder-bias + interval-mode +
  `randomize_joint_stiffness_damping` / `randomize_actuator_deadzone`；plumbing：
  `RobotState.applied_torque`、`ArticulationCfg.joint_vel_limit`、actuator gain-scale/deadzone
- **Observation noise（M2.6 完成）**：`Unoise`/`Gnoise` + `ScaledNoise`/`CorrelatedNoise`/`BiasDrift`
- **Sim2Real 部署文档（M2.7 完成）**：`docs/best-practices/sim2real.{en,zh}.md`
- Curriculum：terrain levels + velocity range
- 传感器：IMU / Contact / FrameTransformer / RayCast(3 模式) / Camera(RGB+depth) / TerrainHeight /
  **ForceTorque（joint-FT，M3.5 完成）**
- **Sim rigid 选项（M3.7 完成）**：`SimulationCfg` 暴露 Genesis `RigidOptions` 的 contact / solver /
  constraint-damping 共 8 字段
- **更多 sub-terrain（M3.2 完成）**：`DiscreteObstacles` / `SteppingStones` / `Fractal`
- **Terrain curriculum 生效（M3.3 完成）**：`SubTerrainCfg.difficulty` + `curriculum=True` 行难度排序
- **Camera segmentation（M3.4 完成）**：`CameraSensorCfg.render_segmentation`（object-index / colorized）
- **多机器人 API（M3.6 完成，ADR-0012 S1–S6）**：`env.articulations[name]` + `SceneEntityCfg.name` /
  `asset_name` / `SensorCfg.entity_name` 路由；单数 `env.robot*` / name-table 访问器已删除
- **Benchmark 命令（M3.8 部分）**：`genelab benchmark --suite` + 回归门
- **8 个 asset_zoo robots（M3.1 完成）**：anymal_c / cartpole / franka / g1 / go1 / **h1** /
  **ur10e** / **allegro**，覆盖 locomotion / arm / dexterous（blob 托管于 `genelab-assets`、md5 验证）
- Recording：NPZ / CSV / video / 实时 PyQt & MPL plots；Teleop bridges：keyboard / DearPyGui
- torchrun 多卡训练
- **架构**：`lint-imports` 必过 CI 分层门禁（6/0）；pyright 在 `src/` 上 0 错误（不再几乎全关）

⚠️ **关键缺口**（详见 §4）—— **M1、M2、M3.1–M3.7 全部完成。** 唯一剩余项受当前环境阻挡（非代码问题）：
- benchmark 真实 reference numbers（M3.8 剩余）—— `genelab benchmark` 命令已落地，但跑 ≥8 个任务
  的真实数字 + vision 任务端到端需 Genesis GPU runtime + 训练好的 checkpoint（本环境无）。同理：新 asset
  的 live-spawn、多机器人 2-robot rollout、任何实际训练都 Genesis-gated。
- 可选扩展：6 轴 wrench / 指尖压力 tactile（M3.5 已交付 joint-FT）；ADR-0010 拆分（待第二实体类型）。

---

## 3. 设计原则

下面这组原则用来在「加新东西」时做取舍。所有 PR 评审都参照它们。

### P1 · 最少绑定（least lock-in）

任何新 API 必须问：

- **能否不绑死某个后端？** 例如 EvalCallback 要能跑在 rsl_rl / skrl / sb3 任一上。
- **能否不绑死 Genesis 特定特性？** 如果一个能力依赖 Genesis 0.4.x 内部接口，必须在 docstring 标注。
- **能否不绑死 NVIDIA GPU？** 至少给一条 CPU 路径用于 CI 与文档示例。

### P2 · 配置即接口（cfg-as-API）

GeneLab 的扩展面是 dataclass cfg + manager。**新功能优先表达为 cfg 字段，而非新类**。

- 反例：为「带 clip 的 JointPositionAction」新建一个 `ClippedJointPositionAction` 类。
- 正例：给 `JointPositionAction` 加 `clip: tuple[float, float] | None = None`。
- 例外：行为差异大到 cfg 选项数量爆炸时（e.g. IK vs JointPos），分两个类。

### P3 · 死字段即 bug（dead fields are bugs）

cfg 字段必须被读取，否则误导用户。当前 `RslRlOnPolicyRunnerCfg.resume / load_run / load_checkpoint` 就是反例。

- 实现一个字段 → 在文档 + 测试中验证它的行为。
- 暂时实现不了 → 用 `# TODO(<issue#>):` 标注，并在初始化时 `warnings.warn`。
- 永远不需要了 → 删掉。

### P4 · 占位即文档（stubs must announce themselves）

`rewards.py:joint_acc_l2` 返回 0 但函数名暗示它在算什么 — 这种 stub 必须 `raise NotImplementedError` 或加 `warnings.warn("stub, returns 0")`。

### P5 · 例子即测试（examples are tests）

每个 `examples/*` 包必须有对应的 `tests/test_*_examples.py`，至少跑 1 个 iteration 的 smoke 训练。CI 必须能在 CPU 上跑通。

### P6 · 默认即基线（defaults must converge）

每个 `examples/*` registered task 必须有**一组「跑就收敛」的默认 cfg**。研究者 fork 后改的应该是 reward weights / DR 范围，而不是「先 debug 为什么训不动」。

### P7 · 文档先行（docs-first for cross-cutting changes）

凡是动 `envs/` / `managers/` / `rl/backends/` / `configs.py` 的 PR，必须同步更新 `docs/concepts/` 或 `docs/best-practices/`。

---

## 4. 里程碑

每个里程碑独立可发布，**不强求按顺序**，但 M1 是 M2/M3 的前置条件（基线数字依赖 eval loop）。

### M1 · Research Reproducibility（核心可复现性）

> **One-liner**: 让任何人 `clone → train → eval → export` 都能拿到与 README 一致的数字。

**状态：✅ 全部完成。**

**目标产物**

| # | 交付 | 说明 |
|---|---|---|
| ✅ M1.1 | `genelab eval <task> <ckpt>` CLI | deterministic rollout × N episodes，输出 `eval.json`（return mean/std、length mean、success rate、wall-clock） |
| ✅ M1.2 | `EvalCallback` 训练期内嵌评估 | 三后端通用，每 K iter 跑一次 eval，更新 `best_model.pt` |
| ✅ M1.3 | `genelab export <task> <ckpt> --format {torchscript,onnx}` | 输出无 rsl_rl/skrl/sb3 依赖的纯 `nn.Module` |
| ✅ M1.4 | 多 seed CLI | `genelab train ... --seeds 1,2,3 --parallel 3` |
| ✅ M1.5 | 死代码清理 | 删除 / 实现 `RslRlOnPolicyRunnerCfg.resume / load_run / load_checkpoint`（已删除） |
| ✅ M1.6 | Stub 标记 | `rewards.py:joint_acc_l2` 等占位项明示（`UserWarning`） |
| ✅ M1.7 | Reference runs 文档 | `docs/best-practices/reference-runs.md` 列出 5 个 registered 任务的 seed 1/2/3 训练曲线、最终 return、收敛步数 |

**API 草案 — `genelab eval`**

```bash
genelab eval Genelab-Inverted-Pendulum-v0 logs/.../model_1000.pt \
    --num-envs 64 --episodes 100 --seed 0 \
    --deterministic --out eval.json
```

输出 `eval.json`：
```json
{
  "task": "GeneLab-Inverted-Pendulum-v0",
  "checkpoint": "logs/.../model_1000.pt",
  "num_episodes": 100,
  "metrics": {
    "return_mean": 487.3,
    "return_std": 22.1,
    "length_mean": 998.4,
    "success_rate": 0.96
  },
  "wall_clock_seconds": 18.2,
  "seed": 0,
  "deterministic": true
}
```

**API 草案 — `genelab export`**

```bash
genelab export GeneLab-Franka-Pick-And-Place-v0 logs/.../model_500.pt \
    --format torchscript --out policy.ts
genelab export GeneLab-Franka-Pick-And-Place-v0 logs/.../model_500.pt \
    --format onnx --out policy.onnx --opset 17
```

导出后的 policy 必须满足：
- 输入：单一 `obs` tensor 或 dict tensor（与训练时 obs group schema 一致）
- 输出：action tensor
- 不依赖 rsl_rl / skrl / sb3，可在仅装 torch 的环境运行
- 包含一段 `metadata.json` 描述 obs schema、action schema、归一化参数

**验收**

- `pytest tests/test_eval.py tests/test_export.py` 全绿
- CI 跑出至少 1 个任务的 reference run，曲线截图入文档
- `genelab eval --help` / `genelab export --help` 在 docs/cli 自动生成

**关键文件**
- 新增：`src/genelab/rl/evaluator.py`、`src/genelab/rl/exporter.py`、`src/genelab/cli/_eval.py`、`src/genelab/cli/_export.py`
- 修改：`src/genelab/rl/backends/{rsl_rl,skrl,sb3}.py`（加 `inference_policy` 统一接口）
- 新增：`tests/test_eval.py`、`tests/test_export.py`、`docs/best-practices/reference-runs.{en,zh}.md`

---

### M2 · Sim2Real Hardening（仿真到现实的鲁棒性）

> **One-liner**: 让用 GeneLab 训出来的 policy 部署到真机时尽量不掉链子。

**状态：✅ 完成（M2.1–M2.7 全部落地，PR #117–#124）。** M2.1 交付了两个可行的新 DR
（`randomize_joint_stiffness_damping` / `randomize_actuator_deadzone`）；其余 M2.1 项已被覆盖或被
Genesis 阻挡（`push_robot` ≈ 现有 `push_by_setting_velocity`；`randomize_imu_bias` ≈ IMU 传感器自带
的 per-env bias；`randomize_restitution` / `randomize_gravity` Genesis 无对应 setter）。**M2 完成。**

**目标产物**

| # | 交付 | 说明 |
|---|---|---|
| ✅ M2.1 | DR 项扩展 | 新增 `randomize_joint_stiffness_damping` / `randomize_actuator_deadzone`（PR #122）；`push_robot`/`randomize_imu_bias` 已被现有功能覆盖；`randomize_restitution`/`randomize_gravity` Genesis 无 setter（详见上方状态说明） |
| ✅ M2.2 | Interval-mode DR | EventManager 支持 `mode="interval"`，让某些 DR 在 episode 中途触发 |
| ✅ M2.3 | Termination 越界保护 | `joint_pos_out_of_limit` / `joint_vel_out_of_limit` / `contact_force_limit`（PR #117/#118/#119） |
| ✅ M2.4 | Reward 硬约束补齐 | `lin_vel_z_l2` / `applied_torque_l2` / `joint_vel_limits` / `base_height_l2` / `alive_bonus`（PR #117/#118） |
| ✅ M2.5 | 学习型 actuator | `MlpResidualActuator`（DCMotor base + TorchScript MLP residual，PR #120） |
| ✅ M2.6 | Observation noise 扩展 | `ScaledNoise`、`CorrelatedNoise`、`BiasDrift`（PR #123） |
| ✅ M2.7 | Deployment recipe 文档 | `docs/best-practices/sim2real.{en,zh}.md`「Harden for Sim2Real」(PR #124) |

**设计要点**

- DR 项命名遵循 `randomize_<物理量>` 模式，方便 grep。
- `mode="interval"` 不引入新 manager，复用 `EventManager` 的现有 hook，加 `interval_range_s: tuple[float, float]` 字段。
- 学习型 actuator 不在 `actuator/` 强行内嵌训练流程，只提供「加载 .pt 权重 + 前向」接口，权重训练放 `examples/` 或下游项目。
- Termination 项默认是 `time_outs=False`（即触发即真终止，而非 timeout），但 cfg 暴露 `time_outs` 字段供用户改。

**验收**

- `examples/unitree/` 的 G1 velocity 加入 M2.1–M2.4 配置后，relative-perf vs 当前实现不下降（同 seed 训练 return 不掉 >10%）。
- Sim2real 文档至少有一个完整 walkthrough（建议用 G1 或 Anymal-C）。

**关键文件**
- 新增 / 修改：`src/genelab/mdp/dr/{joint,external_force,gravity,imu,actuator}.py`
- 修改：`src/genelab/managers/event_manager.py`（interval mode）
- 修改：`src/genelab/mdp/terminations.py`、`src/genelab/mdp/rewards.py`、`src/genelab/mdp/noise.py`
- 新增：`src/genelab/actuator/mlp_residual.py`

---

### M3 · Platform Breadth（平台广度）

> **One-liner**: 把 GeneLab 从「能跑 demo」推到「能做严肃 benchmark」。

**状态：M3.1–M3.7 ✅ 完成（PR #126–#144）；仅 M3.8 的真实数字待跑。** 多机器人 API（M3.6）按
ADR-0012 的 S1–S6 全部落地；M3.1 新增 H1 / UR10e / Allegro 三个 asset（zoo 5 → 8，blob 已托管 + md5
验证）。**唯一剩余**：M3.8 的真实 reference numbers——`genelab benchmark` 命令已落地，但跑 ≥8 个任务的
真实数字 + vision 任务端到端需要 Genesis GPU runtime + 训练好的 checkpoint（本环境无，非代码问题）。

**目标产物**

| # | 交付 | 说明 |
|---|---|---|
| ✅ M3.1 | 资产扩充 | +3：**Unitree H1**（19-DoF 人形, #141）/ **UR10e**（6-DoF 工业臂, #142）/ **Allegro**（16-DoF 灵巧手, #144）。均 Menagerie 镜像，blob 托管于 `genelab-assets`、md5 端到端验证；live spawn 仍 Genesis-gated。zoo 5 → 8 robots，覆盖 locomotion / arm / dexterous |
| ✅ M3.2 | 更多 sub-terrain | `DiscreteObstacles` / `SteppingStones` / `Fractal`（#129）。「gaps」无 Genesis 分支、mesh import 是 `height_field` 另一路径——均后置 |
| ✅ M3.3 | Terrain curriculum 真生效 | `SubTerrainCfg.difficulty` + `curriculum=True` 行难度排序（#130），配合 `mdp.terrain_levels_vel` |
| ✅ M3.4 | Camera segmentation | `CameraSensorCfg.render_segmentation`（object-index / colorized，#131）。point cloud（depth+intrinsics 反投影）后置 |
| ✅ M3.5 | F/T sensor + tactile array | `ForceTorqueSensor`（per-joint 反作用力矩，PR #126）；6 轴 wrench / 指尖压力阵列后置 |
| ✅ M3.6 | 多机器人 API | **ADR-0012 实现完成（S1–S6）**。env 持 `articulations` dict；`SceneEntityCfg.name` / `asset_name` / `SensorCfg.entity_name` 路由到具名实体；S6 删除单数 `env.robot*` + name-table 访问器（破坏性），统一 `env.articulations[name].*`。验收：`tests/test_multi_robot.py`（live 2-robot rollout Genesis-gated） |
| ✅ M3.7 | SimulationCfg 字段扩展 | 暴露 Genesis `RigidOptions`：contact / solver / constraint-damping 共 8 字段（PR #127）。CCD：Genesis 无对应 knob，N/A |
| 🚧 M3.8 | Benchmark suite | **`genelab benchmark --suite suite.json` 命令 + suite/report schema + 回归门（`--reference`/`--tolerance`）已落地**（rl/benchmark.py，mock 单测）。剩「≥8 个任务 + 真实 reference numbers + vision 任务端到端」待 Genesis runtime + checkpoint + asset（受阻，同 M3.1） |

**设计要点**

- 多机器人 API 改动较大。**RFC 已起草：ADR-0012**（采用 explicit-routing / 破坏性设计——
  去掉单数 `env.robot*` 访问器，term 一律经 `asset_name` 路由）。实现按 ADR-0012 的 S1–S6
  分片推进。
- Benchmark suite 不是 examples 的简单堆砌；要有一个统一的 `genelab benchmark` 命令一键 run。
- Camera segmentation 通道写在 `CameraSensorCfg` 上而非新 sensor 类型，遵守 P2。

**验收**

- `genelab list robots` 至少 8 个
- 至少一个 vision-based 任务能端到端训练（如「Franka stacking with depth obs」）
- 多机器人 API（M3.6）：一个 2-robot 示例任务 + 集成测试（两 articulation、各自 action / reward）
  通过，即 ADR-0012 的验收门（S6）。**注意：ADR-0012 选择破坏性迁移，现有 examples 会随 S5 一起迁移、
  不保证向后兼容**（与早期「向后兼容」设想相反，已在 ADR 中记录变更）。

**关键文件**
- 新增：`src/genelab/asset_zoo/{a1,h1,ur10,allegro}.py`
- 修改：`src/genelab/terrains/sub_terrain.py`、`src/genelab/terrains/generator.py`
- 修改：`src/genelab/sensor/camera.py`、新增 `src/genelab/sensor/force_torque.py`
- 修改：`src/genelab/envs/manager_based_rl_env.py`、`src/genelab/managers/scene_entity_cfg.py`
- 修改：`src/genelab/configs.py`（`SimulationCfg`）

---

## 5. 跨里程碑的长期方向

下面是值得做但**不进 M1-M3**的方向。要做的话单独立 milestone。

- **Offline RL / Demo collection 管线**：HDF5 数据集 + `genelab collect` CLI + offline replay buffer adapter。要做需先有个 motivating 用户故事。
- **MultiAgent / Self-play**：M3.6 多机器人 API 已落地（`env.articulations[name]` + 按名路由），
  这一前置已解除；剩下的是 per-agent policy / 对抗式 reward 编排。
- **CleanRL / Tianshou 后端**：后端抽象层做得很干净，接入成本低，但缺少 motivating user。
- **MLflow / Aim logger**：用户够多再加。
- **VR / 3D Spacemouse teleop**：研究阶段非关键。
- **大规模 distributed (multi-node)**：与 GeneLab 的轻量定位冲突，明确不做。

---

## 6. 设计 RFC 模板

跨切面（涉及 `envs/` / `managers/` / `rl/backends/` / `configs.py`）的改动，建议**先提 RFC 后实现**。RFC 写在 `docs/rfcs/NNNN-title.md`（这是新增目录），结构：

```markdown
# RFC NNNN: <Title>

- Author: <handle>
- Status: Draft | Accepted | Rejected | Superseded
- Date: YYYY-MM-DD

## Motivation
（为什么要做）

## Proposal
（API 草案、cfg 字段、行为变更）

## Alternatives
（考虑过的其他方案）

## Compatibility
（向后兼容、迁移路径）

## Open Questions
（待定）
```

第一个候选 RFC 是 **M3.6 多机器人 API**（影响 `ManagerBasedRlEnv` 和所有 manager term 的 `entity_cfg` 解析）。

---

## 7. 贡献者上手路径

对照里程碑挑选合适的入手点：

- **小改动 / 上手第一 PR**：M1.5（清死代码）、M1.6（标 stub）、M2.4 中的单个 reward 项、M2.3 中的单个 termination 项。
- **中等改动 / 已熟悉项目**：M1.1 eval CLI、M2.1 中单个 DR 项、M3.1 中单个 asset。
- **大改动 / 需先提 RFC**：M3.6 多机器人 API、M2.5 学习型 actuator、M3.7 SimulationCfg 扩展。

每个 PR 都对照 §3 的 7 条原则自查。

---

## 8. 评审与发布节奏

- **`main` 是发布分支**：只接 merge from `dev`。
- **`dev` 是集成分支**：feature 分支合到 dev，定期（每 2-4 周）合 main 发版本。
- **每个 release tag** 同步更新 `CHANGELOG.md` 与本 ROADMAP 的「Last reviewed」。
- **每个里程碑发布**做一次 ROADMAP 整体复审，淘汰已完成项、调整下一里程碑优先级。

---

## 9. Architectural Refactor Roadmap / 架构重构路线图

**✅ COMPLETE — R0–R7 + post-R7 cleanup all shipped.** The refactor decoupled the layers,
broke the `rl.runner ↔ rl.backends` cycle, pushed parsing/abstractions to their right homes,
and made the layering a **required CI gate**. 17 production PRs (#81–#115) + docs/test-split.
Per-PR detail lives in the git history, `CHANGELOG.md` [0.2.0], and `plans/adr/` (ADR-0001–0011).

| Phase | Outcome | ADR | PRs |
|---|---|---|---|
| R0 — baseline & tooling | importlinter baseline + optional-dep / help-snapshot tests | — | #81–#83 |
| R1 — break rl.runner↔backends cycle | `rl/_helpers.py`; grimp static-cycle test | 0001 | #84 |
| R2 — small abstractions | `BaseTermManager` + PD-gain / joint-match / algo-taxonomy dedup | 0002, 0003 | #85–#89, #108 |
| R3 — domain-owned parsing | cfg `from_args` / `play_retargeted_keys` classmethods | 0005 | #90, #91 |
| R4 — CLI decomposition | `cli/_{dispatch,multi_seed,distributed,help,resolve}.py` (479 LoC) | 0004, 0011 | #92–#95, #115 |
| R5 — task-reward split | motion-tracking rewards out of `mdp/rewards.py` | 0006 | #97, #98 |
| R6 — vecenv rename | adapters → `rl/vecenvs/` | 0007 | #100 |
| R7 — extensions API + blocking lint | `genelab.extensions`; `lint-imports` required gate (6/0) | 0008, 0009 | #101–#106 |
| post-R7 cleanup | shim removal, ADR-0010 size guard, `test_cli.py` split | — | #108–#115 |
| deferred — entity/articulation split | not done; criteria unmet (reconsidered post-M3.6, still deferred) | 0010 | — |

**Enforced invariants** — six `lint-imports` contracts (see `pyproject.toml [tool.importlinter]`
+ CLAUDE.md "Architectural invariants"): optional-dep boundary, lazy `lab.py`, domain purity
(incl. `asset_zoo` may fetch), `rl.backends ⊬ rl.runner`, torch-free `configs.py`, and
bottom-bands-never-reach-up. A PR adding a cross-layer import fails CI. The R-phase *process*
notes + per-slice lessons (kept for future similar work) live in `CLAUDE.md`.

---

## 附录 A · 当前命名约定速查

| 类别 | 命名模式 | 例 |
|---|---|---|
| Action term | `<动作语义>Action` | `JointPositionAction`、`DifferentialIKAction` |
| Reward term function | `<语义>_<距离度量>` | `track_linear_velocity_xy_exp`、`action_rate_l2` |
| Termination term function | `<条件>` 或 `bad_<x>` | `time_out`、`bad_orientation` |
| DR term function | `<物理量>_<操作>` | `body_com_offset`、`geom_friction` |
| Curriculum term function | `<目标量>` | `terrain_levels_vel` |
| Sensor cfg | `<类型>SensorCfg` | `CameraSensorCfg`、`ContactSensorCfg` |
| Backend module | `rl/backends/<lib>.py` | `rsl_rl.py`、`skrl.py`、`sb3.py` |
| Task ID | `GeneLab-<Robot>-<Task>-v<N>` | `GeneLab-Franka-Pick-And-Place-Cartesian-v0` |

新增功能时**先 grep 现有命名再起名**。

---

## 附录 B · 「我应该做什么」决策树

```
你想加一个新功能？
│
├── 它能用现有 cfg 字段表达吗？
│   ├── 能 → 改 cfg，不加类（遵守 P2）
│   └── 不能 ↓
│
├── 它跨 manager / env / backend 吗？
│   ├── 是 → 先提 RFC（§6）
│   └── 否 ↓
│
├── 它是 M1-M3 路线上的事吗？
│   ├── 是 → 在对应 milestone 下开 PR
│   └── 否 → 先开 issue 讨论必要性
│
└── 同步加测试 + 文档 + CHANGELOG（遵守 P5, P7）
```
