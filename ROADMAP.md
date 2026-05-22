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

## 2. 当前状态快照（dev @ 26c2da5）

✅ **已具备**
- 三 RL 后端 + 后端抽象层（`src/genelab/rl/backends/`）
- Action 项：JointPosition / DifferentialIK / Binary & ContinuousGripper
- 任务示例：Inverted Pendulum、G1 Velocity、Franka Pick-And-Place × 5 变种（含 SAC+HER）
- DR 4 项（COM offset / mass offset / friction / encoder bias）
- Curriculum：terrain levels + velocity range
- 传感器：IMU / Contact / FrameTransformer / RayCast(3 模式) / Camera(RGB+depth) / TerrainHeight
- Recording：NPZ / CSV / video / 实时 PyQt & MPL plots
- Teleop bridges：keyboard / DearPyGui
- torchrun 多卡训练

⚠️ **关键缺口**（按 ROI 排序详见 §4）
- 独立 eval CLI 与 best-model selection
- Policy 导出（TorchScript / ONNX）
- 训练 / 测试基线数字（reference numbers）
- DR 项稀薄、无 interval mode
- Termination 缺关节越界保护，reward 缺 alive_bonus / lin_vel_z / torque_l2
- `config.resume / load_run / load_checkpoint` 三字段是死代码
- `joint_acc_l2` 是 return-0 占位
- pyright 几乎全关，多后端接入后 silent unknown 在扩大
- 多机器人 API 缺位（`articulations["robot"]` 硬编码）
- `sim/` 层是空壳，SimulationCfg 字段过少
- Camera 无 segmentation；无 LIDAR/F-T/tactile
- Terrain curriculum flag 未生效，sub-terrain 仅 5 种

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

**目标产物**

| # | 交付 | 说明 |
|---|---|---|
| M1.1 | `genelab eval <task> <ckpt>` CLI | deterministic rollout × N episodes，输出 `eval.json`（return mean/std、length mean、success rate、wall-clock） |
| M1.2 | `EvalCallback` 训练期内嵌评估 | 三后端通用，每 K iter 跑一次 eval，更新 `best_model.pt` |
| M1.3 | `genelab export <task> <ckpt> --format {torchscript,onnx}` | 输出无 rsl_rl/skrl/sb3 依赖的纯 `nn.Module` |
| M1.4 | 多 seed CLI | `genelab train ... --seeds 1,2,3 --parallel 3` |
| M1.5 | 死代码清理 | 删除 / 实现 `RslRlOnPolicyRunnerCfg.resume / load_run / load_checkpoint` |
| M1.6 | Stub 标记 | `rewards.py:joint_acc_l2` 等占位项明示 |
| M1.7 | Reference runs 文档 | `docs/best-practices/reference-runs.md` 列出 5 个 registered 任务的 seed 1/2/3 训练曲线、最终 return、收敛步数 |

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

**目标产物**

| # | 交付 | 说明 |
|---|---|---|
| M2.1 | DR 项扩展 | 补 `randomize_joint_stiffness_damping` / `randomize_restitution` / `push_robot`（脉冲外力）/ `randomize_imu_bias` / `randomize_gravity` / `randomize_actuator_deadzone` |
| M2.2 | Interval-mode DR | EventManager 支持 `mode="interval"`，让某些 DR 在 episode 中途触发 |
| M2.3 | Termination 越界保护 | `joint_pos_out_of_limit` / `joint_vel_out_of_limit` / `contact_force_limit` |
| M2.4 | Reward 硬约束补齐 | `lin_vel_z_l2` / `applied_torque_l2` / `joint_vel_limits` / `base_height_l2` / `alive_bonus` |
| M2.5 | 学习型 actuator | `MlpResidualActuator`（DCMotor base + MLP residual） |
| M2.6 | Observation noise 扩展 | `ScaledNoise`、`CorrelatedNoise`、`BiasDrift` |
| M2.7 | Deployment recipe 文档 | `docs/best-practices/sim2real.md` 描述「训练时该用哪些 DR、导出时该 dump 什么、部署端该怎么对齐」 |

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

**目标产物**

| # | 交付 | 说明 |
|---|---|---|
| M3.1 | 资产扩充 | 至少 +3 个：A1 / H1 / UR10 / Allegro 任二（按下游需求） |
| M3.2 | 更多 sub-terrain | gaps / stepping stones / discrete obstacles / mesh import |
| M3.3 | Terrain curriculum 真生效 | `TerrainGeneratorCfg.curriculum=True` 时按 mdp/curriculums 的进度自动调难度 |
| M3.4 | Camera segmentation + point cloud | 暴露 Genesis 的 semantic/instance ID 通道 |
| M3.5 | F/T sensor + tactile array | 至少 joint-FT；指尖压力可后置 |
| M3.6 | 多机器人 API | manager 层支持 `entity_cfg.name="robot_a"` 绑定特定 articulation，env 层去除 `["robot"]` 硬编码 |
| M3.7 | SimulationCfg 字段扩展 | 暴露 Genesis 的接触参数、求解器选项、CCD、constraint damping |
| M3.8 | Benchmark suite | 至少 8 个任务（含 locomotion / manipulation / dexterous / vision），每个都有 reference numbers |

**设计要点**

- 多机器人 API 改动较大。**先在 PR 中给一份「API 设计 RFC」征求意见**，再动代码。
- Benchmark suite 不是 examples 的简单堆砌；要有一个统一的 `genelab benchmark` 命令一键 run。
- Camera segmentation 通道写在 `CameraSensorCfg` 上而非新 sensor 类型，遵守 P2。

**验收**

- `genelab list robots` 至少 8 个
- 至少一个 vision-based 任务能端到端训练（如「Franka stacking with depth obs」）
- 多机器人 API 改完后所有现有 examples 不破坏（向后兼容 `articulations["robot"]` 仍工作）

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
- **MultiAgent / Self-play**：受限于 manager-based env 单 robot 假设，需要 M3.6 多机器人 API 先落地。
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

> **Scope of this section.** Internal architectural cleanup derived from
> [`plans/architecture/target-architecture.md`](plans/architecture/target-architecture.md).
> The M1–M3 milestones above ship *features*; the R-phases below ship
> *structural hygiene* that those features rest on. R-phases are
> independent of M1–M3 sequencing — most can interleave with feature work.
>
> Last reviewed: 2026-05-22 · against `dev` @ `fe5f604` (R7 complete).

### 9.0 Status / 当前状态

**Phase:** **R0–R7 COMPLETE** — the architecture refactor is done; importlinter
now runs as a **required** CI gate with a clean **6 contracts / 0 violations**
baseline. Only ADR-0010 (entity/articulation split) remains, deliberately deferred.

| What | Status | Artifact |
|---|---|---|
| Architecture assessment | ✅ written | [`plans/architecture/architecture-assessment.md`](plans/architecture/architecture-assessment.md) |
| Target architecture | ✅ written | [`plans/architecture/target-architecture.md`](plans/architecture/target-architecture.md) |
| ADRs 0001–0010 | ✅ ADR-0004 + ADR-0006 + ADR-0007 + ADR-0008 + ADR-0009 `Accepted` (shipped); ADR-0001 + ADR-0002 + ADR-0003 + ADR-0005 shipped, ready for `Accepted` (pending maintainer review); ADR-0010 `Accepted: deferred` | [`plans/adr/`](plans/adr/) |
| Phase R0 — baseline & tooling | ✅ **complete** (3 / 3 PRs merged) | PR #81 (`c3d851e`), #82 (`b8df293`), #83 (`d7a702e`) |
| Phase R1 — break rl.runner ↔ rl.backends cycle | ✅ **merged** | PR #84 (`04509d9`) — implements ADR-0001 |
| Phase R2 — small abstractions | ✅ **complete** (5 / 5 sub-slices merged) | PRs #85 (`74b2e30`), #86 (`039e502`), #87 (`0367462`), #88 (`a50d031`), #89 (`5571889`) — implement ADR-0003 (+ ADR-0002 for R2.5) |
| Phase R3 — domain-owned parsing | ✅ **complete** (2 / 2 sub-slices merged) | PR #90 (`6b97f6e`), #91 (`03f480b`) — implement ADR-0005 |
| Phase R4 — CLI decomposition | ✅ **complete** (3 / 3 PRs merged) — implements ADR-0004 | PR #92 (`588f5be`), #94 (`43cf463`), #95 (`4ca926d`) |
| Phase R5 — task-specific rewards split | ✅ **complete** (2 / 2 sub-slices merged) — implements ADR-0006 | PR #97 (`9bcbe01`), #98 (`67eef60`) |
| Phase R6 — vecenv rename + colocation | ✅ **complete** (1 PR) — implements ADR-0007 | PR #100 (`c8d2880`) |
| Phase R7 — extensions API + importlinter blocking | ✅ **complete** (6 PRs: 7.1 + docs + 7.3a–d) — implements ADR-0008 + ADR-0009 | PRs #101, #102 (docs), #103, #104, #105, #106 |
| Phase deferred — entity/articulation split | ⏸ deferred (criteria recorded) | [ADR-0010](plans/adr/0010-defer-articulation-split.md) |

**Completed.**

- Planning round (commits `0b4474f` and prior):
  - New: [`plans/architecture/architecture-assessment.md`](plans/architecture/architecture-assessment.md), [`plans/architecture/target-architecture.md`](plans/architecture/target-architecture.md), [`plans/adr/0001-…0010`](plans/adr/), [`plans/adr/README.md`](plans/adr/README.md), [`CLAUDE.md`](CLAUDE.md).
  - Modified: this `ROADMAP.md` (§9 added — refactor phases).
- **PR R0.1 — CLI `--help` snapshot baseline** (PR #81, merged `c3d851e`, 3 commits `c00d8c1` → `7dc47f8` → `c6c7ba1`):
  - New: `tests/test_cli_help_snapshots.py` (123 LoC), `tests/snapshots/help-{root,cache,prof,list,info,play,eval,export,train,project,project_new}.txt` (11 files / 277 lines).
  - Modified: `CHANGELOG.md` (+11 lines under `[Unreleased] · Added`).
  - **Not modified:** any file under `src/genelab/`. R0.1 is test scaffolding only.
- **PR R0.2 — Optional-dep import boundary** (PR #82, merged `b8df293`, 2 commits `9124c89` → `a0997d3`):
  - New: `tests/test_optional_deps.py` (78 LoC). Four parametrized subprocess tests cover `genelab.rl` + each `genelab.rl.backends.<lib>`; `sys.modules[lib] = None` poisons `rsl_rl` / `skrl` / `stable_baselines3` / `tensordict` before the import attempt.
  - Modified: `CHANGELOG.md` (+11 lines under `[Unreleased] · Added`).
  - **Not modified:** any file under `src/genelab/`. R0.2 is test scaffolding only.
- **PR R0.3 — Importlinter lint-only + CI step** (PR #83, merged `d7a702e`, 1 commit `7a429a5`):
  - Modified: `pyproject.toml` (+ `import-linter>=2.11` to `[dependency-groups] dev`; + `[tool.importlinter]` with four contracts and `exclude_type_checking_imports = true` — 66 lines).
  - Modified: `.github/workflows/ci.yml` (+ 6-line non-blocking `Architecture lint (import-linter)` step inside the `lint` job).
  - Modified: `CHANGELOG.md` (+17 lines), `uv.lock` (regenerated by `uv add`).
  - **Not modified:** any file under `src/genelab/`. R0.3 is config + CI scaffolding only.
- **PR R1 — Break `rl.runner ↔ rl.backends` cycle** (PR #84, merged `04509d9`, 1 commit `27cb847`) — implements ADR-0001:
  - New: `src/genelab/rl/_helpers.py` (175 LoC). Houses the nine shared helpers (`build_bridges`, `build_env`, `close_bridges`, `make_random_policy`, `make_zero_policy`, `resolve_env_cfg`, `resolve_log_dir`, `run_play_loop`, `save_run_params`) moved **verbatim** from `runner.py`.
  - New: `tests/test_no_static_cycle.py` (65 LoC). `grimp`-based static-graph test asserting no `rl.backends.<lib>` directly imports `rl.runner`.
  - Modified: `src/genelab/rl/runner.py` (408 → 284 LoC, −124 lines). Drops the nine helper definitions; re-exports them from `rl._helpers` to preserve the public `from genelab.rl.runner import …` path. Keeps `train_task`, `play_task`, and the private `_profile_args`.
  - Modified: `src/genelab/rl/backends/{rsl_rl,sb3,skrl}.py` (×3, +1/-1 each line). Flip `from genelab.rl.runner import (…)` → `from genelab.rl._helpers import (…)`.
  - Modified: `CHANGELOG.md` (+18 lines under `[Unreleased] · Changed`).
  - **First production-code change of the R-chain.** No public API change — re-export shim covers all CLI callers (`cli/_eval.py`, `cli/_export.py`, `cli/__init__.py:_relaunch_under_torchrun`).
- **PR R2.1 — `rl/_algorithm_taxonomy.py` extraction** (PR #85, merged `74b2e30`, 1 commit `dc43dcc`) — implements ADR-0003 sub-slice 1:
  - New: `src/genelab/rl/_algorithm_taxonomy.py` (27 LoC). Holds the two frozensets (`ON_POLICY_ALGORITHMS` = `{"PPO", "A2C"}`, `OFF_POLICY_ALGORITHMS` = `{"SAC", "TD3", "DDPG"}`) previously declared verbatim in both `rl/sb3_config.py` and `rl/skrl_config.py` (jaccard 1.000).
  - Modified: `src/genelab/rl/sb3_config.py`, `src/genelab/rl/skrl_config.py` (−5 / +5 each). Replace inline definitions with `from genelab.rl._algorithm_taxonomy import OFF_POLICY_ALGORITHMS as OFF_POLICY_ALGORITHMS, ON_POLICY_ALGORITHMS` — PEP 484 explicit re-export form.
  - Modified: `CHANGELOG.md` (+11 lines under `[Unreleased] · Changed`).
- **PR R2.2 — `rl/_attach_base.attach_optional_base` collapse** (PR #86, merged `039e502`, 1 commit `f240309`) — implements ADR-0003 sub-slice 2:
  - New: `src/genelab/rl/_attach_base.py` (65 LoC). Single `attach_optional_base` helper takes `base_module` / `base_attr` / `wrapper_name` / `caller_globals` parameters; defers the optional library import via `importlib.import_module` so the helper stays importable when no optional RL libraries are installed.
  - Modified: `src/genelab/rl/{rsl_rl,skrl,sb3}_wrapper.py` (×3, −10 / +5 each). Each drops its `def _attach_*_base()` + bottom call (16 lines) and replaces it with a top-level `from genelab.rl._attach_base import attach_optional_base` plus a 5-line `attach_optional_base(...)` call at module bottom.
  - Modified: `CHANGELOG.md` (+17 lines under `[Unreleased] · Changed`).
  - Net: −27 lines of duplication across the three wrappers; +65 LoC new helper. Verified `RslRlBase in RslRlVecEnvWrapper.__mro__` and the matching checks for skrl + sb3 stay True; class `__name__`s preserved.
  - **Location note:** ADR-0003 names the new module `rl/vecenvs/_attach_base.py`, but `rl/vecenvs/` is created in R6 (ADR-0007). The helper lives flat at `rl/_attach_base.py` until R6 relocates it.
- **PR R2.3 — `mdp/actions/_joint_match.match_joints` extraction** (PR #87, merged `0367462`, 1 commit `77aeb99`) — implements ADR-0003 sub-slice 3:
  - New: `src/genelab/mdp/actions/_joint_match.py` (36 LoC). `match_joints(patterns: Sequence[str], joint_names: Sequence[str]) -> list[int]` — extracts the regex-with-re.escape-fallback joint-matching code shared verbatim (jaccard 1.000) between `BinaryGripperAction.__init__` and `ContinuousGripperAction.__init__`.
  - Modified: `src/genelab/mdp/actions/{binary_gripper,continuous_gripper}.py` (×2, −13 / +4 each). Each calls `match_joints(...)` and raises its own term-specific `ValueError` on zero-match so error strings preserve the correct class name. Both also drop their now-unused `import re`.
  - Modified: `CHANGELOG.md` (+12 lines under `[Unreleased] · Changed`).
- **PR R2.4 — `ActuatorBase._write_pd_gains` extraction** (PR #88, merged `a50d031`, 1 commit `a2db4ef`) — implements ADR-0003 sub-slice 4:
  - Modified: `src/genelab/actuator/actuator_base.py` (+33 LoC). New `_write_pd_gains(gs_handle, *, kp_values, kv_values)` helper alongside the existing `_write_force_range` / `_write_armature` / `_write_friction` helpers. Pulls up the `set_dofs_kp`/`set_dofs_kv` write loop (with TypeError positional/kwarg fallback) shared by both PD actuators (jaccard 0.969).
  - Modified: `src/genelab/actuator/{ideal_pd,implicit_pd}.py`. `IdealPDActuator.initialize` 16 → 5 lines (writes zeros); `ImplicitPDActuator.initialize` 15 → 4 lines (writes `self._stiffness` / `self._damping`). `DCMotorActuator` inherits `IdealPDActuator.initialize` unchanged.
  - Modified: `CHANGELOG.md` (+12 lines under `[Unreleased] · Changed`).
  - **Naming variance:** ADR-0003 names it `_initialize_pd_common`; shipped as `_write_pd_gains` for consistency with the sibling `_write_*` helpers (recorded in ADR-0003 §Implementation notes).
- **PR R2.5 — `BaseTermManager` introduction** (PR #89, merged `5571889`, 2 commits `463f579` + `88454fa`) — implements ADR-0002 (the `_post_init` template-method shape) for the ADR-0003 dedup target; **completes Phase R2**:
  - New: `tests/test_manager_init_order.py` (130 LoC). Init-order gate (3 tests) asserting buffer allocation happens *after* term registration. Committed *first* (`463f579`) and verified green on pure `dev` before the refactor.
  - Modified: `src/genelab/managers/_base.py` (+60 LoC). New `BaseTermManager[TCfg]` generic base: shared `__init__` (deepcopy cfg, build `_term_names`/`_term_cfgs`, run `instantiate_class_term`, call `_post_init`), a `_post_init` hook, and the `num_envs`/`device`/`active_terms` properties.
  - Modified: `src/genelab/managers/{reward_manager,termination_manager}.py`. Both subclass `BaseTermManager[…TermCfg]`; each keeps only its buffer-allocation `_post_init`. `RewardManager.__init__` stashes `_scale_by_dt` then calls super; `TerminationManager` inherits the base `__init__` verbatim. Both drop `deepcopy`/`instantiate_class_term` imports and the 3 duplicated properties.
  - Modified: `CHANGELOG.md` (+19 lines under `[Unreleased] · Changed`).
  - **Init-order risk (ADR-0002 §Risks)** handled by the gate test — verified green pre- and post-refactor.
- **PR R3.1 — `EvalCallbackCfg.from_args` extraction** (PR #90, merged `6b97f6e`, commit `fa9c60a`) — implements ADR-0005 sub-slice 1:
  - Modified: `src/genelab/rl/eval_callback.py` (+26 LoC). New `@classmethod from_args(cls, runner_args: dict[str, str]) -> "EvalCallbackCfg | None"` — parses `--eval-every` / `--eval-episodes` / `--eval-num-envs` / `--eval-seed` verbatim from the runner-arg dict; returns `None` when `--eval-every` is unset (keeps the legacy single-shot `backend.train()` path).
  - Modified: `src/genelab/cli/__init__.py`. `_build_eval_callback` removed; `_dispatch_train` now calls `EvalCallbackCfg.from_args(runner_args)`.
  - New: `tests/test_eval_callback_from_args.py` (5 tests).
  - Modified: `CHANGELOG.md`.
  - **Direction note.** This moves a parser in the *allowed* `cli → rl` direction. It does **not** touch the function-local `from genelab.cli._eval import eval_task` inside `run_with_eval_callback` (the actual `rl → cli` violation) — see the corrected risk note below.
- **PR R3.2 — `SimulationCfg.play_retargeted_keys` extraction** (PR #91, merged `03f480b`, commit `5963aa4`) — implements ADR-0005 sub-slice 2; **completes Phase R3**:
  - Modified: `src/genelab/configs.py` (+18 LoC). New `@staticmethod play_retargeted_keys() -> tuple[str, ...]` returning `("env.simulation.vis", "env.simulation.gpu", "env.simulation.steps", "env.simulation.dt")` verbatim. `configs.py` stays torch-free at import (invariant #5) — the staticmethod adds no imports.
  - Modified: `src/genelab/cli/__init__.py`. `_PLAY_RETARGETED_KEYS` constant removed; the play-mode `env.` → `play_env.` retarget loop calls the method; `SimulationCfg` added to the `from genelab.configs import …` line.
  - New: `tests/test_configs.py` (3 tests).
  - Modified: `CHANGELOG.md`.
- **PR R4.1 — `cli/_distributed.py` extraction** (PR #92, merged `588f5be`, commit `c6adabd`) — implements ADR-0004 sub-slice 1 (first of 3 CLI-decomposition PRs):
  - New: `src/genelab/cli/_distributed.py` (161 LoC). Six multi-GPU-plumbing functions moved **verbatim** — `_relaunch_under_torchrun`, `_resolve_per_rank_num_envs`, `_strip_distributed_flags`, `_strip_flag_value_pairs`, `_extract_log_dir_flag`, `_has_log_dir_flag` — plus the `_STRIPPABLE_DISTRIBUTED_FLAGS` constant. Declares `__all__` of the six (they keep CLI-private underscore names but are the module's external API; `__all__` suppresses pyright `reportUnusedFunction` on the externally-only-called ones).
  - Modified: `src/genelab/cli/__init__.py` (1,020 → 900 LoC, −120). Drops the six definitions; re-exports them via the PEP-484 `from genelab.cli._distributed import _foo as _foo` idiom. The multi-seed pieces (`_STRIPPABLE_MULTI_SEED_FLAGS`, `_strip_multi_seed_flags`) stay until R4.2 and keep calling the re-exported `_strip_flag_value_pairs`.
  - Modified: `CHANGELOG.md` (+14 lines under `[Unreleased] · Changed`).
  - **Zero test edits.** All callers (incl. `tests/test_cli.py`, `tests/test_multi_seed_cli.py`) import via `from genelab.cli import <name>` — preserved by the shim.
- **PR R4.2 — `cli/_multi_seed.py` extraction** (PR #94, merged `43cf463`, commit `1f6178a`) — implements ADR-0004 sub-slice 2:
  - New: `src/genelab/cli/_multi_seed.py` (160 LoC). Four functions moved **verbatim** — `_dispatch_multi_seed_train`, `_parse_seed_list`, `_resolve_multi_seed_parent`, `_strip_multi_seed_flags` — plus the `_STRIPPABLE_MULTI_SEED_FLAGS` constant. Imports the argv-strip helpers (`_extract_log_dir_flag`, `_strip_flag_value_pairs`) from `cli/_distributed.py`. `__all__` of the four; `_RunnableTask` referenced via a `TYPE_CHECKING`-only forward ref (`from __future__ import annotations`) so no runtime `cli → _multi_seed → cli` cycle.
  - Modified: `src/genelab/cli/__init__.py` (900 → 775 LoC, −125; the now-unused `import sys` dropped). Re-exports the four via the `as` idiom.
  - Modified: `tests/test_cli.py` — four `_relaunch_under_torchrun` tests had their `monkeypatch.setattr` target corrected from the stale `genelab.cli.sys.argv` to `genelab.cli._distributed.sys.argv` (that function moved to `_distributed.py` in R4.1; the patch path only resolved because `sys` had stayed imported in `__init__.py`).
  - Modified: `CHANGELOG.md`.
  - `tests/test_multi_seed_cli.py` unchanged — imports the three helpers via `from genelab.cli import` (preserved by the shim).
- **PR R4.3 — `cli/_dispatch.py` extraction** (PR #95, merged `4ca926d`, commit `37f672f`) — implements ADR-0004 sub-slice 3; **completes Phase R4**:
  - New: `src/genelab/cli/_dispatch.py` (167 LoC). `_dispatch_play` + `_dispatch_train` moved **verbatim**. The profiler-kwarg coercion (`_coerce_prof_kwargs` + private `_parse_bool` / `_parse_int` / `_parse_path`) and the `_AGENT_KINDS` set moved **alongside** them — the two dispatch functions are their only users, so co-locating keeps `_dispatch.py` a self-contained leaf and avoids a runtime `cli → _dispatch → cli` cycle. **ADR variance:** ADR-0004 had tentatively kept `_coerce_prof_kwargs` in `__init__.py`; the cycle its R4.2 risk row anticipated forced the co-location. `_RunnableTask` is a `TYPE_CHECKING`-only forward ref; `__all__` lists the two dispatch fns.
  - Modified: `src/genelab/cli/__init__.py` (775 → 645 LoC, −130). Re-exports the two dispatch fns; drops three imports orphaned by the move (`os`, `typing.Any`, `pick_agent_kind`).
  - Modified: `tests/test_cli.py` — four `_relaunch_under_torchrun` `os.execvp` patches repointed to `genelab.cli._distributed.os.execvp` (`os` left `__init__.py`); the `_patch_picker` helper generalized to also patch the `cli._dispatch` consumer site (`pick_agent_kind` is consumed there now; each importing module holds its own binding).
  - Modified: `CHANGELOG.md`.
  - **≤400-LoC target not reached** (645 LoC). All three ADR-0004 modules are extracted, but the residue — 10 Typer command callbacks, `_configured_task` / `_resolve_task`, the override helpers, `_RunnableTask`, and help text — is exactly what ADR-0004 deliberately kept in `__init__.py`. Reaching ≤400 needs a separate follow-up (new concern / ADR).
- **PR R5.1 — `mdp/motion_tracking.py` extraction** (PR #97, merged `9bcbe01`) — implements ADR-0006 sub-slice 1:
  - New: `src/genelab/mdp/motion_tracking.py` (116 LoC). The whole "motion imitation" section moved **verbatim** — six public functions (`motion_global_anchor_position_error_exp`, `motion_global_anchor_orientation_error_exp`, `motion_relative_body_position_error_exp`, `motion_relative_body_orientation_error_exp`, `motion_global_body_linear_velocity_error_exp`, `motion_global_body_angular_velocity_error_exp`) + the shared private helpers `_motion_command` / `_body_index_filter`.
  - Modified: `src/genelab/mdp/rewards.py` (554 → 461 LoC). Re-exports the six (PEP-484 `as` idiom); drops the now-unused `cast` / `MotionCommand` / `quat_error_magnitude` imports. `mdp/__init__.py`, the Unitree G1 example, and tests are **unchanged** (the re-export preserves `genelab.mdp.motion_*` and `genelab.mdp.rewards.motion_*`).
  - **ADR variance:** ADR-0006 §6.1 named only the three jaccard-1.000 functions; the section had grown to six + two helpers, so R5.1 moved the whole coherent block (keeps the shared helpers with their only users). Confirmed with the maintainer.
- **PR R5.2 — `motion_body_error_exp` factory** (PR #98, merged `67eef60`, commit `31ce571`) — implements ADR-0006 sub-slice 2; **completes Phase R5**:
  - Modified: `src/genelab/mdp/motion_tracking.py` (116 → 130 LoC). New `motion_body_error_exp(env, command_name, std, body_names=None, *, quantity)` factory + `_BODY_ERROR_ATTRS` mapping; the three jaccard-1.000 body-error rewards become thin `def` wrappers (kept as `def`, not `functools.partial`, so `__name__` / signature / reward-term logging are unchanged — ADR-0006 R6.2). Factory also exported from the `genelab.mdp` namespace.
  - **Scope:** only the jaccard-1.000 trio (`pos` / `lin_vel` / `ang_vel`), per ADR-0006 and confirmed with the maintainer. The orientation (geodesic) + anchor rewards are structurally different and left unchanged.
  - New: `tests/test_motion_tracking_equivalence.py` (3 tests). Pins the pre-refactor implementations and asserts the factory + wrappers reproduce them bit-for-bit (`torch.equal`), with/without the `body_names` filter, plus a distinct-signal guard. (These rewards had no test coverage before R5.2.)
  - Modified: `src/genelab/mdp/__init__.py` (factory export), `CHANGELOG.md`.
- **PR R6 — vecenv rename + colocation** (PR #100, merged `c8d2880`) — implements ADR-0007:
  - New package `src/genelab/rl/vecenvs/` with `{rsl_rl,sb3,skrl}.py` (the three adapters moved **verbatim** from `rl/<lib>_wrapper.py`; git renames, only the `_attach_base` import path changed) + `_attach_base.py` (relocated from `rl/_attach_base.py`, its R2.2 deferred home) + `__init__.py`. The file tree now pairs each adapter with its same-named trainer under `rl/backends/`.
  - Old paths `rl/{rsl_rl,sb3,skrl}_wrapper.py` are `DeprecationWarning` re-export shims; `genelab.rl.RslRlVecEnvWrapper` is served by a module-level `__getattr__` shim and dropped from `rl.__all__`. All internal callers (3 backends, `skrl_models.py`, pipeline tests) repointed to `rl/vecenvs/<lib>`.
  - New `tests/test_deprecated_imports.py` (the sb3/skrl shim checks run in a **subprocess** to dodge the cv2/Qt collection conflict); `tests/test_optional_deps.py` extended to the three `rl/vecenvs/<lib>` modules.
- **PR R7.1 — public extension API** (PR #101, merged) — implements ADR-0008:
  - New `src/genelab/extensions.py` re-exports `register_{robot,env,task,backend}`, `ROBOTS`/`ENVS`/`TASKS`, `Backend`, `Runnable`. `cli._RunnableTask` Protocol promoted to public `genelab.registry.Runnable`; CLI keeps `_RunnableTask = Runnable` alias and internal annotations migrated to `Runnable` (dropped the two R4 `reportPrivateUsage` ignores). New `tests/test_extensions_api.py` (4 tests).
- **PR R7.2 — extensions docs** (PR #102, merged) — `docs/concepts/extensions.{en,zh}.md` updated to cover the `genelab.extensions` path, `register_backend`, and the `Runnable` / `Backend` contracts (with a runnable `EchoBackend` example). `mkdocs build --strict` green.
- **PR R7.3a — `eval_task` → rl layer** (PR #103, merged) — clears `rl.eval_callback → cli._eval`. `eval_task` moved verbatim from `cli/_eval.py` to new `genelab.rl.eval_task` (runner import made function-local to stay acyclic); `cli/_eval.py` re-exports it.
- **PR R7.3b — distributed helpers → utils** (PR #104, merged) — clears `scene → rl.distributed` (and `envs → scene → rl`). `rl/distributed.py` moved verbatim to `genelab/utils/distributed.py`; old path is a `DeprecationWarning` shim; 6 internal callers repointed. After R7.3a+b, **no `domain → rl` violations remain**.
- **PR R7.3c — split the domain-forbidden contract** (PR #105, merged) — config-only. Split "Domain below cli/rl/utils.download" into "Domain ⊬ cli/rl" + "Domain (except asset_zoo) ⊬ utils.download", recognizing asset fetching as `asset_zoo`'s legitimate downward `domain → utils` dependency. Baseline → 4 kept / 1 broken.
- **PR R7.3d — importlinter blocking flip** (PR #106, merged `fe5f604`) — implements ADR-0009; **completes the refactor**:
  - Replaced the monolithic "Top-down layering" `layers` contract (which forbade legitimate intra-domain imports) with directional `forbidden` contracts: **"rl is below cli"** + **"Infrastructure modules do not import up"** (configs/registry/cache/utils ⊬ cli/rl/domain).
  - Code fix: `PROJECT_ROOT` / `CACHE_DIR` moved to new `genelab/utils/paths.py` (cache re-exports), removing the last `utils.download → cache` infra edge so the strict infra contract passes with no waiver.
  - Flipped `.github/workflows/ci.yml` `lint-imports` to **required** (dropped `continue-on-error`); new `tests/test_importlinter_configured.py` guards the contract set against silent deletion. Final baseline: **6 kept / 0 broken**.

**Tests run.**

- Pre-R0.1: none. The pre-refactor M1 feature work that landed earlier on `dev` (eval / export / backends — see commits `99389d3..1c80b41`) was tested in its own PRs; those results stand.
- R0.1: 11 `--help` snapshot tests green locally (×3 byte-deterministic runs), green under `TERM=dumb COLUMNS=80` (CI-equivalent), green under `FORCE_COLOR=1 CI=true` (reproduces the run-2 ANSI failure mode), green under `env -u COLUMNS -u TERM -u FORCE_COLOR -u NO_COLOR` (stripped env). Final CI run `26228761027`: **test PASS / lint PASS / typecheck PASS**.
- R0.2: 4 parametrized subprocess tests green locally (×2 deterministic runs, ~3.25s each). Negative control confirmed: importing `genelab.rl.skrl_models` under poison correctly fails (verifies the mechanism catches real leaks). Final CI run `26229819224`: **test PASS / lint PASS / typecheck PASS** (after one ruff-format fix in commit `a0997d3`).
- R0.3: `ruff format --check` ✓, `ruff check` ✓, `pyright` 0/0/0, prior R0.1 + R0.2 tests (15 total) green in 4.70s, `lint-imports` surfaces 1 kept / 3 broken contracts as expected (non-blocking via `continue-on-error: true`). Final CI run `26230480745`: **test PASS / lint PASS / typecheck PASS**.
- R1: `ruff check` + `ruff format --check` ✓, `pyright` 0/0/0, full test suite **384 passed in 23.54s** (was 381 pre-R1; +3 from `test_no_static_cycle.py`). `lint-imports` baseline shrunk: contract 2 (`rl.backends does not import rl.runner`) flipped from BROKEN to KEPT. Final CI run `26233596787`: **test PASS / lint PASS / typecheck PASS** (CI's `lint-imports` confirmed `Contracts: 2 kept, 2 broken`).
- R2.1: ruff ✓, pyright 0/0/0 (after switching `# noqa: F401` to `from foo import bar as bar` PEP-484 explicit re-export form — see lessons below), 384 tests pass, `lint-imports` baseline unchanged at 2 kept / 2 broken (intra-`rl/` dedup). Final CI run `26235360172`: **all three jobs PASS**.
- R2.2: ruff ✓ (after moving `from genelab.rl._attach_base import attach_optional_base` to file top to silence E402 — see lessons below), pyright 0/0/0, 384 tests pass, `tests/test_optional_deps.py` 4/4 (invariant #1 preserved by deferred `importlib.import_module`), `lint-imports` baseline unchanged at 2 kept / 2 broken. Final CI run `26236013353`: **all three jobs PASS**.
- R2.3: ruff ✓, pyright 0/0/0 (after widening helper params from `list[str]` to `Sequence[str]` — `cfg.joint_names` is `tuple[str, ...]`), 384 tests pass, `lint-imports` baseline unchanged at 2 kept / 2 broken. `match_joints` smoke-tested directly with 5 cases (basic match, glob, `re.error` fallback, substring search, dedup). Final CI run `26236564576`: **all three jobs PASS**.
- R2.4: ruff ✓, pyright 0/0/0, 384 tests pass, the 9 actuator-specific tests (`tests/test_actuator.py` + `tests/test_articulation_refresh.py`) green — including the 4 that exercise the `_write_pd_gains` path directly. `lint-imports` baseline unchanged at 2 kept / 2 broken. Final CI run `26237393291`: **all three jobs PASS**.
- R2.5: **the init-order gate** (`tests/test_manager_init_order.py`, 3 tests) verified green on pure `dev` (commit `463f579` alone, pre-refactor) **and** after the refactor (commit `88454fa`) — the load-bearing proof that buffer-allocation timing is preserved. ruff ✓, pyright 0/0/0 (the `Generic[TCfg]` typing keeps `cfg`/`_term_cfgs` narrowed per subclass), full suite **387 passed** (was 384; +3 from the gate), manager/reward/metrics tests 48 passed. `lint-imports` baseline unchanged at 2 kept / 2 broken. Final CI run `26266527762`: **all three jobs PASS**.
- R3.1: ruff ✓, pyright 0/0/0, full suite **392 passed** (was 387; +5 from `tests/test_eval_callback_from_args.py`). `lint-imports` baseline **unchanged at 2 kept / 2 broken** — the parser moved in the allowed `cli → rl` direction. `--help` snapshots green (no CLI output drift). All three CI jobs PASS.
- R3.2: ruff ✓, pyright 0/0/0, full suite **395 passed** (was 392; +3 from `tests/test_configs.py`). `lint-imports` baseline unchanged at 2 kept / 2 broken. `genelab play --help` byte-identical (R0.1 snapshot gate green); `configs.py` torch-free at import re-verified. All three CI jobs PASS. Note: `cli/__init__.py` shrank −31 LoC (ADR-0005 estimated ≥35; the difference is the `SimulationCfg` import line added back).
- R4.1: ruff ✓, ruff format ✓, pyright 0/0/0 (after adding `__all__` to `_distributed.py` to silence `reportUnusedFunction` on the externally-only-called functions), full suite **395 passed** (unchanged — no new tests; the move is covered by the existing `test_cli.py` + `test_multi_seed_cli.py`, 98 of which run in 2.6s), `lint-imports` baseline unchanged at 2 kept / 2 broken (move stays within the `cli` package). `--help` snapshots byte-identical. Final CI run `26270577235`: **all three jobs PASS**.
- R4.2: ruff ✓, ruff format ✓, pyright 0/0/0, full suite **395 passed** (unchanged), import smoke confirms no `cli → _multi_seed → cli` cycle, `lint-imports` baseline unchanged at 2 kept / 2 broken (intra-`cli` move). `--help` snapshots byte-identical. Moved bodies verified byte-identical to `HEAD`. The four `genelab.cli.sys.argv` → `genelab.cli._distributed.sys.argv` test repoints were forced by dropping the now-unused `import sys` from `__init__.py`.
- R4.3: ruff ✓, ruff format ✓, pyright 0/0/0, full suite **395 passed** (unchanged), import smoke confirms no `cli → _dispatch → cli` cycle, `lint-imports` baseline unchanged at 2 kept / 2 broken. `--help` snapshots byte-identical. `_dispatch_play` / `_dispatch_train` + the `_parse_*` / `_coerce_prof_kwargs` cluster verified byte-identical to `HEAD`. Five `test_cli.py` failures surfaced mid-implementation (4 `os.execvp` patches + the `_patch_picker` agent-kind site) and were fixed by repointing the monkeypatch targets to the moved functions' new owning modules — same class of fix as R4.2's `sys.argv`.
- R5.1: ruff ✓, ruff format ✓, pyright 0/0/0, full suite **395 passed** (unchanged), `lint-imports` baseline unchanged at 2 kept / 2 broken. Re-export smoke confirms all six motion fns resolve via `mdp` / `mdp.rewards` / `mdp.motion_tracking` to the same objects (no cycle); the moved 96-line block verified byte-identical to `HEAD`. Zero edits to `mdp/__init__.py`, examples, or tests.
- R5.2: ruff ✓, ruff format ✓, pyright 0/0/0, full suite **398 passed** (was 395; +3 from `tests/test_motion_tracking_equivalence.py`), `lint-imports` baseline unchanged at 2 kept / 2 broken. The equivalence tests assert the factory + wrappers reproduce the pinned pre-refactor implementations bit-for-bit (`torch.equal`). The jaccard-1.000 `SIMILAR_TO` triple is gone by construction (the three bodies are now distinct one-liners); graph re-index runs post-merge.
- R6: ruff ✓, ruff format ✓, pyright 0/0/0, full suite **406 passed** (was 398; +5 `test_deprecated_imports` + 3 new `test_optional_deps` vecenv targets), `lint-imports` baseline unchanged at 2 kept / 2 broken (intra-`rl` move). Moved adapter bodies verified to differ from `HEAD` only by the `_attach_base` import line. **Incident:** the first full-suite run SIGABRTed — `test_deprecated_imports` (sorts early) imported the SB3 adapter → cv2 → Qt, poisoning the Genesis PyQt plotter tests; fixed by subprocess-isolating the sb3/skrl shim checks.
- R7.1: ruff ✓, ruff format ✓, pyright 0/0/0, full suite **410 passed** (was 406; +4 `test_extensions_api`), `lint-imports` unchanged at 2 kept / 2 broken. `import genelab` stays torch-free (`extensions.py` is standalone).
- R7.2: docs-only. `mkdocs build --strict` green; the `EchoBackend` example executed end-to-end (`isinstance Backend` True; `select_backend` round-trip).
- R7.3a: ruff ✓, pyright 0/0/0, full suite **410 passed** (unchanged), import smoke confirms no cycle and `cli._eval.eval_task is rl.eval_task.eval_task`. `rl → cli` violation gone.
- R7.3b: ruff ✓, pyright 0/0/0, full suite **411 passed** (+1 deprecation case), no cycle. `scene → rl` (+ `envs → scene → rl`) gone; **no `domain → rl` violations remain**.
- R7.3c: config-only. `lint-imports` **4 kept / 1 broken**; ruff + pyright clean; full suite **411 passed** (unchanged).
- R7.3d: ruff ✓, ruff format ✓, pyright 0/0/0, full suite **413 passed** (was 411; +2 `test_importlinter_configured`), optional-dep boundary still green after the `CACHE_DIR` move, import smoke confirms no cycle. **`lint-imports` 6 kept / 0 broken** — the gate is now required in CI.

**Changes to the dependency graph.** `mcp__codebase-memory-mcp__index_status`
reports **3,982 nodes / 11,433 edges** post-R7 (was 3,829 / 11,427 at the start
of the planning round; dipped to ~3,887 / ~10,905 mid-refactor as R0.3's
`exclude_type_checking_imports` filter + the R1/R2 consolidations collapsed
duplicate edges). The net node/edge growth in R3–R7 reflects **added** structure,
not regressions: new modules (`cli/_distributed.py`, `cli/_multi_seed.py`,
`cli/_dispatch.py`, `mdp/motion_tracking.py`, `rl/vecenvs/{rsl_rl,sb3,skrl,_attach_base}.py`,
`rl/eval_task.py`, `extensions.py`, `utils/distributed.py`, `utils/paths.py`) plus
the **re-export / deprecation shims** (each shim adds import edges:
`rl/{rsl_rl,sb3,skrl}_wrapper.py`, `rl/distributed.py`, `cli/_eval.py`,
`registry.Runnable`'s alias). The cycle (R1) and all five cluster duplications
(R2.1–R2.5) are gone; the SIMILAR_TO jaccard-1.000 motion-tracking triple
collapsed in R5.2; and the cross-layer import baseline went from 24 violations / 3
broken contracts (R0.3) to **0 violations / 6 kept contracts** (R7.3d).

**Risks identified (aggregate view).** Detailed per-ADR; the highest-attention items:

- **R1 (ADR-0001)** — ✅ **resolved by PR #84.** The quiescent `rl.runner ↔ rl.backends` cycle is gone; backends now import from `rl._helpers`. `tests/test_no_static_cycle.py` (grimp-based) prevents regression; matching importlinter contract is now KEPT.
- **R2.5 (ADR-0002)** — ✅ **resolved by PR #89.** `BaseTermManager._post_init` preserves buffer-allocation timing; `tests/test_manager_init_order.py` was verified green pre- and post-refactor. The risk did not materialize.
- **R4 (ADR-0004)** — ✅ **resolved by PRs #92 + #94 + #95.** All three CLI submodules (`_distributed`, `_multi_seed`, `_dispatch`) extracted; R0.1 `--help` snapshots gated every PR (zero drift). The large `test_cli.py` surface (1,414 LoC) needed only mechanical monkeypatch-target repoints in R4.2/R4.3 (functions moved out of `cli/__init__.py`, so `genelab.cli.{sys,os,pick_agent_kind}` patch paths were repointed to the new owning modules). `cli/__init__.py` 1,051 → 645 LoC; the ADR's ≤400 target is a documented follow-up (the residue is Typer wiring ADR-0004 deliberately kept).
- **R5.2 (ADR-0006)** — ✅ **resolved by PR #98.** The parameterization-drift risk was guarded by `tests/test_motion_tracking_equivalence.py`, which asserts the factory + wrappers reproduce the pinned pre-refactor implementations bit-for-bit (`torch.equal`). Scope was held to the jaccard-1.000 trio; the geodesic / anchor rewards (structurally different) were left untouched.
- **R7 (ADR-0009)** — ✅ **resolved by PRs #103–#106.** The blocking flip was gated on clearing the R0.3 baseline (24 cross-layer imports → 21 after R1 → still 21 through R6, since R2–R6 were intra-package / allowed-direction / intra-`cli` moves that didn't touch the cross-layer baseline). R7.3 cleared the rest: **R7.3a** (`rl.eval_callback → cli._eval`, via moving `eval_task` to `rl`), **R7.3b** (`scene → rl.distributed`, via moving the helpers to `utils` — also clearing `envs → scene → rl`), **R7.3c** (`asset_zoo → utils.download` recognized as legitimate via a contract split), **R7.3d** (`utils.download → cache` removed by moving `CACHE_DIR` to `utils.paths`; the monolithic `layers` contract replaced by directional `forbidden` contracts). Final baseline: **0 violations / 6 kept contracts**, now a required CI gate.

**Risks discovered during R0 + R1 (new, not in the original list).**

- **Rich/Typer help output is not deterministic under env-var pinning** (R0.1). `TERM=dumb` / `COLUMNS=N` / `NO_COLOR=1` are honoured differently across Rich versions and CI runners. GitHub Actions sets `FORCE_COLOR=1` which Rich honours *over* `NO_COLOR=1`. The portable mechanism is pinning Typer's module-level constants in a `python -c` wrapper before import: `typer.rich_utils.MAX_WIDTH = 100`, `typer.rich_utils.FORCE_TERMINAL = False`. Any future test that diff-checks Rich-rendered output must follow this pattern. Captured in `tests/test_cli_help_snapshots.py`'s module docstring.
- **Doc drift: ROADMAP §9.0 originally enumerated 9 CLI commands; the real surface has 11** (R0.1). `prof` and the `project` subapp's own `--help` were missed. The R0.1 snapshot baseline covers all 11; this document has been updated. New CLI commands or subapps must be added to both `tests/test_cli_help_snapshots.py:HELP_COMMANDS` *and* the §9.0 list.
- **Local lint gates must be run before pushing** (R0.2). The first R0.2 push failed CI because `ruff format --check` flagged the implicit-string-concatenation in the subprocess wrapper. Triple-quoted strings are ruff-stable. Future test-only PRs should run `uv run ruff format --check` and `uv run ruff check` locally before pushing — both are part of CI's `lint` job.
- **The layering spec in `target-architecture.md` / ADR-0009 has more drift from current code than the assessment suggested** (R0.3). With `exclude_type_checking_imports = true`, the R0.3 baseline showed **24 distinct cross-layer imports** across 3 broken contracts; R1 trimmed 3 of those. Root causes by category (post-R1):
  - ~~3 imports: `rl.backends.{rsl_rl,skrl,sb3} → rl.runner` — addressed by **R1 / ADR-0001**.~~ ✅ Resolved by PR #84.
  - 1 import: `rl.eval_callback → cli._eval` (function-local `from genelab.cli._eval import eval_task` inside `run_with_eval_callback`, now ~l.141) — **NOT addressed by R3** (corrected: R3.1 moved the `--eval-*` *parser* into `EvalCallbackCfg.from_args`, an allowed `cli → rl` move, but left the `rl → cli` runtime import untouched). Needs a separate follow-up: inject `eval_task` into `run_with_eval_callback` rather than function-local importing it. Still gating R7.
  - 1 import: `scene.interactive_scene → rl.distributed (l.139)` — already function-local (lazy) inside `build()`; either move `pin_cuda_device` out of `rl/` or accept a TODO-tagged waiver in R1.
  - 6 imports: `asset_zoo.{cartpole,franka,unitree_g1,unitree_g1_motions,unitree_go1,anymal_c} → utils.download` — asset factories call `fetch_asset` directly; needs a small PR or new ADR.
  - 5+5 imports: `asset_zoo.* → actuator / entity` — asset factories build entities. Probably needs an "assets reach into domain" ADR.
  - ~8 imports: `envs.manager_based_rl_env → {bridges, entity, managers, scene, sensor}` — integration-point realities; some may stay, some can move behind interfaces during R3/R4.
  - Remaining imports (`scene.interactive_scene → {entity, recording, sensor, terrains}`, `entity.articulation → actuator`, `configs → {sensor, entity, recording, terrains}`, `utils.download → cache`) — each needs per-file audit during the relevant R-phase or a dedicated PR.
- **R7 (importlinter blocking flip) — ✅ DONE.** The R0.3 baseline of 24 cross-layer imports is fully cleared (24 → 21 after R1 → 0 after R7.3). Resolution by edge cluster: `rl.backends → rl.runner` ×3 (R1); `rl.eval_callback → cli._eval` (R7.3a, relocate `eval_task`); `scene/envs → rl.distributed` (R7.3b, relocate to `utils`); `asset_zoo → utils.download` ×6 (R7.3c, legitimate — contract split); `utils.download → cache` (R7.3d, relocate `CACHE_DIR`). The remaining ~14 "violations" in the old `layers` contract were **never real** — they were legitimate intra-domain imports the monolithic `layers` contract mis-flagged by treating domain packages as mutually independent; R7.3d's directional `forbidden` contracts stop flagging them. Net: **6 kept / 0 broken**, required in CI.
- **ADR-0001 §Migration plan's test spec was flawed** (R1). It said *"imports `genelab.rl.backends.sb3` in a subprocess and asserts `genelab.rl.runner` is not in `sys.modules` afterwards."* That spec assumed `rl/__init__.py` does not import `runner` — but it does (for the public `play_task` / `train_task` re-exports), so any submodule import of `genelab.rl` unavoidably pulls `runner` into `sys.modules` via parent-package init. The `sys.modules` check can never detect the actual cycle for `rl.backends.*` modules. PR #84 used a `grimp`-based static-graph test instead, which asserts no `rl.backends.<lib>` directly imports `rl.runner` — the real invariant. ADR-0001 §Migration plan has been amended to reflect this. Future "static cycle" tests should use the grimp pattern, not the sys.modules pattern.
- **`# noqa: F401` does not silence pyright's `reportUnusedImport`** (R2.1). For re-export shims, use the PEP-484 explicit-re-export idiom `from foo import bar as bar` — recognized by both ruff and pyright (and mypy). No `__all__` maintenance required. Future R-phases that introduce re-export shims should follow this pattern; `noqa` comments alone will fail CI's typecheck job.
- **Ruff E402 "module level import not at top of file" fires even when the call site is the bottom of the file** (R2.2). Pattern: import the helper at the top with the other imports; place the *call* wherever execution requires it (e.g. at module bottom for "register at module load" patterns). Future R-phases that wire bottom-of-file registration calls should put the imports up top.
- **Match cfg field types when extracting helpers** (R2.3). Dataclass fields often default to `tuple[str, ...]` (immutable) rather than `list[str]`. Helpers that accept inputs originating from cfg should type their parameters as `Sequence[str]` (or `Iterable[str]`) so both `tuple` and `list` callers work without conversion. Pyright catches the mismatch — worth checking dataclass field types before drafting helper signatures.
- **ADR-0003 R2.2 names the new module `rl/vecenvs/_attach_base.py`, but `rl/vecenvs/` is created in R6** (ADR-0007). PR #86 placed the helper flat at `rl/_attach_base.py` and noted the deferred relocation in both the PR description and CHANGELOG. R6 will move it. Same care should be taken for any future R-phase slice that targets a directory the upstream ADR assumes already exists.
- **`Generic[TVar]` is required for a shared base whose `__init__` accepts a `dict[str, <subclass-cfg>]`** (R2.5). `dict` is invariant in its value type, so a base `__init__(cfg: dict[str, ManagerTermBaseCfg])` rejects `dict[str, RewardTermCfg]` at the call site under pyright. `BaseTermManager(Generic[TCfg])` with `TCfg = TypeVar(..., bound=ManagerTermBaseCfg)` lets each subclass narrow (`BaseTermManager[RewardTermCfg]`) without losing call-site type info. Future shared-base extractions over heterogeneous cfg dicts should use the same pattern.
- **Init-order gate tests are cheap insurance for "pull __init__ up to a base" refactors** (R2.5). The `_post_init` template method moves buffer allocation across a method boundary; a 3-assertion test (buffers populated, keys mirror term names, empty-cfg → empty dicts) committed *before* the refactor and verified green pre- and post-move turns a "medium-risk" refactor into a low-risk one. Pattern worth reusing for R4 (CLI split) and any future base-class extraction.
- **`metrics_manager.py` and `curriculum_manager.py` carry the same term-registration loop as the two managers R2.5 consolidated** (discovered during R2.5 audit). They are *out of ADR-0002's scope* (which targets rewards + terminations only) so were left untouched. They are a clean follow-up: adopting `BaseTermManager` there would remove two more copies of the registration loop. Candidate for a future R2.x-style PR or an ADR-0002 addendum.
- **R3 relocates parsers in the *allowed* direction and does not reduce the importlinter baseline** (R3.1, corrected). The pre-R3 plan claimed moving `_build_eval_callback` into `EvalCallbackCfg.from_args` would clear the `rl.eval_callback → cli._eval` violation. It did not: `from_args` is a `cli → rl` move (allowed), whereas the violation is the `rl → cli` runtime import of `eval_task` *inside* `run_with_eval_callback`, which R3 never touched. Lesson: distinguish "where does the parsing live" (R3's concern) from "which module imports which at runtime" (the importlinter concern) — they are orthogonal. Clearing the violation is a separate follow-up (dependency-inject `eval_task`).
- **`__all__` is the right tool for `reportUnusedFunction` on moved private functions** (R4.1). When CLI-package-private helpers (leading-underscore names that tests import via `from genelab.cli import _foo`) move into a new `cli/_*.py`, the ones called *only* from outside that module trip pyright `reportUnusedFunction` at the **def** site. The PEP-484 `as`-idiom re-export fixes `reportUnusedImport` at the *import* site (R2.1 lesson) but not this. Fix: declare an `__all__` listing the moved names in the new module — pyright treats `__all__` members as exported. Underscore names in `__all__` are legal and document the module's external API. Future R4.2/R4.3 (and any private-function relocation) should pre-empt this with `__all__`.
- **A `TYPE_CHECKING`-only forward ref breaks the cycle when a moved function's *annotation* references a symbol left behind in the parent package** (R4.2 + R4.3). `_dispatch_multi_seed_train`, `_dispatch_play`, `_dispatch_train` are all annotated `task: _RunnableTask`, and `_RunnableTask` (a Protocol) stays in `cli/__init__.py` (it moves to `registry.Runnable` only in ADR-0008). Adding `from __future__ import annotations` + `if TYPE_CHECKING: from genelab.cli import _RunnableTask` keeps the annotation a string at runtime, so the new `cli/_*.py` module never imports `cli` at runtime — no `cli → _* → cli` cycle. A targeted `# pyright: ignore[reportPrivateUsage]` on that import covers the private-name lint. Verified with an import smoke test each PR.
- **When the moved function uses a parent-package symbol at *runtime* (not just in an annotation), co-locate that symbol with it** (R4.3). `_dispatch_play` / `_dispatch_train` call `_AGENT_KINDS` and `_coerce_prof_kwargs` (+ its private `_parse_*` helpers) at runtime; all are used *only* by the two dispatch functions. ADR-0004 had said `_coerce_prof_kwargs` "stays in `cli/__init__.py`", but that re-creates the cycle the `TYPE_CHECKING` trick can't fix (the names are evaluated, not stringified). Resolution: move the whole exclusively-used cluster into the new module so it stays a self-contained leaf. Recorded as an ADR variance. Rule of thumb: a moved function's *annotation* deps can stay behind (TYPE_CHECKING ref); its *runtime* deps must either move with it or already live in a non-`cli` / sibling `cli/_*` module.
- **Removing a now-unused `import` from `cli/__init__.py` breaks `monkeypatch.setattr("genelab.cli.<mod>.<attr>", …)` paths that targeted it** (R4.2 + R4.3). `test_cli.py` patches `genelab.cli.sys.argv` and `genelab.cli.os.execvp` to drive `_relaunch_under_torchrun`. Those paths only resolved because `sys` / `os` were imported in `__init__.py`; the function itself moved to `_distributed.py` in R4.1. Once R4.2/R4.3 dropped the orphaned `import sys` / `import os`, the patch paths 404'd. Fix: repoint them to the module that *owns* the function (`genelab.cli._distributed.{sys,os}.…`). Likewise, a picker imported `from … import name` into a new consumer module needs its monkeypatch to target *that* module (each importer holds its own binding) — `_patch_picker` was generalized to patch every consumer site. **Lesson: when relocating a function, audit `tests/` for `monkeypatch.setattr("genelab.cli.<x>…")` strings, not just `from genelab.cli import` statements** — the string paths are invisible to `find_referencing_symbols`.
- **A literal LoC target in an ADR is an estimate, not a guarantee** (R4.3). ADR-0004 set `cli/__init__.py ≤ 400 LoC`; extracting all three named modules landed it at 645. The gap is the Typer command callbacks + task-resolution + override helpers + help text the ADR explicitly chose to keep. Shipped as "ADR scope complete" with the gap documented; reaching ≤400 is a separate concern. When an ADR's quantitative target and its enumerated scope disagree after the work, the enumerated scope is the contract — record the deviation rather than expanding scope silently.
- **An ADR's enumerated symbol list can go stale; re-audit the actual code before a "move" slice** (R5.1). ADR-0006 named three motion-tracking functions; by the time R5.1 ran, the "motion imitation" section of `mdp/rewards.py` had grown to six public functions + two shared private helpers. Moving only the three would have orphaned the shared helpers and left a half-family behind — defeating the ADR's coherence goal. `get_symbols_overview` on the target module before drafting the move surfaced the drift; the whole coherent block moved instead (maintainer-confirmed). Lesson: a relocation slice's scope is "the coherent unit in the code today," not "the symbols the ADR happened to list months ago."
- **Pin the pre-refactor implementation in the test when a refactor *merges* bodies** (R5.2). Unlike a verbatim move (where `git show HEAD:… | diff` proves equivalence), a parameterizing refactor changes the code shape, so there's no in-tree "original" to diff against post-merge. The durable guard is a test that *copies* the original bodies as reference implementations and asserts the new factory/wrappers reproduce them bit-for-bit (`torch.equal`), exercising each parameter branch (here: each `quantity`, with and without the optional filter). These rewards had **zero** prior coverage — the equivalence test is also their first unit test, so it doubles as a regression net.
- **Prefer thin `def` wrappers over `functools.partial` for back-compat shims** (R5.2). ADR-0006 floated `partial`, but `partial` objects have no `__name__` and a noisy `repr` — risky if any consumer logs reward terms by `func.__name__`. A one-line `def` that forwards to the factory keeps the public name a real function with its original signature and `__name__`, for one extra line each. Use `partial` only when the call site truly needs a callable value, not a named function.
- **`subprocess`-isolate any test that imports the SB3 vecenv adapter** (R6). Importing `genelab.rl.vecenvs.sb3` pulls `cv2`, which forces the `xcb` Qt plugin and SIGABRTs Genesis's PyQt plotter tests if it happens in the shared pytest process. `test_sb3_pipeline.py` dodges this only because it sorts *after* the plotter tests; `test_deprecated_imports.py` sorts early (`d`) and crashed the suite on the first run. Fix: run such checks in a subprocess (like `test_optional_deps.py`). See project memory `cv2-qt-plotter-conflict.md`. The backend *modules* are safe in-process (they import the adapter only function-locally); only the adapter modules pull cv2.
- **`importlinter`'s `layers` contract forbids intra-layer imports among pipe-separated siblings** (R7.3d). Listing the 11 domain packages as `a | b | c | …` in one layer makes them mutually *independent* — which wrongly flags every legitimate intra-domain import (`envs → scene`, `mdp → managers`, `entity → actuator`). There is no "allow imports within a layer" flag, and the packages have no shared parent module to collapse them into one layer item. Lesson: to enforce *band ordering* without forbidding intra-band imports, use directional `forbidden` contracts ("lower band ⊬ each higher band"), not a monolithic `layers` contract. The `layers` type fits strict pipelines, not flat bands.
- **A relocation can fix a layering violation more cleanly than DI** (R7.3a/b). Both `rl.eval_callback → cli._eval` and `scene → rl.distributed` were "function imports something a layer up." The instinct is dependency injection (thread the callable down), but the better fix was noticing the imported thing lived in the *wrong layer*: `eval_task`'s body is all `rl`/config-band work (move it to `rl`), and `rl.distributed` is a generic env helper with no `rl` content (move it to `utils`). Relocating to the correct layer removes the edge at its source with no signature churn. Check "is this symbol in the right layer?" before threading injection params.
- **A forbidden contract states intent better than a blanket ban + waivers** (R7.3c). `asset_zoo → utils.download` is a legitimate downward `domain → utils` dependency (the asset catalog fetches assets). Rather than waive 6 edges with `ignore_imports` + TODO, split the contract so the rule reads "term logic must not download, but the asset catalog may." Waivers accrete; a precise contract documents the architecture.

**The refactor is complete.** R0–R7 are all merged; the architecture is enforced
by a required `lint-imports` CI gate (6 contracts, 0 violations). What remains:

1. **ADR review.** ADR-0004 / 0006 / 0007 / 0008 / 0009 are `Accepted` (shipped).
   ADR-0001 / 0002 / 0003 / 0005 are shipped and ready to flip from `Proposed` to
   `Accepted` (one-line edit per file, pending maintainer review). ADR-0010 stays
   deferred.
2. **Follow-up work** (none on a critical path; each is an independent, optional PR):
   - Apply `BaseTermManager` to `metrics_manager.py` + `curriculum_manager.py`
     (the R2.5 dedup leftover — two more copies of the term-registration loop).
   - **Reach the ADR-0004 ≤400-LoC target for `cli/__init__.py`** (now ≈645) by
     extracting the residue ADR-0004 deliberately kept — Typer command callbacks,
     `_configured_task` / `_resolve_task`, override helpers, help text — as a new
     concern (its own ADR).
   - Split the 1,414-LoC `tests/test_cli.py` by concern (ADR-0004 flagged this; the
     R4.2/R4.3 monkeypatch-target churn shows the file has stale module-coupling).
   - Remove the deprecation shims (`rl/{rsl_rl,sb3,skrl}_wrapper.py`,
     `rl/distributed.py`, `cli/_eval.py` re-export, `cli._RunnableTask` alias,
     `rl.RslRlVecEnvWrapper` `__getattr__`) once they've ridden one release on
     `main` (per §9.1 rule 1).
   - Reconsider ADR-0010 (entity/articulation split) against its recorded trigger
     criteria when multi-robot work (M3.6) begins.

**Suggested next slice.** No further R-phase. The highest-value optional follow-up
is the **`BaseTermManager` adoption for `metrics_manager.py` + `curriculum_manager.py`**
— it is the smallest, lowest-risk, fully-understood item (a known dedup with a
proven init-order gate-test pattern from R2.5) and finishes the consolidation R2
started. The R-phase machinery (snapshots, optional-dep tests, the now-blocking
importlinter gate) all stays in place to guard it.

### 9.1 Cross-phase rules

These rules (§9.1) apply to **every** R-phase below. The status panel
in §9.0 records progress against them.

1. **Verify references before removing or relocating code.** Use
   `mcp__serena__find_referencing_symbols` + `mcp__codebase-memory-mcp__search_graph`
   to enumerate every caller of the symbol(s) the PR touches. Paste the
   reference list into the PR description. Do not delete a legacy path
   until its forwarding shim (if any) has been in `main` for one release.
2. **No behavior change unless explicitly called out.** Every PR's first
   reviewer task is to confirm the diff is a structural move, not a
   functional edit. Snapshot tests (R0) catch regressions early.
3. **Boundaries before logic.** A phase that *introduces a seam* (R0
   guardrails, R3 domain-config parsers) must land before the phase that
   *moves logic across that seam* (R4 CLI split, R7 importlinter blocking).
4. **One concept per PR.** A PR may move multiple files but must address
   a single architectural concept. Mixing the cycle-break with the
   rewards-split, for instance, is forbidden.
5. **Each PR ships with**: tests green, snapshot diff empty (where
   applicable), CHANGELOG entry, and — for moves — a `find_referencing_symbols`
   audit pasted into the description.

### 9.2 Phase sequencing

```
R0 ✅─┬──► R1 ✅ ──────────────────────────────────────► R7 ✅
      ├──► R2 ✅ (R2.1–R2.5 all merged) ────────────────► R7 ✅
      ├──► R3 ✅ ──► R4 ✅ (R4.1–R4.3 all merged) ───────► R7 ✅
      ├──► R5 ✅ (R5.1–R5.2 all merged) ────────────────► R7 ✅
      └──► R6 ✅ ───────────────────────────────────────► R7 ✅
```

**All phases complete.** R7 (R7.1 + docs + R7.3a–d) cleared every cross-layer
violation (24 → 0) and flipped `lint-imports` to a required CI gate. The
architecture described in `target-architecture.md` §4–§5 is now executable and
enforced.

---

### Phase R0 — Baseline & Tooling / 基线与工具 ✅ COMPLETE

1. **Goal.** Establish the safety net (snapshot tests, optional-dep test,
   importlinter in lint-only mode) before touching production code.
   Nothing in `src/genelab/` changes.
2. **Scope.**
   - **R0.1 ✅ merged** — CLI `--help` snapshot tests for all 11 commands
     (root, `cache`, `prof`, `list`, `info`, `play`, `eval`, `export`,
     `train`, `project`, `project new`). Test invokes the CLI via a
     `python -c` wrapper that pins `typer.rich_utils.MAX_WIDTH = 100`
     and `typer.rich_utils.FORCE_TERMINAL = False` before importing
     `genelab.cli`, bypassing every host-dependent Rich auto-detection
     branch. Regeneration: `UPDATE_SNAPSHOTS=1 pytest tests/test_cli_help_snapshots.py`.
   - **R0.2 ✅ merged** — Optional-dep subprocess test
     (`tests/test_optional_deps.py`) that boots `import genelab.rl` and
     each `genelab.rl.backends.<lib>` with `rsl_rl` / `skrl` /
     `stable_baselines3` / `tensordict` poisoned in `sys.modules`.
   - **R0.3 ✅ merged** — Layering contract scaffold (`pyproject.toml`
     `[tool.importlinter]` with 4 contracts + `exclude_type_checking_imports
     = true`) + non-blocking `Architecture lint (import-linter)` step in
     CI's `lint` job. Baseline on `dev` post-merge: 1 kept / 3 broken /
     24 distinct cross-layer imports — see §9.0 "Risks discovered during R0".
3. **Non-goals.** No production code change. No new public APIs. No
   importlinter rule is yet blocking (R7 flips that).
4. **Affected modules.** `tests/`, `pyproject.toml`,
   `.github/workflows/ci.yml`. **No file under `src/genelab/` was touched.**
5. **Dependency changes.** `import-linter>=2.11` added to
   `[dependency-groups] dev` in R0.3.
6. **PR slices.**
   - PR0.1: snapshot baseline (`tests/snapshots/` + reader test). ✅ **Merged in PR #81 (`c3d851e`).** Final test file 123 LoC; 11 snapshots (277 lines). CHANGELOG `[Unreleased] · Added` entry included.
   - PR0.2: optional-dep test scaffold. ✅ **Merged in PR #82 (`b8df293`).** `tests/test_optional_deps.py` 78 LoC + CHANGELOG entry. Two commits (one ruff-format fix after the first CI failed).
   - PR0.3: importlinter config (lint-only) + CI step (non-blocking). ✅ **Merged in PR #83 (`d7a702e`).** `pyproject.toml` +66 lines (4 contracts), `.github/workflows/ci.yml` +6 lines, CHANGELOG entry, `uv.lock` regenerated.
7. **Test strategy.** New tests must run green on the current `dev` —
   they are pure observations, not assertions about a target state.
   R0.1's snapshot test was verified green under four envs (clean,
   `TERM=dumb COLUMNS=80`, `FORCE_COLOR=1 CI=true`, stripped) before
   merge. R0.2's poison test was verified with a negative control
   (importing `genelab.rl.skrl_models` under poison correctly fails,
   confirming the mechanism catches real leaks). R0.3's `lint-imports`
   was verified to produce the same baseline locally and in CI
   (1 kept / 3 broken / 24 distinct cross-layer imports / 266 deps).
8. **Risk level.** Low (realised: no rollbacks required).
9. **Rollback.** `git revert <merge-sha>` per PR. No source code was
   affected, so revert would be trivial.
10. **Completion criteria.**
    - 3 PRs merged. **Progress: 3 / 3** ✅.
    - CI shows green snapshot diff and a non-blocking importlinter step. ✅
    - `tests/test_optional_deps.py` passes on the matrix `{rsl_rl, skrl, sb3, tensordict} × {present, absent}`. ✅ (4 tests, all 4 libs poisoned at once per design decision in PR #82.)

---

### Phase R1 — Break `rl.runner` ↔ `rl.backends` cycle / 解开 RL 循环依赖 ✅ COMPLETE (PR #84)

1. **Goal.** Eliminate the static-import cycle currently held together
   by `rl/backends/__init__.py._ensure_loaded` lazy `importlib`. Decision
   recorded in ADR-0001.
2. **Scope.** Extract 9 helpers from `rl/runner.py` into a new
   `rl/_helpers.py`: `build_bridges`, `build_env`, `close_bridges`,
   `make_random_policy`, `make_zero_policy`, `resolve_env_cfg`,
   `resolve_log_dir`, `run_play_loop`, `save_run_params`. Backends switch
   their imports. `runner.py` re-exports the helpers for one release.
3. **Non-goals.** No logic edits in any helper. `train_task` / `play_task`
   bodies are untouched. No file rename.
4. **Affected modules.** `rl/runner.py` (shrinks), `rl/_helpers.py` (new),
   `rl/backends/{rsl_rl,skrl,sb3}.py` (imports flip).
5. **Dependency changes.** New importlinter rule: `rl.backends.* ⊬ rl.runner`
   (added in lint-only mode here; flipped to blocking in R7).
6. **PR slices.** 1 PR. Mechanical move (helper bodies verified in
   `target-architecture.md` Appendix A).
7. **Test strategy.**
   - `test_rl_pipeline.py` + `test_sb3_pipeline.py` + `test_skrl_pipeline.py` green.
   - New `tests/test_no_static_cycle.py`: imports `genelab.rl.backends.sb3`
     in a subprocess and asserts `genelab.rl.runner` is *not* in
     `sys.modules` afterwards (cycle stays broken under future changes).
8. **Risk level.** Low.
9. **Rollback.** `git revert`. Legacy re-export in `rl/runner.py` means
   any external caller is unaffected during the deprecation window;
   revert simply restores the original file layout.
10. **Completion criteria.**
    - `test_no_static_cycle.py` passes.
    - The three pipeline tests pass.
    - importlinter lint-only output shows zero new violations.

---

### Phase R2 — Small abstractions / 抽取重复抽象 ✅ COMPLETE (PRs #85–#89)

1. **Goal.** Collapse five high-jaccard duplicates flagged in the
   assessment (§9 rows 1–6) into shared helpers. Decisions recorded in
   ADR-0002 (BaseTermManager) and ADR-0003 (the other four).
2. **Scope.**
   - **R2.1** `rl/_algorithm_taxonomy.py` — centralize `ON_POLICY_ALGORITHMS`
     / `OFF_POLICY_ALGORITHMS`; both configs re-export.
   - **R2.2** `rl/vecenvs/_attach_base.attach_optional_base` — collapse
     the three `_attach_{sb3,skrl,rsl_rl}_base` helpers (jaccard 0.984–1.000).
   - **R2.3** `mdp/actions/_joint_match.match_joints` — extract regex
     joint-name → indices code from `BinaryGripperAction.__init__` and
     `ContinuousGripperAction.__init__` (jaccard 1.000).
   - **R2.4** `actuator/actuator_base._initialize_pd_common` — pull
     `IdealPDActuator.initialize` / `ImplicitPDActuator.initialize`
     shared body up to the base (jaccard 0.969).
   - **R2.5** `managers/_base.BaseTermManager` — pull `RewardManager.__init__`
     / `TerminationManager.__init__` registration loop into the base
     (jaccard 0.953). Subclasses override `_post_init` for buffer
     allocation.
3. **Non-goals.** No file relocation (that's R6). No public API change
   (subclass constructors keep their signatures).
4. **Affected modules.** Per item:
   - R2.1: `rl/_algorithm_taxonomy.py` (new), `rl/sb3_config.py`,
     `rl/skrl_config.py`.
   - R2.2: `rl/vecenvs/_attach_base.py` (new), 3 existing wrappers.
   - R2.3: `mdp/actions/_joint_match.py` (new), `binary_gripper.py`,
     `continuous_gripper.py`.
   - R2.4: `actuator/actuator_base.py`, `ideal_pd.py`, `implicit_pd.py`.
   - R2.5: `managers/_base.py`, `reward_manager.py`,
     `termination_manager.py`.
5. **Dependency changes.** None external.
6. **PR slices.** 5 PRs (one per sub-item). Each PR independently revertable.
   Each PR's description includes a `find_referencing_symbols` audit
   for the symbol(s) being relocated.
7. **Test strategy.**
   - R2.1: `tests/test_rl_pipeline.py`, `tests/test_sb3_pipeline.py`,
     `tests/test_skrl_pipeline.py`.
   - R2.2: optional-dep test (R0) covers the attach-base degradation paths.
   - R2.3: `tests/test_franka_pick_and_place_examples.py`, `test_ee_delta_ik.py`.
   - R2.4: `tests/test_actuator.py`, `test_articulation_refresh.py`.
   - R2.5: `tests/test_rewards.py`, `test_managers.py`. **Pre-R2.5 gate:**
     add `tests/test_manager_init_order.py` asserting that
     `RewardManager(cfg, env)._episode_sums` and `TerminationManager(cfg,
     env)._term_dones` are populated after construction; landed *before*
     the R2.5 refactor PR.
8. **Risk level.**
   - R2.1 / R2.2 / R2.3 / R2.4: low (mechanical).
   - R2.5: medium — `_post_init` ordering must preserve current
     buffer-allocation timing (target-arch risk R2).
9. **Rollback.** Per-PR `git revert`. Each abstraction is independent;
   reverting one does not affect the others.
10. **Completion criteria.**
    - All 5 PRs merged.
    - Re-running `mcp__codebase-memory-mcp__search_graph` (relation
      `SIMILAR_TO`) on the original symbols shows no jaccard ≥ 0.9
      edges between the now-consolidated pairs.
    - Existing tests green; the new manager-init-order test green.

---

### Phase R3 — Domain owns its parsing / 解析逻辑下沉到 domain — ✅ COMPLETE (PRs #90, #91)

1. **Goal.** Move runner-arg parsing out of `cli/__init__.py` into
   classmethods on the relevant domain configs. Sets the principle for
   future runner args. Decision recorded in ADR-0005.
2. **Scope.**
   - `EvalCallbackCfg.from_args(runner_args: dict[str, str]) → EvalCallbackCfg | None`
     replaces `cli/__init__.py:_build_eval_callback`.
   - `SimulationCfg.play_retargeted_keys() → tuple[str, ...]` replaces
     `cli/__init__.py:_PLAY_RETARGETED_KEYS`.
3. **Non-goals.** No CLI flag changes. No `--help` text changes (R0
   snapshot must match). No structural CLI decomposition (that's R4).
4. **Affected modules.** `rl/eval_callback.py`, `configs.py`,
   `cli/__init__.py`.
5. **Dependency changes.** None. (Note: R3 does **not** clear the
   `rl.eval_callback → cli._eval` importlinter violation — that is a
   separate `rl → cli` runtime import, untouched here. The pre-R3 plan
   wrongly assumed it would; corrected in §9.0.)
6. **PR slices.** 2 PRs — both shipped:
   - PR3.1 ✅ (#90): `EvalCallbackCfg.from_args`; CLI delegates.
   - PR3.2 ✅ (#91): `SimulationCfg.play_retargeted_keys()`; CLI delegates.
7. **Test strategy.**
   - R0 snapshot tests must produce empty diffs.
   - New `tests/test_eval_callback_from_args.py` covering the matrix
     `{--eval-every set / unset} × {--eval-episodes set / unset} × …`.
   - New focused test in `tests/test_configs.py` for
     `play_retargeted_keys`.
8. **Risk level.** Low.
9. **Rollback.** Per-PR revert. CLI keeps the original inline parsing
   commented out during the first deprecation window (so revert is a
   no-op uncomment, not a full body re-paste).
10. **Completion criteria.** ✅ all met.
    - `cli/__init__.py` shrinks by ≥ 35 LoC. → shrank ~31 LoC net
      (R3.2 added back a `SimulationCfg` import line; the estimate did
      not account for it — close enough, no concern).
    - New classmethods tested. → `tests/test_eval_callback_from_args.py`
      (5 tests) + `tests/test_configs.py` (3 tests).
    - Snapshot diff empty. → verified green on both PRs.

---

### Phase R4 — CLI dispatcher decomposition / CLI 拆解 — ✅ COMPLETE (R4.1 #92, R4.2 #94, R4.3 #95)

1. **Goal.** Carve `cli/__init__.py` (1,051 LoC, 40 symbols) into three
   focused submodules. Decision recorded in ADR-0004.
2. **Scope.**
   - `cli/_dispatch.py` ← `_dispatch_play`, `_dispatch_train`.
   - `cli/_multi_seed.py` ← `_dispatch_multi_seed_train`,
     `_parse_seed_list`, `_resolve_multi_seed_parent`,
     `_strip_multi_seed_flags`.
   - `cli/_distributed.py` ← `_relaunch_under_torchrun`,
     `_strip_distributed_flags`, `_resolve_per_rank_num_envs`,
     `_extract_log_dir_flag`, `_has_log_dir_flag`,
     `_strip_flag_value_pairs`.
   - `cli/__init__.py` becomes ≤ 400 LoC of Typer wiring.
3. **Non-goals.** No behavioral change. No flag renames. No command
   reordering. No new helper logic. No public-API changes (CLI
   surface is the only public API here, and it must stay byte-identical).
4. **Affected modules.** `cli/__init__.py` (shrinks), three new
   `cli/_*.py` files.
5. **Dependency changes.** None.
6. **PR slices.** 3 PRs, in order:
   - PR4.1 ✅ (#92, `588f5be`): `cli/_distributed.py` (smallest blast
     radius; pure plumbing). `cli/__init__.py` 1,020 → 900 LoC. Six
     functions + constant moved verbatim; `__all__` added to silence
     `reportUnusedFunction`; re-export shim kept all callers + both CLI
     test modules working with zero edits.
   - PR4.2 ✅ (#94, `43cf463`): `cli/_multi_seed.py` (four functions +
     constant; imports the argv-strip helpers from `_distributed`).
   - PR4.3 ✅ (#95, `4ca926d`): `cli/_dispatch.py` (`_dispatch_play`,
     `_dispatch_train`, + the exclusively-used `_AGENT_KINDS` /
     `_coerce_prof_kwargs` / `_parse_*` cluster co-located to avoid the
     cycle). Completes Phase R4.
7. **Test strategy.**
   - Full `tests/test_cli.py` (1,414 LoC) and `tests/test_multi_seed_cli.py`
     per PR.
   - R0 snapshots: `--help` diff must be empty.
   - Per PR, paste a `find_referencing_symbols` audit for each moved
     function showing only intra-file callers were touched.
8. **Risk level.** Medium — large file, large test surface, but the test
   net is dense and R0 snapshots are baselined.
9. **Rollback.** Per-PR revert. The moved files become orphans on
   revert; their content moves back inline.
10. **Completion criteria.** (all three PRs merged)
    - `cli/__init__.py` ≤ 400 LoC. → ⚠️ **not met: 645 LoC** (1,051 → 900
      → 775 → 645). All three named modules were extracted, but the ≤400
      figure was an estimate that didn't account for the residue ADR-0004
      deliberately keeps in `__init__.py` (10 Typer command callbacks,
      `_configured_task` / `_resolve_task`, override helpers, `_RunnableTask`,
      help text). Reaching ≤400 is a documented follow-up (new concern / ADR).
    - 3 new files created. → ✅ 3 / 3 (`_distributed.py`, `_multi_seed.py`,
      `_dispatch.py`).
    - `tests/test_cli.py` and `tests/test_multi_seed_cli.py` green. → ✅
      (R4.2/R4.3 needed mechanical monkeypatch-target repoints — see lessons).
    - `--help` snapshot diff empty for all 8 commands. → ✅ all three PRs.

---

### Phase R5 — Task-specific rewards out of `mdp/rewards.py` / 拆出任务专属奖励 — ✅ COMPLETE (R5.1 #97, R5.2 #98)

1. **Goal.** Separate motion-tracking rewards from the generic reward
   library; parameterize the near-identical motion variants.
   Decision recorded in ADR-0006.
2. **Scope (as shipped).**
   - Created `mdp/motion_tracking.py`.
   - Moved the **whole** "motion imitation" section verbatim — the six
     public functions + the `_motion_command` / `_body_index_filter`
     helpers (ADR-0006 named only three; the family had grown — variance
     recorded, maintainer-confirmed).
   - Added `motion_body_error_exp(env, command_name, std, body_names=None,
     *, quantity)` factory; the three jaccard-1.000 body-error rewards
     became thin `def` wrappers (kept as `def`, not `partial`, so
     `__name__` / signatures are unchanged).
   - `mdp/rewards.py` re-exports the six (`from genelab.mdp.motion_tracking
     import …`), so `genelab.mdp.motion_*` and `genelab.mdp.rewards.motion_*`
     both keep working.
3. **Non-goals.** No new reward families. No behavioral change to the
   moved rewards (verified numerically — see test strategy).
4. **Affected modules (as shipped).** `mdp/rewards.py` (554 → 460 LoC),
   `mdp/motion_tracking.py` (new, 130 LoC), `mdp/__init__.py` (factory
   export only). `examples/unitree/` and the tests are unchanged.
5. **Dependency changes.** None.
6. **PR slices.** 2 PRs:
   - PR5.1 ✅ (#97, `9bcbe01`): relocate the whole motion family verbatim;
     re-export from `rewards.py`. Pure file move.
   - PR5.2 ✅ (#98, `67eef60`): parameterize via `motion_body_error_exp`;
     the three jaccard-1.000 names become wrappers. Completes Phase R5.
7. **Test strategy (as shipped).**
   - Full suite green per PR (395 → 398).
   - PR5.2 added `tests/test_motion_tracking_equivalence.py` — for each of
     the three reward names, builds a small batch of fake states and asserts
     the wrapper + factory output equals the **pinned original**
     implementation bit-for-bit (`torch.equal`), with and without the
     `body_names` filter, plus a distinct-signal guard.
   - (The Unitree G1 reference-run check from the original plan was not run
     — the bit-for-bit equivalence test is the stronger guarantee, and a
     reference-runs doc does not yet exist; deferred to M1.7.)
8. **Risk level.**
   - PR5.1: low (file move). — borne out.
   - PR5.2: medium (parameterization — target-arch risk R9). — mitigated
     by the bit-equivalence test; scope held to the jaccard-1.000 trio.
9. **Rollback.** Per-PR revert. `mdp/rewards.py` re-export block keeps
   import paths stable in both directions.
10. **Completion criteria.** (all met)
    - ✅ The motion family lives in `mdp/motion_tracking.py`.
    - ✅ The jaccard-1.000 `SIMILAR_TO` triple is gone by construction (the
      three bodies are now distinct one-liners); graph re-index runs
      post-merge to confirm jaccard < 0.9.
    - ✅ Equivalence test green (`torch.equal`).
    - ⚠️ Unitree G1 reference run — superseded by the bit-equivalence test;
      no reference-runs doc exists yet (M1.7).

---

### Phase R6 — VecEnv rename and colocation / VecEnv 重命名 — ✅ COMPLETE (PR #100)

1. **Goal.** Disambiguate "wrapper" (env adapter) from "backend"
   (trainer) by relocating env adapters under `rl/vecenvs/`. Decision
   recorded in ADR-0007.
2. **Scope.**
   - Create `rl/vecenvs/{rsl_rl,sb3,skrl}.py` containing the bodies of
     the current `rl/{rsl_rl,sb3,skrl}_wrapper.py`.
   - Replace the old paths with 3-line deprecation shims that re-export
     from the new paths and emit `DeprecationWarning`.
   - Update `rl/backends/{rsl_rl,sb3,skrl}.py` to import from the new
     path.
   - Remove `RslRlVecEnvWrapper` from `rl/__init__.py:__all__`; keep it
     importable via a module-level `__getattr__` shim that emits
     `DeprecationWarning` for one release.
3. **Non-goals.** No class renames (`RslRlVecEnvWrapper`,
   `GenelabSb3VecEnv`, `GenelabSkrlWrapper` keep their names). No
   behavioral change.
4. **Affected modules.** 3 files moved, 3 shim files left at the old
   paths, `rl/__init__.py`, `rl/backends/*`.
5. **Dependency changes.** None.
6. **PR slices.** 1 PR (mechanical rename + shims). Possibly split
   per-library if review bandwidth is tight.
7. **Test strategy.**
   - Pipeline tests (`test_rl_pipeline.py`, `test_sb3_pipeline.py`,
     `test_skrl_pipeline.py`) green.
   - New `tests/test_deprecated_imports.py` asserts that
     `from genelab.rl.sb3_wrapper import GenelabSb3VecEnv` works and
     emits a `DeprecationWarning`.
   - Optional-dep test (R0) still green for the new paths.
8. **Risk level.** Low. Independent of R1–R5; can land any time after R0.
9. **Rollback.** `git revert`. Shim files revert to their original
   full content; new `rl/vecenvs/` directory becomes orphaned (delete in
   the revert).
10. **Completion criteria.**
    - `rl/vecenvs/{rsl_rl,sb3,skrl}.py` populated.
    - 3 shim files emit `DeprecationWarning`.
    - All pipeline tests green.

---

### Phase R7 — Public extension API + importlinter blocking / 公开扩展接口 + CI 强制分层 — ✅ COMPLETE (PRs #101–#106)

1. **Goal.** Codify the third-party extension contract and flip the
   layering contract from lint-only to a required CI check. Decisions
   recorded in ADR-0008 and ADR-0009.
2. **Scope.**
   - Create `src/genelab/extensions.py` re-exporting `ROBOTS`, `ENVS`,
     `TASKS`, `register_robot`, `register_env`, `register_task`,
     `register_backend`, `Backend`, `Runnable`.
   - Promote `cli/__init__.py:_RunnableTask` (private Protocol) →
     `registry.Runnable` (public Protocol). Old name kept as alias for
     one release.
   - Add `docs/concepts/extensions.{en,zh}.md` documenting the
     contract.
   - Flip the importlinter contract introduced in R0 from non-blocking
     to required.
3. **Non-goals.** No new extension types (no new registry kinds beyond
   the four already present). No new backend implementations. No
   behavioral change to the registry itself.
4. **Affected modules.** `extensions.py` (new), `registry.py`
   (`Runnable`), `cli/__init__.py` (alias), `docs/concepts/`,
   `pyproject.toml` (importlinter blocking), `.github/workflows/ci.yml`.
5. **Dependency changes.** importlinter contract becomes a required
   CI gate.
6. **PR slices.** 3 PRs:
   - PR7.1: `extensions.py` + `registry.Runnable` + `cli/__init__.py`
     alias.
   - PR7.2: `docs/concepts/extensions.{en,zh}.md`.
   - PR7.3: flip importlinter to blocking; add explicit ignores for any
     legitimate exception with a TODO + issue link.
7. **Test strategy.**
   - PR7.1: `tests/test_extensions_api.py` — round-trip register/lookup
     for each of robot/env/task/backend via the new public path.
   - PR7.3: re-run the full test suite under importlinter in blocking
     mode; CI green.
8. **Risk level.**
   - PR7.1 / PR7.2: low (purely additive).
   - PR7.3: medium — stale branches may break on rebase if they
     violate the now-blocking contract.
9. **Rollback.** Per-PR revert.
   - PR7.1 / PR7.2 are additive; revert is safe but rarely needed.
   - PR7.3 can be reverted independently; importlinter falls back to
     lint-only mode.
10. **Completion criteria.**
    - `from genelab.extensions import register_backend, Backend`
      works as documented.
    - Docs page live.
    - CI shows importlinter as a required check on `main`.
    - ADR-0001 through ADR-0010 all merged.

---

### 9.3 Interleaving with M1–M3

R-phases are independent of M1–M3 feature work in most places. Specific
hand-offs:

| Feature work | Recommended R-phase already landed |
|---|---|
| M1.3 `genelab export` follow-ups touching `rl/exporter.py` | R1 (no longer pulls `rl.runner` in transitively from `rl.backends.*`) |
| M2.5 `MlpResidualActuator` | R2.4 (`_initialize_pd_common` is the natural base for the new actuator) |
| M3.6 multi-robot API RFC | R3 + R4 (CLI no longer makes hard assumptions about single-robot env layout) |
| Any new RL backend (CleanRL / Tianshou per §5) | R7 (extension API documented) |
| Any new MDP term family (e.g. dexterous-manipulation rewards) | R5 (precedent for task-specific reward modules) |

The R-phase chain does **not** need to fully complete before M1 / M2 /
M3 features ship. The two roadmaps progress in parallel.

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
