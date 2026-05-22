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
> Last reviewed: 2026-05-22 · against `dev` @ `5571889`.

### 9.0 Status / 当前状态

**Phase:** **R0 + R1 + R2 complete** · R3–R7 not started.

| What | Status | Artifact |
|---|---|---|
| Architecture assessment | ✅ written | [`plans/architecture/architecture-assessment.md`](plans/architecture/architecture-assessment.md) |
| Target architecture | ✅ written | [`plans/architecture/target-architecture.md`](plans/architecture/target-architecture.md) |
| ADRs 0001–0010 | ✅ drafted (all Status: Proposed; ADR-0001 + ADR-0002 + ADR-0003 ready for `Accepted` flip) | [`plans/adr/`](plans/adr/) |
| Phase R0 — baseline & tooling | ✅ **complete** (3 / 3 PRs merged) | PR #81 (`c3d851e`), #82 (`b8df293`), #83 (`d7a702e`) |
| Phase R1 — break rl.runner ↔ rl.backends cycle | ✅ **merged** | PR #84 (`04509d9`) — implements ADR-0001 |
| Phase R2 — small abstractions | ✅ **complete** (5 / 5 sub-slices merged) | PRs #85 (`74b2e30`), #86 (`039e502`), #87 (`0367462`), #88 (`a50d031`), #89 (`5571889`) — implement ADR-0003 (+ ADR-0002 for R2.5) |
| Phase R3 — domain-owned parsing | ⬜ **next recommended** | gated on ADR-0005 |
| Phase R4 — CLI decomposition | ⬜ not started | gated on R3 landing + ADR-0004 |
| Phase R5 — task-specific rewards split | ⬜ not started | gated on ADR-0006 |
| Phase R6 — vecenv rename | ⬜ not started | gated on ADR-0007 |
| Phase R7 — extensions API + importlinter blocking | ⬜ not started | gated on R0–R6 landing + ADR-0008, ADR-0009 |
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

**Changes to the dependency graph.** `mcp__codebase-memory-mcp__index_status` reports **3,887 nodes / 10,905 edges** post-R2.5 (was 3,829 / 11,427 at the start of the planning round). Net deltas across all merged work: +58 nodes (new test files plus the `_helpers.py` / `_algorithm_taxonomy.py` / `_attach_base.py` / `_joint_match.py` modules, the `ActuatorBase._write_pd_gains` method, and the `BaseTermManager` class with its methods/properties) and −522 edges. The large edge reduction is structural: the codebase-memory indexer re-resolves the graph after R0.3's `exclude_type_checking_imports = true` flag and after each consolidation (collapsing duplicate caller / property / method edges into a single canonical helper-import or inherited-base edge). The hotspots flagged in the assessment are being trimmed slice-by-slice; the cycle (R1) and all five cluster duplications (R2.1–R2.5) are gone.

**Risks identified (aggregate view).** Detailed per-ADR; the highest-attention items:

- **R1 (ADR-0001)** — ✅ **resolved by PR #84.** The quiescent `rl.runner ↔ rl.backends` cycle is gone; backends now import from `rl._helpers`. `tests/test_no_static_cycle.py` (grimp-based) prevents regression; matching importlinter contract is now KEPT.
- **R2.5 (ADR-0002)** — ✅ **resolved by PR #89.** `BaseTermManager._post_init` preserves buffer-allocation timing; `tests/test_manager_init_order.py` was verified green pre- and post-refactor. The risk did not materialize.
- **R4 (ADR-0004)** — CLI decomposition has large test surface (`test_cli.py` is 1,414 LoC). Mitigation: R0.1 `--help` snapshots are now in place and gate every PR.
- **R5.2 (ADR-0006)** — parameterizing motion-tracking rewards risks numerical drift. Mitigation: bit-equivalence test ships in the same PR.
- **R7 (ADR-0009)** — flipping importlinter from lint-only to blocking is gated on the R0.3 baseline being clean. Post-R2: **21 cross-layer imports remain across 2 broken contracts** (down from 24 / 3 — R1 trimmed 3; R2.1–R2.5 were intra-package dedup and did not touch the cross-layer baseline).

**Risks discovered during R0 + R1 (new, not in the original list).**

- **Rich/Typer help output is not deterministic under env-var pinning** (R0.1). `TERM=dumb` / `COLUMNS=N` / `NO_COLOR=1` are honoured differently across Rich versions and CI runners. GitHub Actions sets `FORCE_COLOR=1` which Rich honours *over* `NO_COLOR=1`. The portable mechanism is pinning Typer's module-level constants in a `python -c` wrapper before import: `typer.rich_utils.MAX_WIDTH = 100`, `typer.rich_utils.FORCE_TERMINAL = False`. Any future test that diff-checks Rich-rendered output must follow this pattern. Captured in `tests/test_cli_help_snapshots.py`'s module docstring.
- **Doc drift: ROADMAP §9.0 originally enumerated 9 CLI commands; the real surface has 11** (R0.1). `prof` and the `project` subapp's own `--help` were missed. The R0.1 snapshot baseline covers all 11; this document has been updated. New CLI commands or subapps must be added to both `tests/test_cli_help_snapshots.py:HELP_COMMANDS` *and* the §9.0 list.
- **Local lint gates must be run before pushing** (R0.2). The first R0.2 push failed CI because `ruff format --check` flagged the implicit-string-concatenation in the subprocess wrapper. Triple-quoted strings are ruff-stable. Future test-only PRs should run `uv run ruff format --check` and `uv run ruff check` locally before pushing — both are part of CI's `lint` job.
- **The layering spec in `target-architecture.md` / ADR-0009 has more drift from current code than the assessment suggested** (R0.3). With `exclude_type_checking_imports = true`, the R0.3 baseline showed **24 distinct cross-layer imports** across 3 broken contracts; R1 trimmed 3 of those. Root causes by category (post-R1):
  - ~~3 imports: `rl.backends.{rsl_rl,skrl,sb3} → rl.runner` — addressed by **R1 / ADR-0001**.~~ ✅ Resolved by PR #84.
  - 1 import: `rl.eval_callback → cli._eval (l.114)` — addressed by **R3 / ADR-0005**.
  - 1 import: `scene.interactive_scene → rl.distributed (l.139)` — already function-local (lazy) inside `build()`; either move `pin_cuda_device` out of `rl/` or accept a TODO-tagged waiver in R1.
  - 6 imports: `asset_zoo.{cartpole,franka,unitree_g1,unitree_g1_motions,unitree_go1,anymal_c} → utils.download` — asset factories call `fetch_asset` directly; needs a small PR or new ADR.
  - 5+5 imports: `asset_zoo.* → actuator / entity` — asset factories build entities. Probably needs an "assets reach into domain" ADR.
  - ~8 imports: `envs.manager_based_rl_env → {bridges, entity, managers, scene, sensor}` — integration-point realities; some may stay, some can move behind interfaces during R3/R4.
  - Remaining imports (`scene.interactive_scene → {entity, recording, sensor, terrains}`, `entity.articulation → actuator`, `configs → {sensor, entity, recording, terrains}`, `utils.download → cache`) — each needs per-file audit during the relevant R-phase or a dedicated PR.
- **R7 (importlinter blocking flip) is gated on the remaining 21 violations being cleaned up or explicitly waived.** Each R-phase PR should record which baseline violations it eliminates. Status: 3 / 24 resolved (all by R1). R2.1–R2.5 were intra-package dedup and did not touch the cross-layer baseline. **R3 is the next slice expected to reduce it** (`rl.eval_callback → cli._eval`).
- **ADR-0001 §Migration plan's test spec was flawed** (R1). It said *"imports `genelab.rl.backends.sb3` in a subprocess and asserts `genelab.rl.runner` is not in `sys.modules` afterwards."* That spec assumed `rl/__init__.py` does not import `runner` — but it does (for the public `play_task` / `train_task` re-exports), so any submodule import of `genelab.rl` unavoidably pulls `runner` into `sys.modules` via parent-package init. The `sys.modules` check can never detect the actual cycle for `rl.backends.*` modules. PR #84 used a `grimp`-based static-graph test instead, which asserts no `rl.backends.<lib>` directly imports `rl.runner` — the real invariant. ADR-0001 §Migration plan has been amended to reflect this. Future "static cycle" tests should use the grimp pattern, not the sys.modules pattern.
- **`# noqa: F401` does not silence pyright's `reportUnusedImport`** (R2.1). For re-export shims, use the PEP-484 explicit-re-export idiom `from foo import bar as bar` — recognized by both ruff and pyright (and mypy). No `__all__` maintenance required. Future R-phases that introduce re-export shims should follow this pattern; `noqa` comments alone will fail CI's typecheck job.
- **Ruff E402 "module level import not at top of file" fires even when the call site is the bottom of the file** (R2.2). Pattern: import the helper at the top with the other imports; place the *call* wherever execution requires it (e.g. at module bottom for "register at module load" patterns). Future R-phases that wire bottom-of-file registration calls should put the imports up top.
- **Match cfg field types when extracting helpers** (R2.3). Dataclass fields often default to `tuple[str, ...]` (immutable) rather than `list[str]`. Helpers that accept inputs originating from cfg should type their parameters as `Sequence[str]` (or `Iterable[str]`) so both `tuple` and `list` callers work without conversion. Pyright catches the mismatch — worth checking dataclass field types before drafting helper signatures.
- **ADR-0003 R2.2 names the new module `rl/vecenvs/_attach_base.py`, but `rl/vecenvs/` is created in R6** (ADR-0007). PR #86 placed the helper flat at `rl/_attach_base.py` and noted the deferred relocation in both the PR description and CHANGELOG. R6 will move it. Same care should be taken for any future R-phase slice that targets a directory the upstream ADR assumes already exists.
- **`Generic[TVar]` is required for a shared base whose `__init__` accepts a `dict[str, <subclass-cfg>]`** (R2.5). `dict` is invariant in its value type, so a base `__init__(cfg: dict[str, ManagerTermBaseCfg])` rejects `dict[str, RewardTermCfg]` at the call site under pyright. `BaseTermManager(Generic[TCfg])` with `TCfg = TypeVar(..., bound=ManagerTermBaseCfg)` lets each subclass narrow (`BaseTermManager[RewardTermCfg]`) without losing call-site type info. Future shared-base extractions over heterogeneous cfg dicts should use the same pattern.
- **Init-order gate tests are cheap insurance for "pull __init__ up to a base" refactors** (R2.5). The `_post_init` template method moves buffer allocation across a method boundary; a 3-assertion test (buffers populated, keys mirror term names, empty-cfg → empty dicts) committed *before* the refactor and verified green pre- and post-move turns a "medium-risk" refactor into a low-risk one. Pattern worth reusing for R4 (CLI split) and any future base-class extraction.
- **`metrics_manager.py` and `curriculum_manager.py` carry the same term-registration loop as the two managers R2.5 consolidated** (discovered during R2.5 audit). They are *out of ADR-0002's scope* (which targets rewards + terminations only) so were left untouched. They are a clean follow-up: adopting `BaseTermManager` there would remove two more copies of the registration loop. Candidate for a future R2.x-style PR or an ADR-0002 addendum.

**Next steps.**

1. Maintainer review of ADRs 0001–0010. **ADR-0001** (cycle break, shipped), **ADR-0002** (BaseTermManager, shipped via R2.5), and **ADR-0003** (small dedup, all 5 sub-slices shipped) are strong candidates to flip from `Proposed` to `Accepted`.
2. Move accepted ADRs to `Status: Accepted` (one-line edit per file).
3. Start **Phase R3** — CLI domain-owned parsing (see below). It is the gate for R4 (CLI decomposition) and resolves 1 more R7-blocker violation.
4. Parallel candidates (the §9.2 graph allows fan-out): **R5.1** (relocate motion-tracking rewards verbatim — pure file move), **R6** (vecenv rename + colocation — also relocates the `rl/_attach_base.py` deferred home). R4 stays serial after R3.
5. R7 (importlinter blocking flip) remains gated on the 21 remaining R0.3-baseline cross-layer imports being cleaned up or waived. R2.1–R2.5 did not change the cross-layer baseline (intra-package dedup only).
6. **Follow-up dedup** (not on the R-phase critical path): apply `BaseTermManager` to `metrics_manager.py` + `curriculum_manager.py` (2 more copies of the registration loop, discovered during the R2.5 audit).

**Suggested next slice: R3 — domain owns its parsing (ADR-0005).**

- **Scope.** Move runner-arg parsing out of `cli/__init__.py` into classmethods on the relevant domain configs: `EvalCallbackCfg.from_args(runner_args) → EvalCallbackCfg | None` (replaces `cli/__init__.py:_build_eval_callback`) and `SimulationCfg.play_retargeted_keys() → tuple[str, ...]` (replaces `cli/__init__.py:_PLAY_RETARGETED_KEYS`). 2 PRs (R3.1 / R3.2) per ADR-0005 §Migration plan.
- **Why next.** (1) **Unblocks R4** — CLI decomposition is serial after R3 (§9.2). (2) **Resolves 1 more R7-blocker baseline violation** — `rl.eval_callback → cli._eval` flips from broken once the parse logic moves into the config classmethod. (3) **R0.1 `--help` snapshots are the safety net** — already in place; any CLI output drift fails the snapshot test (ADR-0005 §Test strategy requires empty snapshot diff).
- **Production change.** Yes — move parsing logic; CLI delegates to the new classmethods. No CLI flag or `--help` text changes (snapshot-enforced).
- **Risk.** Low. ADR-0005 §Rollback keeps the original inline parsing commented out for one deprecation window.
- **Reviewer effort.** Medium — 2 PRs, each with a focused new test (`tests/test_eval_callback_from_args.py`, a `play_retargeted_keys` test in `tests/test_configs.py`) plus the R0.1 snapshot gate.

**Alternative next slices** (unlocked, parallel-safe with R3):

- **R5.1 (ADR-0006 PR1 of 2)** — relocate motion-tracking rewards verbatim. Pure file move.
- **R6 (ADR-0007)** — vecenv rename + colocation. Independent of everything; relocates the R2.2 `rl/_attach_base.py` deferred home.

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
R0 ✅─┬──► R1 ✅ ──────────────────────────────────────► R7
      ├──► R2 ✅ (R2.1–R2.5 all merged) ────────────────► R7
      ├──► R3 ──► R4 ─────────────────────────────────► R7
      ├──► R5 ─────────────────────────────────────────► R7
      └──► R6 ─────────────────────────────────────────► R7
```

R0, R1, and R2 are complete. R3 / R5 / R6 fan out in parallel; R3 must
land before R4 (smaller CLI seam). R7 is the closer (gated on the
remaining 21 importlinter baseline violations being cleared).

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

### Phase R3 — Domain owns its parsing / 解析逻辑下沉到 domain

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
5. **Dependency changes.** None.
6. **PR slices.** 2 PRs:
   - PR3.1: `EvalCallbackCfg.from_args`; CLI delegates.
   - PR3.2: `SimulationCfg.play_retargeted_keys()`; CLI delegates.
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
10. **Completion criteria.**
    - `cli/__init__.py` shrinks by ≥ 35 LoC.
    - New classmethods tested.
    - Snapshot diff empty.

---

### Phase R4 — CLI dispatcher decomposition / CLI 拆解

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
   - PR4.1: `cli/_distributed.py` (smallest blast radius; pure plumbing).
   - PR4.2: `cli/_multi_seed.py` (depends on `_distributed` only via
     argv-strip helpers, which are now imported).
   - PR4.3: `cli/_dispatch.py` (depends on `_distributed` and `_multi_seed`).
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
10. **Completion criteria.**
    - `cli/__init__.py` ≤ 400 LoC.
    - 3 new files created.
    - `tests/test_cli.py` and `tests/test_multi_seed_cli.py` green.
    - `--help` snapshot diff empty for all 8 commands.

---

### Phase R5 — Task-specific rewards out of `mdp/rewards.py` / 拆出任务专属奖励

1. **Goal.** Separate motion-tracking rewards from the generic reward
   library; parameterize the three near-identical motion variants.
   Decision recorded in ADR-0006.
2. **Scope.**
   - Create `mdp/motion_tracking.py`.
   - Move `motion_relative_body_position_error_exp`,
     `motion_global_body_linear_velocity_error_exp`,
     `motion_global_body_angular_velocity_error_exp` verbatim.
   - Add `motion_body_error_exp(quantity, frame, *, std, …)` factory.
   - Keep the three named functions as thin wrappers for back-compat
     (`from genelab.mdp.rewards import motion_*` keeps working via a
     re-export block).
3. **Non-goals.** No new reward families. No behavioral change to the
   three motion-tracking rewards (verified numerically — see test
   strategy).
4. **Affected modules.** `mdp/rewards.py`, `mdp/motion_tracking.py`
   (new), `examples/unitree/` (no change, just verified).
5. **Dependency changes.** None.
6. **PR slices.** 2 PRs:
   - PR5.1: relocate the three functions verbatim; re-export from
     `rewards.py`. Pure file move.
   - PR5.2: parameterize via `motion_body_error_exp`; the three names
     become wrappers.
7. **Test strategy.**
   - `tests/test_rewards.py` green per PR.
   - PR5.2 adds `tests/test_motion_tracking_equivalence.py` — for each
     of the three reward names, builds a small batch of fake states and
     asserts the wrapper output equals the original implementation
     bit-for-bit (within float tolerance).
   - Run `examples/unitree/g1` training for one chunk per PR; reward
     curves must match the reference run from `docs/best-practices/reference-runs.md`.
8. **Risk level.**
   - PR5.1: low (file move).
   - PR5.2: medium (parameterization — target-arch risk R9).
9. **Rollback.** Per-PR revert. `mdp/rewards.py` re-export block keeps
   import paths stable in both directions.
10. **Completion criteria.**
    - 3 functions live in `mdp/motion_tracking.py`.
    - Re-running `search_graph(relation='SIMILAR_TO')` shows
      jaccard < 0.9 for those names.
    - Equivalence test green.
    - Unitree G1 reference run reproduces existing numbers.

---

### Phase R6 — VecEnv rename and colocation / VecEnv 重命名

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

### Phase R7 — Public extension API + importlinter blocking / 公开扩展接口 + CI 强制分层

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
