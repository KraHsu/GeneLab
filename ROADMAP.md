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
> Last reviewed: 2026-05-21 · against `dev` @ `1c80b41`.

### 9.0 Status / 当前状态

**Phase:** Planning complete · Implementation not started.

| What | Status | Artifact |
|---|---|---|
| Architecture assessment | ✅ written | [`plans/architecture/architecture-assessment.md`](plans/architecture/architecture-assessment.md) |
| Target architecture | ✅ written | [`plans/architecture/target-architecture.md`](plans/architecture/target-architecture.md) |
| ADRs 0001–0010 | ✅ drafted (all Status: Proposed) | [`plans/adr/`](plans/adr/) |
| Phase R0 — baseline & tooling | ⬜ not started | — |
| Phase R1 — break rl.runner ↔ rl.backends cycle | ⬜ not started | gated on ADR-0001 acceptance |
| Phase R2 — small abstractions | ⬜ not started | gated on ADR-0002, ADR-0003 |
| Phase R3 — domain-owned parsing | ⬜ not started | gated on ADR-0005 |
| Phase R4 — CLI decomposition | ⬜ not started | gated on R3 landing + ADR-0004 |
| Phase R5 — task-specific rewards split | ⬜ not started | gated on ADR-0006 |
| Phase R6 — vecenv rename | ⬜ not started | gated on ADR-0007 |
| Phase R7 — extensions API + importlinter blocking | ⬜ not started | gated on R0–R6 landing + ADR-0008, ADR-0009 |
| Phase deferred — entity/articulation split | ⏸ deferred (criteria recorded) | [ADR-0010](plans/adr/0010-defer-articulation-split.md) |

**Completed in this planning round.**

- New: [`plans/architecture/architecture-assessment.md`](plans/architecture/architecture-assessment.md), [`plans/architecture/target-architecture.md`](plans/architecture/target-architecture.md), [`plans/adr/0001-…0010`](plans/adr/), [`plans/adr/README.md`](plans/adr/README.md), [`CLAUDE.md`](CLAUDE.md).
- Modified: this `ROADMAP.md` (§9 added — refactor phases).
- **Not modified:** any file under `src/genelab/`, `tests/`, `examples/`, `docs/`. Production code has not been touched as part of the refactor.

**Tests run.** None — the refactor has not begun, so there is no refactor diff to validate. The pre-refactor M1 feature work that landed earlier on `dev` (eval / export / backends — see commits `99389d3..1c80b41`) was tested in its own PRs; those results stand and are not affected by this planning round.

**Changes to the dependency graph.** None. Re-running `mcp__codebase-memory-mcp__index_repository` produces the same 3,829-node / 11,427-edge graph that was indexed at the start of the planning round. The hotspots and duplications flagged in the assessment are still present.

**Risks identified (aggregate view).** Detailed per-ADR; the highest-attention items:

- **R1 (ADR-0001)** — quiescent `rl.runner ↔ rl.backends` cycle held together by lazy `importlib`. Any contributor "cleaning up" backend imports would break the package; R1 fixes this permanently.
- **R2.5 (ADR-0002)** — `BaseTermManager._post_init` ordering must preserve current buffer-allocation timing in `RewardManager` / `TerminationManager`. Mitigation: pre-R2.5 assertion test gates the refactor.
- **R4 (ADR-0004)** — CLI decomposition has large test surface (`test_cli.py` is 1,414 LoC). Mitigation: R0 `--help` snapshots gate every PR.
- **R5.2 (ADR-0006)** — parameterizing motion-tracking rewards risks numerical drift. Mitigation: bit-equivalence test ships in the same PR.
- **R7 (ADR-0009)** — flipping importlinter from lint-only to blocking may surface latent layering violations not caught in R0. Mitigation: R0 is lint-only for the entire R1–R6 window; baseline must be clean before R7 flips.

**Next steps.**

1. Maintainer review of ADRs 0001–0010. Approve / request-changes per ADR.
2. Move accepted ADRs to `Status: Accepted` (one-line edit per file).
3. Land **PR R0.1** — the suggested next slice (see below). R0.1 is the smallest possible starting move, unblocks R3 and R4, and changes no production code.
4. After R0 lands, fan out: R1 / R5 / R6 can run in parallel; R3 → R4 serial.

**Suggested next slice: PR R0.1 — CLI `--help` snapshot baseline.**

- **Scope.** Add `tests/snapshots/help-*.txt` (one per CLI command — root, `cache`, `list`, `info`, `play`, `eval`, `export`, `train`, `project_new`) + `tests/test_cli_help_snapshots.py` that captures `python -m genelab <cmd> --help` and asserts byte-equality against the snapshots.
- **Why first.** ADR-0004 (CLI decomposition) and ADR-0005 (domain-owned parsing) both depend on having a frozen `--help` baseline. Without R0.1, regressions in those phases are invisible.
- **Production change.** None.
- **Risk.** Low. The snapshots capture what Typer prints today; if a later PR changes the output, the diff is the signal.
- **Reviewer effort.** Small — one new test file, nine snapshot files.

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
R0 ──┬──► R1 ─────────────────────────────────────────► R7
     ├──► R2 (R2.1–R2.5 parallel) ────────────────────► R7
     ├──► R3 ──► R4 ─────────────────────────────────► R7
     ├──► R5 ─────────────────────────────────────────► R7
     └──► R6 ─────────────────────────────────────────► R7
```

R0 is the gate. R1 / R2 / R5 / R6 fan out in parallel. R3 must land
before R4 (smaller CLI seam). R7 is the closer.

---

### Phase R0 — Baseline & Tooling / 基线与工具

1. **Goal.** Establish the safety net (snapshot tests, optional-dep test,
   importlinter in lint-only mode) before touching production code.
   Nothing in `src/genelab/` changes.
2. **Scope.**
   - CLI `--help` snapshot tests for every command (`tests/snapshots/help-*.txt`,
     `tests/test_cli_help_snapshots.py`).
   - Optional-dep subprocess test (`tests/test_optional_deps.py`) — boots
     `import genelab.rl` and each `genelab.rl.backends.<lib>` with
     `rsl_rl` / `skrl` / `stable_baselines3` / `tensordict` poisoned in
     `sys.modules`.
   - Layering contract scaffold (`pyproject.toml` `[tool.importlinter]`)
     in **lint-only** mode — runs in CI but does not fail the build yet.
3. **Non-goals.** No production code change. No new public APIs. No
   importlinter rule is yet blocking.
4. **Affected modules.** `tests/`, `pyproject.toml`,
   `.github/workflows/ci.yml`. **No file under `src/genelab/` is touched.**
5. **Dependency changes.** Add `import-linter` to dev-deps.
6. **PR slices.**
   - PR0.1: snapshot baseline (`tests/snapshots/` + reader test).
   - PR0.2: optional-dep test scaffold.
   - PR0.3: importlinter config (lint-only) + CI step (non-blocking).
7. **Test strategy.** New tests must run green on the current `dev` —
   they are pure observations, not assertions about a target state.
8. **Risk level.** Low.
9. **Rollback.** `git revert <merge-sha>` per PR. No source code is
   affected, so revert is trivial.
10. **Completion criteria.**
    - 3 PRs merged.
    - CI shows green snapshot diff and a non-blocking importlinter step.
    - `tests/test_optional_deps.py` passes on the matrix `{rsl_rl, skrl, sb3, tensordict} × {present, absent}`.

---

### Phase R1 — Break `rl.runner` ↔ `rl.backends` cycle / 解开 RL 循环依赖

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

### Phase R2 — Small abstractions / 抽取重复抽象

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
