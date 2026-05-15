# 发现：list 与 info

`list` 与 `info` 子命令是 GeneLab 的发现入口 —— 它们把已安装的扩展打印成一张
"可用 task / env / robot + 可覆盖 dotted path" 的地图，供 `play` / `train` 的
`--<a.b.c> VALUE` 旗标使用。

## 列举注册表

`list KIND` 枚举三个全局注册表之一（`robots` / `envs` / `tasks`）。命令会先导入所
有已发现的扩展，因此列表反映当前安装集。

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

每行打印注册名、说明，以及一条逐条 detail：task 显示 `env=..., robot=...,
trainable=...`；env / robot 显示首批几个标量 cfg 字段。通过
`register_task(..., examples=[...])` 注册了 `examples=[...]` 字符串的 task 会附带
`(N example(s))` 后缀。

## 查看任务

`info NAME` 解析单个注册名并打印 detail 面板。查找顺序是 task → env → robot。
TTY 下名字找不到时会唤起 `questionary` 选择器，从 task / env / robot 名的并集
中挑一个；非交互环境下沿用原行为，直接退出并列出可用名。

```bash
uv run genelab info GeneLab-Sensors-Showcase-v0
```

Task 面板分三块：

1. **Description 与 meta** —— 一行 description，加 task cfg 上拉出的
   `env=..., robot=..., trainable=...` 元信息条。
2. **Overridable cfg paths** —— 三列表格（`Path` / `Type` / `Default`），
   通过深度优先遍历 cfg dataclass 树生成。以 `_` 开头的字段名被跳过；嵌套
   dataclass 会以父节点 dotted path 为前缀就地展开。
3. **Examples** —— 扩展在注册时提供的原文 `examples=[...]` 行。仅展示，
   `info` 不会执行它们。

env 与 robot 渲染同样的面板，但没有 `env / robot / trainable` 元信息条。

## Override 路径

"Overridable cfg paths" 表里的每一行就是 `play` / `train` 能直接接受的 dotted
path。CLI 把每个 `--<a.b.c> VALUE` 旗标转发给
`genelab.configs.apply_overrides`，后者在 cfg 构建期按目标字段类型 hint 做 coerce。

| `info` 输出行 | 等价的 override 旗标 |
|---|---|
| `env.simulation.vis` (bool, false) | `--vis`（捷径）或 `--env.simulation.vis true` |
| `env.simulation.gpu` (bool, false) | `--gpu`（捷径）或 `--env.simulation.gpu true` |
| `env.simulation.steps` (int, 240) | `--steps 500`（捷径）或 `--env.simulation.steps 500` |
| `env.simulation.dt` (float, 0.01) | `--dt 0.005`（捷径）或 `--env.simulation.dt 0.005` |
| `env.rewards_cfg.track_lin_vel.weight` (float, 1.0) | `--env.rewards_cfg.track_lin_vel.weight 2.0` |
| `env.rewards_cfg.track_lin_vel.params.std` (float, 0.25) | `--env.rewards_cfg.track_lin_vel.params.std 0.5` |
| `env.scene.env_spacing` (tuple, `(2.0, 2.0)`) | `--env.scene.env_spacing 3.0,3.0` |
| `env.scene.batch_render` (bool, false) | `--env.scene.batch_render true` |

四个 simulation 捷径（`--vis` / `--gpu` / `--steps` / `--dt`）在转发前会被改写成
`env.simulation.{vis,gpu,steps,dt}` 形式，因此 profiler trace argv 中显示的形态
与显式 override 完全一致。

## 示例命令清单

针对 inverted-pendulum 任务的发现流程：

```bash
# 确认扩展已安装并挑一个 task id。
uv run genelab list tasks | grep Pendulum

# 钻入单 task —— 从 "Overridable cfg paths" 表里复制一个 dotted path。
uv run genelab info GeneLab-Inverted-Pendulum-v0

# 带 override 跑 task。
uv run genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 240 \
    --env.rewards_cfg.pole_upright.weight 5.0
```

同样的流程对 env 与 robot 也成立：`genelab info FrankaPandaCfg`（robot）会列出
所有 `ArticulationCfg` 字段，包括 per-group 的 `actuators.*.stiffness` 与
`default_joint_pos.*` 键。

## See also

- [Play and Train](play-train.md)
- [Configs](../concepts/configs.md)
- [Managers and MDP terms](../concepts/managers.md)
