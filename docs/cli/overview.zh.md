# CLI 总览

`genelab` CLI 是注册表（registry）与任务配置之上的薄分发层。它不持有 task 逻辑——
发现扩展、解析已注册对象、应用 override，然后调用 task 或 runner。

## 命令模型

```bash
genelab [全局选项] <命令> [参数]
```

| 分组 | 命令 |
|---|---|
| Registry | `list robots\|envs\|tasks`、`info NAME`、`asset list\|info\|download\|purge` |
| Runtime | `play TASK`、`train TASK`、`eval TASK CHECKPOINT`、`export TASK CHECKPOINT`、`benchmark --suite ...` |
| Project | `project new NAME` |
| Utilities | `cache`、`prof open` |

不带子命令运行 `genelab` 会打印 landing 页：包含 quickstart 命令以及已注册的
robots / envs / tasks 数量。

## 全局选项

| 选项 | 含义 |
|---|---|
| `--version` | 打印版本并退出。 |
| `--import MODULE` | 分发前导入扩展模块；可重复。 |
| `--no-entry-points` | 跳过 `genelab.extensions` entry point 组的扩展。 |

## 扩展加载顺序

每个需要 registry 数据的命令按以下顺序加载扩展：

1. 通过 `load_bundled_asset_zoo()` 装入随包的 asset zoo 机器人。
2. 已安装的 `genelab.extensions` entry points（除非传 `--no-entry-points`）。
3. 多次显式的 `--import MODULE`。

日常工作用 entry points，临时实验用 `--import`。

## Overrides

Runtime 命令在 task id 后接受未知的 `--a.b.c VALUE` 选项，由 CLI 转发到
`apply_overrides`：

```bash
genelab play TASK_ID --env.simulation.dt 0.005
```

可用路径见 `genelab info TASK_ID`。

## 交互模式

当 stdin 是 TTY 时，CLI 可以为缺失的 task id、未知的注册名、非法 `--agent` 或
未知 override 路径弹出交互选择器；CI 与脚本中则直接抛出同样的错误。

## 另见

- [发现：list 与 info](list-info.md)
- [play 与 train](play-train.md)
- [Eval 与 export](../concepts/eval-and-export.md)
