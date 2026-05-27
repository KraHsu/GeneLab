# CLI 总览

`genelab` CLI 是注册表和任务配置上的薄调度层。它不拥有任务逻辑；它负责发现扩展、解析已注册对象、应用 override，然后调用 task 或 runner。

## 命令模型

```bash
genelab [全局选项] <命令> [参数]
```

| 区域 | 命令 |
|---|---|
| 注册表发现 | `list robots`、`list envs`、`list tasks`、`info NAME` |
| 运行时 | `play TASK`、`train TASK` |
| 工具 | `cache`、`prof open` |
| 项目骨架 | `project new NAME` |

## 扩展加载顺序

任何需要注册表数据的命令都会按顺序加载：

1. 通过 `load_bundled_asset_zoo()` 加载随包 asset zoo robot。
2. 已安装的 `genelab.extensions` entry point，除非设置 `--no-entry-points`。
3. 重复传入的显式 `--import MODULE`。

日常工作用 entry point，本地实验用 `--import`。

## Overrides

运行时命令在 task id 后接受未知的 `--a.b.c VALUE` 选项。CLI 会把它们转发给
`apply_overrides`。

```bash
genelab play TASK_ID --env.simulation.dt 0.005
```

用 `genelab info TASK_ID` 查看有效路径。

## 交互模式

stdin 是 TTY 时，CLI 可以为缺失 task id、未知注册表名字、非法 `--agent`、未知 override 路径弹出选择器。CI 和脚本中会直接抛出相同错误。

## 另见

- [CLI 参考](../reference/cli.md)
- [发现：list 与 info](list-info.md)
- [play 与 train](play-train.md)
