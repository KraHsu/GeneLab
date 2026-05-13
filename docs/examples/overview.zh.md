# 示例

仓库在 `examples/` 下提供数个参考扩展，同时也是 CLI 与注册表的集成测试。

## genelab_examples

路径：[`examples/genelab_examples/`](https://github.com/KraHsu/GeneLab/tree/main/examples/genelab_examples)

仓库内的标准扩展，接通两个任务：

- **`wuji_hand`** —— 手部操作任务。
- **`rubiks`** —— 魔方任务。

`pyproject.toml` 声明了 `genelab.extensions` entry point，因此安装该包后会被自动发现
（通过项目 `pyproject.toml` 的 `pythonpath` 设置，测试也可以直接 import，而无需安装）。

## unitree

路径：[`examples/unitree/`](https://github.com/KraHsu/GeneLab/tree/main/examples/unitree)

聚焦 Unitree 平台的机器人示例。形态与 `genelab_examples` 相同 —— entry point、`register()`、
按模块拆分的注册文件。

## external_project

路径：[`examples/external_project/`](https://github.com/KraHsu/GeneLab/tree/main/examples/external_project)

下游项目最小模板。`genelab project new` 生成的内容与之结构一致，留在仓库里作为脚手架输出参考。

## 使用示例

```bash
# 确认示例任务可见。
uv run genelab list tasks

# 可视化运行一个任务。
uv run genelab play wuji_hand --vis --steps 200
```

## 另见

- [新建项目](../cli/project-new.md) —— 生成自己的扩展包。
- [扩展加载](../concepts/extensions.md) —— 扩展如何被发现与加载。
