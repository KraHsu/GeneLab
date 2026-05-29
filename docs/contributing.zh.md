# 贡献指南

面向修改 GeneLab 本身的贡献者。下游机器人项目通常应使用扩展包。

## 开发环境

```bash
uv sync --extra torch-cpu
genelab cache
pytest
ruff check
pyright
```

开发 GPU 工作流时，用一个 CUDA extra 替代 `torch-cpu`。

## 文档规则

文档遵循 Diátaxis：

| 类型 | 用途 |
|---|---|
| Tutorial | 从零到可运行结果的学习路径。 |
| How-to | 面向有经验用户的任务指南。 |
| Reference | 事实、flag、API、默认值。 |
| Explanation | 架构和设计原因。 |

尽量让单页只承担一种文档类型，通过交叉链接连接，而不是把长篇概念解释塞进任务指南。

## 代码风格

- 使用 Python 3.12+ 语法。
- 保持注册阶段导入轻量；不要在 import 时启动 Genesis。
- 用户可见旋钮优先使用 dataclass 配置。
- 除非属于框架本身，否则 task/example 代码不要放进 `src/genelab/`。

## PR 前

```bash
pytest
ruff check
pyright
mkdocs build --strict
```

docs-only 改动也要跑 MkDocs strict 构建。
