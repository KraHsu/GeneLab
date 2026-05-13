# 贡献指南

> 详细的贡献者指南目前以英文为权威版本，见仓库根目录
> [`CONTRIBUTING.md`](https://github.com/KraHsu/GeneLab/blob/main/CONTRIBUTING.md)。本页提供
> 关键要点的中文摘要，欢迎贡献完整中文翻译。

## 开发环境

GeneLab 使用 [uv](https://github.com/astral-sh/uv) 并要求 Python `>=3.12`。挑选一个互斥的
`torch-*` extra：

```bash
uv sync --extra torch-cpu     # 或 torch-cu126 / torch-cu128 / torch-cu130
uv run genelab --help         # CLI 烟雾测试
uv run genelab cache          # 创建 Genesis / Matplotlib 用的项目本地 .cache 目录
```

`examples/genelab_examples/` 已加入 `pytest` 的 `pythonpath`，测试可直接 import 而无需安装。

## Python 3.12+ 风格

GeneLab 锁定 Python `>=3.12`。只写现代代码，不要为旧解释器加兼容垫片：

- **不要** `from __future__ import annotations`。注解会立即被求值；只在 `TYPE_CHECKING` 下导入的
  名字在使用点引号化（`def f(env: "ManagerBasedRlEnv") -> ...`）。
- **PEP 604 union**：`X | Y`、`X | None`，不要 `Optional[X]` / `Union[X, Y]`。
- **PEP 585 内置泛型**：`list[T]`、`dict[K, V]`、`tuple[T, ...]`、`type[T]`，不要 `typing.List` 等。
- **PEP 695 泛型**：`class Foo[T]:`、`def f[T](...)`、`type Alias = ...`，不要 `TypeVar + Generic[T]`。
- 用 `collections.abc` 代替 `typing` 中的 `Callable` / `Iterable` / `Iterator` / `Sequence` / `Mapping`。
- 仅以下 `typing` 名字预期出现：`Any`、`cast`、`Protocol`、`Literal`、`Final`、`Annotated`、
  `TYPE_CHECKING`、`runtime_checkable`、`get_args`、`get_origin`、`get_type_hints`。其他名字
  视为可疑。

## PR 前检查

```bash
uv run ruff check          # lint
uv run ruff format --check # 格式（用 `uv run ruff format` 自动修复）
uv run pyright             # src/genelab 的 strict 类型检查
uv run pytest              # 完整测试套件
```

四项均须通过 —— CI 把同一组检查设为每个 PR 的必需 status checks。

## 分支与 PR 流程

`main` 受保护：直接推送会被拒绝，必需的 CI 检查（`lint`、`typecheck`、`test`）必须全绿后才能
合入 PR。

1. 从 `main` 派生分支，使用描述性前缀：`fix/...`、`feat/...`、`ci/...`、`chore/...`、`docs/...`。
2. 本地跑上述检查。
3. 推送并开 PR —— CI 会在每次 push 时自动运行。
4. 评审期间若 `main` 前进，把 `main` 合入分支（或 rebase），让 PR 保持最新。

## 提交信息

- 祈使语气、首字母大写："Bump…"、"Fix…"、"Add…"。不要前缀、不要 emoji。
- 主题行控制在约 72 字符内。正文解释 *why* —— *what* 已经被 diff 表达。
