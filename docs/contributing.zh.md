# 贡献指南

详细的贡献者指南以英文为权威版本，见仓库根目录
[`CONTRIBUTING.md`](https://github.com/KraHsu/GeneLab/blob/main/CONTRIBUTING.md)。

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

## 文档约定

文档位于 `docs/`，由 MkDocs Material 与 `mkdocs-static-i18n` 插件构建。每个内容页都有
`.en.md` / `.zh.md` 双语版本；站点在 `/` 提供英文、`/zh/` 提供中文。下列规则同时适用于两种
语言：

- **双语平行强制要求。** `.en.md` 修改时，同次提交需修改对应的 `.zh.md`。两文件需在标题数与
  顺序、代码块（变量占位符除外）、表格结构、admonition、`See also` 条目上保持一致。
- **章节标题用名词或动名词短语**，不要祈使语气。例如 `Scaffold output` / `生成的目录结构`、
  `Running a task` / `运行任务` —— 而不是 `Play a task` / `运行一个任务`。结尾块统一叫
  `## See also`，不要 `## Next steps`。
- **不要写"本页介绍……"式开场白。** 避免在第一段铺垫"本页覆盖……"、`This walks through…`、
  `In this guide we'll…`。直接进入实质内容。
- **避免第二人称。** 去掉 `you` / `你` / `您`。改用省略主语（操作命令用祈使语气）或名词短语
  （`The CLI exposes…` / `CLI 暴露…`）。
- **正文不要插中段跨页跳转链接。** 不在正文段落里写 `see [Foo](...)` / `详见 [Foo](...)`。
  把相关页面收在结尾的 `## See also` 块里，**≤ 3 条**，只放补充阅读（必要的下一步页面由
  左侧导航承担）。
- **callout 用 admonition，不用 blockquote。** `!!! warning "Title"` / `!!! tip "Title"` /
  `!!! note "Title"`。`>` 只用于真正的引文。
- **稳定显式锚点。** 含非 ASCII 字符、数字前缀或可能改名的标题，追加 `{ #stable-id }`。例如：
  `## 5. Advanced: end-to-end RL on Unitree G1 { #unitree-g1 }`。跨链时使用该 slug，不要使用自动
  生成的 slug。
- **`.zh.md` 中的中英混排空格。** 中文字符与相邻的 ASCII 单词、数字或行内代码之间留一个空格
  （`运行 \`uv sync\``，不是 `运行\`uv sync\``）。
- **`mkdocs build --strict` 必须通过。** 用 `uv sync --extra docs` 安装 docs extra，
  在提交涉及文档的 PR 前运行 `uv run mkdocs build --strict`。strict 模式会因任何未解析的相对链接
  或锚点失败。

## PR 前检查

```bash
uv run ruff check          # lint
uv run ruff format --check # 格式（用 `uv run ruff format` 自动修复）
uv run pyright             # src/genelab 的 strict 类型检查
uv run pytest              # 完整测试套件
```

四项均须通过 —— CI 把同一组检查设为每个 PR 的必需 status checks。

## 分支与 PR 流程

GeneLab 采用两层流程：feature 分支先汇入 `dev`，再由 `dev` 通过 PR 提升到 `main`。`main`
受保护 —— 直接推送会被拒绝，必需的 CI 检查（`lint`、`typecheck`、`test`）必须全绿后才能
合入 PR。

```
feat/* ──┐
fix/*  ──┼──> dev ──PR──> main
docs/* ──┘
```

1. 从 `dev` 派生分支，使用描述性前缀：`fix/...`、`feat/...`、`ci/...`、`chore/...`、`docs/...`。
2. 本地跑上述检查。
3. 推送后选择开 PR 合入 `dev`，或直接本地合入 `dev` 后推送 `dev` —— 视改动大小而定。
4. 把 `dev` 提升到 `main` 时，开 `dev` → `main` 的 PR。CI 在每次 push 时自动运行，全绿后才能合入。
5. 评审期间若 `dev` 前进，把 `dev` 合入分支（或 rebase），让 PR 保持最新。

## 提交信息

- 祈使语气、首字母大写："Bump…"、"Fix…"、"Add…"。不要前缀、不要 emoji。
- 主题行控制在约 72 字符内。正文解释 *why* —— *what* 已经被 diff 表达。
