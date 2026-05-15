# Contributing to GeneLab

## Development setup

GeneLab uses [uv](https://github.com/astral-sh/uv) and requires Python `>=3.12`. Pick exactly one `torch-*` extra — they are mutually exclusive:

```bash
uv sync --extra torch-cpu     # or: torch-cu126 / torch-cu128 / torch-cu130
uv run genelab --help         # smoke-test the CLI
uv run genelab cache          # create project-local .cache dirs for Genesis / Matplotlib
```

The example extension under `examples/genelab_examples/` is on `pytest`'s `pythonpath`, so the test suite can import from it without installation.

## Python 3.12+ style

GeneLab targets Python `>=3.12`. Write modern code only; do not add compatibility shims for older interpreters.

- **No `from __future__ import annotations`.** Annotations are evaluated eagerly. When a name is imported only under `if TYPE_CHECKING:`, quote forward references at the call site, e.g. `def f(env: "ManagerBasedRlEnv") -> ...`.
- **PEP 604 unions.** Use `X | Y` and `X | None`. Never `Optional[X]` or `Union[X, Y]`.
- **PEP 585 builtin generics.** Use `list[T]`, `dict[K, V]`, `tuple[T, ...]`, `type[T]`. Never `typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Type`.
- **PEP 695 generics.** Use `class Foo[T]:`, `def f[T](...)`, and `type Alias = ...`. Never `TypeVar` + `Generic[T]`, never `TypeAlias`.
- **`collections.abc` over `typing`** for `Callable`, `Iterable`, `Iterator`, `Sequence`, `Mapping`, etc.
- **Whitelisted `typing` imports.** Only `Any`, `cast`, `Protocol`, `Literal`, `Final`, `Annotated`, `TYPE_CHECKING`, `runtime_checkable`, `get_args`, `get_origin`, `get_type_hints` are expected to appear. Anything else is suspect.

## Documentation conventions

Docs live under `docs/` and are built by MkDocs Material with the `mkdocs-static-i18n` plugin. Every content page exists in two languages with the `.en.md` / `.zh.md` suffix; the rendered site serves English at `/` and 中文 at `/zh/`. The following rules apply to both languages.

- **Bilingual parity is mandatory.** When a `.en.md` page changes, its `.zh.md` counterpart must change in the same commit. The two files must agree on heading count and order, code blocks (variable placeholders aside), table shape, admonitions, and `See also` entries.
- **Section headings are noun or gerund phrases**, never imperative directives. Use `Scaffold output` / `生成的目录结构`, `Running a task` / `运行任务` — not `Play a task` / `运行一个任务`. Same rule for `## See also` (not `## Next steps`).
- **No chatty openers.** Avoid first-paragraph meta-narration about what the page is about (`This walks through…`, `本节走通…`, `In this guide we'll…`). Open with the substantive statement.
- **Avoid second-person.** Drop `you` / `你` / `您`. Prefer no subject (imperative steps for operational commands) or noun-based phrasing (`The CLI exposes…` / `CLI 暴露…`).
- **No mid-paragraph cross-page jumps.** Do not insert `see [Foo](...)` / `详见 [Foo](...)` inside body text. Collect related-page pointers in a single `## See also` block at the end of the page, capped at **≤ 3 entries** indexing genuinely supplementary reading (not the next required step — that is the left-hand nav's job).
- **Admonitions over blockquotes.** Use `!!! warning "Title"` / `!!! tip "Title"` / `!!! note "Title"` for callouts; reserve `>` blockquotes for actual quotations.
- **Stable explicit anchors** for headings that contain non-ASCII characters, numbered prefixes, or wording likely to change. Append `{ #stable-id }` to the heading, e.g. `## 5. Advanced: end-to-end RL on Unitree G1 { #unitree-g1 }`. Cross-link to the slug, not the auto-generated one.
- **CJK + ASCII spacing in `.zh.md`.** Leave one space between Chinese characters and adjacent ASCII words, numbers, or inline code (`运行 \`uv sync\``, not `运行\`uv sync\``).
- **`mkdocs build --strict` must pass.** Install the docs extra with `uv sync --extra docs` and run `uv run mkdocs build --strict` before opening a doc-touching PR. The flag fails the build on any unresolved relative link or anchor.

## Checks before opening a PR

```bash
uv run ruff check          # lint
uv run ruff format --check # formatting (run `uv run ruff format` to fix)
uv run pyright             # strict type-check on src/genelab
uv run pytest              # full test suite
```

All four must pass — CI enforces the same set as required status checks on every PR.

## Branch and PR workflow

GeneLab uses a two-layer flow: feature branches integrate into `dev` first, then `dev` is promoted to `main` via PR. `main` is protected — direct pushes are rejected and the required CI checks (`lint`, `typecheck`, `test`) must be green before a PR can merge.

```
feat/* ──┐
fix/*  ──┼──> dev ──PR──> main
docs/* ──┘
```

1. Branch off `dev` with a descriptive prefix: `fix/...`, `feat/...`, `ci/...`, `chore/...`, `docs/...`.
2. Run the checks above locally.
3. Push and either open a PR targeting `dev`, or merge into `dev` locally and push `dev` directly — pick whichever fits the size of the change.
4. To promote `dev` to `main`, open a PR `dev` → `main`. CI runs on every push and must be green before merge.
5. If `dev` advances during review of a feature branch, merge `dev` into your branch (or rebase) so the PR stays current.

## Commit messages

- Imperative mood, capitalized first word: "Bump…", "Fix…", "Add…". No prefix, no emoji.
- Keep the subject under ~72 chars. Use the body to explain *why* — the diff already shows *what*.
