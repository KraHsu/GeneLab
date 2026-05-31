# Contributing to GeneLab

## Development setup

GeneLab uses [uv](https://github.com/astral-sh/uv) and requires Python `>=3.12`. Pick exactly one `torch-*` extra — they are mutually exclusive:

```bash
uv sync --extra torch-cpu     # or: torch-cu126 / torch-cu128 / torch-cu130
genelab --help         # smoke-test the CLI
genelab cache          # create project-local .cache dirs for Genesis / Matplotlib
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
- **`mkdocs build --strict` must pass.** Run `mkdocs build --strict` before opening a doc-touching PR — mkdocs is included by default in `uv sync`. The flag fails the build on any unresolved relative link or anchor.

## Checks before opening a PR

```bash
ruff check          # lint
ruff format --check # formatting (run `ruff format` to fix)
pyright             # strict type-check on src/genelab
pytest              # full test suite
```

All four must pass — CI enforces the same set as required status checks on every PR.

## Branch and PR workflow

GeneLab uses a two-layer flow: feature branches integrate into a topical `dev/<topic>` branch first, then `dev/<topic>` is promoted to `main` via PR. Each coherent body of work — an epic, a release prep, a multi-PR feature group — lives on its own `dev/<topic>` integration branch cut from `main`. `main` is protected: direct pushes are rejected and the required CI checks (`lint`, `typecheck`, `test`) must be green before a PR can merge.

```
feat/* ──┐
fix/*  ──┼──> dev/<topic> ──PR──> main
docs/* ──┘     ▲                    ▲
               │                    │
   small work: commit directly      └── hotfix/*: urgent, skips dev/<topic>
```

1. **Starting a new topic.** Cut `dev/<topic>` off `main` with a short kebab-case topic name (e.g. `dev/bump-genesis-1.0`, `dev/sb3-backend`):

   ```bash
   git checkout main && git pull --ff-only
   git checkout -b dev/<topic>
   git push -u origin dev/<topic>
   ```

2. **Feature work.** For a change that warrants its own review PR, branch off the relevant `dev/<topic>` with a descriptive prefix: `fix/...`, `feat/...`, `ci/...`, `chore/...`, `docs/...`. Run the checks above locally. Push and open a PR targeting `dev/<topic>`. Small, self-contained work does not need its own feature branch — committing straight to `dev/<topic>` is fine, since `dev/<topic>` is itself the integration branch and review happens at the `dev/<topic>` → `main` promotion PR.
3. **Promoting to main.** When the topic is complete, open a PR `dev/<topic>` → `main`. CI runs on every push and must be green before merge.
4. **Mid-topic sync.** If `dev/<topic>` advances during review of a feature branch, merge it into your branch (or rebase) so the PR stays current. If `main` advances mid-topic (hotfixes), merge or rebase `main` into `dev/<topic>` so the eventual promotion PR stays clean.
5. **Hotfixes.** Small urgent fixes may branch directly off `main` as `hotfix/...` and PR to `main` without going through a `dev/<topic>` branch.

## Commit messages

- Imperative mood, capitalized first word: "Bump…", "Fix…", "Add…". No prefix, no emoji.
- Keep the subject under ~72 chars. Use the body to explain *why* — the diff already shows *what*.
