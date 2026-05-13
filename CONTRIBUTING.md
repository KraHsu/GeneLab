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

## Checks before opening a PR

```bash
uv run ruff check          # lint
uv run ruff format --check # formatting (run `uv run ruff format` to fix)
uv run pyright             # strict type-check on src/genelab
uv run pytest              # full test suite
```

All four must pass — CI enforces the same set as required status checks on every PR.

## Branch and PR workflow

`main` is protected: direct pushes are rejected and the required CI checks (`lint`, `typecheck`, `test`) must be green before a PR can merge.

1. Branch off `main` with a descriptive prefix: `fix/...`, `feat/...`, `ci/...`, `chore/...`, `docs/...`.
2. Run the checks above locally.
3. Push and open a PR — CI runs automatically on every push.
4. If `main` advances during review, merge it into your branch (or rebase) so the PR stays current.

## Commit messages

- Imperative mood, capitalized first word: "Bump…", "Fix…", "Add…". No prefix, no emoji.
- Keep the subject under ~72 chars. Use the body to explain *why* — the diff already shows *what*.
