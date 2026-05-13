# Contributing to GeneLab

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
uv run ruff check    # lint
uv run pyright       # strict type-check on src/genelab
uv run pytest        # full test suite
```

All three must pass.
