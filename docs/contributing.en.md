# Contributing

This page is for contributors changing GeneLab itself. Downstream robot projects should usually use
an extension package instead.

## Development setup

```bash
uv sync --extra torch-cpu
genelab cache
pytest
ruff check
pyright
```

Pick one CUDA extra instead of `torch-cpu` when developing GPU workflows.

## Documentation rules

The docs follow Diátaxis:

| Type | Purpose |
|---|---|
| Tutorial | Learning path from zero to a working result. |
| How-to | Task-focused instructions for experienced users. |
| Reference | Facts, flags, APIs, and defaults. |
| Explanation | Architecture and design rationale. |

Keep pages in one type where possible and cross-link rather than mixing long conceptual digressions
into task guides.

## Code style

- Use Python 3.12+ syntax.
- Keep registry-time imports light; avoid starting Genesis at import time.
- Prefer dataclass configs for user-visible knobs.
- Keep task/example code outside `src/genelab/` unless it is part of the framework.

## Before a PR

```bash
pytest
ruff check
pyright
mkdocs build --strict
```

For docs-only changes, still run the MkDocs strict build.
