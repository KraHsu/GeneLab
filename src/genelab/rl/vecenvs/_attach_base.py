"""Shared "subclass the optional library's base if installed" helper.

Each of ``vecenvs/rsl_rl.py``, ``vecenvs/skrl.py``, and ``vecenvs/sb3.py``
declares a vec-env adapter (``RslRlVecEnvWrapper`` / ``GenelabSkrlWrapper`` /
``GenelabSb3VecEnv``) that should *additionally* subclass the matching
upstream library's vec-env base class **only when that optional library is
installed**. The three former helpers (``_attach_rsl_rl_base`` /
``_attach_skrl_base`` / ``_attach_sb3_base``) implemented the same recipe
verbatim; they are collapsed into the single :func:`attach_optional_base` below.

The optional-library import is deferred through ``importlib`` so this module
stays importable regardless of which RL backends are installed (preserves
CLAUDE.md invariant #1 — verified by ``tests/test_optional_deps.py``).

The helper rebinds the wrapper class in the caller's module namespace via the
``caller_globals`` argument. Each wrapper module calls it like so::

    attach_optional_base(
        base_module="rsl_rl.env",
        base_attr="VecEnv",
        wrapper_name="RslRlVecEnvWrapper",
        caller_globals=globals(),
    )

Lives under ``rl/vecenvs/`` next to its three callers (relocated here from the
flat ``rl/_attach_base.py``).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["attach_optional_base"]


def attach_optional_base(
    *,
    base_module: str,
    base_attr: str,
    wrapper_name: str,
    caller_globals: dict[str, Any],
) -> None:
    """Re-bind ``caller_globals[wrapper_name]`` to a subclass of the optional base.

    When ``base_module`` is not importable (the optional library is not
    installed), this is a no-op — the wrapper stays a bare ``object``
    subclass and the backend that actually needs the library will fail later
    with a more specific error.
    """
    try:
        mod = importlib.import_module(base_module)
    except ImportError:
        return
    base = getattr(mod, base_attr, None)
    if base is None:
        return

    original: type[Any] = caller_globals[wrapper_name]

    class _Combined(original, base):  # type: ignore[misc, valid-type]
        pass

    _Combined.__name__ = wrapper_name
    _Combined.__qualname__ = wrapper_name
    caller_globals[wrapper_name] = _Combined
