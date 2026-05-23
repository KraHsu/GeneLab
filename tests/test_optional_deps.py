"""Optional-dep boundary test (ROADMAP §9 Phase R0.2).

Locks invariant #1 from ``CLAUDE.md``::

    import genelab.rl  # must succeed without any of:
                       #   rsl_rl, skrl, stable_baselines3, tensordict
                       # Library imports are function-local inside each backend.

Each test spawns a fresh subprocess, poisons the four optional RL
libraries by setting ``sys.modules[name] = None`` (which makes any
subsequent ``import name`` raise ``ImportError`` per Python's
documented behaviour for sentinel ``None`` entries), then attempts to
import one target module.  A non-zero exit code means the target leaks
one of the optional libraries at module-load time and must move the
offending import into a function body before R7 can flip
``import-linter`` from lint-only to blocking.

The four targets are the load-bearing entry points: ``genelab.rl``
itself (re-exports the configs, the backend registry, the runner) plus
each concrete backend module under ``genelab.rl.backends/``.  None of
them should pull ``rsl_rl`` / ``skrl`` / ``stable_baselines3`` /
``tensordict`` at module level; library-specific code lives behind
function-local imports inside the backend methods (``train`` /
``play`` / ``make_inference_setup`` and helpers).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Optional RL libraries that ``import genelab.rl`` must not pull
# transitively.  See ``CLAUDE.md`` §"Architectural invariants" #1.
OPTIONAL_LIBS: tuple[str, ...] = (
    "rsl_rl",
    "skrl",
    "stable_baselines3",
    "tensordict",
)

# Modules whose load path must stay clear of the optional libraries.
# ``genelab.rl`` is the public surface end-users import; the three
# backend modules are loaded by ``genelab.rl.backends._ensure_loaded``
# on first ``select_backend()`` call and must themselves import without
# their RL library being installed (library-specific imports happen
# inside method bodies).
TARGETS: tuple[str, ...] = (
    "genelab.rl",
    "genelab.rl.backends.rsl_rl",
    "genelab.rl.backends.skrl",
    "genelab.rl.backends.sb3",
    # VecEnv adapters (ADR-0007 / R6): each must import without its RL library —
    # the "subclass the upstream base if installed" step defers via importlib.
    "genelab.rl.vecenvs.rsl_rl",
    "genelab.rl.vecenvs.skrl",
    "genelab.rl.vecenvs.sb3",
)

_WRAPPER = """\
import sys
for _lib in {libs!r}:
    sys.modules[_lib] = None
import {target}
"""


@pytest.mark.parametrize("target", TARGETS)
def test_import_with_optional_deps_poisoned(target: str) -> None:
    """``import {target}`` must succeed with the four optional RL libs absent."""
    code = _WRAPPER.format(libs=OPTIONAL_LIBS, target=target)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"`import {target}` failed when {list(OPTIONAL_LIBS)} are poisoned.\n"
        f"This means one of those libraries is imported at module-load time "
        f"somewhere on the import chain — find the offending top-level "
        f"`import` and move it into a function body.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
