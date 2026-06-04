"""Regression guard for the ``rl.runner ↔ rl.backends`` cycle break.

Before the refactor, ``rl/runner.py`` defined nine helpers (``build_bridges``,
``build_env``, ``close_bridges``, ``make_random_policy``, ``make_zero_policy``,
``resolve_env_cfg``, ``resolve_log_dir``, ``run_play_loop``, ``save_run_params``)
that every backend imported back, forming the static cycle:

    rl.runner ── imports ──► rl.backends.__init__
    rl.backends.sb3 ── imports ──► rl.runner

The cycle was held together only by ``rl/backends/__init__.py:_ensure_loaded``
deferring concrete backend imports to first ``select_backend()`` call. Any
contributor who moved a backend ``import`` to module top level (e.g. for
type-checking convenience) would have broken the package at import time.

After the refactor the helpers live in ``rl/_helpers.py`` and the backends import from
there. ``rl/runner.py`` re-exports the same names so external callers using
``from genelab.rl.runner import build_env, …`` keep working.

An earlier design described this test as "imports
``genelab.rl.backends.sb3`` in a subprocess and asserts ``genelab.rl.runner``
is not in ``sys.modules`` afterwards." That spec turned out to be flawed:
``rl/__init__.py`` imports ``rl.runner`` for its public ``play_task`` /
``train_task`` re-exports, so Python pulls ``rl.runner`` into ``sys.modules``
during parent-package init *before* the backend module's own import runs. The
``sys.modules`` check can never detect the cycle for ``rl.backends.*``
modules.

We instead test the *static* import graph via ``grimp`` (a transitive
dependency of ``import-linter``, which the project already pins). The
invariant we want is "no top-level ``from genelab.rl.runner import …`` or
``import genelab.rl.runner`` statement in any ``rl/backends/*.py``" — exactly
what ``grimp.find_modules_directly_imported_by`` reports. This complements
the equivalent ``[tool.importlinter]`` contract with a runtime test
that fails the suite, not just a non-blocking CI lint.
"""

from __future__ import annotations

import pytest

BACKEND_MODULES: tuple[str, ...] = (
    "genelab.rl.backends.rsl_rl",
    "genelab.rl.backends.skrl",
    "genelab.rl.backends.sb3",
)


@pytest.fixture(scope="module")
def genelab_import_graph():
    """Cached grimp graph of the ``genelab`` package."""
    grimp = pytest.importorskip("grimp")
    return grimp.build_graph("genelab")


@pytest.mark.parametrize("backend", BACKEND_MODULES)
def test_backend_does_not_directly_import_runner(backend: str, genelab_import_graph) -> None:
    """``{backend}`` must not have a top-level dependency on ``genelab.rl.runner``."""
    direct = genelab_import_graph.find_modules_directly_imported_by(backend)
    assert "genelab.rl.runner" not in direct, (
        f"{backend} directly imports genelab.rl.runner — the cycle the "
        f"refactor broke has been reintroduced.\n\nBackends must import "
        f"helpers from genelab.rl._helpers.\n"
        f"Direct imports of {backend}: {sorted(direct)}"
    )
