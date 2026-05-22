"""Deprecation shims for the relocated vecenv adapters (ADR-0007 / ROADMAP §9 R6).

The three env adapters moved to ``genelab.rl.vecenvs.<lib>``; the old
``genelab.rl.<lib>_wrapper`` paths and the top-level ``genelab.rl.RslRlVecEnvWrapper``
re-export remain for one release as ``DeprecationWarning``-emitting shims. These
tests assert the old paths still resolve to the moved classes *and* warn.

Each legacy module shim is checked in a **subprocess** (like ``test_optional_deps.py``):
importing the SB3 adapter pulls ``cv2``, which forces the ``xcb`` Qt plugin and
SIGABRTs Genesis's PyQt plotter tests if it happens in the shared pytest process —
see ``cv2-qt-plotter-conflict.md`` in project memory. A subprocess isolates that.
The rsl_rl-only checks (``__getattr__`` re-export, ``__all__``) are cv2-free and run
in-process.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_LEGACY_MODULES = [
    ("genelab.rl.rsl_rl_wrapper", "genelab.rl.vecenvs.rsl_rl", "RslRlVecEnvWrapper"),
    ("genelab.rl.sb3_wrapper", "genelab.rl.vecenvs.sb3", "GenelabSb3VecEnv"),
    ("genelab.rl.skrl_wrapper", "genelab.rl.vecenvs.skrl", "GenelabSkrlWrapper"),
]

_CHECK = """\
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import {legacy} as legacy
    import {new} as new

assert legacy.{attr} is new.{attr}, "legacy shim re-exports a different object"
assert any(
    issubclass(w.category, DeprecationWarning) and {legacy!r} in str(w.message)
    for w in caught
), f"no DeprecationWarning naming {legacy!r}; got {{[str(w.message) for w in caught]}}"
"""


@pytest.mark.parametrize("legacy_path, new_path, attr", _LEGACY_MODULES)
def test_legacy_wrapper_module_warns_and_reexports(
    legacy_path: str, new_path: str, attr: str
) -> None:
    """The old ``genelab.rl.<lib>_wrapper`` path re-exports the moved class and warns."""
    code = _CHECK.format(legacy=legacy_path, new=new_path, attr=attr)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"shim check for {legacy_path} failed:\n{proc.stderr}"


def test_legacy_rl_package_reexport_warns() -> None:
    # rsl_rl adapter is cv2-free, so this is safe in-process.
    import genelab.rl as rl
    from genelab.rl.vecenvs.rsl_rl import RslRlVecEnvWrapper

    with pytest.warns(DeprecationWarning, match="RslRlVecEnvWrapper"):
        resolved = rl.RslRlVecEnvWrapper
    assert resolved is RslRlVecEnvWrapper


def test_rsl_wrapper_dropped_from_rl_all() -> None:
    import genelab.rl as rl

    # Removed from the public ``__all__`` (the deprecation signal); still reachable
    # via the ``__getattr__`` shim above.
    assert "RslRlVecEnvWrapper" not in rl.__all__


def test_legacy_rl_distributed_module_warns_and_reexports() -> None:
    # The torchrun helpers moved to genelab.utils.distributed (R7.3b). The shim has
    # no cv2/Qt dependency, so it is safe in-process; drop any cached copy so the
    # module-level warning re-fires.
    import importlib
    import sys

    sys.modules.pop("genelab.rl.distributed", None)
    with pytest.warns(DeprecationWarning, match="genelab.rl.distributed"):
        legacy = importlib.import_module("genelab.rl.distributed")
    from genelab.utils import distributed as new

    assert legacy.pin_cuda_device is new.pin_cuda_device
    assert legacy.is_main_process is new.is_main_process
