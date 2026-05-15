"""Shared pytest fixtures for the GeneLab suite.

The notable one is :func:`genesis_runtime`: Genesis initializes EGL/OpenGL even when
``show_viewer=False`` (its rigid solver shares state with the viewer), so plain
``ubuntu-latest`` CI runners without ``libEGL`` raise ``AttributeError`` from
PyOpenGL the first time ``gs.Scene(...).build()`` is called. Tests that genuinely
need a Genesis runtime depend on this fixture so they skip cleanly on headless CI
instead of hard-failing.
"""

import ctypes.util
from typing import Any

import pytest


def _has_egl() -> bool:
    """Return True when ``libEGL`` is discoverable on this machine."""
    return ctypes.util.find_library("EGL") is not None


@pytest.fixture(scope="session")
def genesis_runtime() -> Any:
    """Initialized Genesis module, or skip when the renderer is unavailable."""
    if not _has_egl():
        pytest.skip("libEGL not available on this host (typical for headless CI)")
    gs = pytest.importorskip("genesis")
    if not getattr(gs, "_initialized", False):
        try:
            gs.init(backend=gs.cpu, logging_level="warning")
        except Exception as exc:  # noqa: BLE001 - any backend init failure means skip
            pytest.skip(f"Genesis init failed: {exc}")
    return gs
