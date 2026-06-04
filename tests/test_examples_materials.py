"""Smoke tests for the ``genelab_examples.materials_demo`` reference scenes."""

from typing import Any

import pytest

pytest.importorskip("torch")

from genelab.configs import SimulationCfg  # noqa: E402
from genelab.scene import InteractiveScene  # noqa: E402
from genelab_examples.materials_demo import (  # noqa: E402
    deformable_scene_cfg,
    surfaced_scene_cfg,
)


class _Recorder:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs


class _Node:
    def __init__(self, path: str = "") -> None:
        self._path = path

    def __getattr__(self, name: str) -> "_Node":
        return _Node(f"{self._path}.{name}" if self._path else name)

    def __call__(self, **kwargs: Any) -> _Recorder:
        return _Recorder(self._path, **kwargs)


class _FakeGs:
    def __init__(self) -> None:
        self.options = _Node()


def _solver_kwargs(scene_cfg: Any) -> dict[str, Any]:
    scene = InteractiveScene(SimulationCfg(), scene_cfg, device_hint="cpu")
    out: dict[str, Any] = {}
    scene._add_solver_options(_FakeGs(), out)
    return out


def test_deformable_demo_enables_mpm_with_tuned_bounds() -> None:
    out = _solver_kwargs(deformable_scene_cfg())
    assert "mpm_options" in out
    assert out["mpm_options"].kwargs["lower_bound"] == (-1.0, -1.0, 0.0)


def test_surfaced_demo_stays_rigid_only() -> None:
    assert _solver_kwargs(surfaced_scene_cfg()) == {}
