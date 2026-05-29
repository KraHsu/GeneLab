"""Shared fakes for the new Genesis-1.0 sensor wrappers.

These mirror the ``_FakeGsScene`` / ``_FakeGsLink`` pattern from ``test_sensor.py``
but cover ``gs_scene.add_sensor(opts) -> handle`` rather than ``add_camera``. The
recorded options + the swappable ``handle.read()`` return value give per-sensor tests a
deterministic surface to assert against without spinning up a real ``gs.Scene``.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class FakeGsHandle:
    """Stand-in for a Genesis entity handle. The wrapper only reads ``.idx``."""

    idx: int = 0


@dataclass
class FakeArticulation:
    """Stand-in for a GeneLab ``Articulation`` wrapper at pre-build time."""

    link_names: list[str]
    gs_handle: FakeGsHandle = field(default_factory=FakeGsHandle)


class FakeGsSensor:
    """Stand-in for the Genesis sensor handle returned by ``add_sensor``.

    ``read(envs_idx=None)`` returns whatever the test installed via
    ``set_return``; defaults to an empty tensor so a wrapper that never primes a
    return raises a shape mismatch rather than silently passing.
    """

    def __init__(self, opts: Any) -> None:
        self.opts = opts
        self._ret: torch.Tensor = torch.empty(0)
        self.read_calls = 0

    def set_return(self, tensor: torch.Tensor) -> None:
        self._ret = tensor

    def read(self, envs_idx: Any = None) -> torch.Tensor:
        del envs_idx
        self.read_calls += 1
        return self._ret


class FakeGsScene:
    """Stand-in for ``gs.Scene``; records ``add_sensor`` options and hands out fake handles."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.sensors: list[FakeGsSensor] = []

    def add_sensor(self, opts: Any) -> FakeGsSensor:
        handle = FakeGsSensor(opts)
        self.sensors.append(handle)
        return handle
