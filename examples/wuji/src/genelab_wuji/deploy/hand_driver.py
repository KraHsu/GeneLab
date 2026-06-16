"""Hand-driver abstraction: hardware-agnostic interface + mock + real (wujihandpy).

The control loop depends only on ``HandDriverBase``. ``MockHandDriver`` echoes
written targets so the full pipeline runs and tests headlessly; ``WujiHandDriver``
talks to the real hand and is imported lazily so the dependency is optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from genelab_wuji.deploy.config import JOINT_NAMES_20, N_JOINTS, default_joint_pos


def _home_ramp(current: np.ndarray, target: np.ndarray, steps: int) -> np.ndarray:
    """Ease-in-out (smoothstep) interpolation from ``current`` to ``target``.

    Returns ``(steps, 20)`` intermediate targets; the last row equals ``target``
    exactly (smoothstep ``3t²-2t³`` reaches 1 at ``t=1``). Pure/numpy so the ramp
    math is testable without hardware.
    """
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    steps = max(1, int(steps))
    t = (np.arange(1, steps + 1, dtype=float) / steps)[:, None]  # (steps, 1), ends at 1.0
    t_smooth = t * t * (3.0 - 2.0 * t)
    return current[None, :] + t_smooth * (target - current)[None, :]


class HandDriverBase(ABC):
    """Interface every hand backend implements (targets/encoders flattened to 20)."""

    @abstractmethod
    def home(self, duration_s: float = 3.0) -> None:
        """Drive the hand to the home grasp keyframe (ease-in-out ramp over ``duration_s``)."""

    @abstractmethod
    def write_target(self, qpos: np.ndarray) -> None:
        """Command a ``(20,)`` joint position target (JOINT_NAMES_20 order)."""

    @abstractmethod
    def read_encoders(self) -> np.ndarray:
        """Read the actual ``(20,)`` joint positions (JOINT_NAMES_20 order)."""

    def joint_names_in_encoder_order(self) -> tuple[str, ...]:
        """Joint names matching ``read_encoders`` / ``write_target`` indexing."""
        return JOINT_NAMES_20


class MockHandDriver(HandDriverBase):
    """In-memory hand: ``read_encoders`` echoes the last ``write_target``.

    Starts at the home grasp pose so a fresh driver reads a sensible state.
    """

    def __init__(self) -> None:
        self._state = default_joint_pos()

    def home(self, duration_s: float = 3.0) -> None:
        # No hardware to ease; the ramp is a real-driver safety concern only.
        self._state = default_joint_pos()

    def write_target(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (N_JOINTS,):
            raise ValueError(f"qpos shape {qpos.shape}, expected ({N_JOINTS},)")
        self._state = qpos.copy()

    def read_encoders(self) -> np.ndarray:
        return self._state.copy()


class WujiHandDriver(HandDriverBase):
    """Real Wuji hand via ``wujihandpy`` (imported lazily; untested in CI).

    The hardware exposes a (5, 4) array (5 fingers x 4 joints); we flatten to
    (20,) at the boundary, which matches ``JOINT_NAMES_20`` row-major order.
    Use as a context manager so joints are enabled on enter / disabled on exit.
    """

    def __init__(self, effort_limit_nm: float = 0.5) -> None:
        import wujihandpy  # noqa: F401  (fail loudly if the dep is missing)

        self._wujihandpy = wujihandpy
        self.effort_limit_nm = effort_limit_nm
        self._hand: Any = None

    def __enter__(self) -> "WujiHandDriver":
        self._hand = self._wujihandpy.Hand()
        self._hand.write_joint_effort_limit(self.effort_limit_nm)
        self._hand.write_joint_enabled(True)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._hand is not None:
            self._hand.write_joint_enabled(False)
            self._hand = None

    def home(self, duration_s: float = 3.0) -> None:
        """Smoothly ramp from the current pose to the home grasp keyframe.

        Ease-in-out interpolation at 50 Hz over ``duration_s`` so the hand eases
        in rather than snapping (a single instant write can jerk the joints).
        ``duration_s <= 0`` does one immediate write.
        """
        import time

        target = default_joint_pos()
        if duration_s <= 0:
            self.write_target(target)
            return
        steps = max(1, int(duration_s * 50.0))  # 50 Hz smoothing
        dt = duration_s / steps
        for frame in _home_ramp(self.read_encoders(), target, steps):
            self.write_target(frame)
            time.sleep(dt)

    def write_target(self, qpos: np.ndarray) -> None:
        assert self._hand is not None, "enter the WujiHandDriver context first"
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (N_JOINTS,):
            raise ValueError(f"qpos shape {qpos.shape}, expected ({N_JOINTS},)")
        self._hand.write_joint_target_position(qpos.reshape(5, 4))

    def read_encoders(self) -> np.ndarray:
        assert self._hand is not None, "enter the WujiHandDriver context first"
        return np.asarray(self._hand.read_joint_actual_position(), dtype=float).reshape(N_JOINTS)
