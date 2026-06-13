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


class HandDriverBase(ABC):
    """Interface every hand backend implements (targets/encoders flattened to 20)."""

    @abstractmethod
    def home(self) -> None:
        """Drive the hand to the home grasp keyframe."""

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

    def home(self) -> None:
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

    def home(self) -> None:
        self.write_target(default_joint_pos())

    def write_target(self, qpos: np.ndarray) -> None:
        assert self._hand is not None, "enter the WujiHandDriver context first"
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (N_JOINTS,):
            raise ValueError(f"qpos shape {qpos.shape}, expected ({N_JOINTS},)")
        self._hand.write_joint_target_position(qpos.reshape(5, 4))

    def read_encoders(self) -> np.ndarray:
        assert self._hand is not None, "enter the WujiHandDriver context first"
        return np.asarray(self._hand.read_joint_actual_position(), dtype=float).reshape(N_JOINTS)
