"""Closed-loop deploy controller (hardware/sim-agnostic).

Wires the deploy pieces into one control step:

    read encoders -> build policy obs (cube/goal from the observer feed)
        -> ONNX policy -> EMA action -> write joint target to the hand

It depends only on small protocols (a hand driver, a cube source, a goal source,
a callable policy), so it runs headlessly with mocks in tests and with the real
hand + ZMQ + Genesis viewer in ``scripts/play_real.py``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from genelab_wuji.deploy.action import ActionProcessor
from genelab_wuji.deploy.config import N_JOINTS
from genelab_wuji.deploy.obs import DeployObsBuilder


class _Policy(Protocol):
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


class _CubeSource(Protocol):
    def latest(self) -> tuple[np.ndarray, np.ndarray]: ...


class _GoalSource(Protocol):
    def latest(self) -> np.ndarray: ...


class _Driver(Protocol):
    def home(self) -> None: ...
    def write_target(self, qpos: np.ndarray) -> None: ...
    def read_encoders(self) -> np.ndarray: ...


class DeployController:
    """Run the policy in closed loop against a hand driver and observer feeds."""

    def __init__(
        self,
        policy: _Policy,
        driver: _Driver,
        cube_source: _CubeSource,
        goal_source: _GoalSource,
        *,
        default_joint_pos: np.ndarray,
        control_dt: float = 0.05,
        action_scale: float = 0.5,
        ema_alpha: float = 0.5,
        warmup_steps: int = 8,
        joint_pos_limits: tuple[np.ndarray, np.ndarray] | None = None,
        enc_to_policy: np.ndarray | None = None,
    ) -> None:
        self.policy = policy
        self.driver = driver
        self.cube_source = cube_source
        self.goal_source = goal_source
        self.control_dt = control_dt
        # Joint-order remap between the driver (encoder/hardware order) and the policy
        # (Genesis articulation order). ``None`` = identity. ``default_joint_pos`` must be
        # in the SAME order the policy uses (policy order when a remap is given).
        self._enc_to_policy = None if enc_to_policy is None else np.asarray(enc_to_policy)
        self._policy_to_enc = (
            None if self._enc_to_policy is None else np.argsort(self._enc_to_policy)
        )
        self._default = np.asarray(default_joint_pos, dtype=float)
        self._obs = DeployObsBuilder(self._default)
        self._action_proc = ActionProcessor(
            self._default,
            action_scale=action_scale,
            ema_alpha=ema_alpha,
            warmup_steps=warmup_steps,
            joint_pos_limits=joint_pos_limits,
        )
        self._last_action = np.zeros(N_JOINTS)
        self._prev_joint_pos = self._default.copy()

    def _to_policy(self, v: np.ndarray) -> np.ndarray:
        """Reorder an encoder/hardware-order vector into policy order."""
        return v if self._enc_to_policy is None else v[self._enc_to_policy]

    def _to_hardware(self, v: np.ndarray) -> np.ndarray:
        """Reorder a policy-order vector into encoder/hardware order."""
        return v if self._policy_to_enc is None else v[self._policy_to_enc]

    def reset(self) -> None:
        """Home the hand and clear obs/action/velocity state."""
        self.driver.home()
        self._obs.reset()
        self._action_proc.reset()
        self._last_action = np.zeros(N_JOINTS)
        self._prev_joint_pos = self._to_policy(self.driver.read_encoders())

    def step(self) -> dict[str, Any]:
        """Run one control step; return ``{action, target, obs, joint_pos}``."""
        encoder = self.driver.read_encoders()  # hardware (encoder) order
        joint_pos = self._to_policy(encoder)  # policy order
        joint_vel = (joint_pos - self._prev_joint_pos) / self.control_dt
        self._prev_joint_pos = joint_pos

        cube_pos_tag, cube_quat_tag = self.cube_source.latest()
        goal_quat_tag = self.goal_source.latest()

        obs = self._obs.compute(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            cube_pos_tag=cube_pos_tag,
            cube_quat_tag=cube_quat_tag,
            goal_quat_tag=goal_quat_tag,
            last_action=self._last_action,
        )
        action = np.asarray(self.policy(obs), dtype=float)  # policy order
        target = self._action_proc.process(action)  # policy order
        self.driver.write_target(self._to_hardware(target))  # back to hardware order
        self._last_action = action
        # ``joint_pos`` returned in encoder/hardware order (for the viewer's name-based remap).
        return {"action": action, "target": target, "obs": obs, "joint_pos": encoder}
