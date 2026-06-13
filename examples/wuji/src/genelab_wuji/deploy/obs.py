"""Assemble the policy observation vector for deploy (pure numpy).

Reproduces the GeneLab training policy obs group so an exported ONNX policy
receives exactly what it was trained on. No forward kinematics is required: joint
state comes from the hand encoders, cube/goal poses arrive from the observer
already in the wrist-tag frame, and the last action is tracked by the action term.

Term order, per-term history, and the 6D goal-error encoding mirror
``genelab_wuji.reorient.mdp.observations`` and ``...mdp._math``.
"""

from __future__ import annotations

import numpy as np

from genelab_wuji.deploy.frame_transform import quat_apply, quat_conjugate, quat_mul


def _matrix_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix from a wxyz quaternion (columns are the rotated basis)."""
    return np.stack(
        [
            quat_apply(quat_wxyz, np.array([1.0, 0.0, 0.0])),
            quat_apply(quat_wxyz, np.array([0.0, 1.0, 0.0])),
            quat_apply(quat_wxyz, np.array([0.0, 0.0, 1.0])),
        ],
        axis=1,
    )


def goal_rot_err_6d(cube_quat_tag: np.ndarray, goal_quat_tag: np.ndarray) -> np.ndarray:
    """6D rotation error (first two matrix rows) of cube-to-goal, tag frame.

    Mirrors ``genelab_wuji.reorient.mdp.observations.goal_rot_err_6d``:
    ``matrix_to_rotation_6d(matrix_from_quat(cube_quat ∘ goal_quat*))``.
    """
    err_quat = quat_mul(cube_quat_tag, quat_conjugate(goal_quat_tag))
    rot = _matrix_from_quat(err_quat)
    return rot[:2, :].reshape(6)


class DeployObsBuilder:
    """Build the 207-dim policy obs with per-term 3-step history.

    Each term keeps a ``(history_len, dim)`` buffer, term-major oldest->newest.
    ``reset()`` clears the buffers; the first ``compute`` after a reset backfills
    every history slot with the current frame (matches the training CircularBuffer).
    """

    # (name, dim) in the order the GeneLab policy obs group concatenates them.
    _TERMS: tuple[tuple[str, int], ...] = (
        ("joint_pos_rel", 20),
        ("joint_vel_rel", 20),
        ("cube_pos_in_tag", 3),
        ("goal_rot_err_6d", 6),
        ("last_action", 20),
    )

    def __init__(self, default_joint_pos: np.ndarray, history_len: int = 3) -> None:
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=float)
        self.history_len = history_len
        self._buffers: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        """Clear history so the next ``compute`` backfills."""
        self._buffers = {}

    def _push(self, name: str, value: np.ndarray) -> np.ndarray:
        """Append ``value`` to the term buffer (backfill on first frame); flatten."""
        value = np.asarray(value, dtype=float)
        buf = self._buffers.get(name)
        if buf is None or buf.shape != (self.history_len, value.shape[0]):
            buf = np.repeat(value[None, :], self.history_len, axis=0)
        else:
            buf = np.concatenate([buf[1:], value[None, :]], axis=0)
        self._buffers[name] = buf
        return buf.reshape(-1)

    def compute(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        cube_pos_tag: np.ndarray,
        cube_quat_tag: np.ndarray,
        goal_quat_tag: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        """Return the flat policy obs vector for this control step."""
        frame = {
            "joint_pos_rel": np.asarray(joint_pos, dtype=float) - self.default_joint_pos,
            "joint_vel_rel": np.asarray(joint_vel, dtype=float),
            "cube_pos_in_tag": np.asarray(cube_pos_tag, dtype=float),
            "goal_rot_err_6d": goal_rot_err_6d(cube_quat_tag, goal_quat_tag),
            "last_action": np.asarray(last_action, dtype=float),
        }
        blocks = [self._push(name, frame[name]) for name, _dim in self._TERMS]
        return np.concatenate(blocks).astype(np.float32)
