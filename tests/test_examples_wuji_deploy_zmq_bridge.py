"""ZMQ bridge for the deploy pipeline: message parsing + latest()-cache semantics.

The cube observer publishes orientation in scipy xyzw order; the deploy stack runs
on mujoco wxyz. These tests pin the conversion and the receiver's last-valid cache
(so a momentary loss of ``world_fixed`` keeps the last good pose).
"""

from typing import Any

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from genelab_wuji.deploy.zmq_bridge import (  # noqa: E402
    CubeReceiver,
    GoalReceiver,
    cube_msg_from_pose,
    cube_pose_from_msg,
    goal_from_msg,
)


def test_cube_pose_from_msg_converts_xyzw_to_wxyz() -> None:
    msg = {
        "world_fixed": True,
        "cube_size": 0.054,
        "cube1": {
            "position": {"x": 0.1, "y": -0.2, "z": 0.3},
            # scipy xyzw on the wire (w last)
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    pos, quat_wxyz, world_fixed, cube_size = cube_pose_from_msg(msg)

    assert np.allclose(pos, [0.1, -0.2, 0.3])
    assert np.allclose(quat_wxyz, [1.0, 0.0, 0.0, 0.0])  # w first
    assert world_fixed is True
    assert cube_size == pytest.approx(0.054)


def test_cube_pose_from_msg_preserves_quat_component_mapping() -> None:
    # Distinct components so a w<->x swap (or any mislabel) would be caught.
    msg = {
        "world_fixed": False,
        "cube1": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.4},
        },
    }
    _pos, quat_wxyz, world_fixed, cube_size = cube_pose_from_msg(msg)

    assert np.allclose(quat_wxyz, [0.4, 0.1, 0.2, 0.3])  # [w, x, y, z]
    assert world_fixed is False
    assert cube_size is None  # absent in this message


def _cube_msg(pos: list[float], quat_wxyz: list[float], world_fixed: bool) -> dict[str, Any]:
    w, x, y, z = quat_wxyz
    return {
        "world_fixed": world_fixed,
        "cube1": {
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "orientation": {"x": x, "y": y, "z": z, "w": w},
        },
    }


def test_cube_receiver_latest_defaults_to_zeros_identity_before_valid() -> None:
    # No socket: connect=False keeps it inert so we can drive it by hand.
    recv = CubeReceiver(connect=False)
    pos, quat = recv.latest()
    assert np.allclose(pos, np.zeros(3))
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_cube_receiver_caches_last_valid_through_world_unfixed() -> None:
    recv = CubeReceiver(connect=False)

    good_pos = [0.1, 0.2, 0.3]
    good_quat = [0.0, 1.0, 0.0, 0.0]  # wxyz
    recv._update_from_msg(_cube_msg(good_pos, good_quat, world_fixed=True))

    pos, quat = recv.latest()
    assert np.allclose(pos, good_pos)
    assert np.allclose(quat, good_quat)

    # A later sample arrives while calibration is momentarily lost: keep the cache.
    recv._update_from_msg(_cube_msg([9.0, 9.0, 9.0], [1.0, 0.0, 0.0, 0.0], world_fixed=False))
    pos2, quat2 = recv.latest()
    assert np.allclose(pos2, good_pos)
    assert np.allclose(quat2, good_quat)


def test_goal_from_msg_reads_wxyz_orientation() -> None:
    msg = {"goal": {"orientation": {"w": 0.4, "x": 0.1, "y": 0.2, "z": 0.3}}}
    assert np.allclose(goal_from_msg(msg), [0.4, 0.1, 0.2, 0.3])


def test_goal_receiver_defaults_to_identity_then_tracks_latest() -> None:
    recv = GoalReceiver(connect=False)
    assert np.allclose(recv.latest(), [1.0, 0.0, 0.0, 0.0])

    recv._update_from_msg({"goal": {"orientation": {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}}})
    assert np.allclose(recv.latest(), [0.0, 0.0, 1.0, 0.0])


def test_cube_msg_round_trips_through_parser() -> None:
    # The publisher serializes a wxyz pose to the wire (scipy xyzw); the parser
    # converts back to wxyz. Round-trip must be exact so observer and consumer agree.
    pos = np.array([0.12, -0.03, 0.55])
    quat_wxyz = np.array([0.4, 0.1, 0.2, 0.3])
    msg = cube_msg_from_pose(pos, quat_wxyz, world_fixed=True, cube_size=0.054)

    got_pos, got_quat, world_fixed, cube_size = cube_pose_from_msg(msg)
    assert np.allclose(got_pos, pos)
    assert np.allclose(got_quat, quat_wxyz)
    assert world_fixed is True
    assert cube_size == pytest.approx(0.054)
