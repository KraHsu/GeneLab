"""ZMQ pub/sub bridge between deploy processes (ported from wuji-mjlab).

Topology (localhost IPC):

* port 5555 — cube pose: ``cube_world_observer`` -> ``play_real`` / ``toreal_viewer``
* port 5556 — goal orientation: ``toreal_viewer`` -> ``play_real``

The observer publishes orientation in scipy **xyzw** order; everything downstream
runs in mujoco **wxyz**. ``cube_pose_from_msg`` / ``goal_from_msg`` own that
conversion as pure functions so it is testable without sockets; the receiver
classes wrap them with a background thread and a last-valid cache.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np

DEFAULT_CUBE_PORT: int = 5555
"""Default ZMQ port for cube pose (cube_world_observer)."""

DEFAULT_GOAL_PORT: int = 5556
"""Default ZMQ port for goal orientation (toreal_viewer)."""


def cube_pose_from_msg(
    msg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool, float | None]:
    """Parse a cube-pose message into ``(pos, quat_wxyz, world_fixed, cube_size)``.

    The wire orientation is scipy xyzw; the returned quaternion is mujoco wxyz.
    ``cube_size`` is ``None`` when the publisher does not announce it.
    """
    cube = msg["cube1"]
    p = cube["position"]
    q = cube["orientation"]  # scipy xyzw on the wire
    pos = np.array([p["x"], p["y"], p["z"]])
    quat_wxyz = np.array([q["w"], q["x"], q["y"], q["z"]])
    world_fixed = bool(msg.get("world_fixed", False))
    cube_size = msg.get("cube_size")
    return pos, quat_wxyz, world_fixed, (float(cube_size) if cube_size is not None else None)


def cube_msg_from_pose(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    world_fixed: bool,
    cube_size: float | None = None,
) -> dict[str, Any]:
    """Serialize a cube pose to the wire format (orientation as scipy xyzw).

    Inverse of ``cube_pose_from_msg``; used by the observer / any cube feeder.
    """
    msg: dict[str, Any] = {
        "world_fixed": bool(world_fixed),
        "cube1": {
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "orientation": {
                "x": float(quat_wxyz[1]),
                "y": float(quat_wxyz[2]),
                "z": float(quat_wxyz[3]),
                "w": float(quat_wxyz[0]),
            },
        },
    }
    if cube_size is not None:
        msg["cube_size"] = float(cube_size)
    return msg


def goal_from_msg(msg: dict[str, Any]) -> np.ndarray:
    """Parse a goal message into a wxyz quaternion (already wxyz on the wire)."""
    q = msg["goal"]["orientation"]
    return np.array([q["w"], q["x"], q["y"], q["z"]])


_RECEIVER_POLL_INTERVAL: float = 0.001
"""Polling interval for receiver background threads (seconds)."""


class CubeReceiver:
    """Subscribe to cube pose (port 5555) and expose the latest valid pose.

    The background thread parses each message via ``cube_pose_from_msg`` and feeds
    it to ``_update_from_msg``; ``latest()`` returns the last sample seen while
    ``world_fixed`` was True (so a momentary loss of calibration keeps the last
    good pose), defaulting to ``(zeros, identity)`` until the first valid sample.

    Pass ``connect=False`` to skip the socket/thread (used in tests and when a
    process drives ``_update_from_msg`` itself).
    """

    def __init__(
        self,
        port: int = DEFAULT_CUBE_PORT,
        host: str = "localhost",
        *,
        connect: bool = True,
    ) -> None:
        self.port = port
        self._latest_pos = np.zeros(3)
        self._latest_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._cached_valid_pos = np.zeros(3)
        self._cached_valid_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.cube_count = 0
        self.world_fixed = False
        self.cube_size: float | None = None
        self._lock = threading.Lock()
        self._running = connect
        self._socket: Any = None
        self._context: Any = None
        self._thread: Any = None
        if connect:
            import zmq

            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(f"tcp://{host}:{port}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.setsockopt(zmq.RCVHWM, 1)
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self._thread.start()

    def _update_from_msg(self, msg: dict[str, Any]) -> None:
        """Ingest one parsed cube message (also the unit-test entry point)."""
        pos, quat_wxyz, world_fixed, cube_size = cube_pose_from_msg(msg)
        with self._lock:
            self._latest_pos = pos
            self._latest_quat = quat_wxyz
            self.world_fixed = world_fixed
            if cube_size is not None:
                self.cube_size = cube_size
            self.cube_count += 1

    def latest(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cube_pos, cube_quat_wxyz)``, caching the last valid sample."""
        with self._lock:
            if self.cube_count > 0 and self.world_fixed:
                self._cached_valid_pos = self._latest_pos.copy()
                self._cached_valid_quat = self._latest_quat.copy()
            return self._cached_valid_pos.copy(), self._cached_valid_quat.copy()

    def close(self) -> None:
        self._running = False
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()

    def _receiver_loop(self) -> None:
        import zmq

        while self._running:
            try:
                data = self._socket.recv(zmq.NOBLOCK)
                msg = json.loads(data.decode("utf-8"))
                if "cube1" in msg:
                    self._update_from_msg(msg)
            except zmq.Again:
                pass
            except (json.JSONDecodeError, KeyError):
                pass
            time.sleep(_RECEIVER_POLL_INTERVAL)


class GoalReceiver:
    """Subscribe to goal orientation (port 5556) and expose the latest goal.

    ``latest()`` returns identity ``[1, 0, 0, 0]`` until a goal arrives. Pass
    ``connect=False`` to skip the socket/thread (tests / self-driven feeds).
    """

    def __init__(
        self,
        port: int = DEFAULT_GOAL_PORT,
        host: str = "localhost",
        *,
        connect: bool = True,
    ) -> None:
        self.port = port
        self._latest_goal = np.array([1.0, 0.0, 0.0, 0.0])
        self.goal_count = 0
        self._lock = threading.Lock()
        self._running = connect
        self._socket: Any = None
        self._context: Any = None
        self._thread: Any = None
        if connect:
            import zmq

            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(f"tcp://{host}:{port}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.setsockopt(zmq.RCVHWM, 1)
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self._thread.start()

    def _update_from_msg(self, msg: dict[str, Any]) -> None:
        """Ingest one parsed goal message (also the unit-test entry point)."""
        goal = goal_from_msg(msg)
        with self._lock:
            self._latest_goal = goal
            self.goal_count += 1

    def latest(self) -> np.ndarray:
        """Return the latest goal quaternion (wxyz); identity until one arrives."""
        with self._lock:
            return self._latest_goal.copy()

    def close(self) -> None:
        self._running = False
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()

    def _receiver_loop(self) -> None:
        import zmq

        while self._running:
            try:
                data = self._socket.recv(zmq.NOBLOCK)
                msg = json.loads(data.decode("utf-8"))
                if "goal" in msg:
                    self._update_from_msg(msg)
            except zmq.Again:
                pass
            except (json.JSONDecodeError, KeyError):
                pass
            time.sleep(_RECEIVER_POLL_INTERVAL)


class GoalPublisher:
    """Publish goal orientation (wxyz) on port 5556 (toreal_viewer -> play_real)."""

    def __init__(self, port: int = DEFAULT_GOAL_PORT) -> None:
        import zmq

        self.port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{port}")

    def publish(self, goal_quat_wxyz: np.ndarray) -> None:
        msg = {
            "timestamp": time.time(),
            "goal": {
                "orientation": {
                    "w": float(goal_quat_wxyz[0]),
                    "x": float(goal_quat_wxyz[1]),
                    "y": float(goal_quat_wxyz[2]),
                    "z": float(goal_quat_wxyz[3]),
                }
            },
        }
        self._socket.send_string(json.dumps(msg))

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class CubePublisher:
    """Publish cube pose on port 5555 (cube_world_observer -> consumers)."""

    def __init__(self, port: int = DEFAULT_CUBE_PORT) -> None:
        import zmq

        self.port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{port}")

    def publish(
        self,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        *,
        world_fixed: bool,
        cube_size: float | None = None,
    ) -> None:
        import zmq

        msg = cube_msg_from_pose(pos, quat_wxyz, world_fixed=world_fixed, cube_size=cube_size)
        msg["timestamp"] = time.time()
        self._socket.send_string(json.dumps(msg), flags=zmq.NOBLOCK)

    def close(self) -> None:
        self._socket.close()
        self._context.term()
