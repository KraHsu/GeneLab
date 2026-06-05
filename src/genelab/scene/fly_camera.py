"""Free-fly viewer camera: pure pose math for WASD + mouse-look navigation.

A ``FlyCamera`` holds an eye position and a (yaw, pitch) orientation in a **Z-up** world
(Genesis convention): yaw rotates about +Z measured from +X, pitch tilts up (+) / down (-).
It is deliberately Genesis-free so the navigation feel can be unit-tested; the viewer wiring
(mouse / keybinds → these methods → ``viewer.set_camera_pose``) lives in
:mod:`genelab.scene.interactive_scene`.
"""

import math

import numpy as np

# Pitch is clamped just shy of straight up / down so ``forward`` never degenerates and the
# horizon never flips (the classic look-straight-up gimbal snap).
_MAX_PITCH = math.radians(89.0)
_WORLD_UP = np.array([0.0, 0.0, 1.0])


class FlyCamera:
    """Eye position + yaw/pitch, with fly-style move / look operations."""

    def __init__(self, pos: "np.ndarray | tuple[float, float, float]", yaw: float, pitch: float):
        self._pos = np.asarray(pos, dtype=float).copy()
        self._yaw = float(yaw)
        self._pitch = float(np.clip(pitch, -_MAX_PITCH, _MAX_PITCH))

    @classmethod
    def from_look(
        cls,
        pos: "np.ndarray | tuple[float, float, float]",
        lookat: "np.ndarray | tuple[float, float, float]",
    ) -> "FlyCamera":
        """Build a camera at ``pos`` oriented toward ``lookat`` (so the viewer can start framed)."""
        direction = np.asarray(lookat, dtype=float) - np.asarray(pos, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return cls(pos, 0.0, 0.0)
        direction = direction / norm
        yaw = math.atan2(direction[1], direction[0])
        pitch = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
        return cls(pos, yaw, pitch)

    @property
    def forward(self) -> np.ndarray:
        """Unit view direction from yaw/pitch."""
        cp = math.cos(self._pitch)
        return np.array([cp * math.cos(self._yaw), cp * math.sin(self._yaw), math.sin(self._pitch)])

    @property
    def right(self) -> np.ndarray:
        """Unit horizontal right vector (pitch-independent, so strafing stays level)."""
        return np.array([math.sin(self._yaw), -math.cos(self._yaw), 0.0])

    def look(self, dyaw: float = 0.0, dpitch: float = 0.0) -> None:
        """Rotate the view by mouse deltas (radians). Pitch is clamped to avoid a horizon flip."""
        self._yaw += dyaw
        self._pitch = float(np.clip(self._pitch + dpitch, -_MAX_PITCH, _MAX_PITCH))

    def move(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> None:
        """Translate the eye: along the view direction, the horizontal right, and world up.

        Amounts are in world metres (the caller scales by speed × dt). ``up`` is world-Z, so
        Space / Shift raise and lower regardless of where the camera is looking.
        """
        self._pos = self._pos + forward * self.forward + right * self.right + up * _WORLD_UP

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """``(eye, lookat)`` for ``viewer.set_camera_pose`` — lookat is one unit ahead."""
        return self._pos.copy(), self._pos + self.forward


# action name -> (forward, right, up) unit components fed to ``FlyCamera.move``.
_MOVE_ACTIONS: dict[str, tuple[float, float, float]] = {
    "forward": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0),
    "left": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


class FlyCameraController:
    """Gates fly inputs behind an *active* flag (held while the right mouse button is down).

    The viewer wiring feeds it raw input — :meth:`move` per movement key per tick, :meth:`on_look`
    per mouse delta — and the controller ignores everything until :meth:`set_active` ``(True)``,
    so the ordinary orbit controls keep working when fly mode is off.
    """

    def __init__(
        self, camera: FlyCamera, *, move_speed: float = 3.0, look_sensitivity: float = 0.0025
    ):
        self._camera = camera
        self._move_speed = float(move_speed)
        self._look_sensitivity = float(look_sensitivity)
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        self._active = bool(active)

    def reseat(
        self,
        pos: "np.ndarray | tuple[float, float, float]",
        lookat: "np.ndarray | tuple[float, float, float]",
    ) -> None:
        """Re-anchor the fly camera at a live viewer pose (so fly continues from the orbit view)."""
        self._camera = FlyCamera.from_look(pos, lookat)

    def on_look(self, dx: float, dy: float) -> None:
        """Mouse motion → yaw/pitch (no-op unless active).

        ``dx`` / ``dy`` are pyglet drag deltas (y is positive-up). Dragging right looks right and
        dragging up looks up, matching the usual FPS / Blender free-look feel.
        """
        if not self._active:
            return
        self._camera.look(dyaw=-dx * self._look_sensitivity, dpitch=dy * self._look_sensitivity)

    def move(self, action: str, dt: float) -> None:
        """Advance the camera for one held movement key over ``dt`` (no-op unless active)."""
        if not self._active:
            return
        fwd, right, up = _MOVE_ACTIONS[action]
        amount = self._move_speed * dt
        self._camera.move(forward=fwd * amount, right=right * amount, up=up * amount)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._camera.pose()
