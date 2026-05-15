"""Rigid-mount RGB-D camera sensor backed by Genesis's ``BatchRenderer``.

The sensor wraps a single Genesis camera handle bolted to a named articulation link via
a fixed 4×4 transform. Each :meth:`_compute_data` invocation calls
``cam.move_to_attach()`` then ``cam.render(...)``, returning a :class:`CameraData`
dataclass with optional RGB (uint8) and depth (float meters) channels.

The cfg's ``offset_pos`` / ``offset_quat`` are wxyz-quaternion local-frame placement
of the camera with respect to the parent link, matching the convention used by
:class:`FrameTransformerSensor`. ``+x`` is the camera's forward direction in the
local frame (Genesis convention).

Constraints (governed by Genesis's renderer, not this wrapper):

* ``BatchRenderer`` is the only renderer that produces per-env tensors with a leading
  ``num_envs`` batch dimension. It currently requires **Linux x86-64 + CUDA**. The
  scene must be constructed with
  ``gs.Scene(renderer=gs.renderers.BatchRenderer(use_rasterizer=False), ...)`` and
  ``gs.init(backend=gs.cuda)``.
* All BatchRender cameras share resolution; mixing different ``(width, height)`` pairs
  across cameras on the same scene will assert in Genesis.

Out of scope for this revision: ``follow_entity`` chase mode, segmentation / normal
channels, single-env fallback path.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from genelab.sensor.sensor import Sensor, SensorCfg
from genelab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class CameraSensorCfg(SensorCfg):
    """Configuration for :class:`CameraSensor`.

    ``link_name`` resolves through ``env.link_names`` (mirrors :class:`IMUSensorCfg`).
    ``offset_pos`` is the camera origin in the parent link's local frame; ``offset_quat``
    is a wxyz unit quaternion describing the camera orientation relative to the link.
    ``render_rgb`` and ``render_depth`` independently toggle the two channels so a
    depth-only or RGB-only configuration skips the unused render path.
    """

    link_name: str = ""
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    width: int = 128
    height: int = 128
    fov: float = 60.0
    near: float = 0.05
    far: float = 10.0
    render_rgb: bool = True
    render_depth: bool = True

    def build(self) -> "CameraSensor":
        return CameraSensor(self)


@dataclass
class CameraData:
    """Cached per-step camera output. Disabled channels are ``None``."""

    rgb: torch.Tensor | None  # (num_envs, H, W, 3) uint8 when render_rgb else None
    depth: torch.Tensor | None  # (num_envs, H, W) float32 (meters) when render_depth else None


class CameraSensor(Sensor[CameraData]):
    def __init__(self, cfg: CameraSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._link_idx: int = -1
        self._cam: object | None = None

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        if not self._cfg_typed.link_name:
            raise ValueError(f"CameraSensorCfg(name={self._cfg.name!r}) requires link_name")
        if not (self._cfg_typed.render_rgb or self._cfg_typed.render_depth):
            raise ValueError(
                f"CameraSensorCfg(name={self._cfg.name!r}): at least one of "
                f"render_rgb / render_depth must be True"
            )
        try:
            self._link_idx = env.link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"env.link_names={env.link_names!r}"
            ) from exc

        # Resolve Genesis handles directly off the env — no ``import genesis`` needed.
        gs_scene = env.scene.gs_scene
        link = env.robot.get_link(self._cfg_typed.link_name)
        cam = gs_scene.add_camera(
            res=(self._cfg_typed.width, self._cfg_typed.height),
            pos=(0.0, 0.0, 1.0),
            lookat=(1.0, 0.0, 1.0),
            fov=float(self._cfg_typed.fov),
            near=float(self._cfg_typed.near),
            far=float(self._cfg_typed.far),
        )
        cam.attach(link, offset_T=self._build_offset_matrix())
        self._cam = cam

    def _build_offset_matrix(self) -> np.ndarray:
        # ``matrix_from_quat`` accepts wxyz and returns a (B, 3, 3) rotation matrix;
        # transpose into a 4×4 homogeneous transform on the host (Genesis's ``attach``
        # consumes a numpy array regardless of the simulation device).
        quat = torch.tensor([self._cfg_typed.offset_quat], dtype=torch.float32)
        rotation = matrix_from_quat(quat)[0].cpu().numpy()
        offset_T = np.eye(4, dtype=np.float64)
        offset_T[:3, :3] = rotation
        offset_T[:3, 3] = np.asarray(self._cfg_typed.offset_pos, dtype=np.float64)
        return offset_T

    def _compute_data(self) -> CameraData:
        if self._cam is None or self._env is None:
            raise RuntimeError(f"sensor {self._cfg.name!r}: _compute_data called before bind")
        # ``move_to_attach`` is required every step — Genesis does not auto-update the
        # camera pose even when attached to a tracked link.
        self._cam.move_to_attach()  # type: ignore[attr-defined]
        rgb_t, depth_t, _, _ = self._cam.render(  # type: ignore[attr-defined]
            rgb=self._cfg_typed.render_rgb,
            depth=self._cfg_typed.render_depth,
        )
        return CameraData(rgb=rgb_t, depth=depth_t)
