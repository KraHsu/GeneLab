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

    def pre_build_genesis(self, gs_scene: object, entities: dict[str, object]) -> None:
        """Allocate the Genesis camera before ``gs_scene.build``.

        BatchRenderer takes a snapshot of registered cameras at scene-build time and
        freezes the list; adding cameras post-build does not register with the
        renderer. ``InteractiveScene`` calls this method for every sensor cfg right
        after entity spawn and before ``gs_scene.build``, so a CameraSensor attached
        to ``InteractiveSceneCfg.batch_render=True`` lands inside the build snapshot.
        """

        self._validate_cfg()
        robot_wrapper = entities.get("robot")
        if robot_wrapper is None:
            raise ValueError(
                f"sensor {self._cfg.name!r}: pre_build_genesis needs entities['robot']; "
                f"got entities={sorted(entities.keys())!r}"
            )
        robot_handle = getattr(robot_wrapper, "gs_handle", None)
        if robot_handle is None:
            raise RuntimeError(
                f"sensor {self._cfg.name!r}: entities['robot'].gs_handle is None; "
                "call pre_build_genesis after the robot has been spawned"
            )
        link = robot_handle.get_link(self._cfg_typed.link_name)
        self._cam = self._allocate_camera(gs_scene, link)

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        self._validate_cfg()
        try:
            self._link_idx = env.link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"env.link_names={env.link_names!r}"
            ) from exc

        # ``pre_build_genesis`` typically allocates the Genesis camera ahead of
        # ``gs_scene.build`` so BatchRenderer can register it. When that has not run
        # (e.g. tests injecting a fake env, or batch_render=False with Rasterizer),
        # fall back to post-build allocation — the default renderer is happy to
        # accept cameras added either side of build.
        if self._cam is None:
            gs_scene = env.scene.gs_scene
            link = env.robot.get_link(self._cfg_typed.link_name)
            self._cam = self._allocate_camera(gs_scene, link)

    def _validate_cfg(self) -> None:
        if not self._cfg_typed.link_name:
            raise ValueError(f"CameraSensorCfg(name={self._cfg.name!r}) requires link_name")
        if not (self._cfg_typed.render_rgb or self._cfg_typed.render_depth):
            raise ValueError(
                f"CameraSensorCfg(name={self._cfg.name!r}): at least one of "
                f"render_rgb / render_depth must be True"
            )

    def _allocate_camera(self, gs_scene: object, link: object) -> object:
        cam = gs_scene.add_camera(  # type: ignore[attr-defined]
            res=(self._cfg_typed.width, self._cfg_typed.height),
            pos=(0.0, 0.0, 1.0),
            lookat=(1.0, 0.0, 1.0),
            fov=float(self._cfg_typed.fov),
            near=float(self._cfg_typed.near),
            far=float(self._cfg_typed.far),
        )
        cam.attach(link, offset_T=self._build_offset_matrix())
        return cam

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
        rgb_raw, depth_raw, _, _ = self._cam.render(  # type: ignore[attr-defined]
            rgb=self._cfg_typed.render_rgb,
            depth=self._cfg_typed.render_depth,
        )
        # Genesis returns numpy arrays from the default Rasterizer (no batch dim when
        # ``num_envs == 1`` and the camera is not env-separated) and torch tensors from
        # BatchRenderer (always 4D / 3D). Normalise both into torch tensors with a
        # leading batch dimension so consumers can rely on a single shape.
        num_envs = self._env.num_envs
        rgb_out = self._coerce_rgb(rgb_raw, num_envs) if rgb_raw is not None else None
        depth_out = self._coerce_depth(depth_raw, num_envs) if depth_raw is not None else None
        return CameraData(rgb=rgb_out, depth=depth_out)

    @staticmethod
    def _to_tensor(raw: object) -> torch.Tensor:
        """torch.as_tensor with a numpy fallback that copies non-contiguous strides.

        Genesis's offscreen renderer occasionally hands back numpy arrays with
        negative strides (post-flip output buffers); ``torch.as_tensor`` rejects
        those. ``np.ascontiguousarray`` is cheap and produces a layout torch can
        adopt without further copying.
        """

        if isinstance(raw, torch.Tensor):
            return raw
        if isinstance(raw, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(raw))
        return torch.as_tensor(raw)

    @classmethod
    def _coerce_rgb(cls, rgb: object, num_envs: int) -> torch.Tensor:
        """Return ``(num_envs, H, W, 3)`` uint8 torch tensor from numpy / torch input."""

        tensor = cls._to_tensor(rgb)
        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != num_envs and tensor.shape[0] == 1 and num_envs > 1:
            tensor = tensor.expand(num_envs, -1, -1, -1).contiguous()
        return tensor

    @classmethod
    def _coerce_depth(cls, depth: object, num_envs: int) -> torch.Tensor:
        """Return ``(num_envs, H, W)`` float32 torch tensor from numpy / torch input."""

        tensor = cls._to_tensor(depth).to(torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != num_envs and tensor.shape[0] == 1 and num_envs > 1:
            tensor = tensor.expand(num_envs, -1, -1).contiguous()
        return tensor
