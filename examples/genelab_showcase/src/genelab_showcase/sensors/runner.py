"""Sensors-showcase runner: scripted joint-1 sweep; sensors stream to live windows.

The numeric sensors (IMU, frame-transformer, joint force/torque) stream to live PyQt curves
declared as ``recordings`` on the scene cfg. This runner drives the motion, draws the wrist
camera's view frustum in the main viewer, and renders the camera RGB + depth into a live
pyqtgraph image window (Qt, smooth). Nothing is written to disk.
"""

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from genelab_showcase._viz import LazyQtWindows
from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from genelab.sensor import CameraSensor

# Camera near / far (m) used as the depth window's display range (matches the camera cfg).
_DEPTH_RANGE = (0.02, 2.5)


class SensorsShowcaseRunner(ShowcaseRunner):
    """Wave Franka joint 1 ±0.5 rad on a 4-second period so every sensor reading moves."""

    task_slug = "sensors"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg)
        self._joint1_idx: int | None = None
        self._frustum_node: object | None = None
        self._img_interval = 2  # refresh the camera window every N steps
        self._qt = LazyQtWindows()
        self._win: Any = None
        self._rgb_item: Any = None
        self._depth_item: Any = None

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        # Action shape is (num_envs, num_actions). The Franka cfg yields 9 actions
        # (7 arm + 2 finger). Drive joint1 with a slow sine; leave others at default.
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        if self._joint1_idx is None:
            self._joint1_idx = env.articulations["robot"].joint_names.index("joint1")
        # period ≈ 4 s with dt 0.02 (decimation 2 × physics 0.01) → 200 steps.
        phase = 2.0 * math.pi * step / 200.0
        action[:, self._joint1_idx] = math.sin(phase) * 0.5
        return action

    def _build_cam_window(self, app: Any) -> None:
        import pyqtgraph as pg

        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle("hand camera (RGB | depth)")
        win.resize(700, 320)
        rgb_vb = win.addViewBox(row=0, col=0)
        rgb_vb.setAspectLocked(True)
        rgb_vb.invertY(True)  # image row 0 at the top
        self._rgb_item = pg.ImageItem(axisOrder="row-major")
        rgb_vb.addItem(self._rgb_item)
        depth_vb = win.addViewBox(row=0, col=1)
        depth_vb.setAspectLocked(True)
        depth_vb.invertY(True)
        self._depth_item = pg.ImageItem(axisOrder="row-major")
        depth_vb.addItem(self._depth_item)
        cmap = pg.colormap.get("inferno")
        if cmap is not None:
            self._depth_item.setLookupTable(cmap.getLookupTable())
        self._depth_item.setLevels(_DEPTH_RANGE)
        win.show()
        self._win = win

    def _post_step(self, env: "ManagerBasedRlEnv", step: int) -> None:
        if step % self._img_interval:
            return
        cam_sensor = cast("CameraSensor", env.sensors["hand_camera"])
        data = cam_sensor.data  # renders the camera → refreshes its transform + gives frames

        # Wrist-camera view frustum in the main viewer (camera moved with joint1).
        cam = cam_sensor.gs_camera
        gs_scene = env.scene.gs_scene
        if cam is not None:
            if self._frustum_node is not None:
                gs_scene.clear_debug_object(self._frustum_node)
            self._frustum_node = gs_scene.draw_debug_frustum(cam, color=(0.2, 0.8, 1.0, 0.35))

        # Live RGB + depth in the pyqtgraph window.
        if not self._qt.ensure(self._build_cam_window, label="hand camera"):
            return
        if data.rgb is not None:
            self._rgb_item.setImage(
                data.rgb[0].detach().cpu().numpy(), autoLevels=False, levels=(0, 255)
            )
        if data.depth is not None:
            self._depth_item.setImage(
                data.depth[0].detach().cpu().numpy(), autoLevels=False, levels=_DEPTH_RANGE
            )
        self._qt.process()
