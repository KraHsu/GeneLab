"""Sensors-showcase runner: scripted joint-1 sweep, dumps RGB+depth+IMU+FrameTx."""

import math
from typing import TYPE_CHECKING, cast

import torch

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from genelab.sensor import (
        CameraData,
        FrameTransformerData,
        IMUData,
    )


class SensorsShowcaseRunner(ShowcaseRunner):
    """Wave Franka joint 1 ±0.5 rad on a 4-second period; dump every 20 steps.

    Each dump writes:

    * ``rgb_<step>.png`` — wrist-camera RGB (160×120, uint8)
    * ``depth_<step>.png`` — wrist-camera depth visualised as 16-bit gray
    * ``frame.log`` — append-only text record of IMU and FrameTransformer state
    """

    task_slug = "sensors"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg, log_interval=20)
        self._joint1_idx: int | None = None

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

    def _dump(self, env: "ManagerBasedRlEnv", step: int) -> None:
        root = self.log_root()
        cam = cast("CameraData", env.sensors["hand_camera"].data)
        imu = cast("IMUData", env.sensors["hand_imu"].data)
        ft = cast("FrameTransformerData", env.sensors["hand_in_base"].data)

        if cam.rgb is not None:
            _save_rgb(cam.rgb, root / f"rgb_{step:04d}.png")
        if cam.depth is not None:
            _save_depth(cam.depth, root / f"depth_{step:04d}.png")

        log = root / "frame.log"
        with log.open("a", encoding="utf-8") as fh:
            ori = imu.orientation[0].tolist()
            lin = imu.lin_acc_b[0].tolist()
            ang = imu.ang_acc_b[0].tolist()
            hand_pos = ft.target_pos_source[0, 0].tolist()
            link7_pos = ft.target_pos_source[0, 1].tolist()
            fh.write(
                f"step={step:04d}  "
                f"imu_q={ori}  imu_lin_acc={lin}  imu_ang_acc={ang}  "
                f"hand_in_base={hand_pos}  link7_in_base={link7_pos}\n"
            )


def _save_rgb(rgb: torch.Tensor, path: object) -> None:
    """Save a single-env RGB tensor (uint8, H×W×3 or 1×H×W×3) as PNG."""

    from PIL import Image  # type: ignore[import-not-found]

    arr = rgb.detach().cpu()
    if arr.dim() == 4:
        arr = arr[0]
    Image.fromarray(arr.numpy(), mode="RGB").save(str(path))


def _save_depth(depth: torch.Tensor, path: object) -> None:
    """Save a depth map (meters, H×W or 1×H×W) as 16-bit PNG.

    Depth is rescaled to the [near, far] range used by the showcase camera
    (0.02–2.5 m) so the resulting grayscale band exercises most of the 16-bit
    dynamic range.
    """

    from PIL import Image  # type: ignore[import-not-found]

    arr = depth.detach().cpu().float()
    if arr.dim() == 3:
        arr = arr[0]
    # Clamp to expected range and scale to uint16.
    arr = arr.clamp(0.02, 2.5)
    scaled = ((arr - 0.02) / (2.5 - 0.02) * 65535.0).clamp(0, 65535).to(torch.uint16)
    Image.fromarray(scaled.numpy(), mode="I;16").save(str(path))
