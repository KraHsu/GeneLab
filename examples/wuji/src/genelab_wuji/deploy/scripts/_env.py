"""Shared helpers to build the Genesis reorient scene for deploy visualization.

Both ``toreal_viewer`` (real2sim) and ``play_real`` (control mirror) need the same
play-mode reorient env plus a way to (a) read the wrist-tag world pose and (b) set
the cube / hand pose each frame. Heavy imports (genesis, torch, the env) are kept
inside the functions so importing this module is cheap and headless-safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_reorient_env() -> Any:
    """Build the play-mode reorient env (hand + cube) with auto-reset disabled."""
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab_wuji.reorient.env_cfg import wuji_hand_reorient_env_cfg

    cfg = wuji_hand_reorient_env_cfg(play=True)
    cfg.auto_reset = False  # we drive poses by hand; never teleport on "done"
    env = ManagerBasedRlEnv(cfg)
    env.reset()
    return env


def tag_world_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Wrist-tag pose in sim-world coordinates (numpy ``(3,)`` / ``(4,)``).

    Reuses the task's own ``_tag_pose`` so the viewer frame matches the obs frame.
    """
    from genelab_wuji.reorient.mdp.observations import _tag_pose

    tag_pos, tag_quat = _tag_pose(env)  # torch (B, 3) / (B, 4)
    return tag_pos[0].detach().cpu().numpy(), tag_quat[0].detach().cpu().numpy()


def set_cube_pose(env: Any, pos_w: np.ndarray, quat_w: np.ndarray) -> None:
    """Kinematically place the sim cube at the given world pose (zero velocity)."""
    import torch

    handle = env.scene["object"].gs_handle
    device = env.device
    pos = torch.tensor(pos_w, dtype=torch.float, device=device).unsqueeze(0)
    quat = torch.tensor(quat_w, dtype=torch.float, device=device).unsqueeze(0)
    zeros = torch.zeros(1, 3, device=device)
    for setter, value in (("set_pos", pos), ("set_quat", quat), ("set_vel", zeros), ("set_ang", zeros)):
        fn = getattr(handle, setter, None)
        if fn is not None:
            fn(value)
