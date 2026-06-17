"""Shared helpers to build the Genesis reorient scene for deploy visualization.

Both ``toreal_viewer`` (real2sim) and ``play_real`` (control mirror) need the same
play-mode reorient env plus a way to (a) read the wrist-tag world pose and (b) set
the cube / hand pose each frame. Heavy imports (genesis, torch, the env) are kept
inside the functions so importing this module is cheap and headless-safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_reorient_env(num_envs: int | None = None) -> Any:
    """Build the play-mode reorient env (hand + cube) with auto-reset disabled.

    ``num_envs`` overrides the cfg's play default (4); deploy viewers only ever use
    env 0, so single-process tools (e.g. ``calib_check``) pass ``1`` to avoid
    rendering parallel copies.
    """
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab_wuji.reorient.env_cfg import wuji_hand_reorient_env_cfg

    cfg = wuji_hand_reorient_env_cfg(play=True)
    cfg.auto_reset = False  # we drive poses by hand; never teleport on "done"
    if num_envs is not None:
        cfg.simulation.num_envs = num_envs
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


def set_goal_marker(env: Any, quat_w: np.ndarray) -> None:
    """Orient the play-mode ``goal_marker`` entity to the target (viewer only).

    The marker sits at a fixed display pose (``GOAL_MARKER_POS``); we only rewrite
    its orientation so play_real's viewer shows the current goal. No-op if the scene
    has no goal marker (e.g. a non-play scene).
    """
    import torch

    try:
        handle = env.scene["goal_marker"].gs_handle
    except (KeyError, AttributeError):
        return
    quat = torch.tensor(quat_w, dtype=torch.float, device=env.device).unsqueeze(0)
    set_quat = getattr(handle, "set_quat", None)
    if set_quat is not None:
        set_quat(quat)


def set_hand_joints(env: Any, qpos_encoder_order: np.ndarray) -> None:
    """Kinematically render hand encoder readings on the sim robot (calib viewer).

    ``qpos_encoder_order`` is the ``(20,)`` joint vector in ``JOINT_NAMES_20`` /
    encoder order; it is reordered (by name) into the articulation's ``joint_names``
    order and written as joint state (zero velocity). Teleport-per-tick: the next
    write re-syncs, so a single physics step can't drift the rendered pose.
    """
    import torch

    from genelab_wuji.deploy.config import JOINT_NAMES_20

    robot = env.scene["robot"]
    enc_index = {name: i for i, name in enumerate(JOINT_NAMES_20)}
    missing = [n for n in robot.joint_names if n not in enc_index]
    if missing:
        raise ValueError(f"robot joints absent from encoder order JOINT_NAMES_20: {missing}")
    perm = [enc_index[n] for n in robot.joint_names]  # joint_names[i] = encoder[perm[i]]
    reordered = np.asarray(qpos_encoder_order, dtype=float)[perm]

    device = env.device
    jp = torch.tensor(reordered, dtype=torch.float, device=device).unsqueeze(0)
    jv = torch.zeros_like(jp)
    env_ids = torch.arange(env.num_envs, device=device)
    robot.write_joint_state(jp, jv, env_ids)
