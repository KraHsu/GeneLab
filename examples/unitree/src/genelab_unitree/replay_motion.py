"""Reference-clip replay for the G1 motion-imitation task.

Snaps the robot to the vendored ``dance1_subject2`` NPZ frame-by-frame inside the
Genesis viewer — no policy, no physics interaction. Useful for visually verifying
that the joint / body ordering between the NPZ and the runtime robot is correct
before training.

Run:

    uv run python -m genelab_unitree.replay_motion
"""

import numpy as np
import torch

from genelab.asset_zoo.unitree_g1_motions import (
    G1_MJLAB_BODY_NAMES,
    G1_MJLAB_JOINT_NAMES,
    g1_lafan1_dance1_subject2,
)
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
from genelab_unitree.g1.tracking_env_cfg import unitree_g1_tracking_env_cfg


def main() -> None:
    cfg = unitree_g1_tracking_env_cfg(play=True)
    cfg.simulation.vis = True
    cfg.simulation.num_envs = 1
    env = ManagerBasedRlEnv(cfg)

    npz = np.load(g1_lafan1_dance1_subject2())
    device = env.device

    pelvis_idx = G1_MJLAB_BODY_NAMES.index("pelvis")
    joint_perm = torch.tensor(
        [G1_MJLAB_JOINT_NAMES.index(n) for n in env.articulations["robot"].joint_names],
        dtype=torch.long,
        device=device,
    )

    body_pos = torch.as_tensor(npz["body_pos_w"], dtype=torch.float32, device=device)
    body_quat = torch.as_tensor(npz["body_quat_w"], dtype=torch.float32, device=device)
    body_lin = torch.as_tensor(npz["body_lin_vel_w"], dtype=torch.float32, device=device)
    body_ang = torch.as_tensor(npz["body_ang_vel_w"], dtype=torch.float32, device=device)
    joint_pos = torch.as_tensor(npz["joint_pos"], dtype=torch.float32, device=device)[:, joint_perm]
    joint_vel = torch.as_tensor(npz["joint_vel"], dtype=torch.float32, device=device)[:, joint_perm]

    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    origin = env.env_origins[0:1]

    env.reset()
    for t in range(joint_pos.shape[0]):
        env.write_root_state_to_sim(
            body_pos[t : t + 1, pelvis_idx] + origin,
            body_quat[t : t + 1, pelvis_idx],
            body_lin[t : t + 1, pelvis_idx],
            body_ang[t : t + 1, pelvis_idx],
            env_ids,
        )
        env.write_joint_state_to_sim(joint_pos[t : t + 1], joint_vel[t : t + 1], env_ids)
        env.scene.step()
        if env.viewer_closed:
            break


if __name__ == "__main__":
    main()
