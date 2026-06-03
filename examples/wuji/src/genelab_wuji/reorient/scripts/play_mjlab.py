"""Visualize the GeneLab-trained reorient policy in wuji-mjlab's full env + native viewer.

Same bridge idea as ``sim2sim_mjlab`` but for interactive viewing: wuji-mjlab's
``ManagerBasedRlEnv`` (full scene + goal command visualization) and its ``NativeMujocoViewer``
drive the loop; the policy callback rebuilds the GeneLab actor observation (joint-major remap,
3-step history, world-frame goal error) from the env state and maps the action back.

Run inside the wuji-mjlab environment with a display and GeneLab on PYTHONPATH, e.g.::

    DISPLAY=:0 PYTHONPATH=<genelab>/src:<genelab>/examples/wuji/src \
      <wuji-mjlab>/.pixi/envs/default/bin/python -m genelab_wuji.reorient.scripts.play_mjlab \
      --policy <checkpoint.pt>
"""

import argparse

import torch
import wuji_mjlab.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from wuji_mjlab.tasks.reorient.mdp.observations import cube_pos_in_tag
from wuji_mjlab.utils.cli_override_utils import DEFAULT_TASK_ALIASES, resolve_task

from genelab_wuji.reorient.scripts._mjlab_bridge import (
    POLICY_JOINTS,
    ObsHistory,
    default_policy_joint_pos,
    goal_err_6d,
    load_policy,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, help="rsl_rl checkpoint .pt")
    args = p.parse_args()

    task_id = resolve_task(
        "WujiHand_Reorient", task_aliases=DEFAULT_TASK_ALIASES, all_tasks=list_tasks()
    )
    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = ManagerBasedRlEnv(cfg=cfg, device=device)
    policy_net = load_policy(args.policy, device)
    robot = base.scene["robot"]
    command = base.command_manager.get_term("reorient_command")
    entity_joints = list(robot.joint_names)
    obs_ids = torch.tensor([entity_joints.index(p) for p in POLICY_JOINTS], device=device)
    act_ids = torch.tensor([list(POLICY_JOINTS).index(m) for m in entity_joints], device=device)
    default_p = default_policy_joint_pos(device).unsqueeze(0)
    hist = ObsHistory()
    last_action = [torch.zeros(20, device=device)]
    prev_len = [-1]

    def policy(_obs):
        r = base.scene["robot"]
        obj = base.scene["object"]
        # Backfill history on episode reset (single env): episode_length_buf == 0.
        if int(base.episode_length_buf[0]) == 0 and prev_len[0] != 0:
            hist.reset()
            last_action[0] = torch.zeros(20, device=device)
        prev_len[0] = int(base.episode_length_buf[0])
        joint_pos_rel = (r.data.joint_pos[:, obs_ids] - default_p)[0]
        joint_vel = r.data.joint_vel[:, obs_ids][0]
        cube_in_tag = cube_pos_in_tag(base, injection_prob=0.0)[0]
        g6 = goal_err_6d(obj.data.root_link_quat_w[0], command.goal_quat[0])
        obs = hist.build(joint_pos_rel, joint_vel, cube_in_tag, g6, last_action[0])
        action = policy_net(obs.unsqueeze(0))[0]
        last_action[0] = action.clone()
        return action[act_ids].unsqueeze(0)

    env = RslRlVecEnvWrapper(base, clip_actions=None)
    print(
        "launching wuji-mjlab native viewer (full env + command viz)... close window to stop",
        flush=True,
    )
    NativeMujocoViewer(env, policy).run()
    env.close()


if __name__ == "__main__":
    main()
