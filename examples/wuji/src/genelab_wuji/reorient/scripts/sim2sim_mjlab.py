"""sim2sim: evaluate the GeneLab-trained reorient policy in wuji-mjlab's own environment.

The *simulation* is 100% wuji-mjlab — its ``scene_builder`` (hand + cube + contacts), physics,
``apply_action`` (EMA / scale / warmup), goal sampling, and drop + hold/success criterion.
Only the policy and its observation/action adapter are GeneLab's (see ``_mjlab_bridge``):
the GeneLab actor observation is rebuilt from the wuji-mjlab scene state with the
joint-major <-> finger-major remap and the 3-step history.

Run inside the wuji-mjlab environment with GeneLab on PYTHONPATH, e.g.::

    PYTHONPATH=<genelab>/src:<genelab>/examples/wuji/src \
      <wuji-mjlab>/.pixi/envs/default/bin/python -m genelab_wuji.reorient.scripts.sim2sim_mjlab \
      --policy <checkpoint.pt> --trials 100
"""

import argparse

import mujoco
import numpy as np
import torch
from wuji_mjlab.tasks.reorient.tooling.scene_builder import (
    HoldState,
    apply_action,
    build_reorient_scene,
    check_hold,
    quat_error_magnitude,
    random_quat_uniform,
    reset_scene,
    set_goal_mocap,
)

from genelab.utils.math import quat_apply_inverse
from genelab_wuji.reorient.scripts._mjlab_bridge import (
    MJLAB_JOINTS,
    POLICY_JOINTS,
    ObsHistory,
    default_policy_joint_pos,
    goal_err_6d,
    load_policy,
    tag_frame,
)

_M_TO_P = [MJLAB_JOINTS.index(p) for p in POLICY_JOINTS]  # mjlab-order array -> policy order
_P_TO_M = [POLICY_JOINTS.index(m) for m in MJLAB_JOINTS]  # policy-order array -> mjlab order


def run(
    policy_path: str, trials: int, seed: int, timeout: float = 14.0, tilt_deg: float = 0.0
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scene = build_reorient_scene(sim_dt=0.01, ctrl_dt=0.05, cube_edge_m=0.054)
    # Emulate a tilted hardware mount by tilting gravity in the (level-hand) eval scene
    # about +X. Same gravity-in-palm effect as pitching the hand; measures robustness to
    # a mount tilt (e.g. the real ~10 deg) without moving the fixed-base hand.
    if tilt_deg:
        import math

        g0 = float(np.linalg.norm(scene.model.opt.gravity)) or 9.81
        rad = math.radians(tilt_deg)
        scene.model.opt.gravity[:] = [0.0, g0 * math.sin(rad), -g0 * math.cos(rad)]
    policy = load_policy(policy_path)
    tag_pos, tag_quat = tag_frame()
    default_p = default_policy_joint_pos()
    dof_adr = np.array([scene.model.joint("robot/" + n).dofadr[0] for n in MJLAB_JOINTS])
    default_mj = scene.default_joint_pos.copy()
    hist = ObsHistory()

    n_success = n_drop = n_timeout = 0
    total_reaches = 0
    prev_target = default_mj.copy()
    last_action = torch.zeros(20)
    episode_step = 0
    need_reset = True

    for _ in range(trials):
        if need_reset:
            reset_scene(scene)
            hist.reset()
            prev_target = default_mj.copy()
            last_action = torch.zeros(20)
            episode_step = 0
        cube_q = scene.cube_quat
        goal = random_quat_uniform()
        for _ in range(1000):
            goal = random_quat_uniform()
            if quat_error_magnitude(goal, cube_q) >= np.pi / 2:
                break
        set_goal_mocap(scene, goal)
        mujoco.mj_forward(scene.model, scene.data)
        goal_t = torch.tensor(np.asarray(goal), dtype=torch.float32)

        start = scene.data.time
        result = "timeout"
        hold = HoldState()
        reaches = 0
        above = True
        while scene.data.time - start < timeout:
            qpos = torch.tensor(scene.data.qpos[scene.joint_qpos_adr][_M_TO_P], dtype=torch.float32)
            qvel = torch.tensor(scene.data.qvel[dof_adr][_M_TO_P], dtype=torch.float32)
            cube_pos = torch.tensor(scene.cube_pos, dtype=torch.float32)
            cube_quat = torch.tensor(scene.cube_quat, dtype=torch.float32)
            cube_in_tag = quat_apply_inverse(
                tag_quat.unsqueeze(0), (cube_pos - tag_pos).unsqueeze(0)
            )[0]
            obs = hist.build(
                qpos - default_p, qvel, cube_in_tag, goal_err_6d(cube_quat, goal_t), last_action
            )
            action = policy(obs.unsqueeze(0)).squeeze(0)
            last_action = action.clone()
            prev_target = apply_action(
                scene,
                action.numpy()[_P_TO_M],
                prev_target,
                episode_step,
                action_scale=0.5,
                ema_alpha=0.5,
                warmup_time_s=0.4,
            )
            episode_step += 1
            for _ in range(scene.n_substeps):
                mujoco.mj_step(scene.model, scene.data)
            if scene.cube_pos[2] < scene.default_cube_pos[2] - 0.15:
                result = "drop"
                break
            err = quat_error_magnitude(scene.cube_quat, np.asarray(goal))
            if err < 0.2:
                if above:
                    reaches += 1
                above = False
            else:
                above = True
            hold, events = check_hold(
                err, hold, threshold=0.2, success_hold_steps=5, goal_switch_delay=20
            )
            if events.success_achieved:
                result = "success"
                break
        total_reaches += reaches
        if result == "success":
            n_success += 1
            need_reset = False
        elif result == "drop":
            n_drop += 1
            need_reset = True
        else:
            n_timeout += 1
            need_reset = True

    return {
        "success_rate": n_success / trials,
        "drop_rate": n_drop / trials,
        "timeout_rate": n_timeout / trials,
        "mean_goal_reaches": total_reaches / trials,
        "trials": trials,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, help="rsl_rl checkpoint .pt")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gravity-tilt",
        type=float,
        default=0.0,
        help="tilt gravity by this many degrees (emulates a tilted hardware mount)",
    )
    args = p.parse_args()
    r = run(args.policy, args.trials, args.seed, tilt_deg=args.gravity_tilt)
    print(
        f"sim2sim (wuji-mjlab env, tilt={args.gravity_tilt:.0f}deg) over {r['trials']} trials: "
        f"success_rate={r['success_rate']:.2f} drop_rate={r['drop_rate']:.2f} "
        f"timeout_rate={r['timeout_rate']:.2f} mean_goal_reaches={r['mean_goal_reaches']:.2f}"
    )


if __name__ == "__main__":
    main()
