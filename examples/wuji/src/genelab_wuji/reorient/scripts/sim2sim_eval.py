"""Sim2sim eval: run the Genesis-trained reorient policy in MuJoCo.

Cross-simulator validation — the policy is trained in GeneLab (Genesis) and evaluated here
in MuJoCo using the same hand XML (``right_mjlab.xml``, native MuJoCo position actuators
with the calibrated kp/kv) plus a free cube. The GeneLab MDP (obs / EMA action / SO(3)
goal state machine / success criterion) is reconstructed against MuJoCo state so the
exported TorchScript policy sees exactly the observation it was trained on.

Run:
    python -m genelab_wuji.reorient.scripts.sim2sim_eval \
        --policy /path/to/policy.ts --trials 100
"""

import argparse
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import nn

from genelab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
)

from genelab_wuji.reorient.asset import resolve_reorient_mjcf
from genelab_wuji.reorient.constants import (
    REORIENT_CUBE_HALF_EXTENT,
    REORIENT_CUBE_INIT_POS,
    REORIENT_JOINT_POS,
    REORIENT_ROBOT_ROOT_POS,
    REORIENT_ROBOT_ROOT_ROT,
    TAG_IN_PALM_POS,
    TAG_IN_PALM_QUAT_WXYZ,
)
from genelab_wuji.reorient.mdp._math import matrix_to_rotation_6d, random_quat

# MuJoCo qpos/ctrl order (right_mjlab.xml, finger-major) vs the policy's joint order.
# Genesis enumerates joints joint-major (all joint1s, then all joint2s, ...), so the obs
# the policy was trained on — and the actions it emits — are in that order. We remap.
_MUJOCO_JOINTS = tuple(f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))
_POLICY_JOINTS = tuple(f"right_finger{f}_joint{j}" for j in range(1, 5) for f in range(1, 6))
_FINGER_JOINTS = _MUJOCO_JOINTS  # backward-compat alias
_ACTION_SCALE = 0.5
_EMA_ALPHA = 0.5
_WARMUP_STEPS = 8  # 0.4 s / 0.05 s control dt
_DECIMATION = 5
_SUCCESS_THRESHOLD = 0.2
_SUCCESS_HOLD_STEPS = 5
_GOAL_SWITCH_DELAY = 20


def load_policy(checkpoint_path: str):
    """Load the actor + empirical obs-normalizer from an rsl_rl checkpoint.

    The TorchScript export drops the running-mean/std normalizer, so we rebuild the policy
    straight from ``actor_state_dict`` (the same weights Genesis play uses).
    """
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].float()
    std = sd["obs_normalizer._std"].float()
    mlp = nn.Sequential(
        nn.Linear(69, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 20),
    )
    mlp.load_state_dict({k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")})
    mlp.eval()

    def policy(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return mlp((obs - mean) / std)

    return policy


def build_model() -> mujoco.MjModel:
    """Assemble the palm-up hand + free cube + floor into one MuJoCo model."""
    spec = mujoco.MjSpec.from_file(str(resolve_reorient_mjcf()))
    palm = spec.body("right_palm_link")
    palm.pos = list(REORIENT_ROBOT_ROOT_POS)
    palm.quat = list(REORIENT_ROBOT_ROOT_ROT)
    cube = spec.worldbody.add_body(name="cube", pos=list(REORIENT_CUBE_INIT_POS))
    cube.add_freejoint()
    edge = REORIENT_CUBE_HALF_EXTENT
    cube.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[edge, edge, edge],
        mass=0.12,
        contype=3,
        conaffinity=3,
        rgba=[0.85, 0.3, 0.3, 1.0],
    )
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1])
    return spec.compile()


@dataclass
class _Frames:
    tag_pos: torch.Tensor
    tag_quat: torch.Tensor


def _tag_frame() -> _Frames:
    palm_pos = torch.tensor(REORIENT_ROBOT_ROOT_POS)
    palm_quat = torch.tensor(REORIENT_ROBOT_ROOT_ROT)
    tag_quat = quat_mul(palm_quat, torch.tensor(TAG_IN_PALM_QUAT_WXYZ))
    tag_pos = palm_pos + quat_apply(palm_quat, torch.tensor(TAG_IN_PALM_POS))
    return _Frames(tag_pos=tag_pos, tag_quat=tag_quat)


@dataclass
class EvalResult:
    success_rate: float
    drop_rate: float
    timeout_rate: float
    mean_goal_reaches: float
    trials: int


def run_eval(
    policy_path: str,
    trials: int = 100,
    max_control_steps: int = 280,
    drop_height: float = 0.35,
    seed: int = 0,
) -> EvalResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model()
    data = mujoco.MjData(model)
    policy = load_policy(policy_path)

    mj_order = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(20)]
    m2p = torch.tensor([mj_order.index(j) for j in _POLICY_JOINTS])  # mujoco -> policy order
    p2m = torch.tensor([_POLICY_JOINTS.index(j) for j in mj_order])  # policy -> mujoco order
    layout = _Layout(
        default_mj=torch.tensor([REORIENT_JOINT_POS[n] for n in mj_order]),
        default_policy=torch.tensor([REORIENT_JOINT_POS[n] for n in _POLICY_JOINTS]),
        lo_policy=torch.tensor(model.jnt_range[:20, 0], dtype=torch.float32)[m2p],
        hi_policy=torch.tensor(model.jnt_range[:20, 1], dtype=torch.float32)[m2p],
        m2p=m2p,
        p2m=p2m,
    )
    frames = _tag_frame()

    n_success = n_drop = n_timeout = total_goals = 0
    for trial in range(trials):
        goals = _run_trial(model, data, policy, layout, frames, max_control_steps, drop_height)
        total_goals += goals.reaches
        if goals.dropped:
            n_drop += 1
        elif goals.reaches >= 1:
            n_success += 1
        else:
            n_timeout += 1
    return EvalResult(
        success_rate=n_success / trials,
        drop_rate=n_drop / trials,
        timeout_rate=n_timeout / trials,
        mean_goal_reaches=total_goals / trials,
        trials=trials,
    )


@dataclass
class _TrialOutcome:
    reaches: int
    dropped: bool


@dataclass
class _Layout:
    """Joint-order bridge between MuJoCo (qpos/ctrl) and the policy (Genesis order)."""

    default_mj: torch.Tensor  # home pose in MuJoCo joint order (reset / ctrl base)
    default_policy: torch.Tensor  # home pose in policy joint order (obs / action base)
    lo_policy: torch.Tensor
    hi_policy: torch.Tensor
    m2p: torch.Tensor  # index: policy_order_vec = mujoco_vec[m2p]
    p2m: torch.Tensor  # index: mujoco_order_vec = policy_vec[p2m]


def _run_trial(model, data, policy, layout: _Layout, frames, max_steps, drop_height):
    mujoco.mj_resetData(model, data)
    # reset hand to home keyframe (MuJoCo order), cube to init pos + random orientation
    data.qpos[:20] = layout.default_mj.numpy()
    data.qpos[20:23] = np.array(REORIENT_CUBE_INIT_POS) + np.random.uniform(-0.01, 0.01, 3)
    data.qpos[23:27] = random_quat(1, "cpu")[0].numpy()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    goal_quat = quat_mul(frames.tag_quat.unsqueeze(0), random_quat(1, "cpu"))[0]
    prev_target = layout.default_policy.clone()  # policy order
    last_action = torch.zeros(20)
    hold = window = reaches = 0
    in_window = False

    for step in range(max_steps):
        cube_pos = torch.tensor(data.qpos[20:23], dtype=torch.float32)
        cube_quat = torch.tensor(data.qpos[23:27], dtype=torch.float32)
        # --- observation (policy joint order) ---
        qpos = torch.tensor(data.qpos[:20], dtype=torch.float32)[layout.m2p]
        qvel = torch.tensor(data.qvel[:20], dtype=torch.float32)[layout.m2p]
        joint_pos_rel = qpos - layout.default_policy
        cube_in_tag = quat_apply_inverse(frames.tag_quat, cube_pos - frames.tag_pos)
        err_quat = quat_mul(cube_quat, quat_conjugate(goal_quat))
        goal_6d = matrix_to_rotation_6d(matrix_from_quat(err_quat))
        obs = torch.cat([joint_pos_rel, qvel, cube_in_tag, goal_6d, last_action])
        action = policy(obs.unsqueeze(0)).squeeze(0)  # policy order
        last_action = action.clone()
        # --- EMA joint-position-offset action (policy order) ---
        raw_target = torch.minimum(
            torch.maximum(
                layout.default_policy + _ACTION_SCALE * action.clamp(-1, 1), layout.lo_policy
            ),
            layout.hi_policy,
        )
        smoothed = _EMA_ALPHA * raw_target + (1 - _EMA_ALPHA) * prev_target
        target = layout.default_policy if step < _WARMUP_STEPS else smoothed
        prev_target = target.clone()
        data.ctrl[:] = target[layout.p2m].numpy()  # back to MuJoCo ctrl order
        for _ in range(_DECIMATION):
            mujoco.mj_step(model, data)
        # --- SO(3) goal state machine ---
        err = float(quat_error_magnitude(cube_quat.unsqueeze(0), goal_quat.unsqueeze(0))[0])
        within = err < _SUCCESS_THRESHOLD
        if not in_window:
            hold = hold + 1 if within else 0
            if hold >= _SUCCESS_HOLD_STEPS:
                in_window, window, hold, reaches = True, 0, 0, reaches + 1
        else:
            window += 1
            if window >= _GOAL_SWITCH_DELAY:
                goal_quat = quat_mul(frames.tag_quat.unsqueeze(0), random_quat(1, "cpu"))[0]
                in_window, window, hold = False, 0, 0
        if float(data.qpos[22]) < drop_height:
            return _TrialOutcome(reaches=reaches, dropped=True)
    return _TrialOutcome(reaches=reaches, dropped=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, help="rsl_rl checkpoint .pt (actor + obs normalizer)")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--steps", type=int, default=280)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    r = run_eval(args.policy, trials=args.trials, max_control_steps=args.steps, seed=args.seed)
    print(
        f"sim2sim (MuJoCo) over {r.trials} trials: "
        f"success_rate={r.success_rate:.2f} drop_rate={r.drop_rate:.2f} "
        f"timeout_rate={r.timeout_rate:.2f} mean_goal_reaches={r.mean_goal_reaches:.2f}"
    )


if __name__ == "__main__":
    main()
