"""Probe: measured base velocity of a trained go2w policy under fixed commands.

Builds the stage-1 (lock_wheels) play env headless with N envs, pins the twist command to a
single fixed value (degenerate ranges, no standing/forward split), loads the rsl_rl
checkpoint, and reports the mean measured base velocity over the last K control steps.
Answers "does the policy actually move?" with numbers instead of a viewer.

Usage: python scripts/go2w_probe_tracking.py <checkpoint> [--lock-wheels] [--num-envs 64]
"""

import argparse
from pathlib import Path
from typing import Any, cast

import torch


def probe(checkpoint: Path, lock_wheels: bool, num_envs: int, case: dict[str, float]) -> None:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.rl.backends.rsl_rl import _runner_cfg_to_dict
    from genelab.rl.vecenvs.rsl_rl import RslRlVecEnvWrapper
    from genelab_unitree.go2w import unitree_go2w_velocity_env_cfg
    from genelab_unitree.go2w.ppo_cfg import unitree_go2w_ppo_runner_cfg
    from rsl_rl.runners import OnPolicyRunner

    cfg = unitree_go2w_velocity_env_cfg(play=True, lock_wheels=lock_wheels)
    cfg.simulation.num_envs = num_envs
    cfg.simulation.vis = False
    cfg.auto_reset = True
    cfg.bridges_cfg = {}
    twist = cfg.commands_cfg["twist"]
    twist.ranges.lin_vel_x = (case["vx"], case["vx"])
    twist.ranges.lin_vel_y = (case["vy"], case["vy"])
    twist.ranges.ang_vel_z = (case["wz"], case["wz"])
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 0.0

    env = ManagerBasedRlEnv(cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = OnPolicyRunner(
            cast(Any, wrapped),
            _runner_cfg_to_dict(unitree_go2w_ppo_runner_cfg()),
            log_dir=None,
            device=str(wrapped.device),
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=str(wrapped.device))

        # Disable auto-reset during measurement: a viewer robot never resets, so a stumble
        # leaves it in a degraded state for good. Per-env means + fall counts expose what an
        # env-mean with auto-reset hides (half the envs tracking, half fallen averages to 50%).
        env._auto_reset = False
        obs = wrapped.get_observations()
        per_env = torch.zeros(num_envs, 3, device=env.device)
        fallen = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        warmup, measure = 100, 200
        for t in range(warmup + measure):
            with torch.no_grad():
                action = policy(obs)
            obs, _, _, _ = wrapped.step(action)
            fallen |= env.termination_manager.terminated
            if t >= warmup:
                per_env[:, :2] += env.sensors["imu_lin_vel"].data[:, :2]
                per_env[:, 2] += env.sensors["imu_ang_vel"].data[:, 2]
        per_env /= measure
        # The commanded axis for this case (exactly one is non-zero).
        axis, target = next(
            (i, v) for i, v in enumerate((case["vx"], case["vy"], case["wz"])) if v != 0.0
        )
        signed = per_env[:, axis] * (1.0 if target > 0 else -1.0)
        ok = (signed > 0.5 * abs(target)) & ~fallen
        med = signed.median().item()
        p10 = signed.quantile(0.1).item()
        print(
            f"cmd(vx={case['vx']:.1f} vy={case['vy']:.1f} wz={case['wz']:.1f}) -> "
            f"mean vx={per_env[:, 0].mean():.3f} vy={per_env[:, 1].mean():.3f} "
            f"wz={per_env[:, 2].mean():.3f} | cmd-axis median={med:.3f} p10={p10:.3f} "
            f"fallen={int(fallen.sum())}/{num_envs} tracking>50%={int(ok.sum())}/{num_envs}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--lock-wheels", action="store_true")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument(
        "--case",
        choices=["vy", "vy-", "wz", "wz-", "vx", "vx-", "wz.2", "wz.4", "wz.6", "vy.3"],
        default="vy",
    )
    a = p.parse_args()
    cases = {
        "vy": {"vx": 0.0, "vy": 0.8, "wz": 0.0},
        "vy-": {"vx": 0.0, "vy": -0.8, "wz": 0.0},
        "wz": {"vx": 0.0, "vy": 0.0, "wz": 1.0},
        "wz-": {"vx": 0.0, "vy": 0.0, "wz": -1.0},
        "vx": {"vx": 0.8, "vy": 0.0, "wz": 0.0},
        "vx-": {"vx": -0.8, "vy": 0.0, "wz": 0.0},
        "wz.2": {"vx": 0.0, "vy": 0.0, "wz": 0.2},
        "wz.4": {"vx": 0.0, "vy": 0.0, "wz": 0.4},
        "wz.6": {"vx": 0.0, "vy": 0.0, "wz": 0.6},
        "vy.3": {"vx": 0.0, "vy": 0.3, "wz": 0.0},
    }
    probe(a.checkpoint, a.lock_wheels, a.num_envs, cases[a.case])
