"""Offline demonstration collection for the HER pick-and-place task.

Run :class:`FrankaPickAndPlaceFsm` in the goal-conditioned env for a fixed
number of env steps, capture every transition in the HER ``Dict``
(observation, achieved_goal, desired_goal) format, and save the result as
``.npz``. The trained-side counterpart lives in
``genelab.rl.backends.sb3._maybe_prefill_demos``, which calls
:meth:`HerReplayBuffer.add` over the same transitions in time order before
``model.learn`` starts.

The demos use the env's auto-reset path: when an env finishes its episode
mid-collection the underlying env resets it and the FSM's per-env phase is
zeroed for the same ids, so each env yields a continuous stream of
``reach -> grasp -> lift -> place`` trajectories back-to-back.

Run from the repo root::

    uv run python -m genelab_franka_pick_and_place.collect_demos \\
        --num-envs 32 --steps 6400 --out /tmp/franka_pp_demos.npz
"""

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from genelab.cache import ensure_project_cache
from genelab.lab import TASKS
from genelab.registry import load_extension_module

if TYPE_CHECKING:
    import torch


def _to_numpy(t: "torch.Tensor") -> np.ndarray:
    # ``.cpu()`` of a CUDA tensor copies into a fresh CPU buffer, but the
    # underlying torch tensor reused by the reward manager / observation
    # manager would still alias across step calls if we kept the numpy view
    # alive. ``.copy()`` makes the captured array self-contained.
    return t.detach().cpu().numpy().copy()


def _her_obs(obs_dict: dict[str, "torch.Tensor"]) -> dict[str, np.ndarray]:
    """Convert the env's per-group obs dict into the SB3 ``HerReplayBuffer``
    ``Dict`` observation format (matches what ``GenelabSb3VecEnv`` exposes)."""
    return {
        "observation": _to_numpy(obs_dict["policy"]),
        "achieved_goal": _to_numpy(obs_dict["achieved_goal"]),
        "desired_goal": _to_numpy(obs_dict["desired_goal"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument(
        "--steps",
        type=int,
        default=6400,
        help="Total env steps to collect (per env, so total transitions = steps x num_envs).",
    )
    parser.add_argument("--episode-length-s", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("franka_pp_demos.npz"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    ensure_project_cache()
    load_extension_module("genelab_franka_pick_and_place.tasks")
    task = TASKS.get("GeneLab-Franka-Pick-And-Place-sb3-her-v0")
    env_cfg = task.cfg.env
    env_cfg.simulation.num_envs = int(args.num_envs)
    env_cfg.simulation.vis = False
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.seed = int(args.seed)

    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    from genelab_franka_pick_and_place.demo_fsm import FrankaPickAndPlaceFsm

    env = ManagerBasedRlEnv(env_cfg)
    fsm = FrankaPickAndPlaceFsm(env)

    obs_dict, _ = env.reset()
    cur_obs = _her_obs(obs_dict)

    obs_obs, obs_ag, obs_dg = [], [], []
    next_obs_obs, next_obs_ag, next_obs_dg = [], [], []
    actions, rewards, dones, truncateds = [], [], [], []

    start = time.time()
    completed_episodes = 0
    for t in range(int(args.steps)):
        action = fsm.step()
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        nxt = _her_obs(obs_dict)
        done = (terminated | truncated).to(torch.bool)

        obs_obs.append(cur_obs["observation"])
        obs_ag.append(cur_obs["achieved_goal"])
        obs_dg.append(cur_obs["desired_goal"])
        next_obs_obs.append(nxt["observation"])
        next_obs_ag.append(nxt["achieved_goal"])
        next_obs_dg.append(nxt["desired_goal"])
        actions.append(_to_numpy(action))
        rewards.append(_to_numpy(reward.reshape(-1)))
        dones.append(_to_numpy(done.reshape(-1)))
        truncateds.append(_to_numpy(truncated.reshape(-1).to(torch.bool)))

        if done.any():
            reset_ids = done.nonzero(as_tuple=False).flatten()
            completed_episodes += int(reset_ids.numel())
            fsm.reset(env_ids=reset_ids)

        cur_obs = nxt

        if (t + 1) % 200 == 0:
            elapsed = time.time() - start
            print(
                f"  step {t + 1:>5}/{args.steps}  "
                f"episodes done={completed_episodes}  "
                f"fps={(t + 1) * env.num_envs / max(elapsed, 1e-6):.0f}"
            )

    total = time.time() - start
    print(
        f"collected {args.steps} steps x {env.num_envs} envs = {args.steps * env.num_envs} transitions in {total:.1f}s"
    )
    print(f"complete episodes captured: {completed_episodes}")

    out: dict[str, np.ndarray] = {
        "obs_observation": np.stack(obs_obs).astype(np.float32),
        "obs_achieved_goal": np.stack(obs_ag).astype(np.float32),
        "obs_desired_goal": np.stack(obs_dg).astype(np.float32),
        "next_obs_observation": np.stack(next_obs_obs).astype(np.float32),
        "next_obs_achieved_goal": np.stack(next_obs_ag).astype(np.float32),
        "next_obs_desired_goal": np.stack(next_obs_dg).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "rewards": np.stack(rewards).astype(np.float32),
        "dones": np.stack(dones).astype(np.bool_),
        "truncateds": np.stack(truncateds).astype(np.bool_),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # pyright's numpy stub for ``savez_compressed`` confuses kwarg-spread with the
    # ``allow_pickle`` keyword; the call is canonical numpy usage.
    np.savez_compressed(args.out, **out)  # pyright: ignore[reportArgumentType]
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
