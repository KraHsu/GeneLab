"""Stable-Baselines3 SAC + HER config for the Franka pick-and-place task.

The companion :func:`franka_pick_and_place_compute_reward` mirrors the env's
reward terms over the (achieved_goal, desired_goal) batches that
``HerReplayBuffer`` hands the callback. SB3 trains in environment timesteps
rather than learning iterations.
"""

import numpy as np

from genelab.rl import Sb3AgentCfg, Sb3ExperimentCfg, Sb3HerCfg, Sb3PolicyCfg

from genelab_franka.constants import CUBE_HALF, DISTANCE_THRESHOLD


def franka_pick_and_place_compute_reward(
    achieved_goal: np.ndarray, desired_goal: np.ndarray, info: object
) -> np.ndarray:
    """Sparse goal reward + lift bonus for HER goal relabelling.

    ``-1`` while the cube is farther than ``DISTANCE_THRESHOLD`` from the goal,
    ``0`` once within threshold, **plus** a per-step lift bonus that ramps from
    ``0`` (cube on table) to ``+0.2`` (cube 10 cm above table). The lift term
    mirrors :func:`genelab_franka.mdp.lift_bonus` exactly so the
    online env reward and HER's relabelled reward share one shape — HER
    relabels ``desired_goal`` only, so the cube-z component of ``achieved_goal``
    is identical in both branches. Vectorized over a batch of goal pairs
    (``HerReplayBuffer`` passes ``(batch, 3)`` numpy arrays)."""
    distance = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
    sparse = -(distance > DISTANCE_THRESHOLD).astype(np.float32)
    cube_z = np.asarray(achieved_goal)[..., 2]
    lift = np.clip(cube_z - CUBE_HALF, 0.0, 0.10) / 0.10
    return sparse + np.float32(0.2) * lift.astype(np.float32)


def franka_pick_and_place_sb3_cfg() -> Sb3AgentCfg:
    """SAC + Hindsight Experience Replay config.

    ``HerReplayBuffer`` works in whole episodes: both ``learning_starts`` and
    ``buffer_size`` must cover ``num_envs x episode_length``. The SB3 backend
    raises either to that floor automatically, so the values here only bind when
    they already exceed it. HER does not benefit from massive parallelism — a
    modest ``--num-envs`` (≤ ~64) keeps ``buffer_size`` worth many episodes per
    env; with thousands of envs the buffer holds barely one episode each.

    ``ent_coef`` is pinned via ``extra_kwargs`` (SAC accepts it that way; the
    dataclass field on ``Sb3AgentCfg`` is consumed by PPO/A2C only). The
    panda-gym default ``"auto"`` collapsed α to ~1e-4 inside the first 10% of
    training in earlier runs, killing exploration.

    ``gradient_steps`` is pinned at 8 — SB3 SAC's default is 1, which with
    ``train_freq=1`` and ``num_envs=64`` would land at ~31 k gradient updates
    over the 2 M-step budget (1 update per 64 transitions). HER on Franka
    needs ~100 k+ updates to climb past the demo-prefill plateau; 8 updates
    per rollout brings the ratio to one update per 8 transitions and the
    total to ~250 k gradient steps, which converges to high success rate
    on a single H200 in under an hour.
    """
    return Sb3AgentCfg(
        algorithm="SAC",
        seed=42,
        total_timesteps=2_000_000,
        learning_rate=1.0e-3,
        discount_factor=0.95,
        batch_size=2048,
        buffer_size=1_000_000,
        tau=0.05,
        learning_starts=4000,
        train_freq=1,
        extra_kwargs={"ent_coef": 0.1, "gradient_steps": 8},
        policy=Sb3PolicyCfg(net_arch=(512, 512, 512), activation="relu"),
        experiment=Sb3ExperimentCfg(
            experiment_name="franka_pick_and_place",
            logger="tensorboard",
        ),
        her=Sb3HerCfg(
            enabled=True,
            obs_group="policy",
            achieved_goal_group="achieved_goal",
            desired_goal_group="desired_goal",
            n_sampled_goal=4,
            goal_selection_strategy="future",
            compute_reward=franka_pick_and_place_compute_reward,
        ),
    )
