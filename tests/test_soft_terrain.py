"""Tests for the ``genelab_soft_terrain`` example (ADR-0001 stage-0 capstone).

Cfg-level checks run everywhere; the build + step end-to-end test is gated on the
``genesis_runtime`` fixture so it skips cleanly on headless CI.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.terrains.deformable import compliant_normal_force  # noqa: E402
from genelab_soft_terrain import go1_soft_stand_env_cfg  # noqa: E402

_GO1_FEET = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")


# Foot geom site offset below the calf link origin (genelab.asset_zoo.unitree_go1).
_FOOT_SITE_DROP = 0.213


def test_env_cfg_wires_deformable_terrain_to_the_go1_feet() -> None:
    cfg = go1_soft_stand_env_cfg()
    assert cfg.deformable_terrain is not None
    assert cfg.deformable_terrain.foot_link_names == _GO1_FEET


def test_spawn_and_surface_keep_the_feet_above_the_backstop_plane() -> None:
    cfg = go1_soft_stand_env_cfg()
    terrain = cfg.deformable_terrain
    assert terrain is not None
    # With the calves settling near the virtual surface, the foot geoms (~0.213 m below)
    # must stay above the z=0 backstop plane.
    assert terrain.surface_height - _FOOT_SITE_DROP > 0.0
    # Spawn the base above the surface so the feet drop onto it rather than through it.
    assert cfg.robot.init_pos[2] >= terrain.surface_height


def test_register_adds_the_soft_stand_env() -> None:
    from genelab.registry import ENVS
    from genelab_soft_terrain.tasks import SOFT_STAND_ENV_NAME, register

    register()
    assert SOFT_STAND_ENV_NAME in ENVS


# --------------------------------------------------- end-to-end (Genesis runtime)


def test_soft_terrain_supports_the_robot_at_a_settled_equilibrium(genesis_runtime: Any) -> None:
    """Held in its home stance, the Go1 is supported entirely by the analytic terrain:
    the total compliance force balances the robot's weight and every foot sinks (none
    rests on the z=0 backstop)."""
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    cfg = go1_soft_stand_env_cfg()
    cfg.simulation.num_envs = 1
    cfg.simulation.gpu = False
    cfg.simulation.vis = False
    cfg.device = "cpu"
    env = ManagerBasedRlEnv(cfg)

    hold_home_pose = torch.zeros_like(env.action_manager.action)
    for _ in range(200):
        env.step(hold_home_pose)

    driver = env.deformable_terrain
    assert driver is not None
    terrain = cfg.deformable_terrain
    state = driver.terrain.state
    robot = env.articulations["robot"]
    weight = float(robot.gs_handle.get_mass()) * 9.81
    # Total injected vertical force (k*d + c*ḋ) — the force actually holding the robot up.
    support = float(
        compliant_normal_force(state.depth, state.depth_rate, terrain.k, terrain.c).sum()
    )

    # The analytic soft terrain carries the robot's full weight (slack for leg inertia).
    assert support == pytest.approx(weight, rel=0.1)
    # Every foot is supported by sinking into the surface, not resting on the backstop.
    assert float(state.depth.min()) > 0.0
