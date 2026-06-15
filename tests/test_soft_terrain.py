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
    assert terrain is not None
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


def test_g1_mattress_cfg_wires_the_pneumatic_chamber_to_the_g1_feet() -> None:
    from functools import partial

    from genelab.terrains import pneumatic_normal_force
    from genelab_soft_terrain.g1_mattress import g1_mattress_velocity_env_cfg

    cfg = g1_mattress_velocity_env_cfg()
    terrain = cfg.deformable_terrain
    assert terrain is not None
    assert terrain.foot_link_names == ("left_ankle_roll_link", "right_ankle_roll_link")
    # The chamber, not per-foot springs: pneumatic law with a bound capacity.
    assert isinstance(terrain.normal_law, partial)
    assert terrain.normal_law.func is pneumatic_normal_force
    assert terrain.normal_law.keywords["capacity"] > 0.0
    # Spawn above the chamber surface so the feet drop onto it, not through it.
    assert cfg.robot.init_pos[2] > terrain.surface_height


def test_g1_mattress_randomizes_the_chamber_in_training_only() -> None:
    from genelab_soft_terrain.g1_mattress import g1_mattress_velocity_env_cfg

    assert "randomize_chamber" in g1_mattress_velocity_env_cfg(play=False).events_cfg
    assert "randomize_chamber" not in g1_mattress_velocity_env_cfg(play=True).events_cfg


def test_g1_mattress_single_support_pays_only_for_one_lifted_foot() -> None:
    """The anti-standing gradient: reward fires only with exactly one foot in the
    chamber and an active movement command — double support, full flight, and
    commanded standing all earn zero."""
    from types import SimpleNamespace

    from genelab_soft_terrain.g1_mattress import single_support

    depth = torch.tensor([[0.05, 0.06], [0.05, 0.0], [0.0, 0.0], [0.05, 0.0]])
    cmd = torch.tensor([[0.5, 0.0, 0.0]] * 3 + [[0.0, 0.0, 0.0]])  # last env: standing
    env = SimpleNamespace(
        deformable_terrain=SimpleNamespace(
            terrain=SimpleNamespace(state=SimpleNamespace(depth=depth))
        ),
        command_manager=SimpleNamespace(get_command=lambda name: cmd),
    )

    reward = single_support(env, command_name="twist")
    assert reward.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_g1_mattress_chamber_carries_the_humanoid_at_equilibrium(genesis_runtime: Any) -> None:
    """Held in its home pose, the G1 is supported entirely by the shared air chamber:
    the pneumatic force balances the robot's weight and both feet sink into the
    chamber (no rigid floor under them)."""
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab_soft_terrain.g1_mattress import g1_mattress_velocity_env_cfg

    cfg = g1_mattress_velocity_env_cfg()
    cfg.simulation.num_envs = 1
    cfg.simulation.gpu = False
    cfg.simulation.vis = False
    cfg.events_cfg.pop("randomize_chamber")  # the calibrated point, not a random draw
    cfg.device = "cpu"
    env = ManagerBasedRlEnv(cfg)

    hold_home_pose = torch.zeros_like(env.action_manager.action)
    for _ in range(100):
        env.step(hold_home_pose)

    driver = env.deformable_terrain
    assert driver is not None
    terrain = cfg.deformable_terrain
    assert terrain is not None
    state = driver.terrain.state
    robot = env.articulations["robot"]
    weight = float(robot.gs_handle.get_mass()) * 9.81
    support = float(
        terrain.normal_law(state.depth, state.depth_rate, driver.terrain.k, terrain.c).sum()
    )

    # The chamber carries the humanoid's full weight (slack for the settling dynamics).
    assert support == pytest.approx(weight, rel=0.15)
    # Both feet sink into the chamber, well clear of bottom-out.
    assert float(state.depth.min()) > 0.0
    assert float(state.depth.sum()) < 0.3  # capacity is 0.35; bottom-out would be ~there


def test_air_mattress_demo_tosses_resting_balls_on_impact(genesis_runtime: Any) -> None:
    """One sealed chamber: a ball slamming down pressurizes the support under ALL balls,
    so the balls resting elsewhere on the mattress get tossed upward — coupling that
    independent per-contact springs cannot produce. Also: the chamber (not the floor)
    is the only support, so no ball ever reaches the ground plane."""
    from genelab_soft_terrain.air_mattress import run

    metrics = run(steps=500, show_viewer=False)
    assert metrics["toss"] > 0.02  # resting balls jump >2 cm when the dropped ball lands
    assert metrics["min_z"] > 0.15  # nobody fell through the chamber to the floor
