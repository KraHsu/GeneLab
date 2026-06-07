"""``Genelab-Velocity-Rough-Unitree-Go1-v0``: Go1 quadruped complex-terrain locomotion.

The cfg-level and registration checks run everywhere (no simulator). The build +
step smoke is gated on the ``genesis_runtime`` fixture so it skips cleanly on
EGL-less CI runners, mirroring ``tests/test_g1_velocity_rough.py``.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module  # noqa: E402
from genelab.sensor import GridPattern  # noqa: E402

ROUGH_TASK_ID = "Genelab-Velocity-Rough-Unitree-Go1-v0"
ROUGH_ENV_NAME = "go1-velocity-rough-env"
# Feet are geoms on the *_calf bodies (no foot links exist in the MJCF), so the calf
# links are what bears foot-ground contact.
_GO1_FEET = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
# 17 x 11 = 187 rays — the trunk height-scan grid the policy must see.
_HEIGHT_SCAN_RAYS = GridPattern(resolution=0.1, size=(1.6, 1.0)).num_rays()


def _flat_cfg(play: bool = True):
    from genelab_unitree.go1 import unitree_go1_velocity_env_cfg

    return unitree_go1_velocity_env_cfg(play=play)


def _rough_cfg(play: bool = True):
    from genelab_unitree.go1 import unitree_go1_velocity_rough_env_cfg

    return unitree_go1_velocity_rough_env_cfg(play=play)


def test_rough_task_registers() -> None:
    load_extension_module("genelab_unitree.tasks")
    # The Go1 robot itself is registered by `genelab.asset_zoo.unitree_go1`, not the example.
    assert "go1" in ROBOTS.names()
    assert ROUGH_ENV_NAME in ENVS.names()
    assert ROUGH_TASK_ID in TASKS.names()


def test_flat_cfg_composes_go1_quadruped() -> None:
    cfg = _flat_cfg(play=True)
    # 12-DoF quadruped: the default-pose regex map fans out to all 12 actuated joints.
    assert cfg.robot is not None
    # The foot contact sensor covers all four feet, in FR/FL/RR/RL order, with air-time
    # tracking on (feet_air_time / feet_slip / clearance all read it).
    sensors = {s.name: s for s in cfg.scene.sensors}
    feet = sensors["feet_ground_contact"]
    assert tuple(feet.link_names) == _GO1_FEET
    assert feet.track_air_time is True


def test_flat_cfg_uses_quadruped_reward_set() -> None:
    rewards = _flat_cfg(play=True).rewards_cfg
    # Velocity tracking (positive) + the Isaac-Lab quadruped penalty set.
    expected = {
        "track_lin_vel",
        "track_ang_vel",
        "lin_vel_z",
        "flat_orientation",
        "base_height",
        "ang_vel_xy",
        "joint_torques",
        "joint_acc",
        "action_rate",
        "dof_pos_limits",
        "air_time",
        "foot_slip",
    }
    assert expected <= set(rewards)
    # Humanoid-only shaping terms (G1) must not leak into the quadruped cfg.
    assert "upright" not in rewards
    assert "pose" not in rewards


def test_rough_cfg_swaps_flat_ground_for_complex_terrain() -> None:
    from genelab.terrains import (
        DiscreteObstaclesCfg,
        PyramidStairsCfg,
        RandomRoughCfg,
        SlopeCfg,
    )

    rough = _rough_cfg(play=True)
    flat = _flat_cfg(play=True)

    # Flat ground has no terrain; rough swaps in a curriculum heightfield grid.
    assert flat.scene.terrain is None
    assert rough.scene.terrain is not None
    assert rough.scene.terrain.curriculum is True
    assert (rough.scene.terrain.num_rows, rough.scene.terrain.num_cols) == (10, 10)

    # "Complex" = a mix of locomotion-relevant sub-terrain types (not a single rough
    # patch), so the policy must handle stairs, clutter, slopes and rough ground.
    kinds = {type(st) for st in rough.scene.terrain.sub_terrains.values()}
    assert {PyramidStairsCfg, DiscreteObstaclesCfg, RandomRoughCfg, SlopeCfg} <= kinds


def test_rough_cfg_adds_trunk_height_scan() -> None:
    from genelab.sensor import TerrainHeightSensorCfg

    rough = _rough_cfg(play=True)
    flat = _flat_cfg(play=True)

    # The terrain-ahead scan is added to the actor (noised) and the critic (clean copy);
    # flat ground has no such observation.
    assert "height_scan" in rough.observations_cfg["policy"].terms
    assert "height_scan" in rough.observations_cfg["critic"].terms
    assert "height_scan" not in flat.observations_cfg["policy"].terms

    # Backed by a trunk-mounted grid sensor returning the full per-ray grid.
    sensors = {s.name: s for s in rough.scene.sensors}
    assert "height_scan" in sensors
    scan = sensors["height_scan"]
    assert isinstance(scan, TerrainHeightSensorCfg)
    assert scan.link_name == "trunk"
    assert scan.pattern.num_rays() == _HEIGHT_SCAN_RAYS


def test_rough_cfg_adds_terrain_level_curriculum() -> None:
    from genelab.mdp import terrain_levels_vel

    rough = _rough_cfg(play=True)
    flat = _flat_cfg(play=True)

    # The terrain-level curriculum promotes / demotes envs by walked distance; it sits
    # alongside the inherited command-range curriculum. Flat ground has no terrain levels.
    assert "terrain_levels" in rough.curriculum_cfg
    assert "command_vel" in rough.curriculum_cfg
    assert "terrain_levels" not in flat.curriculum_cfg

    term = rough.curriculum_cfg["terrain_levels"]
    assert term.func is terrain_levels_vel
    assert term.params["command_name"] == "twist"
    # Promotion threshold = half a sub-terrain cell (subterrain_size[0] / 2 = 4.0).
    assert term.params["distance_threshold"] == 4.0


def test_cfg_penalizes_undesired_thigh_contacts() -> None:
    from genelab.mdp import contact_force_limit

    cfg = _flat_cfg(play=True)

    # A safety penalty discourages the thighs (knees) from striking the ground or
    # obstacles — only the feet should bear load. Calves are excluded: the foot geom lives
    # on the calf, so calf contact is normal stance, not a fault.
    assert "undesired_contacts" in cfg.rewards_cfg
    term = cfg.rewards_cfg["undesired_contacts"]
    assert term.func is contact_force_limit
    assert term.weight < 0

    # Backed by a contact sensor over the four thigh links.
    sensors = {s.name: s for s in cfg.scene.sensors}
    body_contact = sensors[term.params["sensor_name"]]
    assert set(body_contact.link_names) == {f"{leg}_thigh" for leg in ("FR", "FL", "RR", "RL")}


def test_play_and_train_configs_split_correctly() -> None:
    train = _rough_cfg(play=False)
    play = _rough_cfg(play=True)

    # Training is a large parallel rollout on the GPU backend with runtime push DR for
    # robustness; play is a single interactive robot with the disturbance dropped and
    # auto-reset off so it keeps walking under teleop.
    assert train.simulation.num_envs == 4096
    assert play.simulation.num_envs == 1
    assert train.simulation.gpu is True and play.simulation.gpu is True
    assert train.auto_reset is True
    assert play.auto_reset is False
    assert "push_robot" in train.events_cfg
    assert "push_robot" not in play.events_cfg


def test_play_wires_imgui_twist_sliders() -> None:
    # Interactive play exposes in-viewer ImGui sliders (vx, vy, ωz) that drive the "twist"
    # command, mirroring the G1 example. Gated on imgui_bundle (the overlay dependency).
    pytest.importorskip("imgui_bundle")
    from genelab.bridges.imgui import ImGuiTwistBridgeCfg

    for cfg in (_flat_cfg(play=True), _rough_cfg(play=True)):
        assert cfg.simulation.viewer_imgui is True
        teleop = cfg.bridges_cfg.get("teleop")
        assert isinstance(teleop, ImGuiTwistBridgeCfg)
        assert teleop.command_name == "twist"

    # Training carries no teleop overlay.
    train = _flat_cfg(play=False)
    assert train.simulation.viewer_imgui is False
    assert "teleop" not in train.bridges_cfg


def test_rough_task_agent_uses_heteroscedastic_std_floor() -> None:
    from genelab.rl import RslRlOnPolicyRunnerCfg

    load_extension_module("genelab_unitree.tasks")
    task = TASKS.get(ROUGH_TASK_ID)

    assert task.cfg.trainable is True
    assert isinstance(task.cfg.agent, RslRlOnPolicyRunnerCfg)
    # The curriculum stays trainable without de-learning via a per-state action std with
    # an exploration floor (a global std over-specialises and floors the adaptive LR).
    dist = task.cfg.agent.actor.distribution_cfg
    assert dist is not None
    assert dist["class_name"] == "HeteroscedasticGaussianDistribution"
    assert dist["std_range"] == (0.3, 2.0)

    # The flat task uses a plain global Gaussian std (no curriculum to de-learn against),
    # not the rough task's heteroscedastic head — but it must still declare a distribution
    # (rsl_rl leaves self.distribution unset otherwise, crashing on the first act()).
    flat_task = TASKS.get("Genelab-Velocity-Flat-Unitree-Go1-v0")
    flat_dist = flat_task.cfg.agent.actor.distribution_cfg
    assert flat_dist is not None
    assert flat_dist["class_name"] == "GaussianDistribution"


def _build_cpu_env(cfg, num_envs: int = 4):
    """Instantiate ``cfg`` on the CPU backend with a small env count for CI speed."""
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    cfg.simulation.num_envs = num_envs
    cfg.simulation.gpu = False
    cfg.simulation.vis = False
    cfg.device = "cpu"
    return ManagerBasedRlEnv(cfg)


def test_rough_env_builds_steps_and_grows_actor_obs_by_height_scan(
    genesis_runtime: Any,
) -> None:
    """Smoke-build the complex-terrain env, step once, and confirm the terrain / obs /
    curriculum / foot-contact wiring resolves against the real simulator."""
    del genesis_runtime  # fixture only guards Genesis availability

    rough_env = _build_cpu_env(_rough_cfg(play=True))
    try:
        rough_env.reset()
        zero_actions = torch.zeros(
            rough_env.num_envs, rough_env.action_manager.total_action_dim, device=rough_env.device
        )
        rough_env.step(zero_actions)

        assert rough_env.scene.terrain is not None
        assert "height_scan" in rough_env.observation_manager.active_terms["policy"]
        assert rough_env.sensors["height_scan"].data.shape == (
            rough_env.num_envs,
            _HEIGHT_SCAN_RAYS,
        )
        assert "terrain_levels" in rough_env.curriculum_manager.active_terms
        rough_policy_dim = rough_env.observation_manager.group_obs_dim("policy")
    finally:
        rough_env.close()

    flat_env = _build_cpu_env(_flat_cfg(play=True))
    try:
        flat_env.reset()
        flat_policy_dim = flat_env.observation_manager.group_obs_dim("policy")
    finally:
        flat_env.close()

    # The only actor-obs delta vs flat is the 187-ray height scan.
    assert rough_policy_dim - flat_policy_dim == _HEIGHT_SCAN_RAYS
