"""``Genelab-Velocity-Rough-Unitree-G1-v0``: terrain + height_scan + curriculum wiring.

The cfg-level and registration checks run everywhere (no simulator). The build +
step smoke is gated on the ``genesis_runtime`` fixture so it skips cleanly on
EGL-less CI runners, mirroring ``tests/test_m4_sensor_integration.py``.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.mdp import terrain_levels_vel  # noqa: E402
from genelab.registry import ENVS, TASKS, load_extension_module  # noqa: E402
from genelab.rl import RslRlOnPolicyRunnerCfg  # noqa: E402
from genelab.sensor import GridPattern, TerrainHeightSensorCfg  # noqa: E402

ROUGH_TASK_ID = "Genelab-Velocity-Rough-Unitree-G1-v0"
ROUGH_ENV_NAME = "g1-velocity-rough-env"
# 17 x 11 = 187 rays — the torso height-scan grid the policy must see.
_HEIGHT_SCAN_RAYS = GridPattern(resolution=0.1, size=(1.6, 1.0)).num_rays()


def _flat_cfg(play: bool = True):
    from genelab_unitree.g1 import unitree_g1_velocity_env_cfg

    return unitree_g1_velocity_env_cfg(play=play)


def _rough_cfg(play: bool = True):
    from genelab_unitree.g1 import unitree_g1_velocity_rough_env_cfg

    return unitree_g1_velocity_rough_env_cfg(play=play)


def _build_cpu_env(cfg, num_envs: int = 4):
    """Instantiate ``cfg`` on the CPU backend with a small env count for CI speed."""
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    cfg.simulation.num_envs = num_envs
    cfg.simulation.gpu = False
    cfg.simulation.vis = False
    cfg.device = "cpu"
    return ManagerBasedRlEnv(cfg)


def test_rough_cfg_adds_terrain_height_scan_and_curriculum() -> None:
    rough = _rough_cfg(play=True)
    flat = _flat_cfg(play=True)

    # Rough swaps the flat ground for a 10-level (rows) x 10-column mixed heightfield.
    assert rough.scene.terrain is not None
    assert flat.scene.terrain is None
    assert (rough.scene.terrain.num_rows, rough.scene.terrain.num_cols) == (10, 10)

    # The two terrain-sensitive foot penalties are lightened on rough terrain only;
    # the flat task keeps the heavier shared weights.
    assert rough.rewards_cfg["foot_clearance"].weight == -0.5
    assert rough.rewards_cfg["foot_swing_height"].weight == -0.25
    assert flat.rewards_cfg["foot_clearance"].weight == -2.0
    assert flat.rewards_cfg["foot_swing_height"].weight == -1.0

    # height_scan observation added to the actor (and a clean copy to the critic).
    assert "height_scan" in rough.observations_cfg["policy"].terms
    assert "height_scan" not in flat.observations_cfg["policy"].terms
    assert "height_scan" in rough.observations_cfg["critic"].terms

    # The backing sensor is a 187-ray torso grid.
    sensors = {s.name: s for s in rough.scene.sensors}
    assert "height_scan" in sensors
    height_scan = sensors["height_scan"]
    assert isinstance(height_scan, TerrainHeightSensorCfg)
    assert height_scan.pattern.num_rays() == _HEIGHT_SCAN_RAYS == 187

    # terrain_levels curriculum sits alongside the inherited command_vel term.
    assert "terrain_levels" in rough.curriculum_cfg
    assert rough.curriculum_cfg["terrain_levels"].func is terrain_levels_vel
    assert rough.curriculum_cfg["terrain_levels"].params["distance_threshold"] == 4.0
    assert rough.curriculum_cfg["terrain_levels"].params["command_name"] == "twist"
    assert "command_vel" in rough.curriculum_cfg
    assert "terrain_levels" not in flat.curriculum_cfg


def test_play_launches_a_single_g1() -> None:
    # Play is an interactive single-robot teleop session, so it builds exactly one G1
    # (the ImGui twist sliders drive that one robot). Both the flat and rough velocity
    # configs share this default.
    assert _rough_cfg(play=True).simulation.num_envs == 1
    assert _flat_cfg(play=True).simulation.num_envs == 1


def test_play_wires_imgui_twist_sliders() -> None:
    # The single-robot play session exposes in-viewer ImGui sliders for (vx, vy, ωz) that
    # drive the "twist" command. Wiring is gated on imgui_bundle (the overlay dependency),
    # so skip cleanly when it's absent — the play path still works, just without sliders.
    pytest.importorskip("imgui_bundle")
    from genelab.bridges.imgui import ImGuiTwistBridgeCfg

    cfg = _rough_cfg(play=True)
    assert cfg.simulation.viewer_imgui is True
    teleop = cfg.bridges_cfg.get("teleop")
    assert isinstance(teleop, ImGuiTwistBridgeCfg)
    assert teleop.command_name == "twist"


def test_twist_command_is_direct_yaw_rate() -> None:
    # The third twist component is a directly-commanded yaw rate (vw = ωz) the policy must
    # track, not a heading target — so (vx, vy, vw) are independent commands and the teleop wz
    # slider takes effect instead of being overridden by the heading PD.
    cmd = _rough_cfg(play=False).commands_cfg["twist"]
    assert cmd.heading_command is False
    assert cmd.rel_heading_envs == 0.0


def test_play_disables_auto_reset_for_teleop() -> None:
    # During interactive teleop the robot should keep going, not snap back to spawn every 20s
    # (time-out) or on a stumble. Training keeps auto-reset so episodes roll over normally.
    assert _rough_cfg(play=True).auto_reset is False
    assert _rough_cfg(play=False).auto_reset is True


def test_training_cfg_keeps_full_env_count_and_no_teleop() -> None:
    # The single-G1 / slider wiring is a play-only affordance; training must stay a large
    # parallel rollout with no viewer overlay or teleop bridge.
    cfg = _rough_cfg(play=False)
    assert cfg.simulation.num_envs == 4096
    assert cfg.simulation.viewer_imgui is False
    assert "teleop" not in cfg.bridges_cfg


def test_rough_task_registers_and_is_trainable() -> None:
    load_extension_module("genelab_unitree.tasks")
    assert ROUGH_TASK_ID in TASKS.names()
    assert ROUGH_ENV_NAME in ENVS.names()

    task = TASKS.get(ROUGH_TASK_ID)
    assert task.cfg.trainable is True
    assert task.cfg.env is not None
    assert task.cfg.play_env is not None
    assert isinstance(task.cfg.agent, RslRlOnPolicyRunnerCfg)
    assert task.cfg.agent.max_iterations == 6_000
    # State-dependent action std with an exploration floor: this is the fix that makes
    # the curriculum trainable without de-learning (a global std over-specialises and
    # floors the adaptive LR; the floor keeps minimum exploration).
    dist = task.cfg.agent.actor.distribution_cfg
    assert dist is not None
    assert dist["class_name"] == "HeteroscedasticGaussianDistribution"
    assert dist["std_range"] == (0.3, 2.0)
    # Same GPU-backend guard the flat tasks carry; play renders, so it needs it too.
    assert task.cfg.env.simulation.gpu is True
    assert task.cfg.play_env.simulation.gpu is True
    # Play mode drops the push DR and shrinks num_envs.
    assert "push_robot" not in task.cfg.play_env.events_cfg
    assert "push_robot" in task.cfg.env.events_cfg
    assert task.cfg.play_env.simulation.num_envs <= task.cfg.env.simulation.num_envs


def test_rough_env_builds_steps_and_grows_actor_obs_by_height_scan(
    genesis_runtime: Any,
) -> None:
    """Smoke-build the rough env, step once, and confirm the terrain/obs/curriculum wiring."""
    del genesis_runtime  # fixture only guards Genesis availability

    rough_env = _build_cpu_env(_rough_cfg(play=True))
    try:
        rough_env.reset()
        zero_actions = torch.zeros(
            rough_env.num_envs, rough_env.action_manager.total_action_dim, device=rough_env.device
        )
        rough_env.step(zero_actions)

        # Rough scene is backed by a real heightfield.
        assert rough_env.scene.terrain is not None
        # The actor observation group includes the height scan ...
        assert "height_scan" in rough_env.observation_manager.active_terms["policy"]
        # ... and the live sensor returns the full 187-ray grid.
        assert rough_env.sensors["height_scan"].data.shape == (
            rough_env.num_envs,
            _HEIGHT_SCAN_RAYS,
        )
        # The terrain curriculum term is registered.
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
