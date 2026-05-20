"""Registration smoke test for the Franka pick-and-place SAC+HER extension."""

from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module


def test_franka_pick_and_place_extension_registers() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")

    assert "GeneLab-Franka-Pick-And-Place-v0" in TASKS.names()
    assert "franka-pick-and-place" in ROBOTS.names()
    assert "franka-pick-and-place-env" in ENVS.names()


def test_franka_pick_and_place_task_is_goal_conditioned() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")
    task = TASKS.get("GeneLab-Franka-Pick-And-Place-v0")

    from genelab.rl import Sb3AgentCfg, select_backend

    agent = task.cfg.agent
    assert isinstance(agent, Sb3AgentCfg)
    assert agent.algorithm == "SAC"
    assert agent.her.enabled is True
    assert agent.her.compute_reward is not None
    assert select_backend(agent).name == "sb3"

    # Env exposes the four observation groups SAC+HER + the asymmetric critic need.
    groups = task.cfg.env.observations_cfg
    assert "policy" in groups
    assert "critic" in groups
    assert "achieved_goal" in groups
    assert "desired_goal" in groups
    # Reward shape: sparse goal + lift bonus.
    assert set(task.cfg.env.rewards_cfg) == {"sparse_goal", "lift_bonus"}
    # Cube friction + arm/hand actuator overrides are applied.
    assert task.cfg.env.scene.entities["cube"].friction == 1.0
    assert task.cfg.env.robot.actuators["panda_arm"].stiffness == 2000.0
    assert task.cfg.env.robot.actuators["panda_hand"].velocity_limit == 1.0
