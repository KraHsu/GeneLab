"""Registration smoke test for the Franka pick-and-place extension."""

from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module


def test_franka_pick_and_place_extension_registers() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")

    assert "GeneLab-Franka-Pick-And-Place-v0" in TASKS.names()
    assert "franka-pick-and-place" in ROBOTS.names()
    assert "franka-pick-and-place-env" in ENVS.names()


def test_franka_pick_and_place_cartesian_extension_registers() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")

    assert "GeneLab-Franka-Pick-And-Place-Cartesian-v0" in TASKS.names()
    assert "franka-pick-and-place-cartesian-env" in ENVS.names()


def test_franka_pick_and_place_skrl_task_routes_to_skrl_backend() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")

    assert "GeneLab-Franka-Pick-And-Place-skrl-v0" in TASKS.names()
    task = TASKS.get("GeneLab-Franka-Pick-And-Place-skrl-v0")

    from genelab.rl import SkrlAgentCfg, select_backend

    assert isinstance(task.cfg.agent, SkrlAgentCfg)
    assert select_backend(task.cfg.agent).name == "skrl"


def test_franka_pick_and_place_sb3_task_routes_to_sb3_backend() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")

    assert "GeneLab-Franka-Pick-And-Place-sb3-v0" in TASKS.names()
    task = TASKS.get("GeneLab-Franka-Pick-And-Place-sb3-v0")

    from genelab.rl import Sb3AgentCfg, select_backend

    assert isinstance(task.cfg.agent, Sb3AgentCfg)
    assert select_backend(task.cfg.agent).name == "sb3"


def test_franka_pick_and_place_sb3_her_task_is_goal_conditioned() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")

    assert "GeneLab-Franka-Pick-And-Place-sb3-her-v0" in TASKS.names()
    task = TASKS.get("GeneLab-Franka-Pick-And-Place-sb3-her-v0")

    from genelab.rl import Sb3AgentCfg

    agent = task.cfg.agent
    assert isinstance(agent, Sb3AgentCfg)
    assert agent.algorithm == "SAC"
    assert agent.her.enabled is True
    assert agent.her.compute_reward is not None
    # The HER env exposes the goal-conditioned observation groups.
    groups = task.cfg.env.observations_cfg
    assert "achieved_goal" in groups
    assert "desired_goal" in groups
