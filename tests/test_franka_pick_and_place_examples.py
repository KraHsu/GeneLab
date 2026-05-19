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
