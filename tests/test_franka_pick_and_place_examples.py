"""Registration smoke test for the Franka pick-and-place extension."""

from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module


def test_franka_pick_and_place_extension_registers() -> None:
    load_extension_module("genelab_franka_pick_and_place.tasks")

    assert "GeneLab-Franka-Pick-And-Place-v0" in TASKS.names()
    assert "franka-pick-and-place" in ROBOTS.names()
    assert "franka-pick-and-place-env" in ENVS.names()
