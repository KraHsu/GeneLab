"""Registration smoke test for the inverted-pendulum extension."""

from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module


def test_inverted_pendulum_extension_registers() -> None:
    load_extension_module("genelab_inverted_pendulum.tasks")

    assert "GeneLab-Inverted-Pendulum-v0" in TASKS.names()
    assert "GeneLab-Double-Inverted-Pendulum-v0" in TASKS.names()

    assert "inverted-pendulum" in ROBOTS.names()
    assert "double-inverted-pendulum" in ROBOTS.names()

    assert "inverted-pendulum-env" in ENVS.names()
    assert "double-inverted-pendulum-env" in ENVS.names()
