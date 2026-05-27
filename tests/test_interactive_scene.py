"""Acceptance test for the M1 :class:`InteractiveScene` + entity wrappers.

Builds a scene that mixes two articulations and one rigid box, steps it for 100 frames, and
checks the public ``keys`` / ``articulations`` / ``rigid_objects`` surfaces. ``importorskip``
gates the test on Genesis availability so the rest of the suite still passes on machines
without it.
"""

from typing import Any

import pytest

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import ArticulationCfg, RigidObjectCfg
from genelab.scene import InteractiveScene
from genelab.scene.interactive_scene import _resolve_use_gpu


@pytest.mark.parametrize(
    ("gpu", "device", "expected"),
    [
        # ``None`` follows the device — the default that removes the CPU-backend footgun.
        (None, "cuda", True),
        (None, "cuda:0", True),
        (None, "cpu", False),
        # Explicit flags override the device.
        (True, "cpu", True),
        (False, "cuda", False),
    ],
)
def test_resolve_use_gpu(gpu: bool | None, device: str, expected: bool) -> None:
    assert _resolve_use_gpu(gpu, device) is expected


def test_simulation_cfg_gpu_defaults_to_none() -> None:
    """Default ``gpu=None`` => backend follows ``device`` (no silent CPU fallback)."""
    assert SimulationCfg().gpu is None


def test_interactive_scene_two_articulations_and_rigid_object(genesis_runtime: Any) -> None:
    del genesis_runtime  # fixture guards EGL/runtime availability; module obj unused
    from genelab_inverted_pendulum.single.robot import get_inverted_pendulum_robot_cfg

    ip_path = get_inverted_pendulum_robot_cfg().to_entity_cfg().mjcf_path
    pendulum_actuators = {
        "all": ImplicitPDActuatorCfg(
            target_names_expr=(".*",), stiffness=0.0, damping=0.0, action_scale=0.0
        ),
    }

    sim_cfg = SimulationCfg(num_envs=1, dt=0.01, substeps=1, vis=False, gpu=False)
    scene_cfg = InteractiveSceneCfg(
        env_spacing=(2.0, 2.0),
        entities={
            "pendulum_a": ArticulationCfg(mjcf_path=ip_path, actuators=pendulum_actuators),
            "pendulum_b": ArticulationCfg(
                mjcf_path=ip_path, init_pos=(1.0, 0.0, 0.5), actuators=pendulum_actuators
            ),
            "obstacle": RigidObjectCfg(morph="box", size=(0.2, 0.2, 0.2), init_pos=(0.5, 0.5, 0.1)),
        },
    )
    scene = InteractiveScene(sim_cfg, scene_cfg, device_hint="cpu")
    try:
        scene.build()
        assert set(scene.keys()) == {"pendulum_a", "pendulum_b", "obstacle"}
        assert set(scene.articulations.keys()) == {"pendulum_a", "pendulum_b"}
        assert set(scene.rigid_objects.keys()) == {"obstacle"}
        for _ in range(100):
            scene.step()
            scene.refresh_state()
        # After stepping, articulation root state has been refreshed.
        rs_a = scene.articulations["pendulum_a"].data
        assert rs_a.root_pos.shape[-1] == 3
    finally:
        scene.close()
