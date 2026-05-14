"""Acceptance test for the M1 :class:`InteractiveScene` + entity wrappers.

Builds a scene that mixes two articulations and one rigid box, steps it for 100 frames, and
checks the public ``keys`` / ``articulations`` / ``rigid_objects`` surfaces. ``importorskip``
gates the test on Genesis availability so the rest of the suite still passes on machines
without it.
"""

import pytest

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import ArticulationCfg, RigidObjectCfg
from genelab.scene import InteractiveScene


def test_interactive_scene_two_articulations_and_rigid_object() -> None:
    pytest.importorskip("genesis")
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
