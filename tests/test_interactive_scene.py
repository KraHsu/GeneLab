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


def test_camera_framing_forwarded_to_viewer_options() -> None:
    # A configured camera pose frames the viewer (so zoom pivots on the subject); when
    # unset, no camera keys are forwarded and Genesis keeps its own defaults.
    from genelab.scene.interactive_scene import _viewer_option_kwargs

    framed = SimulationCfg(vis=True, camera_pos=(2.0, 0.0, 1.0), camera_lookat=(0.0, 0.0, 0.8))
    kwargs = _viewer_option_kwargs(framed, enable_gui=False)
    assert kwargs["camera_pos"] == (2.0, 0.0, 1.0)
    assert kwargs["camera_lookat"] == (0.0, 0.0, 0.8)

    bare = _viewer_option_kwargs(SimulationCfg(vis=True), enable_gui=True)
    assert "camera_pos" not in bare
    assert "camera_lookat" not in bare
    assert bare["enable_gui"] is True


def test_fly_camera_disables_genesis_default_keybinds() -> None:
    # Genesis' default controls bind W/A/S/D (world-frame, camera-rotation, save-image, wireframe)
    # — which collide with fly-mode WASD whenever the ImGui overlay is off (with the overlay on,
    # Genesis already disables them). Forcing them off when fly_camera is on makes fly own the
    # keyboard regardless of the overlay; when fly_camera is off, Genesis keeps its defaults.
    from genelab.scene.interactive_scene import _viewer_option_kwargs

    flying = _viewer_option_kwargs(SimulationCfg(vis=True, fly_camera=True), enable_gui=False)
    assert flying["enable_default_keybinds"] is False

    no_fly = _viewer_option_kwargs(SimulationCfg(vis=True, fly_camera=False), enable_gui=False)
    assert "enable_default_keybinds" not in no_fly


def test_terrain_scene_collapses_env_grid(genesis_runtime: Any) -> None:
    """Regression: a height-field terrain is one shared mesh spanning all envs, with per-env
    spawn origins spread across it. Genesis's ``env_spacing`` grid replicates+offsets the
    whole scene per env, so on top of a terrain it renders N overlapping terrain copies with
    robots buried in the neighbours. ``build()`` must collapse the grid to zero when a
    terrain is present — so the parallel-env offsets are all identical (no spread).
    """
    del genesis_runtime
    from genelab.terrains import FlatPatchCfg, TerrainGeneratorCfg

    sim_cfg = SimulationCfg(num_envs=4, dt=0.01, substeps=1, vis=False, gpu=False)
    terrain = TerrainGeneratorCfg(
        num_rows=2, num_cols=2, subterrain_size=(4.0, 4.0), sub_terrains={"flat": FlatPatchCfg()}
    )
    # A non-zero configured spacing that the terrain must override to (0, 0).
    scene_cfg = InteractiveSceneCfg(
        env_spacing=(2.5, 2.5),
        terrain=terrain,
        entities={
            "obstacle": RigidObjectCfg(morph="box", size=(0.2, 0.2, 0.2), init_pos=(0.0, 0.0, 1.0)),
        },
    )
    scene = InteractiveScene(sim_cfg, scene_cfg, device_hint="cpu")
    try:
        scene.build()
        origins = scene.env_origins  # (num_envs, 3); reads Genesis' per-env offset
        spread = (origins - origins[0]).abs().max().item()
        assert spread < 1e-6, (
            f"terrain scene grid-offset its envs by up to {spread:.3f} m — env_spacing "
            f"was not collapsed, so the terrain is replicated/overlapping per env"
        )
    finally:
        scene.close()
