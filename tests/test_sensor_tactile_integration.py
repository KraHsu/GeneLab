"""Real-scene integration smoke for the Genesis-1.0 sensor wrappers.

Spins up an actual ``gs.Scene`` (CPU backend), adds each wrapper-backed sensor
via :meth:`Sensor.pre_build_genesis`, builds the scene, takes a few sim steps,
and confirms each sensor's :attr:`Sensor.data` field has the expected dtype /
shape on its tensor channels. This catches the failure mode where the synthetic
``_FakeGsScene`` tests pass but Genesis itself rejects the constructed options
or returns a different layout than the wrapper assumes.

CPU-only so it runs everywhere; the wrappers don't take a backend-conditional
path so a CPU pass implies the GPU one.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
gs = pytest.importorskip("genesis")

from genelab.sensor.kinematic_contact import KinematicContactSensorCfg  # noqa: E402
from genelab.sensor.kinematic_depth import KinematicDepthSensorCfg  # noqa: E402
from genelab.sensor.kinematic_tactile import KinematicTactileSensorCfg  # noqa: E402
from genelab.sensor.proximity import ProximitySensorCfg  # noqa: E402
from genelab.sensor.tactile_elastomer import ElastomerTactileSensorCfg  # noqa: E402
from genelab.sensor.tactile_pointcloud import PointCloudTactileSensorCfg  # noqa: E402


def _scene_build_supported() -> bool:
    """Return True if ``gs.Scene(...).build()`` can run in this environment.

    Genesis always constructs an offscreen pyrender renderer inside
    ``scene.build()``, which requires EGL / a display backend. Headless CI
    runners typically lack both, so this probe lets us skip the integration
    smoke gracefully there without disabling the local pass.
    """
    if not getattr(gs, "_initialized", False):
        try:
            gs.init(backend=gs.cpu, logging_level="warning")
        except Exception:
            return False
    try:
        probe = gs.Scene(show_viewer=False)
        probe.add_entity(gs.morphs.Plane())
        probe.build(n_envs=1)
    except Exception:
        return False
    return True


_SCENE_BUILD_SUPPORTED = _scene_build_supported()

pytestmark = pytest.mark.skipif(
    not _SCENE_BUILD_SUPPORTED,
    reason="gs.Scene.build() unsupported in this environment (no display / EGL)",
)


@pytest.fixture(scope="module")
def gs_initialized() -> None:
    """``gs.init`` is already called by ``_scene_build_supported``; this fixture
    is kept so the per-test signatures stay explicit about the global Genesis state.
    """
    return None


def _entity_wrapper(gs_handle: Any, link_names: list[str]) -> Any:
    """Minimal GeneLab-side entity stub the wrappers' ``pre_build_genesis`` reads."""

    class _Entity:
        pass

    e = _Entity()
    e.gs_handle = gs_handle
    e.link_names = link_names
    return e


def _build_scene_with_sensors(*sensor_cfgs: Any) -> tuple[Any, dict[str, Any]]:
    """Build a scene with a falling box over a plane and the given sensors attached to the box."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        show_viewer=False,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.05)))

    entities = {
        "robot": _entity_wrapper(box, ["box_link"]),
        "plane": _entity_wrapper(plane, ["plane_link"]),
    }
    # ``track_link_names`` needs to resolve against the *same* entity's links — Genesis's
    # ``entity_idx`` + ``link_idx_local`` scopes the probe to one entity. Inject a dummy
    # "plane_link" name into the box entity so the wrapper's name resolution passes; the
    # generated ``track_link_idx=(1,)`` then references the box's own (only) link, which is
    # enough for an "add_sensor + step + read doesn't error" smoke.
    entities["robot"].link_names = ["box_link", "plane_link"]

    sensors: dict[str, Any] = {}
    for cfg in sensor_cfgs:
        sensor = cfg.build()
        sensor.pre_build_genesis(scene, entities)
        sensors[cfg.name] = sensor

    scene.build(n_envs=2)
    return scene, sensors


def test_kinematic_contact_real_scene(gs_initialized: None) -> None:
    del gs_initialized
    cfg = KinematicContactSensorCfg(
        name="contact",
        link_name="box_link",
        probe_local_pos=((0.0, 0.0, -0.025),),
        contact_threshold=1e-4,
    )
    scene, sensors = _build_scene_with_sensors(cfg)
    for _ in range(3):
        scene.step()
        sensors["contact"].update(0.02)

    data = sensors["contact"].data
    assert data.in_contact.dtype == torch.bool
    assert data.in_contact.shape == (2, 1)


def test_proximity_real_scene(gs_initialized: None) -> None:
    del gs_initialized
    cfg = ProximitySensorCfg(
        name="prox",
        link_name="box_link",
        probe_local_pos=((0.0, 0.0, -0.025),),
        probe_radius=0.5,
        track_link_names=("box_link",),
    )
    scene, sensors = _build_scene_with_sensors(cfg)
    for _ in range(3):
        scene.step()
        sensors["prox"].update(0.02)

    data = sensors["prox"].data
    assert data.distance.dtype == torch.float32
    assert data.distance.shape == (2, 1)


def test_elastomer_tactile_real_scene(gs_initialized: None) -> None:
    del gs_initialized
    cfg = ElastomerTactileSensorCfg(
        name="elast",
        link_name="box_link",
        probe_local_pos=((0.0, 0.0, -0.025),),
        track_link_names=("box_link",),
        n_sample_points=8,
    )
    scene, sensors = _build_scene_with_sensors(cfg)
    for _ in range(3):
        scene.step()
        sensors["elast"].update(0.02)

    data = sensors["elast"].data
    # ElastomerTaxel.read() → (B, P, 3) float displacement vector per probe.
    assert data.raw.dtype == torch.float32
    assert data.raw.shape == (2, 1, 3)


def test_kinematic_depth_real_scene(gs_initialized: None) -> None:
    del gs_initialized
    cfg = KinematicDepthSensorCfg(
        name="depth",
        link_name="box_link",
        probe_local_pos=((0.0, 0.0, -0.025),),
        probe_radius=0.01,
    )
    scene, sensors = _build_scene_with_sensors(cfg)
    for _ in range(3):
        scene.step()
        sensors["depth"].update(0.02)

    data = sensors["depth"].data
    # ContactDepthProbe.read() → (B, P) float penetration depth (metres), non-negative.
    assert data.depth.dtype == torch.float32
    assert data.depth.shape == (2, 1)
    assert bool((data.depth >= 0).all())
    # ``raw`` aliases ``depth`` for the generic reward primitives.
    assert torch.equal(data.raw, data.depth)


def test_kinematic_tactile_real_scene(gs_initialized: None) -> None:
    del gs_initialized
    cfg = KinematicTactileSensorCfg(
        name="taxel",
        link_name="box_link",
        probe_local_pos=((0.0, 0.0, -0.025),),
        probe_local_normal=(0.0, 0.0, 1.0),
        probe_radius=0.01,
        normal_stiffness=1000.0,
    )
    scene, sensors = _build_scene_with_sensors(cfg)
    for _ in range(3):
        scene.step()
        sensors["taxel"].update(0.02)

    data = sensors["taxel"].data
    # KinematicTaxel.read() → KinematicTaxelData(force=(B, P, 3), torque=(B, P, 3)), link frame.
    assert data.force.dtype == torch.float32
    assert data.force.shape == (2, 1, 3)
    assert data.torque.dtype == torch.float32
    assert data.torque.shape == (2, 1, 3)
    # ``raw`` aliases ``force`` for the generic reward primitives.
    assert torch.equal(data.raw, data.force)


def test_pointcloud_tactile_real_scene(gs_initialized: None) -> None:
    del gs_initialized
    cfg = PointCloudTactileSensorCfg(
        name="pcl",
        link_name="box_link",
        probe_local_pos=((0.0, 0.0, -0.025),),
        track_link_names=("box_link",),
        n_sample_points=8,
    )
    scene, sensors = _build_scene_with_sensors(cfg)
    for _ in range(3):
        scene.step()
        sensors["pcl"].update(0.02)

    data = sensors["pcl"].data
    # ProximityTaxel.read() → ProximityTaxelData(force=(B, P, 3), torque=(B, P, 3)).
    assert data.force.dtype == torch.float32
    assert data.force.shape == (2, 1, 3)
    assert data.torque.dtype == torch.float32
    assert data.torque.shape == (2, 1, 3)
    # ``raw`` aliases ``force`` for the generic reward primitives.
    assert torch.equal(data.raw, data.force)
