"""Tests for the bundled Rubik's-cube example: MJCF generation, the registry
robot, and the kinematic / force-driven cube controllers.

These exercise ``genelab_examples`` robot logic (not the CLI) — they were split
out of ``tests/test_cli.py`` to keep that module focused on the CLI surface.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from genelab.configs import apply_overrides
from genelab.registry import TASKS, load_extension_module
from genelab_examples.robots import create_rubiks_robot
from genelab_examples.rubiks.assets import RubiksCubeSpec, iter_cubie_coords, to_mjcf_xml
from genelab_examples.rubiks.sim import (
    ControlledEntityLike,
    ForceDrivenCubeConfig,
    ForceDrivenCubeController,
    RubiksCubeController,
    select_rotation_axis,
)

type FloatArray = NDArray[np.floating]


class _FakeLink:
    def __init__(self, q_start: int, idx: int | None = None) -> None:
        self.q_start = q_start
        self.dof_start = (q_start // 7) * 6
        self.idx = q_start // 7 if idx is None else idx


class _FakeQpos:
    def __init__(self, array: FloatArray) -> None:
        self.array = array

    def detach(self) -> "_FakeQpos":
        return self

    def cpu(self) -> "_FakeQpos":
        return self

    def numpy(self) -> FloatArray:
        return self.array


class _FakeTensor:
    def __init__(self, array: object) -> None:
        self.array = np.asarray(array, dtype=float)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> FloatArray:
        return self.array


class _FakeEntity:
    q_start = 0
    dof_start = 0

    def __init__(self, spec: RubiksCubeSpec) -> None:
        self._links = [_FakeLink(i * 7) for i in range(27)]
        self.qpos = np.zeros(27 * 7, dtype=float)
        self.dofs_velocity = np.zeros(27 * 6, dtype=float)
        for i, coord in enumerate(iter_cubie_coords()):
            self.qpos[i * 7 : i * 7 + 3] = np.asarray(coord, dtype=float) * spec.spacing
            self.qpos[i * 7 + 3 : i * 7 + 7] = (1.0, 0.0, 0.0, 0.0)

    @property
    def links(self) -> Sequence[_FakeLink]:
        return self._links

    def get_qpos(self) -> _FakeQpos:
        return _FakeQpos(self.qpos.copy())

    def set_qpos(self, qpos: FloatArray, zero_velocity: bool = True) -> None:
        _ = zero_velocity
        self.qpos = np.asarray(qpos, dtype=float).copy()

    def get_links_pos(self, ref: str = "link_origin") -> _FakeTensor:
        _ = ref
        return _FakeTensor(self.qpos.reshape(27, 7)[:, :3].copy())

    def get_dofs_velocity(self) -> _FakeTensor:
        return _FakeTensor(self.dofs_velocity.copy())

    def set_dofs_velocity(self, velocity: FloatArray, skip_forward: bool = False) -> None:
        _ = skip_forward
        self.dofs_velocity = np.asarray(velocity, dtype=float).copy()


class _FakeSolver:
    def __init__(self) -> None:
        self.welds: list[tuple[int, int]] = []
        self.hinges: list[tuple[int, int, tuple[float, ...], tuple[float, ...], float]] = []
        self.torques: list[tuple[FloatArray, tuple[int, ...], str]] = []

    def add_weld_constraint(self, link_a: int, link_b: int) -> None:
        pair = (int(link_a), int(link_b))
        assert pair not in self.welds
        self.welds.append(pair)

    def delete_weld_constraint(self, link_a: int, link_b: int) -> None:
        self.welds.remove((int(link_a), int(link_b)))

    def add_hinge_constraint(
        self,
        link_a: int,
        link_b: int,
        anchor: FloatArray,
        axis: FloatArray,
        span: float = 0.1,
    ) -> None:
        self.hinges.append((int(link_a), int(link_b), tuple(anchor), tuple(axis), float(span)))

    def delete_hinge_constraint(self, link_a: int, link_b: int) -> None:
        for i, hinge in enumerate(self.hinges):
            if hinge[:2] == (int(link_a), int(link_b)):
                del self.hinges[i]
                return
        raise ValueError("hinge not found")

    def apply_links_external_torque(
        self,
        torque: object,
        links_idx: list[int] | None = None,
        ref: str = "link_com",
    ) -> None:
        self.torques.append((np.asarray(torque, dtype=float), tuple(links_idx or ()), ref))


class _ForceFakeEntity(_FakeEntity):
    def __init__(self, spec: RubiksCubeSpec, solver: _FakeSolver) -> None:
        super().__init__(spec)
        self._solver = solver
        self.ang = np.zeros((27, 3), dtype=float)
        self.quat = np.zeros((27, 4), dtype=float)
        self.quat[:, 0] = 1.0

    @property
    def solver(self) -> _FakeSolver:
        return self._solver

    def get_links_ang(self) -> _FakeTensor:
        return _FakeTensor(self.ang.copy())

    def get_links_quat(self) -> _FakeTensor:
        return _FakeTensor(self.quat.copy())


def _controlled(entity: _ForceFakeEntity) -> ControlledEntityLike:
    return entity


def test_rubiks_robot_registry_writes_turnable_mjcf(tmp_path: Path) -> None:
    robot = create_rubiks_robot()
    output = robot.write_asset(tmp_path / "rubiks.xml")

    text = output.read_text()
    assert text.count('<body name="cubie_') == 27
    assert text.count("_sticker_") == 54
    assert "<equality>" not in text


def test_rubiks_robot_registry_can_write_welded_mjcf(tmp_path: Path) -> None:
    load_extension_module("genelab_examples.tasks")
    task = TASKS.get("GeneLab-Rubiks-Play-v0")
    apply_overrides(task.cfg, {"env.robot.welded": "true"})
    robot = create_rubiks_robot(task.cfg.env.robot)

    output = robot.write_asset(tmp_path / "rubiks_welded.xml")

    assert "<equality>" in output.read_text()


def test_controller_turn_updates_layer_mapping() -> None:
    spec = RubiksCubeSpec(gap=0.0)
    entity = _FakeEntity(spec)
    controller = RubiksCubeController(entity, spec)

    controller.queue_turn(axis="z", layer="+", turns=1, steps=3)
    while controller.is_turning:
        controller.step()

    moved_link = controller.link_by_coord[(-1, 1, 1)]
    q = entity.qpos[moved_link * 7 : moved_link * 7 + 7]
    np.testing.assert_allclose(q[:3], (-spec.spacing, spec.spacing, spec.spacing), atol=1e-7)
    np.testing.assert_allclose(q[3:], (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), atol=1e-7)


def test_select_rotation_axis_uses_dominant_thresholded_component() -> None:
    assert select_rotation_axis(np.array([0.2, -1.4, 0.9]), threshold=1.0) == 1
    assert select_rotation_axis(np.array([0.2, -0.8, 0.9]), threshold=1.0) is None


def test_force_controller_starts_solid_and_groups_each_axis() -> None:
    spec = RubiksCubeSpec(gap=0.0)
    solver = _FakeSolver()
    entity = _ForceFakeEntity(spec, solver)
    controller = ForceDrivenCubeController(_controlled(entity), spec=spec)

    assert controller.mode == "solid"
    assert len(solver.welds) == 54
    for link_a, link_b in solver.welds:
        coord_a = np.asarray(controller.coord_by_link[link_a])
        coord_b = np.asarray(controller.coord_by_link[link_b])
        np.testing.assert_allclose(np.abs(coord_b - coord_a).sum(), 1)
    for axis in range(3):
        assert [len(controller.layer_link_indices(axis, layer)) for layer in (-1, 0, 1)] == [
            9,
            9,
            9,
        ]


def test_force_controller_switches_to_jointed_and_back_when_settled() -> None:
    spec = RubiksCubeSpec(gap=0.0)
    solver = _FakeSolver()
    entity = _ForceFakeEntity(spec, solver)
    controller = ForceDrivenCubeController(
        _controlled(entity),
        spec=spec,
        config=ForceDrivenCubeConfig(
            enter_ang_vel=1.0,
            exit_angle=0.08,
            exit_ang_vel=0.25,
            settle_steps=2,
            joint_spring=0.0,
            joint_damping=0.0,
        ),
    )

    entity.ang[:, 2] = 1.2
    controller.step()

    assert controller.mode == "jointed"
    assert controller.active_axis == 2
    assert len(solver.welds) == 36
    for link_a, link_b in solver.welds:
        assert controller.coord_by_link[link_a][2] == controller.coord_by_link[link_b][2]
    assert len(solver.hinges) == 2

    entity.ang[:] = 0.0
    controller.step()
    controller.step()

    assert controller.mode == "solid"
    assert len(solver.welds) == 54
    assert len(solver.hinges) == 0


def test_force_config_rejects_negative_separation_error() -> None:
    try:
        ForceDrivenCubeConfig(max_separation_error=-1.0)
    except ValueError as exc:
        assert "max_separation_error" in str(exc)
    else:
        raise AssertionError("expected negative max_separation_error to fail")


def test_force_config_rejects_negative_turn_gain() -> None:
    try:
        ForceDrivenCubeConfig(turn_gain=-1.0)
    except ValueError as exc:
        assert "turn_gain" in str(exc)
    else:
        raise AssertionError("expected negative turn_gain to fail")


def test_force_controller_projects_separated_cubies_before_rewelding() -> None:
    spec = RubiksCubeSpec(gap=0.0)
    solver = _FakeSolver()
    entity = _ForceFakeEntity(spec, solver)
    controller = ForceDrivenCubeController(
        _controlled(entity),
        spec=spec,
        config=ForceDrivenCubeConfig(
            enter_ang_vel=1.0,
            exit_angle=0.08,
            exit_ang_vel=0.25,
            settle_steps=1,
            joint_spring=0.0,
            joint_damping=0.0,
            max_separation_error=0.01,
        ),
    )

    entity.ang[:, 2] = 1.2
    controller.step()
    entity.ang[:] = 0.0
    separated = controller.link_by_coord[(1, 1, 1)]
    entity.qpos[separated * 7] += 0.03
    controller.step()

    assert controller.mode == "solid"
    assert len(solver.welds) == 54
    assert len(solver.hinges) == 0


def test_force_controller_projects_solid_velocity_to_one_rigid_body() -> None:
    spec = RubiksCubeSpec(gap=0.0)
    solver = _FakeSolver()
    entity = _ForceFakeEntity(spec, solver)
    controller = ForceDrivenCubeController(_controlled(entity), spec=spec)

    picked = controller.link_by_coord[(1, 1, 1)]
    entity.dofs_velocity[picked * 6 : picked * 6 + 3] = (0.0, 0.0, 9.0)
    controller.step()

    velocities = entity.dofs_velocity.reshape(27, 6)
    np.testing.assert_allclose(velocities[:, 2].mean(), 9.0 / 27.0, atol=1e-7)
    assert velocities[:, 2].max() < 1.0
    np.testing.assert_allclose(
        velocities[:, 3:], np.repeat(velocities[0:1, 3:], 27, axis=0), atol=1e-7
    )


def test_generated_mjcf_disables_cubie_self_collision_masks() -> None:
    spec = RubiksCubeSpec()
    text = to_mjcf_xml(spec)

    assert 'contype="1"' in text
    assert 'conaffinity="0"' in text
