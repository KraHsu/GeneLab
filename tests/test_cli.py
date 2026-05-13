from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from genelab_examples.rubiks.assets import RubiksCubeSpec, iter_cubie_coords, to_mjcf_xml
from genelab_examples.wuji_hand.assets import (
    candidate_mjcf_paths,
    load_trajectory,
    resolve_mjcf_path,
    wuji_joint_names,
)
from genelab import __doc__ as genelab_doc
from genelab.cli import main
from genelab.configs import ManagerBasedEnvCfg, apply_overrides
from genelab_examples.wuji_hand.assets import DEFAULT_DESC_DIR, DEFAULT_TRAJECTORY
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    load_entrypoint_extensions,
    load_extension_module,
)
from genelab_examples.robots import create_rubiks_robot
from genelab_examples.rubiks.sim import (
    ControlledEntityLike,
    ForceDrivenCubeConfig,
    ForceDrivenCubeController,
    RubiksCubeController,
    select_rotation_axis,
)
from genelab_examples.wuji_hand.sim import build_joint_mapping, trajectory_target

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


def test_cli_outputs_registered_hint(capsys: pytest.CaptureFixture[str]) -> None:
    main([])

    assert "genelab list tasks" in capsys.readouterr().out


def test_core_does_not_register_example_tasks_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--no-entry-points", "list", "tasks"])

    assert "GeneLab-Rubiks-Play-v0" not in capsys.readouterr().out


def test_example_extension_registers_robots_envs_and_tasks() -> None:
    load_extension_module("genelab_examples.tasks")

    assert "rubiks-cube" in ROBOTS.names()
    assert "wuji-hand" in ROBOTS.names()
    assert "rubiks-play" in ENVS.names()
    assert "wuji-hand-playback" in ENVS.names()
    assert "GeneLab-Rubiks-Play-v0" in TASKS.names()
    assert "GeneLab-Wuji-Hand-Playback-v0" in TASKS.names()


def test_list_tasks_shows_registered_bindings(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "genelab_examples.tasks", "list", "tasks"])

    out = capsys.readouterr().out
    assert "GeneLab-Rubiks-Play-v0" in out
    assert "env=rubiks-play" in out
    assert "robot=rubiks-cube" in out
    assert "GeneLab-Wuji-Hand-Playback-v0" in out


def test_package_positions_genelab_as_genesis_powered_lab_api() -> None:
    assert genelab_doc is not None
    assert "Isaac Lab API" in genelab_doc
    assert ManagerBasedEnvCfg.__name__ == "ManagerBasedEnvCfg"


def test_wuji_default_trajectory_is_bundled_with_package_assets() -> None:
    assert "genelab_examples/wuji_hand/data/wave.npy" in DEFAULT_TRAJECTORY.as_posix()
    assert DEFAULT_TRAJECTORY.exists()


def test_wuji_default_description_assets_are_bundled_with_package_assets() -> None:
    assert "genelab_examples/wuji_hand/description" in DEFAULT_DESC_DIR.as_posix()
    assert resolve_mjcf_path(DEFAULT_DESC_DIR, "left").exists()
    assert resolve_mjcf_path(DEFAULT_DESC_DIR, "right").exists()
    assert (DEFAULT_DESC_DIR / "meshes" / "left" / "left_palm_link.STL").exists()
    assert (DEFAULT_DESC_DIR / "meshes" / "right" / "right_palm_link.STL").exists()


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


def test_config_overrides_update_nested_task_config() -> None:
    load_extension_module("genelab_examples.tasks")
    task = TASKS.get("GeneLab-Wuji-Hand-Playback-v0")

    apply_overrides(
        task.cfg, {"env.robot.side": "left", "env.scene.steps": "3", "env.reset_interval": "0"}
    )

    assert task.cfg.env.robot.side == "left"
    assert task.cfg.env.scene.steps == 3
    assert task.cfg.env.reset_interval == 0


def test_cli_run_args_accept_flags_after_task() -> None:
    from genelab.cli import normalize_argv, parse_run_args

    argv = normalize_argv(
        ["play", "--steps", "5", "GeneLab-Rubiks-Play-v0", "--vis", "--env.robot.gap", "0.002"]
    )
    assert argv is not None
    task_id, overrides = parse_run_args(argv[1:])

    assert task_id == "GeneLab-Rubiks-Play-v0"
    assert overrides == {
        "env.scene.steps": "5",
        "env.scene.vis": "true",
        "env.robot.gap": "0.002",
    }


def test_cli_parses_agent_flag_value() -> None:
    from genelab.cli import parse_run_args

    task_id, overrides = parse_run_args(
        ["External-Fake-Task-v0", "--agent", "random", "--num_envs", "4"]
    )

    assert task_id == "External-Fake-Task-v0"
    assert overrides == {"agent": "random", "num_envs": "4"}


def test_cli_rejects_invalid_agent_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    from genelab import cli as cli_module

    def _fake_play(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("play_task should not run when --agent value is invalid")

    fake_rl = type("FakeRl", (), {"play_task": staticmethod(_fake_play), "AgentKind": str})
    monkeypatch.setitem(__import__("sys").modules, "genelab.rl", fake_rl)

    try:
        cli_module.main(
            ["--import", "tests.fake_extension", "play", "External-Fake-Task-v0", "--agent", "bogus"]
        )
    except SystemExit as exc:
        assert "--agent" in str(exc)
    else:
        raise AssertionError("expected invalid --agent value to exit")


def test_cli_routes_agent_through_play_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from genelab import cli as cli_module

    captured: dict[str, object] = {}

    def _fake_play(task_id: str, **kwargs: object) -> None:
        captured["task_id"] = task_id
        captured.update(kwargs)

    fake_rl = type("FakeRl", (), {"play_task": staticmethod(_fake_play), "AgentKind": str})
    monkeypatch.setitem(__import__("sys").modules, "genelab.rl", fake_rl)

    cli_module.main(
        ["--import", "tests.fake_extension", "play", "External-Fake-Task-v0", "--agent", "random"]
    )

    assert captured["task_id"] == "External-Fake-Task-v0"
    assert captured["agent"] == "random"


def test_core_namespaces_do_not_import_example_objects() -> None:
    import genelab.envs as envs
    import genelab.robots as robots
    import genelab.tasks as tasks

    assert not hasattr(robots, "create_rubiks_robot")
    assert not hasattr(envs, "create_rubiks_env")
    assert not hasattr(tasks, "rubiks_play_task_cfg")


def test_cli_import_loads_external_task(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "tests.fake_extension", "list", "tasks"])

    out = capsys.readouterr().out
    assert "External-Fake-Task-v0" in out
    assert "env=fake-extension-env" in out
    assert "robot=fake-extension-robot" in out


def test_cli_imported_external_task_can_run(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "tests.fake_extension", "play", "External-Fake-Task-v0", "--steps", "7"])

    assert "played External-Fake-Task-v0 for 7 steps" in capsys.readouterr().out


def test_project_new_creates_importable_external_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["project", "new", "sample-project", "--path", str(tmp_path)])

    project = tmp_path / "sample-project"
    package_src = project / "src" / "sample_project"
    assert (project / "pyproject.toml").exists()
    assert (package_src / "tasks.py").exists()
    assert "genelab = { path = " in (project / "pyproject.toml").read_text()

    monkeypatch.syspath_prepend(str(project / "src"))
    main(["--import", "sample_project.tasks", "list", "tasks"])

    assert "SampleProject-Example-v0" in capsys.readouterr().out


def test_entrypoint_extensions_load_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEntryPoint:
        name = "fake"
        value = "tests.fake_entrypoint_extension:register"
        dist = None

        def load(self):
            def register():
                register_task(
                    "EntryPoint-Fake-Task-v0",
                    lambda: None,
                    description="Task from a fake entry point.",
                )

            return register

    def fake_entry_points(*, group: str | None = None) -> list[_FakeEntryPoint]:
        assert group == "genelab.extensions"
        return [_FakeEntryPoint()]

    from genelab import registry
    from genelab.registry import register_task

    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        fake_entry_points,
    )  # pyright: ignore[reportUnknownArgumentType]

    load_entrypoint_extensions()

    assert "EntryPoint-Fake-Task-v0" in TASKS.names()


def test_train_registered_task_reports_unimplemented() -> None:
    try:
        main(["--import", "genelab_examples.tasks", "train", "GeneLab-Rubiks-Play-v0"])
    except SystemExit as exc:
        assert "training is not implemented" in str(exc)
    else:
        raise AssertionError("expected train to report unimplemented")


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


class _FakeWujiJoint:
    def __init__(self, dof_idx: int) -> None:
        self.dofs_idx_local = [dof_idx]


class _FakeWujiEntity:
    n_dofs = 24

    def __init__(self, names: list[str]) -> None:
        self.names = {name: i + 4 for i, name in enumerate(names)}

    def get_joint(self, name: str) -> _FakeWujiJoint:
        if name not in self.names:
            raise KeyError(name)
        return _FakeWujiJoint(self.names[name])

    def get_dofs_limit(self, dof_indices: list[int]) -> tuple[FloatArray, FloatArray]:
        n_dofs = len(dof_indices)
        return np.full(n_dofs, -1.0, dtype=np.float32), np.full(n_dofs, 1.0, dtype=np.float32)

    def set_dofs_kp(self, kp: FloatArray, dofs_idx_local: list[int]) -> None:
        _ = (kp, dofs_idx_local)

    def set_dofs_kv(self, kv: FloatArray, dofs_idx_local: list[int]) -> None:
        _ = (kv, dofs_idx_local)

    def control_dofs_position(self, position: FloatArray, dof_indices: list[int]) -> None:
        _ = (position, dof_indices)

    def set_dofs_position(
        self, position: FloatArray, dof_indices: list[int], zero_velocity: bool
    ) -> None:
        _ = (position, dof_indices, zero_velocity)


def test_wuji_hand_is_registered_for_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "genelab_examples.tasks", "list", "tasks"])

    assert "GeneLab-Wuji-Hand-Playback-v0" in capsys.readouterr().out


def test_wuji_joint_names_match_reference_order() -> None:
    names = wuji_joint_names("right")

    assert len(names) == 20
    assert names[:4] == [
        "right_finger1_joint1",
        "right_finger1_joint2",
        "right_finger1_joint3",
        "right_finger1_joint4",
    ]
    assert names[-1] == "right_finger5_joint4"


def test_wuji_mjcf_resolution_uses_flat_layout(tmp_path: Path) -> None:
    desc = tmp_path / "description"
    mjcf = desc / "mjcf" / "left.xml"
    mjcf.parent.mkdir(parents=True)
    mjcf.write_text("<mujoco />")

    assert candidate_mjcf_paths(desc, "left")[0] == mjcf
    assert resolve_mjcf_path(desc, "left") == mjcf


def test_wuji_trajectory_loader_requires_2d_array(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(20, dtype=np.float32))

    try:
        load_trajectory(path)
    except ValueError as exc:
        assert "2D trajectory" in str(exc)
    else:
        raise AssertionError("expected non-2D trajectory to fail")


def test_wuji_joint_mapping_skips_missing_joints() -> None:
    names = wuji_joint_names("right")
    entity = _FakeWujiEntity(names[:3] + names[4:])

    mapping = build_joint_mapping(entity, "right")

    assert mapping.mujoco_indices.tolist() == list(range(3)) + list(range(4, 20))
    assert mapping.dof_indices[:3] == [4, 5, 6]
    assert mapping.missing_joint_names == ["right_finger1_joint4"]


def test_wuji_trajectory_target_clips_to_dof_limits() -> None:
    mapping = build_joint_mapping(_FakeWujiEntity(wuji_joint_names("left")[:2]), "left")
    frame = np.zeros(20, dtype=np.float32)
    frame[:2] = [-1.0, 2.0]

    target = trajectory_target(
        frame,
        mapping,
        lower=np.array([-0.2, -0.5], dtype=np.float32),
        upper=np.array([0.7, 1.1], dtype=np.float32),
    )

    np.testing.assert_allclose(target, [-0.2, 1.1])
