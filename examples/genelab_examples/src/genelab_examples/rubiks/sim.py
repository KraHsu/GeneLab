"""Rubik's-cube controllers and simulation helpers."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from genelab_examples.rubiks.assets import (
    Axis,
    Coord,
    Layer,
    RubiksCubeSpec,
    cubie_center,
    iter_cubie_coords,
)

type FloatArray = NDArray[np.floating]
type AnyArray = NDArray[np.generic]
type ArrayLike = object

_AXIS_NAMES = ("x", "y", "z")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}
_LAYER_INDEX = {
    "-": -1,
    "n": -1,
    "negative": -1,
    "0": 0,
    "c": 0,
    "center": 0,
    "+": 1,
    "p": 1,
    "positive": 1,
}


class _ArrayProvider(Protocol):
    def numpy(self) -> AnyArray: ...


class _CpuTensorLike(Protocol):
    def cpu(self) -> _ArrayProvider: ...


class _TensorLike(Protocol):
    def detach(self) -> _CpuTensorLike: ...


class _LinkLike(Protocol):
    q_start: int
    dof_start: int
    idx: int


class _SolverLike(Protocol):
    def add_weld_constraint(self, link_a: int, link_b: int) -> None: ...

    def delete_weld_constraint(self, link_a: int, link_b: int) -> None: ...

    def add_hinge_constraint(
        self,
        link_a: int,
        link_b: int,
        anchor: FloatArray,
        axis: FloatArray,
        span: float,
    ) -> None: ...

    def delete_hinge_constraint(self, link_a: int, link_b: int) -> None: ...

    def apply_links_external_torque(
        self,
        torque: object,
        links_idx: list[int],
        ref: str,
    ) -> None: ...


class _SimLike(Protocol):
    rigid_solver: _SolverLike


class ControllerSceneLike(Protocol):
    sim: _SimLike

    def step(self) -> None: ...


class ControlledEntityLike(Protocol):
    q_start: int
    dof_start: int

    @property
    def solver(self) -> _SolverLike: ...

    @property
    def links(self) -> Sequence[_LinkLike]: ...

    def get_links_ang(self) -> ArrayLike: ...

    def get_links_quat(self) -> ArrayLike: ...

    def get_links_pos(self, ref: str = "link_origin") -> ArrayLike: ...

    def get_qpos(self) -> ArrayLike: ...

    def set_qpos(self, qpos: FloatArray, zero_velocity: bool) -> None: ...

    def get_dofs_velocity(self) -> ArrayLike: ...

    def set_dofs_velocity(self, velocity: FloatArray, skip_forward: bool) -> None: ...


class _KinematicEntityLike(Protocol):
    q_start: int

    @property
    def links(self) -> Sequence[_LinkLike]: ...

    def get_qpos(self) -> ArrayLike: ...

    def set_qpos(self, qpos: FloatArray, zero_velocity: bool) -> None: ...


@dataclass(frozen=True)
class LayerTurn:
    """A queued quarter-turn request for ``RubiksCubeController``."""

    axis: int
    layer: int
    turns: int = 1
    steps: int = 30


@dataclass(frozen=True)
class ForceDrivenCubeConfig:
    """Thresholds and gains for physically switching cube articulation modes."""

    enter_ang_vel: float = 1.0
    exit_angle: float = 0.08
    exit_ang_vel: float = 0.25
    settle_steps: int = 10
    joint_spring: float = 0.08
    joint_damping: float = 0.012
    turn_gain: float = 0.12
    hinge_span: float | None = None
    max_separation_error: float = 0.01
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.enter_ang_vel <= 0:
            raise ValueError("enter_ang_vel must be positive")
        if self.exit_angle < 0:
            raise ValueError("exit_angle must be non-negative")
        if self.exit_ang_vel < 0:
            raise ValueError("exit_ang_vel must be non-negative")
        if self.settle_steps < 1:
            raise ValueError("settle_steps must be at least 1")
        if self.joint_spring < 0:
            raise ValueError("joint_spring must be non-negative")
        if self.joint_damping < 0:
            raise ValueError("joint_damping must be non-negative")
        if self.turn_gain < 0:
            raise ValueError("turn_gain must be non-negative")
        if self.hinge_span is not None and self.hinge_span <= 0:
            raise ValueError("hinge_span must be positive")
        if self.max_separation_error < 0:
            raise ValueError("max_separation_error must be non-negative")


@dataclass
class _JointedMode:
    axis: int
    direction: int
    steps: int = 0
    settled_steps: int = 0


class ForceDrivenCubeController:
    """Switch a cube between whole-body and three-link physical articulation.

    The cube starts as 27 free cubies that are dynamically welded into one rigid
    body. Once the welded body has enough angular velocity about any world axis,
    it is re-constrained into three slabs along that axis. Each slab remains a
    rigid body, and the side slabs are connected to the center slab by hinge-like
    pairs of point constraints. No layer pose is animated with ``set_qpos``.
    """

    def __init__(
        self,
        entity: ControlledEntityLike,
        scene: ControllerSceneLike | None = None,
        spec: RubiksCubeSpec | None = None,
        config: ForceDrivenCubeConfig | None = None,
    ) -> None:
        self.entity = entity
        self.scene = scene
        self.spec = spec or RubiksCubeSpec()
        self.config = config or ForceDrivenCubeConfig()
        if not self.spec.include_hidden_core:
            raise ValueError("ForceDrivenCubeController requires include_hidden_core=True")

        self.coords_by_link: list[Coord] = list(iter_cubie_coords(self.spec.include_hidden_core))
        self.link_by_coord: dict[Coord, int] = {
            coord: i for i, coord in enumerate(self.coords_by_link)
        }
        self.coord_by_link: dict[int, Coord] = dict(enumerate(self.coords_by_link))
        self.mode: str = "solid"
        self._jointed: _JointedMode | None = None
        self._solid_welds: list[tuple[int, int]] = []
        self._group_welds: list[tuple[int, int]] = []
        self._hinge_pairs: list[tuple[int, int]] = []

        self._enter_solid_mode()

    @property
    def active_axis(self) -> int | None:
        return None if self._jointed is None else self._jointed.axis

    @property
    def joint_angles(self) -> tuple[float, float]:
        if self._jointed is None:
            return (0.0, 0.0)
        return self._joint_angles(self._jointed.axis)

    def layer_link_indices(self, axis: int, layer: int) -> list[int]:
        return self._layer_link_indices(axis, layer)

    def global_link(self, link_i: int) -> int:
        return self._global_link(link_i)

    def solid_angular_velocity(self) -> FloatArray:
        return self._solid_angular_velocity()

    def step(self) -> str:
        """Update constraints and forces for one simulation step."""

        self._project_to_allowed_motion()
        if self._jointed is None:
            angular_velocity = self._solid_angular_velocity()
            axis = select_rotation_axis(angular_velocity, self.config.enter_ang_vel)
            if axis is not None:
                direction = 1 if angular_velocity[axis] >= 0 else -1
                self._enter_jointed_mode(axis, direction)
            return self.mode

        self._jointed.steps += 1
        angles = self._joint_angles(self._jointed.axis)
        rel_vels = self._joint_relative_velocities(self._jointed.axis)
        self._apply_joint_restoring_torques(self._jointed.axis, angles, rel_vels)

        angle_velocity_settled = (
            max(abs(angle) for angle in angles) <= self.config.exit_angle
            and max(abs(vel) for vel in rel_vels) <= self.config.exit_ang_vel
        )
        lattice_error = self._max_lattice_error()
        lattice_settled = lattice_error <= self.config.max_separation_error
        if angle_velocity_settled and not lattice_settled:
            self._log(
                "waiting to re-weld; "
                f"lattice error {lattice_error:.4g} exceeds {self.config.max_separation_error:.4g}"
            )
        settled = angle_velocity_settled and lattice_settled
        self._jointed.settled_steps = self._jointed.settled_steps + 1 if settled else 0
        if self._jointed.settled_steps >= self.config.settle_steps:
            self._enter_solid_mode()
        return self.mode

    def _enter_jointed_mode(self, axis: int, direction: int) -> None:
        self._clear_solid_welds()
        self._clear_group_welds_and_hinges()

        for link_a, link_b in self._neighbor_weld_pairs(skip_axis=axis):
            self._add_weld(link_a, link_b, self._group_welds)

        center = self._layer_representative(axis, 0)
        axis_vec = _axis_vector(axis)
        for layer in (-1, 1):
            side = self._layer_representative(axis, layer)
            anchor = np.zeros(3, dtype=float)
            anchor[axis] = layer * self.spec.spacing * 0.5
            self._solver.add_hinge_constraint(
                self._global_link(center),
                self._global_link(side),
                anchor,
                axis_vec,
                span=self._hinge_span(),
            )
            self._hinge_pairs.append((self._global_link(center), self._global_link(side)))

        self._jointed = _JointedMode(axis=axis, direction=direction)
        self.mode = "jointed"
        self._log(f"entered jointed mode on {_AXIS_NAMES[axis]} axis")

    def _enter_solid_mode(self) -> None:
        self._clear_group_welds_and_hinges()
        self._clear_solid_welds()

        for link_a, link_b in self._neighbor_weld_pairs():
            self._add_weld(link_a, link_b, self._solid_welds)

        self._jointed = None
        self.mode = "solid"
        self._log("entered solid mode")

    def _add_weld(self, link_a: int, link_b: int, target: list[tuple[int, int]]) -> None:
        global_a = self._global_link(link_a)
        global_b = self._global_link(link_b)
        self._solver.add_weld_constraint(global_a, global_b)
        target.append((global_a, global_b))

    def _clear_solid_welds(self) -> None:
        for link_a, link_b in reversed(self._solid_welds):
            self._solver.delete_weld_constraint(link_a, link_b)
        self._solid_welds.clear()

    def _clear_group_welds_and_hinges(self) -> None:
        for link_a, link_b in reversed(self._hinge_pairs):
            self._solver.delete_hinge_constraint(link_a, link_b)
        self._hinge_pairs.clear()

        for link_a, link_b in reversed(self._group_welds):
            self._solver.delete_weld_constraint(link_a, link_b)
        self._group_welds.clear()

    def _layer_link_indices(self, axis: int, layer: int) -> list[int]:
        return [i for i, coord in self.coord_by_link.items() if coord[axis] == layer]

    def _neighbor_weld_pairs(self, skip_axis: int | None = None) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for coord in self.coords_by_link:
            for axis in range(3):
                if axis == skip_axis:
                    continue
                if coord[axis] >= 1:
                    continue
                neighbor = list(coord)
                neighbor[axis] += 1
                neighbor_coord = cast(Coord, tuple(neighbor))
                pairs.append((self.link_by_coord[coord], self.link_by_coord[neighbor_coord]))
        return pairs

    def _layer_representative(self, axis: int, layer: int) -> int:
        coord = [0, 0, 0]
        coord[axis] = layer
        return self.link_by_coord[cast(Coord, tuple(coord))]

    def _solid_angular_velocity(self) -> FloatArray:
        return self._links_ang().mean(axis=0)

    def _joint_relative_velocities(self, axis: int) -> tuple[float, float]:
        angular_velocity = self._links_ang()
        center = self._layer_representative(axis, 0)
        axis_vec = _axis_vector(axis)
        out: list[float] = []
        for layer in (-1, 1):
            side = self._layer_representative(axis, layer)
            rel = angular_velocity[side] - angular_velocity[center]
            out.append(float(np.dot(rel, axis_vec)))
        return out[0], out[1]

    def _joint_angles(self, axis: int) -> tuple[float, float]:
        quats = self._links_quat()
        center = self._layer_representative(axis, 0)
        axis_vec = _axis_vector(axis)
        out: list[float] = []
        for layer in (-1, 1):
            side = self._layer_representative(axis, layer)
            out.append(_twist_angle(quats[center], quats[side], axis_vec))
        return out[0], out[1]

    def _apply_joint_restoring_torques(
        self, axis: int, angles: tuple[float, float], rel_vels: tuple[float, float]
    ) -> None:
        if self.config.joint_spring == 0 and self.config.joint_damping == 0:
            return

        center = self._layer_representative(axis, 0)
        center_global = self._global_link(center)
        axis_vec = _axis_vector(axis)
        for layer, angle, rel_vel in zip((-1, 1), angles, rel_vels):
            side = self._layer_representative(axis, layer)
            side_global = self._global_link(side)
            torque_mag = -(self.config.joint_spring * angle + self.config.joint_damping * rel_vel)
            torque = axis_vec * torque_mag
            self._solver.apply_links_external_torque(
                [torque], links_idx=[side_global], ref="link_com"
            )
            self._solver.apply_links_external_torque(
                [-torque], links_idx=[center_global], ref="link_com"
            )

    def _links_ang(self) -> FloatArray:
        return _as_link_array(self.entity.get_links_ang(), len(self.coords_by_link), 3)

    def _links_quat(self) -> FloatArray:
        return _as_link_array(self.entity.get_links_quat(), len(self.coords_by_link), 4)

    def _links_pos(self) -> FloatArray:
        try:
            value = self.entity.get_links_pos(ref="link_origin")
        except TypeError:
            value = self.entity.get_links_pos()
        return _as_link_array(value, len(self.coords_by_link), 3)

    def _max_lattice_error(self) -> float:
        positions = self._links_pos()
        max_error = 0.0
        for link_a, link_b in self._neighbor_weld_pairs():
            distance = float(np.linalg.norm(positions[link_b] - positions[link_a]))
            max_error = max(max_error, abs(distance - self.spec.spacing))
        return max_error

    def _project_to_allowed_motion(self) -> None:
        positions = self._links_pos()
        qpos = _to_numpy(self.entity.get_qpos()).copy()
        if self._jointed is None:
            rotation, translation = _fit_lattice_pose(
                positions,
                self.coords_by_link,
                list(range(len(self.coords_by_link))),
                self.spec.spacing,
            )
            core_qs = self._q_slice(self.link_by_coord[(0, 0, 0)])
            qpos_rotation = _quat_to_matrix(qpos[core_qs.start + 3 : core_qs.stop])
            if _rotation_distance(rotation, qpos_rotation) < _rotation_distance(
                rotation, np.eye(3)
            ):
                rotation = qpos_rotation
            reference = _lattice_positions(
                self.coords_by_link,
                list(range(len(self.coords_by_link))),
                self.spec.spacing,
                rotation,
                translation,
            )
            self._write_projected_pose(qpos, reference, _matrix_to_quat(rotation))
            self.entity.set_qpos(qpos, zero_velocity=False)
            self._sync_projected_velocity()
            return

        axis = self._jointed.axis
        center_links = self._layer_link_indices(axis, 0)
        center_rotation, center_translation = _fit_lattice_pose(
            positions, self.coords_by_link, center_links, self.spec.spacing
        )
        center_qs = self._q_slice(self._layer_representative(axis, 0))
        qpos_center_rotation = _quat_to_matrix(qpos[center_qs.start + 3 : center_qs.stop])
        if _rotation_distance(center_rotation, qpos_center_rotation) < _rotation_distance(
            center_rotation, np.eye(3)
        ):
            center_rotation = qpos_center_rotation
        center_reference = _lattice_positions(
            self.coords_by_link,
            center_links,
            self.spec.spacing,
            center_rotation,
            center_translation,
        )
        self._write_projected_pose(qpos, center_reference, _matrix_to_quat(center_rotation))

        axis_vec = _axis_vector(axis)
        axis_world = center_rotation @ axis_vec
        center = self._layer_representative(axis, 0)
        angular_velocity = self._links_ang()
        for layer in (-1, 1):
            side = self._layer_representative(axis, layer)
            side_links = self._layer_link_indices(axis, layer)
            side_rotation, _ = _fit_lattice_pose(
                positions, self.coords_by_link, side_links, self.spec.spacing
            )
            angle = _rotation_twist_angle(center_rotation.T @ side_rotation, axis_vec)
            rel_twist_velocity = float(
                np.dot(angular_velocity[side] - angular_velocity[center], axis_world)
            )
            angle += self.config.turn_gain * rel_twist_velocity
            rotation = center_rotation @ _axis_rotation_matrix(axis, angle)
            reference = _lattice_positions(
                self.coords_by_link,
                side_links,
                self.spec.spacing,
                rotation,
                center_translation,
            )
            self._write_projected_pose(qpos, reference, _matrix_to_quat(rotation))

        self.entity.set_qpos(qpos, zero_velocity=False)
        self._sync_projected_velocity(axis_world)

    def _q_slice(self, link_i: int) -> slice:
        start = self.entity.links[link_i].q_start - self.entity.q_start
        return slice(start, start + 7)

    def _write_projected_pose(
        self, qpos: FloatArray, reference: dict[int, FloatArray], quat: FloatArray
    ) -> None:
        for link_i, position in reference.items():
            qs = self._q_slice(link_i)
            qpos[qs.start : qs.start + 3] = position
            qpos[qs.start + 3 : qs.stop] = quat

    def _sync_projected_velocity(self, axis_world: FloatArray | None = None) -> None:
        velocity = _to_numpy(self.entity.get_dofs_velocity()).copy()
        positions = self._links_pos()
        linear_velocity, angular_velocity = self._velocity_components(velocity)
        if self._jointed is None:
            self._write_solid_projected_velocity(
                velocity, positions, linear_velocity, angular_velocity
            )
            self.entity.set_dofs_velocity(velocity, skip_forward=False)
            return

        axis = self._jointed.axis
        if axis_world is None:
            axis_world = _axis_vector(axis)
        axis_world = np.asarray(axis_world, dtype=float)
        axis_world /= np.linalg.norm(axis_world) + 1e-12
        self._write_jointed_projected_velocity(
            velocity, positions, linear_velocity, angular_velocity, axis_world
        )
        self.entity.set_dofs_velocity(velocity, skip_forward=False)

    def _velocity_components(self, velocity: FloatArray) -> tuple[FloatArray, FloatArray]:
        linear = np.zeros((len(self.coords_by_link), 3), dtype=float)
        angular = np.zeros((len(self.coords_by_link), 3), dtype=float)
        for link_i in range(len(self.coords_by_link)):
            ds = self._dof_slice(link_i)
            linear[link_i] = velocity[ds.start : ds.start + 3]
            angular[link_i] = velocity[ds.start + 3 : ds.stop]
        return linear, angular

    def _write_solid_projected_velocity(
        self,
        velocity: FloatArray,
        positions: FloatArray,
        linear_velocity: FloatArray,
        angular_velocity: FloatArray,
    ) -> None:
        links = list(range(len(self.coords_by_link)))
        center = positions[links].mean(axis=0)
        base_linear, base_angular = _fit_solid_velocity(
            positions,
            linear_velocity,
            angular_velocity,
            links,
            self.spec.mass_per_cubie,
            self.spec.cubie_inertia,
            center,
        )
        for link_i in links:
            link_linear = np.asarray(
                base_linear + np.cross(base_angular, positions[link_i] - center), dtype=float
            )
            self._write_link_velocity(velocity, link_i, link_linear, base_angular)

    def _write_jointed_projected_velocity(
        self,
        velocity: FloatArray,
        positions: FloatArray,
        linear_velocity: FloatArray,
        angular_velocity: FloatArray,
        axis_world: FloatArray,
    ) -> None:
        assert self._jointed is not None
        axis = self._jointed.axis
        center_links = self._layer_link_indices(axis, 0)
        center = positions[center_links].mean(axis=0)
        base_linear, base_angular, relative_twists = _fit_jointed_velocity(
            positions,
            linear_velocity,
            angular_velocity,
            list(range(len(self.coords_by_link))),
            self.coords_by_link,
            axis,
            axis_world,
            self.spec.mass_per_cubie,
            self.spec.cubie_inertia,
            center,
        )
        for layer in (-1, 0, 1):
            rel_twist = relative_twists.get(layer, 0.0)
            layer_angular = base_angular + axis_world * rel_twist
            for link_i in self._layer_link_indices(axis, layer):
                link_linear = np.asarray(
                    base_linear + np.cross(layer_angular, positions[link_i] - center), dtype=float
                )
                self._write_link_velocity(velocity, link_i, link_linear, layer_angular)

    def _write_link_velocity(
        self,
        velocity: FloatArray,
        link_i: int,
        linear_velocity: FloatArray,
        angular_velocity: FloatArray,
    ) -> None:
        ds = self._dof_slice(link_i)
        velocity[ds.start : ds.start + 3] = linear_velocity
        velocity[ds.start + 3 : ds.stop] = angular_velocity

    @property
    def _solver(self) -> _SolverLike:
        if self.scene is not None:
            return self.scene.sim.rigid_solver
        return self.entity.solver

    def _global_link(self, link_i: int) -> int:
        link = self.entity.links[link_i]
        return int(link.idx if hasattr(link, "idx") else link_i)

    def _dof_slice(self, link_i: int) -> slice:
        link = self.entity.links[link_i]
        start = link.dof_start - self.entity.dof_start
        return slice(start, start + 6)

    def _hinge_span(self) -> float:
        return self.config.hinge_span or max(self.spec.cubie_size, self.spec.full_extent * 0.5)

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[genelab] {message}")


class RubiksCubeController:
    """Move cubie free joints so a selected layer turns as a Rubik's-cube move.

    Genesis does not currently expose a native Rubik's-cube mechanism primitive.
    This controller implements action-level physical interaction: cubies are
    free rigid links, and a layer turn is applied as a short kinematic trajectory
    over those links, then snapped to the exact 90-degree lattice pose.
    """

    def __init__(self, entity: _KinematicEntityLike, spec: RubiksCubeSpec | None = None) -> None:
        self.entity = entity
        self.spec = spec or RubiksCubeSpec()
        self.coords_by_link: list[Coord] = list(iter_cubie_coords(self.spec.include_hidden_core))
        self.link_by_coord: dict[Coord, int] = {
            coord: i for i, coord in enumerate(self.coords_by_link)
        }
        self.coord_by_link: dict[int, Coord] = dict(enumerate(self.coords_by_link))
        self._active: _ActiveTurn | None = None

    @property
    def is_turning(self) -> bool:
        return self._active is not None

    def queue_turn(
        self, axis: Axis = "z", layer: Layer = 1, turns: int = 1, steps: int = 30
    ) -> None:
        if self._active is not None:
            raise RuntimeError("a layer turn is already active")
        axis_i = normalize_axis(axis)
        layer_i = normalize_layer(layer)
        turns_i = int(turns)
        if turns_i == 0:
            return
        steps_i = max(1, int(steps))
        link_indices = [i for i, coord in self.coord_by_link.items() if coord[axis_i] == layer_i]
        if len(link_indices) != 9:
            raise RuntimeError(f"expected 9 cubies in layer, got {len(link_indices)}")

        qpos = _to_numpy(self.entity.get_qpos()).copy()
        start_pos = {
            i: qpos[self._q_slice(i).start : self._q_slice(i).start + 3].copy()
            for i in link_indices
        }
        start_quat = {
            i: qpos[self._q_slice(i).start + 3 : self._q_slice(i).stop].copy() for i in link_indices
        }
        self._active = _ActiveTurn(
            axis=axis_i,
            layer=layer_i,
            turns=turns_i,
            steps=steps_i,
            step=0,
            link_indices=tuple(link_indices),
            start_pos=start_pos,
            start_quat=start_quat,
            qpos=qpos,
        )

    def step(self, scene: ControllerSceneLike | None = None) -> bool:
        """Advance one queued turn step; returns True once the turn completes."""

        if self._active is None:
            return False

        active = self._active
        active.step += 1
        fraction = active.step / active.steps
        angle = active.turns * math.pi / 2.0 * fraction
        rot = _axis_angle_quat(active.axis, angle)
        matrix = _axis_rotation_matrix(active.axis, angle)

        for link_i in active.link_indices:
            qs = self._q_slice(link_i)
            active.qpos[qs.start : qs.start + 3] = matrix @ active.start_pos[link_i]
            active.qpos[qs.start + 3 : qs.stop] = _quat_mul(rot, active.start_quat[link_i])

        is_done = active.step >= active.steps
        if is_done:
            self._snap_active_turn()

        self.entity.set_qpos(active.qpos, zero_velocity=True)
        if scene is not None:
            scene.step()

        if is_done:
            self._active = None
        return is_done

    def turn(
        self,
        scene: ControllerSceneLike,
        axis: Axis = "z",
        layer: Layer = 1,
        turns: int = 1,
        steps: int = 30,
    ) -> None:
        """Run a complete layer turn, stepping the Genesis scene along the way."""

        self.queue_turn(axis=axis, layer=layer, turns=turns, steps=steps)
        while self._active is not None:
            self.step(scene)

    def _snap_active_turn(self) -> None:
        assert self._active is not None
        active = self._active
        quarter_turns = active.turns % 4
        angle = quarter_turns * math.pi / 2.0
        rot = _axis_angle_quat(active.axis, angle)
        matrix = _axis_rotation_matrix(active.axis, angle)
        new_coord_by_link = self.coord_by_link.copy()

        for link_i in active.link_indices:
            old_coord = self.coord_by_link[link_i]
            new_coord = _round_coord(matrix @ np.asarray(old_coord, dtype=float))
            new_coord_by_link[link_i] = new_coord

            qs = self._q_slice(link_i)
            active.qpos[qs.start : qs.start + 3] = np.asarray(
                cubie_center(new_coord, self.spec), dtype=float
            )
            active.qpos[qs.start + 3 : qs.stop] = _quat_mul(rot, active.start_quat[link_i])

        self.coord_by_link = new_coord_by_link
        self.link_by_coord = {coord: link_i for link_i, coord in new_coord_by_link.items()}

    def _q_slice(self, link_i: int) -> slice:
        start = self.entity.links[link_i].q_start - self.entity.q_start
        return slice(start, start + 7)


@dataclass
class _ActiveTurn:
    axis: int
    layer: int
    turns: int
    steps: int
    step: int
    link_indices: tuple[int, ...]
    start_pos: dict[int, FloatArray]
    start_quat: dict[int, FloatArray]
    qpos: FloatArray


def normalize_axis(axis: Axis) -> int:
    try:
        key = axis.lower() if isinstance(axis, str) else axis
        return _AXIS_INDEX[key]
    except KeyError as exc:
        raise ValueError("axis must be one of x/y/z or 0/1/2") from exc


def normalize_layer(layer: Layer) -> int:
    if isinstance(layer, str):
        value = layer.lower()
        if value not in _LAYER_INDEX:
            raise ValueError("layer must be -1/0/1, n/c/p, or -/0/+")
        return _LAYER_INDEX[value]
    if int(layer) not in (-1, 0, 1):
        raise ValueError("layer must be -1, 0, or 1")
    return int(layer)


def select_rotation_axis(angular_velocity: FloatArray, threshold: float) -> int | None:
    """Return the dominant world axis if angular speed exceeds ``threshold``."""

    values = np.asarray(angular_velocity, dtype=float)
    if values.shape != (3,):
        raise ValueError("angular_velocity must be a 3-vector")
    axis = int(np.argmax(np.abs(values)))
    if abs(values[axis]) < threshold:
        return None
    return axis


def _axis_angle_quat(axis: int, angle: float) -> FloatArray:
    quat = np.zeros(4, dtype=float)
    quat[0] = math.cos(angle / 2.0)
    quat[axis + 1] = math.sin(angle / 2.0)
    return quat


def _axis_rotation_matrix(axis: int, angle: float) -> FloatArray:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == 0:
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=float)
    if axis == 1:
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=float)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float)


def _axis_vector(axis: int) -> FloatArray:
    vector = np.zeros(3, dtype=float)
    vector[axis] = 1.0
    return vector


def _fit_lattice_pose(
    positions: FloatArray,
    coords_by_link: list[Coord],
    links: list[int],
    spacing: float,
) -> tuple[FloatArray, FloatArray]:
    group_positions = positions[links]
    group_lattice = np.asarray([coords_by_link[link_i] for link_i in links], dtype=float) * spacing
    lattice_center = group_lattice.mean(axis=0)
    position_center = group_positions.mean(axis=0)
    covariance = (group_lattice - lattice_center).T @ (group_positions - position_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = position_center - rotation @ lattice_center
    return rotation, translation


def _lattice_positions(
    coords_by_link: list[Coord],
    links: list[int],
    spacing: float,
    rotation: FloatArray,
    translation: FloatArray,
) -> dict[int, FloatArray]:
    return {
        link_i: translation + rotation @ (np.asarray(coords_by_link[link_i], dtype=float) * spacing)
        for link_i in links
    }


def _fit_solid_velocity(
    positions: FloatArray,
    linear_velocity: FloatArray,
    angular_velocity: FloatArray,
    links: list[int],
    mass: float,
    inertia: float,
    center: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    rel_positions = positions[links] - center
    link_linear = linear_velocity[links]
    link_angular = angular_velocity[links]
    base_linear = link_linear.mean(axis=0)

    inertia_matrix = np.eye(3, dtype=float) * (len(links) * inertia)
    angular_momentum = np.zeros(3, dtype=float)
    for rel_pos, linear, angular in zip(rel_positions, link_linear, link_angular):
        angular_momentum += mass * np.cross(rel_pos, linear - base_linear) + inertia * angular
        inertia_matrix += mass * (
            (float(np.dot(rel_pos, rel_pos)) * np.eye(3)) - np.outer(rel_pos, rel_pos)
        )

    base_angular = np.linalg.pinv(inertia_matrix, rcond=1e-10) @ angular_momentum
    return base_linear, base_angular


def _fit_jointed_velocity(
    positions: FloatArray,
    linear_velocity: FloatArray,
    angular_velocity: FloatArray,
    links: list[int],
    coords_by_link: list[Coord],
    axis: int,
    axis_world: FloatArray,
    mass: float,
    inertia: float,
    center: FloatArray,
) -> tuple[FloatArray, FloatArray, dict[int, float]]:
    rows: list[FloatArray] = []
    values: list[FloatArray] = []
    linear_weight = math.sqrt(mass)
    angular_weight = math.sqrt(inertia)
    axis_world = np.asarray(axis_world, dtype=float)
    axis_world /= np.linalg.norm(axis_world) + 1e-12

    for link_i in links:
        layer = coords_by_link[link_i][axis]
        alpha_col = 6 if layer == -1 else 7 if layer == 1 else None
        rel_pos = positions[link_i] - center

        linear_row = np.zeros((3, 8), dtype=float)
        linear_row[:, 0:3] = np.eye(3)
        linear_row[:, 3:6] = -_skew(rel_pos)
        if alpha_col is not None:
            linear_row[:, alpha_col] = np.cross(axis_world, rel_pos)
        rows.append(linear_row * linear_weight)
        values.append(linear_velocity[link_i] * linear_weight)

        angular_row = np.zeros((3, 8), dtype=float)
        angular_row[:, 3:6] = np.eye(3)
        if alpha_col is not None:
            angular_row[:, alpha_col] = axis_world
        rows.append(angular_row * angular_weight)
        values.append(angular_velocity[link_i] * angular_weight)

    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(values), rcond=None)
    return solution[0:3], solution[3:6], {-1: float(solution[6]), 0: 0.0, 1: float(solution[7])}


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)


def _rotation_twist_angle(rotation: FloatArray, axis: FloatArray) -> float:
    quat = _matrix_to_quat(rotation)
    if quat[0] < 0:
        quat = -quat
    projected = np.asarray(axis, dtype=float) * float(np.dot(quat[1:], axis))
    twist = np.concatenate(([quat[0]], projected))
    norm = np.linalg.norm(twist)
    if norm <= 1e-12:
        return 0.0
    twist /= norm
    return float(2.0 * math.atan2(np.dot(twist[1:], axis), twist[0]))


def _matrix_to_quat(matrix: FloatArray) -> FloatArray:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array(
                (
                    (m[2, 1] - m[1, 2]) / s,
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                )
            )
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array(
                (
                    (m[0, 2] - m[2, 0]) / s,
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s,
                )
            )
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array(
                (
                    (m[1, 0] - m[0, 1]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s,
                )
            )
    return quat / np.linalg.norm(quat)


def _quat_to_matrix(quat: FloatArray) -> FloatArray:
    w, x, y, z = np.asarray(quat, dtype=float)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _rotation_distance(left: FloatArray, right: FloatArray) -> float:
    delta = np.asarray(left, dtype=float).T @ np.asarray(right, dtype=float)
    value = (float(np.trace(delta)) - 1.0) * 0.5
    return abs(math.acos(max(-1.0, min(1.0, value))))


def _quat_conj(quat: FloatArray) -> FloatArray:
    out = np.asarray(quat, dtype=float).copy()
    out[1:] *= -1.0
    return out


def _twist_angle(base_quat: FloatArray, target_quat: FloatArray, axis: FloatArray) -> float:
    rel = _quat_mul(
        np.asarray(target_quat, dtype=float), _quat_conj(np.asarray(base_quat, dtype=float))
    )
    if rel[0] < 0:
        rel = -rel
    projected = np.asarray(axis, dtype=float) * float(np.dot(rel[1:], axis))
    twist = np.concatenate(([rel[0]], projected))
    norm = np.linalg.norm(twist)
    if norm <= 1e-12:
        return 0.0
    twist /= norm
    return float(2.0 * math.atan2(np.dot(twist[1:], axis), twist[0]))


def _as_link_array(value: ArrayLike, n_links: int, width: int) -> FloatArray:
    array = _to_numpy(value)
    array = np.asarray(array, dtype=float)
    if array.shape == (n_links, width):
        return array
    if array.shape == (1, n_links, width):
        return array[0]
    if array.ndim == 3 and array.shape[1:] == (n_links, width):
        return array[0]
    return array.reshape(n_links, width)


def _quat_mul(left: FloatArray, right: FloatArray) -> FloatArray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    out = np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=float,
    )
    return out / np.linalg.norm(out)


def _round_coord(values: FloatArray) -> Coord:
    x, y, z = (int(round(v)) for v in values)
    return x, y, z


def _to_numpy(value: ArrayLike) -> FloatArray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float)
    if hasattr(value, "detach"):
        tensor = cast(_TensorLike, value)
        return np.asarray(tensor.detach().cpu().numpy(), dtype=float)
    return np.asarray(value, dtype=float)
