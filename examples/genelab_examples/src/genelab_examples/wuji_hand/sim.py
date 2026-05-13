"""Wuji hand playback simulation helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from genelab_examples.wuji_hand.assets import (
    DEFAULT_DESC_DIR,
    DEFAULT_TRAJECTORY,
    SIDES,
    load_trajectory,
    resolve_description_dir,
    resolve_mjcf_path,
    wuji_joint_names,
)

type FloatArray = NDArray[np.float32]
type IntArray = NDArray[np.int64]
type AnyArray = NDArray[np.generic]
type ArrayLike = object


@dataclass(frozen=True)
class JointMapping:
    """Map MuJoCo trajectory columns to Genesis local DOF indices."""

    mujoco_indices: IntArray
    dof_indices: list[int]
    joint_names: list[str]
    missing_joint_names: list[str]


@dataclass(frozen=True)
class WujiHandRunConfig:
    side: str = "right"
    desc_dir: Path = DEFAULT_DESC_DIR
    trajectory: Path = DEFAULT_TRAJECTORY
    vis: bool = False
    gpu: bool = False
    steps: int = 0
    dt: float = 0.01
    reset_interval: int = 500

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.steps < 0:
            raise ValueError("steps must be >= 0; use 0 for an infinite viewer loop")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.reset_interval < 0:
            raise ValueError("reset_interval must be >= 0; use 0 to disable periodic hard reset")


class _JointLike(Protocol):
    dofs_idx_local: list[int]


class _ArrayProvider(Protocol):
    def numpy(self) -> AnyArray: ...


class _CpuTensorLike(Protocol):
    def cpu(self) -> _ArrayProvider: ...


class _TensorLike(Protocol):
    def detach(self) -> _CpuTensorLike: ...


class _EntityLike(Protocol):
    n_dofs: int

    def get_joint(self, name: str) -> _JointLike: ...

    def get_dofs_limit(self, dof_indices: list[int]) -> tuple[ArrayLike, ArrayLike]: ...

    def set_dofs_kp(self, kp: FloatArray, dofs_idx_local: list[int]) -> None: ...

    def set_dofs_kv(self, kv: FloatArray, dofs_idx_local: list[int]) -> None: ...

    def control_dofs_position(self, position: FloatArray, dof_indices: list[int]) -> None: ...

    def set_dofs_position(
        self, position: FloatArray, dof_indices: list[int], zero_velocity: bool
    ) -> None: ...


class _ForceRangeEntityLike(_EntityLike, Protocol):
    def get_dofs_force_range(self, dof_indices: list[int]) -> tuple[ArrayLike, ArrayLike]: ...

    def set_dofs_force_range(
        self, lower: FloatArray, upper: FloatArray, dofs_idx_local: list[int]
    ) -> None: ...


class _ViewerLike(Protocol):
    def is_alive(self) -> bool: ...


class _SceneLike(Protocol):
    viewer: _ViewerLike

    def add_entity(self, *args: object, **kwargs: object) -> _EntityLike: ...

    def build(self) -> None: ...

    def step(self) -> None: ...


def _genesis() -> Any:
    # Genesis is optional at import time and does not ship complete type stubs.
    import genesis as gs  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    return cast(Any, gs)


def build_joint_mapping(entity: _EntityLike, side: str) -> JointMapping:
    mujoco_indices: list[int] = []
    dof_indices: list[int] = []
    mapped_joint_names: list[str] = []
    missing_joint_names: list[str] = []

    for mujoco_idx, name in enumerate(wuji_joint_names(side)):
        try:
            joint = entity.get_joint(name)
        except Exception:  # Genesis raises a package-specific exception for unknown joints.
            missing_joint_names.append(name)
            continue
        local_dofs = list(joint.dofs_idx_local)
        if not local_dofs:
            missing_joint_names.append(name)
            continue
        mujoco_indices.append(mujoco_idx)
        dof_indices.append(int(local_dofs[0]))
        mapped_joint_names.append(name)

    if not dof_indices:
        raise RuntimeError(f"No controllable Wuji joints were found for side={side!r}")
    return JointMapping(
        mujoco_indices=np.asarray(mujoco_indices, dtype=np.int64),
        dof_indices=dof_indices,
        joint_names=mapped_joint_names,
        missing_joint_names=missing_joint_names,
    )


def _as_numpy(value: ArrayLike) -> FloatArray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    if hasattr(value, "detach"):
        tensor = cast(_TensorLike, value)
        return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    if hasattr(value, "cpu"):
        tensor = cast(_CpuTensorLike, value)
        return np.asarray(tensor.cpu().numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def dof_limits(entity: _EntityLike, dof_indices: list[int]) -> tuple[FloatArray, FloatArray]:
    lower, upper = entity.get_dofs_limit(dof_indices)
    lower_np = _as_numpy(lower).astype(np.float32, copy=False)
    upper_np = _as_numpy(upper).astype(np.float32, copy=False)
    return lower_np, upper_np


def trajectory_target(
    frame: FloatArray,
    mapping: JointMapping,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    target = frame[mapping.mujoco_indices].astype(np.float32, copy=False)
    return np.clip(target, lower, upper)


def apply_wuji_gains(entity: _EntityLike, dof_indices: list[int]) -> None:
    """Use moderate defaults when the imported actuators are not PD-reducible."""
    n_dofs = len(dof_indices)
    entity.set_dofs_kp(np.full(n_dofs, 0.8, dtype=np.float32), dofs_idx_local=dof_indices)
    entity.set_dofs_kv(np.full(n_dofs, 0.04, dtype=np.float32), dofs_idx_local=dof_indices)
    if hasattr(entity, "get_dofs_force_range"):
        force_entity = cast(_ForceRangeEntityLike, entity)
        lower, upper = force_entity.get_dofs_force_range(dof_indices)
        lower_np = _as_numpy(lower).astype(np.float32, copy=False)
        upper_np = _as_numpy(upper).astype(np.float32, copy=False)
        if np.all(np.isfinite(lower_np)) and np.all(np.isfinite(upper_np)):
            force_entity.set_dofs_force_range(lower_np, upper_np, dofs_idx_local=dof_indices)


def run_wuji_hand(config: WujiHandRunConfig) -> None:
    gs = _genesis()

    desc_dir = resolve_description_dir(config.desc_dir)
    mjcf_path = resolve_mjcf_path(desc_dir, config.side)
    trajectory = load_trajectory(config.trajectory)

    gs.init(backend=gs.gpu if config.gpu else gs.cpu, precision="32")
    try:
        scene = cast(
            _SceneLike,
            gs.Scene(
                sim_options=gs.options.SimOptions(dt=config.dt),
                rigid_options=gs.options.RigidOptions(
                    enable_self_collision=True,
                    max_collision_pairs=4096,
                    constraint_timeconst=max(0.01, config.dt),
                    iterations=80,
                    ls_iterations=80,
                    tolerance=1e-6,
                    noslip_iterations=5,
                ),
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(0.35, -0.45, 0.32),
                    camera_lookat=(0.0, 0.0, 0.08),
                    camera_fov=35,
                    max_FPS=60,
                ),
                vis_options=gs.options.VisOptions(
                    ambient_light=(0.35, 0.35, 0.35),
                    lights=[
                        {
                            "type": "directional",
                            "dir": (-1.0, -1.0, -1.0),
                            "color": (1.0, 1.0, 1.0),
                            "intensity": 5.0,
                        },
                    ],
                ),
                show_viewer=config.vis,
            ),
        )
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0))
        hand = scene.add_entity(
            gs.morphs.MJCF(
                file=str(mjcf_path),
                pos=(0.0, 0.0, 0.08),
                requires_jac_and_IK=False,
                recompute_inertia=False,
            ),
            material=gs.materials.Rigid(friction=1.0),
            name=f"wuji_hand_{config.side}",
        )
        scene.build()

        mapping = build_joint_mapping(hand, config.side)
        lower, upper = dof_limits(hand, mapping.dof_indices)
        apply_wuji_gains(hand, mapping.dof_indices)

        if mapping.missing_joint_names:
            print("[WARN] Missing Wuji joints skipped: " + ", ".join(mapping.missing_joint_names))
        print(
            f"Wuji {config.side} hand running in Genesis: "
            f"asset={mjcf_path}, trajectory={trajectory.shape[0]}x{trajectory.shape[1]}, "
            f"mapped_joints={len(mapping.dof_indices)}"
        )

        default_target = np.clip(np.zeros(len(mapping.dof_indices), dtype=np.float32), lower, upper)
        count = 0
        while config.steps == 0 or count < config.steps:
            if config.vis and not scene.viewer.is_alive():
                break
            if config.reset_interval and count % config.reset_interval == 0:
                hand.set_dofs_position(default_target, mapping.dof_indices, zero_velocity=True)
                hand.control_dofs_position(default_target, mapping.dof_indices)
                if count == 0 or config.vis:
                    print("[INFO] Resetting Wuji hand state...")

            frame = trajectory[count % len(trajectory)]
            target = trajectory_target(frame, mapping, lower, upper)
            hand.control_dofs_position(target, mapping.dof_indices)
            scene.step()
            count += 1

        print(f"Simulated Wuji {config.side} hand for {count} Genesis steps")
    finally:
        gs.destroy()
