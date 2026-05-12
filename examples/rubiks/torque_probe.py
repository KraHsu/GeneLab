"""Apply scripted torques to the Rubik cube simulation and print mode/angle diagnostics."""

import argparse
from pathlib import Path
import tempfile
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from genelab.cache import ensure_project_cache
from genelab_examples.rubiks.assets import RubiksCubeSpec, write_mjcf
from genelab_examples.rubiks.sim import (
    ControlledEntityLike,
    ControllerSceneLike,
    ForceDrivenCubeConfig,
    ForceDrivenCubeController,
)

type FloatArray = NDArray[np.floating]


class _RigidSolverLike(Protocol):
    def apply_links_external_torque(
        self, torque: FloatArray, links_idx: list[int], ref: str
    ) -> None: ...


class _SimLike(Protocol):
    rigid_solver: _RigidSolverLike


class _ViewerLike(Protocol):
    def is_alive(self) -> bool: ...


class _SceneLike(Protocol):
    sim: _SimLike
    viewer: _ViewerLike

    def add_entity(self, *args: object, **kwargs: object) -> object: ...

    def build(self) -> None: ...

    def step(self) -> None: ...


def _genesis() -> Any:
    # Genesis is optional at type-check time and does not ship complete stubs.
    import genesis as gs  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    return cast(Any, gs)


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe cube turning by applying a scripted external torque."
    )
    parser.add_argument(
        "--axis", choices=("x", "y", "z"), default="z", help="Axis used for torque and diagnostics."
    )
    parser.add_argument(
        "--layer",
        type=int,
        choices=(-1, 0, 1),
        default=1,
        help="Layer/slab to torque after jointed mode.",
    )
    parser.add_argument("--steps", type=int, default=240, help="Total simulation steps.")
    parser.add_argument(
        "--solid-steps", type=int, default=25, help="Initial steps applying torque to all cubies."
    )
    parser.add_argument("--torque", type=float, default=0.035, help="Torque magnitude in N*m.")
    parser.add_argument(
        "--turn-gain",
        type=float,
        default=0.25,
        help="Controller turn_gain to use during the probe.",
    )
    parser.add_argument(
        "--enter-ang-vel",
        type=float,
        default=0.25,
        help="Angular velocity threshold for jointed mode.",
    )
    parser.add_argument("--print-every", type=int, default=10, help="Diagnostic print interval.")
    parser.add_argument(
        "-v", "--vis", action="store_true", help="Show the Genesis viewer while probing."
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        help="Keep stepping after the probe so the viewer stays open; implies --vis.",
    )
    args = parser.parse_args()

    ensure_project_cache()
    gs = _genesis()

    axis = _axis_index(args.axis)
    axis_vec = np.zeros(3, dtype=float)
    axis_vec[axis] = 1.0

    spec = RubiksCubeSpec()
    asset_path = write_mjcf(Path(tempfile.gettempdir()) / "rubiks_cube_torque_probe.xml", spec)

    gs.init(backend=gs.cpu, precision="32")
    scene = cast(
        _SceneLike,
        gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.003, substeps=4),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
                enable_self_collision=False,
                max_collision_pairs=4096,
                max_dynamic_constraints=128,
                constraint_timeconst=0.01,
                iterations=100,
                ls_iterations=100,
                tolerance=1e-6,
                noslip_iterations=5,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.45, -0.55, 0.35),
                camera_lookat=(0.0, 0.0, 0.08),
                camera_fov=35,
                max_FPS=60,
            ),
            show_viewer=args.vis or args.hold_open,
        ),
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0))
    cube = scene.add_entity(
        gs.morphs.MJCF(file=str(asset_path), pos=(0.0, 0.0, 0.25), requires_jac_and_IK=False),
        material=gs.materials.Rigid(rho=500.0, friction=1.2),
        name="rubiks_cube",
    )
    scene.build()

    controller = ForceDrivenCubeController(
        cast(ControlledEntityLike, cube),
        scene=cast(ControllerSceneLike, scene),
        spec=spec,
        config=ForceDrivenCubeConfig(
            enter_ang_vel=args.enter_ang_vel,
            joint_spring=0.0,
            joint_damping=0.002,
            settle_steps=999999,
            turn_gain=args.turn_gain,
            verbose=False,
        ),
    )

    all_links = [controller.global_link(i) for i in range(len(controller.coords_by_link))]
    layer_links = [
        controller.global_link(i) for i in controller.layer_link_indices(axis, args.layer)
    ]
    torque = axis_vec * args.torque

    for step in range(args.steps):
        if step < args.solid_steps or controller.mode == "solid":
            targets = all_links
            per_link_torque = torque / len(targets)
        else:
            targets = layer_links
            per_link_torque = torque / len(targets)
        scene.sim.rigid_solver.apply_links_external_torque(
            np.tile(per_link_torque, (len(targets), 1)),
            links_idx=targets,
            ref="link_com",
        )

        mode = controller.step()
        scene.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            angles = controller.joint_angles
            mean_ang = controller.solid_angular_velocity()
            print(
                f"step={step:04d} mode={mode:7s} active_axis={controller.active_axis} "
                f"mean_ang={mean_ang.round(4).tolist()} joint_angles=({angles[0]:.4f}, {angles[1]:.4f})"
            )

    if args.hold_open:
        print(
            "Probe finished; keeping viewer open. Close the viewer window or press Ctrl+C to stop."
        )
        try:
            while scene.viewer.is_alive():
                scene.step()
        except KeyboardInterrupt:
            pass

    gs.destroy()


if __name__ == "__main__":
    main()
