"""Genesis environment registrations for the external example tasks."""

from dataclasses import replace
from typing import Any, Protocol, cast

from genelab.registry import ENVS, register_env

from genelab_examples.gui_panels.config import GuiPanelsEnvCfg
from genelab_examples.gui_panels.env import create_gui_panels_env
from genelab_examples.robots import create_rubiks_robot
from genelab_examples.rubiks.assets import RubiksCubeSpec
from genelab_examples.rubiks.config import RubiksEnvCfg
from genelab_examples.rubiks.sim import (
    ControlledEntityLike,
    ControllerSceneLike,
    ForceDrivenCubeController,
    RubiksCubeController,
)


class _ViewerLike(Protocol):
    def add_plugin(self, plugin: object) -> None: ...


class _RubiksEntityLike(ControlledEntityLike, Protocol):
    n_links: int
    n_geoms: int
    n_vgeoms: int
    n_equalities: int


class _SceneLike(ControllerSceneLike, Protocol):
    viewer: _ViewerLike

    def add_entity(self, *args: object, **kwargs: object) -> _RubiksEntityLike: ...

    def build(self) -> None: ...

    def step(self) -> None: ...


def _genesis() -> Any:
    # Genesis is optional at import time and does not ship complete type stubs.
    import genesis as gs  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    return cast(Any, gs)


class RubiksPlayEnv:
    """Genesis environment for the force-driven Rubik's cube demo."""

    def __init__(self, cfg: RubiksEnvCfg | None = None) -> None:
        self.cfg = cfg or RubiksEnvCfg()

    def play(self) -> None:
        from genelab.cache import ensure_project_cache

        ensure_project_cache()

        gs = _genesis()

        cfg = self.cfg
        if cfg.interaction.interactive_force and not cfg.simulation.vis:
            raise SystemExit("--env.interaction.interactive_force requires --vis")

        robot = create_rubiks_robot(cfg.robot)
        spec = cfg.robot.spec()
        asset_path = robot.write_asset()

        gs.init(backend=gs.gpu if cfg.simulation.gpu else gs.cpu, precision="32")
        try:
            scene = cast(
                _SceneLike,
                gs.Scene(
                    sim_options=gs.options.SimOptions(
                        dt=cfg.simulation.dt, substeps=cfg.simulation.substeps
                    ),
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
                    show_viewer=cfg.simulation.vis,
                ),
            )
            scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0))
            cube = scene.add_entity(
                gs.morphs.MJCF(
                    file=str(asset_path),
                    pos=cfg.robot.spawn_pos,
                    requires_jac_and_IK=False,
                ),
                material=gs.materials.Rigid(rho=cfg.robot.rho, friction=cfg.robot.friction),
                name="rubiks_cube",
            )
            if cfg.interaction.interactive_force:
                scene.viewer.add_plugin(
                    gs.vis.viewer_plugins.MouseInteractionPlugin(
                        use_force=True,
                        spring_const=cfg.interaction.mouse_spring,
                        max_spring_distance=cfg.interaction.mouse_max_distance,
                        max_force=cfg.interaction.mouse_max_force,
                        max_torque=cfg.interaction.mouse_max_torque,
                        color=(0.1, 0.6, 0.8, 0.6),
                    )
                )
            scene.build()

            controller = self._make_controller(scene, cube, spec)
            if cfg.interaction.interactive_force:
                print(
                    "Interactive force mode: left-drag a cubie in the viewer to apply real spring forces "
                    f"(spring={cfg.interaction.mouse_spring:g}, "
                    f"max-distance={cfg.interaction.mouse_max_distance:g}, "
                    f"max-force={cfg.interaction.mouse_max_force:g}, "
                    f"max-torque={cfg.interaction.mouse_max_torque:g})."
                )

            for _ in range(max(0, cfg.simulation.steps)):
                if controller is not None:
                    controller.step()
                scene.step()

            mode = "welded" if spec.weld_cubies else "force-driven"
            print(
                f"Simulated {mode} cube: "
                f"{cube.n_links} cubies, {cube.n_geoms} collision geoms, "
                f"{cube.n_vgeoms} visual geoms, {cube.n_equalities} weld constraints"
            )
        finally:
            gs.destroy()

    def _make_controller(
        self, scene: _SceneLike, cube: _RubiksEntityLike, spec: RubiksCubeSpec
    ) -> ForceDrivenCubeController | None:
        cfg = self.cfg
        if spec.weld_cubies:
            print("Cube is statically welded; physical three-link mode is disabled.")
            return None
        if cfg.legacy_turn.enabled:
            legacy = RubiksCubeController(cube, spec)
            legacy.turn(
                scene,
                axis=cfg.legacy_turn.axis,
                layer=cfg.legacy_turn.layer,
                turns=cfg.legacy_turn.turns,
                steps=cfg.legacy_turn.steps,
            )
            return None
        force_cfg = replace(
            cfg.force_controller, verbose=cfg.force_controller.verbose or cfg.simulation.vis
        )
        return ForceDrivenCubeController(cube, scene=scene, spec=spec, config=force_cfg)


def create_rubiks_env(cfg: RubiksEnvCfg | None = None) -> RubiksPlayEnv:
    return RubiksPlayEnv(cfg)


def register() -> None:
    if "rubiks-play" not in ENVS:
        register_env(
            "rubiks-play",
            lambda: create_rubiks_env(),
            description="Example Genesis scene for the force-driven Rubik's cube.",
            cfg_type=RubiksEnvCfg,
        )
    if "gui-panels-demo" not in ENVS:
        register_env(
            "gui-panels-demo",
            lambda: create_gui_panels_env(),
            description="Cookbook scene showing how to add common ImGui widgets to the viewer.",
            cfg_type=GuiPanelsEnvCfg,
            examples=["genelab play GeneLab-GUI-Panels-Demo-v0 --vis"],
        )
