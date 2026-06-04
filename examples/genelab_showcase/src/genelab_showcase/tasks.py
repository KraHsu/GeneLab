"""Showcase task registrations: nine play-only TaskCfgs.

Each task is a thin wrapper around a :class:`~genelab_showcase.runner.ShowcaseRunner`
subclass. The factory is the task class itself (matching the
``examples/inverted_pendulum`` pattern); the registry calls it lazily on
``genelab info`` / ``genelab play`` so ``genelab list tasks`` does not trigger any
asset download.

No training agent is configured — every task carries ``trainable=False``. The CLI
``play`` command therefore bypasses the rsl_rl runner and calls our ``play()`` method
directly, which builds the env, runs the scripted action loop, and dumps the
showcase's evidence under ``logs/showcase/<slug>/``.
"""

from collections.abc import Callable
from typing import cast

from genelab.configs import TaskCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.registry import TASKS, register_task

from genelab_showcase.actuators.env_cfg import (
    actuator_showcase_env_cfg,
    mlp_residual_actuator_showcase_env_cfg,
)
from genelab_showcase.actuators.runner import (
    ActuatorShowcaseRunner,
    MlpResidualActuatorShowcaseRunner,
)
from genelab_showcase.contact.env_cfg import contact_showcase_env_cfg
from genelab_showcase.contact.runner import ContactShowcaseRunner
from genelab_showcase.materials.runner import MaterialsDemoRunner
from genelab_showcase.materials.scenes import (
    FLOAT_DENSITY_RANGE,
    SOFT_STIFFNESS_RANGE,
    MaterialsDemoCfg,
    fluid_demo_cfg,
    get_float_density,
    get_soft_stiffness,
    set_float_density,
    set_soft_stiffness,
    soft_body_demo_cfg,
)
from genelab_showcase.curriculum.env_cfg import curriculum_showcase_env_cfg
from genelab_showcase.curriculum.runner import CurriculumShowcaseRunner
from genelab_showcase.raycast.env_cfg import raycast_showcase_env_cfg
from genelab_showcase.raycast.runner import RayCastShowcaseRunner
from genelab_showcase.recording.env_cfg import recording_showcase_env_cfg
from genelab_showcase.recording.runner import RecordingShowcaseRunner
from genelab_showcase.runner import ShowcaseRunner
from genelab_showcase.sensors.env_cfg import sensors_showcase_env_cfg
from genelab_showcase.sensors.runner import SensorsShowcaseRunner
from genelab_showcase.tactile.env_cfg import tactile_showcase_env_cfg
from genelab_showcase.tactile.runner import TactileShowcaseRunner
from genelab_showcase.terrain.env_cfg import terrain_showcase_env_cfg
from genelab_showcase.terrain.runner import TerrainShowcaseRunner

SENSORS_TASK_ID = "GeneLab-Sensors-Showcase-v0"
TACTILE_TASK_ID = "GeneLab-Tactile-Showcase-v0"
RAYCAST_TASK_ID = "GeneLab-RayCast-Showcase-v0"
CONTACT_TASK_ID = "GeneLab-Contact-Showcase-v0"
TERRAIN_TASK_ID = "GeneLab-Terrain-Showcase-v0"
CURRICULUM_TASK_ID = "GeneLab-Curriculum-Showcase-v0"
ACTUATOR_TASK_ID = "GeneLab-Actuator-Showcase-v0"
MLP_RESIDUAL_ACTUATOR_TASK_ID = "GeneLab-MlpResidual-Actuator-Showcase-v0"
RECORDING_TASK_ID = "GeneLab-Recording-Showcase-v0"
SOFTBODY_TASK_ID = "GeneLab-SoftBody-Showcase-v0"
FLUID_TASK_ID = "GeneLab-Fluid-Showcase-v0"


class _ShowcaseTaskBase:
    """Shared TaskCfg-backed wrapper. Subclasses override the four class attributes."""

    task_id: str = ""
    env_name: str = ""
    robot_name: str = ""
    runner_cls: type[ShowcaseRunner] = ShowcaseRunner
    env_factory: Callable[[], ManagerBasedRlEnvCfg] | None = None

    def __init__(self) -> None:
        if self.env_factory is None:
            raise RuntimeError(f"{type(self).__name__}.env_factory is unset")
        self.cfg = TaskCfg(
            name=self.task_id,
            env_name=self.env_name,
            robot_name=self.robot_name,
            env=self.env_factory(),
            trainable=False,
        )

    def play(self, *, max_steps: int | None = None) -> None:
        # IMPORTANT: pass ``self.cfg.env`` (which the CLI may have mutated via
        # ``--vis`` / ``--steps`` / ``--env.x.y.z`` overrides) so the override actually
        # reaches the env build. Re-calling ``env_factory()`` here would discard the
        # patched cfg and silently ignore the CLI flags.
        #
        # ``max_steps`` is the ``--max-steps`` hard cap, threaded into the runner so the
        # viewer-gated loop honors it exactly like RL playback does.
        env_cfg = cast(ManagerBasedRlEnvCfg, self.cfg.env)
        self.runner_cls(env_cfg).play(max_steps=max_steps)

    def train(self) -> None:
        raise NotImplementedError(f"showcase task {self.task_id} is play-only")


class SensorsShowcaseTask(_ShowcaseTaskBase):
    task_id = SENSORS_TASK_ID
    env_name = "sensors-showcase-env"
    robot_name = "franka"
    runner_cls = SensorsShowcaseRunner
    env_factory = staticmethod(sensors_showcase_env_cfg)


class TactileShowcaseTask(_ShowcaseTaskBase):
    task_id = TACTILE_TASK_ID
    env_name = "tactile-showcase-env"
    robot_name = "tactile-plate"
    runner_cls = TactileShowcaseRunner
    env_factory = staticmethod(tactile_showcase_env_cfg)


class RayCastShowcaseTask(_ShowcaseTaskBase):
    task_id = RAYCAST_TASK_ID
    env_name = "raycast-showcase-env"
    robot_name = "franka"
    runner_cls = RayCastShowcaseRunner
    env_factory = staticmethod(raycast_showcase_env_cfg)


class ContactShowcaseTask(_ShowcaseTaskBase):
    task_id = CONTACT_TASK_ID
    env_name = "contact-showcase-env"
    robot_name = "g1"
    runner_cls = ContactShowcaseRunner
    env_factory = staticmethod(contact_showcase_env_cfg)


class TerrainShowcaseTask(_ShowcaseTaskBase):
    task_id = TERRAIN_TASK_ID
    env_name = "terrain-showcase-env"
    robot_name = "g1"
    runner_cls = TerrainShowcaseRunner
    env_factory = staticmethod(terrain_showcase_env_cfg)


class CurriculumShowcaseTask(_ShowcaseTaskBase):
    task_id = CURRICULUM_TASK_ID
    env_name = "curriculum-showcase-env"
    robot_name = "g1"
    runner_cls = CurriculumShowcaseRunner
    env_factory = staticmethod(curriculum_showcase_env_cfg)


class ActuatorShowcaseTask(_ShowcaseTaskBase):
    task_id = ACTUATOR_TASK_ID
    env_name = "actuator-showcase-env"
    robot_name = "franka"
    runner_cls = ActuatorShowcaseRunner
    env_factory = staticmethod(actuator_showcase_env_cfg)


class MlpResidualActuatorShowcaseTask(_ShowcaseTaskBase):
    task_id = MLP_RESIDUAL_ACTUATOR_TASK_ID
    env_name = "mlp-residual-actuator-showcase-env"
    robot_name = "franka"
    runner_cls = MlpResidualActuatorShowcaseRunner
    env_factory = staticmethod(mlp_residual_actuator_showcase_env_cfg)


class RecordingShowcaseTask(_ShowcaseTaskBase):
    task_id = RECORDING_TASK_ID
    env_name = "recording-showcase-env"
    robot_name = "franka"
    runner_cls = RecordingShowcaseRunner
    env_factory = staticmethod(recording_showcase_env_cfg)


class _MaterialsDemoTask:
    """Material demo wrapper exposing one material property as a viewer slider.

    Unlike :class:`_ShowcaseTaskBase`, the env is a plain :class:`MaterialsDemoCfg`
    (``simulation`` + ``scene``), not a ``ManagerBasedRlEnvCfg`` — so ``genelab play``
    routes it to this class's own ``play()`` instead of the RL helper. The ``--vis`` /
    ``--steps`` / ``--env.scene.*`` overrides still land on ``cfg.env`` exactly as for
    every other task, and the runner reads the (possibly overridden) slider value back
    from the cfg before play.
    """

    task_id: str = ""
    env_name: str = ""
    cfg_factory: "Callable[[], MaterialsDemoCfg] | None" = None
    slider_label: str = ""
    slider_range: tuple[float, float] = (0.0, 1.0)
    get_value: "Callable[[MaterialsDemoCfg], float] | None" = None
    apply_value: "Callable[[MaterialsDemoCfg, float], None] | None" = None

    def __init__(self) -> None:
        if self.cfg_factory is None:
            raise RuntimeError(f"{type(self).__name__}.cfg_factory is unset")
        self.cfg = TaskCfg(
            name=self.task_id,
            env_name=self.env_name,
            robot_name="none",
            env=self.cfg_factory(),
            trainable=False,
        )

    def play(self, *, max_steps: int | None = None) -> None:
        env = cast(MaterialsDemoCfg, self.cfg.env)
        assert self.get_value is not None and self.apply_value is not None
        MaterialsDemoRunner(
            env,
            slider_label=self.slider_label,
            slider_range=self.slider_range,
            slider_value=self.get_value(env),
            apply_value=self.apply_value,
        ).play(max_steps=max_steps)

    def train(self) -> None:
        raise NotImplementedError(f"material demo {self.task_id} is play-only")


class SoftBodyShowcaseTask(_MaterialsDemoTask):
    task_id = SOFTBODY_TASK_ID
    env_name = "softbody-showcase-env"
    cfg_factory = staticmethod(soft_body_demo_cfg)
    slider_label = "stiffness"
    slider_range = SOFT_STIFFNESS_RANGE
    get_value = staticmethod(get_soft_stiffness)
    apply_value = staticmethod(set_soft_stiffness)


class FluidShowcaseTask(_MaterialsDemoTask):
    task_id = FLUID_TASK_ID
    env_name = "fluid-showcase-env"
    cfg_factory = staticmethod(fluid_demo_cfg)
    slider_label = "float density"
    slider_range = FLOAT_DENSITY_RANGE
    get_value = staticmethod(get_float_density)
    apply_value = staticmethod(set_float_density)


_MATERIAL_TASK_CLASSES: tuple[type[_MaterialsDemoTask], ...] = (
    SoftBodyShowcaseTask,
    FluidShowcaseTask,
)


_TASK_CLASSES: tuple[type[_ShowcaseTaskBase], ...] = (
    SensorsShowcaseTask,
    TactileShowcaseTask,
    RayCastShowcaseTask,
    ContactShowcaseTask,
    TerrainShowcaseTask,
    CurriculumShowcaseTask,
    ActuatorShowcaseTask,
    MlpResidualActuatorShowcaseTask,
    RecordingShowcaseTask,
)


_DESCRIPTIONS: dict[str, str] = {
    SENSORS_TASK_ID: (
        "Franka with CameraSensor + IMUSensor + FrameTransformerSensor + ForceTorqueSensor."
    ),
    TACTILE_TASK_ID: (
        "Flat dense tactile plate pressing a relief of shapes; per-probe depth → live heatmap."
    ),
    RAYCAST_TASK_ID: "Franka with RayCastSensor in three patterns (grid / ring / hemisphere).",
    CONTACT_TASK_ID: "Unitree G1 with ContactSensor air-time tracking on both feet.",
    TERRAIN_TASK_ID: "Unitree G1 dropped on a 1×5 row of the five built-in sub-terrains.",
    CURRICULUM_TASK_ID: "Unitree G1 on a 5×5 RandomRough grid driven by terrain_levels_vel.",
    ACTUATOR_TASK_ID: "Franka with IdealPDActuator on the arm (force-channel control).",
    MLP_RESIDUAL_ACTUATOR_TASK_ID: (
        "Franka with MlpResidualActuator on the arm (DC-motor base + TorchScript residual)."
    ),
    RECORDING_TASK_ID: "Franka with live PyQt/MPL plots and NPZ/CSV data dumps from an IMU.",
    SOFTBODY_TASK_ID: (
        "A FEM-elastic jelly ball dropped onto the ground, with a viewer stiffness slider."
    ),
    FLUID_TASK_ID: (
        "An MPM-liquid pool with a light box floating and a gold ball sinking (density slider)."
    ),
}


def register() -> None:
    """Register every showcase task. Idempotent under repeated imports."""

    all_classes: tuple[type[_ShowcaseTaskBase] | type[_MaterialsDemoTask], ...] = (
        *_TASK_CLASSES,
        *_MATERIAL_TASK_CLASSES,
    )
    for cls in all_classes:
        if cls.task_id in TASKS:
            continue
        register_task(
            cls.task_id,
            cls,
            description=_DESCRIPTIONS[cls.task_id],
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {cls.task_id} --vis",
                f"genelab play {cls.task_id} --vis --max-steps 400",
                f"genelab play {cls.task_id} --steps 400",
            ],
        )
