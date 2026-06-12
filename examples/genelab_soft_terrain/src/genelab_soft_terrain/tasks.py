"""GeneLab extension entry point: registers the Go1 soft-terrain envs + trainable task."""

from genelab.configs import TaskCfg
from genelab.registry import ENVS, TASKS, register_env, register_task
from genelab_soft_terrain.go1_soft_env_cfg import go1_soft_stand_env_cfg
from genelab_soft_terrain.go1_soft_velocity_env_cfg import go1_soft_velocity_env_cfg
from genelab_soft_terrain.ppo_cfg import go1_soft_velocity_ppo_runner_cfg
from genelab_soft_terrain.sand import go1_sand_velocity_env_cfg

AIR_MATTRESS_TASK_ID = "Genelab-Air-Mattress-Demo-v0"
AIR_MATTRESS_ENV_NAME = "air-mattress-demo-env"
SOFT_STAND_ENV_NAME = "go1-soft-stand-env"
SOFT_VELOCITY_ENV_NAME = "go1-soft-velocity-env"
SOFT_VELOCITY_TASK_ID = "Genelab-Velocity-Soft-Unitree-Go1-v0"
SAND_VELOCITY_ENV_NAME = "go1-sand-velocity-env"
SAND_VELOCITY_TASK_ID = "Genelab-Velocity-Sand-Unitree-Go1-v0"
GO1_ROBOT_NAME = "go1"


def _build_soft_stand_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(go1_soft_stand_env_cfg(play=play))


def _build_soft_velocity_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(go1_soft_velocity_env_cfg(play=play))


class Go1SoftVelocityTask:
    """Trainable Go1 velocity-tracking on analytic deformable terrain."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=SOFT_VELOCITY_TASK_ID,
            env_name=SOFT_VELOCITY_ENV_NAME,
            robot_name=GO1_ROBOT_NAME,
            env=go1_soft_velocity_env_cfg(play=False),
            play_env=go1_soft_velocity_env_cfg(play=True),
            agent=go1_soft_velocity_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self, *, max_steps: int | None = None) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None, max_steps=max_steps)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


class AirMattressDemoTask:
    """Play-only demo: three balls coupled through one pneumatic air chamber.

    The env is a plain :class:`~genelab_soft_terrain.air_mattress.AirMattressDemoCfg`
    (simulation + scene + chamber model), not a ``ManagerBasedRlEnvCfg`` — ``genelab
    play`` routes to this class's own :meth:`play`, which steps the chamber loop with
    the in-viewer panel (capacity / stiffness sliders, fill readout, Drop button).
    """

    def __init__(self) -> None:
        from genelab_soft_terrain.air_mattress import air_mattress_demo_cfg

        self.cfg = TaskCfg(
            name=AIR_MATTRESS_TASK_ID,
            env_name=AIR_MATTRESS_ENV_NAME,
            robot_name="none",
            env=air_mattress_demo_cfg(),
            trainable=False,
        )

    def play(self, *, max_steps: int | None = None) -> None:
        from genelab_soft_terrain.air_mattress import AirMattressDemoCfg, run

        env = self.cfg.env
        assert isinstance(env, AirMattressDemoCfg)
        # ~60 s of show, re-dropping every ~5 s, paced to real time.
        run(env, steps=max_steps or 12_000, redrop_every=1_000, realtime=True)

    def train(self) -> None:
        raise NotImplementedError(f"{AIR_MATTRESS_TASK_ID} is play-only")


class Go1SandVelocityTask:
    """Trainable Go1 velocity-tracking on the analytic *sand* model (validate on MPM)."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=SAND_VELOCITY_TASK_ID,
            env_name=SAND_VELOCITY_ENV_NAME,
            robot_name=GO1_ROBOT_NAME,
            env=go1_sand_velocity_env_cfg(play=False),
            play_env=go1_sand_velocity_env_cfg(play=True),
            agent=go1_soft_velocity_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self, *, max_steps: int | None = None) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None, max_steps=max_steps)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


def register() -> None:
    if SOFT_STAND_ENV_NAME not in ENVS:
        register_env(
            SOFT_STAND_ENV_NAME,
            lambda: _build_soft_stand_env(play=False),
            description=(
                "Unitree Go1 standing on analytic deformable (soft) terrain — held up by the "
                "compliance force over a virtual surface, no rigid floor under the feet "
                "(ADR-0001 stage 0)."
            ),
            cfg_type=type(None),
        )
    if SOFT_VELOCITY_ENV_NAME not in ENVS:
        register_env(
            SOFT_VELOCITY_ENV_NAME,
            lambda: _build_soft_velocity_env(play=False),
            description="Unitree Go1 velocity tracking on analytic deformable terrain (ADR-0001 stage 1).",
            cfg_type=type(None),
        )
    if SOFT_VELOCITY_TASK_ID not in TASKS:
        register_task(
            SOFT_VELOCITY_TASK_ID,
            Go1SoftVelocityTask,
            description="PPO velocity tracking for Unitree Go1 on analytic deformable (soft) terrain.",
            cfg_type=TaskCfg,
            examples=[
                f"genelab train {SOFT_VELOCITY_TASK_ID} --num_envs 4096",
                f"genelab play {SOFT_VELOCITY_TASK_ID} --checkpoint PATH/model.pt",
            ],
        )
    if AIR_MATTRESS_TASK_ID not in TASKS:
        register_task(
            AIR_MATTRESS_TASK_ID,
            AirMattressDemoTask,
            description="Play-only pneumatic air-mattress demo: three balls coupled "
            "through one sealed chamber (cross-contact coupling).",
            cfg_type=TaskCfg,
            examples=[f"genelab play {AIR_MATTRESS_TASK_ID}"],
        )
    if SAND_VELOCITY_TASK_ID not in TASKS:
        register_task(
            SAND_VELOCITY_TASK_ID,
            Go1SandVelocityTask,
            description="PPO velocity tracking for Unitree Go1 on the analytic *sand* model "
            "(validate on granular MPM).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab train {SAND_VELOCITY_TASK_ID} --num_envs 4096",
                f"genelab play {SAND_VELOCITY_TASK_ID} --checkpoint PATH/model.pt",
            ],
        )
