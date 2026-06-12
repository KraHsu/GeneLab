"""GeneLab extension entry point: registers the Go1 soft-terrain envs + trainable task."""

from genelab.configs import TaskCfg
from genelab.registry import ENVS, TASKS, register_env, register_task
from genelab_soft_terrain.go1_soft_env_cfg import go1_soft_stand_env_cfg
from genelab_soft_terrain.go1_soft_velocity_env_cfg import go1_soft_velocity_env_cfg
from genelab_soft_terrain.ppo_cfg import go1_soft_velocity_ppo_runner_cfg
from genelab_soft_terrain.sand import go1_sand_velocity_env_cfg

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
