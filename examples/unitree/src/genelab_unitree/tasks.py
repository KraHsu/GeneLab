"""GeneLab extension entry point: registers the G1 velocity tracking + motion imitation tasks."""

from genelab.configs import TaskCfg
from genelab.registry import (
    ENVS,
    TASKS,
    register_env,
    register_task,
)
from genelab_unitree.g1 import (
    unitree_g1_ppo_runner_cfg,
    unitree_g1_tracking_env_cfg,
    unitree_g1_tracking_ppo_runner_cfg,
    unitree_g1_velocity_env_cfg,
    unitree_g1_velocity_rough_env_cfg,
    unitree_g1_velocity_rough_ppo_runner_cfg,
)
from genelab_unitree.go1 import (
    unitree_go1_ppo_runner_cfg,
    unitree_go1_velocity_env_cfg,
    unitree_go1_velocity_rough_env_cfg,
    unitree_go1_velocity_rough_ppo_runner_cfg,
)
from genelab_unitree.go2w import (
    unitree_go2w_ppo_runner_cfg,
    unitree_go2w_velocity_env_cfg,
)

VELOCITY_TASK_ID = "Genelab-Velocity-Flat-Unitree-G1-v0"
VELOCITY_ROUGH_TASK_ID = "Genelab-Velocity-Rough-Unitree-G1-v0"
TRACKING_TASK_ID = "Genelab-Tracking-Flat-Unitree-G1-v0"
# Matches the name registered by `genelab.asset_zoo.unitree_g1`.
ROBOT_NAME = "g1"
VELOCITY_ENV_NAME = "g1-velocity-flat-env"
VELOCITY_ROUGH_ENV_NAME = "g1-velocity-rough-env"
TRACKING_ENV_NAME = "g1-tracking-flat-env"

GO1_VELOCITY_TASK_ID = "Genelab-Velocity-Flat-Unitree-Go1-v0"
GO1_VELOCITY_ROUGH_TASK_ID = "Genelab-Velocity-Rough-Unitree-Go1-v0"
# Matches the name registered by `genelab.asset_zoo.unitree_go1`.
GO1_ROBOT_NAME = "go1"
GO1_VELOCITY_ENV_NAME = "go1-velocity-flat-env"
GO1_VELOCITY_ROUGH_ENV_NAME = "go1-velocity-rough-env"

GO2W_VELOCITY_TASK_ID = "Genelab-Velocity-Flat-Unitree-Go2W-v0"
# Matches the name registered by `genelab.asset_zoo.unitree_go2w`.
GO2W_ROBOT_NAME = "go2w"
GO2W_VELOCITY_ENV_NAME = "go2w-velocity-flat-env"
# Wheeled-legged curriculum stage 1: wheels locked into rigid feet so the legs must learn to
# step (incl. sideways crab-walk). Stage 2 is the regular GO2W_VELOCITY task, warm-started
# from this stage's checkpoint with the wheels rolling again.
GO2W_CRAB_STAGE1_TASK_ID = "Genelab-Velocity-Flat-Unitree-Go2W-CrabStage1-v0"
GO2W_CRAB_STAGE1_ENV_NAME = "go2w-crab-stage1-env"


def _build_velocity_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_g1_velocity_env_cfg(play=play))


def _build_velocity_rough_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_g1_velocity_rough_env_cfg(play=play))


def _build_tracking_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_g1_tracking_env_cfg(play=play))


class G1VelocityTask:
    """Trainable task wrapper. Delegates to ``genelab.rl.runner`` for play / train."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=VELOCITY_TASK_ID,
            env_name=VELOCITY_ENV_NAME,
            robot_name=ROBOT_NAME,
            env=unitree_g1_velocity_env_cfg(play=False),
            play_env=unitree_g1_velocity_env_cfg(play=True),
            agent=unitree_g1_ppo_runner_cfg(),
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


class G1VelocityRoughTask:
    """Trainable task wrapper for velocity tracking on rough terrain."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=VELOCITY_ROUGH_TASK_ID,
            env_name=VELOCITY_ROUGH_ENV_NAME,
            robot_name=ROBOT_NAME,
            env=unitree_g1_velocity_rough_env_cfg(play=False),
            play_env=unitree_g1_velocity_rough_env_cfg(play=True),
            agent=unitree_g1_velocity_rough_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


class G1TrackingTask:
    """Motion-imitation task wrapper.

    The clip defaults to the asset-zoo LAFAN1 ``dance1_subject2`` NPZ. Swap it by editing
    ``unitree_g1_tracking_env_cfg`` or assigning a new path into
    ``self.cfg.env.commands_cfg["motion"].motion_file``.
    """

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=TRACKING_TASK_ID,
            env_name=TRACKING_ENV_NAME,
            robot_name=ROBOT_NAME,
            env=unitree_g1_tracking_env_cfg(play=False),
            play_env=unitree_g1_tracking_env_cfg(play=True),
            agent=unitree_g1_tracking_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self, *, max_steps: int | None = None) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, agent="zero", max_steps=max_steps)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


class Go1VelocityTask:
    """Trainable Go1 flat-ground velocity-tracking task wrapper."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=GO1_VELOCITY_TASK_ID,
            env_name=GO1_VELOCITY_ENV_NAME,
            robot_name=GO1_ROBOT_NAME,
            env=unitree_go1_velocity_env_cfg(play=False),
            play_env=unitree_go1_velocity_env_cfg(play=True),
            agent=unitree_go1_ppo_runner_cfg(),
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


class Go1VelocityRoughTask:
    """Trainable Go1 complex-terrain velocity-tracking task wrapper."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=GO1_VELOCITY_ROUGH_TASK_ID,
            env_name=GO1_VELOCITY_ROUGH_ENV_NAME,
            robot_name=GO1_ROBOT_NAME,
            env=unitree_go1_velocity_rough_env_cfg(play=False),
            play_env=unitree_go1_velocity_rough_env_cfg(play=True),
            agent=unitree_go1_velocity_rough_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


class Go2WVelocityTask:
    """Trainable Go2-W flat-ground velocity-tracking task wrapper (wheeled quadruped)."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=GO2W_VELOCITY_TASK_ID,
            env_name=GO2W_VELOCITY_ENV_NAME,
            robot_name=GO2W_ROBOT_NAME,
            env=unitree_go2w_velocity_env_cfg(play=False),
            play_env=unitree_go2w_velocity_env_cfg(play=True),
            agent=unitree_go2w_ppo_runner_cfg(),
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


class Go2WCrabStage1Task:
    """Stage 1 of the Go2-W wheeled-legged curriculum: wheels locked, learn the legged
    (incl. lateral crab-walk) gait. Warm-start the regular Go2-W task from its checkpoint."""

    def __init__(self) -> None:
        agent = unitree_go2w_ppo_runner_cfg()
        # Keep stage-1 logs / checkpoints separate from the rolling-wheel task.
        agent.experiment_name = "go2w_crab_stage1"
        self.cfg = TaskCfg(
            name=GO2W_CRAB_STAGE1_TASK_ID,
            env_name=GO2W_CRAB_STAGE1_ENV_NAME,
            robot_name=GO2W_ROBOT_NAME,
            env=unitree_go2w_velocity_env_cfg(play=False, lock_wheels=True),
            play_env=unitree_go2w_velocity_env_cfg(play=True, lock_wheels=True),
            agent=agent,
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


def _build_go2w_velocity_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_go2w_velocity_env_cfg(play=play))


def _build_go2w_crab_stage1_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_go2w_velocity_env_cfg(play=play, lock_wheels=True))


def _build_go1_velocity_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_go1_velocity_env_cfg(play=play))


def _build_go1_velocity_rough_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(unitree_go1_velocity_rough_env_cfg(play=play))


def register() -> None:
    # The G1 robot itself is registered by `genelab.asset_zoo.unitree_g1`; no robot registration here.
    if VELOCITY_ENV_NAME not in ENVS:
        register_env(
            VELOCITY_ENV_NAME,
            lambda: _build_velocity_env(play=False),
            description="Unitree G1 velocity tracking on flat ground.",
            cfg_type=type(None),
        )
    if VELOCITY_ROUGH_ENV_NAME not in ENVS:
        register_env(
            VELOCITY_ROUGH_ENV_NAME,
            lambda: _build_velocity_rough_env(play=False),
            description="Unitree G1 velocity tracking on heightfield rough terrain.",
            cfg_type=type(None),
        )
    if TRACKING_ENV_NAME not in ENVS:
        register_env(
            TRACKING_ENV_NAME,
            lambda: _build_tracking_env(play=False),
            description="Unitree G1 motion imitation (BeyondMimic-style) on flat ground.",
            cfg_type=type(None),
        )
    if VELOCITY_TASK_ID not in TASKS:
        register_task(
            VELOCITY_TASK_ID,
            G1VelocityTask,
            description="PPO velocity tracking for Unitree G1 (flat).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {VELOCITY_TASK_ID}",
                f"genelab play {VELOCITY_TASK_ID} --agent random",
                f"genelab play {VELOCITY_TASK_ID} --agent trained --checkpoint PATH/model.pt",
                f"genelab play {VELOCITY_TASK_ID} --num_envs 1 --checkpoint PATH/model.pt  "
                "# in-viewport ImGui sliders (uv sync --extra imgui)",
                f"genelab train {VELOCITY_TASK_ID} --num_envs 4096",
                f"genelab train {VELOCITY_TASK_ID} --num_envs 4096 --gpus 2",
            ],
        )
    if VELOCITY_ROUGH_TASK_ID not in TASKS:
        register_task(
            VELOCITY_ROUGH_TASK_ID,
            G1VelocityRoughTask,
            description="PPO velocity tracking for Unitree G1 on rough terrain.",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {VELOCITY_ROUGH_TASK_ID}",
                f"genelab play {VELOCITY_ROUGH_TASK_ID} --agent trained --checkpoint PATH/model.pt",
                f"genelab train {VELOCITY_ROUGH_TASK_ID} --num_envs 4096",
                f"genelab train {VELOCITY_ROUGH_TASK_ID} --num_envs 4096 --gpus 2",
            ],
        )
    if TRACKING_TASK_ID not in TASKS:
        register_task(
            TRACKING_TASK_ID,
            G1TrackingTask,
            description=(
                "PPO motion imitation for Unitree G1 on flat ground. "
                "Default clip: asset-zoo LAFAN1 dance1_subject2."
            ),
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {TRACKING_TASK_ID} --agent zero --vis",
                f"genelab train {TRACKING_TASK_ID} --num_envs 4096",
                f"genelab train {TRACKING_TASK_ID} --num_envs 4096 --gpus 2",
            ],
        )
    if GO1_VELOCITY_ENV_NAME not in ENVS:
        register_env(
            GO1_VELOCITY_ENV_NAME,
            lambda: _build_go1_velocity_env(play=False),
            description="Unitree Go1 velocity tracking on flat ground.",
            cfg_type=type(None),
        )
    if GO1_VELOCITY_ROUGH_ENV_NAME not in ENVS:
        register_env(
            GO1_VELOCITY_ROUGH_ENV_NAME,
            lambda: _build_go1_velocity_rough_env(play=False),
            description="Unitree Go1 velocity tracking on complex heightfield terrain.",
            cfg_type=type(None),
        )
    if GO1_VELOCITY_TASK_ID not in TASKS:
        register_task(
            GO1_VELOCITY_TASK_ID,
            Go1VelocityTask,
            description="PPO velocity tracking for Unitree Go1 quadruped (flat).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {GO1_VELOCITY_TASK_ID}",
                f"genelab train {GO1_VELOCITY_TASK_ID} --num_envs 4096",
            ],
        )
    if GO1_VELOCITY_ROUGH_TASK_ID not in TASKS:
        register_task(
            GO1_VELOCITY_ROUGH_TASK_ID,
            Go1VelocityRoughTask,
            description="PPO velocity tracking for Unitree Go1 quadruped on complex terrain.",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {GO1_VELOCITY_ROUGH_TASK_ID}",
                f"genelab play {GO1_VELOCITY_ROUGH_TASK_ID} --agent trained --checkpoint PATH/model.pt",
                f"genelab train {GO1_VELOCITY_ROUGH_TASK_ID} --num_envs 4096",
                f"genelab train {GO1_VELOCITY_ROUGH_TASK_ID} --num_envs 4096 --gpus 2",
            ],
        )
    if GO2W_VELOCITY_ENV_NAME not in ENVS:
        register_env(
            GO2W_VELOCITY_ENV_NAME,
            lambda: _build_go2w_velocity_env(play=False),
            description="Unitree Go2-W wheeled quadruped velocity tracking on flat ground.",
            cfg_type=type(None),
        )
    if GO2W_VELOCITY_TASK_ID not in TASKS:
        register_task(
            GO2W_VELOCITY_TASK_ID,
            Go2WVelocityTask,
            description=(
                "PPO velocity tracking for Unitree Go2-W wheeled quadruped (flat); position "
                "legs + velocity wheels."
            ),
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {GO2W_VELOCITY_TASK_ID}",
                f"genelab train {GO2W_VELOCITY_TASK_ID} --num_envs 4096",
            ],
        )
    if GO2W_CRAB_STAGE1_ENV_NAME not in ENVS:
        register_env(
            GO2W_CRAB_STAGE1_ENV_NAME,
            lambda: _build_go2w_crab_stage1_env(play=False),
            description="Go2-W with wheels locked (crab-walk stage 1 of the wheeled-legged curriculum).",
            cfg_type=type(None),
        )
    if GO2W_CRAB_STAGE1_TASK_ID not in TASKS:
        register_task(
            GO2W_CRAB_STAGE1_TASK_ID,
            Go2WCrabStage1Task,
            description=(
                "Go2-W wheeled-legged curriculum stage 1: wheels locked into rigid feet so the "
                "legs learn to step (incl. sideways). Warm-start the regular Go2-W task from this."
            ),
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {GO2W_CRAB_STAGE1_TASK_ID}",
                f"genelab train {GO2W_CRAB_STAGE1_TASK_ID} --num_envs 4096",
            ],
        )
