"""GeneLab extension entry point: registers single + double inverted-pendulum tasks."""

from genelab.configs import TaskCfg
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    register_env,
    register_robot,
    register_task,
)

from genelab_inverted_pendulum.double import (
    DoubleInvertedPendulumRobotCfg,
    double_inverted_pendulum_env_cfg,
    double_inverted_pendulum_ppo_runner_cfg,
    get_double_inverted_pendulum_robot_cfg,
)
from genelab_inverted_pendulum.single import (
    InvertedPendulumRobotCfg,
    get_inverted_pendulum_robot_cfg,
    inverted_pendulum_env_cfg,
    inverted_pendulum_memory_env_cfg,
    inverted_pendulum_memory_mlp_runner_cfg,
    inverted_pendulum_memory_rnn_runner_cfg,
    inverted_pendulum_ppo_runner_cfg,
    inverted_pendulum_skrl_agent_cfg,
)

INVERTED_PENDULUM_TASK_ID = "GeneLab-Inverted-Pendulum-v0"
INVERTED_PENDULUM_SKRL_TASK_ID = "GeneLab-Inverted-Pendulum-Skrl-v0"
DOUBLE_PENDULUM_TASK_ID = "GeneLab-Double-Inverted-Pendulum-v0"
# "Recall the target": a task that genuinely needs memory, so an LSTM beats an MLP.
MEMORY_RNN_TASK_ID = "GeneLab-Inverted-Pendulum-Memory-Rnn-v0"
MEMORY_MLP_TASK_ID = "GeneLab-Inverted-Pendulum-Memory-Mlp-v0"
ROBOT_NAME_SINGLE = "inverted-pendulum"
ROBOT_NAME_DOUBLE = "double-inverted-pendulum"
ENV_NAME_SINGLE = "inverted-pendulum-env"
ENV_NAME_DOUBLE = "double-inverted-pendulum-env"
ENV_NAME_MEMORY = "inverted-pendulum-memory-env"


def _build_single_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(inverted_pendulum_env_cfg(play=play))


def _build_double_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(double_inverted_pendulum_env_cfg(play=play))


def _build_memory_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(inverted_pendulum_memory_env_cfg(play=play))


class InvertedPendulumTask:
    """Trainable single inverted-pendulum task. Delegates to ``genelab.rl.runner``."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=INVERTED_PENDULUM_TASK_ID,
            env_name=ENV_NAME_SINGLE,
            robot_name=ROBOT_NAME_SINGLE,
            env=inverted_pendulum_env_cfg(play=False),
            play_env=inverted_pendulum_env_cfg(play=True),
            agent=inverted_pendulum_ppo_runner_cfg(),
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


class InvertedPendulumSkrlTask:
    """Single inverted-pendulum trained by the **skrl** PPO backend.

    Reuses the same env/robot as :class:`InvertedPendulumTask`; only the agent cfg
    differs (``SkrlAgentCfg`` instead of ``RslRlOnPolicyRunnerCfg``), which is what
    routes it to the skrl backend. Exists so ``genelab train/play/eval/export``
    exercise skrl at runtime.
    """

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=INVERTED_PENDULUM_SKRL_TASK_ID,
            env_name=ENV_NAME_SINGLE,
            robot_name=ROBOT_NAME_SINGLE,
            env=inverted_pendulum_env_cfg(play=False),
            play_env=inverted_pendulum_env_cfg(play=True),
            agent=inverted_pendulum_skrl_agent_cfg(),
            trainable=True,
        )

    def play(self, *, max_steps: int | None = None) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None, max_steps=max_steps)

    def train(self) -> None:
        from genelab.rl import SkrlAgentCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, SkrlAgentCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


class DoubleInvertedPendulumTask:
    """Trainable double inverted-pendulum task."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=DOUBLE_PENDULUM_TASK_ID,
            env_name=ENV_NAME_DOUBLE,
            robot_name=ROBOT_NAME_DOUBLE,
            env=double_inverted_pendulum_env_cfg(play=False),
            play_env=double_inverted_pendulum_env_cfg(play=True),
            agent=double_inverted_pendulum_ppo_runner_cfg(),
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


class _MemoryTask:
    """A "recall the target" cart-pole trained by rsl_rl PPO.

    The RNN showcase and its MLP baseline share the same memory env and differ only in the
    agent cfg (recurrent vs feed-forward); each subclass binds a task id + agent-cfg factory.
    """

    task_id: str

    def __init__(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg

        agent = self._agent_cfg()
        assert isinstance(agent, RslRlOnPolicyRunnerCfg)  # narrows for train()
        self.cfg = TaskCfg(
            name=self.task_id,
            env_name=ENV_NAME_MEMORY,
            robot_name=ROBOT_NAME_SINGLE,
            env=inverted_pendulum_memory_env_cfg(play=False),
            play_env=inverted_pendulum_memory_env_cfg(play=True),
            agent=agent,
            trainable=True,
        )

    @staticmethod
    def _agent_cfg():
        raise NotImplementedError

    def play(self, *, max_steps: int | None = None) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None, max_steps=max_steps)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError(f"agent cfg has unexpected type {type(agent).__name__}")
        train_task(self.cfg.name, agent)


class MemoryRnnTask(_MemoryTask):
    """Recall-the-target cart-pole with an LSTM policy — needs memory, so the RNN solves it."""

    task_id = MEMORY_RNN_TASK_ID
    _agent_cfg = staticmethod(inverted_pendulum_memory_rnn_runner_cfg)


class MemoryMlpTask(_MemoryTask):
    """Recall-the-target cart-pole with an MLP — structurally cannot remember the target."""

    task_id = MEMORY_MLP_TASK_ID
    _agent_cfg = staticmethod(inverted_pendulum_memory_mlp_runner_cfg)


def register() -> None:
    if ROBOT_NAME_SINGLE not in ROBOTS:
        register_robot(
            ROBOT_NAME_SINGLE,
            get_inverted_pendulum_robot_cfg,
            description="Cart with a single hinge-mounted inverted pole.",
            cfg_type=InvertedPendulumRobotCfg,
        )
    if ROBOT_NAME_DOUBLE not in ROBOTS:
        register_robot(
            ROBOT_NAME_DOUBLE,
            get_double_inverted_pendulum_robot_cfg,
            description="Cart with two serially linked inverted poles.",
            cfg_type=DoubleInvertedPendulumRobotCfg,
        )
    if ENV_NAME_SINGLE not in ENVS:
        register_env(
            ENV_NAME_SINGLE,
            lambda: _build_single_env(play=False),
            description="Single inverted-pendulum balancing on a flat plane.",
            cfg_type=type(None),
        )
    if ENV_NAME_DOUBLE not in ENVS:
        register_env(
            ENV_NAME_DOUBLE,
            lambda: _build_double_env(play=False),
            description="Double inverted-pendulum balancing on a flat plane.",
            cfg_type=type(None),
        )
    if ENV_NAME_MEMORY not in ENVS:
        register_env(
            ENV_NAME_MEMORY,
            lambda: _build_memory_env(play=False),
            description="Cart-pole that must recall a target flashed only at the episode start.",
            cfg_type=type(None),
        )
    if INVERTED_PENDULUM_TASK_ID not in TASKS:
        register_task(
            INVERTED_PENDULUM_TASK_ID,
            InvertedPendulumTask,
            description="PPO inverted-pendulum balancing (single pole).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {INVERTED_PENDULUM_TASK_ID} --vis",
                f"genelab play {INVERTED_PENDULUM_TASK_ID} --agent trained "
                "--checkpoint PATH/model.pt",
                f"genelab train {INVERTED_PENDULUM_TASK_ID} --num_envs 4096 --max_iterations 100",
            ],
        )
    if INVERTED_PENDULUM_SKRL_TASK_ID not in TASKS:
        register_task(
            INVERTED_PENDULUM_SKRL_TASK_ID,
            InvertedPendulumSkrlTask,
            description="skrl PPO inverted-pendulum balancing (single pole).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {INVERTED_PENDULUM_SKRL_TASK_ID} --vis",
                f"genelab train {INVERTED_PENDULUM_SKRL_TASK_ID} --num_envs 4096 --max_iterations 4800",
                f"genelab eval {INVERTED_PENDULUM_SKRL_TASK_ID} PATH/checkpoints/agent_<N>.pt",
            ],
        )
    if DOUBLE_PENDULUM_TASK_ID not in TASKS:
        register_task(
            DOUBLE_PENDULUM_TASK_ID,
            DoubleInvertedPendulumTask,
            description="PPO double-inverted-pendulum balancing (two stacked poles).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {DOUBLE_PENDULUM_TASK_ID} --vis",
                f"genelab play {DOUBLE_PENDULUM_TASK_ID} --agent trained "
                "--checkpoint PATH/model.pt",
                f"genelab train {DOUBLE_PENDULUM_TASK_ID} --num_envs 4096 --max_iterations 200",
            ],
        )
    if MEMORY_RNN_TASK_ID not in TASKS:
        register_task(
            MEMORY_RNN_TASK_ID,
            MemoryRnnTask,
            description="LSTM PPO on the 'recall the target' cart-pole — the task that needs memory.",
            cfg_type=TaskCfg,
            examples=[
                f"genelab train {MEMORY_RNN_TASK_ID} --num_envs 2048 --max_iterations 400",
                f"genelab play {MEMORY_RNN_TASK_ID} --agent trained --checkpoint PATH/model.pt --vis",
            ],
        )
    if MEMORY_MLP_TASK_ID not in TASKS:
        register_task(
            MEMORY_MLP_TASK_ID,
            MemoryMlpTask,
            description="Feed-forward baseline on the recall task — cannot remember the target.",
            cfg_type=TaskCfg,
            examples=[
                f"genelab train {MEMORY_MLP_TASK_ID} --num_envs 2048 --max_iterations 400",
                f"genelab eval {MEMORY_MLP_TASK_ID} PATH/model.pt",
            ],
        )
