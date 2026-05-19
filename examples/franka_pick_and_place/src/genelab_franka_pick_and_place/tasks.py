"""GeneLab extension entry point: registers the Franka pick-and-place task."""

from genelab.configs import TaskCfg
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    register_env,
    register_robot,
    register_task,
)

from genelab_franka_pick_and_place.env_cfg import (
    franka_pick_and_place_cartesian_env_cfg,
    franka_pick_and_place_env_cfg,
)
from genelab_franka_pick_and_place.ppo_cfg import franka_pick_and_place_ppo_runner_cfg
from genelab_franka_pick_and_place.robot import (
    FrankaPickAndPlaceRobotCfg,
    get_franka_pick_and_place_robot_cfg,
)
from genelab_franka_pick_and_place.skrl_cfg import franka_pick_and_place_skrl_cfg

TASK_ID = "GeneLab-Franka-Pick-And-Place-v0"
CARTESIAN_TASK_ID = "GeneLab-Franka-Pick-And-Place-Cartesian-v0"
SKRL_TASK_ID = "GeneLab-Franka-Pick-And-Place-skrl-v0"
ROBOT_NAME = "franka-pick-and-place"
ENV_NAME = "franka-pick-and-place-env"
CARTESIAN_ENV_NAME = "franka-pick-and-place-cartesian-env"


def _build_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(franka_pick_and_place_env_cfg(play=play))


def _build_cartesian_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(franka_pick_and_place_cartesian_env_cfg(play=play))


class FrankaPickAndPlaceTask:
    """Trainable Franka pick-and-place task. Delegates to ``genelab.rl.runner``."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=TASK_ID,
            env_name=ENV_NAME,
            robot_name=ROBOT_NAME,
            env=franka_pick_and_place_env_cfg(play=False),
            play_env=franka_pick_and_place_env_cfg(play=True),
            agent=franka_pick_and_place_ppo_runner_cfg(),
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


class FrankaPickAndPlaceCartesianTask:
    """Cartesian (panda-gym 4-DoF) variant of :class:`FrankaPickAndPlaceTask` —
    swaps the action surface for ``DifferentialIKAction`` (orientation locked) +
    ``ContinuousGripperAction`` while reusing the same scene, reward shaping, and
    PPO runner config."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=CARTESIAN_TASK_ID,
            env_name=CARTESIAN_ENV_NAME,
            robot_name=ROBOT_NAME,
            env=franka_pick_and_place_cartesian_env_cfg(play=False),
            play_env=franka_pick_and_place_cartesian_env_cfg(play=True),
            agent=franka_pick_and_place_ppo_runner_cfg(),
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


class FrankaPickAndPlaceSkrlTask:
    """skrl variant of :class:`FrankaPickAndPlaceTask` — same scene / reward shaping,
    trained through the skrl backend (PPO) instead of RSL-RL. The backend is selected
    automatically from the ``SkrlAgentCfg`` agent config."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=SKRL_TASK_ID,
            env_name=ENV_NAME,
            robot_name=ROBOT_NAME,
            env=franka_pick_and_place_env_cfg(play=False),
            play_env=franka_pick_and_place_env_cfg(play=True),
            agent=franka_pick_and_place_skrl_cfg(),
            trainable=True,
        )

    def play(self) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None)

    def train(self) -> None:
        from genelab.rl import train_task

        train_task(self.cfg.name, self.cfg.agent)


def register() -> None:
    if ROBOT_NAME not in ROBOTS:
        register_robot(
            ROBOT_NAME,
            get_franka_pick_and_place_robot_cfg,
            description="Franka Panda arm + parallel gripper at the world origin (pick-and-place).",
            cfg_type=FrankaPickAndPlaceRobotCfg,
        )
    if ENV_NAME not in ENVS:
        register_env(
            ENV_NAME,
            lambda: _build_env(play=False),
            description="Franka pick-and-place on a flat plane with a 4 cm cube.",
            cfg_type=type(None),
        )
    if CARTESIAN_ENV_NAME not in ENVS:
        register_env(
            CARTESIAN_ENV_NAME,
            lambda: _build_cartesian_env(play=False),
            description=(
                "Franka pick-and-place with 4-DoF Cartesian control "
                "(differential IK + binary gripper, panda-gym parity)."
            ),
            cfg_type=type(None),
        )
    if TASK_ID not in TASKS:
        register_task(
            TASK_ID,
            FrankaPickAndPlaceTask,
            description="PPO Franka Panda pick-and-place (panda-gym-style dense reward).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {TASK_ID} --vis",
                f"genelab play {TASK_ID} --agent trained --checkpoint PATH/model.pt",
                f"genelab train {TASK_ID} --num-envs 2048 --max-iterations 500",
            ],
        )
    if CARTESIAN_TASK_ID not in TASKS:
        register_task(
            CARTESIAN_TASK_ID,
            FrankaPickAndPlaceCartesianTask,
            description=(
                "PPO Franka Panda pick-and-place with 4-DoF Cartesian (EE-delta IK + "
                "continuous gripper) action space — panda-gym ``PandaPickAndPlace`` parity."
            ),
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {CARTESIAN_TASK_ID} --vis",
                f"genelab train {CARTESIAN_TASK_ID} --num-envs 16 --max-iterations 2",
                f"genelab train {CARTESIAN_TASK_ID} --num-envs 2048 --max-iterations 500",
            ],
        )
    if SKRL_TASK_ID not in TASKS:
        register_task(
            SKRL_TASK_ID,
            FrankaPickAndPlaceSkrlTask,
            description=(
                "skrl PPO Franka Panda pick-and-place — same task as "
                f"{TASK_ID}, trained through the skrl backend instead of RSL-RL."
            ),
            cfg_type=TaskCfg,
            examples=[
                f"genelab play {SKRL_TASK_ID} --vis",
                f"genelab train {SKRL_TASK_ID} --num-envs 16 --max-iterations 2",
                f"genelab train {SKRL_TASK_ID} --num-envs 2048 --max-iterations 12000",
            ],
        )
