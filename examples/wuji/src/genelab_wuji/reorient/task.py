"""Registration for the WUJI-hand SO(3) in-hand reorientation task (RSL-RL PPO)."""

from genelab.configs import TaskCfg
from genelab.registry import ENVS, TASKS, register_env, register_task

from genelab_wuji.reorient.env_cfg import wuji_hand_reorient_env_cfg
from genelab_wuji.reorient.ppo import wuji_hand_reorient_ppo_runner_cfg

TASK_ID = "Genelab-Reorient-Wuji-Hand-v0"
ENV_NAME = "wuji-reorient"
ROBOT_NAME = "wuji-hand"


def _build_env(play: bool = False):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(wuji_hand_reorient_env_cfg(play=play))


class WujiReorientTask:
    """Trainable in-hand reorientation task; delegates to ``genelab.rl`` for train / play."""

    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name=TASK_ID,
            env_name=ENV_NAME,
            robot_name=ROBOT_NAME,
            env=wuji_hand_reorient_env_cfg(play=False),
            play_env=wuji_hand_reorient_env_cfg(play=True),
            agent=wuji_hand_reorient_ppo_runner_cfg(),
            trainable=True,
        )

    def play(self) -> None:
        from genelab.rl import play_task

        play_task(self.cfg.name, checkpoint=None)

    def train(self) -> None:
        from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

        agent = self.cfg.agent
        if not isinstance(agent, RslRlOnPolicyRunnerCfg):
            raise TypeError("reorient task requires an RSL-RL on-policy runner cfg")
        train_task(self.cfg.name, agent)


def register() -> None:
    if ENV_NAME not in ENVS:
        register_env(
            ENV_NAME,
            lambda: _build_env(play=False),
            description="WUJI hand in-hand SO(3) cube reorientation (fixed-base, palm-up).",
            cfg_type=type(None),
        )
    if TASK_ID not in TASKS:
        register_task(
            TASK_ID,
            WujiReorientTask,
            description="RSL-RL PPO in-hand cube reorientation for the WUJI Hand (SO(3) goals).",
            cfg_type=TaskCfg,
            examples=[
                f"genelab train {TASK_ID} --num_envs 4096 --gpu",
                f"genelab play {TASK_ID} --checkpoint PATH/model.pt --vis",
            ],
        )
