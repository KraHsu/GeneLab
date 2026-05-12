"""Task registrations for the external example tasks."""

from genelab.configs import TaskCfg
from genelab.registry import TASKS, register_task

from genelab_examples import envs, robots
from genelab_examples.envs import create_rubiks_env, create_wuji_env
from genelab_examples.rubiks.config import RubiksEnvCfg
from genelab_examples.wuji_hand.config import WujiEnvCfg


class RegisteredTask:
    """A runnable task resolved from the task registry."""

    def __init__(self, cfg: TaskCfg) -> None:
        self.cfg = cfg

    def play(self) -> None:
        if self.cfg.env_name == "rubiks-play":
            if not isinstance(self.cfg.env, RubiksEnvCfg):
                raise TypeError("rubiks-play tasks require RubiksEnvCfg")
            create_rubiks_env(self.cfg.env).play()
            return
        if self.cfg.env_name == "wuji-hand-playback":
            if not isinstance(self.cfg.env, WujiEnvCfg):
                raise TypeError("wuji-hand-playback tasks require WujiEnvCfg")
            create_wuji_env(self.cfg.env).play()
            return
        raise RuntimeError(f"no play runner for env {self.cfg.env_name!r}")

    def train(self) -> None:
        if not self.cfg.trainable:
            raise NotImplementedError(f"training is not implemented for task {self.cfg.name}")
        raise NotImplementedError(f"training runner is not implemented for task {self.cfg.name}")


def rubiks_play_task_cfg() -> TaskCfg:
    return TaskCfg(
        name="GeneLab-Rubiks-Play-v0",
        env_name="rubiks-play",
        robot_name="rubiks-cube",
        env=RubiksEnvCfg(),
        trainable=False,
    )


def wuji_hand_playback_task_cfg() -> TaskCfg:
    return TaskCfg(
        name="GeneLab-Wuji-Hand-Playback-v0",
        env_name="wuji-hand-playback",
        robot_name="wuji-hand",
        env=WujiEnvCfg(),
        trainable=False,
    )


def create_task(cfg: TaskCfg) -> RegisteredTask:
    return RegisteredTask(cfg)


def register() -> None:
    robots.register()
    envs.register()
    if "GeneLab-Rubiks-Play-v0" not in TASKS:
        register_task(
            "GeneLab-Rubiks-Play-v0",
            lambda: create_task(rubiks_play_task_cfg()),
            description="Example force-driven Rubik's cube Genesis scene.",
            cfg_type=TaskCfg,
        )
    if "GeneLab-Wuji-Hand-Playback-v0" not in TASKS:
        register_task(
            "GeneLab-Wuji-Hand-Playback-v0",
            lambda: create_task(wuji_hand_playback_task_cfg()),
            description="Example fixed-trajectory Wuji hand Genesis scene.",
            cfg_type=TaskCfg,
        )
