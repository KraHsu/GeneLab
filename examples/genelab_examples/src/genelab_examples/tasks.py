"""Task registrations for the external example tasks."""

from genelab.configs import TaskCfg
from genelab.registry import TASKS, register_task

from genelab_examples import envs, robots
from genelab_examples.envs import create_rubiks_env
from genelab_examples.gui_panels.config import GuiPanelsEnvCfg
from genelab_examples.gui_panels.env import create_gui_panels_env
from genelab_examples.rubiks.config import RubiksEnvCfg


class RegisteredTask:
    """A runnable task resolved from the task registry."""

    def __init__(self, cfg: TaskCfg) -> None:
        self.cfg = cfg

    def play(self, *, max_steps: int | None = None) -> None:
        # Rubiks and the GUI-panels demo are scene-playback demos with their own fixed
        # loops, so the ``--max-steps`` hard cap (``max_steps``) does not apply here;
        # accepted to satisfy the ``Runnable`` contract.
        if self.cfg.env_name == "rubiks-play":
            if not isinstance(self.cfg.env, RubiksEnvCfg):
                raise TypeError("rubiks-play tasks require RubiksEnvCfg")
            create_rubiks_env(self.cfg.env).play()
            return
        if self.cfg.env_name == "gui-panels-demo":
            if not isinstance(self.cfg.env, GuiPanelsEnvCfg):
                raise TypeError("gui-panels-demo tasks require GuiPanelsEnvCfg")
            create_gui_panels_env(self.cfg.env).play()
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


def gui_panels_demo_task_cfg() -> TaskCfg:
    return TaskCfg(
        name="GeneLab-GUI-Panels-Demo-v0",
        env_name="gui-panels-demo",
        robot_name="none",
        env=GuiPanelsEnvCfg(),
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
    if "GeneLab-GUI-Panels-Demo-v0" not in TASKS:
        register_task(
            "GeneLab-GUI-Panels-Demo-v0",
            lambda: create_task(gui_panels_demo_task_cfg()),
            description="Cookbook of common ImGui viewer panels (sliders, toggles, dropdowns…).",
            cfg_type=TaskCfg,
            examples=["genelab play GeneLab-GUI-Panels-Demo-v0 --vis"],
        )
