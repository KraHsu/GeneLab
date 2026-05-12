"""Environment runners owned by the downstream package."""

from my_robot_project.config import MyEnvCfg


class MyPickPlaceEnv:
    def __init__(self, cfg: MyEnvCfg | None = None) -> None:
        self.cfg = cfg or MyEnvCfg()

    def play(self) -> None:
        print(f"Run MyPickPlaceEnv with Genesis for {self.cfg.scene.steps} steps")
