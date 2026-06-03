"""Genesis environment registrations for the Wuji hand example tasks."""

from genelab.registry import ENVS, register_env

from genelab_wuji.wuji_hand.config import WujiEnvCfg
from genelab_wuji.wuji_hand.sim import WujiHandRunConfig, run_wuji_hand


class WujiHandPlaybackEnv:
    """Genesis environment for fixed-trajectory Wuji hand playback."""

    def __init__(self, cfg: WujiEnvCfg | None = None) -> None:
        self.cfg = cfg or WujiEnvCfg()

    def play(self) -> None:
        from genelab.cache import ensure_project_cache

        ensure_project_cache()

        cfg = self.cfg
        run_wuji_hand(
            WujiHandRunConfig(
                side=cfg.robot.side,
                desc_dir=cfg.robot.desc_dir,
                trajectory=cfg.robot.trajectory,
                vis=cfg.simulation.vis,
                gpu=cfg.simulation.gpu,
                steps=cfg.simulation.steps,
                dt=cfg.simulation.dt,
                reset_interval=cfg.reset_interval,
            )
        )


def create_wuji_env(cfg: WujiEnvCfg | None = None) -> WujiHandPlaybackEnv:
    return WujiHandPlaybackEnv(cfg)


def register() -> None:
    if "wuji-hand-playback" not in ENVS:
        register_env(
            "wuji-hand-playback",
            lambda: create_wuji_env(),
            description="Example Genesis scene that plays a fixed Wuji hand trajectory.",
            cfg_type=WujiEnvCfg,
        )
