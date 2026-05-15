"""Contact-showcase runner: prints per-foot contact + air-time evolution."""

from typing import TYPE_CHECKING, cast

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from genelab.sensor import ContactData


class ContactShowcaseRunner(ShowcaseRunner):
    """Drives G1 at its default pose so both feet settle into ground contact.

    Output dump (``logs/showcase/contact/air_time.log``) at each interval:

    ::

        step=NNNN
            left_ankle_roll_link  contact=BOOL  current_contact=… current_air=…
                                  last_contact=…    last_air=…
            right_ankle_roll_link contact=BOOL  …
    """

    task_slug = "contact"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg, log_interval=20)

    def _dump(self, env: "ManagerBasedRlEnv", step: int) -> None:
        root = self.log_root()
        data = cast("ContactData", env.sensors["feet_contact"].data)
        link_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        with (root / "air_time.log").open("a", encoding="utf-8") as fh:
            fh.write(f"step={step:04d}\n")
            for idx, name in enumerate(link_names):
                found = bool(data.found[0, idx])
                cur_contact = float(data.current_contact_time[0, idx])
                cur_air = float(data.current_air_time[0, idx])
                last_contact = float(data.last_contact_time[0, idx])
                last_air = float(data.last_air_time[0, idx])
                fh.write(
                    f"    {name}  contact={found}  "
                    f"current_contact={cur_contact:.3f}  current_air={cur_air:.3f}  "
                    f"last_contact={last_contact:.3f}  last_air={last_air:.3f}\n"
                )
