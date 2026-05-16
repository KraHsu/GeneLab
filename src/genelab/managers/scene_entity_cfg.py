"""Scene-entity configuration: a uniform name → index selector for manager terms.

``SceneEntityCfg`` is the single shared way to point a reward / event / observation
at a subset of the scene entity's joints or links. Configs declare names; the
manager-construction pipeline (``managers._base.instantiate_class_term``) calls
:meth:`SceneEntityCfg.resolve` once at startup to convert those names into
integer index tuples. Terms then index ``robot_state`` tensors directly.

mjlab parity for ``SceneEntityCfg``, slimmed to GeneLab's surface:

* ``link_names`` / ``link_ids`` (mjlab: ``body_names`` — same concept, just the
  Genesis-side name).
* ``joint_names`` / ``joint_ids``.
* No ``geom_names`` / ``site_names`` / ``actuator_names`` etc. — Genesis doesn't
  surface those as separate selectable collections in GeneLab today. Add fields
  here when a consumer materialises.

Resolution is idempotent: once ``link_ids`` is set, a second ``resolve`` call is
a no-op. Each manager deep-copies its cfg at construction, so the same
``SceneEntityCfg`` instance never gets resolved against two different envs.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class SceneEntityCfg:
    """Declarative selector for joints / links inside the scene's main articulation.

    Pass one as the ``asset_cfg`` parameter of a manager term::

        RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "asset_cfg": SceneEntityCfg(
                    "robot", link_names=("left_ankle_roll_link", "right_ankle_roll_link")
                ),
                "command_name": "twist",
            },
        )

    After the manager finishes construction the ``link_ids`` / ``joint_ids``
    tuples are populated and the term function reads them directly — no string
    matching on the hot path.
    """

    name: str = "robot"
    """Scene-entity name. GeneLab currently exposes a single articulation ("robot");
    the field is retained for mjlab parity and to leave room for multi-entity scenes."""

    link_names: tuple[str, ...] | None = None
    """Names of the links this term acts on. ``None`` means "no link selection"."""

    link_ids: tuple[int, ...] | None = None
    """Populated by :meth:`resolve` from ``link_names``. Don't set manually."""

    joint_names: tuple[str, ...] | None = None
    """Names of the joints this term acts on. ``None`` means "no joint selection"."""

    joint_ids: tuple[int, ...] | None = None
    """Populated by :meth:`resolve` from ``joint_names``. Don't set manually."""

    def resolve(self, env: "ManagerBasedRlEnv") -> None:
        """Convert configured ``*_names`` into index tuples against ``env``'s tables.

        Idempotent: re-running on an already-resolved cfg is a no-op. Raises
        :class:`ValueError` if any configured name isn't in the env's enumeration
        — fail fast at manager construction rather than at first reward call.
        """
        if self.link_names is not None and self.link_ids is None:
            missing = [n for n in self.link_names if n not in env.link_names]
            if missing:
                raise ValueError(
                    f"SceneEntityCfg(name={self.name!r}): link(s) {missing!r} not in "
                    f"env.link_names={env.link_names!r}"
                )
            self.link_ids = tuple(env.link_names.index(n) for n in self.link_names)
        if self.joint_names is not None and self.joint_ids is None:
            missing = [n for n in self.joint_names if n not in env.joint_names]
            if missing:
                raise ValueError(
                    f"SceneEntityCfg(name={self.name!r}): joint(s) {missing!r} not in "
                    f"env.joint_names={env.joint_names!r}"
                )
            self.joint_ids = tuple(env.joint_names.index(n) for n in self.joint_names)
