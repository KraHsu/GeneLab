"""Name-keyed entity resolution for sensors (ROADMAP M3.6 / ADR-0012 S4).

Sensors attach to / read from one scene entity named by ``SensorCfg.entity_name``. These
helpers look it up in ``env.articulations[name]`` and fall back to the singular primary
(``env.robot`` / ``env.robot_state`` / ``env.articulation``) so single-robot scenes and
fake-env tests (which expose those directly) are unchanged.

These mirror ``genelab.mdp._helpers.resolve_*`` but live in the sensor layer: ``mdp._helpers``
imports ``sensor.contact``, so a sensor importing it would form a cycle.
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from genelab.entity import Articulation, RobotState
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def entity_articulation(env: "ManagerBasedRlEnv", name: str) -> "Articulation":
    """The named :class:`Articulation`, else the primary ``env.articulation``.

    Falls back to ``env`` itself when there's no articulation accessor — minimal fake-env
    tests expose ``link_names`` / ``joint_names`` at env level (duck-typed, hence the cast).
    """
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name]
    return cast("Articulation", getattr(env, "articulation", env))


def entity_state(env: "ManagerBasedRlEnv", name: str) -> "RobotState":
    """The named entity's ``RobotState``, else the primary ``env.robot_state``."""
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name].data
    # Fallback only reached by minimal fake envs that expose ``robot_state`` directly; the
    # real env always has ``articulations`` (so it never lands here).
    return env.robot_state  # pyright: ignore[reportAttributeAccessIssue]


def entity_handle(env: "ManagerBasedRlEnv", name: str) -> Any:
    """The named entity's raw Genesis handle, else the primary ``env.robot``."""
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name].gs_handle
    return env.robot  # pyright: ignore[reportAttributeAccessIssue]
