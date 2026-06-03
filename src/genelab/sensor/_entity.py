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
    from genelab.contracts import EnvContext


def entity_articulation(env: "EnvContext", name: str) -> "Articulation":
    """The named :class:`Articulation`, else the primary ``env.articulation``.

    Falls back to ``env`` itself when there's no articulation accessor — minimal fake-env
    tests expose ``link_names`` / ``joint_names`` at env level (duck-typed, hence the cast).
    """
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name]
    return cast("Articulation", getattr(env, "articulation", env))


def entity_state(env: "EnvContext", name: str) -> "RobotState":
    """The named entity's ``RobotState``, else the primary ``env.robot_state``."""
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name].data
    # Fallback only reached by minimal fake envs that expose ``robot_state`` directly; the
    # real env always has ``articulations`` (so it never lands here).
    return env.robot_state  # pyright: ignore[reportAttributeAccessIssue]


def entity_handle(env: "EnvContext", name: str) -> Any:
    """The named entity's raw Genesis handle, else the primary ``env.robot``."""
    arts = getattr(env, "articulations", None)
    if arts is not None and name in arts:
        return arts[name].gs_handle
    return env.robot  # pyright: ignore[reportAttributeAccessIssue]


def prebuild_link_names(entity: Any) -> list[str]:
    """Link-name list resolvable inside ``Sensor.pre_build_genesis`` (before ``gs_scene.build``).

    The GeneLab :class:`Articulation` wrapper only fills its ``link_names`` at post-build
    ``bind`` time, but the SDF tactile sensors must map a link name to its local index *before*
    build to register their Genesis options. Genesis already populates ``gs_handle.links`` at
    ``add_entity`` time, so derive the names from there when the wrapper's own list is still
    empty. An already-populated ``entity.link_names`` wins — the fake entities in the sensor
    unit tests set it directly and carry no ``gs_handle.links``.
    """
    names = list(getattr(entity, "link_names", None) or [])
    if names:
        return names
    links = getattr(getattr(entity, "gs_handle", None), "links", None) or []
    return [getattr(link, "name", f"link_{i}") for i, link in enumerate(links)]
