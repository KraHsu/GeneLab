"""Conformance + drift guard for the ``genelab.contracts`` ports.

Two layers protect the ``EnvContext`` / ``SceneContext`` protocols:

* **pyright (CI typecheck)** — the real, signature-level conformance check, via the
  ``TYPE_CHECKING`` cast assignments in ``manager_based_rl_env.py`` /
  ``interactive_scene.py`` (the adapters declaring they implement the ports).
* **this test (headless)** — pins each port's member set against drift and checks,
  by class introspection (no instantiation, so no Genesis runtime), that the
  concrete classes expose the protocol's *class-level* members.

``ManagerBasedRlEnv``'s managers / ``cfg`` / ``common_step_counter`` are instance
attributes set in ``__init__`` (not visible on the class object), so this test
cannot see them — pyright covers those. ``SceneContext``'s members are all
``InteractiveScene`` properties, so it is fully checkable here.
"""

from genelab.contracts import EnvContext, SceneContext

# Frozen port surfaces — update in the same PR that intentionally widens a protocol.
EXPECTED_ENV_CONTEXT = frozenset(
    {
        "device",
        "num_envs",
        "dt",
        "physics_dt",
        "max_episode_length",
        "max_episode_length_s",
        "num_actions",
        "episode_length_buf",
        "env_origins",
        "viewer_closed",
        "common_step_counter",
        "scene",
        "articulations",
        "sensors",
        "deformable_terrain",
        "cfg",
        "action_manager",
        "command_manager",
        "observation_manager",
        "reward_manager",
        "termination_manager",
        "event_manager",
        "curriculum_manager",
        "metrics_manager",
        "write_joint_state_to_sim",
        "write_root_state_to_sim",
    }
)

EXPECTED_SCENE_CONTEXT = frozenset(
    {
        "num_envs",
        "device",
        "env_origins",
        "gs_scene",
        "terrain",
        "viewer_closed",
        "sensors",
        "recorder_bridge",
        "articulations",
        "rigid_objects",
    }
)

# EnvContext members that ManagerBasedRlEnv exposes as instance attributes (set in
# __init__), hence invisible to dir(class). pyright validates these via the cast
# assignment in manager_based_rl_env.py; this test checks only the rest.
_ENV_INSTANCE_ATTRS = frozenset(
    {
        "cfg",
        "common_step_counter",
        "action_manager",
        "command_manager",
        "observation_manager",
        "reward_manager",
        "termination_manager",
        "event_manager",
        "curriculum_manager",
        "metrics_manager",
    }
)


def test_protocol_surfaces_match_snapshot() -> None:
    """Pin the port surfaces so a protocol change is explicit and reviewed."""
    assert set(EnvContext.__protocol_attrs__) == EXPECTED_ENV_CONTEXT
    assert set(SceneContext.__protocol_attrs__) == EXPECTED_SCENE_CONTEXT


def test_scene_context_satisfied_by_interactive_scene() -> None:
    """Every SceneContext member is an InteractiveScene property (fully checkable)."""
    from genelab.scene.interactive_scene import InteractiveScene

    missing = set(SceneContext.__protocol_attrs__) - set(dir(InteractiveScene))
    assert not missing, f"InteractiveScene missing SceneContext members: {sorted(missing)}"


def test_env_context_class_level_members_present() -> None:
    """The class-level (property/method) EnvContext members exist on ManagerBasedRlEnv.

    Instance attributes (``_ENV_INSTANCE_ATTRS``) are excluded — pyright checks those.
    """
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    class_level = set(EnvContext.__protocol_attrs__) - _ENV_INSTANCE_ATTRS
    missing = class_level - set(dir(ManagerBasedRlEnv))
    assert not missing, f"ManagerBasedRlEnv missing EnvContext members: {sorted(missing)}"
