"""Contract pins for every Genesis method GeneLab calls through a ``getattr`` guard.

GeneLab's write seams duck-type against the Genesis handle
(``getattr(handle, "set_...", None)``) so unit-test fakes and multiple entity types
stay supported. The cost of that pattern is #242: a method the upstream API never had
(or renamed) silently no-ops instead of raising — ``push_by_setting_velocity`` was dead
for every rigid articulation because ``RigidEntity`` never had ``set_vel`` / ``set_ang``.

These tests pin every ``getattr``-guarded name against the *installed* Genesis, so an
upstream rename surfaces as a test failure at the next version bump instead of a silent
physics change. Import-only (no ``gs.init`` scene build), so they also run on headless CI.
"""

import pytest

pytest.importorskip("genesis")

# Every RigidEntity method reached via ``getattr(handle, name, None)`` in src/genelab
# and examples/ (grep for ``getattr(`` + ``"set_/get_/control_"`` to regenerate).
# Names in dead *fallback* arms of ``x or y`` chains are deliberately not pinned
# (``set_dofs_friction``, ``set_dofs_position_target``) — their primary arm is.
RIGID_ENTITY_SEAMS = (
    # root / DoF write seams
    "set_pos",
    "set_quat",
    "set_dofs_position",
    "set_dofs_velocity",
    "set_dofs_kp",
    "set_dofs_kv",
    "set_dofs_force_range",
    "set_dofs_armature",
    "set_dofs_frictionloss",
    "set_friction_ratio",
    "set_mass_shift",
    "control_dofs_position",
    "control_dofs_force",
    "control_dofs_velocity",
    # read seams
    "get_dofs_force",
    "get_dofs_control_force",
    "get_dofs_armature",
    "get_dofs_limit",
    "get_links_inertial_mass",
    "get_links_net_contact_force",
    "get_contacts",
    "get_vel",
    "get_ang",
)


def test_rigid_entity_seams_exist() -> None:
    from genesis.engine.entities import RigidEntity

    missing = [name for name in RIGID_ENTITY_SEAMS if not hasattr(RigidEntity, name)]
    assert not missing, (
        f"Genesis RigidEntity no longer has {missing} — the GeneLab call sites guarding "
        f"these with getattr(..., None) would silently no-op (see #242). Update the seam "
        f"or this pin."
    )


def test_free_joint_surface_for_root_velocity_writes() -> None:
    """``write_root_velocity`` needs joints with n_dofs / dofs_idx_local for free-joint discovery."""
    from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint

    for attr in ("n_dofs", "dofs_idx_local", "dof_start"):
        assert hasattr(RigidJoint, attr), f"RigidJoint lost {attr}"


def test_viewer_hud_seam() -> None:
    """The teleop HUD writes through the inner pyrender viewer's set_message_text."""
    viewer_mod = pytest.importorskip("genesis.ext.pyrender.viewer")
    assert hasattr(viewer_mod.Viewer, "set_message_text")
    assert hasattr(viewer_mod.Viewer, "register_keybinds")


def test_rigid_solver_gravity_seam() -> None:
    """``mdp.dr.gravity`` writes per-env gravity through ``sim.rigid_solver.set_gravity``."""
    import genesis as gs

    try:
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.cpu, logging_level="error")
        from genesis.engine.solvers import RigidSolver
    except Exception as exc:  # noqa: BLE001 - headless CI may fail at init, not import
        pytest.skip(f"Genesis solver import needs a runtime: {exc}")
    assert hasattr(RigidSolver, "set_gravity")
