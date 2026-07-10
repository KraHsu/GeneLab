"""Root-velocity write-back for Genesis entity handles.

Genesis's ``RigidEntity`` API is asymmetric: ``get_vel`` / ``get_ang`` exist, but the
matching setters ``set_vel`` / ``set_ang`` exist only on FEM / tool entities. Writing a
floating base's velocity therefore goes through ``set_dofs_velocity`` on the base free
joint's 6 DoFs — indices 0–2 carry world-frame linear velocity, 3–5 world-frame angular
velocity. Every GeneLab site that overwrites root velocity (event terms, the articulation
writer, example resets) must route through :func:`write_root_velocity`; calling
``getattr(handle, "set_vel", ...)`` directly silently no-ops on rigid entities (#242).
"""

from typing import Any

import torch


def base_dof_indices(handle: Any) -> list[int] | None:
    """The 6 free-joint DoF indices of ``handle``'s floating base, or ``None`` if fixed-based.

    Scans ``handle.joints`` for the first joint with ≥ 6 DoFs (same idiom as the
    articulation binder's free-joint detection) and returns its first six
    ``dofs_idx_local`` entries, falling back to ``dof_start`` arithmetic for handles
    that don't expose the index list.
    """
    joints = getattr(handle, "joints", None) or []
    for joint in joints:
        if int(getattr(joint, "n_dofs", 1)) < 6:
            continue
        idx = getattr(joint, "dofs_idx_local", None)
        if idx is not None:
            idx = [int(i) for i in idx]
            if len(idx) >= 6:
                return idx[:6]
        start = int(getattr(joint, "dof_start", 0))
        return list(range(start, start + 6))
    return None


def write_root_velocity(
    handle: Any,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
    env_ids: torch.Tensor | None = None,
) -> bool:
    """Overwrite ``handle``'s world-frame root velocity; returns ``True`` if written.

    ``lin_vel_w`` / ``ang_vel_w`` are ``(n, 3)`` aligned with ``env_ids`` (``None`` →
    all envs). Rigid entities take the free-joint ``set_dofs_velocity`` path; entities
    exposing direct ``set_vel`` / ``set_ang`` setters (FEM / tool entities, test fakes)
    take those. ``False`` means the handle has neither — e.g. a fixed-base articulation,
    which has no root velocity to write.
    """
    set_dofs_velocity = getattr(handle, "set_dofs_velocity", None)
    base_idx = base_dof_indices(handle)
    if set_dofs_velocity is not None and base_idx is not None:
        velocity = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
        set_dofs_velocity(velocity, base_idx, envs_idx=env_ids)
        return True
    wrote = False
    for name, value in (("set_vel", lin_vel_w), ("set_ang", ang_vel_w)):
        fn = getattr(handle, name, None)
        if fn is None:
            continue
        try:
            fn(value, envs_idx=env_ids)
        except TypeError:
            fn(value)
        wrote = True
    return wrote
