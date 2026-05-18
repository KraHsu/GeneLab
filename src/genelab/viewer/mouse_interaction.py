"""Subclass of ``MouseInteractionPlugin`` patched for Genesis' batched tensor outputs.

The upstream plugin (``genesis.vis.viewer_plugins.MouseInteractionPlugin``) assumes
``link.get_pos()`` / ``get_quat()`` / ``get_vel()`` / ``get_ang()`` return un-batched
``(3,)`` / ``(4,)`` arrays, while the link's ``inertial_pos`` / ``inertial_quat`` are
already un-batched. With the current Genesis (post-build), the dynamic getters return
``(num_envs, ...)`` tensors — even at ``num_envs=1`` — so the inertial / link arrays no
longer share shapes and ``_np_quat_mul`` asserts inside ``_apply_spring_force``.

This subclass squeezes the leading batch dim from the dynamic per-link arrays so the
rest of the spring-force math (which operates on a single env's COM frame) works.

Genesis 0.4.7 added an ``envs_idx`` kwarg to ``get_pos`` / ``get_quat`` / ``get_vel`` /
``get_ang``; the ``on_draw`` wrapper forwards ``*args, **kwargs`` so calls like
``get_pos(envs_idx=...)`` from the upstream ``on_draw`` reach the underlying method.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np

import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.viewer_plugins import MouseInteractionPlugin


def _squeeze_env(arr: np.ndarray) -> np.ndarray:
    """Drop a leading length-1 batch dim if present, leaving ``(3,)`` / ``(4,)``."""
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr[0]
    return arr


class GeneLabMouseInteractionPlugin(MouseInteractionPlugin):
    """Mouse-interaction plugin compatible with Genesis' batched link APIs."""

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> Any:
        # The base class computes ``_held_point_local`` via
        # ``inv_transform_by_trans_quat(ray_hit.position, link_pos, link_quat)``;
        # with batched link state ``(1, 3)`` / ``(1, 4)`` this broadcasts into a
        # ``(1, 3)`` held point, which then breaks every later use. Squeeze after.
        result = super().on_mouse_press(x, y, button, modifiers)
        if self._held_point_local is not None:
            self._held_point_local = _squeeze_env(np.asarray(self._held_point_local))
        return result

    def _apply_spring_force(self, control_point: np.ndarray, dt: float) -> None:
        if not self._held_link:
            return

        link_pos = _squeeze_env(tensor_to_array(self._held_link.get_pos()))
        link_quat = _squeeze_env(tensor_to_array(self._held_link.get_quat()))
        lin_vel = _squeeze_env(tensor_to_array(self._held_link.get_vel()))
        ang_vel = _squeeze_env(tensor_to_array(self._held_link.get_ang()))

        held_point_world = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)

        inertial_pos = tensor_to_array(cast(Any, self._held_link.inertial_pos))
        inertial_quat = tensor_to_array(cast(Any, self._held_link.inertial_quat))
        world_principal_quat = gu.transform_quat_by_quat(inertial_quat, link_quat)

        arm_in_principal = gu.inv_transform_by_trans_quat(
            self._held_point_local, inertial_pos, inertial_quat
        )
        arm_in_world = gu.transform_by_quat(arm_in_principal, world_principal_quat)

        R_world = gu.quat_to_R(world_principal_quat)
        inertia_world = R_world @ self._held_link.inertial_i @ R_world.T
        inv_inertia_world = np.linalg.inv(inertia_world)

        pos_err_v = control_point - held_point_world
        mass = self._held_link.get_mass()
        if mass is None or mass <= 0:
            return
        inv_mass = float(1.0 / mass)

        total_impulse = np.zeros(3, dtype=control_point.dtype)
        total_torque_impulse = np.zeros(3, dtype=control_point.dtype)

        for i in range(3):
            body_point_vel = lin_vel + np.cross(ang_vel, arm_in_world)
            vel_err_v = -body_point_vel

            direction = np.zeros(3, dtype=control_point.dtype)
            direction[i] = 1.0

            pos_err = float(np.dot(direction, pos_err_v))
            vel_err = float(np.dot(direction, vel_err_v))

            arm_x_dir = np.cross(arm_in_world, direction)
            rot_mass = float(np.dot(arm_x_dir, inv_inertia_world @ arm_x_dir))
            virtual_mass = 1.0 / (inv_mass + rot_mass + 1e-12)

            damping_coeff = 2.0 * np.sqrt(self.spring_const * virtual_mass)
            impulse = (self.spring_const * pos_err + damping_coeff * vel_err) * dt

            lin_vel = lin_vel + direction * impulse * inv_mass
            ang_vel = ang_vel + inv_inertia_world @ (arm_x_dir * impulse)

            total_impulse[i] += impulse
            total_torque_impulse += arm_x_dir * impulse

        self._held_link.solver.apply_links_external_force(
            total_impulse / dt, (self._held_link.idx,), ref="link_com", local=False
        )
        self._held_link.solver.apply_links_external_torque(
            total_torque_impulse / dt, (self._held_link.idx,), ref="link_com", local=False
        )

    def on_draw(self) -> None:
        # The base class's on_draw also dereferences batched ``link.get_pos`` / ``get_quat``
        # outputs through ``gu.transform_by_trans_quat`` to position the debug sphere/line.
        # Pre-squeeze by temporarily wrapping the link's accessors.
        link = self._held_link
        if link is None:
            super().on_draw()
            return

        orig_get_pos = link.get_pos
        orig_get_quat = link.get_quat

        def _wrap(fn: Callable[..., Any]) -> Callable[..., np.ndarray]:
            def _wrapped(*args: Any, **kwargs: Any) -> np.ndarray:
                out = fn(*args, **kwargs)
                arr = tensor_to_array(out)
                return _squeeze_env(arr)

            return _wrapped

        try:
            link.get_pos = _wrap(orig_get_pos)  # type: ignore[method-assign]
            link.get_quat = _wrap(orig_get_quat)  # type: ignore[method-assign]
            super().on_draw()
        finally:
            link.get_pos = orig_get_pos  # type: ignore[method-assign]
            link.get_quat = orig_get_quat  # type: ignore[method-assign]
