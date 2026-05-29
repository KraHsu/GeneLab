"""Reusable reward term functions for locomotion tasks.

Split into per-family modules (ADR-0015): :mod:`~genelab.mdp.rewards.tracking`,
:mod:`~genelab.mdp.rewards.regularization`, and :mod:`~genelab.mdp.rewards.gait`.
This package re-exports every public term so ``from genelab.mdp.rewards import
<term>`` and ``genelab.mdp.rewards.<term>`` resolve exactly as before the split.
"""

from genelab.mdp.rewards.gait import (
    angular_momentum_penalty,
    body_angular_velocity_penalty,
    feet_air_time,
    feet_clearance,
    feet_slip,
    feet_swing_height,
    self_collision_cost,
    soft_landing,
)
from genelab.mdp.rewards.regularization import (
    action_rate_l2,
    alive_bonus,
    applied_torque_l2,
    base_height_l2,
    flat_orientation_l2,
    joint_acc_l2,
    joint_pos_limits,
    joint_vel_limits,
    lin_vel_z_l2,
    upright_exp,
    variable_posture,
)
from genelab.mdp.rewards.energy import (
    energy_budget,
    kinetic_energy_l2,
    potential_energy,
)
from genelab.mdp.rewards.tactile import (
    contact_count,
    contact_intensity_l2,
    slip_penalty,
)
from genelab.mdp.rewards.tracking import (
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
    track_angular_velocity_z_exp,
    track_linear_velocity_xy_exp,
)

__all__ = [
    "action_rate_l2",
    "alive_bonus",
    "angular_momentum_penalty",
    "applied_torque_l2",
    "base_height_l2",
    "body_angular_velocity_penalty",
    "contact_count",
    "contact_intensity_l2",
    "energy_budget",
    "feet_air_time",
    "feet_clearance",
    "feet_slip",
    "feet_swing_height",
    "flat_orientation_l2",
    "joint_acc_l2",
    "joint_pos_limits",
    "joint_vel_limits",
    "kinetic_energy_l2",
    "lin_vel_z_l2",
    "motion_global_anchor_orientation_error_exp",
    "motion_global_anchor_position_error_exp",
    "motion_global_body_angular_velocity_error_exp",
    "motion_global_body_linear_velocity_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "potential_energy",
    "self_collision_cost",
    "slip_penalty",
    "soft_landing",
    "track_angular_velocity_z_exp",
    "track_linear_velocity_xy_exp",
    "upright_exp",
    "variable_posture",
]
