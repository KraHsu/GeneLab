"""Tensor-vectorized hand-crafted controller for the HER pick-and-place env.

Produces a (num_envs, 4) action per env step by running each env through a
shared finite state machine on the live env state. Reasonable enough to be
used as the action source for offline demonstration collection — running this
in parallel through the Cartesian + sparse-reward env produces full
``reach -> grasp -> lift -> place`` trajectories whose transitions
:func:`genelab_franka_pick_and_place.collect_demos.main` saves to disk.

Each env carries its own ``phase`` (which of the 6 stages) and ``frames``
(steps spent in the current stage) so envs progress independently. Call
:meth:`FrankaPickAndPlaceFsm.reset` (with optional ``env_ids``) on env reset
to clear that state for the auto-reset envs only — the controller does NOT
read the env's done flag itself.
"""

from typing import TYPE_CHECKING

import torch

from genelab_franka_pick_and_place.constants import (
    CUBE_ENTITY_NAME,
    CUBE_HALF,
    DISTANCE_THRESHOLD,
    EE_LINK,
)

# Must match ``genelab_franka_pick_and_place.mdp._GOAL_ATTR``: the env attribute
# under which ``resample_goal_uniform`` caches the per-env desired-goal buffer.
# Duplicated here to keep the FSM independent of mdp's private surface.
_GOAL_ATTR = "_franka_pp_goal_w"

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


PHASE_APPROACH = 0  # xy align above the cube, gripper open, z elevated
PHASE_DESCEND = 1  # lower z to grasp height while xy tracks the cube
PHASE_CLOSE = 2  # hold position, drive gripper -1 for a few steps
PHASE_LIFT = 3  # +z while keeping gripper closed
PHASE_TO_GOAL = 4  # move EE (+held cube) toward desired_goal in 3D
PHASE_HOLD = 5  # gentle xy/z tracking of the goal, gripper closed


class FrankaPickAndPlaceFsm:
    """Vectorized FSM producing demo actions for ``num_envs`` parallel envs.

    Constants are tuned for the HER variant's physics overrides (panda-gym
    neutral pose, stiff arm PD, fast gripper, friction=1.0). The controller
    relies on the env exposing the same internals used by the MDP terms — cube
    rigid handle under ``scene.rigid_objects[CUBE_ENTITY_NAME]``, hand link
    index resolved against ``articulation.link_names``, and the per-env goal
    buffer cached on the env at ``mdp._GOAL_ATTR``.
    """

    # Target hand-link z values, picked from the FSM probe that succeeded.
    APPROACH_Z_OFFSET = 0.08  # above cube_top
    GRASP_Z = 0.112  # hand_z so fingertips wrap the cube on the table
    LIFT_TARGET_Z = 0.30
    GOAL_Z_OFFSET = 0.10  # hand_z above the goal so fingertips hold the cube there

    XY_TOL = 0.005
    Z_TOL = 0.008
    CLOSE_STEPS = 4
    SUCCESS_DIST = DISTANCE_THRESHOLD - 0.005  # finish slightly inside the threshold

    def __init__(self, env: "ManagerBasedRlEnv") -> None:
        self._env = env
        self._n = env.num_envs
        self._device = env.device
        self._cube_handle = env.scene.rigid_objects[CUBE_ENTITY_NAME].gs_handle
        self._art = env.articulation
        self._hand_idx = self._art.link_names.index(EE_LINK)
        finger_joint_idx = [i for i, n in enumerate(self._art.joint_names) if "finger_joint" in n]
        self._finger_joint_idx = torch.tensor(
            finger_joint_idx, dtype=torch.long, device=self._device
        )
        self.phase = torch.zeros(self._n, dtype=torch.long, device=self._device)
        self.frames = torch.zeros(self._n, dtype=torch.long, device=self._device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear phase/frame counters for the given envs (all envs if ``None``)."""
        if env_ids is None:
            self.phase.fill_(0)
            self.frames.fill_(0)
        else:
            self.phase[env_ids] = 0
            self.frames[env_ids] = 0

    # -- state accessors -----------------------------------------------------

    def _cube_pos(self) -> torch.Tensor:
        raw = self._cube_handle.get_pos()
        pos = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(self._n, -1)
        return pos.to(self._device)

    def _ee_pos(self) -> torch.Tensor:
        return self._env.robot_state.link_pos[:, self._hand_idx, :]

    def _goal_pos(self) -> torch.Tensor:
        return getattr(self._env, _GOAL_ATTR)

    def _gripper_width(self) -> torch.Tensor:
        return self._env.robot_state.joint_pos.index_select(1, self._finger_joint_idx).mean(dim=1)

    # -- main step -----------------------------------------------------------

    def step(self) -> torch.Tensor:
        """Compute the (B, 4) action for the current env state and advance phases."""
        cube = self._cube_pos()
        ee = self._ee_pos()
        goal = self._goal_pos()

        action = torch.zeros(self._n, 4, device=self._device)
        action[:, 3] = 1.0  # gripper open by default

        approach_z = cube[:, 2] + CUBE_HALF + self.APPROACH_Z_OFFSET
        cube_xy_err = torch.linalg.vector_norm(cube[:, :2] - ee[:, :2], dim=-1)

        # APPROACH: xy chase the cube, z hover above it
        mask = self.phase == PHASE_APPROACH
        if mask.any():
            delta = torch.stack(
                [
                    cube[:, 0] - ee[:, 0],
                    cube[:, 1] - ee[:, 1],
                    approach_z - ee[:, 2],
                ],
                dim=-1,
            )
            action[mask, :3] = (delta[mask] / 0.05).clamp(-1.0, 1.0)
            z_err = (approach_z - ee[:, 2]).abs()
            ready = (cube_xy_err < self.XY_TOL * 2) & (z_err < self.Z_TOL * 2)
            self.phase[mask & ready] = PHASE_DESCEND

        # DESCEND: keep xy locked on the cube while dropping to grasp height
        mask = self.phase == PHASE_DESCEND
        if mask.any():
            delta = torch.stack(
                [
                    cube[:, 0] - ee[:, 0],
                    cube[:, 1] - ee[:, 1],
                    self.GRASP_Z - ee[:, 2],
                ],
                dim=-1,
            )
            action[mask, :3] = (delta[mask] / 0.05).clamp(-1.0, 1.0)
            z_err = (self.GRASP_Z - ee[:, 2]).abs()
            ready = (z_err < self.Z_TOL) & (cube_xy_err < self.XY_TOL * 2)
            advance = mask & ready
            self.phase[advance] = PHASE_CLOSE
            self.frames[advance] = 0

        # CLOSE: stay put, drive gripper down for a few steps
        mask = self.phase == PHASE_CLOSE
        if mask.any():
            action[mask, :3] = 0.0
            action[mask, 3] = -1.0
            self.frames[mask] += 1
            self.phase[mask & (self.frames >= self.CLOSE_STEPS)] = PHASE_LIFT

        # LIFT: hold xy, command +z aggressively, gripper closed
        mask = self.phase == PHASE_LIFT
        if mask.any():
            dz = self.LIFT_TARGET_Z - ee[:, 2]
            action[mask, 0] = 0.0
            action[mask, 1] = 0.0
            action[mask, 2] = (dz[mask] / 0.05).clamp(-1.0, 1.0)
            action[mask, 3] = -1.0
            # Advance once we've cleared the table top by a good margin.
            advance = mask & (ee[:, 2] > 0.22)
            self.phase[advance] = PHASE_TO_GOAL

        # TO_GOAL: chase the desired goal position in 3D (hand at goal + offset)
        mask = self.phase == PHASE_TO_GOAL
        if mask.any():
            target = torch.stack(
                [
                    goal[:, 0] - ee[:, 0],
                    goal[:, 1] - ee[:, 1],
                    (goal[:, 2] + self.GOAL_Z_OFFSET) - ee[:, 2],
                ],
                dim=-1,
            )
            action[mask, :3] = (target[mask] / 0.05).clamp(-1.0, 1.0)
            action[mask, 3] = -1.0
            cube_to_goal = torch.linalg.vector_norm(cube - goal, dim=-1)
            self.phase[mask & (cube_to_goal < self.SUCCESS_DIST)] = PHASE_HOLD

        # HOLD: gentle tracking of the goal, gripper still closed
        mask = self.phase == PHASE_HOLD
        if mask.any():
            target = torch.stack(
                [
                    goal[:, 0] - ee[:, 0],
                    goal[:, 1] - ee[:, 1],
                    (goal[:, 2] + self.GOAL_Z_OFFSET) - ee[:, 2],
                ],
                dim=-1,
            )
            action[mask, :3] = (target[mask] / 0.05).clamp(-1.0, 1.0) * 0.5
            action[mask, 3] = -1.0

        return action.clamp(-1.0, 1.0)
