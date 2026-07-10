"""SO(3) in-hand reorientation goal command with a hold-and-advance state machine.

Goals are sampled uniformly on SO(3) in the wrist "tag" frame, then composed to world. The
term tracks an APPROACHING -> SUCCESS_WINDOW cycle: while the cube orientation error stays
below ``success_threshold`` for ``success_hold_steps`` consecutive control steps the env
enters a success window; after ``goal_switch_delay`` window steps a new goal is sampled.

The hand is fixed-base, so the palm (root) orientation is constant — the tag frame is
precomputed from the robot root rotation and the static tag-in-palm transform.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.entity import write_root_velocity
from genelab.managers.command_manager import CommandTerm, CommandTermCfg
from genelab.utils.math import quat_error_magnitude, quat_mul

from genelab_wuji.reorient.constants import (
    GOAL_MARKER_POS,
    REORIENT_ROBOT_ROOT_ROT,
    TAG_IN_PALM_QUAT_WXYZ,
)
from genelab_wuji.reorient.mdp._math import random_quat

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class InHandReorientCommandCfg(CommandTermCfg):
    object_name: str = "object"
    success_threshold: float = 0.2
    success_hold_steps: int = 5
    goal_switch_delay: int = 20
    tag_in_palm_quat: tuple[float, float, float, float] = TAG_IN_PALM_QUAT_WXYZ
    robot_root_rot: tuple[float, float, float, float] = REORIENT_ROBOT_ROOT_ROT
    # debug_vis (play): pose a textured marker entity at the goal orientation.
    marker_name: str = "goal_marker"
    marker_pos: tuple[float, float, float] = GOAL_MARKER_POS
    class_type: type[CommandTerm] | None = None

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = InHandReorientCommand


class InHandReorientCommand(CommandTerm):
    cfg: InHandReorientCommandCfg  # type: ignore[assignment]

    def __init__(self, cfg: InHandReorientCommandCfg, env: "EnvContext") -> None:
        super().__init__(cfg, env)
        dev = self.device
        n = self.num_envs
        # Constant tag frame in world: root_rot (palm) composed with tag-in-palm.
        root = torch.tensor(cfg.robot_root_rot, dtype=torch.float, device=dev)
        tag_in_palm = torch.tensor(cfg.tag_in_palm_quat, dtype=torch.float, device=dev)
        self._tag_quat_w = quat_mul(root.unsqueeze(0), tag_in_palm.unsqueeze(0))  # (1, 4)

        self._goal_quat_w = torch.zeros(n, 4, device=dev)
        self._goal_quat_w[:, 0] = 1.0
        self._ori_error = torch.zeros(n, device=dev)
        self._hold_counter = torch.zeros(n, dtype=torch.long, device=dev)
        self._window_timer = torch.zeros(n, dtype=torch.long, device=dev)
        self._reward_hold = torch.zeros(n, dtype=torch.long, device=dev)
        self._goal_reach_count = torch.zeros(n, dtype=torch.long, device=dev)
        # Pre-reset snapshot of the ended episode's goal count, read by the success curriculum.
        self._goal_reach_snapshot = torch.zeros(n, dtype=torch.long, device=dev)
        self._in_window = torch.zeros(n, dtype=torch.bool, device=dev)
        self._within = torch.zeros(n, dtype=torch.bool, device=dev)
        self._success_achieved = torch.zeros(n, dtype=torch.bool, device=dev)
        self._goal_switched = torch.zeros(n, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ buffers
    @property
    def command(self) -> torch.Tensor:
        return self._goal_quat_w

    @property
    def goal_quat(self) -> torch.Tensor:
        return self._goal_quat_w

    @property
    def ori_error(self) -> torch.Tensor:
        return self._ori_error

    @property
    def within_threshold(self) -> torch.Tensor:
        return self._within

    @property
    def in_success_window(self) -> torch.Tensor:
        return self._in_window

    @property
    def reward_hold_counter(self) -> torch.Tensor:
        return self._reward_hold

    @property
    def hold_counter(self) -> torch.Tensor:
        return self._hold_counter

    @property
    def window_timer(self) -> torch.Tensor:
        return self._window_timer

    @property
    def goal_reach_count(self) -> torch.Tensor:
        return self._goal_reach_count

    # ------------------------------------------------------------------ helpers
    def _cube_quat(self) -> torch.Tensor:
        handle = self._env.scene[self.cfg.object_name].gs_handle
        quat = torch.as_tensor(handle.get_quat(), device=self.device, dtype=torch.float)
        if quat.dim() == 1:
            quat = quat.unsqueeze(0).expand(self.num_envs, -1)
        return quat

    def _sample_goal(self, env_ids: torch.Tensor) -> None:
        rel = random_quat(int(env_ids.numel()), self.device)
        self._goal_quat_w[env_ids] = quat_mul(self._tag_quat_w.expand(env_ids.numel(), -1), rel)

    # ------------------------------------------------------------------ lifecycle
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Mid-episode goal switch: new goal + reset the per-goal hold state. NOTE: the
        # cumulative ``goal_reach_count`` is intentionally NOT cleared here — only on a full
        # episode reset (see ``reset``).
        self._sample_goal(env_ids)
        self._hold_counter[env_ids] = 0
        self._window_timer[env_ids] = 0
        self._reward_hold[env_ids] = 0
        self._in_window[env_ids] = False

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = env_ids
        # Snapshot the ended episode's goal count before clearing — the success curriculum
        # (which runs after this reset) reads the snapshot.
        self._goal_reach_snapshot[ids] = self._goal_reach_count[ids]
        super().reset(env_ids)  # resamples goal + clears per-goal hold state
        self._goal_reach_count[ids] = 0

    @property
    def goal_reach_snapshot(self) -> torch.Tensor:
        return self._goal_reach_snapshot

    def _update_command(self) -> None:
        cube_quat = self._cube_quat()
        err = quat_error_magnitude(cube_quat, self._goal_quat_w)
        self._ori_error = err
        within = err < self.cfg.success_threshold
        self._within = within
        self._success_achieved = torch.zeros_like(self._success_achieved)
        self._goal_switched = torch.zeros_like(self._goal_switched)

        approaching = ~self._in_window
        # APPROACHING: count consecutive in-threshold steps; reset on miss.
        self._hold_counter = torch.where(
            approaching & within, self._hold_counter + 1, self._hold_counter
        )
        self._hold_counter = torch.where(
            approaching & ~within, torch.zeros_like(self._hold_counter), self._hold_counter
        )
        reached = approaching & (self._hold_counter >= self.cfg.success_hold_steps)
        if reached.any():
            self._in_window = self._in_window | reached
            self._window_timer = torch.where(
                reached, torch.zeros_like(self._window_timer), self._window_timer
            )
            self._hold_counter = torch.where(
                reached, torch.zeros_like(self._hold_counter), self._hold_counter
            )
            self._reward_hold = torch.where(
                reached, torch.zeros_like(self._reward_hold), self._reward_hold
            )
            self._goal_reach_count = self._goal_reach_count + reached.long()
            self._success_achieved = reached

        # SUCCESS_WINDOW: advance the window timer + escalating hold bonus.
        self._window_timer = torch.where(
            self._in_window, self._window_timer + 1, self._window_timer
        )
        self._reward_hold = torch.where(
            self._in_window & within, self._reward_hold + 1, self._reward_hold
        )
        advance = self._in_window & (self._window_timer >= self.cfg.goal_switch_delay)
        adv_ids = advance.nonzero(as_tuple=False).flatten()
        if adv_ids.numel() > 0:
            self._resample_command(adv_ids)
            self._goal_switched[adv_ids] = True

        # Publish a per-env success flag for the eval harness (gymnasium-style
        # ``info["is_success"]``): an episode counts as a success if it reorients the cube to
        # at least one SO(3) goal. Set before the env's per-step auto-reset clears the count.
        extras = getattr(self._env, "_extras", None)
        if extras is not None:
            extras["is_success"] = self._goal_reach_count >= 1

        if self.cfg.debug_vis:
            self._pose_goal_marker()

    def _pose_goal_marker(self) -> None:
        """Place a textured marker entity at the goal orientation (viewer only)."""
        scene = self._env.scene
        try:
            handle = scene[self.cfg.marker_name].gs_handle  # type: ignore[index]
        except (KeyError, Exception):  # noqa: BLE001 - marker absent (e.g. training scene)
            return
        n = self.num_envs
        pos = torch.tensor(self.cfg.marker_pos, device=self.device).expand(n, -1)
        zeros = torch.zeros(n, 3, device=self.device)
        for setter, value in (
            ("set_pos", pos),
            ("set_quat", self._goal_quat_w),
        ):
            fn = getattr(handle, setter, None)
            if fn is not None:
                fn(value)
        write_root_velocity(handle, zeros, zeros)

    @property
    def success_achieved(self) -> torch.Tensor:
        return self._success_achieved

    @property
    def goal_switched(self) -> torch.Tensor:
        return self._goal_switched
