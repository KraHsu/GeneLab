"""Contact-showcase runner: G1 marches in place; foot contact drawn live, no disk dumps.

A virtual spring suspends the pelvis (so the passive G1 stays upright) while this runner
scripts an alternating in-place step by flexing each leg's hip + knee + ankle in turn. Each
foot's time-since-lift-off streams to the PyQt plot on the env cfg; this runner draws a
contact-state sphere under each foot (green = planted, red = airborne; radius ~ contact force).
"""

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from genelab_showcase.contact.env_cfg import FOOT_LINKS
from genelab_showcase.runner import ShowcaseRunner, SpringCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from genelab.sensor import ContactData

# The leg joints flexed to lift a foot: (hip pitch, knee, ankle pitch), per side.
_LEFT_LEG: tuple[str, str, str] = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
)
_RIGHT_LEG: tuple[str, str, str] = (
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)
# Per-actuator-group action_scale turns these into a high-knee lift (~70 deg hip, ~60 deg knee
# above default): big enough to clear the ground cleanly so the contact <-> air switch is
# obvious. Hip is negative (pitch flexion).
_HIP_AMP, _KNEE_AMP, _ANKLE_AMP = -3.0, 4.0, 2.0
_STEP_PERIOD = 120  # control steps per full left+right cycle (~2.4 s at 50 Hz)

# Contact-sphere drawing.
_MARKER_Z_OFFSET = 0.03  # m below the ankle-roll link origin, ≈ the sole
_FORCE_RADIUS_SCALE = 0.0005  # m per N — grows the sphere with contact force
_FORCE_RADIUS_CAP = 400.0  # N — clamps the sphere size on hard landings
_MARKER_BASE_RADIUS = 0.03  # m


class ContactShowcaseRunner(ShowcaseRunner):
    """G1 marches in place under a pelvis spring; draws per-foot contact spheres live."""

    task_slug = "contact"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        # Partial gravity comp so the feet stay loaded (contact registers); a stiff vertical
        # PD holds the pelvis up against the planted-leg squat so the body marches tall.
        super().__init__(
            env_cfg,
            spring=SpringCfg(
                link_name="pelvis", gravity_comp=0.9, stiffness=8000.0, damping=1000.0
            ),
        )
        self._joint_idx: dict[str, int] | None = None
        self._foot_idx: tuple[int, int] = (-1, -1)
        self._nodes: list[Any] = []

    def _ensure_indices(self, env: "ManagerBasedRlEnv") -> None:
        if self._joint_idx is not None:
            return
        robot = env.articulations["robot"]
        self._joint_idx = {n: robot.joint_names.index(n) for n in (*_LEFT_LEG, *_RIGHT_LEG)}
        self._foot_idx = (
            robot.link_names.index(FOOT_LINKS[0]),
            robot.link_names.index(FOOT_LINKS[1]),
        )

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        self._ensure_indices(env)
        assert self._joint_idx is not None
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        phase = 2.0 * math.pi * step / _STEP_PERIOD
        # Lift left on the positive half of the cycle, right on the negative half.
        sides = ((max(0.0, math.sin(phase)), _LEFT_LEG), (max(0.0, -math.sin(phase)), _RIGHT_LEG))
        for lift, leg in sides:
            hip, knee, ankle = (self._joint_idx[j] for j in leg)
            action[:, hip] = _HIP_AMP * lift
            action[:, knee] = _KNEE_AMP * lift
            action[:, ankle] = _ANKLE_AMP * lift
        return action

    def _post_step(self, env: "ManagerBasedRlEnv", step: int) -> None:
        self._ensure_indices(env)
        gs_scene = env.scene.gs_scene
        for node in self._nodes:
            gs_scene.clear_debug_object(node)
        self._nodes = []
        robot = env.articulations["robot"]
        foot_pos = robot.data.link_pos[0]  # (num_links, 3), normalized to env 0
        data = cast("ContactData", env.sensors["feet_contact"].data)
        found = data.found[0]
        force_norm = data.force_norm[0]
        for i, link_idx in enumerate(self._foot_idx):
            pos = foot_pos[link_idx].detach().cpu().numpy().copy()
            pos[2] -= _MARKER_Z_OFFSET  # drop the marker toward the sole
            planted = bool(found[i])
            color = (0.2, 1.0, 0.35, 0.9) if planted else (1.0, 0.3, 0.2, 0.9)
            radius = _MARKER_BASE_RADIUS + _FORCE_RADIUS_SCALE * min(
                float(force_norm[i]), _FORCE_RADIUS_CAP
            )
            self._nodes.append(gs_scene.draw_debug_sphere(pos, radius=radius, color=color))
