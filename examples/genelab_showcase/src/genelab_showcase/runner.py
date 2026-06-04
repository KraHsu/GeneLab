"""Shared play-only runner used by every showcase task.

The runner owns the env lifecycle, drives a fixed-length scripted-action loop, and
calls the subclass-supplied :py:meth:`_dump` hook every ``log_interval`` steps so each
showcase can render its own visual / textual evidence (PNG dump, histogram print,
joint-tracking error, ...).

Subclasses override :py:meth:`_scripted_action` to emit the per-step action tensor and
:py:meth:`_dump` to extract sensor / curriculum / robot state. The base class handles
device placement, log-directory creation, and viewer ticking — none of that should
duplicate per showcase.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from genelab.cache import ensure_project_cache
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from genelab.utils.math import quat_apply

if TYPE_CHECKING:
    pass

# World up axis, reused by the virtual-spring uprighting torque.
_WORLD_UP: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class SpringCfg:
    """A virtual "marionette" spring that suspends a robot link so it cannot fully fall.

    Applies, every control step, an external force + torque on ``link_name`` via Genesis's
    ``rigid_solver.apply_links_external_force`` / ``apply_links_external_torque`` (the same
    primitive the viewer's mouse-drag uses):

    * a **vertical** spring ``F_z = stiffness * (target_z - z) - damping * v_z`` that holds
      the link near ``target_z`` (captured from the link's height at the first step when
      ``target_z is None``);
    * optional **horizontal centering** that pulls the link back toward its anchor x/y so
      the robot does not drift off-camera;
    * optional **uprighting torque** that rotates the base's up-axis back toward world up,
      so the humanoid stays standing instead of toppling.

    Gains are intentionally exposed so each showcase can tune the "hung by a rope" feel.
    """

    link_name: str = "pelvis"
    target_z: float | None = None
    # ``gravity_comp`` feeds forward this fraction of the robot's weight as a steady upward
    # force, so the PD term only has to correct deviations (robust to robot mass). 1.0 =
    # fully suspended (feet may dangle); lower it (e.g. 0.6) when the feet should stay
    # loaded, as in the contact showcase.
    gravity_comp: float = 1.0
    stiffness: float = 800.0
    damping: float = 100.0
    horizontal: bool = True
    horizontal_stiffness: float = 120.0
    horizontal_damping: float = 24.0
    keep_upright: bool = True
    # Validated on the 33 kg Unitree G1: ``upright_stiffness=400`` holds it perfectly
    # vertical; pushing past ~1500 destabilises the contact solver (explodes).
    upright_stiffness: float = 400.0
    upright_damping: float = 40.0


class ShowcaseRunner:
    """Base class for the six play-only showcases.

    The runner builds a :class:`ManagerBasedRlEnv` from ``env_cfg``, then drives
    ``num_steps`` calls to :py:meth:`env.step` with the action returned from
    :py:meth:`_scripted_action`. Every ``log_interval`` steps it calls
    :py:meth:`_dump` so each subclass can write whatever evidence is relevant for its
    feature (PNG frames, histograms, tracking error, ...).

    ``log_root`` is created lazily on first dump. The default lives under
    ``logs/showcase/<task_slug>/``.
    """

    task_slug: str = "showcase"

    def __init__(
        self,
        env_cfg: ManagerBasedRlEnvCfg,
        *,
        num_steps: int | None = None,
        log_interval: int = 20,
        realtime: bool = True,
        realtime_scale: float = 1.0,
        spring: SpringCfg | None = None,
    ) -> None:
        self.env_cfg = env_cfg
        # Default ``num_steps`` from ``env.simulation.steps`` so the CLI ``--steps``
        # override (which lands on ``env.simulation.steps``) actually drives the loop
        # length. Subclasses can still pin a hard default via ``num_steps=...``.
        self.num_steps = int(num_steps if num_steps is not None else env_cfg.simulation.steps)
        self.log_interval = max(1, int(log_interval))
        self._log_root: Path | None = None
        self._env: ManagerBasedRlEnv | None = None
        # G3 real-time pacing: hold playback to ``1 / realtime_scale`` of wall-clock so
        # motion reads naturally in the viewer instead of blurring past at sim speed.
        # Only paces when the viewer is shown — headless / CI runs stay full-speed.
        self.realtime = bool(realtime)
        self.realtime_scale = max(1e-6, float(realtime_scale))
        # G4 virtual spring (None = off). Resolved link + captured anchor are lazy.
        self.spring = spring
        self._spring_link: Any = None
        self._spring_weight: float = 0.0
        self._spring_anchor_xy: torch.Tensor | None = None
        self._spring_target_z: float | None = None if spring is None else spring.target_z

    # ------------------------------------------------------------------ hooks

    def _scripted_action(self, env: ManagerBasedRlEnv, step: int) -> torch.Tensor:
        """Return the per-step action tensor of shape ``(num_envs, num_actions)``.

        Default: zero action (each `JointPositionActionCfg` with
        ``use_default_offset=True`` interprets this as "go to ``default_joint_pos``").
        """

        return torch.zeros(env.num_envs, env.num_actions, device=env.device)

    def _dump(self, env: ManagerBasedRlEnv, step: int) -> None:
        """Subclass hook — called every ``log_interval`` steps. Default: no-op."""

    def _post_step(self, env: ManagerBasedRlEnv, step: int) -> None:
        """Subclass hook — called every step after ``env.step`` while the viewer is open.

        The home for per-step debug visualisation (camera frustum, ray casts, contact
        markers). Default: no-op. Only fires when the viewer is shown, so headless runs
        skip it. Implementations may use ``env.scene.gs_scene.draw_debug_*`` and should
        clear their previous debug objects to avoid accumulation.
        """

    # ------------------------------------------------------------------ utilities

    def log_root(self) -> Path:
        """Lazily create and return the per-showcase log directory."""

        if self._log_root is None:
            root = Path("logs") / "showcase" / self.task_slug
            root.mkdir(parents=True, exist_ok=True)
            self._log_root = root
        return self._log_root

    def _step_dt(self) -> float:
        """Wall-clock seconds one control step should occupy at 1× real time."""

        decimation = int(getattr(self.env_cfg, "decimation", 1) or 1)
        return float(self.env_cfg.simulation.dt) * decimation

    def _apply_spring(self, env: ManagerBasedRlEnv) -> None:
        """G4: apply the virtual-spring force/torque on the configured link (if any).

        No-op when ``self.spring is None``. Reaches the Genesis rigid solver through the
        robot's ``gs_handle`` to apply a world-frame linear force (vertical hold +
        optional horizontal centering) and a world-frame restoring torque (uprighting).
        """

        spr = self.spring
        if spr is None:
            return
        robot = env.articulations["robot"]
        if self._spring_link is None:
            local_idx = robot.link_names.index(spr.link_name)
            self._spring_link = robot.gs_handle.links[local_idx]
            self._spring_weight = float(robot.gs_handle.get_mass()) * 9.81
        link = self._spring_link
        rs = robot.data
        n = env.num_envs
        device = env.device

        pos = rs.root_pos  # (n, 3)
        vel = rs.root_lin_vel_w  # (n, 3)
        # Default the hold height to the robot's configured standing height (``init_pos``),
        # not the spawn z — tasks often drop the robot from above, so the spawn frame is
        # mid-air and would make the spring hold it too high. Fall back to the spawn z.
        if self._spring_target_z is None:
            init_pos = getattr(robot.cfg, "init_pos", None)
            self._spring_target_z = (
                float(init_pos[2]) if init_pos is not None else float(pos[:, 2].mean().item())
            )
        if self._spring_anchor_xy is None:
            self._spring_anchor_xy = pos[:, :2].clone()

        force = torch.zeros(n, 1, 3, device=device)
        # Gravity feed-forward + PD around the target height.
        force[:, 0, 2] = (
            spr.gravity_comp * self._spring_weight
            + spr.stiffness * (self._spring_target_z - pos[:, 2])
            - spr.damping * vel[:, 2]
        )
        if spr.horizontal:
            force[:, 0, :2] = (
                spr.horizontal_stiffness * (self._spring_anchor_xy - pos[:, :2])
                - spr.horizontal_damping * vel[:, :2]
            )
        link.solver.apply_links_external_force(force, (link.idx,), ref="link_com")

        if spr.keep_upright:
            world_up = torch.tensor(_WORLD_UP, device=device).expand(n, 3)
            up_w = quat_apply(rs.root_quat, world_up)  # base up-axis in world
            err = torch.cross(up_w, world_up, dim=-1)  # rotates up_w back toward world up
            torque = (
                spr.upright_stiffness * err - spr.upright_damping * rs.root_ang_vel_w
            ).unsqueeze(1)
            link.solver.apply_links_external_torque(torque, (link.idx,), ref="link_com")

    # ------------------------------------------------------------------ play

    def play(self, *, max_steps: int | None = None) -> None:
        """Build the env, run the scripted loop, close cleanly.

        The kernel sets :py:attr:`env.viewer_closed` when the user closes the
        Genesis viewer; the loop polls that flag and breaks. No need to catch
        ``GenesisException`` here.

        Loop length follows the same soft-config / hard-cap split as the RL play
        helper (:func:`genelab.rl.runner.play_task`), so ``genelab play`` behaves
        identically whichever runner backs the task:

        * ``max_steps`` set (the ``--max-steps`` flag): a hard cap that always wins —
          the loop stops after exactly that many steps, viewer open or not.
        * else viewer shown (``vis=True``): unbounded — run until the user closes the
          window. The soft ``simulation.steps`` config is ignored, exactly as it is
          for RL playback with a viewer.
        * else headless: cap at ``self.num_steps`` (the soft ``simulation.steps``
          config, what ``--steps`` sets) so a no-window run is a bounded smoke rollout
          instead of looping forever.
        """

        ensure_project_cache()
        env = ManagerBasedRlEnv(self.env_cfg)
        self._env = env
        # Reset env-scoped lazy spring state: ``play()`` builds a fresh env each call, so a
        # cached link/solver handle from a previous run would point into a destroyed scene.
        self._spring_link = None
        self._spring_weight = 0.0
        self._spring_anchor_xy = None
        self._spring_target_z = None if self.spring is None else self.spring.target_z
        # Debug drawing / pacing only make sense with a viewer (headless stays full-speed).
        viewer_shown = bool(self.env_cfg.simulation.vis)
        # Resolve the loop bound: hard cap wins; else viewer => unbounded; else soft config.
        # ``None`` means "no step bound — only ``viewer_closed`` ends the loop".
        if max_steps is not None:
            bound: int | None = int(max_steps)
        elif viewer_shown:
            bound = None
        else:
            bound = self.num_steps
        pace = self.realtime and viewer_shown
        step_budget = self._step_dt() / self.realtime_scale
        next_tick = time.monotonic()
        last_step = 0
        try:
            env.reset()
            step = 0
            while bound is None or step < bound:
                action = self._scripted_action(env, step)
                self._apply_spring(env)
                env.step(action)
                last_step = step
                if env.viewer_closed:
                    break
                if step % self.log_interval == 0:
                    self._dump(env, step)
                if viewer_shown:
                    self._post_step(env, step)
                if pace:
                    next_tick += step_budget
                    sleep_for = next_tick - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        # Fell behind (slow step); resync so we don't accumulate debt.
                        next_tick = time.monotonic()
                step += 1
            # Dump one final frame so the last state is captured — but only when the
            # viewer is still open. If the user closed the window, the offscreen
            # renderer is torn down with it, so a final camera-backed ``_dump`` would
            # raise ``GenesisException("Viewer already closed.")``. The in-loop dumps
            # already captured the run, so skipping the final one on close is safe.
            if not env.viewer_closed:
                self._dump(env, last_step)
        finally:
            env.close()
            self._env = None
