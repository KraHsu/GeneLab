"""Genesis-backed manager-based RL environment.

This is a slim port of ``envs.manager_based_rl_env`` adapted to Genesis. The env owns
the seven manager-style MDP hooks and orchestrates an :class:`~genelab.scene.InteractiveScene`
that in turn owns the Genesis ``gs.Scene``, an articulated robot, and any extra rigid bodies.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from genelab.bridges.base import BridgeCfg
from genelab.configs import ManagerBasedEnvCfg
from genelab.entity import Articulation, ArticulationCfg
from genelab.managers import (
    ActionManager,
    ActionTermCfg,
    CommandManager,
    CommandTermCfg,
    CurriculumManager,
    CurriculumTermCfg,
    EventManager,
    EventTermCfg,
    MetricsManager,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationManager,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)
from genelab.scene import InteractiveScene
from genelab.sensor import Sensor

# Back-compat alias: existing extension packages import ``RobotEntityCfg`` from this module.
# The new canonical name is :class:`genelab.entity.ArticulationCfg`. Plain class alias so the
# name remains callable as a constructor (PEP 695 ``type`` aliases are not callable).
RobotEntityCfg = ArticulationCfg


@dataclass
class ManagerBasedRlEnvCfg(ManagerBasedEnvCfg):
    """RL-flavored env config: composes per-manager term dictionaries.

    Backwards-compatible with ``ManagerBasedEnvCfg`` so existing tooling (``apply_overrides``,
    CLI listing) keeps working.
    """

    decimation: int = 4
    episode_length_s: float = 20.0
    # When False, ``step`` stops auto-resetting envs that hit a termination — used by
    # interactive single-robot play (teleop) so the robot keeps going instead of snapping back
    # on time-out / stumble. Training leaves it True so episodes roll over normally.
    auto_reset: bool = True
    seed: int | None = None
    scale_rewards_by_dt: bool = True

    robot: ArticulationCfg = field(default_factory=ArticulationCfg)
    # Multi-robot: when non-empty, ``robots`` spawns
    # one articulation per entry (keyed by name) and ``robot`` is ignored. When empty, the
    # env falls back to ``{"robot": robot}`` — so single-robot tasks are unchanged. The
    # entity named ``"robot"`` (or the first key) is the *primary*, backing the singular
    # ``env.robot`` / ``env.robot_state`` accessors until they are removed in a later slice.
    robots: dict[str, ArticulationCfg] = field(default_factory=dict)

    actions_cfg: dict[str, ActionTermCfg] = field(default_factory=dict)
    observations_cfg: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    rewards_cfg: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations_cfg: dict[str, TerminationTermCfg] = field(default_factory=dict)
    commands_cfg: dict[str, CommandTermCfg] = field(default_factory=dict)
    events_cfg: dict[str, EventTermCfg] = field(default_factory=dict)
    curriculum_cfg: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    metrics_cfg: dict[str, MetricsTermCfg] = field(default_factory=dict)
    # Play-only: consumed by ``genelab.rl.runner.play_task`` to drive live I/O
    # (keyboard teleop, ImGui sliders, ROS2 publishers, …). ``train_task``
    # ignores this field. Empty by default — bridges are opt-in per env cfg.
    bridges_cfg: dict[str, BridgeCfg] = field(default_factory=dict)


def _resolve_entity_cfgs(cfg: ManagerBasedRlEnvCfg) -> dict[str, ArticulationCfg]:
    """Entities to spawn: ``cfg.robots`` when non-empty, else ``{"robot": cfg.robot}``."""
    if cfg.robots:
        return dict(cfg.robots)
    return {"robot": cfg.robot}


def _primary_entity_name(entity_cfgs: dict[str, ArticulationCfg]) -> str:
    """The primary entity name — ``"robot"`` if present, else the first declared key."""
    return "robot" if "robot" in entity_cfgs else next(iter(entity_cfgs))


class ManagerBasedRlEnv:
    """Genesis-backed manager-based RL environment."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg) -> None:
        self.cfg = cfg
        self._device = cfg.device
        self._num_envs = max(1, int(cfg.simulation.num_envs))
        self._dt_sim = float(cfg.simulation.dt)
        self._decimation = int(cfg.decimation)
        self._auto_reset = bool(cfg.auto_reset)
        self._step_dt = self._dt_sim * self._decimation
        self._max_episode_length = max(1, int(math.ceil(cfg.episode_length_s / self._step_dt)))

        self._scene = InteractiveScene(cfg.simulation, cfg.scene, device_hint=cfg.device)
        entity_cfgs = _resolve_entity_cfgs(cfg)
        for name, robot_cfg in entity_cfgs.items():
            self._scene.add_entity(name, robot_cfg)
        self._scene.build()
        # Genesis may have refined the device string (e.g. ``cuda:0``); pick it up.
        self._device = self._scene.device
        self._articulations: dict[str, Articulation] = {
            name: self._scene.articulations[name] for name in entity_cfgs
        }
        self._primary_name = _primary_entity_name(entity_cfgs)
        # ``_articulation`` is the primary entity, backing the singular ``env.robot`` /
        # ``env.robot_state`` accessors (removed in a later slice once terms route by name).
        self._articulation: Articulation = self._articulations[self._primary_name]

        self._episode_length_buf = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        # Global control-step counter (transitions seen across all envs / resets). Curriculum
        # terms read it to ramp difficulty over training time — see ``mdp.commands_vel``.
        self.common_step_counter: int = 0
        self._extras: dict[str, Any] = {}

        # Managers (order matters: actions before observations/rewards; commands before obs)
        self.action_manager = ActionManager(cfg.actions_cfg, self)
        self.command_manager = CommandManager(cfg.commands_cfg, self)
        self.observation_manager = ObservationManager(cfg.observations_cfg, self)
        self.reward_manager = RewardManager(
            cfg.rewards_cfg, self, scale_by_dt=cfg.scale_rewards_by_dt
        )
        self.termination_manager = TerminationManager(cfg.terminations_cfg, self)
        self.event_manager = EventManager(cfg.events_cfg, self)
        self.curriculum_manager = CurriculumManager(cfg.curriculum_cfg, self)
        self.metrics_manager = MetricsManager(cfg.metrics_cfg, self)

        # Sensors are pre-built by ``InteractiveScene.build`` so any Genesis resources
        # snapshotted at scene-build time (e.g. BatchRenderer cameras) are already
        # registered. Bind every pre-built sensor against the env after managers are
        # constructed but before the initial reset — observation_manager.compute can
        # then read sensor.data on the first frame.
        self._sensors: dict[str, Sensor[Any]] = dict(self._scene.sensors)
        for sensor in self._sensors.values():
            sensor.bind(self)

        # Recorder bridge (when the scene declared ``recordings``): now that sensors
        # are bound and the env is fully constructed, patch per-recorder sample rates
        # for sensor-name sources to the control-step decimation.
        bridge = self._scene.recorder_bridge
        if bridge is not None:
            bridge.bind_env(self)

        self.event_manager.apply("startup")
        self._articulation.refresh()
        # Initial reset of all envs so the policy sees a well-defined obs.
        self.reset()

    # ------------------------------------------------------------------ properties

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def dt(self) -> float:
        return self._step_dt

    @property
    def physics_dt(self) -> float:
        return self._dt_sim

    @property
    def max_episode_length(self) -> int:
        return self._max_episode_length

    @property
    def max_episode_length_s(self) -> float:
        return float(self.cfg.episode_length_s)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_length_buf

    @property
    def num_actions(self) -> int:
        return int(self.action_manager.total_action_dim)

    @property
    def scene(self) -> InteractiveScene:
        """The live :class:`InteractiveScene`. Distinct from ``env.cfg.scene`` (cfg dataclass)."""
        return self._scene

    @property
    def viewer_closed(self) -> bool:
        """``True`` once the user closes the Genesis viewer mid-rollout.

        The kernel catches Genesis's ``GenesisException("Viewer closed.")`` inside
        :py:meth:`InteractiveScene.step` and flips this flag; loop drivers (RL
        runners, showcase scripts, custom rollouts) should poll it and break
        instead of writing their own try / except.
        """
        return self._scene.viewer_closed

    @property
    def articulations(self) -> dict[str, Articulation]:
        """All spawned articulations keyed by name.

        Multi-robot terms / sensors select an entity via ``env.articulations[name]``
        (``name`` from ``SceneEntityCfg.name`` / ``asset_name``). Single-robot tasks have
        a single ``"robot"`` entry that the singular ``env.robot*`` accessors alias.
        """
        return self._articulations

    @property
    def sensors(self) -> dict[str, Sensor[Any]]:
        return self._sensors

    @property
    def env_origins(self) -> torch.Tensor:
        """Per-env world-frame offset ``[num_envs, 3]``; zeros when Genesis uses local frames."""
        return self._scene.env_origins

    # The singular entity accessors (``robot`` / ``robot_state`` /
    # ``articulation``) and the name-table convenience properties (``joint_names`` /
    # ``link_names`` / ``default_joint_pos`` / ``actuators`` / ``joint_*_limits``) were
    # removed. Reach an entity by name: ``env.articulations[name]`` (its ``.gs_handle`` /
    # ``.data`` / ``.joint_names`` / …). Single-robot code uses ``env.articulations["robot"]``.

    # ------------------------------------------------------------------ reference state

    def write_joint_state_to_sim(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write actuated-joint positions / velocities for ``env_ids`` (size ``[N, num_dofs]``)."""
        self._articulation.write_joint_state(joint_pos, joint_vel, env_ids)

    def write_root_state_to_sim(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write floating-base pose + velocity for ``env_ids`` (each size ``[N, 3 or 4]``)."""
        self._articulation.write_root_state(
            root_pos, root_quat, root_lin_vel_w, root_ang_vel_w, env_ids
        )

    # ------------------------------------------------------------------ rollout

    def reset(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self._reset_idx(env_ids)
        self._articulation.refresh()
        # Re-prime the joint-acceleration baseline to the reset velocities so the first
        # post-reset step doesn't finite-difference across the discontinuity.
        self._articulation.reset_acceleration(env_ids)
        obs = self.observation_manager.compute()
        return obs, dict(self._extras)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        # Reset joint state to default before events run, so events can layer randomisation.
        self._articulation.reset(env_ids)
        self.event_manager.apply("reset", env_ids)
        self.command_manager.reset(env_ids)
        self.action_manager.reset(env_ids)
        for sensor in self._sensors.values():
            sensor.reset(env_ids)
        bridge = self._scene.recorder_bridge
        if bridge is not None:
            bridge.on_reset(env_ids)
        reward_extras = self.reward_manager.reset(env_ids)
        term_extras = self.termination_manager.reset(env_ids)
        curr_extras = self.curriculum_manager.compute(env_ids)
        metrics_extras = self.metrics_manager.reset(env_ids)
        self._episode_length_buf[env_ids] = 0
        self._extras["log"] = {
            **reward_extras,
            **term_extras,
            **curr_extras,
            **metrics_extras,
        }

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # ``action_manager.process_action`` copies into a pre-allocated buffer, so we don't
        # need ``.clone()`` here; the ``.to`` call is a no-op when the action is already on
        # the right device.
        action = action.to(self._device, non_blocking=True)
        self.action_manager.process_action(action)
        # Render only on the final physics tick of the decimation loop. Genesis's viewer
        # update calls ``rate.sleep`` at ``ViewerOptions.max_FPS``, so updating every
        # tick coupled the wall-clock rate to the physics rate and produced slow-motion
        # playback at large decimation values.
        last_tick = self._decimation - 1
        for tick in range(self._decimation):
            self.action_manager.apply_action()
            self._scene.step(update_visualizer=(tick == last_tick))
        self._articulation.refresh()
        # Advance the joint-acceleration finite difference once per control step (Genesis
        # has no DoF-acceleration getter); mdp.joint_acc_l2 reads the result.
        self._articulation.update_acceleration(self._step_dt)
        for sensor in self._sensors.values():
            sensor.update(self._step_dt)
        self._episode_length_buf += 1
        self.common_step_counter += 1
        # Resample commands and trigger interval-mode events.
        self.command_manager.compute(self._step_dt)
        self.event_manager.apply("interval", dt=self._step_dt)

        reward = self.reward_manager.compute(self._step_dt)
        self.metrics_manager.compute()
        dones = self.termination_manager.compute()
        time_outs = self.termination_manager.time_outs
        terminated = self.termination_manager.terminated

        # Auto-reset envs that hit done (disabled for interactive teleop via cfg.auto_reset).
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if self._auto_reset and reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            self._articulation.refresh()
            self._articulation.reset_acceleration(reset_ids)
        obs = self.observation_manager.compute()
        # The RSL-RL wrapper shallow-copies ``extras`` itself before mutating, so we hand back
        # the live dict to skip an unnecessary per-step copy.
        return obs, reward, terminated, time_outs, self._extras

    def close(self) -> None:
        self._scene.close()


if TYPE_CHECKING:
    from typing import cast

    from genelab.contracts import EnvContext

    # ManagerBasedRlEnv is the adapter for the EnvContext port. This
    # type-only assignment makes pyright fail CI if the env stops conforming.
    _env_context_conformance: EnvContext = cast("ManagerBasedRlEnv", ...)
