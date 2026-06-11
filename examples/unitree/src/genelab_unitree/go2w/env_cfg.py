"""Velocity-tracking env config for the Unitree Go2-W wheeled quadruped on flat ground.

Go2-W mixes two control modes, so the action space is split:

* the 12 leg joints (hip / thigh / calf) are **position**-controlled via
  ``JointPositionAction`` (the policy commands a posture offset);
* the 4 wheels are **velocity**-controlled via ``JointVelocityAction`` (the policy commands
  a wheel angular velocity that rolls the robot — the new core velocity-action path).

Like the Go1 example the deployable actor is proprioception-only (no base linear velocity —
real hardware lacks the sensor); a privileged critic gets the true base velocity for a usable
value function (asymmetric actor-critic).

NOTE: wheel *positions* spin unbounded, so ``joint_pos`` here includes a wrapping signal for
the 4 wheels. Wheel *velocities* (the signal that matters) are clean. A training-grade config
should exclude wheel positions from the actor obs (or sin/cos encode them); kept simple here.

ROBUSTNESS / sim2sim (Genesis -> MuJoCo): the bare single-frame policy transfers poorly
(spin-in-place / strafe falls) because it overfits Genesis dynamics and can't read velocity
trends. Two levers harden it, both wired below for training only:

* **Frame stacking** — the actor / critic obs are the last ``_OBS_HISTORY`` (=5) control-step
  frames concatenated **frame-major, oldest -> newest**: ``[f₀ | f₁ | … | f₄]`` where each
  ``fᵢ`` is the full single-frame term concat in declared order. A deployment bridge MUST feed
  the policy this same stacked layout (push one frame per control step, backfill on reset).
* **Domain randomization + action perturbation** — startup-mode DR on the physical constants
  the policy can't observe (wheel friction, trunk mass / COM, PD gains, encoder bias) plus a
  per-env action latency and per-step action noise. All disabled in ``play`` so the deployed
  policy is clean.
"""

import math
from typing import TYPE_CHECKING

from genelab import mdp
from genelab.asset_zoo.unitree_go2w import UnitreeGo2WCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.actions.joint_velocity import JointVelocityActionCfg
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from genelab.mdp.noise import Unoise
from genelab.sensor import BodyVelocitySensorCfg, ContactSensorCfg, SensorCfg

if TYPE_CHECKING:
    from genelab.terrains import TerrainGeneratorCfg

# Floating-base body (go2w.xml <body name="base_link">).
_BASE_LINK = "base_link"
# Leg joints (position-controlled) and wheel joints (velocity-controlled).
_LEG_JOINTS_RE = r".*_(hip|thigh|calf)_joint"
_WHEEL_JOINTS_RE = r".*_wheel_joint"
# Wheels are the ground contact; thighs/calves should not bear load.
_WHEEL_LINKS: tuple[str, ...] = ("FR_wheel_link", "FL_wheel_link", "RR_wheel_link", "RL_wheel_link")
_THIGH_CALF_LINKS: tuple[str, ...] = tuple(
    f"{leg}_{seg}" for leg in ("FR", "FL", "RR", "RL") for seg in ("thigh", "calf")
)
# Standing trunk height for the base-height penalty target (matches the asset init z).
_GO2W_BASE_HEIGHT_TARGET = 0.40
# Raw action [-1, 1] -> wheel angular velocity (rad/s). Starting value; tune against rolling.
_WHEEL_VEL_SCALE = 20.0
# Frame stacking: feed the actor (and critic) the last N control-step frames so a
# proprioception-only policy can infer base/wheel velocity and contact trends it can't read
# instantaneously. ~100 ms of history at the 50 Hz control rate — the main lever for the
# rough sim2sim (Genesis -> MuJoCo) transfer the bare single-frame policy fails at.
_OBS_HISTORY = 5


def _trunk_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(name="robot", link_names=(_BASE_LINK,))


def _obs_terms() -> dict[str, ObservationTermCfg]:
    """Actor / critic-shared proprioception. No base *linear* velocity (deployability)."""
    return {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.sensor_data, params={"sensor_name": "imu_ang_vel"}, noise=Unoise(-0.2, 0.2)
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(-0.05, 0.05)
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(-1.5, 1.5)),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "twist"}
        ),
    }


def _velocity_env_cfg_base(
    *,
    play: bool = False,
    lock_wheels: bool = False,
    terrain: "TerrainGeneratorCfg | None" = None,
    extra_sensors: tuple[SensorCfg, ...] = (),
    extra_actor_obs: dict[str, ObservationTermCfg] | None = None,
    extra_critic_obs: dict[str, ObservationTermCfg] | None = None,
    extra_curriculum: dict[str, CurriculumTermCfg] | None = None,
) -> ManagerBasedRlEnvCfg:
    """Shared velocity-tracking skeleton for the Unitree Go2-W.

    ``lock_wheels`` is the **wheeled-legged curriculum's stage 1**: the wheel velocity command
    is zeroed (``scale=0``) and the wheel joints are stiffly damped into rigid round feet, so
    the robot cannot roll and must *step* — including sideways via hip abduction — to track any
    command. This bootstraps a legged crab-walk gait that pure-wheeled training never discovers
    (rolling wheels can't strafe — Go2-W is a skid-steer platform). Stage 2 (``lock_wheels=False``,
    the shipped config) warm-starts from the stage-1 checkpoint and lets the wheels roll again;
    the 16-dim action space is identical across stages so the policy transfers."""
    policy_terms = _obs_terms()
    policy_terms.update(extra_actor_obs or {})
    critic_terms = _obs_terms()
    # Privileged critic obs (training-only): true base linear velocity the deployable actor
    # can't sense — gives the value function an accurate read on velocity-tracking error.
    critic_terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.sensor_data, params={"sensor_name": "imu_lin_vel"}
    )
    critic_terms.update(extra_critic_obs or {})

    # Stage 1 locks the wheels into rigid feet: kill the wheel velocity command and crank the
    # wheel velocity-tracking gain + torque cap so any roll is strongly resisted.
    robot = UnitreeGo2WCfg()
    if lock_wheels:
        robot.actuators["wheel"].damping = 20.0
        robot.actuators["wheel"].effort_limit = 50.0
    else:
        # Stiffer-than-asset wheel damping (0.5 -> 5.0): a stance wheel commanded to zero
        # velocity actually brakes instead of free-rolling out from under the leg. Probed:
        # at 0.5 the stage-2 crab-walk is fall-prone (stance feet roll away mid-step) and
        # the policy abandons lateral stepping; the firmer gain also tracks the commanded
        # wheel speed better for skid-steer yaw.
        robot.actuators["wheel"].damping = 5.0
    wheel_scale = 0.0 if lock_wheels else _WHEEL_VEL_SCALE

    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4096 if not play else 1, dt=0.002, substeps=1, vis=play, gpu=True
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.5, 2.5),
            terrain=terrain,
            sensors=(
                BodyVelocitySensorCfg(name="imu_lin_vel", link_name=_BASE_LINK, measure="lin_vel"),
                BodyVelocitySensorCfg(name="imu_ang_vel", link_name=_BASE_LINK, measure="ang_vel"),
                # Wheel-ground contact (the wheels are the feet).
                ContactSensorCfg(
                    name="wheel_contact", link_names=_WHEEL_LINKS, track_air_time=True
                ),
                # Thigh / calf contact for the undesired-contacts safety penalty.
                ContactSensorCfg(name="body_contact", link_names=_THIGH_CALF_LINKS),
                *extra_sensors,
            ),
        ),
        decimation=10,
        episode_length_s=20.0,
        device="cuda",
        robot=robot,
        actions_cfg={
            # Legs: position control (posture offset from default). Wheels: velocity control.
            "joint_pos": JointPositionActionCfg(
                asset_name="robot", joint_names=(_LEG_JOINTS_RE,), use_default_offset=True
            ),
            "wheel_vel": JointVelocityActionCfg(
                asset_name="robot", joint_names=(_WHEEL_JOINTS_RE,), scale=wheel_scale
            ),
        },
        observations_cfg={
            "policy": ObservationGroupCfg(
                enable_corruption=True, terms=policy_terms, history_length=_OBS_HISTORY
            ),
            "critic": ObservationGroupCfg(
                enable_corruption=False, terms=critic_terms, history_length=_OBS_HISTORY
            ),
        },
        commands_cfg={
            "twist": UniformVelocityCommandCfg(
                asset_name="robot",
                resampling_time_range=(3.0, 8.0),
                rel_standing_envs=0.1,
                rel_heading_envs=0.0,
                rel_forward_envs=0.1,
                heading_command=False,
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-1.0, 1.0),
                    heading=(-math.pi, math.pi),
                ),
            )
        },
        rewards_cfg={
            # --- velocity tracking (positive) ---
            "track_lin_vel": RewardTermCfg(
                func=mdp.track_linear_velocity_xy_exp,
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            "track_ang_vel": RewardTermCfg(
                func=mdp.track_angular_velocity_z_exp,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            # --- base stability ---
            "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
            "ang_vel_xy": RewardTermCfg(
                func=mdp.body_angular_velocity_penalty,
                weight=-0.05,
                params={"asset_cfg": _trunk_cfg()},
            ),
            "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
            "base_height": RewardTermCfg(
                func=mdp.base_height_l2,
                weight=-1.0,
                params={"target_height": _GO2W_BASE_HEIGHT_TARGET},
            ),
            # --- actuation / smoothness ---
            "joint_torques": RewardTermCfg(func=mdp.applied_torque_l2, weight=-2.0e-4),
            "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-5),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
            "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            # --- safety: only the wheels should touch the ground ---
            "undesired_contacts": RewardTermCfg(
                func=mdp.contact_force_limit,
                weight=-1.0,
                params={"sensor_name": "body_contact", "max_force": 1.0},
            ),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(
                func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
            ),
        },
        events_cfg={
            "reset_base": EventTermCfg(
                mode="reset",
                func=mdp.reset_root_state_uniform,
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
                    "velocity_range": {},
                },
            ),
            "reset_robot_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_by_offset,
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
        curriculum_cfg={
            "command_vel": CurriculumTermCfg(
                func=mdp.commands_vel,
                params={
                    "command_name": "twist",
                    "velocity_stages": [
                        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)}
                    ],
                },
            ),
            **(extra_curriculum or {}),
        },
    )

    if lock_wheels:
        # Legged gait shaping (the Go1 recipe) — without it stage 1 converges to the
        # stand-still optimum (probed: a 6k run without these tracked 0–16 % of any
        # command). ``feet_air_time`` cannot be earned standing (command-gated), so it is
        # the gradient that forces the legs to step; ``feet_slip`` stops the planted
        # wheels skating across the ground instead of lifting.
        cfg.rewards_cfg["air_time"] = RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.25,
            params={
                "sensor_name": "wheel_contact",
                "threshold_min": 0.1,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        )
        cfg.rewards_cfg["foot_slip"] = RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": "wheel_contact",
                "asset_cfg": SceneEntityCfg(name="robot", link_names=_WHEEL_LINKS),
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        )
    else:
        # Rolling-wheel (stage 2 / shipped) task: a *lateral-gated* air_time. The wheels
        # cannot roll sideways (skid-steer), so stepping is the only way to track vy — but
        # without a standing reward gradient the warm-started policy abandons the crab-walk
        # (probed: vy tracking collapsed 93 % -> 4 % mid-stage-2). Gating on |cmd_vy| only
        # keeps rolling optimal for pure forward/yaw commands, where the gate stays closed.
        # No foot_slip here — it would penalize normal wheel rolling. Weight 0.5: probed at
        # 0.25 the term contributed ~0.02/step and was drowned out by the stability terms.
        cfg.rewards_cfg["air_time"] = RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.5,
            params={
                "sensor_name": "wheel_contact",
                "threshold_min": 0.1,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.1,
                # vy AND wz: the wheels can't strafe, and they can't scrub-turn *slowly*
                # either (stiction deadband near zero wheel speed — probed: wz=0.2 tracked
                # 32 % while wz=1.0 hit 96 %). Stepping must stay rewarded for both lateral
                # and rotation demand; pure-vx keeps the gate closed (rolling is optimal).
                "command_axes": (1, 2),
            },
        )
        # Unsaturated lateral-tracking gradient: the exp kernel's pull vanishes once the
        # policy fully ignores vy (exp(-0.64/0.25) ≈ 0.08 at a 0.8 m/s error — flat), which
        # is how the warm-started crab-walk died (93 % -> 4 % vy tracking, probed). The L1
        # error keeps a constant-magnitude push back toward lateral stepping at any distance.
        cfg.rewards_cfg["vy_error_l1"] = RewardTermCfg(
            func=mdp.velocity_tracking_error_l1,
            weight=-0.5,
            params={"command_name": "twist", "axes": (1,)},
        )
        # Yaw needs the same unsaturated pull: probed at 6k, in-place rotation collapsed to
        # 2 % of the command while mixed-command yaw still scored ~0.46 reward — the exp
        # kernel masks per-command abandonment, the L1 doesn't. Weight stays −0.5: a −1.0
        # attempt made small-yaw marginally better but the policy abandoned the lateral
        # gait wholesale (±vy fell 64/64) — the slow-yaw deadband is solved by stepping
        # turns (air_time gate covers wz) rather than a steeper penalty.
        cfg.rewards_cfg["wz_error_l1"] = RewardTermCfg(
            func=mdp.angular_velocity_tracking_error_l1,
            weight=-0.5,
            params={"command_name": "twist"},
        )

    if not play:
        cfg.events_cfg["push_robot"] = EventTermCfg(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            func=mdp.push_by_setting_velocity,
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
        )

        # --- sim2real startup DR (sampled once per env at training start) ---
        # The Genesis-trained policy overfits Genesis dynamics; randomizing the physical
        # constants the policy can't observe is what closes the gap to MuJoCo's contact /
        # actuator model — the cause of the spin-in-place / strafe falls. All startup-mode
        # (Isaac Lab's standard sim2real pattern): one constant per env for the whole episode.
        #
        # Wheel-ground friction: the wheels are the only contact, so the rolling/turning
        # dynamics are dominated by this coefficient — the single highest-leverage term for
        # the in-place-rotation and lateral-slide failures. ``shared_random`` so all four
        # wheels of an env share one ground material.
        cfg.events_cfg["wheel_friction"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", link_names=_WHEEL_LINKS),
                "ranges": (0.3, 1.5),
                "shared_random": True,
            },
        )
        # Trunk payload + COM offset: models an unknown load / build tolerance so the policy
        # can't assume a perfectly centered, fixed-mass base.
        cfg.events_cfg["base_mass"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.body_mass_offset,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", link_names=(_BASE_LINK,)),
                "ranges": (-1.0, 2.0),
            },
        )
        cfg.events_cfg["base_com"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", link_names=(_BASE_LINK,)),
                "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        # PD-gain calibration sweep (±20%): the leg-position and wheel-velocity gains differ
        # between Genesis and MuJoCo; randomizing them keeps the policy from baking in one
        # sim's exact actuator response.
        cfg.events_cfg["joint_gains"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.randomize_joint_stiffness_damping,
            params={
                "asset_cfg": SceneEntityCfg(name="robot"),
                "stiffness_range": (0.8, 1.2),
                "damping_range": (0.8, 1.2),
            },
        )
        # Per-joint encoder bias: a small constant offset between perceived and physical joint
        # angle the policy must learn to compensate. Only the position-controlled legs feel it
        # (``JointPositionAction`` subtracts the bias from its PD target); the velocity-controlled
        # wheel entries are written but inert (``JointVelocityAction`` never reads them), so the
        # whole-robot cfg is equivalent to a leg-only filter without enumerating joint names
        # (``SceneEntityCfg`` matches literal names, not the leg regex).
        cfg.events_cfg["encoder_bias"] = EventTermCfg(
            mode="startup",
            func=mdp.dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg(name="robot"),
                "bias_range": (-0.015, 0.015),
            },
        )

        # --- training-only action perturbation (sim2real control-loop hardening) ---
        # A few-step random latency plus per-step noise so the policy tolerates the stale,
        # noisy command path of a real / cross-sim deploy instead of assuming an instant,
        # exact actuator response.
        cfg.action_noise_std = 0.05
        cfg.action_delay_steps = (0, 2)

    if play:
        cfg.auto_reset = False
        cfg.simulation.camera_lookat = (0.0, 0.0, 0.3)
        cfg.simulation.camera_pos = (2.0, -2.0, 1.0)
        try:
            import imgui_bundle  # noqa: F401

            from genelab.bridges.imgui import ImGuiTwistBridgeCfg

            cfg.simulation.viewer_imgui = True
            cfg.bridges_cfg["teleop"] = ImGuiTwistBridgeCfg(
                command_name="twist",
                vx_range=(-1.5, 1.5),
                vy_range=(-1.0, 1.0),
                wz_range=(-1.0, 1.0),
                default_vx=0.5,
                default_vy=0.0,
                default_wz=0.0,
            )
        except ImportError:
            pass

    return cfg


def unitree_go2w_velocity_env_cfg(
    play: bool = False, lock_wheels: bool = False
) -> ManagerBasedRlEnvCfg:
    """Flat-ground velocity-tracking env config for the Unitree Go2-W.

    ``lock_wheels=True`` is the wheeled-legged curriculum's stage-1 crab-walk variant (wheels
    immobilized so the legs must learn to step, including sideways); the default rolls the
    wheels (the shipped sim2sim-hardened task)."""
    return _velocity_env_cfg_base(play=play, lock_wheels=lock_wheels)
