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
    terrain: "TerrainGeneratorCfg | None" = None,
    extra_sensors: tuple[SensorCfg, ...] = (),
    extra_actor_obs: dict[str, ObservationTermCfg] | None = None,
    extra_critic_obs: dict[str, ObservationTermCfg] | None = None,
    extra_curriculum: dict[str, CurriculumTermCfg] | None = None,
) -> ManagerBasedRlEnvCfg:
    """Shared velocity-tracking skeleton for the Unitree Go2-W."""
    policy_terms = _obs_terms()
    policy_terms.update(extra_actor_obs or {})
    critic_terms = _obs_terms()
    # Privileged critic obs (training-only): true base linear velocity the deployable actor
    # can't sense — gives the value function an accurate read on velocity-tracking error.
    critic_terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.sensor_data, params={"sensor_name": "imu_lin_vel"}
    )
    critic_terms.update(extra_critic_obs or {})

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
        robot=UnitreeGo2WCfg(),
        actions_cfg={
            # Legs: position control (posture offset from default). Wheels: velocity control.
            "joint_pos": JointPositionActionCfg(
                asset_name="robot", joint_names=(_LEG_JOINTS_RE,), use_default_offset=True
            ),
            "wheel_vel": JointVelocityActionCfg(
                asset_name="robot", joint_names=(_WHEEL_JOINTS_RE,), scale=_WHEEL_VEL_SCALE
            ),
        },
        observations_cfg={
            "policy": ObservationGroupCfg(enable_corruption=True, terms=policy_terms),
            "critic": ObservationGroupCfg(enable_corruption=False, terms=critic_terms),
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

    if not play:
        cfg.events_cfg["push_robot"] = EventTermCfg(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            func=mdp.push_by_setting_velocity,
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
        )

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


def unitree_go2w_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-ground velocity-tracking env config for the Unitree Go2-W."""
    return _velocity_env_cfg_base(play=play)
