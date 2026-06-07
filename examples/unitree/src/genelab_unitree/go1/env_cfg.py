"""Velocity-tracking env config for the Unitree Go1 quadruped on flat ground.

Shared skeleton ``_velocity_env_cfg_base`` that both the flat
(``unitree_go1_velocity_env_cfg``) and the complex-terrain
(``rough_env_cfg.unitree_go1_velocity_rough_env_cfg``) factories build on. Keeping
the body here means the two configs can never drift on the rewards / events / DR
they share — the rough task only injects a heightfield terrain, a trunk-mounted
``height_scan`` sensor + observation, and the ``terrain_levels`` curriculum.

The robot cfg (12-DoF, hip/thigh/calf ImplicitPD groups, 4 feet) is registered by
``genelab.asset_zoo.unitree_go1``; this module only composes it into an MDP.
"""

import math
from typing import TYPE_CHECKING

from genelab import mdp
from genelab.asset_zoo.unitree_go1 import UnitreeGo1Cfg
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
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from genelab.mdp.noise import Unoise
from genelab.sensor import (
    BodyVelocitySensorCfg,
    ContactSensorCfg,
    SensorCfg,
)

if TYPE_CHECKING:
    from genelab.terrains import TerrainGeneratorCfg

# Base / trunk link — Go1's floating-base body (go1.xml <body name="trunk">).
_TRUNK_LINK = "trunk"
# Feet. The Menagerie MJCF has no foot *bodies* — each foot is a geom (class "foot") on
# the corresponding *_calf body, so the calf link is what Genesis exposes and what bears
# foot-ground contact. The foot site sits at (0, 0, -0.213) in the calf frame; offsetting
# the reward/contact-evaluation point there keeps feet_slip / clearance measured at the
# contact point (stationary during stance) rather than the calf origin (which swings).
_GO1_FOOT_LINKS: tuple[str, ...] = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
_GO1_FOOT_SITE_OFFSET: tuple[float, float, float] = (0.0, 0.0, -0.213)
# Thigh links (knees). Only the feet should bear load, so contact on the upper legs is
# penalised by the ``undesired_contacts`` safety term. Calves are excluded — the foot
# geom lives on the calf, so calf contact is normal stance, not a fault.
_GO1_BODY_CONTACT_LINKS: tuple[str, ...] = ("FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh")
# Standing trunk height above the ground for the ``base_height`` penalty target
# (Isaac Lab Go1 reference).
_GO1_BASE_HEIGHT_TARGET = 0.34


def _feet_cfg() -> SceneEntityCfg:
    """Selector for the four feet — calf links, evaluated at the foot-site offset."""
    return SceneEntityCfg(
        name="robot",
        link_names=_GO1_FOOT_LINKS,
        link_offsets=(_GO1_FOOT_SITE_OFFSET,) * len(_GO1_FOOT_LINKS),
    )


def _trunk_cfg() -> SceneEntityCfg:
    """Selector pointing at the floating-base trunk link."""
    return SceneEntityCfg(name="robot", link_names=(_TRUNK_LINK,))


def _single_play_spawn_xy(terrain: "TerrainGeneratorCfg | None") -> tuple[float, float]:
    """World (x, y) the lone play robot spawns at — used only to frame the viewer camera.

    Flat ground spawns at the origin. On a height-field it spawns at one terrain cell whose
    column is chosen deterministically from ``terrain.seed`` (mirroring
    ``TerrainImporter.init_per_env_state``), so the camera can be framed before the scene
    is built.
    """
    if terrain is None:
        return 0.0, 0.0
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(terrain.seed + 1)
    col = int(torch.randint(0, terrain.num_cols, (1,), generator=gen, device="cpu").item())
    px, py, _ = terrain.pos
    sx, sy = terrain.subterrain_size
    return px + 0.5 * sx, py + (col + 0.5) * sy


def _obs_terms() -> dict[str, ObservationTermCfg]:
    """Actor / critic-shared proprioceptive observation terms (noise on the actor copy).

    Deliberately excludes base *linear* velocity: real Go1 hardware has no sensor for it,
    so a deployable actor must not depend on it. Velocity tracking is learned from the
    command + proprioception.
    """
    return {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "imu_ang_vel"},
            noise=Unoise(-0.2, 0.2),
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
    """Shared velocity-tracking skeleton for the Unitree Go1 quadruped."""
    policy_terms = _obs_terms()
    policy_terms.update(extra_actor_obs or {})
    critic_terms = _obs_terms()
    # Privileged critic obs (training-only, never deployed): the true base linear velocity
    # the deployable actor can't sense. An asymmetric actor-critic — the value function gets
    # an accurate read on velocity-tracking error, so the proprioception-only actor can still
    # learn to translate to the commanded speed instead of stepping in place.
    critic_terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.sensor_data, params={"sensor_name": "imu_lin_vel"}
    )
    critic_terms.update(extra_critic_obs or {})

    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4096 if not play else 1,
            dt=0.002,
            substeps=1,
            vis=play,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.5, 2.5),
            terrain=terrain,
            sensors=(
                BodyVelocitySensorCfg(
                    name="imu_lin_vel",
                    link_name=_TRUNK_LINK,
                    measure="lin_vel",
                ),
                BodyVelocitySensorCfg(
                    name="imu_ang_vel",
                    link_name=_TRUNK_LINK,
                    measure="ang_vel",
                ),
                ContactSensorCfg(
                    name="feet_ground_contact",
                    link_names=_GO1_FOOT_LINKS,
                    track_air_time=True,
                ),
                # Thigh / calf contact for the undesired-contacts safety penalty.
                ContactSensorCfg(
                    name="body_contact",
                    link_names=_GO1_BODY_CONTACT_LINKS,
                ),
                *extra_sensors,
            ),
        ),
        decimation=10,
        episode_length_s=20.0,
        device="cuda",
        robot=UnitreeGo1Cfg(),
        actions_cfg={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(".*",),
                use_default_offset=True,
            )
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
                # Lowered from 0.2: the strict-forward group over-represented forward
                # commands, compounding the quadruped's natural forward bias so the policy
                # never learned to walk backward. 0.1 keeps some forward emphasis while
                # letting the free group sample vx symmetrically in [-1, 1].
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
                # Tracking is weighted heavily (2.0 / 1.0) relative to the gait-shaping
                # terms. An earlier 1.0 / 0.5 with air_time=0.5 let the policy farm the
                # air-time reward by stepping in place (achieving ~0.05 m/s against a
                # ~0.65 m/s command); boosting the tracking gradient and trimming air_time
                # makes actually translating the dominant way to earn reward.
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            "track_ang_vel": RewardTermCfg(
                func=mdp.track_angular_velocity_z_exp,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            "air_time": RewardTermCfg(
                func=mdp.feet_air_time,
                weight=0.25,
                params={
                    "sensor_name": "feet_ground_contact",
                    "threshold_min": 0.1,
                    "threshold_max": 0.5,
                    "command_name": "twist",
                    "command_threshold": 0.1,
                },
            ),
            # --- base-stability penalties (Isaac Lab quadruped set) ---
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
                params={"target_height": _GO1_BASE_HEIGHT_TARGET},
            ),
            # --- actuation / smoothness penalties ---
            "joint_torques": RewardTermCfg(func=mdp.applied_torque_l2, weight=-2.0e-4),
            # NOTE: not Isaac Lab's -2.5e-7. That value is tuned for its *physics-step*
            # acceleration (Δv / ~0.005s, large spiky values); GeneLab's joint_acc is a
            # *control-step* finite difference (Δv / step_dt = 0.02s, smoothed over the
            # decimation window), so the raw magnitude is ~100x smaller and -2.5e-7 left
            # the term inert (~360x below the tracking reward). -1.0e-5 scale-matches it to
            # the sibling smoothness penalties (≈ action_rate's weighted ~0.02/step). A
            # starting point, not a training-tuned value — sweep it against gait quality.
            "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-5),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
            "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            # --- safety ---
            "undesired_contacts": RewardTermCfg(
                func=mdp.contact_force_limit,
                weight=-1.0,
                params={"sensor_name": "body_contact", "max_force": 1.0},
            ),
            # --- gait shaping ---
            "foot_slip": RewardTermCfg(
                func=mdp.feet_slip,
                weight=-0.1,
                params={
                    "sensor_name": "feet_ground_contact",
                    "asset_cfg": _feet_cfg(),
                    "command_name": "twist",
                    "command_threshold": 0.05,
                },
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
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "yaw": (-math.pi, math.pi),
                    },
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
                        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
                    ],
                },
            ),
            **(extra_curriculum or {}),
        },
    )

    if not play:
        # Runtime DR: random pushes test the policy's recovery on complex terrain.
        cfg.events_cfg["push_robot"] = EventTermCfg(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            func=mdp.push_by_setting_velocity,
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        )

    if play:
        # Interactive teleop: don't snap the lone robot back to spawn on a stumble or
        # time-out — let it keep walking under slider control until the viewer closes.
        cfg.auto_reset = False

        # Frame the viewer on the single play robot (Go1 trunk sits ~0.34 m up, so aim
        # lower and closer than the G1 humanoid). ``camera_lookat`` is also the trackball
        # pivot, so aiming it at the robot is what lets the mouse-wheel zoom in on it.
        look_x, look_y = _single_play_spawn_xy(terrain)
        cfg.simulation.camera_lookat = (look_x, look_y, 0.3)
        cfg.simulation.camera_pos = (look_x + 2.0, look_y - 2.0, 1.0)

        # Auto-attach the in-viewport ImGui teleop bridge: three sliders (vx, vy, ωz) that
        # drive the single robot's ``twist`` command. Guarded on imgui_bundle (the 'imgui'
        # extra) so a user without it still gets a working play path — just no sliders.
        try:
            import imgui_bundle  # noqa: F401  # the ImGui overlay needs it at draw time

            from genelab.bridges.imgui import ImGuiTwistBridgeCfg

            cfg.simulation.viewer_imgui = True  # enable the overlay that hosts the sliders
            cfg.bridges_cfg["teleop"] = ImGuiTwistBridgeCfg(
                command_name="twist",
                # Slider ranges bracket the training command ranges (vx/vy/ωz ∈ [-1, 1]).
                vx_range=(-1.5, 1.5),
                vy_range=(-1.0, 1.0),
                wz_range=(-1.0, 1.0),
                # Start in the forward-walking slice so iter-0 is in-distribution; drag vx
                # to 0 to test standing.
                default_vx=0.5,
                default_vy=0.0,
                default_wz=0.0,
            )
        except ImportError:
            pass

    return cfg


def unitree_go1_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-ground velocity-tracking env config for the Unitree Go1."""
    return _velocity_env_cfg_base(play=play)
