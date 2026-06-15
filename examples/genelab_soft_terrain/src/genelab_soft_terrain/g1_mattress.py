"""Trainable Unitree G1 velocity tracking on a pneumatic air mattress.

The humanoid counterpart of the Go1 sand task: instead of a granular medium, the feet
stand on **one sealed air chamber** (:func:`genelab.terrains.pneumatic_normal_force` as
the terrain's ``normal_law``). The support under each foot is the shared chamber
pressure — an instantaneous function of the *total* volume both feet displace — so
weight transfer onto one foot softens the support under the other. That cross-foot
coupling (plus the progressive stiffening as the chamber fills) is what makes mattress
locomotion a different balance problem from a per-foot spring surface.

As in the rest of the soft-terrain suite there is no rigid floor under the feet
(``ground_plane_collision=False``): the analytic chamber force is the only support, so
Genesis contact sensors are blind and the reward set is built from non-contact terms
plus the terrain-derived gait shaping. The mattress box in the play scene is render-only.
"""

import math
from functools import partial

from genelab import mdp
from genelab.asset_zoo.unitree_g1 import UnitreeG1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from genelab.materials import SurfaceCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from genelab.mdp.noise import Unoise
from genelab.sensor import BodyVelocitySensorCfg
from genelab.terrains import DeformableTerrainCfg, pneumatic_normal_force
from genelab_soft_terrain.go1_soft_velocity_env_cfg import track_velocity_l1

_TORSO_LINK = "torso_link"
_PELVIS_LINK = "pelvis"
# IMU site offset from the pelvis link origin (g1.xml's <site name="imu_in_pelvis">).
_IMU_OFFSET = (0.04525, 0.0, -0.08339)
G1_FOOT_LINKS: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
# Foot-site offset from each ankle_roll_link origin (g1.xml's left_foot/right_foot site):
# support + traction act at the sole, ~0.037 m below the ankle link origin.
G1_FOOT_OFFSET = (0.04, 0.0, -0.037)

# Mattress geometry / chamber model (same family as the air_mattress demo, scaled up to
# carry the ~35 kg humanoid instead of three 0.1 m balls).
_TOP = 0.4  # undisturbed chamber top surface, m
_CAPACITY = 0.35  # total displaced depth at chamber bottom-out, m
# N per m of total displaced depth at small fill. At ~340 N robot weight the two feet
# settle ~8.5 cm in (headroom ~0.76): a firm camping mattress, well clear of bottom-out.
_PRESSURE_STIFFNESS = 3000.0
_DAMPING = 150.0
_FRICTION = 0.8  # rubberized fabric: grippy
_LATERAL_STIFFNESS = 3000.0  # planted-foot anchor (stick-slip), as in the soft suite
# Pelvis sits 0.76 m above the feet in the home pose; equilibrium sinkage ~8.5 cm.
_BASE_HEIGHT_TARGET = _TOP - 0.085 + 0.76

# Per-joint posture tolerance for the ``pose`` reward (std of the walking band, copied
# from the G1 flat-ground example so mattress gaits stay human-shaped, not flailing).
_G1_POSE_STD: dict[str, float] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
}

_MATTRESS = DeformableTerrainCfg(
    k=_PRESSURE_STIFFNESS,
    c=_DAMPING,
    mu=_FRICTION,
    lateral_stiffness=_LATERAL_STIFFNESS,
    surface_height=_TOP,
    normal_law=partial(pneumatic_normal_force, capacity=_CAPACITY),
    foot_link_names=G1_FOOT_LINKS,
    foot_offset=G1_FOOT_OFFSET,
)


def _torso_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(name="robot", link_names=(_TORSO_LINK,))


def single_support(env, command_name: str, command_threshold: float = 0.1):  # type: ignore[no-untyped-def]
    """Reward exactly-one-foot chamber contact while commanded to move.

    Standing earns the posture/upright/tracking floor (~0.27 per step) and can never
    earn this term, so it is the gradient toward *lifting a foot* — the move three
    training runs never discovered on their own (``feet_air_time`` pinned at ~0; the
    chamber's full pressure acts on any foot with depth > 0, so a foot only unloads
    once it clears the undisturbed surface, ~8.5 cm above the loaded stance level).
    """
    import torch

    driver = env.deformable_terrain
    assert driver is not None, "single_support requires cfg.deformable_terrain"
    contacts = (driver.terrain.state.depth > 0.0).sum(dim=-1)
    cmd = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd[:, :2], dim=-1) > command_threshold
    return (contacts == 1).float() * moving.float()


def _proprio_obs() -> dict[str, ObservationTermCfg]:
    """Deployable proprioception (no base linear velocity, as in the rest of the suite)."""
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


def g1_mattress_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Trainable velocity-tracking env for the G1 on the pneumatic mattress."""
    robot = UnitreeG1Cfg()
    bx, by, bz = robot.init_pos  # bz = 0.76: pelvis height with the feet at z=0
    robot.init_pos = (bx, by, _TOP + bz + 0.04)  # feet drop ~4 cm onto the chamber surface

    policy_terms = _proprio_obs()
    critic_terms = _proprio_obs()
    # Privileged critic obs: true base linear velocity + per-foot chamber sinkage.
    critic_terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.sensor_data, params={"sensor_name": "imu_lin_vel"}
    )
    critic_terms["terrain_sinkage"] = ObservationTermCfg(func=mdp.terrain_sinkage)

    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4096 if not play else 1,
            # G1 actuator stiffness needs the finer solver step (as in the flat-ground G1
            # example); control stays at 50 Hz via the deeper decimation.
            dt=0.002,
            substeps=1,
            vis=play,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.5, 2.5),
            terrain=None,
            ground_plane_collision=False,
            ground_plane_height=0.0,
            sensors=(
                BodyVelocitySensorCfg(
                    name="imu_lin_vel",
                    link_name=_PELVIS_LINK,
                    offset=_IMU_OFFSET,
                    measure="lin_vel",
                ),
                BodyVelocitySensorCfg(
                    name="imu_ang_vel",
                    link_name=_PELVIS_LINK,
                    offset=_IMU_OFFSET,
                    measure="ang_vel",
                ),
            ),
        ),
        decimation=10,
        episode_length_s=20.0,
        device="cuda",
        robot=robot,
        deformable_terrain=_MATTRESS,
        actions_cfg={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot", joint_names=(".*",), use_default_offset=True
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
                # 10% standing envs: balancing in place on the chamber is itself a skill
                # worth training (the mattress never stops moving under the feet).
                rel_standing_envs=0.1,
                rel_heading_envs=0.0,
                rel_forward_envs=0.2,
                heading_command=False,
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.8, 0.8),
                    lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-1.0, 1.0),
                    heading=(-math.pi, math.pi),
                ),
            )
        },
        rewards_cfg={
            # --- velocity tracking ---
            "track_lin_vel": RewardTermCfg(
                func=mdp.track_linear_velocity_xy_exp,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            # No-floor movement penalty (see the soft-terrain suite): standing while
            # commanded to move costs ~-|cmd| per step, so the policy must translate.
            "track_lin_vel_l1": RewardTermCfg(
                func=track_velocity_l1, weight=1.5, params={"command_name": "twist"}
            ),
            "track_ang_vel": RewardTermCfg(
                func=mdp.track_angular_velocity_z_exp,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            # --- humanoid posture ---
            "upright": RewardTermCfg(
                func=mdp.upright_exp,
                weight=1.0,
                params={"std": math.sqrt(0.2), "asset_cfg": _torso_cfg()},
            ),
            "pose": RewardTermCfg(
                func=mdp.variable_posture,
                weight=1.0,
                params={
                    "command_name": "twist",
                    "walking_threshold": 0.05,
                    "running_threshold": 1.5,
                    "default_std": 0.3,
                    "std_standing": {".*": 0.05},
                    "std_walking": _G1_POSE_STD,
                    "std_running": _G1_POSE_STD,  # command range never reaches running
                },
            ),
            # --- base stability ---
            # Light: some vertical motion is intrinsic to standing on an air chamber.
            "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.5),
            "ang_vel_xy": RewardTermCfg(
                func=mdp.body_angular_velocity_penalty,
                weight=-0.05,
                params={"asset_cfg": _torso_cfg()},
            ),
            "base_height": RewardTermCfg(
                func=mdp.base_height_l2, weight=-1.0, params={"target_height": _BASE_HEIGHT_TARGET}
            ),
            # --- actuation / smoothness ---
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
            "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            # --- gait shaping (terrain-derived contact: the chamber, not a contact
            #     sensor, knows when a foot is planted) ---
            "single_support": RewardTermCfg(
                func=single_support,
                weight=0.5,
                params={"command_name": "twist", "command_threshold": 0.1},
            ),
            "feet_air_time": RewardTermCfg(
                func=mdp.terrain_feet_air_time,
                weight=2.0,
                params={
                    "command_name": "twist",
                    "threshold_min": 0.0,
                    "threshold_max": 0.5,
                    "command_threshold": 0.1,
                },
            ),
            "feet_slip": RewardTermCfg(
                func=mdp.terrain_feet_slip,
                weight=-0.1,
                params={"command_name": "twist", "command_threshold": 0.1},
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
                    "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi, math.pi)},
                    "velocity_range": {},
                },
            ),
            "reset_robot_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_by_offset,
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
    )

    if not play:
        # Interval pushes (the proven G1 flat-ground recipe carries the same event):
        # recovering from a shove is what forces the policy to discover protective
        # *steps*. Without it, mattress training settles into a stand-then-tip local
        # optimum with feet_air_time ~ 0 — observed twice, at entropy_coef 0.005 and
        # 0.01 (plateau at ~97-step episodes, action std collapsing).
        cfg.events_cfg["push_robot"] = EventTermCfg(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            func=mdp.push_by_setting_velocity,
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        )
        # Domain-randomize the chamber per env on every reset: firmer / softer inflation
        # and slicker / grippier fabric, bracketing the calibrated point. (``capacity``
        # is bound inside the normal law, not a per-env tensor, so it stays fixed.)
        # Bootstrap note (the sand curriculum lesson): train the gait at the calibrated
        # point first — pop this event and warm-start the DR run from that checkpoint.
        cfg.events_cfg["randomize_chamber"] = EventTermCfg(
            mode="reset",
            func=mdp.randomize_terrain_params,
            params={"ranges": {"k": (1500.0, 6000.0), "mu": (0.5, 1.0)}},
        )

    if play:
        # Render-only mattress under the lone play robot (training never renders it; a
        # per-env 4 m box would also overlap the 2.5 m env grid visually).
        cfg.scene.entities = {
            "mattress": RigidObjectCfg(
                morph="box",
                size=(4.0, 4.0, _TOP),
                init_pos=(0.0, 0.0, _TOP / 2),
                fixed=True,
                collision=False,  # the chamber model, not the box, is the support
                surface=SurfaceCfg(color=(0.55, 0.65, 0.95)),
            )
        }
        cfg.auto_reset = False
        cfg.simulation.camera_lookat = (0.0, 0.0, 0.9)
        cfg.simulation.camera_pos = (2.4, -2.4, 1.6)
        from importlib.util import find_spec

        if find_spec("imgui_bundle") is not None:  # the in-viewport overlay needs it
            from genelab_soft_terrain.teleop import ImGuiTwistResetBridgeCfg

            cfg.simulation.viewer_imgui = True
            cfg.bridges_cfg["teleop"] = ImGuiTwistResetBridgeCfg(
                command_name="twist",
                vx_range=(-0.8, 0.8),
                vy_range=(-0.5, 0.5),
                wz_range=(-1.0, 1.0),
                default_vx=0.4,  # start walking forward (in-distribution); drag to 0 to stand
                default_vy=0.0,
                default_wz=0.0,
            )

    return cfg
