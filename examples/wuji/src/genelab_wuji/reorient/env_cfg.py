"""Reorient environment assembler: WUJI hand + cube + SO(3) goal task.

``wuji_hand_reorient_env_cfg(play)`` wires the manager term groups into a single
``ManagerBasedRlEnvCfg``. The hand is fixed-base palm-up; a free cube starts above the
fingertip cage and must be reoriented to a stream of SO(3) goals without dropping it.
"""

import math

from genelab import mdp
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import ArticulationCfg, RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp import Gnoise, Unoise
from genelab.sensor import SelfContactSensorCfg

from genelab_wuji.reorient.asset import resolve_cube_mesh, wuji_hand_reorient_cfg
from genelab_wuji.reorient.constants import (
    GOAL_MARKER_POS,
    REORIENT_CUBE_HALF_EXTENT,
    REORIENT_CUBE_INIT_POS,
)
from genelab_wuji.reorient.mdp import cage, curriculums, events, metrics, observations, rewards
from genelab_wuji.reorient.mdp.actions import JointPositionOffsetEMAActionCfg
from genelab_wuji.reorient.mdp.commands import InHandReorientCommandCfg
from genelab_wuji.reorient.mdp.sensors import HandObjectContactSensorCfg

_JOINTS = (r"right_finger[1-5]_joint[1-4]",)
_FAR_RESAMPLE = (1.0e6, 1.0e6)  # command runs its own state machine; never auto-resample


def _policy_obs(play: bool) -> ObservationGroupCfg:
    # 3-step observation history (term-major) matches mjlab's actor obs. Heavier observation
    # noise (training only) is a robustness lever: it stops the policy from relying on precise
    # state estimates, part of why the learned feedback gait overfits to one engine's exact
    # contact response. Eval (play) runs nominal/clean.
    return ObservationGroupCfg(
        enable_corruption=not play,
        terms={
            "joint_pos": ObservationTermCfg(
                func=observations.joint_pos_rel_history, noise=Unoise(n_min=-0.10, n_max=0.10)
            ),
            "joint_vel": ObservationTermCfg(
                func=observations.joint_vel_rel_history, noise=Unoise(n_min=-0.5, n_max=0.5)
            ),
            "cube_pos_in_tag": ObservationTermCfg(
                func=observations.cube_pos_in_tag_history,
                noise=Unoise(n_min=-0.02, n_max=0.02),
            ),
            "goal_rot_err_6d": ObservationTermCfg(
                func=observations.goal_rot_err_6d_history,
                params={"command_name": "reorient_command"},
                noise=Gnoise(std=0.08),
            ),
            "last_action": ObservationTermCfg(func=observations.last_action_history),
        },
    )


def _critic_obs() -> ObservationGroupCfg:
    return ObservationGroupCfg(
        enable_corruption=False,
        terms={
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "cube_pos_in_tag": ObservationTermCfg(func=observations.cube_pos_in_tag),
            "goal_rot_err_6d": ObservationTermCfg(
                func=observations.goal_rot_err_6d, params={"command_name": "reorient_command"}
            ),
            "last_action": ObservationTermCfg(func=mdp.last_action),
            "state_progress": ObservationTermCfg(
                func=observations.command_state_progress,
                params={"command_name": "reorient_command"},
            ),
            "cage_counter": ObservationTermCfg(
                func=observations.cage_counter_progress, params={"max_outside_steps": 10}
            ),
        },
    )


def _scene_entities(play: bool, size: float) -> dict[str, ArticulationCfg | RigidObjectCfg]:
    """Scene objects: a plain box for training (clean collision); for play, a UV-textured
    cube (visible faces) plus a textured goal-marker the command poses at the target."""
    if not play:
        return {
            "object": RigidObjectCfg(
                morph="box",
                size=(size, size, size),
                init_pos=REORIENT_CUBE_INIT_POS,
                init_quat=(1.0, 0.0, 0.0, 0.0),
                fixed=False,
                friction=1.0,
                density=580.0,
            )
        }
    cube_mesh = resolve_cube_mesh()
    return {
        "object": RigidObjectCfg(
            morph="mesh",
            file=cube_mesh,
            init_pos=REORIENT_CUBE_INIT_POS,
            init_quat=(1.0, 0.0, 0.0, 0.0),
            fixed=False,
            friction=1.0,
            density=580.0,
        ),
        "goal_marker": RigidObjectCfg(
            morph="mesh",
            file=cube_mesh,
            init_pos=GOAL_MARKER_POS,
            init_quat=(1.0, 0.0, 0.0, 0.0),
            fixed=False,
            friction=1.0,
            density=580.0,
        ),
    }


def wuji_hand_reorient_env_cfg(play: bool = False, num_envs: int = 8192) -> ManagerBasedRlEnvCfg:
    """Build the WUJI-hand SO(3) reorientation task config."""
    size = 2.0 * REORIENT_CUBE_HALF_EXTENT
    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4 if play else num_envs,
            dt=0.01,
            substeps=1,
            vis=play,
            gpu=not play,
            # Per-env model params so dof/link DR (PD gains, frictionloss, mass/inertia)
            # actually applies per environment. Genesis defaults these off, which silently
            # no-ops per-env dof DR; training-only (eval/play uses nominal params).
            batch_dofs_info=not play,
            batch_links_info=not play,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(0.75, 0.75),
            entities=_scene_entities(play, size),
            sensors=(
                HandObjectContactSensorCfg(
                    name="hand_object_contact", object_name="object", force_threshold=0.05
                ),
                SelfContactSensorCfg(name="finger_collision", force_threshold=1.0),
            ),
        ),
        decimation=5,
        episode_length_s=50.0 if not play else 60.0,
        device="cpu" if play else "cuda",
        robot=wuji_hand_reorient_cfg(),
        actions_cfg={
            "joint_pos": JointPositionOffsetEMAActionCfg(
                asset_name="robot",
                joint_names=_JOINTS,
                scale=0.5,
                use_default_offset=True,
                ema_alpha=0.5,
                warmup_time_s=0.4,
                action_noise_std=0.0 if play else 0.1,
            )
        },
        observations_cfg={"policy": _policy_obs(play), "critic": _critic_obs()},
        commands_cfg={
            "reorient_command": InHandReorientCommandCfg(
                asset_name="robot",
                resampling_time_range=_FAR_RESAMPLE,
                # Training starts loose and the success curriculum tightens it toward 0.2;
                # play/eval uses the tight target threshold directly (real success criterion).
                success_threshold=0.2 if play else 0.8,
                success_hold_steps=5,
                goal_switch_delay=20,
                debug_vis=play,
            )
        },
        rewards_cfg={
            "orientation_alignment": RewardTermCfg(
                func=rewards.orientation_alignment,
                weight=15.0,
                params={"command_name": "reorient_command", "bound": 0.2, "margin": math.pi},
            ),
            "hold_escalation": RewardTermCfg(
                func=rewards.hold_escalation,
                weight=11.4,
                params={"command_name": "reorient_command"},
            ),
            "cage_escape": RewardTermCfg(
                func=cage.cage_escape_penalty, weight=-500.0, params={"margin": 0.01}
            ),
            "hand_pose": RewardTermCfg(func=rewards.hand_pose_penalty, weight=-0.2),
            "action_rate": RewardTermCfg(func=rewards.action_rate_combined, weight=-1.0),
            "torque": RewardTermCfg(func=rewards.torque_penalty, weight=-24.0),
            "tip_slide": RewardTermCfg(
                func=rewards.tip_slide_penalty,
                weight=-0.3,
                params={"sensor_name": "hand_object_contact"},
            ),
            "palm_detach": RewardTermCfg(
                func=rewards.palm_detach_reward,
                weight=0.5,
                params={"sensor_name": "hand_object_contact"},
            ),
            "finger_collision": RewardTermCfg(
                func=rewards.finger_self_collision_penalty,
                weight=-1.0,
                params={"sensor_name": "finger_collision"},
            ),
        },
        metrics_cfg={
            "fingertip_contact_count": MetricsTermCfg(
                func=metrics.fingertip_contact_count,
                params={"sensor_name": "hand_object_contact"},
            ),
            "finger_collision_frequency": MetricsTermCfg(
                func=metrics.finger_collision_frequency,
                params={"sensor_name": "finger_collision"},
            ),
            "cube_height_above_palm": MetricsTermCfg(func=metrics.cube_height_above_palm),
            "goals_reached": MetricsTermCfg(
                func=metrics.success_rate, params={"command_name": "reorient_command"}
            ),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "cage_drop": TerminationTermCfg(func=cage.cage_drop, params={"max_outside_steps": 10}),
        },
        curriculum_cfg=_curriculum_cfg(play),
        events_cfg=_events_cfg(play),
    )
    return cfg


def _curriculum_cfg(play: bool) -> dict[str, CurriculumTermCfg]:
    """Training-only adaptive schedules: success-tolerance tightening + disturbance ramp."""
    if play:
        return {}
    return {
        "success_curriculum": CurriculumTermCfg(
            func=curriculums.reorient_success_curriculum,
            params={
                "command_name": "reorient_command",
                "count_threshold": 3,
                "delta_per_loop": 0.08,
                "threshold_start": 0.8,
                "threshold_end": 0.2,
            },
        ),
        "adaptive_episode": CurriculumTermCfg(
            func=curriculums.adaptive_episode_curriculum,
            params={"target_steps": 800},
        ),
    }


def _events_cfg(play: bool) -> dict[str, EventTermCfg]:
    """Reset events + (training-only) robustness domain randomization & disturbance.

    The sim2sim transfer gap is a closed-loop feedback overfit (the policy learns a gait
    tuned to Genesis's exact contact response), so the DR here targets *robustness of the
    feedback loop* rather than matching any single contact parameter: per-env friction (hand
    + cube), mass (hand links + cube), COM, PD gains, encoder bias, plus a frequent linear +
    angular cube velocity disturbance. Action noise (in the action term) and heavier
    observation noise (policy obs group) complete the recipe. All disabled in play (eval).
    """
    robot = SceneEntityCfg("robot")
    cfg: dict[str, EventTermCfg] = {
        "reset_object": EventTermCfg(
            func=events.reset_object_orientation, mode="reset", params={"pos_noise": 0.01}
        ),
        "reset_joints": EventTermCfg(func=mdp.reset_joints_to_default, mode="reset"),
        "reset_cage": EventTermCfg(func=events.reset_cage_state, mode="reset"),
    }
    if play:
        return cfg
    cfg.update(
        {
            "robot_friction": EventTermCfg(
                func=mdp.dr.geom_friction,
                mode="startup",
                params={"asset_cfg": robot, "ranges": (0.5, 1.5)},
            ),
            "cube_physics": EventTermCfg(
                func=events.randomize_cube_physics,
                mode="startup",
                params={
                    "object_name": "object",
                    "friction_range": (0.5, 1.5),
                    "mass_shift_range": (-0.02, 0.02),
                },
            ),
            "robot_mass": EventTermCfg(
                func=mdp.dr.body_mass_offset,
                mode="startup",
                params={"asset_cfg": robot, "ranges": (-0.003, 0.003)},
            ),
            "robot_com": EventTermCfg(
                func=mdp.dr.body_com_offset,
                mode="startup",
                params={
                    "asset_cfg": robot,
                    "ranges": {0: (-0.003, 0.003), 1: (-0.003, 0.003), 2: (-0.003, 0.003)},
                },
            ),
            "pd_gains": EventTermCfg(
                func=mdp.dr.randomize_joint_stiffness_damping,
                mode="startup",
                params={
                    "asset_cfg": robot,
                    "stiffness_range": (0.75, 1.5),
                    "damping_range": (0.5, 2.0),
                },
            ),
            "encoder_bias": EventTermCfg(
                func=mdp.dr.encoder_bias,
                mode="startup",
                params={"asset_cfg": robot, "bias_range": (-0.01, 0.01)},
            ),
            # Per-env joint dry-friction (stiction): the MJCF has none, so the policy
            # never learns to overcome the real hand's static friction and reorients too
            # slowly on hardware. Real per-env DR now that batch_dofs_info is on (the
            # earlier global-baseline attempt hurt; this samples per env/joint).
            "dof_frictionloss": EventTermCfg(
                func=mdp.dr.dof_frictionloss,
                mode="startup",
                params={"asset_cfg": robot, "friction_range": (0.0, 0.03)},
            ),
            # Per-env joint armature (rotor inertia) scale — robustness to inertia
            # calibration error (mjlab parity). Now per-env via batch_dofs_info.
            "dof_armature": EventTermCfg(
                func=mdp.dr.dof_armature,
                mode="startup",
                params={"asset_cfg": robot, "scale_range": (0.75, 1.3)},
            ),
            "object_disturbance": EventTermCfg(
                func=events.apply_velocity_disturbance,
                mode="interval",
                interval_range_s=(0.4, 1.2),
                params={
                    "min_speed": 0.05,
                    "max_speed": 0.15,
                    "min_angular": 0.5,
                    "max_angular": 2.0,
                },
            ),
        }
    )
    return cfg
