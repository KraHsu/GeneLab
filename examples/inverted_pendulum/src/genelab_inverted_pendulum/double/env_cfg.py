"""Flat-ground double inverted-pendulum env config (manager-based + Genesis)."""

from typing import TYPE_CHECKING

from genelab import mdp
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.noise import Unoise
from genelab.recording import PyQtPlotCfg, RecordingCfg
from genelab.sensor import BodyVelocitySensorCfg

from genelab_inverted_pendulum import mdp as ip_mdp
from genelab_inverted_pendulum.double.constants import (
    CART_JOINT,
    CART_POSITION_LIMIT,
    POLE_1_ANGLE_LIMIT,
    POLE_1_JOINT,
    POLE_2_ANGLE_LIMIT,
    POLE_2_JOINT,
    POLE_2_LINK,
    POLE_HINGE_JOINTS,
)
from genelab_inverted_pendulum.double.robot import get_double_inverted_pendulum_robot_cfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _poles_state(env: "ManagerBasedRlEnv") -> dict[str, object]:
    """Per-tick (angle, angular velocity) of both pole joints for env 0.

    Joint convention: each pole's joint angle is 0 when that joint is at its
    "upright" rest pose — the same zero used by ``double_pole_upright``. Note
    that pole_2's angle is measured relative to pole_1's axis, not the world
    vertical: if pole_1 is tilted by α and pole_2's joint reads 0, pole_2 is
    also tilted by α in world frame.
    """
    i1 = env.joint_names.index(POLE_1_JOINT)
    i2 = env.joint_names.index(POLE_2_JOINT)
    return {
        "angle (rad)": env.robot_state.joint_pos[0, [i1, i2]],
        "ang_vel (rad/s)": env.robot_state.joint_vel[0, [i1, i2]],
    }


def _obs_terms() -> dict[str, ObservationTermCfg]:
    return {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(-0.005, 0.005),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.1,
            noise=Unoise(-0.05, 0.05),
        ),
        "pole_tip_ang_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "pole_tip_ang_vel"},
            scale=0.2,
            noise=Unoise(-0.1, 0.1),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }


def double_inverted_pendulum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-ground double-inverted-pendulum env config."""
    robot_entity_cfg = get_double_inverted_pendulum_robot_cfg().to_entity_cfg()

    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4096 if not play else 1,
            dt=0.005,
            substeps=1,
            vis=play,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(3.0, 3.0),
            mouse_interaction=play,
            sensors=(
                BodyVelocitySensorCfg(
                    name="pole_tip_ang_vel",
                    link_name=POLE_2_LINK,
                    measure="ang_vel",
                ),
            ),
            recordings=(
                (
                    RecordingCfg(
                        name="poles_state",
                        source=_poles_state,
                        env_idx=None,  # callable already returns a dict (no leading env dim)
                        outputs=(
                            PyQtPlotCfg(
                                title="Double inverted pendulum — pole state",
                                labels={
                                    "angle (rad)": ("pole_1", "pole_2"),
                                    "ang_vel (rad/s)": ("pole_1", "pole_2"),
                                },
                                history_length=400,
                            ),
                        ),
                    ),
                )
                if play
                else ()
            ),
        ),
        decimation=2,
        episode_length_s=12.0,
        device="cuda",
        robot=robot_entity_cfg,
        actions_cfg={
            "cart": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(CART_JOINT,),
                use_default_offset=True,
            ),
        },
        observations_cfg={
            "policy": ObservationGroupCfg(enable_corruption=True, terms=_obs_terms()),
            "critic": ObservationGroupCfg(enable_corruption=False, terms=_obs_terms()),
        },
        rewards_cfg={
            "alive": RewardTermCfg(func=ip_mdp.alive_bonus, weight=1.0),
            "poles_upright": RewardTermCfg(
                func=ip_mdp.double_pole_upright,
                weight=4.0,
                params={"joint_names": POLE_HINGE_JOINTS},
            ),
            "poles_aligned": RewardTermCfg(
                func=ip_mdp.double_pole_alignment,
                weight=-0.5,
                params={"joint_names": POLE_HINGE_JOINTS},
            ),
            "cart_position": RewardTermCfg(func=ip_mdp.cart_position_l2, weight=-0.05),
            "cart_velocity": RewardTermCfg(func=ip_mdp.cart_velocity_l2, weight=-0.005),
            "poles_velocity": RewardTermCfg(
                func=ip_mdp.double_pole_velocity_l2,
                weight=-0.005,
                params={"joint_names": POLE_HINGE_JOINTS},
            ),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "poles_fell": TerminationTermCfg(
                func=ip_mdp.any_pole_angle_exceeds,
                params={
                    "limits": {
                        POLE_1_JOINT: POLE_1_ANGLE_LIMIT,
                        POLE_2_JOINT: POLE_2_ANGLE_LIMIT,
                    },
                },
            ),
            "cart_out": TerminationTermCfg(
                func=ip_mdp.cart_position_exceeds,
                params={"limit": CART_POSITION_LIMIT, "joint_name": CART_JOINT},
            ),
        },
        events_cfg={
            "reset_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_to_default,
                params={"pos_jitter": 0.05, "vel_jitter": 0.05},
            ),
        },
    )
    if not play:
        cfg.events_cfg["push_cart"] = EventTermCfg(
            mode="interval",
            interval_range_s=(3.0, 6.0),
            func=ip_mdp.push_cart_by_setting_joint_velocity,
            params={"velocity_range": (-0.3, 0.3), "joint_name": CART_JOINT},
        )
    return cfg
