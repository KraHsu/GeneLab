"""Manager-based RL env config for the Franka pick-and-place task.

Two action surfaces are exposed:

* :func:`franka_pick_and_place_env_cfg` — original 9-DoF joint-position control
  (arm 7 + fingers 2) used by ``GeneLab-Franka-Pick-And-Place-v0``.
* :func:`franka_pick_and_place_cartesian_env_cfg` — 4-DoF Cartesian control
  ``(dx, dy, dz, gripper)`` via :class:`DifferentialIKAction` +
  :class:`BinaryGripperAction`, the panda-gym ``PandaPickAndPlace`` action
  space, used by ``GeneLab-Franka-Pick-And-Place-Cartesian-v0``.
"""

from genelab import mdp
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.binary_gripper import BinaryGripperActionCfg
from genelab.mdp.actions.ee_delta_ik import DifferentialIKActionCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.noise import Unoise

from genelab_franka_pick_and_place import mdp as fpp_mdp
from genelab_franka_pick_and_place.constants import (
    ARM_JOINTS_REGEX,
    CUBE_ENTITY_NAME,
    CUBE_HALF,
    CUBE_NOMINAL_XY,
    CUBE_SIZE,
    EE_LINK,
    FINGER_JOINTS_REGEX,
)
from genelab_franka_pick_and_place.robot import get_franka_pick_and_place_robot_cfg


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
        "ee_pos": ObservationTermCfg(
            func=fpp_mdp.ee_pos_obs,
            noise=Unoise(-0.002, 0.002),
        ),
        "cube_pos": ObservationTermCfg(
            func=fpp_mdp.cube_pos_obs,
            noise=Unoise(-0.002, 0.002),
        ),
        "goal_pos": ObservationTermCfg(func=fpp_mdp.goal_pos_obs),
        "cube_to_goal": ObservationTermCfg(func=fpp_mdp.cube_to_goal_vec_obs),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }


def _build_env_cfg(
    *,
    actions_cfg: dict[str, ActionTermCfg],
    requires_jac_and_ik: bool,
    play: bool,
) -> ManagerBasedRlEnvCfg:
    """Shared body for the joint-position and Cartesian variants — only the
    ``actions_cfg`` dict and the morph's IK flag differ between them."""
    robot_entity_cfg = get_franka_pick_and_place_robot_cfg(
        requires_jac_and_ik=requires_jac_and_ik,
    ).to_entity_cfg()

    cube_cfg = RigidObjectCfg(
        morph="box",
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        init_pos=(CUBE_NOMINAL_XY[0], CUBE_NOMINAL_XY[1], CUBE_HALF),
        init_quat=(1.0, 0.0, 0.0, 0.0),
        fixed=False,
    )

    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=2048 if not play else 1,
            dt=0.01,
            substeps=2,
            vis=play,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.0, 2.0),
            mouse_interaction=play,
            entities={CUBE_ENTITY_NAME: cube_cfg},
        ),
        decimation=5,  # 100 Hz physics / 5 = 20 Hz control (panda-gym parity).
        episode_length_s=5.0,
        device="cuda",
        robot=robot_entity_cfg,
        actions_cfg=actions_cfg,
        observations_cfg={
            "policy": ObservationGroupCfg(enable_corruption=True, terms=_obs_terms()),
            "critic": ObservationGroupCfg(enable_corruption=False, terms=_obs_terms()),
        },
        rewards_cfg={
            "ee_to_cube": RewardTermCfg(func=fpp_mdp.ee_to_cube_distance, weight=-1.0),
            "cube_to_goal": RewardTermCfg(func=fpp_mdp.cube_to_goal_distance, weight=-1.0),
            "success": RewardTermCfg(func=fpp_mdp.success_bonus, weight=5.0),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "cube_dropped": TerminationTermCfg(func=fpp_mdp.cube_dropped),
        },
        events_cfg={
            "reset_arm": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_to_default,
            ),
            "reset_cube": EventTermCfg(
                mode="reset",
                func=fpp_mdp.reset_cube_uniform,
            ),
            "resample_goal": EventTermCfg(
                mode="reset",
                func=fpp_mdp.resample_goal_uniform,
            ),
        },
    )


def franka_pick_and_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the joint-position env config (9-DoF). ``play=True`` switches to a
    single visualised env."""
    return _build_env_cfg(
        actions_cfg={
            "arm_and_hand": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(ARM_JOINTS_REGEX, FINGER_JOINTS_REGEX),
                use_default_offset=True,
            ),
        },
        requires_jac_and_ik=False,
        play=play,
    )


def franka_pick_and_place_cartesian_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the Cartesian env config (4-DoF, panda-gym parity):
    ``DifferentialIKAction`` on the arm + ``BinaryGripperAction`` on the fingers."""
    return _build_env_cfg(
        actions_cfg={
            "arm": DifferentialIKActionCfg(
                asset_name="robot",
                body_name=EE_LINK,
                joint_names=(ARM_JOINTS_REGEX,),
                use_orientation=False,
                scale=0.05,
            ),
            "gripper": BinaryGripperActionCfg(
                asset_name="robot",
                joint_names=(FINGER_JOINTS_REGEX,),
                open_pos=0.04,
                closed_pos=0.0,
            ),
        },
        requires_jac_and_ik=True,
        play=play,
    )
