"""Manager-based RL env config for the Franka pick-and-place SAC+HER task.

The single env config exposes:

* 4-DoF Cartesian action surface ``(dx, dy, dz, gripper)`` —
  :class:`DifferentialIKAction` (orientation locked to the panda-gym downward
  pose) on the seven arm joints + :class:`ContinuousGripperAction` on the
  fingers.
* Goal-conditioned observation groups (``policy`` / ``critic`` /
  ``achieved_goal`` / ``desired_goal``) + TimeFeature so SB3's
  ``HerReplayBuffer`` sees a Dict obs and the value head can resolve the
  episode timeout.
* Sparse goal reward + a per-step lift bonus, mirrored exactly in
  :func:`franka_pick_and_place_compute_reward` so HER relabelling and the
  online env reward share one shape.
* Actuator overrides on the Franka robot — arm PD stiffened so it can lift
  the cube without sagging, hand actuator sped up so a single ``-1`` gripper
  action closes the fingers in one env step, cube friction raised so the
  fingers can actually carry it.
"""

from genelab import mdp
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.continuous_gripper import ContinuousGripperActionCfg
from genelab.mdp.actions.ee_delta_ik import DifferentialIKActionCfg
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
from genelab_franka_pick_and_place.robot import (
    PANDA_GYM_NEUTRAL_POSE,
    get_franka_pick_and_place_robot_cfg,
)


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
        # TimeFeature mirrors ``sb3_contrib.common.wrappers.TimeFeatureWrapper``;
        # with a fixed-length episode + HER relabelling, hiding the remaining
        # time makes the env non-Markov for the value head.
        "time_feature": ObservationTermCfg(func=fpp_mdp.time_feature_obs),
    }


def franka_pick_and_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the SAC+HER pick-and-place env config (``play=True`` switches to a
    single visualised env)."""
    robot_entity_cfg = get_franka_pick_and_place_robot_cfg(
        requires_jac_and_ik=True,
        default_joint_pos=PANDA_GYM_NEUTRAL_POSE,
    ).to_entity_cfg()
    # Actuator overrides scoped to this task — Isaac Lab's stock Franka gains
    # were tuned for the unloaded arm with slow finger commands; both fail
    # once the gripper has to actually catch and carry the cube.
    # * panda_arm stiffness 400 -> 2000: stock PD sags below where it can
    #   recover once carrying the cube; stiffer PD holds the arm against the
    #   combined gravity + cube load. Damping bumped to keep ~critical.
    # * panda_hand effort 20 -> 100, velocity 0.2 -> 1.0: defaults close the
    #   fingers at ~0.1 rad/s; new limits shut the fingers in roughly one env
    #   step, fast enough to catch the cube before it falls out.
    arm_actuator = robot_entity_cfg.actuators["panda_arm"]
    arm_actuator.stiffness = 2000.0
    arm_actuator.damping = 200.0
    hand_actuator = robot_entity_cfg.actuators["panda_hand"]
    hand_actuator.effort_limit = 100.0
    hand_actuator.velocity_limit = 1.0

    cube_cfg = RigidObjectCfg(
        morph="box",
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        init_pos=(CUBE_NOMINAL_XY[0], CUBE_NOMINAL_XY[1], CUBE_HALF),
        init_quat=(1.0, 0.0, 0.0, 0.0),
        fixed=False,
        # Genesis's built-in rigid friction is too low to keep the cube wedged
        # between the parallel fingers; ``1.0`` matches the panda-gym MuJoCo
        # cube and is what makes the gripper actually carry it on lift-off.
        friction=1.0,
    )

    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            # HER + SAC does not benefit from massive parallelism — the
            # replay buffer holds whole episodes, and with thousands of envs
            # each env contributes barely one episode before SAC's
            # gradient_steps=1 default leaves the data under-trained. See the
            # cfg comment in :func:`franka_pick_and_place_sb3_cfg`.
            num_envs=64 if not play else 1,
            dt=0.01,
            substeps=2,
            vis=play,
            # Run Genesis on the GPU. ``SimulationCfg.gpu`` defaults to False (CPU
            # backend); without this the sim runs on CPU with the policy/tensors on
            # cuda — GPU idle and far slower (same regression fixed for G1).
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.0, 2.0),
            mouse_interaction=play,
            entities={CUBE_ENTITY_NAME: cube_cfg},
        ),
        decimation=5,  # 100 Hz physics / 5 = 20 Hz control (panda-gym parity).
        episode_length_s=5.0,
        device="cuda",
        # HER's ``compute_reward`` returns the per-step reward at unit scale (no
        # dt multiplier), so the in-buffer reward must match — otherwise online
        # transitions hit {-dt, 0} while relabelled ones hit {-1, 0}.
        scale_rewards_by_dt=False,
        robot=robot_entity_cfg,
        actions_cfg={
            "arm": DifferentialIKActionCfg(
                asset_name="robot",
                body_name=EE_LINK,
                joint_names=(ARM_JOINTS_REGEX,),
                use_orientation=False,
                lock_orientation=True,
                scale=0.05,
            ),
            "gripper": ContinuousGripperActionCfg(
                asset_name="robot",
                joint_names=(FINGER_JOINTS_REGEX,),
                open_pos=0.04,
                closed_pos=0.0,
                speed=0.1,
            ),
        },
        observations_cfg={
            "policy": ObservationGroupCfg(enable_corruption=True, terms=_obs_terms()),
            "critic": ObservationGroupCfg(enable_corruption=False, terms=_obs_terms()),
            # achieved_goal = cube_pos, desired_goal = goal_pos — both clean
            # (no observation noise) so HER relabelling sees exact positions.
            "achieved_goal": ObservationGroupCfg(
                enable_corruption=False,
                terms={"cube_pos": ObservationTermCfg(func=fpp_mdp.cube_pos_obs)},
            ),
            "desired_goal": ObservationGroupCfg(
                enable_corruption=False,
                terms={"goal_pos": ObservationTermCfg(func=fpp_mdp.goal_pos_obs)},
            ),
        },
        rewards_cfg={
            "sparse_goal": RewardTermCfg(func=fpp_mdp.sparse_goal_reward, weight=1.0),
            "lift_bonus": RewardTermCfg(func=fpp_mdp.lift_bonus, weight=0.2),
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
