"""Actuator showcase env: Franka with IdealPDActuator on the arm (force-channel).

Diff vs ``genelab.asset_zoo.FrankaPandaCfg``:

* The seven arm joints use :class:`IdealPDActuator` rather than the default implicit
  PD. The actuator zeros Genesis's internal PD gains at bind time and drives the
  joints by writing torque through ``control_dofs_force`` every control step.
* Effort and velocity limits are unchanged from the implicit-PD defaults so the only
  observable difference at runtime is the control channel.
* The hand keeps :class:`ImplicitPDActuator` — gripper closing is a stiff position
  task and benefits from Genesis's internal solver.
"""

from genelab import mdp
from genelab.actuator import (
    IdealPDActuatorCfg,
    ImplicitPDActuatorCfg,
    MlpResidualActuatorCfg,
)
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg

from genelab_showcase.actuators.residual_net import residual_net_path


def actuator_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Franka with IdealPD on the arm; otherwise identical to the sensors showcase."""

    robot_cfg = FrankaPandaCfg()
    # Replace the arm actuator with an IdealPD group (same stiffness / damping numbers,
    # but force-channel control).
    robot_cfg.actuators = {
        "panda_arm": IdealPDActuatorCfg(
            target_names_expr=(r"^joint[1-7]$",),
            stiffness=400.0,
            damping=80.0,
            effort_limit=87.0,
            velocity_limit=2.175,
            action_scale=0.5,
        ),
        "panda_hand": ImplicitPDActuatorCfg(
            target_names_expr=(r"finger_joint.*",),
            stiffness=1.0e4,
            damping=200.0,
            effort_limit=20.0,
            velocity_limit=0.2,
            action_scale=0.04,
        ),
    }
    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=1,
            dt=0.01,
            substeps=2,
            steps=200,
            vis=False,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(env_spacing=(2.0, 2.0)),
        decimation=2,
        episode_length_s=20.0,
        device="cuda",
        robot=robot_cfg,
        actions_cfg={
            "panda_arm": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(r"^joint[1-7]$",),
                use_default_offset=True,
            ),
            "panda_hand": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(r"finger_joint.*",),
                use_default_offset=True,
            ),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        events_cfg={
            "reset_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_to_default,
                params={"pos_jitter": 0.0, "vel_jitter": 0.0},
            ),
        },
    )


def mlp_residual_actuator_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Franka arm on an :class:`MlpResidualActuator`: DC-motor torque base + a learned residual.

    Identical to :func:`actuator_showcase_env_cfg` except the seven arm joints use an
    ``MlpResidualActuatorCfg``. ``network_file`` points at a tiny TorchScript net
    (generated under ``CACHE_DIR``) applying a small velocity-damping residual, and
    ``residual_scale > 0``, so the MLP-residual code path runs — not the DC-motor
    fallback. The residual is tiny relative to the 87 Nm effort budget, so the only
    observable change from the DC-motor base is the loaded correction.
    """

    cfg = actuator_showcase_env_cfg()
    # Swap only the arm group; the hand keeps the base showcase's ImplicitPD. The
    # factory materializes the residual net to CACHE_DIR and points network_file at it.
    cfg.robot.actuators["panda_arm"] = MlpResidualActuatorCfg(
        target_names_expr=(r"^joint[1-7]$",),
        stiffness=400.0,
        damping=80.0,
        effort_limit=87.0,
        velocity_limit=2.175,
        saturation_effort=87.0,
        action_scale=0.5,
        network_file=residual_net_path(),
        residual_scale=1.0,
    )
    return cfg
