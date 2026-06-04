"""Unit test for the MlpResidualActuator wired into the actuator showcase.

Asserts the showcase env cfg puts an ``MlpResidualActuatorCfg`` on the Franka arm
with a real ``network_file`` + ``residual_scale > 0``, that the generated TorchScript
net round-trips (load + forward), and that the loaded residual actually changes the
actuator's torque (the residual code path runs, not the DC-motor fallback). No Genesis
runtime needed — the actuator is built directly from the cfg with fake joint ids.
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from genelab.actuator import MlpResidualActuator, MlpResidualActuatorCfg  # noqa: E402

_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def _arm_cfg() -> MlpResidualActuatorCfg:
    from genelab_showcase.actuators.env_cfg import mlp_residual_actuator_showcase_env_cfg

    # The showcase spawns two robots; the residual arm is on the primary "robot".
    cfg = mlp_residual_actuator_showcase_env_cfg()
    robot = cfg.robots["robot"] if cfg.robots else cfg.robot
    arm = robot.actuators["panda_arm"]
    assert isinstance(arm, MlpResidualActuatorCfg)
    return arm


def _build_arm_actuator(cfg: MlpResidualActuatorCfg) -> MlpResidualActuator:
    ids = torch.arange(len(_ARM_JOINTS))
    return cfg.class_type(  # type: ignore[return-value]
        cfg,
        name="panda_arm",
        joint_ids=ids,
        dof_ids=ids,
        joint_names=_ARM_JOINTS,
        num_envs=1,
        device="cpu",
    )


def test_showcase_arm_uses_mlp_residual_with_network_file() -> None:
    cfg = _arm_cfg()
    assert cfg.network_file is not None
    assert cfg.residual_scale > 0.0
    assert cfg.velocity_limit is not None  # required by the DC-motor base


def test_showcase_residual_net_round_trips() -> None:
    from genelab_showcase.actuators.residual_net import residual_net_path

    net = torch.jit.load(residual_net_path()).eval()
    # Contract: last dim [pos_error, joint_vel] -> per-joint residual torque.
    feats = torch.tensor([[[0.5, 2.0]]])  # (B=1, J=1, 2)
    out = net(feats)
    assert out.shape == (1, 1, 1)
    # Generated net applies residual = 80 * pos_error (velocity weight 0).
    assert out.reshape(-1).item() == pytest.approx(40.0, abs=1e-4)


def test_showcase_residual_path_executes() -> None:
    """The loaded residual changes the torque vs the pure-DC-motor fallback."""
    cfg = _arm_cfg()
    actuator = _build_arm_actuator(cfg)
    fallback = _build_arm_actuator(dataclasses.replace(cfg, network_file=None))

    # Unsaturated operating point (small position error, zero velocity) so the residual
    # (driven by pos_error) is exercised and not masked by the effort clamp.
    joint_pos = torch.zeros(1, len(_ARM_JOINTS))
    joint_vel = torch.zeros(1, len(_ARM_JOINTS))
    target = torch.full((1, len(_ARM_JOINTS)), 0.1)

    tau = actuator.compute(joint_pos, joint_vel, target)
    tau_fallback = fallback.compute(joint_pos, joint_vel, target)
    diff = (tau - tau_fallback)[0]
    # residual = residual_scale * (80 * pos_error) = 80 * 0.1 = 8.0 on every arm joint.
    # atol 1e-3 absorbs float32 noise from the (cancelling) DC-motor base; the 8.0
    # residual signal is far larger.
    assert torch.allclose(diff, torch.full((len(_ARM_JOINTS),), 8.0), atol=1e-3)
