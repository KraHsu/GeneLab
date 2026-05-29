# Actuators

Actuators are the layer between policy actions and Genesis joint control. They let a robot config
choose how each joint group is driven without changing the action term or task logic.

## Why a dedicated actuator layer

Different robots need different control assumptions. A simple cart can use implicit PD targets; a
legged robot may need torque limits and DC motor saturation. GeneLab keeps those mechanics in
actuator configs attached to an `ArticulationCfg`.

## Built-in models

| Model | Behavior |
|---|---|
| `ImplicitPDActuator` | Uses Genesis/simulator implicit PD control. |
| `IdealPDActuator` | Computes PD torque in Python and writes force targets. |
| `DCMotorActuator` | Extends ideal PD with motor limits and saturation behavior. |
| `MlpResidualActuator` | Extends `DCMotorActuator` with a TorchScript residual torque model. |

`MujocoStyleActuatorCfg` is a convenience config for `IdealPDActuator` — see
[MJCF-style actuator configuration](#mjcf-style-actuator-configuration) below.

Actuators match joint groups by configured names or expressions, then expose dimensions and control
logic to action terms.

## Using `MlpResidualActuatorCfg`

Use `MlpResidualActuatorCfg` when a robot already has a usable DC-motor model but hardware logs show
a repeatable torque-tracking gap. The actuator loads a TorchScript module from `network_file` and
adds its output to the DC-motor torque:

```python
from genelab.actuator import MlpResidualActuatorCfg

robot_cfg.actuators["legs"] = MlpResidualActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_knee_joint", ".*_ankle_joint"),
    stiffness=40.0,
    damping=1.0,
    effort_limit=120.0,
    velocity_limit=30.0,
    saturation_effort=120.0,
    action_scale=0.25,
    network_file="assets/actuators/leg_residual.pt",
    residual_scale=0.5,
)
```

The TorchScript module receives a tensor whose last dimension is `[target_pos - joint_pos,
joint_vel]` and returns one residual torque per joint. A standard MLP with `nn.Linear(2, hidden)` as
its first layer satisfies that contract. Set `network_file=None` to keep the same config shape while
falling back to plain `DCMotorActuator` behavior.

`velocity_limit` is required because `MlpResidualActuatorCfg` inherits the DC-motor torque-speed
model. `effort_limit` or `saturation_effort` must define the final torque budget; the residual output
is clamped back into that budget after it is added.

For a runnable example, `GeneLab-MlpResidual-Actuator-Showcase-v0` (in `examples/genelab_showcase`)
drives the Franka arm with an `MlpResidualActuator` whose tiny TorchScript residual is generated on
first use:

```bash
genelab play GeneLab-MlpResidual-Actuator-Showcase-v0 --steps 5
```

## MJCF-style actuator configuration { #mjcf-style-actuator-configuration }

`MujocoStyleActuatorCfg` translates a Mujoco general actuator into an `IdealPDActuator`. Use it when
a robot declares actuators in MJCF as `dyntype=none`, `gaintype=fixed`, `biastype=affine`, and
hand-writing the equivalent `stiffness` / `damping` would obscure the original source of truth.

```python
from genelab.actuator import MujocoStyleActuatorCfg

robot_cfg.actuators["arm"] = MujocoStyleActuatorCfg(
    target_names_expr=("joint[1-7]",),
    gear=1.0,
    bias_prm=(0.0, -200.0, -5.0),  # constant, qpos, qvel terms
    effort_limit=80.0,
)
# Equivalent to IdealPDActuatorCfg(stiffness=200.0, damping=5.0, effort_limit=80.0).
```

Any other dyntype / gaintype / biastype combination is rejected (Genesis's own MJCF parser also logs
a warning for the same combination), and `bias_prm[0]` must be zero — `IdealPDActuator` has no
constant bias-torque term. The final `stiffness` / `damping` are computed as `-gear * bias_prm[1]`
and `-gear * bias_prm[2]`.

## Design guidance

Keep actuator grouping aligned with robot mechanics. Avoid one giant actuator if arm, hand, and base
joints need different gains, limits, or action scales.

## Where to continue

- [Task design](../best-practices/task-design.md)
- [API Reference](../api/reference.md)
