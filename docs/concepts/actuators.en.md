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

Actuators match joint groups by configured names or expressions, then expose dimensions and control
logic to action terms.

## Use `MlpResidualActuatorCfg`

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
uv run genelab play GeneLab-MlpResidual-Actuator-Showcase-v0 --steps 5
```

## Design guidance

Keep actuator grouping aligned with robot mechanics. Avoid one giant actuator if arm, hand, and base
joints need different gains, limits, or action scales.

## Where to continue

- [Task design](../best-practices/task-design.md)
- [API Reference](../api/reference.md)
