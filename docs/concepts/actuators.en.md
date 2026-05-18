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

Actuators match joint groups by configured names or expressions, then expose dimensions and control
logic to action terms.

## Design guidance

Keep actuator grouping aligned with robot mechanics. Avoid one giant actuator if arm, hand, and base
joints need different gains, limits, or action scales.

## Where to continue

- [Task design](../best-practices/task-design.md)
- [API Reference](../api/reference.md)
