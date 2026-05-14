# Actuators

`genelab.actuator` is the electromechanical layer between a manager-based RL policy and
Genesis. Each `ActuatorBase` owns a regex-matched slice of an articulation's actuated
joints and decides how the per-step joint position target reaches the simulator: either
through Genesis's implicit PD law, or through a Python-side torque computation pushed via
`control_dofs_force`.

## Why a dedicated layer

Real robots are not monolithic — a single Unitree G1 carries six distinct motor classes
spread across knees, hips, shoulders, and ankles. Each class has different stiffness,
damping, effort, velocity, armature, and friction. Folding all of that into flat
`joint_kp` / `joint_kv` dicts on `ArticulationCfg` loses the per-motor identity and
prevents modelling velocity-dependent saturation. The actuator namespace restores the
group abstraction and adds three torque models.

## Three shipped models

| Class | Channel | Torque computation |
|---|---|---|
| `ImplicitPDActuator` | `implicit_pd` | None — Genesis solves `tau = kp*(q* - q) - kv*q_dot` internally |
| `IdealPDActuator` | `force` | `tau = clip(kp*(q* - q) - kv*q_dot, ±effort_limit)` |
| `DCMotorActuator` | `force` | `IdealPD` plus linear de-rating in the driving direction |

`ImplicitPDActuator` is the numerical equivalent of the pre-M2 baseline — Genesis is
called with `set_dofs_kp` / `set_dofs_kv` and the per-step target reaches the engine
through `control_dofs_position`. `IdealPDActuator` zeroes the simulator-side PD and
becomes the canonical torque-control path for explicit effort limits. `DCMotorActuator`
adds a torque-speed curve:

`tau_max(q_dot) = saturation_effort * clip(1 - |q_dot| / velocity_limit, 0, 1)`

The de-rating only applies to the driving direction (the sign of `tau_pd` matches the
sign of `q_dot`); reverse-braking torque retains the full `effort_limit`. This matches
the Isaac Lab `DCMotor` semantics — the back-EMF saturation does not penalise
regenerative braking.

## Composing onto an articulation

`ArticulationCfg.actuators` is a `dict[str, ActuatorBaseCfg]`. Every actuated joint must
be covered by exactly one group; both unmatched and conflicting regex matches raise
`ValueError` at `Articulation.bind` time. Passive joints get explicit zero-gain
`ImplicitPDActuatorCfg` entries so the topology is visible in the config.

```python
from genelab.actuator import DCMotorActuatorCfg, ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg

cfg = ArticulationCfg(
    mjcf_path="/path/to/robot.xml",
    default_joint_pos={"cart_slide": 0.0, "pole_hinge": 0.0},
    actuators={
        "cart": ImplicitPDActuatorCfg(
            target_names_expr=("cart_slide",), stiffness=80.0, damping=8.0,
            action_scale=1.0,
        ),
        "pole": ImplicitPDActuatorCfg(
            target_names_expr=("pole_hinge",), stiffness=0.0, damping=0.0,
        ),
    },
)
```

`target_names_expr` is a tuple of regex patterns matched against the articulation's
joint names. Each group's `action_scale` is published through
`Articulation.action_scale_tensor`, which `JointPositionAction` uses by default when
`scale` is `None` on the action config.

## Failure modes worth knowing

* **Empty `actuators` dict** — the articulation rejects an articulation with no declared
  actuators. Even a passive pendulum needs a zero-gain group covering its hinge.
* **Unmatched joint** — `ValueError` lists the joints that no regex covered.
* **Conflicting groups** — `ValueError` lists the joints that more than one group matched
  along with the group names involved.
* **`DCMotorActuatorCfg.velocity_limit is None`** — the de-rating curve has no
  breakpoint; `__post_init__` raises before the actuator gets a chance to run.

## Switching G1 between models

The bundled `examples/unitree` extension declares six `DCMotorActuatorCfg` groups (5020,
7520_14, 7520_22, 4010, waist, ankle). To run the same robot under the simulator's
implicit PD law instead, swap each `DCMotorActuatorCfg(...)` for an
`ImplicitPDActuatorCfg(...)` with the same `stiffness` / `damping` / `effort_limit`. The
articulation makes no global assumption about which channel is active — every group
chooses individually.

## See also

- [Configs](configs.md)
- [Sensors](sensors.md)
