# Sensors

Genesis does not expose every signal through an MJCF sensor block, so GeneLab provides a
backend-aware sensor abstraction. Sensors produce cached data that observations, rewards,
recordings, and custom code can consume.

## Lifecycle

Sensors are declared in `InteractiveSceneCfg.sensors`. `ManagerBasedRlEnv` builds and binds them
after the scene exists. Each control step invalidates cached data; the first access to `sensor.data`
computes fresh output. Reset clears per-env sensor state.

This lazy cache keeps repeated observation and reward reads consistent within one control step.

## Built-in sensor families

| Sensor | Use case |
|---|---|
| `BodyVelocitySensor` | Link linear or angular velocity at an offset. |
| `IMUSensor` | Orientation, projected gravity, linear/angular acceleration. |
| `CameraSensor` | RGB-D camera attached to a link. |
| `ContactSensor` | Per-link contact force and optional air/contact time. |
| `FrameTransformerSensor` | Target frame poses relative to a source frame. |
| `RayCastSensor` | Grid, ring, or hemisphere ray patterns. |
| `TerrainHeightSensor` | Downward height scan for terrain-aware locomotion. |
| `SelfContactSensor` | Self-collision/contact signals. |
| `RootAngularMomentumSensor` | Root angular momentum diagnostics. |

## Sensor data in MDP terms

Observation terms can read sensors through `mdp.sensor_data` or a custom callable. Rewards and
metrics can access `env.sensors[name].data` directly.

Keep sensor names stable; they become config paths, recording sources, and term params.

## Camera caveat

Parallel RGB-D rendering requires CUDA and `InteractiveSceneCfg.batch_render=True`. Treat camera
tasks as a separate hardware profile from light state-based RL tasks.

## Where to continue

- [Recording and plotting](recording.md)
- [Module Map](../reference/module-map.md)
- [API Reference](../api/reference.md)
