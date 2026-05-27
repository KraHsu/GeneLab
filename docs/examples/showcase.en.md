# Showcase

`examples/genelab_showcase` contains focused tasks for individual framework capabilities. Use it
when you want a runnable example of one subsystem without reading a full robot task.

## Capabilities

| Area | Demonstrates |
|---|---|
| Sensors | Body velocity, IMU-like state, contact, ray-cast, terrain height, joint force/torque. |
| Contact | Contact forces, air time, landing and slip metrics. |
| Terrain | Generated terrain grids and height scans. |
| Curriculum | Terrain-level progression. |
| Actuators | Switching actuator models (IdealPD, MLP-residual) and action behavior. |
| Recording | Live plots, files, and video-oriented recording configs. |

## Run

```bash
uv pip install -e examples/genelab_showcase
genelab list tasks
genelab play <showcase-task-id> --vis --steps 300
```

Use `genelab info <showcase-task-id>` to inspect the exact override paths for each showcase task.

## See also

- [Sensors](../concepts/sensors.md)
- [Recording and plotting](../concepts/recording.md)
- [Terrains](../concepts/terrains.md)
