# Recording and Live Plotting

Recording turns selected runtime signals into live plots, files, or video without embedding output
code in the environment loop.

## Mental model

`RecordingCfg` describes one data source and one or more output sinks. The scene converts these
configs into Genesis recorders during build, then binds them to the live env after sensors exist.

```text
RecordingCfg(source, field, env_idx)
└── PyQtPlotCfg / MPLPlotCfg / MPLImagePlotCfg / NPZFileCfg / CSVFileCfg / VideoFileCfg
```

## Sources

| Source | Meaning |
|---|---|
| Sensor name | Read `env.sensors[name].data`, optionally walking `field`. |
| Callable with no args | Called directly. |
| Callable with `env` arg | Called with the live `ManagerBasedRlEnv`. |

`env_idx` squeezes a batched tensor for per-env plotting. Use `env_idx=None` when the source already
returns a scalar or when a file should capture the full batch.

## Outputs

| Output | Use case |
|---|---|
| `PyQtPlotCfg` | Live PyQtGraph line plots (time series). |
| `MPLPlotCfg` | Live Matplotlib line plots (time series). |
| `MPLImagePlotCfg` | Live Matplotlib image window for a `CameraSensor` source — `field="rgb"` for the frame, `field="depth"` for the depth map. |
| `NPZFileCfg` | Compressed array dumps at cleanup/reset. |
| `CSVFileCfg` | Row-oriented streaming output. |
| `VideoFileCfg` | Camera frames to `.mp4`. |

Line plots (`PyQtPlotCfg` / `MPLPlotCfg`) consume 1-D channel data; `MPLImagePlotCfg` consumes a 2-D
frame and requires a camera source. All plot windows auto-detect the display and no-op on a headless
host.

## Where to continue

- [Sensors](sensors.md)
- [Debug Common Failures](../best-practices/debugging.md)
- [API Reference](../api/reference.md)
