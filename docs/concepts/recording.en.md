# Recording and Live Plotting

Recording turns selected runtime signals into live plots, files, or video without embedding output
code in the environment loop.

## Mental model

`RecordingCfg` describes one data source and one or more output sinks. The scene converts these
configs into Genesis recorders during build, then binds them to the live env after sensors exist.

```text
RecordingCfg(source, field, env_idx)
└── PyQtPlotCfg / MPLPlotCfg / NPZFileCfg / CSVFileCfg / VideoFileCfg
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
| `PyQtPlotCfg` | Live PyQtGraph plots. |
| `MPLPlotCfg` | Live Matplotlib plots. |
| `NPZFileCfg` | Compressed array dumps at cleanup/reset. |
| `CSVFileCfg` | Row-oriented streaming output. |
| `VideoFileCfg` | Camera frames to `.mp4`. |

## Where to continue

- [Sensors](sensors.md)
- [Debug Common Failures](../best-practices/debugging.md)
- [API Reference](../api/reference.md)
