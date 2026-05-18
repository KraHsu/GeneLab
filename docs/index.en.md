---
hide:
  - toc
---

<div class="gl-hero" markdown>

<p class="gl-hero__eyebrow">Research framework · Built on Genesis</p>

# GeneLab

<p class="gl-hero__lead" markdown>
Isaac Lab-shaped primitives for RL and robotics research, running on the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator.
Small registries, manager-based MDP, explicit Genesis backend — a stable
package layout and CLI for downstream robotics projects.
</p>

<div class="gl-hero__cta" markdown>
[Start the tutorial](tutorial.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/KraHsu/GeneLab){ .md-button }
</div>

</div>

<div class="gl-terminal" aria-hidden="true">
  <div class="gl-terminal__head">
    <i></i><i></i><i></i>
    <span class="gl-terminal__title">~/genelab</span>
  </div>
<pre><span class="gl-prompt">$</span> uv run genelab list tasks

  Registered tasks <span class="gl-comment">(4 discovered)</span>
  ──────────────────────────────────────────────────────
  <span class="gl-key">GeneLab-Inverted-Pendulum-v0</span>          trainable
  <span class="gl-key">GeneLab-Double-Inverted-Pendulum-v0</span>   trainable
  <span class="gl-key">Genelab-Velocity-Flat-Unitree-G1-v0</span>   trainable
  <span class="gl-key">Genelab-Tracking-Flat-Unitree-G1-v0</span>   trainable

<span class="gl-prompt">$</span> uv run genelab play GeneLab-Inverted-Pendulum-v0 --vis
</pre>
</div>

## Quick start

<div class="grid cards gl-cards" markdown>

-   <span class="gl-card__num">01</span> :material-school-outline:{ .lg .middle } **Tutorial**

    ---

    Build the first robot experiment, then continue to Unitree G1.

    [Tutorial →](tutorial.md)

-   <span class="gl-card__num">02</span> :material-download:{ .lg .middle } **Install**

    ---

    Set up `uv`, pick a `torch-*` extra, and verify the CLI.

    [Installation →](getting-started/installation.md)

-   <span class="gl-card__num">03</span> :material-console-line:{ .lg .middle } **CLI**

    ---

    `play`, `train`, `project new`, and the dotted override grammar.

    [CLI overview →](cli/overview.md)

-   <span class="gl-card__num">04</span> :material-package-variant-closed:{ .lg .middle } **Extend**

    ---

    Write a downstream extension that registers robots, envs, and tasks.

    [Extensions →](concepts/extensions.md)

</div>

## Abilities

<ul class="gl-abilities">
  <li><code>Discover</code><span>Register and inspect robots, environments, and tasks from installed or explicit extensions.</span></li>
  <li><code>Compose</code><span>Build Genesis scenes from robot assets, rigid objects, terrains, sensors, and recordings.</span></li>
  <li><code>Shape MDPs</code><span>Define actions, commands, observations, rewards, terminations, events, curricula, and metrics as named terms.</span></li>
  <li><code>Run</code><span>Use one CLI for headless rollouts, viewer playback, config overrides, and task introspection.</span></li>
  <li><code>Train</code><span>Launch RSL-RL PPO runs, checkpoint replay, profiler traces, and multi-GPU training.</span></li>
  <li><code>Extend</code><span>Keep downstream robot projects in independent Python packages with entry-point discovery.</span></li>
</ul>
