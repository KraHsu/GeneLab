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
[Get started](getting-started/installation.md){ .md-button .md-button--primary }
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

-   <span class="gl-card__num">01</span> :material-download:{ .lg .middle } **Install**

    ---

    Set up `uv`, pick a `torch-*` extra, and verify the CLI.

    [Installation →](getting-started/installation.md)

-   <span class="gl-card__num">02</span> :material-rocket-launch-outline:{ .lg .middle } **Run**

    ---

    List registered tasks and play one in a Genesis viewer.

    [Quickstart →](getting-started/quickstart.md)

-   <span class="gl-card__num">03</span> :material-console-line:{ .lg .middle } **CLI**

    ---

    `play`, `train`, `project new`, and the dotted override grammar.

    [CLI overview →](cli/overview.md)

-   <span class="gl-card__num">04</span> :material-package-variant-closed:{ .lg .middle } **Extend**

    ---

    Write a downstream extension that registers robots, envs, and tasks.

    [Extensions →](concepts/extensions.md)

</div>

## Modules

<ul class="gl-modules">
  <li><code>genelab.registry</code><span>Registries, registration helpers, extension loading.</span></li>
  <li><code>genelab.configs</code><span>Reusable dataclass configs (<code>ManagerBasedEnvCfg</code>, <code>TaskCfg</code>).</span></li>
  <li><code>genelab.lab</code><span>Public facade for registry + manager-based environment primitives.</span></li>
  <li><code>genelab.cli</code><span>Typer + Rich dispatcher; <code>play</code> / <code>train</code> / <code>info</code> / <code>project new</code>.</span></li>
  <li><code>genelab.envs</code> / <code>robots</code> / <code>tasks</code><span>Core registry helper namespaces.</span></li>
  <li><code>genelab.actuator</code> / <code>scene</code> / <code>sensor</code> / <code>terrains</code><span>Extension namespaces for robotics research code.</span></li>
</ul>
