---
hide:
  - toc
---

<div class="gl-hero" markdown>

<p class="gl-hero__eyebrow">研究框架 · 基于 Genesis</p>

# GeneLab

<p class="gl-hero__lead" markdown>
面向强化学习与机器人研究的 Isaac Lab 风格 API，由
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 提供仿真后端。
小型注册表、manager-based MDP、显式的 Genesis 后端 —— 为下游机器人项目
提供稳定的包结构与 CLI。
</p>

<div class="gl-hero__cta" markdown>
[开始教程](tutorial.md){ .md-button .md-button--primary }
[查看 GitHub](https://github.com/KraHsu/GeneLab){ .md-button }
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

## 快速开始

<div class="grid cards gl-cards" markdown>

-   <span class="gl-card__num">01</span> :material-school-outline:{ .lg .middle } **教程**

    ---

    构建第一个机器人实验，然后继续到 Unitree G1。

    [教程 →](tutorial.md)

-   <span class="gl-card__num">02</span> :material-download:{ .lg .middle } **安装**

    ---

    准备 `uv`，挑选一个 `torch-*` extra，并验证 CLI。

    [安装 →](getting-started/installation.md)

-   <span class="gl-card__num">03</span> :material-console-line:{ .lg .middle } **CLI**

    ---

    `play`、`train`、`project new`，以及点分路径 override 语法。

    [CLI 总览 →](cli/overview.md)

-   <span class="gl-card__num">04</span> :material-package-variant-closed:{ .lg .middle } **扩展**

    ---

    编写下游扩展包，注册机器人、环境与任务。

    [扩展加载 →](concepts/extensions.md)

</div>

## 能力速览

<ul class="gl-abilities">
  <li><code>发现</code><span>从已安装或显式导入的扩展中注册并检查机器人、环境和任务。</span></li>
  <li><code>组合</code><span>用机器人资产、刚体、地形、传感器和录制配置构建 Genesis scene。</span></li>
  <li><code>塑造 MDP</code><span>把 action、command、observation、reward、termination、event、curriculum、metric 定义成命名 term。</span></li>
  <li><code>运行</code><span>用统一 CLI 做 headless rollout、viewer 回放、配置 override 和任务检查。</span></li>
  <li><code>训练</code><span>启动 RSL-RL PPO、checkpoint 回放、profiler trace 和多 GPU 训练。</span></li>
  <li><code>扩展</code><span>把下游机器人项目放在独立 Python 包中，并通过 entry point 自动发现。</span></li>
</ul>
