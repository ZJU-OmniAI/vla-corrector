# Paper Summary for README and Project Page

Source: the unpacked LaTeX paper source, main file `neurips_2026.tex`.

This note extracts public-facing facts from the LaTeX paper source. It is intended to keep README, GitHub Pages, and media copy aligned with the paper.

## Basic Information

- Paper title: **VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon**
- Authors: Yi Pan, Miao Pan, Qi Lu, Jiaming Huang, Man Zhang, Wenqi Zhang
- Institution: Zhejiang University
- Project name: VLA-Corrector
- Paper link: Coming soon
- arXiv link: Coming soon
- Repository: https://github.com/ZJU-OmniAI/vla-corrector
- Project page: https://zju-omniai.github.io/vla-corrector/

## One-Sentence Summary

VLA-Corrector adds a lightweight inference-time detect-and-correct layer to action-chunked VLA policies, using latent visual dynamics monitoring, event-triggered truncation, and OGG-guided recovery replanning to mitigate open-loop blind spots without retraining the VLA backbone.

## Abstract Core

Action-chunked VLA policies reduce policy-call frequency by executing multiple predicted actions under a fixed action horizon, but this creates an open-loop blind spot where fresh observations are not used until the horizon ends. VLA-Corrector monitors latent visual dynamics, detects persistent deviation, truncates stale actions, and applies corrective replanning with Online Gradient Guidance. This produces an event-triggered adaptive action horizon that preserves long-horizon efficiency when the chunk remains reliable and invokes short-horizon correction when drift appears.

## Research Problem

The paper studies how fixed action horizons in action-chunked VLA policies create a performance-efficiency trade-off. Longer horizons reduce policy calls but weaken reactivity and allow errors to accumulate. Shorter horizons improve responsiveness but require much more frequent VLA inference.

## Open-Loop Blind Spot

Explicitly stated in the paper. The introduction describes an open-loop blind spot in which the robot keeps executing stale queued actions while fresh observations are ignored until the horizon ends. The paper discusses risks such as lack of real-time reactivity and compounding errors that may push the robot into states from which the next replan cannot recover.

## Core Idea

VLA-Corrector decouples action generation from execution monitoring:

- the frozen VLA backbone generates action chunks;
- an external latent dynamics corrector predicts short-horizon visual latent residuals;
- LVM compares expected and observed latent evolution online;
- persistent mismatch triggers truncation of stale queued actions;
- OGG guides the next recovery replan.

## 40M Corrector Support

Explicitly supported. The paper states that the external corrector is a residual MLP with four hidden layers of width `[2048, 2048, 2048, 2048]`, and that correctors used in experiments contain approximately **38--42M parameters**, referred to as a lightweight **~40M MLP corrector**.

## Method Modules

- External latent dynamics corrector `M_phi`
- Latent-space Vision Monitor (LVM)
- Robust event-triggered truncation with sliding-window median, MAD thresholds, hysteresis, and persistence checking
- Online Gradient Guidance (OGG) for the single policy call immediately after an interrupt
- Event-triggered adaptive action horizon

## Training Flow

1. Fine-tune or obtain a VLA policy on the benchmark training set.
2. Freeze the VLA backbone.
3. Use the VLA visual encoder to extract visual latents from demonstration trajectories.
4. Train the corrector to predict short-horizon latent residuals.
5. Use a loss combining residual magnitude matching and cosine directional consistency.

Implementation details from the appendix:

- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Schedule: cosine annealing
- Training: 30 epochs with early stopping patience 5
- Deployed h1-k10 correctors: batch size 512 and cosine-based training loss

## Inference Flow

1. The base VLA generates an action chunk.
2. The controller executes queued actions under the current horizon.
3. LVM compares expected and actual latent dynamics from fresh observations.
4. Persistent drift triggers an interrupt event.
5. Remaining queued actions are discarded.
6. The next policy call is performed under OGG-guided corrective mode.
7. Subsequent policy calls return to standard inference unless another interrupt is detected.

## Benchmarks and Backbones

- Simulation benchmarks: MetaWorld and LIBERO
- Real robot platform: AgileX PiPER 6-DoF arm
- Main backbone: PI0.5
- Cross-architecture evaluation: SmolVLA and X-VLA
- Metrics: task success rate, policy calls, success-per-call efficiency, post-interrupt recovery rate, and inference-time overhead where applicable

## Main Results

### MetaWorld

Average success rate across difficulty splits:

- PI0.5: 48.70 -> 64.35, +15.65
- SmolVLA: 61.90 -> 66.65, +4.75
- X-VLA: 55.55 -> 59.60, +4.05

Largest split-level gain reported: PI0.5 Very Hard, 41.0 -> 65.0.

### LIBERO

PI0.5 few-shot fine-tuned model:

- Baseline: 94.00 average success
- + VLA-Corrector: 97.80 average success
- Fully fine-tuned baseline in reported setting: 96.95

### Policy-Call Efficiency

Largest reported success-per-call gains:

- PI0.5: 29.9%
- SmolVLA: 45.3%
- X-VLA: 39.1%

### Real World

AgileX PiPER average success across three task groups:

- PI0.5 baseline: 55.6%
- + VLA-Corrector: 73.3%
- Largest gain: disturbance recovery group, +28.3 points

## Ablation and Analysis

- Truncation only: 48.70 -> 60.35 MetaWorld average success
- Truncation + OGG: 48.70 -> 64.35
- Decoupled LVM + OGG outperforms an internal detection head in the reported ablation
- LVM-40M is the default; 10M is weaker and 160M gives almost no additional average gain
- 83.7% of truncations occur in manually labeled critical phases
- OGG improves post-interrupt recovery and adds event-triggered wall-clock overhead

## Limitations

The paper discusses:

- additional wall-clock computation from OGG;
- the fact that VLA-Corrector improves inference-time recovery but does not replace the recovery capability of the underlying VLA backbone;
- real-world failures when disturbances exceed reachable regions, happen after unfavorable gripper poses, require force-sensitive contact handling, or involve visual ambiguity.

## README-Suitable English

VLA-Corrector is a lightweight inference-time detect-and-correct framework for action-chunked VLA policies. It monitors latent visual dynamics, truncates stale queued actions when persistent drift is detected, and guides the next recovery replan without retraining the frozen VLA backbone.

## Project-Page-Suitable English

A ~40M external latent dynamics corrector for mitigating blind spots in open-loop VLA execution: monitor latent visual dynamics, interrupt stale action chunks, and recover through OGG-guided replanning.

## Chinese Media Angles

- 用 40M 小模型补上机器人开环控制盲区
- 从开环执行到纠错执行
- 不重训大 VLA，而是给 VLA 加一个轻量动作纠错器
- 面向具身智能长程接触操作中的动作漂移和扰动恢复
