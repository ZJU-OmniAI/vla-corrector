# VLA-Corrector

**Lightweight detect-and-correct inference for adaptive action horizons in action-chunked VLA policies.**

[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/ZJU-OmniAI/vla-corrector)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zju-omniai.github.io/vla-corrector/)
[![Paper](https://img.shields.io/badge/Paper-Coming%20soon-lightgrey)](#citation)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20soon-lightgrey)](#citation)

- **Code:** https://github.com/ZJU-OmniAI/vla-corrector
- **Project page:** https://zju-omniai.github.io/vla-corrector/
- **Paper:** Coming soon
- **arXiv:** Coming soon

## Overview

Vision-Language-Action (VLA) policies often generate an action chunk in one policy call and then execute several actions before querying the policy again. This reduces policy-call frequency and keeps actions temporally smooth, but it also creates an **open-loop blind spot**: fresh observations arrive during execution, while the policy keeps following queued actions until the fixed action horizon ends. In contact-rich manipulation, small pose drift, slippage, collision, or disturbance can accumulate before the next replan.

**VLA-Corrector** is a lightweight inference-time framework for action-chunked VLA policies. It keeps the VLA backbone frozen and adds an external latent dynamics corrector. During rollout, the corrector monitors whether the observed visual evolution matches the expected local dynamics. When persistent drift is detected, VLA-Corrector truncates stale queued actions and applies Online Gradient Guidance (OGG) to the next recovery replan.

The result is an **event-triggered adaptive action horizon**: long chunks are preserved when they remain reliable, while corrective replanning is invoked when execution begins to drift. This differs from fine-tuning the whole VLA: the trainable component is a separate lightweight latent dynamics module trained on frozen VLA features.

## Key Ideas

- **Open-loop blind-spot monitoring:** a Latent-space Vision Monitor (LVM) compares expected and observed latent visual dynamics during action-chunk execution.
- **Lightweight external corrector:** the paper reports residual MLP correctors with approximately **38--42M parameters**, referred to as a lightweight ~40M MLP corrector.
- **Event-triggered truncation:** persistent visual-dynamics mismatch interrupts the current chunk and discards stale remaining actions.
- **Corrective re-inference:** OGG guides the single policy call immediately after an interrupt toward a recovery-oriented latent direction.
- **Frozen VLA backbone:** VLA-Corrector augments PI0.5, SmolVLA, and X-VLA style backbones without retraining their policy weights for the corrector module.

## Method

VLA-Corrector decouples action generation from execution monitoring:

```text
Observation + Instruction
        |
        v
Base action-chunked VLA policy
        |
        v
Queued action chunk ------ fresh observations during execution
        |                                      |
        v                                      v
Execute actions                         Latent-space Vision Monitor
        |                                      |
        +----------- persistent drift? --------+
                         |
             no          |          yes
        keep executing   |   truncate stale actions
                         v
                  OGG-guided recovery replan
```

The external corrector is trained after the VLA backbone has been fine-tuned on the benchmark training set. The VLA visual encoder extracts frozen visual latents from demonstration trajectories. Given a transition `(o_t, a_t, o_{t+k})`, the corrector predicts the short-horizon latent residual induced by the executed action. The training objective combines residual magnitude matching and directional consistency.

During deployment, LVM computes an inconsistency score between expected and observed latent residuals. A robust thresholding state machine based on a sliding window, median absolute deviation, hysteresis, and persistence checking decides whether an interrupt event should be triggered. After an interrupt, the next policy call receives OGG, which aligns the predicted action effect with a corrective latent direction.

Paper figures:

- [Method overview](docs/assets/images/method_overview.pdf)
- [Open-loop execution comparison](docs/assets/images/open_loop_vs_corrected_execution.pdf)
- [Performance-efficiency trade-off](docs/assets/images/performance_efficiency_pareto.pdf)

## Installation

```bash
conda env create -f environment.yml
conda activate lerobot
python -m pip install -e . --no-build-isolation
```

For PushT simulation smoke tests:

```bash
python -m pip install -e '.[pusht]' --no-build-isolation
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

The exported environment name is `lerobot`. You can edit the `name:` field in `environment.yml` before creating the environment.

## Data and Checkpoints

This repository does **not** include:

- datasets;
- demo data;
- training outputs;
- Hugging Face pretrained weights;
- fine-tuned policy checkpoints;
- trained corrector checkpoints;
- wandb logs or caches.

Prepare or specify these paths yourself:

```text
<DATASET_DIR>             # Source LeRobot, MetaWorld, or LIBERO dataset
<EXTRACTED_CACHE_DIR>     # Extracted latent cache from siglip_dynamics.extract
<POLICY_CHECKPOINT>       # Base or fine-tuned PI0.5, SmolVLA, or X-VLA policy checkpoint
<CORRECTOR_CHECKPOINT>    # Trained latent dynamics corrector checkpoint directory
<OUTPUT_DIR>              # Local output directory, usually under outputs/
```

Pretrained weights are not included. If you use Hugging Face model IDs, the underlying libraries may download them automatically. If you work offline, download the required weights yourself and pass local paths through command-line arguments.

Known model names referenced by the code include:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
google/paligemma-3b-pt-224
lerobot/fast-action-tokenizer
```

Fine-tuned checkpoints are not included in this repository. Please specify your own checkpoint paths with `--policy.path` and `--safety_model_path`.

## Quick Start

Main modified evaluation entry point:

```bash
python -m lerobot.scripts.lerobot_eval_modified_detection --help
```

Minimal PushT smoke training from a local LeRobot-format dataset:

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<LOCAL_REPO_ID> \
  --dataset.root=<LOCAL_LEROBOT_DATASET_DIR> \
  --policy.type=act \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --policy.chunk_size=1 \
  --policy.n_action_steps=1 \
  --batch_size=2 \
  --steps=1 \
  --eval_freq=0 \
  --save_freq=1 \
  --num_workers=0 \
  --output_dir=outputs/train/pusht_smoke
```

Minimal PushT smoke evaluation from a local policy checkpoint:

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/train/pusht_smoke/checkpoints/000001/pretrained_model \
  --policy.device=cpu \
  --env.type=pusht \
  --env.obs_type=environment_state_agent_pos \
  --env.episode_length=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval/pusht_smoke
```

## Training

The detect-and-correct pipeline trains an external latent dynamics corrector on frozen VLA visual features:

1. Fine-tune or obtain a VLA policy checkpoint for the benchmark.
2. Freeze the VLA and extract visual latents from demonstration trajectories.
3. Train the external corrector to predict short-horizon latent residuals.
4. Use the trained corrector as `--safety_model_path` during evaluation.

Latent extraction:

```bash
python -m siglip_dynamics.extract \
  --dataset-path <DATASET_DIR> \
  --dataset-repo-id <DATASET_REPO_ID> \
  --dataset-loader parquet \
  --dataset-format <metaworld_or_libero> \
  --output-path <EXTRACTED_CACHE_DIR> \
  --encoder-backend <pi05_or_smolvla_or_xvla> \
  --use-normalized-delta-action \
  --encoder-policy-path <POLICY_CHECKPOINT> \
  --encoder-local-files-only
```

Corrector training:

```bash
torchrun --nproc_per_node=1 -m siglip_dynamics.train \
  --model-type mlp \
  --h-window 1 \
  --k-step-list 10 \
  --dataset-path <EXTRACTED_CACHE_DIR> \
  --batch-size 512 \
  --epochs 30 \
  --train-loss-type cosine \
  --checkpoint-dir <CORRECTOR_CHECKPOINT>
```

Optional train-ratio sweep:

```bash
python -m siglip_dynamics.train_split_sweep \
  --dataset-path <EXTRACTED_CACHE_DIR> \
  --output-dir outputs/sweeps/siglip_dynamics \
  --model-types mlp transformer dit \
  --train-ratios 0.1 0.25 0.5 0.75 1.0 \
  --k-step 10 \
  --mlp-h-window 1 \
  --seq-h-window 5 \
  --epochs 30 \
  --train-loss-type both
```

The original LeRobot policy training entry point is retained:

```bash
python -m lerobot.scripts.lerobot_train --help
```

## Evaluation

PI0.5-style modified evaluation:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless

python -m lerobot.scripts.lerobot_eval_modified_detection \
  --policy.path=<POLICY_CHECKPOINT> \
  --policy.device=cuda \
  --policy.n_action_steps=50 \
  --policy.chunk_size=50 \
  --policy.compile_model=false \
  --env.type=metaworld \
  --env.task=<TASK_SPLIT> \
  --env.episode_length=300 \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --seed=1000 \
  --safety_model_path=<CORRECTOR_CHECKPOINT> \
  --safety_k=10 \
  --guidance_eta=1 \
  --guidance_apply_every=1 \
  --guidance_loss_objective=attract_delta_z_correction \
  --guidance_compare_baseline=true \
  --meltdown_cooldown_steps=10 \
  --output_dir=<OUTPUT_DIR> \
  --save_analysis=false \
  --save_raw_video=false \
  --save_summary_csv=true \
  --save_summary_json=true
```

SmolVLA and X-VLA use the same entry point with backbone-specific policy arguments. For example, SmolVLA may specify:

```bash
--policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

X-VLA may require observation-key remapping:

```bash
--rename_map='{"observation.image":"observation.images.image"}'
```

Full evaluation requires simulator dependencies, GPU resources, datasets, policy checkpoints, and trained corrector checkpoints. Missing checkpoints should produce explicit path errors rather than private-path assumptions.

## Results

The following results are summarized from the paper LaTeX draft. See the paper for full tables, task protocols, and appendix details.

### MetaWorld Cross-Architecture Evaluation

Success rate (%), averaged across difficulty splits:

| Backbone | Baseline Avg. | + VLA-Corrector Avg. | Absolute gain |
| --- | ---: | ---: | ---: |
| PI0.5 | 48.70 | 64.35 | +15.65 |
| SmolVLA | 61.90 | 66.65 | +4.75 |
| X-VLA | 55.55 | 59.60 | +4.05 |

The largest reported split-level gain is on the PI0.5 Very Hard split, from 41.0% to 65.0%.

### LIBERO Sample Efficiency

The paper reports that a PI0.5 few-shot fine-tuned model improves from 94.00% to 97.80% average success when augmented with VLA-Corrector, compared with 96.95% for the fully fine-tuned baseline in the reported setting.

### Policy-Call Efficiency

The paper reports positive success-per-call efficiency gains across PI0.5, SmolVLA, and X-VLA. The largest reported gains reach 29.9% for PI0.5, 45.3% for SmolVLA, and 39.1% for X-VLA.

### Real-World Evaluation

On an AgileX PiPER 6-DoF arm with PI0.5 as the backbone, the paper reports average success improving from 55.6% to 73.3% across three task groups. Gains are largest in disturbance recovery tasks, where the remaining action chunk is likely to become stale.

### Ablation and Analysis

- Truncation alone improves MetaWorld average success from 48.70% to 60.35%.
- Truncation plus OGG improves the average to 64.35%.
- 83.7% of truncations occur in manually labeled critical phases in the reported analysis.
- Increasing LVM capacity from 10M to 40M substantially improves success, while 160M provides almost no additional average gain in the reported setting.
- OGG introduces additional wall-clock inference cost and is applied only to recovery queries after interrupt events.

## Repository Structure

```text
.
├── src/lerobot/                 # LeRobot-based codebase and modified VLA policies
│   ├── scripts/                 # Training, evaluation, and modified detection entry points
│   ├── policies/pi05_modified/  # PI0.5 detect-and-correct wrapper
│   ├── policies/smolvla_modified/
│   ├── policies/xvla_modified/
│   └── safety/                  # Runtime dynamics predictor loader
├── src/siglip_dynamics/         # Latent extraction and corrector training
├── docs/                        # English GitHub Pages project page and technical notes
├── media/                       # Non-Pages media materials, including Chinese press copy
├── examples/                    # LeRobot examples retained from the base project
├── tests/                       # Source tests; large artifacts are excluded
├── environment.yml
└── requirements.txt
```

## GitHub Pages

The project page is served from the `/docs` directory via GitHub Pages.

To enable it:

```text
Settings -> Pages -> Build and deployment -> Source: Deploy from a branch
Branch: main
Folder: /docs
```

Expected URL:

```text
https://zju-omniai.github.io/vla-corrector/
```

## Citation

Paper and arXiv links are coming soon. Until a public paper entry is available, please cite this repository as:

```bibtex
@misc{pan2026vlacorrector,
  title        = {VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon},
  author       = {Pan, Yi and Pan, Miao and Lu, Qi and Huang, Jiaming and Zhang, Man and Zhang, Wenqi},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/ZJU-OmniAI/vla-corrector},
  note         = {Paper and arXiv coming soon}
}
```

## Acknowledgements

This project builds on Hugging Face LeRobot and uses components from Hugging Face Transformers and Datasets. The paper evaluates on MetaWorld, LIBERO, and a real AgileX PiPER setup, and studies PI0.5, SmolVLA, and X-VLA style VLA backbones. Please also follow the licenses and terms of the corresponding upstream projects, datasets, and model checkpoints.
