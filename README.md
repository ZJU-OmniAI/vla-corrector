# VLA Corrector

VLA Corrector is an event-triggered detect-then-correct framework for chunked vision-language-action policies. It monitors latent visual dynamics during rollout, detects when the current action chunk is drifting into an unsafe or inconsistent state, purges stale queued actions, and optionally applies local guidance during the next replan.

The code is built on top of LeRobot and adds modified evaluation/policy modules for PI05, SmolVLA, and XVLA, plus a lightweight SigLIP/VLM latent dynamics predictor used for online detection and guidance.

## Directory Structure

```text
.
├── src/lerobot/                 # LeRobot-based codebase and modified VLA policies
│   ├── scripts/                 # Evaluation and training entry points
│   ├── policies/pi05_modified/  # PI05 detect-and-correct policy wrapper
│   ├── policies/smolvla_modified/
│   ├── policies/xvla_modified/
│   └── safety/                  # Runtime dynamics predictor loader
├── src/siglip_dynamics/         # Latent extraction and safety predictor training
├── docs/                        # Extra notes for multi-backend extraction/training
├── examples/                    # LeRobot examples retained from the base project
├── tests/                       # Lightweight source tests; large artifacts are excluded
├── environment.yml
└── requirements.txt
```

## Installation

```bash
conda env create -f environment.yml
conda activate lerobot
python -m pip install -e . --no-build-isolation
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

The exported environment name is `lerobot`. You can edit the `name:` field in `environment.yml` before creation if that conflicts with an existing environment. `--no-build-isolation` uses the build tools already provided by the conda environment and avoids extra downloads during editable installation.

## Data And Weights

This repository does not include datasets, demo data, training outputs, pretrained model weights, or fine-tuned checkpoints.

Prepare the following paths yourself:

```text
<DATASET_DIR>             # Source LeRobot/MetaWorld/LIBERO dataset
<EXTRACTED_CACHE_DIR>     # Extracted latent cache from siglip_dynamics.extract
<POLICY_CHECKPOINT>       # Base or fine-tuned PI05/SmolVLA/XVLA policy checkpoint
<SAFETY_CHECKPOINT_DIR>   # Trained latent dynamics predictor checkpoint directory
```

Pretrained weights are not included in this repository. If you use Hugging Face model IDs, the underlying libraries may download them automatically. If you work offline, download them yourself and pass local paths with the command line arguments.

Known pretrained/backbone names used by the code include:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
google/paligemma-3b-pt-224
lerobot/fast-action-tokenizer
```

Fine-tuned checkpoints are not included in this repository. Please specify your own checkpoint path with `--policy.path` and `--safety_model_path`.

## Inference

Main entry point:

```bash
python -m lerobot.scripts.lerobot_eval_modified_detection
```

PI05 example:

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
  --env.task=medium \
  --env.episode_length=300 \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --seed=1000 \
  --safety_model_path=<SAFETY_CHECKPOINT_DIR> \
  --safety_k=10 \
  --guidance_eta=0.1 \
  --guidance_apply_every=3 \
  --guidance_loss_objective=attract_delta_z_correction \
  --guidance_compare_baseline=true \
  --meltdown_cooldown_steps=10 \
  --output_dir=outputs/eval/pi05_medium \
  --save_analysis=false \
  --save_raw_video=false \
  --save_summary_csv=true \
  --save_summary_json=true
```

SmolVLA uses the same entry point and adds the VLM path when needed:

```bash
python -m lerobot.scripts.lerobot_eval_modified_detection \
  --policy.path=<POLICY_CHECKPOINT> \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.device=cuda \
  --policy.n_action_steps=50 \
  --policy.chunk_size=50 \
  --env.type=metaworld \
  --env.task=medium \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --safety_model_path=<SAFETY_CHECKPOINT_DIR> \
  --safety_k=10 \
  --guidance_eta=1 \
  --guidance_apply_every=1 \
  --output_dir=outputs/eval/smolvla_medium
```

XVLA commonly needs an observation-key rename:

```bash
python -m lerobot.scripts.lerobot_eval_modified_detection \
  --policy.path=<POLICY_CHECKPOINT> \
  --policy.device=cuda \
  --policy.action_mode=auto \
  --policy.n_action_steps=32 \
  --policy.chunk_size=32 \
  --env.type=metaworld \
  --env.task=medium \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --rename_map='{"observation.image":"observation.images.image"}' \
  --safety_model_path=<SAFETY_CHECKPOINT_DIR> \
  --safety_k=10 \
  --guidance_eta=0.1 \
  --guidance_apply_every=3 \
  --output_dir=outputs/eval/xvla_medium
```

## Training

The detect-and-correct pipeline needs a latent dynamics predictor. A typical workflow is:

1. Extract latent visual features from a dataset and policy checkpoint.
2. Train an MLP/Transformer/DiT dynamics predictor on the extracted cache.
3. Use the trained predictor as `--safety_model_path` during evaluation.

Latent extraction:

```bash
python -m siglip_dynamics.extract \
  --dataset-path <DATASET_DIR> \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --output-path <EXTRACTED_CACHE_DIR> \
  --encoder-backend pi05 \
  --use-normalized-delta-action \
  --encoder-policy-path <POLICY_CHECKPOINT> \
  --encoder-local-files-only
```

Train the safety predictor:

```bash
torchrun --nproc_per_node=1 -m siglip_dynamics.train \
  --model-type mlp \
  --h-window 1 \
  --k-step-list 10 \
  --dataset-path <EXTRACTED_CACHE_DIR> \
  --batch-size 512 \
  --epochs 30 \
  --train-loss-type cosine \
  --checkpoint-dir <SAFETY_CHECKPOINT_DIR>
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

The original LeRobot policy training entry point is also retained:

```bash
python -m lerobot.scripts.lerobot_train --help
```

## Notes

- `eval.batch_size` must be `1` for `lerobot_eval_modified_detection`.
- `safety_history_frames` currently supports `1`.
- `policy.n_action_steps` must be positive and no larger than `policy.chunk_size`.
- Default generated files should go under `outputs/`, which is ignored by git.
- Full evaluation requires the target simulator dependencies, GPU resources, datasets, policy checkpoints, and safety predictor checkpoints.
- If an evaluation fails because a dataset, pretrained model, or fine-tuned checkpoint is missing, prepare that artifact locally and rerun with the corresponding argument.

## Acknowledgements

This project is based on Hugging Face LeRobot and uses components from Hugging Face Transformers, Datasets, and model checkpoints/backbones released by their respective authors. The added code focuses on latent-dynamics detection, action-queue meltdown handling, and event-triggered guidance for VLA policies.
