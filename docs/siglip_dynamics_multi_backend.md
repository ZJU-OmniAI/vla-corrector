# SigLIP Dynamics Multi-Backend Extraction and Training

This note documents the latent-dynamics extraction and corrector-training utilities used by VLA-Corrector. The code supports multiple VLA visual backbones and multiple robot benchmarks, while keeping generated caches and checkpoints outside the repository.

## Supported Backends

- Visual backends: `pi05`, `smolvla`, `xvla`
- Dataset families: `libero`, `metaworld`
- Corrector model types: `mlp`, `transformer`, `dit`

Use the Python environment defined by this repository. Do not run these scripts from an unrelated virtual environment, because the LeRobot policy wrappers and modified VLA modules must be importable.

## Preprocessing Alignment

The extraction script normalizes image layout, image scale, and visual-encoder inputs before computing frozen visual latents.

Implementation entry point:

```bash
python -m siglip_dynamics.extract
```

Relevant implementation areas:

- image layout conversion and value-range normalization in `src/siglip_dynamics/extract.py`;
- PI0.5 image preparation and resize-with-pad handling;
- SmolVLA image preparation and visual tower embedding extraction;
- X-VLA image preparation and visual embedding extraction;
- unified policy-path handling through `--encoder-policy-path`.

The extraction output is intended to match each backbone's expected preprocessing path while exposing a common latent-cache format for corrector training.

## Extraction

Recommended command shape:

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

Notes:

- `--encoder-policy-path` should point to the user-provided VLA checkpoint or local model directory.
- `--encoder-local-files-only` is useful when pretrained weights have already been downloaded.
- The output path should be outside tracked source directories, for example under `outputs/` or another ignored workspace.
- Datasets, extracted caches, and checkpoints are not included in the repository.

## Extraction Verification

Structure-only verification:

```bash
python -m siglip_dynamics.verify_extraction \
  --cache-dir <EXTRACTED_CACHE_DIR> \
  --check-structure-only
```

Strict verification can recompute embeddings and compare them with the stored cache:

```bash
python -m siglip_dynamics.verify_extraction \
  --dataset-path <DATASET_DIR> \
  --cache-dir <EXTRACTED_CACHE_DIR> \
  --encoder-backend <pi05_or_smolvla_or_xvla> \
  --encoder-policy-path <POLICY_CHECKPOINT>
```

## Corrector Training

The paper trains an external latent dynamics corrector after the VLA backbone has been obtained. The corrector is trained on frozen visual latents to predict short-horizon latent residuals.

Basic MLP training:

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

The training script can read metadata from the extracted cache to align latent and action dimensions automatically.

## Output Format

A typical extracted cache contains:

```text
metadata.json
z_q.npy
z_scale.npy
actions.npy
episode_index.npy
```

These files can be large and must not be committed. Keep them in ignored paths such as `outputs/`, external storage, or a local experiment directory.

## Runtime Use

After training, pass the corrector checkpoint directory to the modified evaluation entry point:

```bash
python -m lerobot.scripts.lerobot_eval_modified_detection \
  --policy.path=<POLICY_CHECKPOINT> \
  --safety_model_path=<CORRECTOR_CHECKPOINT> \
  --safety_k=10 \
  --guidance_eta=1 \
  --output_dir=<OUTPUT_DIR>
```

Full evaluation still requires benchmark-specific dependencies, datasets, policy checkpoints, and trained corrector checkpoints prepared by the user.
