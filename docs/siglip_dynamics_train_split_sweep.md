# Train-Ratio Sweep for the External Corrector

This note describes `siglip_dynamics.train_split_sweep`, which evaluates how the external latent dynamics corrector behaves as the amount of demonstration data changes. The script is intended for data-efficiency analysis and for producing aggregate metrics from repeated train-ratio experiments.

Implementation entry point:

```bash
python -m siglip_dynamics.train_split_sweep
```

## Experiment Logic

The script uses an episode-level split:

1. Shuffle all episodes with `split_seed`.
2. Hold out a fixed validation set.
3. Hold out a fixed test set.
4. Use the remaining episodes as the training pool.
5. For each `train_ratio`, train a new model from scratch using the corresponding fraction of the fixed training pool.
6. Evaluate every ratio on the same validation and test sets.
7. If multiple split seeds are provided, repeat the whole pipeline and report aggregated statistics.

This design avoids incremental training across ratios and makes the validation and test sets consistent for a fair data-efficiency comparison.

## Main Metrics

The most important test metrics are:

- mean cosine similarity between predicted and target latent residuals;
- mean squared error;
- aggregate statistics across split seeds when multiple seeds are used.

Cosine similarity closer to `1.0` indicates better directional alignment of the predicted latent residual.

## Outputs

The output directory usually contains:

```text
split_manifest.json
results.csv
summary_by_ratio.csv
test_tsne_subset.npz
test_tsne_points.csv
test_tsne.png
summary_plots/
```

Important files:

- `split_manifest.json`: records the train, validation, and test episode assignment for each split seed.
- `results.csv`: one row per individual run.
- `summary_by_ratio.csv`: aggregated mean and variance grouped by model type and train ratio.
- `test_tsne_subset.npz`: lightweight sampled predictions and targets for visualization.
- `test_predictions.npz`: optional full predictions, saved only when explicitly requested.

The script does not save full prediction tensors by default because they can be very large.

## Dependencies

Core training uses PyTorch and the dependencies already declared by this repository. Optional visualization uses packages such as `matplotlib` and `scikit-learn`. If optional visualization dependencies are unavailable, training and metric export can still run while plots are skipped.

## Common Parameters

- `--dataset-path`: extracted cache directory containing `metadata.json`, `z_q.npy`, `z_scale.npy`, `actions.npy`, and `episode_index.npy`.
- `--output-dir`: output directory for metrics and optional plots.
- `--model-types`: one or more model classes, such as `mlp transformer dit`.
- `--train-ratios`: fractions of the fixed training pool, for example `0.1 0.25 0.5 0.75 1.0`.
- `--val-ratio`: fixed validation ratio.
- `--test-ratio`: fixed test ratio.
- `--split-seeds`: one or more random seeds for repeated episode-level splits.
- `--k-step`: prediction interval for `t -> t+k`.
- `--mlp-h-window`: history window for the MLP model. The default MLP path expects `1`.
- `--seq-h-window`: history window for Transformer or DiT models.
- `--save-full-predictions-npz`: explicitly save full test predictions. This is off by default.

## Examples

Run an MLP-only train-ratio sweep:

```bash
python -m siglip_dynamics.train_split_sweep \
  --dataset-path <EXTRACTED_CACHE_DIR> \
  --output-dir outputs/sweeps/siglip_dynamics \
  --model-types mlp \
  --train-ratios 0.1 0.25 0.5 0.75 1.0 \
  --k-step 10 \
  --mlp-h-window 1 \
  --epochs 30 \
  --train-loss-type both
```

Run multiple model types:

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

Run repeated splits:

```bash
python -m siglip_dynamics.train_split_sweep \
  --dataset-path <EXTRACTED_CACHE_DIR> \
  --output-dir outputs/sweeps/siglip_dynamics \
  --model-types mlp \
  --train-ratios 0.2 0.4 0.6 0.8 1.0 \
  --split-seeds 42 123 999 \
  --k-step 10 \
  --mlp-h-window 1 \
  --epochs 30
```

## Interpreting Results

If the test cosine score improves and test MSE decreases as the train ratio grows, the corrector is benefiting from additional demonstration data. If the curve saturates early, the corrector may already have enough examples of local on-track latent dynamics for that setting. Large variance across split seeds suggests that the current data amount or split difficulty is unstable and should be analyzed before drawing conclusions.

The paper reports that corrector performance improves with additional data and then shows diminishing returns in the reported MetaWorld data-efficiency study.
