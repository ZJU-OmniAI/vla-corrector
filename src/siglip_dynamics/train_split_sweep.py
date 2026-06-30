from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    from .config import LossType, ModelScale, ModelType, SiglipMLPConfig
    from .dataset import SiglipDynamicsDataset
    from .dit_transformer import SiglipDiTTransformer
    from .MLP import SiglipResidualMLP
    from .transformer import SiglipResidualTransformer
except ImportError:
    from config import LossType, ModelScale, ModelType, SiglipMLPConfig
    from dataset import SiglipDynamicsDataset
    from dit_transformer import SiglipDiTTransformer
    from MLP import SiglipResidualMLP
    from transformer import SiglipResidualTransformer


@dataclass
class RunSpec:
    model_type: str
    train_ratio: float
    h_window: int
    k_step: int


class IndexedSubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, sample_indices: list[int], sample_episode_ids: np.ndarray):
        self.base_dataset = base_dataset
        self.sample_indices = list(sample_indices)
        self.sample_episode_ids = sample_episode_ids

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base_idx = int(self.sample_indices[idx])
        item = dict(self.base_dataset[base_idx])
        item["sample_index"] = torch.tensor(base_idx, dtype=torch.int64)
        item["episode_id"] = torch.tensor(int(self.sample_episode_ids[base_idx]), dtype=torch.int64)
        return item


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_serializable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_serializable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_serializable(v) for v in x]
    if isinstance(x, (ModelScale, ModelType, LossType)):
        return x.value
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def _format_ratio_tag(ratio: float) -> str:
    return f"r{ratio:.3f}".rstrip("0").rstrip(".")


def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()


def _combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: LossType,
    cosine_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred, target)
    cos = _cosine_loss(pred, target)
    if loss_type == LossType.MSE:
        total = mse
    elif loss_type == LossType.COSINE:
        total = cos
    elif loss_type == LossType.BOTH:
        total = mse + cosine_loss_weight * cos
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    return total, mse, cos


def _flatten_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return F.cosine_similarity(pred_flat, target_flat, dim=-1)


def _build_model(
    *,
    model_type: ModelType,
    scale: ModelScale,
    token_dim: int,
    action_dim: int,
    action_embed_dim: int,
    dropout: float,
    rope_theta: float,
    ada_rmsnorm_eps: float,
) -> nn.Module:
    if model_type == ModelType.MLP:
        cfg = SiglipMLPConfig(
            token_dim=token_dim,
            action_dim=action_dim,
            action_embed_dim=action_embed_dim,
            dropout=dropout,
            rope_theta=rope_theta,
            ada_rmsnorm_eps=ada_rmsnorm_eps,
            scale=scale,
        )
        return SiglipResidualMLP(cfg)

    d_model = 768 if scale != ModelScale.M4 else 512
    n_heads = 12 if scale != ModelScale.M4 else 8
    n_layers = 4 if scale == ModelScale.M20 else (2 if scale == ModelScale.M4 else 8)

    if model_type == ModelType.TRANSFORMER:
        return SiglipResidualTransformer(
            token_dim=token_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    if model_type == ModelType.DIT:
        return SiglipDiTTransformer(
            token_dim=token_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            rope_theta=rope_theta,
            ada_rmsnorm_eps=ada_rmsnorm_eps,
        )

    raise ValueError(f"Unsupported model_type={model_type}")


def _forward(model: nn.Module, model_type: ModelType, batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    z_t = batch["z_t"].to(device, non_blocking=True)
    z_hist = batch["z_hist"].to(device, non_blocking=True)
    a_t = batch["a_t"].to(device, non_blocking=True)
    delta_z = batch["delta_z"].to(device, non_blocking=True)

    model_input = z_t if model_type == ModelType.MLP else z_hist
    pred = model(model_input, a_t)
    return pred, delta_z


def _resolve_dims_from_metadata(dataset_path: Path, args: argparse.Namespace) -> tuple[int, int]:
    token_dim = int(args.token_dim)
    action_dim = int(args.action_dim)
    meta_path = dataset_path / "metadata.json"
    if not meta_path.exists():
        return token_dim, action_dim

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    token_dim = int(meta.get("d_dim", token_dim))
    action_dim = int(meta.get("action_dim", action_dim))
    return token_dim, action_dim


def _sample_episode_ids(dataset: SiglipDynamicsDataset) -> np.ndarray:
    if getattr(dataset, "mode", "") == "quantized_frames":
        out = np.empty(len(dataset), dtype=np.int64)
        for i, (t, _t_next) in enumerate(dataset.pairs):
            out[i] = int(dataset.ep_idx[t])
        return out

    # Legacy triplets fallback: treat each sample as its own episode.
    return np.arange(len(dataset), dtype=np.int64)


def _episode_order(sample_episode_ids: np.ndarray, split_seed: int) -> np.ndarray:
    unique_eps = np.unique(sample_episode_ids)
    if unique_eps.size < 4:
        raise ValueError(f"Need at least 4 episodes for fixed train/val/test sweep, got {unique_eps.size}")

    rng = np.random.default_rng(split_seed)
    shuffled = unique_eps.copy()
    rng.shuffle(shuffled)
    return shuffled.astype(np.int64)


def _fixed_episode_split(
    episode_order: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    min_val_episodes: int,
    min_test_episodes: int,
) -> dict[str, np.ndarray]:
    total_eps = int(episode_order.size)
    if total_eps < 4:
        raise ValueError(f"Need at least 4 episodes, got {total_eps}")
    if min_test_episodes < 1:
        raise ValueError(f"min_test_episodes must be >= 1, got {min_test_episodes}")
    if min_val_episodes < 1:
        raise ValueError(f"min_val_episodes must be >= 1, got {min_val_episodes}")
    if min_test_episodes + min_val_episodes >= total_eps:
        raise ValueError(
            f"min_val_episodes + min_test_episodes leaves no room for training. "
            f"total_eps={total_eps}, min_val_episodes={min_val_episodes}, min_test_episodes={min_test_episodes}"
        )
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")

    val_count = max(min_val_episodes, int(round(total_eps * val_ratio)))
    test_count = max(min_test_episodes, int(round(total_eps * test_ratio)))
    if val_count + test_count >= total_eps:
        # Prefer preserving the requested test holdout, then shrink val.
        val_count = min(val_count, total_eps - test_count - 1)
    if val_count + test_count >= total_eps:
        raise ValueError(
            f"Fixed split leaves no room for train pool: total_eps={total_eps}, "
            f"val_count={val_count}, test_count={test_count}"
        )

    test_eps = episode_order[:test_count]
    val_eps = episode_order[test_count : test_count + val_count]
    train_pool_eps = episode_order[test_count + val_count :]
    if train_pool_eps.size < 1:
        raise ValueError("Fixed split produced zero training episodes.")

    return {
        "train_pool_episode_ids": train_pool_eps.astype(np.int64),
        "val_episode_ids": val_eps.astype(np.int64),
        "test_episode_ids": test_eps.astype(np.int64),
    }


def _train_pool_subset_for_ratio(
    train_pool_episode_ids: np.ndarray,
    *,
    train_ratio: float,
    min_train_episodes: int,
) -> np.ndarray:
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0,1], got {train_ratio}")
    total_train_pool = int(train_pool_episode_ids.size)
    if total_train_pool < min_train_episodes:
        raise ValueError(
            f"train_pool has too few episodes: total_train_pool={total_train_pool}, "
            f"min_train_episodes={min_train_episodes}"
        )
    requested = max(min_train_episodes, int(round(total_train_pool * train_ratio)))
    requested = min(requested, total_train_pool)
    return train_pool_episode_ids[:requested].astype(np.int64)


def _indices_from_episode_ids(sample_episode_ids: np.ndarray, episode_ids: np.ndarray) -> list[int]:
    mask = np.isin(sample_episode_ids, episode_ids)
    return np.flatnonzero(mask).astype(np.int64).tolist()


def _count_samples_per_episode(sample_episode_ids: np.ndarray, indices: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for idx in indices:
        ep = int(sample_episode_ids[idx])
        counts[ep] = counts.get(ep, 0) + 1
    return counts


def _train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_type: ModelType,
    loss_type: LossType,
    cosine_loss_weight: float,
    grad_clip_norm: float,
    device: torch.device,
    desc: str,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    mse_sum = 0.0
    cos_loss_sum = 0.0
    cos_sum = 0.0
    count = 0

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False) if tqdm is not None else loader
    for batch in pbar:
        pred, delta_z = _forward(model, model_type, batch, device)
        loss, mse, cos_loss = _combined_loss(pred, delta_z, loss_type, cosine_loss_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        bsz = int(delta_z.shape[0])
        loss_sum += float(loss.item()) * bsz
        mse_sum += float(mse.item()) * bsz
        cos_loss_sum += float(cos_loss.item()) * bsz
        cos_sum += float(_flatten_cosine(pred, delta_z).sum().item())
        count += bsz

        if tqdm is not None:
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", cos=f"{float((1.0 - cos_loss.item())):.4f}")

    denom = max(1, count)
    return {
        "loss": loss_sum / denom,
        "mse": mse_sum / denom,
        "cosine_loss": cos_loss_sum / denom,
        "cosine": cos_sum / denom,
    }


@torch.no_grad()
def _evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    model_type: ModelType,
    loss_type: LossType,
    cosine_loss_weight: float,
    device: torch.device,
    collect_full_predictions: bool,
    tsne_max_samples_per_group: int,
    tsne_seed: int,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    mse_sum = 0.0
    cos_loss_sum = 0.0
    cos_sum = 0.0
    pred_norm_sum = 0.0
    target_norm_sum = 0.0
    count = 0

    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    cosine_chunks: list[np.ndarray] = []
    sample_idx_chunks: list[np.ndarray] = []
    episode_id_chunks: list[np.ndarray] = []
    tsne_pred_samples: list[np.ndarray] = []
    tsne_target_samples: list[np.ndarray] = []
    tsne_sample_idx_samples: list[np.int64] = []
    tsne_episode_id_samples: list[np.int64] = []
    tsne_seen = 0
    tsne_rng = np.random.default_rng(tsne_seed)

    pbar = tqdm(loader, desc="eval", dynamic_ncols=True, leave=False) if tqdm is not None else loader
    for batch in pbar:
        pred, delta_z = _forward(model, model_type, batch, device)
        loss, mse, cos_loss = _combined_loss(pred, delta_z, loss_type, cosine_loss_weight)
        cosine_per_sample = _flatten_cosine(pred, delta_z)

        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = delta_z.reshape(delta_z.shape[0], -1)

        bsz = int(delta_z.shape[0])
        loss_sum += float(loss.item()) * bsz
        mse_sum += float(mse.item()) * bsz
        cos_loss_sum += float(cos_loss.item()) * bsz
        cos_sum += float(cosine_per_sample.sum().item())
        pred_norm_sum += float(torch.norm(pred_flat, dim=-1).sum().item())
        target_norm_sum += float(torch.norm(target_flat, dim=-1).sum().item())
        count += bsz

        if collect_full_predictions:
            pred_chunks.append(pred.detach().cpu().numpy().astype(np.float16))
            target_chunks.append(delta_z.detach().cpu().numpy().astype(np.float16))
            sample_idx_chunks.append(batch["sample_index"].detach().cpu().numpy().astype(np.int64))
            episode_id_chunks.append(batch["episode_id"].detach().cpu().numpy().astype(np.int64))

        cosine_chunks.append(cosine_per_sample.detach().cpu().numpy().astype(np.float32))

        if tsne_max_samples_per_group > 0:
            pred_for_tsne = pred.detach()
            target_for_tsne = delta_z.detach()
            if pred_for_tsne.ndim >= 3:
                pred_for_tsne = pred_for_tsne.mean(dim=1)
                target_for_tsne = target_for_tsne.mean(dim=1)

            pred_np = pred_for_tsne.cpu().numpy().astype(np.float16)
            target_np = target_for_tsne.cpu().numpy().astype(np.float16)
            sample_idx_np = batch["sample_index"].detach().cpu().numpy().astype(np.int64)
            episode_id_np = batch["episode_id"].detach().cpu().numpy().astype(np.int64)

            for i in range(pred_np.shape[0]):
                tsne_seen += 1
                if len(tsne_pred_samples) < tsne_max_samples_per_group:
                    tsne_pred_samples.append(pred_np[i])
                    tsne_target_samples.append(target_np[i])
                    tsne_sample_idx_samples.append(sample_idx_np[i])
                    tsne_episode_id_samples.append(episode_id_np[i])
                    continue

                replace_idx = int(tsne_rng.integers(0, tsne_seen))
                if replace_idx < tsne_max_samples_per_group:
                    tsne_pred_samples[replace_idx] = pred_np[i]
                    tsne_target_samples[replace_idx] = target_np[i]
                    tsne_sample_idx_samples[replace_idx] = sample_idx_np[i]
                    tsne_episode_id_samples[replace_idx] = episode_id_np[i]

    denom = max(1, count)
    out: dict[str, Any] = {
        "loss": loss_sum / denom,
        "mse": mse_sum / denom,
        "cosine_loss": cos_loss_sum / denom,
        "cosine": cos_sum / denom,
        "pred_norm": pred_norm_sum / denom,
        "target_norm": target_norm_sum / denom,
        "count": count,
    }

    cosine_arr = np.concatenate(cosine_chunks, axis=0) if cosine_chunks else np.empty((0,), dtype=np.float32)
    out.update(
        {
            "cosine_per_sample": cosine_arr,
            "cosine_median": float(np.median(cosine_arr)) if cosine_arr.size else float("nan"),
            "cosine_std": float(np.std(cosine_arr)) if cosine_arr.size else float("nan"),
            "cosine_p25": float(np.percentile(cosine_arr, 25)) if cosine_arr.size else float("nan"),
            "cosine_p75": float(np.percentile(cosine_arr, 75)) if cosine_arr.size else float("nan"),
        }
    )

    if collect_full_predictions:
        pred_arr = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.empty((0,), dtype=np.float16)
        target_arr = np.concatenate(target_chunks, axis=0) if target_chunks else np.empty((0,), dtype=np.float16)
        sample_idx_arr = np.concatenate(sample_idx_chunks, axis=0) if sample_idx_chunks else np.empty((0,), dtype=np.int64)
        episode_id_arr = np.concatenate(episode_id_chunks, axis=0) if episode_id_chunks else np.empty((0,), dtype=np.int64)
        out.update(
            {
                "pred": pred_arr,
                "target": target_arr,
                "sample_index": sample_idx_arr,
                "episode_id": episode_id_arr,
            }
        )

    if tsne_pred_samples:
        out.update(
            {
                "tsne_pred": np.stack(tsne_pred_samples, axis=0),
                "tsne_target": np.stack(tsne_target_samples, axis=0),
                "tsne_sample_index": np.asarray(tsne_sample_idx_samples, dtype=np.int64),
                "tsne_episode_id": np.asarray(tsne_episode_id_samples, dtype=np.int64),
            }
        )

    return out


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )


def _save_predictions_npz(run_dir: Path, eval_metrics: dict[str, Any]) -> Path:
    pred_path = run_dir / "test_predictions.npz"
    np.savez_compressed(
        pred_path,
        pred=eval_metrics["pred"],
        target=eval_metrics["target"],
        cosine_per_sample=eval_metrics["cosine_per_sample"],
        sample_index=eval_metrics["sample_index"],
        episode_id=eval_metrics["episode_id"],
    )
    return pred_path


def _save_tsne_subset_npz(run_dir: Path, eval_metrics: dict[str, Any]) -> Path:
    tsne_path = run_dir / "test_tsne_subset.npz"
    np.savez_compressed(
        tsne_path,
        pred=eval_metrics["tsne_pred"],
        target=eval_metrics["tsne_target"],
        sample_index=eval_metrics["tsne_sample_index"],
        episode_id=eval_metrics["tsne_episode_id"],
    )
    return tsne_path


def _plot_tsne(run_dir: Path, eval_metrics: dict[str, Any], max_samples_per_group: int, seed: int) -> dict[str, str]:
    logger = logging.getLogger("siglip_dynamics.train_split_sweep")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as e:
        logger.warning("Skip t-SNE plot because optional dependency is missing: %s", e)
        return {}

    pred = eval_metrics.get("tsne_pred")
    target = eval_metrics.get("tsne_target")
    if pred is None or target is None:
        pred = eval_metrics.get("pred")
        target = eval_metrics.get("target")
    if pred is None or target is None:
        logger.warning("Skip t-SNE plot because there are no evaluation predictions.")
        return {}

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    if pred.size == 0 or target.size == 0:
        logger.warning("Skip t-SNE plot because there are no evaluation predictions.")
        return {}

    # Use token-mean pooling before t-SNE to keep the representation meaningful
    # while making large [L,D] latent tensors tractable on CPU.
    if pred.ndim >= 3:
        pred_for_tsne = pred.mean(axis=1)
        target_for_tsne = target.mean(axis=1)
    else:
        pred_for_tsne = pred
        target_for_tsne = target

    pred_flat = pred_for_tsne.reshape(pred_for_tsne.shape[0], -1)
    target_flat = target_for_tsne.reshape(target_for_tsne.shape[0], -1)

    rng = np.random.default_rng(seed)

    def _sample_rows(x: np.ndarray) -> np.ndarray:
        if x.shape[0] <= max_samples_per_group or max_samples_per_group <= 0:
            return x
        idx = rng.choice(x.shape[0], size=max_samples_per_group, replace=False)
        return x[idx]

    pred_sampled = _sample_rows(pred_flat)
    target_sampled = _sample_rows(target_flat)
    joined = np.concatenate([target_sampled, pred_sampled], axis=0)
    labels = np.array(["ground_truth"] * target_sampled.shape[0] + ["prediction"] * pred_sampled.shape[0])

    if joined.shape[0] < 4:
        logger.warning("Skip t-SNE plot because sampled evaluation set is too small: %d", joined.shape[0])
        return {}

    if joined.shape[1] > 50 and joined.shape[0] > 10:
        pca_dim = min(50, joined.shape[0] - 1, joined.shape[1])
        joined = PCA(n_components=pca_dim, random_state=seed).fit_transform(joined)

    perplexity = min(30, max(5, (joined.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)
    coords = tsne.fit_transform(joined)

    png_path = run_dir / "test_tsne.png"
    csv_path = run_dir / "test_tsne_points.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label"])
        for (x, y), label in zip(coords, labels, strict=True):
            writer.writerow([float(x), float(y), label])

    plt.figure(figsize=(7.5, 6.2))
    for label, color in [("ground_truth", "#1f77b4"), ("prediction", "#d62728")]:
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.65, c=color, label=label)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.2)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()

    return {
        "tsne_png": str(png_path),
        "tsne_csv": str(csv_path),
        "tsne_repr": "token_mean_pool_then_flatten",
    }


def _plot_summary_curves(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    logger = logging.getLogger("siglip_dynamics.train_split_sweep")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning("Skip summary plots because matplotlib is missing: %s", e)
        return

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_model.setdefault(str(row["model_type"]), []).append(row)

    plot_dir = output_dir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for model_type, rows in by_model.items():
        rows = sorted(rows, key=lambda x: float(x["actual_train_pool_ratio_mean"]))
        ratios = [float(r["actual_train_pool_ratio_mean"]) for r in rows]
        mean_cos = [float(r["test_cosine_mean_mean"]) for r in rows]
        mean_cos_std = [float(r["test_cosine_mean_std"]) for r in rows]
        median_cos = [float(r["test_cosine_median_mean"]) for r in rows]
        mse_vals = [float(r["test_mse_mean"]) for r in rows]
        mse_std = [float(r["test_mse_std"]) for r in rows]

        plt.figure(figsize=(7.2, 5.0))
        plt.plot(ratios, mean_cos, marker="o", linewidth=2, label="test mean cosine")
        if any(v > 0 for v in mean_cos_std):
            lower = [m - s for m, s in zip(mean_cos, mean_cos_std, strict=True)]
            upper = [m + s for m, s in zip(mean_cos, mean_cos_std, strict=True)]
            plt.fill_between(ratios, lower, upper, alpha=0.18, label="mean cosine ± std")
        plt.plot(ratios, median_cos, marker="s", linewidth=2, label="test median cosine")
        plt.xlabel("Train ratio (fraction of fixed train pool)")
        plt.ylabel("Cosine similarity")
        plt.ylim(min(0.0, min(mean_cos + median_cos) - 0.05), min(1.0, max(mean_cos + median_cos) + 0.05))
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{model_type}_cosine_vs_ratio.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close()

        plt.figure(figsize=(7.2, 5.0))
        plt.plot(ratios, mse_vals, marker="o", linewidth=2, color="#ff7f0e")
        if any(v > 0 for v in mse_std):
            lower = [m - s for m, s in zip(mse_vals, mse_std, strict=True)]
            upper = [m + s for m, s in zip(mse_vals, mse_std, strict=True)]
            plt.fill_between(ratios, lower, upper, alpha=0.18, color="#ff7f0e")
        plt.xlabel("Train ratio (fraction of fixed train pool)")
        plt.ylabel("Test MSE")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{model_type}_mse_vs_ratio.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close()


def _run_single_experiment(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    token_dim: int,
    action_dim: int,
    sample_episode_ids: np.ndarray,
    split_manifest: dict[str, Any],
    spec: RunSpec,
    device: torch.device,
) -> dict[str, Any]:
    logger = logging.getLogger("siglip_dynamics.train_split_sweep")
    model_type = ModelType(spec.model_type)
    ratio_tag = _format_ratio_tag(spec.train_ratio)
    split_seed = int(split_manifest["split_seed"])
    run_dir = Path(args.output_dir) / model_type.value / f"seed_{split_seed}" / ratio_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    train_pool_eps = np.asarray(split_manifest["train_pool_episode_ids"], dtype=np.int64)
    val_eps = np.asarray(split_manifest["val_episode_ids"], dtype=np.int64)
    test_eps = np.asarray(split_manifest["test_episode_ids"], dtype=np.int64)
    total_episode_count = int(split_manifest["total_episode_count"])
    train_eps = _train_pool_subset_for_ratio(
        train_pool_eps,
        train_ratio=spec.train_ratio,
        min_train_episodes=args.min_train_episodes,
    )

    dataset = SiglipDynamicsDataset(
        str(dataset_path),
        k_step=spec.k_step,
        h_window=spec.h_window,
        max_samples=args.max_samples,
    )

    train_indices = _indices_from_episode_ids(sample_episode_ids, train_eps)
    val_indices = _indices_from_episode_ids(sample_episode_ids, val_eps)
    test_indices = _indices_from_episode_ids(sample_episode_ids, test_eps)

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            f"Empty split detected for model={model_type.value}, ratio={spec.train_ratio}: "
            f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}"
        )

    train_ds = IndexedSubsetDataset(dataset, train_indices, sample_episode_ids)
    val_ds = IndexedSubsetDataset(dataset, val_indices, sample_episode_ids)
    test_ds = IndexedSubsetDataset(dataset, test_indices, sample_episode_ids)

    train_loader = _make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = _make_loader(val_ds, args.eval_batch_size, False, args.num_workers, device)
    test_loader = _make_loader(test_ds, args.eval_batch_size, False, args.num_workers, device)

    model = _build_model(
        model_type=model_type,
        scale=ModelScale(args.scale),
        token_dim=token_dim,
        action_dim=action_dim,
        action_embed_dim=args.action_embed_dim,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        ada_rmsnorm_eps=args.ada_rmsnorm_eps,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    loss_type = LossType(args.train_loss_type)

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    no_improve = 0
    history: list[dict[str, Any]] = []

    logger.info(
        "[Run] model=%s ratio=%.3f h=%d k=%d | train_eps=%d val_eps=%d test_eps=%d | train=%d val=%d test=%d",
        model_type.value,
        spec.train_ratio,
        spec.h_window,
        spec.k_step,
        len(train_eps),
        len(val_eps),
        len(test_eps),
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            model_type=model_type,
            loss_type=loss_type,
            cosine_loss_weight=args.cosine_loss_weight,
            grad_clip_norm=args.grad_clip_norm,
            device=device,
            desc=f"train {model_type.value} {ratio_tag} e{epoch}",
        )
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            model_type=model_type,
            loss_type=loss_type,
            cosine_loss_weight=args.cosine_loss_weight,
            device=device,
            collect_full_predictions=False,
            tsne_max_samples_per_group=0,
            tsne_seed=args.tsne_seed,
        )
        scheduler.step()

        epoch_s = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_cosine": train_metrics["cosine"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_cosine": val_metrics["cosine"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_s,
        }
        history.append(row)
        logger.info(
            "[Epoch] model=%s ratio=%.3f epoch=%d/%d | train_loss=%.6f train_cos=%.6f | val_loss=%.6f val_cos=%.6f | %.1fs",
            model_type.value,
            spec.train_ratio,
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["cosine"],
            val_metrics["loss"],
            val_metrics["cosine"],
            epoch_s,
        )

        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            no_improve = 0
            best_state = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(
                    "[EarlyStop] model=%s ratio=%.3f epoch=%d best_epoch=%d best_val=%.6f",
                    model_type.value,
                    spec.train_ratio,
                    epoch,
                    best_epoch,
                    best_val,
                )
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state["state_dict"])
    eval_metrics = _evaluate(
        model=model,
        loader=test_loader,
        model_type=model_type,
        loss_type=loss_type,
        cosine_loss_weight=args.cosine_loss_weight,
        device=device,
        collect_full_predictions=bool(args.save_full_predictions_npz),
        tsne_max_samples_per_group=args.tsne_max_samples_per_group,
        tsne_seed=args.tsne_seed,
    )

    ckpt_path = run_dir / "best_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": model_type.value,
            "token_dim": token_dim,
            "action_dim": action_dim,
            "train_ratio": spec.train_ratio,
            "h_window": spec.h_window,
            "k_step": spec.k_step,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        ckpt_path,
    )

    predictions_path = ""
    if bool(args.save_full_predictions_npz):
        predictions_path = str(_save_predictions_npz(run_dir, eval_metrics))

    tsne_subset_path = ""
    if "tsne_pred" in eval_metrics and "tsne_target" in eval_metrics:
        tsne_subset_path = str(_save_tsne_subset_npz(run_dir, eval_metrics))

    tsne_outputs = _plot_tsne(run_dir, eval_metrics, args.tsne_max_samples_per_group, args.tsne_seed)

    run_config = {
        "args": _to_serializable(vars(args)),
        "split_seed": split_seed,
        "model_type": model_type.value,
        "train_ratio": spec.train_ratio,
        "actual_train_pool_ratio": float(len(train_eps) / max(1, len(train_pool_eps))),
        "actual_train_ratio": float(len(train_eps) / max(1, total_episode_count)),
        "actual_val_ratio": float(len(val_eps) / max(1, total_episode_count)),
        "actual_test_ratio": float(len(test_eps) / max(1, total_episode_count)),
        "h_window": spec.h_window,
        "k_step": spec.k_step,
        "token_dim": token_dim,
        "action_dim": action_dim,
        "train_pool_episode_ids": train_pool_eps,
        "train_episode_ids": train_eps,
        "val_episode_ids": val_eps,
        "test_episode_ids": test_eps,
        "train_pool_episode_sample_counts": _count_samples_per_episode(sample_episode_ids, _indices_from_episode_ids(sample_episode_ids, train_pool_eps)),
        "train_episode_sample_counts": _count_samples_per_episode(sample_episode_ids, train_indices),
        "val_episode_sample_counts": _count_samples_per_episode(sample_episode_ids, val_indices),
        "test_episode_sample_counts": _count_samples_per_episode(sample_episode_ids, test_indices),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    result = {
        "split_seed": split_seed,
        "model_type": model_type.value,
        "train_ratio": spec.train_ratio,
        "ratio_tag": ratio_tag,
        "actual_train_pool_ratio": float(len(train_eps) / max(1, len(train_pool_eps))),
        "actual_train_ratio": float(len(train_eps) / max(1, total_episode_count)),
        "actual_val_ratio": float(len(val_eps) / max(1, total_episode_count)),
        "actual_test_ratio": float(len(test_eps) / max(1, total_episode_count)),
        "h_window": spec.h_window,
        "k_step": spec.k_step,
        "train_pool_episode_count": int(len(train_pool_eps)),
        "train_episode_count": int(len(train_eps)),
        "val_episode_count": int(len(val_eps)),
        "test_episode_count": int(len(test_eps)),
        "train_sample_count": int(len(train_indices)),
        "val_sample_count": int(len(val_indices)),
        "test_sample_count": int(len(test_indices)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_loss": float(eval_metrics["loss"]),
        "test_mse": float(eval_metrics["mse"]),
        "test_cosine_loss": float(eval_metrics["cosine_loss"]),
        "test_cosine_mean": float(eval_metrics["cosine"]),
        "test_cosine_median": float(eval_metrics["cosine_median"]),
        "test_cosine_std": float(eval_metrics["cosine_std"]),
        "test_cosine_p25": float(eval_metrics["cosine_p25"]),
        "test_cosine_p75": float(eval_metrics["cosine_p75"]),
        "test_pred_norm": float(eval_metrics["pred_norm"]),
        "test_target_norm": float(eval_metrics["target_norm"]),
        "checkpoint_path": str(ckpt_path),
        "predictions_path": predictions_path,
        "tsne_subset_path": tsne_subset_path,
        "tsne_png": tsne_outputs.get("tsne_png", ""),
        "tsne_csv": tsne_outputs.get("tsne_csv", ""),
        "tsne_repr": tsne_outputs.get("tsne_repr", ""),
    }
    with open(run_dir / "result_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    logger.info(
        "[Done] model=%s ratio=%.3f | test_cos_mean=%.6f test_cos_median=%.6f test_mse=%.6f",
        model_type.value,
        spec.train_ratio,
        result["test_cosine_mean"],
        result["test_cosine_median"],
        result["test_mse"],
    )
    return result


def _write_summary(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: _to_serializable(v) for k, v in row.items()})


def _aggregate_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not summary_rows:
        return []

    group_keys = ["model_type", "train_ratio", "h_window", "k_step"]
    numeric_keys = [
        "actual_train_pool_ratio",
        "actual_train_ratio",
        "actual_val_ratio",
        "actual_test_ratio",
        "train_pool_episode_count",
        "train_episode_count",
        "val_episode_count",
        "test_episode_count",
        "train_sample_count",
        "val_sample_count",
        "test_sample_count",
        "best_epoch",
        "best_val_loss",
        "test_loss",
        "test_mse",
        "test_cosine_loss",
        "test_cosine_mean",
        "test_cosine_median",
        "test_cosine_std",
        "test_cosine_p25",
        "test_cosine_p75",
        "test_pred_norm",
        "test_target_norm",
    ]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        out = {k: v for k, v in zip(group_keys, key, strict=True)}
        out["num_seeds"] = len(rows)
        out["split_seeds"] = [int(r["split_seed"]) for r in rows]
        for metric in numeric_keys:
            vals = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std())
            out[f"{metric}_min"] = float(vals.min())
            out[f"{metric}_max"] = float(vals.max())
        aggregated.append(out)
    return aggregated


def _write_aggregated_summary(output_dir: Path, aggregated_rows: list[dict[str, Any]]) -> None:
    if not aggregated_rows:
        return
    summary_json = output_dir / "summary_aggregated.json"
    summary_csv = output_dir / "summary_aggregated.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(aggregated_rows, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    fieldnames = list(aggregated_rows[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow({k: _to_serializable(v) for k, v in row.items()})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep multiple train-data ratios on extracted LIBERO/MetaWorld latent caches, "
            "train mlp/transformer/dit on a fixed train pool with fixed validation/test splits, "
            "and save prediction-vs-ground-truth artifacts."
        )
    )
    p.add_argument("--dataset-path", type=str, required=True, help="Extracted cache directory containing metadata.json and *.npy")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints, summaries, predictions, and t-SNE plots")
    p.add_argument("--model-types", type=str, nargs="+", choices=["mlp", "transformer", "dit"], default=["mlp"])
    p.add_argument(
        "--train-ratios",
        type=float,
        nargs="+",
        required=True,
        help="Fractions of the fixed train pool to use, e.g. 0.1 0.25 0.5 1.0",
    )
    p.add_argument(
        "--val-episode-ratio",
        type=float,
        default=0.1,
        help="Episode fraction reserved as fixed validation set",
    )
    p.add_argument(
        "--test-episode-ratio",
        type=float,
        default=0.1,
        help="Episode fraction reserved as fixed held-out test set",
    )
    p.add_argument(
        "--min-val-episodes",
        type=int,
        default=1,
        help="Always keep at least this many episodes in the validation split",
    )
    p.add_argument(
        "--min-test-episodes",
        type=int,
        default=1,
        help="Always keep at least this many episodes in the test split",
    )
    p.add_argument(
        "--min-train-episodes",
        type=int,
        default=1,
        help="For each ratio, use at least this many episodes from the fixed train pool",
    )
    p.add_argument("--split-seeds", type=int, nargs="+", default=[42], help="Random seed list for repeated split experiments")

    p.add_argument("--k-step", type=int, default=10, help="Target horizon k for delta_z prediction")
    p.add_argument("--mlp-h-window", type=int, default=1, help="History length used by MLP (must stay 1)")
    p.add_argument("--seq-h-window", type=int, default=1, help="History length used by transformer and dit")
    p.add_argument("--max-samples", type=int, default=0, help="Optional cap on dataset pairs; 0 means use all")

    p.add_argument("--scale", type=str, choices=["4m", "20m", "100m", "custom"], default="20m")
    p.add_argument("--token-dim", type=int, default=2048, help="Fallback only; overwritten by metadata.json when available")
    p.add_argument("--action-dim", type=int, default=7, help="Fallback only; overwritten by metadata.json when available")
    p.add_argument("--action-embed-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--ada-rmsnorm-eps", type=float, default=1e-6)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--train-loss-type", type=str, choices=["mse", "cosine", "both"], default="both")
    p.add_argument("--cosine-loss-weight", type=float, default=1.0)

    p.add_argument("--tsne-max-samples-per-group", type=int, default=1500, help="Max GT / prediction points each for t-SNE")
    p.add_argument("--tsne-seed", type=int, default=123)
    p.add_argument(
        "--save-full-predictions-npz",
        action="store_true",
        help="Save full test_predictions.npz with all prediction/target tensors. Disabled by default because it can be very large.",
    )
    return p


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("siglip_dynamics.train_split_sweep")
    args = build_arg_parser().parse_args()

    for ratio in args.train_ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"Each train ratio must be in (0,1], got {ratio}")
    if not (0.0 < args.val_episode_ratio < 1.0):
        raise ValueError("--val-episode-ratio must be in (0,1)")
    if not (0.0 < args.test_episode_ratio < 1.0):
        raise ValueError("--test-episode-ratio must be in (0,1)")
    if args.val_episode_ratio + args.test_episode_ratio >= 1.0:
        raise ValueError("--val-episode-ratio + --test-episode-ratio must be < 1")
    if args.min_val_episodes < 1:
        raise ValueError("--min-val-episodes must be >= 1")
    if args.min_test_episodes < 1:
        raise ValueError("--min-test-episodes must be >= 1")
    if args.min_train_episodes < 1:
        raise ValueError("--min-train-episodes must be >= 1")
    if args.mlp_h_window != 1:
        raise ValueError("MLP only supports --mlp-h-window=1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    _set_seed(int(args.split_seeds[0]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    token_dim, action_dim = _resolve_dims_from_metadata(dataset_path, args)

    # Use the largest history requirement to build a split manifest once.
    max_h_window = max(args.mlp_h_window, args.seq_h_window)
    split_probe_dataset = SiglipDynamicsDataset(
        str(dataset_path),
        k_step=args.k_step,
        h_window=max_h_window,
        max_samples=args.max_samples,
    )
    sample_episode_ids = _sample_episode_ids(split_probe_dataset)
    summary_rows: list[dict[str, Any]] = []
    specs: list[RunSpec] = []
    for model_type in args.model_types:
        h_window = args.mlp_h_window if model_type == "mlp" else args.seq_h_window
        specs.extend(RunSpec(model_type=model_type, train_ratio=float(r), h_window=h_window, k_step=args.k_step) for r in args.train_ratios)

    split_manifests: list[dict[str, Any]] = []
    for split_seed in args.split_seeds:
        episode_order = _episode_order(sample_episode_ids, int(split_seed))
        fixed_split = _fixed_episode_split(
            episode_order,
            val_ratio=args.val_episode_ratio,
            test_ratio=args.test_episode_ratio,
            min_val_episodes=args.min_val_episodes,
            min_test_episodes=args.min_test_episodes,
        )
        split_manifests.append(
            {
                "split_seed": int(split_seed),
                "total_episode_count": int(episode_order.size),
                "episode_order": episode_order,
                **fixed_split,
            }
        )

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_path": str(dataset_path),
                "token_dim": token_dim,
                "action_dim": action_dim,
                "k_step": args.k_step,
                "max_h_window_for_split": max_h_window,
                "total_samples": int(len(split_probe_dataset)),
                "unique_episode_count": int(np.unique(sample_episode_ids).size),
                "val_episode_ratio": args.val_episode_ratio,
                "test_episode_ratio": args.test_episode_ratio,
                "min_val_episodes": args.min_val_episodes,
                "min_test_episodes": args.min_test_episodes,
                "min_train_episodes": args.min_train_episodes,
                "split_manifests": split_manifests,
            },
            f,
            ensure_ascii=False,
            indent=2,
            cls=NumpyEncoder,
        )

    logger.info("Planned runs: %d (model-ratio) x %d seed(s) = %d", len(specs), len(split_manifests), len(specs) * len(split_manifests))
    for split_manifest in split_manifests:
        for spec in specs:
            result = _run_single_experiment(
                args=args,
                dataset_path=dataset_path,
                token_dim=token_dim,
                action_dim=action_dim,
                sample_episode_ids=sample_episode_ids,
                split_manifest=split_manifest,
                spec=spec,
                device=device,
            )
            summary_rows.append(result)
            _write_summary(output_dir, summary_rows)

    aggregated_rows = _aggregate_summary(summary_rows)
    _write_aggregated_summary(output_dir, aggregated_rows)
    _plot_summary_curves(output_dir, aggregated_rows)
    logger.info("All runs completed. Summary saved to %s", output_dir)


if __name__ == "__main__":
    main()
