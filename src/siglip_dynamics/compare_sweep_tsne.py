from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for compare_sweep_tsne.py") from exc

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for compare_sweep_tsne.py") from exc

try:
    from .config import LossType, ModelScale, ModelType
    from .dataset import SiglipDynamicsDataset
    from .train_split_sweep import (
        IndexedSubsetDataset,
        _build_model,
        _forward,
        _indices_from_episode_ids,
        _make_loader,
        _sample_episode_ids,
    )
except ImportError:
    from config import LossType, ModelScale, ModelType
    from dataset import SiglipDynamicsDataset
    from train_split_sweep import (
        IndexedSubsetDataset,
        _build_model,
        _forward,
        _indices_from_episode_ids,
        _make_loader,
        _sample_episode_ids,
    )


LOGGER = logging.getLogger("siglip_dynamics.compare_sweep_tsne")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _pool_for_tsne(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 3:
        return x.mean(dim=1)
    return x


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gather_completed_runs(base_dir: Path, loss_tag: str) -> dict[tuple[str, int, float], dict[str, Any]]:
    out: dict[tuple[str, int, float], dict[str, Any]] = {}
    for metrics_path in sorted(base_dir.glob("**/result_metrics.json")):
        metrics = _load_json(metrics_path)
        run_dir = metrics_path.parent
        run_config_path = run_dir / "run_config.json"
        ckpt_path = run_dir / "best_model.pt"
        if not run_config_path.exists() or not ckpt_path.exists():
            continue
        key = (str(metrics["model_type"]), int(metrics["split_seed"]), float(metrics["train_ratio"]))
        out[key] = {
            "loss_tag": loss_tag,
            "base_dir": str(base_dir),
            "metrics_path": str(metrics_path),
            "run_config_path": str(run_config_path),
            "ckpt_path": str(ckpt_path),
            "metrics": metrics,
            "run_config": _load_json(run_config_path),
        }
    return out


def _build_loaded_model(run: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, ModelType]:
    run_cfg = run["run_config"]
    args = run_cfg["args"]
    model_type = ModelType(run_cfg["model_type"])
    model = _build_model(
        model_type=model_type,
        scale=ModelScale(args["scale"]),
        token_dim=int(run_cfg["token_dim"]),
        action_dim=int(run_cfg["action_dim"]),
        action_embed_dim=int(args["action_embed_dim"]),
        dropout=float(args["dropout"]),
        rope_theta=float(args["rope_theta"]),
        ada_rmsnorm_eps=float(args["ada_rmsnorm_eps"]),
    )
    ckpt = torch.load(run["ckpt_path"], map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, model_type


def _select_common_runs(
    mse_runs: dict[tuple[str, int, float], dict[str, Any]],
    cosine_runs: dict[tuple[str, int, float], dict[str, Any]],
    *,
    model_type: str,
    split_seed: int,
    ratios: list[float] | None,
) -> list[tuple[str, int, float]]:
    common = sorted(set(mse_runs).intersection(cosine_runs), key=lambda x: x[2])
    filtered = []
    ratio_filter = None if not ratios else {float(r) for r in ratios}
    for key in common:
        mt, seed, ratio = key
        if mt != model_type or seed != split_seed:
            continue
        if ratio_filter is not None and float(ratio) not in ratio_filter:
            continue
        filtered.append(key)
    return filtered


def _eval_two_models_for_ratio(
    mse_run: dict[str, Any],
    cosine_run: dict[str, Any],
    *,
    device: torch.device,
    num_workers: int,
    max_eval_samples: int,
    subset_seed: int,
) -> dict[str, np.ndarray]:
    mse_cfg = mse_run["run_config"]
    cosine_cfg = cosine_run["run_config"]
    mse_args = mse_cfg["args"]
    cosine_args = cosine_cfg["args"]

    dataset_path = Path(mse_args["dataset_path"])
    if str(dataset_path) != str(cosine_args["dataset_path"]):
        raise ValueError("Dataset path mismatch between mse and cosine runs.")
    if list(mse_cfg["test_episode_ids"]) != list(cosine_cfg["test_episode_ids"]):
        raise ValueError("Test episode ids mismatch between mse and cosine runs.")

    dataset = SiglipDynamicsDataset(
        str(dataset_path),
        k_step=int(mse_cfg["k_step"]),
        h_window=int(mse_cfg["h_window"]),
        max_samples=int(mse_args.get("max_samples", 0)),
    )
    sample_episode_ids = _sample_episode_ids(dataset)
    test_episode_ids = np.asarray(mse_cfg["test_episode_ids"], dtype=np.int64)
    test_indices = _indices_from_episode_ids(sample_episode_ids, test_episode_ids)
    if max_eval_samples > 0 and len(test_indices) > max_eval_samples:
        rng = np.random.default_rng(subset_seed)
        chosen = np.sort(rng.choice(np.asarray(test_indices, dtype=np.int64), size=max_eval_samples, replace=False))
        test_indices = chosen.astype(np.int64).tolist()
    test_ds = IndexedSubsetDataset(dataset, test_indices, sample_episode_ids)
    batch_size = int(mse_args.get("eval_batch_size", 256))
    loader = _make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, device=device)

    mse_model, mse_model_type = _build_loaded_model(mse_run, device)
    cosine_model, cosine_model_type = _build_loaded_model(cosine_run, device)

    gt_chunks: list[np.ndarray] = []
    mse_chunks: list[np.ndarray] = []
    cosine_chunks: list[np.ndarray] = []
    sample_index_chunks: list[np.ndarray] = []
    episode_id_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            mse_pred, delta_z = _forward(mse_model, mse_model_type, batch, device)
            cosine_pred, _ = _forward(cosine_model, cosine_model_type, batch, device)

            gt_chunks.append(_pool_for_tsne(delta_z).cpu().numpy().astype(np.float32))
            mse_chunks.append(_pool_for_tsne(mse_pred).cpu().numpy().astype(np.float32))
            cosine_chunks.append(_pool_for_tsne(cosine_pred).cpu().numpy().astype(np.float32))
            sample_index_chunks.append(batch["sample_index"].cpu().numpy().astype(np.int64))
            episode_id_chunks.append(batch["episode_id"].cpu().numpy().astype(np.int64))

    return {
        "target": np.concatenate(gt_chunks, axis=0),
        "pred_mse": np.concatenate(mse_chunks, axis=0),
        "pred_cosine": np.concatenate(cosine_chunks, axis=0),
        "sample_index": np.concatenate(sample_index_chunks, axis=0),
        "episode_id": np.concatenate(episode_id_chunks, axis=0),
    }


def _sample_common_rows(arrays: dict[str, np.ndarray], max_points: int, seed: int) -> dict[str, np.ndarray]:
    n = arrays["target"].shape[0]
    if max_points <= 0 or n <= max_points:
        return arrays
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return {k: v[idx] for k, v in arrays.items()}


def _compute_tsne(points: np.ndarray, seed: int) -> np.ndarray:
    joined = points
    if joined.shape[0] < 4:
        raise ValueError(f"Too few points for t-SNE: {joined.shape[0]}")
    if joined.shape[1] > 50 and joined.shape[0] > 10:
        pca_dim = min(50, joined.shape[0] - 1, joined.shape[1])
        joined = PCA(n_components=pca_dim, random_state=seed).fit_transform(joined)
    perplexity = min(30, max(5, (joined.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)
    return tsne.fit_transform(joined)


def _save_ratio_outputs(
    out_dir: Path,
    ratio: float,
    sampled: dict[str, np.ndarray],
    coords: np.ndarray,
    mse_metrics: dict[str, Any],
    cosine_metrics: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = sampled["target"].shape[0]
    labels = (
        ["Ground truth"] * n
        + ["Prediction (MSE loss)"] * n
        + ["Prediction (Cosine loss)"] * n
    )
    sample_index = np.concatenate([sampled["sample_index"]] * 3, axis=0)
    episode_id = np.concatenate([sampled["episode_id"]] * 3, axis=0)

    csv_path = out_dir / f"ratio_{ratio:g}_tsne_points.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "sample_index", "episode_id"])
        for i, ((x, y), label) in enumerate(zip(coords, labels, strict=True)):
            writer.writerow([float(x), float(y), label, int(sample_index[i]), int(episode_id[i])])

    colors = {
        "Ground truth": "#1f77b4",
        "Prediction (MSE loss)": "#ff7f0e",
        "Prediction (Cosine loss)": "#2ca02c",
    }
    plt.figure(figsize=(7.6, 6.4))
    start = 0
    for label in ["Ground truth", "Prediction (MSE loss)", "Prediction (Cosine loss)"]:
        end = start + n
        block = coords[start:end]
        plt.scatter(block[:, 0], block[:, 1], s=12, alpha=0.60, c=colors[label], label=label)
        start = end
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.18)
    plt.legend(frameon=True)
    plt.title(
        f"Train ratio={ratio:g} | "
        f"MSE cosine={mse_metrics['test_cosine_mean']:.3f}, "
        f"Cosine-loss cosine={cosine_metrics['test_cosine_mean']:.3f}"
    )
    plt.tight_layout()
    png_path = out_dir / f"ratio_{ratio:g}_tsne.png"
    pdf_path = out_dir / f"ratio_{ratio:g}_tsne.pdf"
    plt.savefig(png_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close()

    manifest = {
        "train_ratio": ratio,
        "sample_count_per_group": int(n),
        "mse_metrics": mse_metrics,
        "cosine_metrics": cosine_metrics,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }
    with open(out_dir / f"ratio_{ratio:g}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-ratio t-SNE comparisons between MSE-loss and cosine-loss sweeps.")
    parser.add_argument("--mse_dir", type=Path, required=True)
    parser.add_argument("--cosine_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_type", type=str, default="mlp")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs="*", default=None)
    parser.add_argument("--max_points_per_group", type=int, default=1500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--max_eval_samples", type=int, default=3000)
    return parser


def main() -> None:
    _setup_logging()
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)

    mse_runs = _gather_completed_runs(args.mse_dir, "mse_loss")
    cosine_runs = _gather_completed_runs(args.cosine_dir, "cosine_loss")
    common_keys = _select_common_runs(
        mse_runs,
        cosine_runs,
        model_type=args.model_type,
        split_seed=args.split_seed,
        ratios=args.ratios,
    )
    if not common_keys:
        raise RuntimeError("No common completed runs found between mse_dir and cosine_dir for the requested filters.")

    summary_rows = []
    for key in common_keys:
        _, _, ratio = key
        LOGGER.info("Processing ratio=%.3f", ratio)
        mse_run = mse_runs[key]
        cosine_run = cosine_runs[key]
        arrays = _eval_two_models_for_ratio(
            mse_run,
            cosine_run,
            device=device,
            num_workers=args.num_workers,
            max_eval_samples=args.max_eval_samples,
            subset_seed=args.tsne_seed + int(ratio * 1000),
        )
        sampled = _sample_common_rows(arrays, args.max_points_per_group, seed=args.tsne_seed + int(ratio * 1000))
        joined = np.concatenate([sampled["target"], sampled["pred_mse"], sampled["pred_cosine"]], axis=0)
        coords = _compute_tsne(joined, seed=args.tsne_seed + int(ratio * 1000))

        ratio_out_dir = args.output_dir / f"ratio_{ratio:g}"
        _save_ratio_outputs(
            ratio_out_dir,
            ratio,
            sampled,
            coords,
            mse_run["metrics"],
            cosine_run["metrics"],
        )
        summary_rows.append(
            {
                "train_ratio": ratio,
                "sample_count_per_group": int(sampled["target"].shape[0]),
                "mse_test_cosine_mean": mse_run["metrics"]["test_cosine_mean"],
                "cosine_test_cosine_mean": cosine_run["metrics"]["test_cosine_mean"],
                "mse_test_mse": mse_run["metrics"]["test_mse"],
                "cosine_test_mse": cosine_run["metrics"]["test_mse"],
                "png_path": str(ratio_out_dir / f"ratio_{ratio:g}_tsne.png"),
                "pdf_path": str(ratio_out_dir / f"ratio_{ratio:g}_tsne.pdf"),
            }
        )

    summary_csv = args.output_dir / "tsne_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "train_ratio",
                "sample_count_per_group",
                "mse_test_cosine_mean",
                "cosine_test_cosine_mean",
                "mse_test_mse",
                "cosine_test_mse",
                "png_path",
                "pdf_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(args.output_dir / "tsne_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    LOGGER.info("Finished. Wrote %s", summary_csv)


if __name__ == "__main__":
    main()
