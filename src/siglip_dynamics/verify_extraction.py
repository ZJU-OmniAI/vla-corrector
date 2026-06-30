from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

try:
    from .extract import _build_extractor, _load_dataset
except ImportError:
    from extract import _build_extractor, _load_dataset


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _dequantize(z_q: np.ndarray, z_scale: np.ndarray) -> np.ndarray:
    return z_q.astype(np.float32) * z_scale.astype(np.float32)[:, None]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0.0:
        return 0.0
    return float(np.dot(af, bf) / denom)


def _resolve_backend_args(args: argparse.Namespace, meta: dict) -> argparse.Namespace:
    encoder_info = meta.get("encoder_info", {}) if isinstance(meta, dict) else {}

    backend = args.encoder_backend or str(meta.get("encoder_backend", "pi05"))

    local_pi0_config_name = args.local_pi0_config_name or str(encoder_info.get("local_pi0_config_name", ""))
    local_pi0_checkpoint_dir = args.local_pi0_checkpoint_dir or str(
        encoder_info.get("local_pi0_checkpoint_dir", "")
    )
    encoder_policy_path = args.encoder_policy_path or str(encoder_info.get("encoder_policy_path", ""))

    return argparse.Namespace(
        encoder_backend=backend,
        local_pi0_config_name=local_pi0_config_name,
        local_pi0_checkpoint_dir=local_pi0_checkpoint_dir,
        encoder_policy_path=encoder_policy_path,
        encoder_local_files_only=args.encoder_local_files_only,
        encoder_revision=args.encoder_revision,
        device=args.device,
    )


def _check_shapes(z_q: np.ndarray, z_scale: np.ndarray, actions: np.ndarray, ep_idx: np.ndarray) -> None:
    if z_q.shape[0] != z_scale.shape[0] or z_q.shape[0] != actions.shape[0] or z_q.shape[0] != ep_idx.shape[0]:
        raise ValueError(
            f"Frame count mismatch: z_q={z_q.shape[0]} z_scale={z_scale.shape[0]} "
            f"actions={actions.shape[0]} ep={ep_idx.shape[0]}"
        )
    if z_scale.shape[:2] != z_q.shape[:2]:
        raise ValueError(f"z_scale shape mismatch: z_scale={z_scale.shape} z_q={z_q.shape}")


def verify(args: argparse.Namespace) -> None:
    logger = logging.getLogger("siglip_dynamics.verify_extraction")
    extracted = Path(args.extracted_path)
    if not extracted.exists():
        raise FileNotFoundError(f"Extracted path does not exist: {extracted}")

    meta_path = extracted / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    z_q = np.load(extracted / "z_q.npy", mmap_mode="r")
    z_scale = np.load(extracted / "z_scale.npy", mmap_mode="r")
    actions = np.load(extracted / "actions.npy", mmap_mode="r")
    ep_idx = np.load(extracted / "episode_index.npy", mmap_mode="r")

    logger.info(
        "Cache summary: frames=%d L=%d D=%d action_dim=%d",
        z_q.shape[0],
        z_q.shape[1],
        z_q.shape[2],
        actions.shape[1],
    )
    logger.info(
        "Metadata: dataset_format=%s image_key=%s wrist_key=%s use_wrist=%s encoder_backend=%s",
        meta.get("dataset_format"),
        meta.get("image_key"),
        meta.get("wrist_image_key"),
        meta.get("use_wrist_image"),
        meta.get("encoder_backend"),
    )

    _check_shapes(z_q, z_scale, actions, ep_idx)

    # Optional strict check: recompute embeddings for a few frames and compare against cache.
    if args.num_samples <= 0:
        logger.info("Skip embedding recompute check (--num-samples <= 0).")
        return

    backend_args = _resolve_backend_args(args, meta)
    try:
        extractor = _build_extractor(backend_args, logger)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Skip embedding recompute check (cannot build extractor backend=%s): %s",
            backend_args.encoder_backend,
            e,
        )
        return

    ds, _kwargs = _load_dataset(args, logger)

    n = int(z_q.shape[0])
    count = min(max(1, args.num_samples), n)
    indices = np.linspace(0, n - 1, num=count, dtype=int).tolist()

    image_key = str(meta.get("image_key", "image"))
    wrist_key = str(meta.get("wrist_image_key", ""))
    use_wrist = bool(meta.get("use_wrist_image", False))

    cos_vals: list[float] = []
    mae_vals: list[float] = []
    for idx in indices:
        item = ds[idx]
        z_ref = extractor.encode_item(
            item,
            image_key=image_key,
            wrist_image_key=wrist_key,
            use_wrist_image=use_wrist,
        )
        z_cache = _dequantize(z_q[idx], z_scale[idx])
        cos = _cosine_similarity(z_ref, z_cache)
        mae = float(np.mean(np.abs(z_ref - z_cache)))
        cos_vals.append(cos)
        mae_vals.append(mae)
        logger.info("[sample=%d] cosine=%.6f mae=%.6f", idx, cos, mae)

    logger.info(
        "Recompute check done: backend=%s samples=%d | cosine(mean/min)=%.6f/%.6f | mae(mean/max)=%.6f/%.6f",
        backend_args.encoder_backend,
        len(indices),
        float(np.mean(cos_vals)),
        float(np.min(cos_vals)),
        float(np.mean(mae_vals)),
        float(np.max(mae_vals)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate extracted latent cache integrity and optional embedding consistency.")

    p.add_argument("--dataset-path", type=str, required=True, help="Source dataset path or repo_id (same as extract.py)")
    p.add_argument("--dataset-repo-id", type=str, default="", help="Optional explicit repo_id for local datasets")
    p.add_argument("--extracted-path", type=str, required=True, help="Path to extracted cache (metadata.json + npy files)")

    p.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="How many frames to recompute and compare. Set 0 to skip recompute.",
    )
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument(
        "--encoder-backend",
        type=str,
        choices=["pi05", "smolvla", "xvla"],
        default="",
        help="Override backend for recompute. Empty means read from metadata.json",
    )

    # PI05 backend args
    p.add_argument("--local-pi0-config-name", type=str, default="")
    p.add_argument("--local-pi0-checkpoint-dir", type=str, default="")

    # SmolVLA / XVLA backend args
    p.add_argument("--encoder-policy-path", type=str, default="")
    p.add_argument("--encoder-revision", type=str, default=None)
    p.add_argument("--encoder-local-files-only", action="store_true", default=False)

    return p


def main() -> None:
    _setup_logging()
    args = build_arg_parser().parse_args()
    verify(args)


if __name__ == "__main__":
    main()
