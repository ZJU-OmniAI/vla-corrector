from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lerobot>=0.8
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # legacy lerobot
    except ImportError:
        LeRobotDataset = None  # type: ignore[assignment]


# ---------------------------
# Generic helpers
# ---------------------------
def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _to_numpy(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_float_tensor(x: object, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(_to_numpy(x), dtype=torch.float32, device=device)


def _to_float_array_1d(x: object) -> np.ndarray:
    return _to_numpy(x).astype(np.float32).reshape(-1)


def _to_bchw_float(x: object, device: torch.device) -> torch.Tensor:
    """
    Convert image-like input to BCHW float32.
    Accepts HWC/CHW/BHWC/BCHW.
    """
    img = _to_float_tensor(x, device)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise ValueError(f"Expected image rank 3/4, got shape={tuple(img.shape)}")

    # If channels-last (BHWC), convert to BCHW.
    if img.shape[-1] in (1, 3):
        img = img.permute(0, 3, 1, 2)
    if img.shape[1] not in (1, 3):
        raise ValueError(f"Expected 1/3 channels after conversion, got shape={tuple(img.shape)}")
    return img


def _normalize_to_zero_one(img_bchw: torch.Tensor) -> torch.Tensor:
    min_v = float(torch.min(img_bchw).item()) if img_bchw.numel() else 0.0
    max_v = float(torch.max(img_bchw).item()) if img_bchw.numel() else 1.0

    if min_v < 0.0:
        # Already in [-1,1]
        if min_v >= -1.0 and max_v <= 1.0:
            img_bchw = (img_bchw + 1.0) * 0.5
        else:
            raise ValueError(f"Unexpected image value range [{min_v:.4f}, {max_v:.4f}]")
    elif max_v > 1.0:
        # Assume uint8-like range.
        img_bchw = img_bchw / 255.0

    return img_bchw.clamp(0.0, 1.0)


def _resolve_action_key(item: dict, preferred_key: str) -> str:
    if preferred_key and preferred_key in item:
        return preferred_key
    for alt in ("actions", "action"):
        if alt in item:
            return alt
    raise KeyError(f"action_key='{preferred_key}' not found in dataset sample keys={list(item.keys())}")


def _choose_key(
    item: dict,
    *,
    preferred: str,
    candidates: tuple[str, ...],
    required: bool,
    name: str,
) -> str:
    if preferred and preferred in item:
        return preferred
    for key in candidates:
        if key in item:
            return key
    if required:
        raise KeyError(
            f"Unable to resolve {name}. preferred='{preferred}', candidates={candidates}, keys={list(item.keys())}"
        )
    return ""


def _infer_dataset_format(first_item: dict) -> str:
    keys = set(first_item.keys())
    if "observation.image" in keys:
        return "metaworld"
    if "image" in keys or "wrist_image" in keys:
        return "libero"
    if "observation.images.image" in keys or "observation.images.image2" in keys:
        return "libero"
    return "libero"


def _resolve_layout(first_item: dict, args: argparse.Namespace) -> tuple[str, str, str, bool]:
    if args.dataset_format == "auto":
        dataset_format = _infer_dataset_format(first_item)
    else:
        dataset_format = args.dataset_format

    if dataset_format == "metaworld":
        image_key = _choose_key(
            first_item,
            preferred=args.image_key,
            candidates=(
                "observation.image",
                "image",
                "observation.images.image",
            ),
            required=True,
            name="image_key",
        )
        wrist_key = _choose_key(
            first_item,
            preferred=args.wrist_image_key,
            candidates=(
                "",
                "wrist_image",
                "observation.images.image2",
            ),
            required=False,
            name="wrist_image_key",
        )
        use_wrist = bool(args.use_wrist_image and wrist_key and (wrist_key in first_item))
        return dataset_format, image_key, wrist_key, use_wrist

    image_key = _choose_key(
        first_item,
        preferred=args.image_key,
        candidates=(
            "image",
            "observation.images.image",
            "observation.image",
        ),
        required=True,
        name="image_key",
    )
    wrist_key = _choose_key(
        first_item,
        preferred=args.wrist_image_key,
        candidates=(
            "wrist_image",
            "image2",
            "observation.images.image2",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ),
        required=False,
        name="wrist_image_key",
    )
    use_wrist = bool(args.use_wrist_image and wrist_key and (wrist_key in first_item))
    return dataset_format, image_key, wrist_key, use_wrist


def _filter_kwargs_by_signature(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


class LocalParquetDataset:
    """Lightweight local parquet dataset adapter to avoid remote sync side effects."""

    def __init__(self, dataset_path: str):
        from datasets import load_dataset

        root = Path(dataset_path)
        data_dir = root / "data" if (root / "data").exists() else root
        self._root = root
        self._data_dir = data_dir
        self._ds = load_dataset("parquet", data_dir=str(data_dir), split="train")

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self._ds[idx])
        # Convert PIL images to numpy arrays to match downstream preprocessing.
        for k, v in list(item.items()):
            if hasattr(v, "size") and hasattr(v, "mode"):
                item[k] = np.asarray(v)
        return item


def _looks_like_local_parquet_tree(ds_path: Path) -> bool:
    if not ds_path.exists():
        return False
    data_dir = ds_path / "data" if (ds_path / "data").exists() else ds_path
    return any(data_dir.rglob("file-*.parquet")) or any(data_dir.rglob("episode_*.parquet"))


def _load_dataset(args: argparse.Namespace, logger: logging.Logger):
    dataset_loader = str(getattr(args, "dataset_loader", "auto")).lower()
    allow_dataset_download = bool(getattr(args, "allow_dataset_download", False))
    ds_path = Path(args.dataset_path)

    # Prefer local parquet loading first to avoid implicit HF sync writing into source dataset dir.
    if dataset_loader in ("auto", "parquet") and _looks_like_local_parquet_tree(ds_path):
        try:
            ds = LocalParquetDataset(args.dataset_path)
            logger.info(
                "Dataset loaded via LocalParquetDataset(data_dir=%s)",
                str((ds_path / "data") if (ds_path / "data").exists() else ds_path),
            )
            return ds, {"loader": "local_parquet", "dataset_path": str(ds_path)}
        except Exception as e:  # noqa: BLE001
            if dataset_loader == "parquet":
                raise
            logger.warning("LocalParquetDataset load failed, fallback to LeRobotDataset: %s", e)

    if dataset_loader == "parquet":
        raise RuntimeError(
            f"--dataset-loader=parquet requested but local parquet loading failed for {args.dataset_path}"
        )

    if LeRobotDataset is None:
        raise RuntimeError("lerobot not installed in current env.")

    repo_hint = args.dataset_repo_id.strip()
    candidates: list[dict] = []

    if ds_path.exists():
        if repo_hint:
            candidates.append(
                {
                    "repo_id": repo_hint,
                    "root": str(ds_path),
                    "download_videos": False,
                }
            )
            candidates.append(
                {
                    "repo_id": repo_hint,
                    "root": str(ds_path.parent),
                    "download_videos": False,
                }
            )
        guessed_repo = ds_path.name
        candidates.append(
            {
                "repo_id": f"lerobot/{guessed_repo}",
                "root": str(ds_path),
                "download_videos": False,
            }
        )
        candidates.append(
            {
                "repo_id": guessed_repo,
                "root": str(ds_path.parent),
                "download_videos": False,
            }
        )
        candidates.append({"repo_id": str(ds_path), "download_videos": False})
    else:
        if repo_hint:
            candidates.append({"repo_id": repo_hint, "download_videos": False})
        candidates.append({"repo_id": args.dataset_path, "download_videos": False})

    dedup: list[dict] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for kw in candidates:
        filtered = _filter_kwargs_by_signature(LeRobotDataset, kw)
        if "repo_id" not in filtered:
            continue
        key = tuple(sorted((k, str(v)) for k, v in filtered.items()))
        if key not in seen:
            seen.add(key)
            dedup.append(filtered)

    errors: list[str] = []
    for kw in dedup:
        try:
            # Prevent accidental on-disk sync into source dataset directory unless explicitly allowed.
            if (not allow_dataset_download) and ds_path.exists():
                old_offline = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    ds = LeRobotDataset(**kw)
                finally:
                    if old_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = old_offline
            else:
                ds = LeRobotDataset(**kw)
            logger.info("Dataset loaded via LeRobotDataset(%s)", kw)
            return ds, kw
        except Exception as e:  # noqa: BLE001
            errors.append(f"{kw} -> {type(e).__name__}: {e}")

    joined = "\n".join(errors[-6:])
    raise RuntimeError(
        "Failed to load dataset with all candidate signatures.\n"
        f"dataset_path={args.dataset_path}, dataset_repo_id={args.dataset_repo_id}\n"
        f"Recent errors:\n{joined}"
    )


def _estimate_size_bytes(num_frames: int, l_tokens: int, d_dim: int, action_dim: int) -> int:
    # z_q int8 + z_scale fp16 + action fp16 + ep int32
    return (
        num_frames * l_tokens * d_dim
        + num_frames * l_tokens * 2
        + num_frames * action_dim * 2
        + num_frames * 4
    )


def _quantize_per_token(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # z: [L,D] float32 -> q: int8 [L,D], scale: float16 [L]
    absmax = np.max(np.abs(z), axis=1)
    scale = np.maximum(absmax / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(z / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def _load_action_quantiles_stats(
    *,
    dataset_path: str,
    dataset_repo_id: str,
    action_key: str,
    action_dim: int,
) -> tuple[dict[str, np.ndarray], str, str]:
    candidates: list[Path] = []
    ds_path = Path(dataset_path)
    if ds_path.exists():
        candidates.extend(
            [
                ds_path / "meta" / "stats.json",
                ds_path / "stats.json",
            ]
        )

    stats_path: Path | None = None
    stats_json: dict[str, Any] | None = None
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                stats_json = json.load(f)
            stats_path = p
            break

    if stats_json is None:
        raise FileNotFoundError(
            "Cannot find stats.json for quantile normalization. "
            f"Tried: {[str(p) for p in candidates]} | dataset_repo_id={dataset_repo_id!r}"
        )

    key_candidates = [k for k in [action_key, "action", "actions"] if k]
    chosen_key = ""
    chosen_val: dict[str, Any] | None = None
    for k in key_candidates:
        v = stats_json.get(k)
        if isinstance(v, dict):
            chosen_key = k
            chosen_val = v
            break
    if chosen_val is None:
        raise KeyError(
            f"Action stats not found in {stats_path}. "
            f"Tried keys={key_candidates}, available keys={list(stats_json.keys())[:20]}"
        )

    q01 = _to_float_array_1d(chosen_val.get("q01"))
    q99 = _to_float_array_1d(chosen_val.get("q99"))
    if q01.shape[0] != action_dim or q99.shape[0] != action_dim:
        raise ValueError(
            "Action quantile dims mismatch: "
            f"q01={q01.shape}, q99={q99.shape}, expected action_dim={action_dim}"
        )

    return {"q01": q01, "q99": q99}, str(stats_path), chosen_key


def _normalize_action_quantiles(action_raw: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = stats["q01"]
    q99 = stats["q99"]
    denom = np.maximum(q99 - q01, 1e-6)
    normalized = 2.0 * (action_raw - q01) / denom - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


# ---------------------------
# Extractor backends
# ---------------------------
def _ensure_openpi_importable() -> None:
    env_openpi_src = os.getenv("OPENPI_SRC", "").strip()
    candidates = [
        Path(env_openpi_src) if env_openpi_src else None,
    ]
    valid: list[str] = []
    for p in candidates:
        if p is None:
            continue
        ps = str(p)
        if p.exists() and ps not in valid:
            valid.append(ps)
    # Keep priority order stable: earlier candidates should appear earlier in sys.path.
    for ps in reversed(valid):
        if ps in sys.path:
            sys.path.remove(ps)
        sys.path.insert(0, ps)


def _ensure_lerobot_importable() -> None:
    env_lerobot_src = os.getenv("LEROBOT_SRC", "").strip()
    repo_src = Path(__file__).resolve().parents[1]
    candidates = [
        Path(env_lerobot_src) if env_lerobot_src else None,
        repo_src,
    ]
    valid: list[str] = []
    for p in candidates:
        if p is None:
            continue
        ps = str(p)
        if p.exists() and ps not in valid:
            valid.append(ps)
    for ps in reversed(valid):
        if ps in sys.path:
            sys.path.remove(ps)
        sys.path.insert(0, ps)

    # If a different/older lerobot package is already imported (e.g. lerobot.common only),
    # clear it so subsequent imports resolve from the configured source tree.
    loaded = sys.modules.get("lerobot")
    loaded_file = str(getattr(loaded, "__file__", "") or "")
    if loaded is not None and valid:
        if not any(loaded_file.startswith(v) for v in valid):
            for mod_name in list(sys.modules.keys()):
                if mod_name == "lerobot" or mod_name.startswith("lerobot."):
                    del sys.modules[mod_name]


class BaseVisualExtractor:
    backend_name = "base"

    def encode_item(self, item: dict, *, image_key: str, wrist_image_key: str, use_wrist_image: bool) -> np.ndarray:
        imgs = [item[image_key]]
        if use_wrist_image and wrist_image_key and wrist_image_key in item:
            imgs.append(item[wrist_image_key])

        feats: list[torch.Tensor] = []
        for raw_img in imgs:
            feats.append(self.encode_single_image(raw_img))
        z = torch.cat(feats, dim=1)  # [1,L,D]
        return z[0].to(dtype=torch.float32).cpu().numpy()

    def encode_single_image(self, img_like: object) -> torch.Tensor:
        raise NotImplementedError

    def backend_info(self) -> dict[str, Any]:
        return {"backend": self.backend_name}


class Pi05Extractor(BaseVisualExtractor):
    """
    PI05 SigLIP extractor based on lerobot PI05Policy.
    Preprocess matches lerobot PI05 image pipeline:
      BCHW/BHWC handling -> resize_with_pad(224) -> [0,1] to [-1,1]
    """

    backend_name = "pi05"

    def __init__(
        self,
        policy_path: str,
        *,
        device: str = "cuda",
        local_files_only: bool = False,
        revision: str | None = None,
    ):
        if not policy_path:
            raise ValueError("pi05 backend requires --encoder-policy-path")

        _ensure_lerobot_importable()
        try:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        except SyntaxError as e:
            raise RuntimeError(
                "Failed to import lerobot PI05Policy. Use lerobot conda env Python (>=3.12)."
            ) from e

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = PI05Policy.from_pretrained(
            policy_path,
            local_files_only=local_files_only,
            revision=revision,
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.model = self.policy.model
        self.model_dtype = next(self.model.parameters()).dtype
        self.image_resolution = tuple(getattr(self.policy.config, "image_resolution", (224, 224)))
        self.policy_path = policy_path

    def _resize_with_pad_zero_one(self, images_bhwc: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Aligned with lerobot PI05 modeling_pi05.resize_with_pad_torch behavior for float inputs in [0,1].
        if images_bhwc.shape[-1] <= 4:
            images = images_bhwc.permute(0, 3, 1, 2)
        else:
            images = images_bhwc

        _b, _c, cur_h, cur_w = images.shape
        ratio = max(cur_w / width, cur_h / height)
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)
        resized = F.interpolate(
            images,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        pad_h0, rem_h = divmod(height - resized_h, 2)
        pad_h1 = pad_h0 + rem_h
        pad_w0, rem_w = divmod(width - resized_w, 2)
        pad_w1 = pad_w0 + rem_w
        padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=0.0)
        return padded.permute(0, 2, 3, 1)

    def _prepare_image(self, img_like: object) -> torch.Tensor:
        img = _to_bchw_float(img_like, self.device)
        img = _normalize_to_zero_one(img)

        # PI05 preprocess: BCHW -> BHWC -> resize_with_pad -> [-1,1] -> BCHW
        img = img.permute(0, 2, 3, 1)
        target_h, target_w = self.image_resolution
        if img.shape[1:3] != (target_h, target_w):
            img = self._resize_with_pad_zero_one(img, target_h, target_w)
        img = img * 2.0 - 1.0
        img = img.permute(0, 3, 1, 2)
        return img.to(dtype=self.model_dtype)

    @torch.no_grad()
    def encode_single_image(self, img_like: object) -> torch.Tensor:
        x = self._prepare_image(img_like)
        return self.model.paligemma_with_expert.embed_image(x)

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "policy_path": self.policy_path,
            "loader": "lerobot.PI05Policy.from_pretrained",
            "preprocess": {
                "pipeline": "lerobot_pi05_style",
                "input_range": "[0,1]",
                "resize_with_pad": list(self.image_resolution),
                "output_range": "[-1,1]",
                "layout": "BCHW",
            },
            "official_refs": [
                "lerobot/src/lerobot/policies/pi05/modeling_pi05.py:_preprocess_images",
                "lerobot/src/lerobot/policies/pi05/modeling_pi05.py:resize_with_pad_torch",
                "lerobot/src/lerobot/policies/pi05/modeling_pi05.py:PI05Policy.from_pretrained",
            ],
        }


class SmolVLAExtractor(BaseVisualExtractor):
    """
    SmolVLA visual extractor.
    Uses official SmolVLA preprocessing and vision embed path:
      prepare_images: resize_with_pad(config.resize_imgs_with_padding, pad=0), [0,1] -> [-1,1]
      embed_image: SmolVLMWithExpertModel.embed_image
    """

    backend_name = "smolvla"

    def __init__(
        self,
        policy_path: str,
        *,
        device: str = "cuda",
        local_files_only: bool = False,
        revision: str | None = None,
    ):
        if not policy_path:
            raise ValueError("smolvla backend requires --encoder-policy-path")

        _ensure_lerobot_importable()
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad as smol_resize_with_pad
        except SyntaxError as e:
            raise RuntimeError(
                "Failed to import lerobot SmolVLA modules. Use lerobot conda env Python (>=3.12)."
            ) from e

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = SmolVLAPolicy.from_pretrained(
            policy_path,
            local_files_only=local_files_only,
            revision=revision,
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.model = self.policy.model
        self.resize_with_pad = smol_resize_with_pad
        self.model_dtype = next(self.model.parameters()).dtype
        self.policy_path = policy_path

    def _prepare_image(self, img_like: object) -> torch.Tensor:
        img = _to_bchw_float(img_like, self.device)
        img = _normalize_to_zero_one(img)

        resize_cfg = getattr(self.policy.config, "resize_imgs_with_padding", None)
        if resize_cfg is not None:
            img = self.resize_with_pad(img, *resize_cfg, pad_value=0)

        # Same as SmolVLAPolicy.prepare_images
        img = img * 2.0 - 1.0
        return img.to(dtype=self.model_dtype)

    @torch.no_grad()
    def encode_single_image(self, img_like: object) -> torch.Tensor:
        x = self._prepare_image(img_like)
        return self.model.vlm_with_expert.embed_image(x)

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "encoder_policy_path": self.policy_path,
            "preprocess": {
                "pipeline": "smolvla_prepare_images",
                "resize_with_pad": list(getattr(self.policy.config, "resize_imgs_with_padding", ()) or []),
                "pad_value": 0,
                "input_range": "[0,1]",
                "output_range": "[-1,1]",
                "layout": "BCHW",
            },
            "official_refs": [
                "lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:prepare_images",
                "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:embed_image",
            ],
        }


class XVLAExtractor(BaseVisualExtractor):
    """
    XVLA visual extractor.
    Uses XVLA processor + model path:
      xvla_image_to_float -> xvla_imagenet_normalize -> resize_with_pad(optional) -> vlm._encode_image
    """

    backend_name = "xvla"

    def __init__(
        self,
        policy_path: str,
        *,
        device: str = "cuda",
        local_files_only: bool = False,
        revision: str | None = None,
    ):
        if not policy_path:
            raise ValueError("xvla backend requires --encoder-policy-path")

        _ensure_lerobot_importable()
        try:
            from lerobot.datasets.factory import IMAGENET_STATS
            from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
            from lerobot.policies.xvla.modeling_xvla import resize_with_pad as xvla_resize_with_pad
        except SyntaxError as e:
            raise RuntimeError(
                "Failed to import lerobot XVLA modules. Use lerobot conda env Python (>=3.12)."
            ) from e

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = XVLAPolicy.from_pretrained(
            policy_path,
            local_files_only=local_files_only,
            revision=revision,
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.model = self.policy.model
        self.resize_with_pad = xvla_resize_with_pad
        self.imagenet_mean = torch.tensor(IMAGENET_STATS["mean"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor(IMAGENET_STATS["std"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.model_dtype = next(self.model.parameters()).dtype
        self.policy_path = policy_path

    def _prepare_image(self, img_like: object) -> torch.Tensor:
        img = _to_bchw_float(img_like, self.device)

        # XVLAImageToFloatProcessorStep semantics.
        max_v = float(torch.max(img).item()) if img.numel() else 1.0
        min_v = float(torch.min(img).item()) if img.numel() else 0.0
        if max_v > 1.0:
            if min_v < 0.0 or max_v > 255.0:
                raise ValueError(
                    f"XVLA preprocess expects image range in [0,255] or [0,1], got [{min_v:.4f}, {max_v:.4f}]"
                )
            img = img / 255.0
        img = img.clamp(0.0, 1.0)

        # XVLAImageNetNormalizeProcessorStep semantics.
        img = (img - self.imagenet_mean) / self.imagenet_std

        resize_cfg = getattr(self.policy.config, "resize_imgs_with_padding", None)
        if resize_cfg is not None:
            img = self.resize_with_pad(img, *resize_cfg, pad_value=0.0)

        return img.to(dtype=self.model_dtype)

    @torch.no_grad()
    def encode_single_image(self, img_like: object) -> torch.Tensor:
        x = self._prepare_image(img_like)
        return self.model.vlm._encode_image(x)

    def backend_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "encoder_policy_path": self.policy_path,
            "preprocess": {
                "pipeline": "xvla_image_to_float + xvla_imagenet_normalize + optional_resize_with_pad",
                "input_range": "[0,255] or [0,1]",
                "after_to_float": "[0,1]",
                "normalize": "imagenet(mean,std)",
                "resize_with_pad": list(getattr(self.policy.config, "resize_imgs_with_padding", ()) or []),
                "layout": "BCHW",
            },
            "official_refs": [
                "lerobot/src/lerobot/policies/xvla/processor_xvla.py:XVLAImageToFloatProcessorStep",
                "lerobot/src/lerobot/policies/xvla/processor_xvla.py:XVLAImageNetNormalizeProcessorStep",
                "lerobot/src/lerobot/policies/xvla/modeling_xvla.py:_prepare_images",
                "lerobot/src/lerobot/policies/xvla/modeling_xvla.py:forward_vlm",
            ],
        }


# Backward-compatible alias used by verify script and old callsites.
LocalSiglipExtractor = Pi05Extractor


def _build_extractor(args: argparse.Namespace, logger: logging.Logger) -> BaseVisualExtractor:
    backend = str(args.encoder_backend).lower()
    if backend == "pi05":
        logger.info("Using extractor backend: pi05")
        # 优先使用统一参数，兼容旧参数
        policy_path = args.encoder_policy_path or args.local_pi0_checkpoint_dir
        if not policy_path:
            raise ValueError("pi05 backend requires --encoder-policy-path or --local-pi0-checkpoint-dir")
        return Pi05Extractor(
            policy_path=policy_path,
            device=args.device,
            local_files_only=args.encoder_local_files_only,
            revision=args.encoder_revision,
        )

    if backend == "smolvla":
        logger.info("Using extractor backend: smolvla")
        return SmolVLAExtractor(
            policy_path=args.encoder_policy_path,
            device=args.device,
            local_files_only=args.encoder_local_files_only,
            revision=args.encoder_revision,
        )

    if backend == "xvla":
        logger.info("Using extractor backend: xvla")
        return XVLAExtractor(
            policy_path=args.encoder_policy_path,
            device=args.device,
            local_files_only=args.encoder_local_files_only,
            revision=args.encoder_revision,
        )

    raise ValueError(f"Unsupported encoder_backend={backend}")


# ---------------------------
# Main extraction
# ---------------------------
def extract(args: argparse.Namespace) -> None:
    logger = logging.getLogger("siglip_dynamics.extract")

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, ds_kwargs = _load_dataset(args, logger)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Empty dataset.")
    logger.info("Dataset loaded: %d frames", n)

    extractor = _build_extractor(args, logger)

    first = ds[0]
    dataset_format, image_key, wrist_image_key, use_wrist_image = _resolve_layout(first, args)
    action_key = _resolve_action_key(first, args.action_key)
    logger.info(
        "Resolved layout | format=%s image_key=%s wrist_image_key=%s use_wrist_image=%s action_key=%s",
        dataset_format,
        image_key,
        wrist_image_key if wrist_image_key else "<none>",
        use_wrist_image,
        action_key,
    )
    logger.info("Sample keys: %s", sorted(first.keys()))

    z0 = extractor.encode_item(
        first,
        image_key=image_key,
        wrist_image_key=wrist_image_key,
        use_wrist_image=use_wrist_image,
    )
    l_tokens, d_dim = int(z0.shape[0]), int(z0.shape[1])
    action_dim = int(_to_numpy(first[action_key]).shape[-1])

    action_stats = None
    action_stats_path = ""
    action_stats_key = ""
    if args.use_normalized_delta_action:
        action_stats, action_stats_path, action_stats_key = _load_action_quantiles_stats(
            dataset_path=args.dataset_path,
            dataset_repo_id=args.dataset_repo_id,
            action_key=action_key,
            action_dim=action_dim,
        )
        logger.info(
            "Using normalized actions (QUANTILES) | stats=%s | key=%s",
            action_stats_path,
            action_stats_key,
        )

    est = _estimate_size_bytes(n, l_tokens, d_dim, action_dim)
    logger.info(
        "Target cache shape: frames=%d L=%d D=%d action_dim=%d | estimated size=%.2f GB",
        n,
        l_tokens,
        d_dim,
        action_dim,
        est / 1e9,
    )
    if est > int(args.max_output_gb * 1e9):
        raise RuntimeError(
            f"Estimated cache size {est/1e9:.2f}GB exceeds max-output-gb={args.max_output_gb}. "
            "Use fewer cameras / lower dims / higher cap."
        )

    z_q = np.lib.format.open_memmap(out_dir / "z_q.npy", mode="w+", dtype=np.int8, shape=(n, l_tokens, d_dim))
    z_scale = np.lib.format.open_memmap(out_dir / "z_scale.npy", mode="w+", dtype=np.float16, shape=(n, l_tokens))
    actions = np.lib.format.open_memmap(out_dir / "actions.npy", mode="w+", dtype=np.float16, shape=(n, action_dim))
    episode_index = np.lib.format.open_memmap(out_dir / "episode_index.npy", mode="w+", dtype=np.int32, shape=(n,))

    for i in range(n):
        item = ds[i]
        z = extractor.encode_item(
            item,
            image_key=image_key,
            wrist_image_key=wrist_image_key,
            use_wrist_image=use_wrist_image,
        )
        q, s = _quantize_per_token(z)
        z_q[i] = q
        z_scale[i] = s

        act_key_i = _resolve_action_key(item, args.action_key)
        action_raw = _to_float_array_1d(item[act_key_i])
        ep_i = int(_to_numpy(item["episode_index"]).item())

        if args.use_normalized_delta_action:
            assert action_stats is not None
            # 官方做法：直接使用归一化后的绝对动作，不计算增量
            action_out = _normalize_action_quantiles(action_raw, action_stats)
        else:
            action_out = action_raw

        actions[i] = action_out.astype(np.float16)

        episode_index[i] = ep_i

        if (i + 1) % 5000 == 0:
            logger.info("Encoded %d / %d frames (%.1f%%)", i + 1, n, (i + 1) * 100.0 / n)

    z_q.flush()
    z_scale.flush()
    actions.flush()
    episode_index.flush()

    resolved_action_key = _resolve_action_key(first, args.action_key)
    if args.use_normalized_delta_action:
        action_representation = "normalized_action"
        action_delta_definition = ""
        action_normalization = {
            "mode": "quantile_q01_q99_to_-1_1",
            "stats_source": action_stats_path,
            "stats_key": action_stats_key,
            "clip": True,
        }
    else:
        action_representation = "raw_action"
        action_delta_definition = ""
        action_normalization = {"mode": "NONE"}

    metadata = {
        "format": "siglip_frame_cache_v2",
        "dataset_format": dataset_format,
        "dataset_path": args.dataset_path,
        "dataset_kwargs": ds_kwargs,
        "encoder_backend": args.encoder_backend,
        "encoder_info": extractor.backend_info(),
        "num_frames": n,
        "l_tokens": l_tokens,
        "d_dim": d_dim,
        "action_dim": action_dim,
        "dtype_z_q": "int8",
        "dtype_z_scale": "float16",
        "dtype_actions": "float16",
        "image_key": image_key,
        "wrist_image_key": wrist_image_key,
        "action_key": resolved_action_key,
        "action_representation": action_representation,
        "action_delta_definition": action_delta_definition,
        "action_normalization": action_normalization,
        "use_wrist_image": use_wrist_image,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Extraction complete: %s", out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract quantized visual-latent frame cache from LeRobot dataset.")

    p.add_argument("--dataset-path", type=str, required=True, help="Dataset path or repo_id")
    p.add_argument("--dataset-repo-id", type=str, default="", help="Optional explicit repo_id for local datasets")
    p.add_argument(
        "--dataset-loader",
        type=str,
        choices=["auto", "parquet", "lerobot"],
        default="auto",
        help="auto: prefer local parquet (no remote sync), fallback to LeRobotDataset",
    )
    p.add_argument(
        "--allow-dataset-download",
        action="store_true",
        default=False,
        help="Allow LeRobotDataset to auto-download missing files into dataset path",
    )
    p.add_argument(
        "--dataset-format",
        type=str,
        choices=["auto", "libero", "metaworld"],
        default="auto",
        help="auto: infer from sample keys; libero/metaworld: force a specific layout",
    )

    p.add_argument("--output-path", type=str, required=True, help="Output directory for extracted cache")
    p.add_argument("--max-output-gb", type=float, default=500.0, help="Abort if estimated size exceeds this cap")

    p.add_argument("--image-key", type=str, default="")
    p.add_argument("--wrist-image-key", type=str, default="")
    p.add_argument("--action-key", type=str, default="actions")
    p.add_argument(
        "--use-normalized-delta-action",
        action="store_true",
        default=False,
        help=(
            "Store quantile-normalized actions in actions.npy: "
            "normalize action with q01/q99 from dataset stats to [-1,1]. "
            "This matches the official lerobot preprocessing for training."
        ),
    )
    p.add_argument("--use-wrist-image", action="store_true", default=True)
    p.add_argument("--no-use-wrist-image", dest="use_wrist_image", action="store_false")

    p.add_argument(
        "--encoder-backend",
        type=str,
        choices=["pi05", "smolvla", "xvla"],
        default="pi05",
        help="Visual encoder backend used for Z_t extraction",
    )

    # PI05 backend args (kept names for backward compatibility)
    p.add_argument(
        "--local-pi0-config-name",
        type=str,
        default=os.getenv("LOCAL_PI0_CONFIG_NAME"),
        help="(optional) legacy arg; not required for lerobot PI05 loading",
    )
    p.add_argument(
        "--local-pi0-checkpoint-dir",
        type=str,
        default=os.getenv("LOCAL_PI0_CHECKPOINT_DIR"),
        help="PI05 pretrained_model directory OR model.safetensors path",
    )

    # SmolVLA / XVLA backend args
    p.add_argument(
        "--encoder-policy-path",
        type=str,
        default="",
        help="Policy path or HF model id for smolvla/xvla backend",
    )
    p.add_argument(
        "--encoder-revision",
        type=str,
        default=None,
        help="Optional HF revision for smolvla/xvla backend",
    )
    p.add_argument(
        "--encoder-local-files-only",
        action="store_true",
        default=False,
        help="Use local files only when loading smolvla/xvla policy",
    )

    p.add_argument("--device", type=str, default="cuda")
    return p


def main() -> None:
    _setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
