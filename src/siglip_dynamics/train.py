from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path

# Keep GPU visibility fixed to avoid occupying other users' devices.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
import wandb
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    from .config import LossType, ModelScale, ModelType, SiglipMLPConfig, TrainConfig
    from .dataset import SiglipDynamicsDataset
    from .dit_transformer import SiglipDiTTransformer
    from .MLP import SiglipResidualMLP
    from .transformer import SiglipResidualTransformer
except ImportError:
    from config import LossType, ModelScale, ModelType, SiglipMLPConfig, TrainConfig
    from dataset import SiglipDynamicsDataset
    from dit_transformer import SiglipDiTTransformer
    from MLP import SiglipResidualMLP
    from transformer import SiglipResidualTransformer


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _cuda_mem_gib(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.memory_allocated(device) / (1024**3)
    except Exception:
        return 0.0


def _make_progress_bar(total_steps: int, desc: str, *, enabled: bool):
    if (not enabled) or total_steps <= 0 or tqdm is None:
        return None
    return tqdm(
        total=total_steps,
        desc=desc,
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
    )


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _init_distributed_if_needed(cfg: TrainConfig) -> tuple[torch.device, int, int, int]:
    if not _is_distributed():
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        return device, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA. No CUDA device is available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    return device, rank, local_rank, world_size


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _reduce_sum(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()


def _combined_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_type: LossType, cosine_loss_weight: float
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


def _shape_of(x: object) -> str:
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, np.ndarray):
        return str(x.shape)
    if isinstance(x, (list, tuple)):
        return f"len={len(x)}"
    return str(type(x))


def _log_stage_shapes(logger: logging.Logger, stage: str, **kwargs: object) -> None:
    parts = [f"{k}={_shape_of(v)}" for k, v in kwargs.items()]
    logger.info("[SHAPE][%s] %s", stage, " | ".join(parts))


def _build_config(args: argparse.Namespace) -> TrainConfig:
    model_cfg = SiglipMLPConfig(
        token_dim=args.token_dim,
        action_dim=args.action_dim,
        action_embed_dim=args.action_embed_dim,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        ada_rmsnorm_eps=args.ada_rmsnorm_eps,
        scale=ModelScale(args.scale),
        custom_widths=tuple(args.custom_widths),
    )
    return TrainConfig(
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        model=model_cfg,
        model_type=ModelType(args.model_type),
        h_window=args.h_window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        train_loss_type=LossType(args.train_loss_type),
        cosine_loss_weight=args.cosine_loss_weight,
        log_shapes_every_epoch=args.log_shapes_every_epoch,
        k_step=args.k_step,
        max_samples=args.max_samples,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
    )


def _unique_int_values(values: list[int], default_value: int, *, name: str) -> list[int]:
    merged = list(values) if values else [default_value]
    out: list[int] = []
    seen: set[int] = set()
    for v in merged:
        if v < 1:
            raise ValueError(f"{name} must be >= 1, got {v}")
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _sanitize_tag_value(x: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(x))


def _run_tag(h_window: int, k_step: int) -> str:
    return f"h{_sanitize_tag_value(h_window)}_k{_sanitize_tag_value(k_step)}"


def _build_run_plan(args: argparse.Namespace) -> tuple[list[tuple[int, int]], bool]:
    h_values = _unique_int_values(args.h_window_list or [], args.h_window, name="h-window")
    k_values = _unique_int_values(args.k_step_list or [], args.k_step, name="k-step")
    plan = [(h, k) for h in h_values for k in k_values]
    grid_requested = bool(args.h_window_list) or bool(args.k_step_list)
    return plan, grid_requested


def train(cfg: TrainConfig) -> None:
    logger = logging.getLogger("siglip_dynamics.train")
    device, rank, local_rank, world_size = _init_distributed_if_needed(cfg)
    is_main = rank == 0
    # Keep seed deterministic but different across ranks.
    _set_seed(cfg.seed + rank)
    if cfg.h_window < 1:
        raise ValueError("h_window must be >= 1")
    if cfg.model_type == ModelType.MLP and cfg.h_window != 1:
        raise ValueError("MLP currently expects h_window=1. Use model_type=transformer for history input.")
    logger.info(
        "Train start | distributed=%s world_size=%d rank=%d local_rank=%d device=%s",
        _is_distributed(),
        world_size,
        rank,
        local_rank,
        device,
    )

    try:
        dataset_path = Path(cfg.dataset_path)
        if not cfg.dataset_path:
            raise ValueError("--dataset-path is required.")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        # If extracted cache metadata exists, align model dims automatically.
        meta_path = dataset_path / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                data_action_dim = int(meta.get("action_dim", 0) or 0)
                data_token_dim = int(meta.get("d_dim", 0) or 0)
                if data_action_dim > 0 and cfg.model.action_dim != data_action_dim:
                    logger.warning(
                        "Auto-align action_dim from metadata: %d -> %d",
                        cfg.model.action_dim,
                        data_action_dim,
                    )
                    cfg.model.action_dim = data_action_dim
                if data_token_dim > 0 and cfg.model.token_dim != data_token_dim:
                    logger.warning(
                        "Auto-align token_dim from metadata: %d -> %d",
                        cfg.model.token_dim,
                        data_token_dim,
                    )
                    cfg.model.token_dim = data_token_dim
                if is_main:
                    logger.info(
                        "Dataset metadata | format=%s action_dim=%s token_dim=%s hints(image_key=%s wrist_key=%s)",
                        meta.get("dataset_format", "<unknown>"),
                        meta.get("action_dim", "<missing>"),
                        meta.get("d_dim", "<missing>"),
                        meta.get("image_key", "<missing>"),
                        meta.get("wrist_image_key", "<missing>"),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to parse metadata.json for auto-dim alignment: %s", e)

        ds = SiglipDynamicsDataset(
            str(dataset_path),
            k_step=cfg.k_step,
            h_window=cfg.h_window,
            max_samples=cfg.max_samples,
        )

        total = len(ds)
        if total < 2:
            raise ValueError(f"Dataset too small: total={total}. Need at least 2 samples for train/val split.")
        raw_val_size = int(total * cfg.val_ratio)
        val_size = min(max(1, raw_val_size), total - 1)
        train_size = total - val_size
        train_ds, val_ds = random_split(
            ds, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed)
        )

        # For large cached tensor datasets, multiprocessing workers are prone to OOM kill
        # in constrained/containerized environments. Keep dataloader single-process by default
        # for stability; users can still increase workers later if needed.
        effective_workers = 0
        if cfg.num_workers > 0 and is_main:
            logger.warning(
                "Overriding num_workers=%d -> 0 for stability (avoid DataLoader worker SIGKILL/OOM).",
                cfg.num_workers,
            )
        use_pin_memory = torch.cuda.is_available() and device.type == "cuda"
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            if _is_distributed()
            else None
        )
        val_sampler = (
            DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            if _is_distributed()
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=effective_workers,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=effective_workers,
            pin_memory=use_pin_memory,
        )
        train_steps = len(train_loader)
        val_steps = len(val_loader)
        if train_steps == 0 or val_steps == 0:
            raise ValueError(
                f"Empty dataloader detected: train_steps={train_steps}, val_steps={val_steps}, "
                f"train_size={train_size}, val_size={val_size}, batch_size={cfg.batch_size}, world_size={world_size}"
            )
        if is_main:
            logger.info(
                "Data split | total=%d train=%d val=%d val_ratio=%.4f",
                total,
                train_size,
                val_size,
                cfg.val_ratio,
            )
            logger.info(
                "Dataloader | train_steps=%d val_steps=%d batch_size=%d workers=%d pin_memory=%s",
                train_steps,
                val_steps,
                cfg.batch_size,
                effective_workers,
                use_pin_memory,
            )
            if tqdm is None:
                logger.warning("tqdm is not installed; progress bars are disabled. Install `tqdm` to enable.")

        if cfg.model_type == ModelType.MLP:
            model = SiglipResidualMLP(cfg.model).to(device)
        elif cfg.model_type == ModelType.TRANSFORMER:
            model = SiglipResidualTransformer(
                token_dim=cfg.model.token_dim,
                action_dim=cfg.model.action_dim,
                d_model=768 if cfg.model.scale != ModelScale.M4 else 512,
                n_heads=12 if cfg.model.scale != ModelScale.M4 else 8,
                n_layers=4 if cfg.model.scale == ModelScale.M20 else (2 if cfg.model.scale == ModelScale.M4 else 8),
                dropout=cfg.model.dropout,
            ).to(device)
        elif cfg.model_type == ModelType.DIT:
            model = SiglipDiTTransformer(
                token_dim=cfg.model.token_dim,
                action_dim=cfg.model.action_dim,
                d_model=768 if cfg.model.scale != ModelScale.M4 else 512,
                n_heads=12 if cfg.model.scale != ModelScale.M4 else 8,
                n_layers=4 if cfg.model.scale == ModelScale.M20 else (2 if cfg.model.scale == ModelScale.M4 else 8),
                dropout=cfg.model.dropout,
                rope_theta=cfg.model.rope_theta,
                ada_rmsnorm_eps=cfg.model.ada_rmsnorm_eps,
            ).to(device)
        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")
        if is_main:
            logger.info("Model params: %.2fM", model.num_parameters() / 1e6)

        if _is_distributed():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and str(device).startswith("cuda"):
            # Single-process multi-GPU fallback for users not launching with torchrun.
            model = torch.nn.DataParallel(model)
            if is_main:
                logger.info("Using DataParallel across %d visible GPUs.", torch.cuda.device_count())

        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
        )

        ckpt_dir = Path(cfg.checkpoint_dir)
        if is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        best_val_mse = float("inf")
        best_val_cos_loss = float("inf")
        no_improve = 0
        if cfg.wandb_run_name:
            run_name = cfg.wandb_run_name
        else:
            run_name = f"{cfg.model_type.value}_{cfg.model.scale.value}_k{cfg.k_step}_h{cfg.h_window}"
        if is_main:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity if cfg.wandb_entity else None,
                group=cfg.wandb_group if cfg.wandb_group else None,
                name=run_name,
                config=_to_serializable(dataclasses.asdict(cfg)),
                reinit=True,
            )
            wandb.watch(model, log="gradients", log_freq=100)

        heartbeat_steps = max(1, int(os.environ.get("SIGLIP_TRAIN_HEARTBEAT_STEPS", "20")))
        heartbeat_secs = max(5.0, float(os.environ.get("SIGLIP_TRAIN_HEARTBEAT_SECS", "60")))
        data_wait_warn_s = max(1.0, float(os.environ.get("SIGLIP_DATA_WAIT_WARN_SECS", "90")))
        step_time_warn_s = max(1.0, float(os.environ.get("SIGLIP_STEP_WARN_SECS", "120")))

        for epoch in range(1, cfg.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_start_t = time.perf_counter()
            if is_main:
                logger.info("Epoch %d/%d start | train_steps=%d val_steps=%d", epoch, cfg.epochs, train_steps, val_steps)

            model.train()
            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_cos_loss_sum = 0.0
            train_cos_sum = 0.0
            train_pred_norm_sum = 0.0
            train_real_norm_sum = 0.0
            train_count = 0
            last_grad_norm = 0.0
            train_stage_start_t = time.perf_counter()
            last_train_hb_t = train_stage_start_t
            train_iter = iter(train_loader)
            train_pbar = _make_progress_bar(
                train_steps,
                f"train e{epoch}",
                enabled=is_main,
            )
            for step_idx in range(train_steps):
                fetch_start_t = time.perf_counter()
                batch = next(train_iter)
                data_wait_s = time.perf_counter() - fetch_start_t

                step_start_t = time.perf_counter()
                z_t = batch["z_t"].to(device, non_blocking=True)
                z_hist = batch["z_hist"].to(device, non_blocking=True)
                a_t = batch["a_t"].to(device, non_blocking=True)
                delta_z = batch["delta_z"].to(device, non_blocking=True)

                model_input = z_t if cfg.model_type == ModelType.MLP else z_hist
                pred = model(model_input, a_t)
                loss, mse_loss_t, cos_loss_t = _combined_loss(
                    pred, delta_z, cfg.train_loss_type, cfg.cosine_loss_weight
                )
                if is_main and step_idx == 0 and (cfg.log_shapes_every_epoch or epoch == 1):
                    _log_stage_shapes(
                        logger,
                        f"train/epoch_{epoch}",
                        z_t=z_t,
                        z_hist=z_hist,
                        model_input=model_input,
                        a_t=a_t,
                        delta_z=delta_z,
                        pred=pred,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                last_grad_norm = float(grad_norm)
                step_time_s = time.perf_counter() - step_start_t

                bsz = int(z_t.shape[0])
                train_count += bsz
                train_loss_sum += float(loss.item()) * bsz
                train_mse_sum += float(mse_loss_t.item()) * bsz
                train_cos_loss_sum += float(cos_loss_t.item()) * bsz
                pred_flat = pred.reshape(pred.shape[0], -1)
                real_flat = delta_z.reshape(delta_z.shape[0], -1)
                train_cos_sum += float(F.cosine_similarity(pred_flat, real_flat, dim=-1).sum().item())
                train_pred_norm_sum += float(torch.norm(pred_flat, dim=-1).sum().item())
                train_real_norm_sum += float(torch.norm(real_flat, dim=-1).sum().item())
                cur_step = step_idx + 1

                if train_pbar is not None:
                    train_pbar.update(1)
                    train_pbar.set_postfix(
                        loss=f"{float(loss.item()):.4f}",
                        mse=f"{float(mse_loss_t.item()):.4f}",
                        cos=f"{float(cos_loss_t.item()):.4f}",
                        wait=f"{data_wait_s:.2f}s",
                        step=f"{step_time_s:.2f}s",
                    )
                if is_main and data_wait_s >= data_wait_warn_s:
                    logger.warning(
                        "[STALL?][train] epoch=%d step=%d/%d data_wait=%.2fs >= %.2fs",
                        epoch,
                        cur_step,
                        train_steps,
                        data_wait_s,
                        data_wait_warn_s,
                    )
                if is_main and step_time_s >= step_time_warn_s:
                    logger.warning(
                        "[STALL?][train] epoch=%d step=%d/%d step_time=%.2fs >= %.2fs",
                        epoch,
                        cur_step,
                        train_steps,
                        step_time_s,
                        step_time_warn_s,
                    )
                now_t = time.perf_counter()
                if is_main and (
                    cur_step == train_steps
                    or cur_step % heartbeat_steps == 0
                    or (now_t - last_train_hb_t) >= heartbeat_secs
                ):
                    mem_gib = _cuda_mem_gib(device)
                    logger.info(
                        "[HB][train] epoch=%d step=%d/%d loss=%.6f mse=%.6f cos_loss=%.6f "
                        "data_wait=%.3fs step=%.3fs grad=%.4f mem=%.2fGiB",
                        epoch,
                        cur_step,
                        train_steps,
                        float(loss.item()),
                        float(mse_loss_t.item()),
                        float(cos_loss_t.item()),
                        data_wait_s,
                        step_time_s,
                        last_grad_norm,
                        mem_gib,
                    )
                    last_train_hb_t = now_t
            if train_pbar is not None:
                train_pbar.close()
            train_stage_s = time.perf_counter() - train_stage_start_t

            train_loss = _reduce_sum(train_loss_sum, device)
            train_mse = _reduce_sum(train_mse_sum, device)
            train_cos_loss = _reduce_sum(train_cos_loss_sum, device)
            train_cos = _reduce_sum(train_cos_sum, device)
            train_pred_norm = _reduce_sum(train_pred_norm_sum, device)
            train_real_norm = _reduce_sum(train_real_norm_sum, device)
            train_count_total = _reduce_sum(float(train_count), device)
            denom_train = max(1.0, train_count_total)
            train_loss /= denom_train
            train_mse /= denom_train
            train_cos_loss /= denom_train
            train_cos /= denom_train
            train_pred_norm /= denom_train
            train_real_norm /= denom_train

            model.eval()
            val_loss_sum = 0.0
            val_mse_sum = 0.0
            val_cos_loss_sum = 0.0
            val_cos_sum = 0.0
            val_pred_norm_sum = 0.0
            val_real_norm_sum = 0.0
            val_count = 0
            val_stage_start_t = time.perf_counter()
            last_val_hb_t = val_stage_start_t
            val_iter = iter(val_loader)
            val_pbar = _make_progress_bar(
                val_steps,
                f"val   e{epoch}",
                enabled=is_main,
            )
            with torch.no_grad():
                for step_idx in range(val_steps):
                    fetch_start_t = time.perf_counter()
                    batch = next(val_iter)
                    data_wait_s = time.perf_counter() - fetch_start_t

                    step_start_t = time.perf_counter()
                    z_t = batch["z_t"].to(device, non_blocking=True)
                    z_hist = batch["z_hist"].to(device, non_blocking=True)
                    a_t = batch["a_t"].to(device, non_blocking=True)
                    delta_z = batch["delta_z"].to(device, non_blocking=True)
                    model_input = z_t if cfg.model_type == ModelType.MLP else z_hist
                    pred = model(model_input, a_t)
                    val_loss_t, val_mse_t, val_cos_loss_t = _combined_loss(
                        pred, delta_z, cfg.train_loss_type, cfg.cosine_loss_weight
                    )
                    if is_main and step_idx == 0 and (cfg.log_shapes_every_epoch or epoch == 1):
                        _log_stage_shapes(
                            logger,
                            f"val/epoch_{epoch}",
                            z_t=z_t,
                            z_hist=z_hist,
                            model_input=model_input,
                            a_t=a_t,
                            delta_z=delta_z,
                            pred=pred,
                        )
                    bsz = int(z_t.shape[0])
                    val_count += bsz
                    val_loss_sum += float(val_loss_t.item()) * bsz
                    val_mse_sum += float(val_mse_t.item()) * bsz
                    val_cos_loss_sum += float(val_cos_loss_t.item()) * bsz
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    real_flat = delta_z.reshape(delta_z.shape[0], -1)
                    val_cos_sum += float(F.cosine_similarity(pred_flat, real_flat, dim=-1).sum().item())
                    val_pred_norm_sum += float(torch.norm(pred_flat, dim=-1).sum().item())
                    val_real_norm_sum += float(torch.norm(real_flat, dim=-1).sum().item())
                    step_time_s = time.perf_counter() - step_start_t
                    cur_step = step_idx + 1

                    if val_pbar is not None:
                        val_pbar.update(1)
                        val_pbar.set_postfix(
                            loss=f"{float(val_loss_t.item()):.4f}",
                            wait=f"{data_wait_s:.2f}s",
                            step=f"{step_time_s:.2f}s",
                        )
                    if is_main and data_wait_s >= data_wait_warn_s:
                        logger.warning(
                            "[STALL?][val] epoch=%d step=%d/%d data_wait=%.2fs >= %.2fs",
                            epoch,
                            cur_step,
                            val_steps,
                            data_wait_s,
                            data_wait_warn_s,
                        )
                    if is_main and step_time_s >= step_time_warn_s:
                        logger.warning(
                            "[STALL?][val] epoch=%d step=%d/%d step_time=%.2fs >= %.2fs",
                            epoch,
                            cur_step,
                            val_steps,
                            step_time_s,
                            step_time_warn_s,
                        )
                    now_t = time.perf_counter()
                    if is_main and (
                        cur_step == val_steps
                        or cur_step % heartbeat_steps == 0
                        or (now_t - last_val_hb_t) >= heartbeat_secs
                    ):
                        logger.info(
                            "[HB][val] epoch=%d step=%d/%d loss=%.6f mse=%.6f cos_loss=%.6f "
                            "data_wait=%.3fs step=%.3fs",
                            epoch,
                            cur_step,
                            val_steps,
                            float(val_loss_t.item()),
                            float(val_mse_t.item()),
                            float(val_cos_loss_t.item()),
                            data_wait_s,
                            step_time_s,
                        )
                        last_val_hb_t = now_t
            if val_pbar is not None:
                val_pbar.close()
            val_stage_s = time.perf_counter() - val_stage_start_t
            epoch_s = time.perf_counter() - epoch_start_t

            val_loss = _reduce_sum(val_loss_sum, device)
            val_mse = _reduce_sum(val_mse_sum, device)
            val_cos_loss = _reduce_sum(val_cos_loss_sum, device)
            val_cos = _reduce_sum(val_cos_sum, device)
            val_pred_norm = _reduce_sum(val_pred_norm_sum, device)
            val_real_norm = _reduce_sum(val_real_norm_sum, device)
            val_count_total = _reduce_sum(float(val_count), device)
            denom_val = max(1.0, val_count_total)
            val_loss /= denom_val
            val_mse /= denom_val
            val_cos_loss /= denom_val
            val_cos /= denom_val
            val_pred_norm /= denom_val
            val_real_norm /= denom_val

            if is_main:
                logger.info(
                    "Epoch %d done | train_total=%.6f train_mse=%.6f train_cos_loss=%.6f "
                    "| val_total=%.6f val_mse=%.6f val_cos_loss=%.6f | train_cos=%.4f val_cos=%.4f "
                    "| time(train/val/epoch)=%.1fs/%.1fs/%.1fs",
                    epoch,
                    train_loss,
                    train_mse,
                    train_cos_loss,
                    val_loss,
                    val_mse,
                    val_cos_loss,
                    train_cos,
                    val_cos,
                    train_stage_s,
                    val_stage_s,
                    epoch_s,
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/loss_mse": train_mse,
                        "train/loss_cosine": train_cos_loss,
                        "train/cosine": train_cos,
                        "train/pred_norm": train_pred_norm,
                        "train/real_norm": train_real_norm,
                        "val/loss": val_loss,
                        "val/loss_mse": val_mse,
                        "val/loss_cosine": val_cos_loss,
                        "val/cosine": val_cos,
                        "val/pred_norm": val_pred_norm,
                        "val/real_norm": val_real_norm,
                        "optim/lr": optimizer.param_groups[0]["lr"],
                        "optim/grad_norm": last_grad_norm,
                        "data/train_samples": len(train_ds),
                        "data/val_samples": len(val_ds),
                        "data/train_steps": train_steps,
                        "data/val_steps": val_steps,
                        "dist/world_size": world_size,
                        "time/train_s": train_stage_s,
                        "time/val_s": val_stage_s,
                        "time/epoch_s": epoch_s,
                        "time/train_step_s": train_stage_s / max(1, train_steps),
                        "time/val_step_s": val_stage_s / max(1, val_steps),
                    }
                )

            scheduler.step()

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                no_improve = 0
                if is_main:
                    model_to_save = model.module if isinstance(model, DDP) else model
                    if isinstance(model_to_save, torch.nn.DataParallel):
                        model_to_save = model_to_save.module
                    torch.save(
                        {
                            "state_dict": model_to_save.state_dict(),
                            "model_cfg": _to_serializable(dataclasses.asdict(cfg.model)),
                            "train_cfg": _to_serializable(dataclasses.asdict(cfg)),
                        },
                        ckpt_dir / "best_model.pt",
                    )
                    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
                        json.dump(_to_serializable(dataclasses.asdict(cfg)), f, ensure_ascii=False, indent=2)
                    logger.info("Saved best checkpoint to %s", ckpt_dir / "best_model.pt")
                    wandb.log({"val/best_loss": best_val, "val/best_epoch": epoch})
            else:
                no_improve += 1
                if is_main:
                    wandb.log({"early_stop/no_improve_epochs": no_improve})

            # Always track two explicit checkpoints by objective metric.
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                if is_main:
                    model_to_save = model.module if isinstance(model, DDP) else model
                    if isinstance(model_to_save, torch.nn.DataParallel):
                        model_to_save = model_to_save.module
                    torch.save(
                        {
                            "state_dict": model_to_save.state_dict(),
                            "model_cfg": _to_serializable(dataclasses.asdict(cfg.model)),
                            "train_cfg": _to_serializable(dataclasses.asdict(cfg)),
                            "best_metric": "val_mse",
                            "best_metric_value": best_val_mse,
                            "best_epoch": epoch,
                        },
                        ckpt_dir / "best_mse_model.pt",
                    )
                    logger.info("Saved best MSE checkpoint to %s", ckpt_dir / "best_mse_model.pt")
                    wandb.log({"val/best_mse": best_val_mse, "val/best_mse_epoch": epoch})

            if val_cos_loss < best_val_cos_loss:
                best_val_cos_loss = val_cos_loss
                if is_main:
                    model_to_save = model.module if isinstance(model, DDP) else model
                    if isinstance(model_to_save, torch.nn.DataParallel):
                        model_to_save = model_to_save.module
                    torch.save(
                        {
                            "state_dict": model_to_save.state_dict(),
                            "model_cfg": _to_serializable(dataclasses.asdict(cfg.model)),
                            "train_cfg": _to_serializable(dataclasses.asdict(cfg)),
                            "best_metric": "val_cosine_loss",
                            "best_metric_value": best_val_cos_loss,
                            "best_epoch": epoch,
                        },
                        ckpt_dir / "best_cosine_model.pt",
                    )
                    logger.info("Saved best cosine checkpoint to %s", ckpt_dir / "best_cosine_model.pt")
                    wandb.log({"val/best_cosine_loss": best_val_cos_loss, "val/best_cosine_epoch": epoch})

            stop_now = no_improve >= cfg.patience
            if _is_distributed():
                stop_tensor = torch.tensor([1 if stop_now else 0], dtype=torch.int32, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(stop_tensor.item())
            if stop_now:
                if is_main:
                    logger.info("Early stop triggered at epoch %d", epoch)
                break

        if is_main:
            wandb.finish()
    finally:
        _cleanup_distributed()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SigLIP-latent residual MLP")
    p.add_argument("--dataset-path", type=str, default="")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/siglip_dynamics",
    )
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--scale", type=str, choices=["4m", "20m", "100m", "custom"], default="20m")
    p.add_argument("--model-type", type=str, choices=["mlp", "transformer", "dit"], default="mlp")
    p.add_argument("--h-window", type=int, default=1)
    p.add_argument(
        "--h-window-list",
        type=int,
        nargs="*",
        default=None,
        help="Run multiple h-window values in one command (Cartesian product with --k-step-list).",
    )
    p.add_argument("--token-dim", type=int, default=2048)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--action-embed-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--ada-rmsnorm-eps", type=float, default=1e-6)
    p.add_argument("--custom-widths", type=int, nargs="*", default=[2048, 2048, 2048, 2048])

    p.add_argument("--batch-size", type=int, default=8, help="Per-process batch size (per GPU under torchrun).")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--train-loss-type", type=str, choices=["mse", "cosine", "both"], default="both")
    p.add_argument("--cosine-loss-weight", type=float, default=1.0)
    p.add_argument("--log-shapes-every-epoch", action="store_true", default=True)
    p.add_argument("--no-log-shapes-every-epoch", dest="log_shapes_every_epoch", action="store_false")

    p.add_argument("--k-step", type=int, default=10)
    p.add_argument(
        "--k-step-list",
        type=int,
        nargs="*",
        default=None,
        help="Run multiple k-step values in one command (Cartesian product with --h-window-list).",
    )
    p.add_argument("--max-samples", type=int, default=0)

    # W&B
    p.add_argument("--wandb-project", type=str, default="siglip-dynamics")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-group", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    return p


def _to_serializable(x):
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_serializable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_serializable(v) for v in x]
    if isinstance(x, (ModelScale, ModelType, LossType)):
        return x.value
    return x


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("siglip_dynamics.main")
    parser = build_arg_parser()
    args = parser.parse_args()
    base_cfg = _build_config(args)

    run_plan, grid_requested = _build_run_plan(args)
    if not grid_requested:
        train(base_cfg)
        return

    total = len(run_plan)
    logger.info("Grid training plan: %d run(s): %s", total, run_plan)
    base_checkpoint_dir = Path(base_cfg.checkpoint_dir)
    base_run_name = base_cfg.wandb_run_name

    for idx, (h_window, k_step) in enumerate(run_plan, start=1):
        tag = _run_tag(h_window, k_step)
        run_checkpoint_dir = str(base_checkpoint_dir / tag)
        run_name = f"{base_run_name}_{tag}" if base_run_name else ""
        run_cfg = dataclasses.replace(
            base_cfg,
            h_window=h_window,
            k_step=k_step,
            checkpoint_dir=run_checkpoint_dir,
            wandb_run_name=run_name,
        )
        logger.info(
            "[Run %d/%d] start | h_window=%d k_step=%d checkpoint_dir=%s",
            idx,
            total,
            h_window,
            k_step,
            run_checkpoint_dir,
        )
        train(run_cfg)
        logger.info("[Run %d/%d] done | tag=%s", idx, total, tag)


if __name__ == "__main__":
    main()
