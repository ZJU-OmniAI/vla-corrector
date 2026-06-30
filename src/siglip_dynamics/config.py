from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class ModelScale(str, Enum):
    # Four tiers: 4M / 20M / 100M / custom
    M4 = "4m"
    M20 = "20m"
    M100 = "100m"
    CUSTOM = "custom"


class ModelType(str, Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"
    DIT = "dit"


class LossType(str, Enum):
    MSE = "mse"
    COSINE = "cosine"
    BOTH = "both"


@dataclass
class SiglipMLPConfig:
    token_dim: int = 2048
    action_dim: int = 7
    action_embed_dim: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    ada_rmsnorm_eps: float = 1e-6
    scale: ModelScale = ModelScale.M20

    # Used only when scale == CUSTOM
    custom_widths: Sequence[int] = field(default_factory=lambda: (2048, 2048, 2048, 2048))

    def layer_widths(self) -> list[int]:
        if self.scale == ModelScale.M4:
            return [1024, 1024, 1024]
        if self.scale == ModelScale.M20:
            return [2048, 2048, 2048, 2048]
        if self.scale == ModelScale.M100:
            return [4096, 4096, 4096, 4096, 4096, 4096]
        return list(self.custom_widths)


@dataclass
class TrainConfig:
    dataset_path: str = ""
    checkpoint_dir: str = "outputs/siglip_dynamics"
    device: str = "cuda"

    model: SiglipMLPConfig = field(default_factory=SiglipMLPConfig)
    model_type: ModelType = ModelType.MLP
    h_window: int = 1  # history frames for input, >=1

    batch_size: int = 2048
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 4
    seed: int = 42
    grad_clip_norm: float = 1.0
    patience: int = 10
    train_loss_type: LossType = LossType.BOTH
    cosine_loss_weight: float = 1.0
    log_shapes_every_epoch: bool = True

    k_step: int = 10
    max_samples: int = 0  # 0 means no limit

    # W&B config
    wandb_project: str = "siglip-dynamics"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_run_name: str = ""
