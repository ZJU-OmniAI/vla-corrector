#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


@dataclass
class SafetyMLPConfig:
    token_dim: int
    action_dim: int
    action_embed_dim: int = 256
    dropout: float = 0.0
    scale: str = "20m"
    custom_widths: tuple[int, ...] = (2048, 2048, 2048, 2048)

    def layer_widths(self) -> list[int]:
        scale = str(self.scale).lower()
        if scale == "4m":
            return [1024, 1024, 1024]
        if scale == "20m":
            return [2048, 2048, 2048, 2048]
        if scale == "100m":
            return [4096, 4096, 4096, 4096, 4096, 4096]
        return list(self.custom_widths)


class _MLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class SiglipResidualMLP(nn.Module):
    """Predicts delta_z from z_t and action_t.

    Inputs:
      - z_t: (L, D) or (B, L, D)
      - a_t: (A,) or (B, A)
    Output:
      - delta_z: same rank as z_t
    """

    def __init__(self, cfg: SafetyMLPConfig):
        super().__init__()
        self.cfg = cfg

        widths = cfg.layer_widths()
        if len(widths) < 2:
            raise ValueError("Safety MLP needs at least 2 layer widths.")

        in_dim = cfg.token_dim + cfg.action_embed_dim
        self.action_proj = nn.Linear(cfg.action_dim, cfg.action_embed_dim)
        self.in_proj = nn.Linear(in_dim, widths[0])
        self.blocks = nn.ModuleList([_MLPBlock(w, cfg.dropout) for w in widths])

        transitions = []
        for i in range(len(widths) - 1):
            if widths[i] == widths[i + 1]:
                transitions.append(nn.Identity())
            else:
                transitions.append(
                    nn.Sequential(nn.LayerNorm(widths[i]), nn.Linear(widths[i], widths[i + 1]), nn.SiLU())
                )
        self.transitions = nn.ModuleList(transitions)

        self.out_norm = nn.LayerNorm(widths[-1])
        self.out_proj = nn.Linear(widths[-1], cfg.token_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(0)
            squeeze_batch = True
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(0)

        if z_t.ndim != 3:
            raise ValueError(f"z_t must be (L,D) or (B,L,D), got {tuple(z_t.shape)}")
        if a_t.ndim != 2:
            raise ValueError(f"a_t must be (A,) or (B,A), got {tuple(a_t.shape)}")
        if z_t.shape[0] != a_t.shape[0]:
            if z_t.shape[0] == 1 and a_t.shape[0] > 1 and a_t.shape[-1] == self.cfg.action_dim:
                a_t = a_t[:1]
            else:
                raise ValueError(f"Batch mismatch: z_t batch={z_t.shape[0]}, a_t batch={a_t.shape[0]}")
        if z_t.shape[-1] != self.cfg.token_dim:
            raise ValueError(f"token_dim mismatch: {z_t.shape[-1]} != {self.cfg.token_dim}")
        if a_t.shape[-1] != self.cfg.action_dim:
            raise ValueError(f"action_dim mismatch: {a_t.shape[-1]} != {self.cfg.action_dim}")

        bsz, lsz, _ = z_t.shape
        a_emb = self.action_proj(a_t)[:, None, :].expand(bsz, lsz, -1)
        x = torch.cat([z_t, a_emb], dim=-1)
        x = self.in_proj(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.out_norm(x)
        out = self.out_proj(x)
        return out[0] if squeeze_batch else out


def _arch_from_scale(scale: str) -> tuple[int, int, int]:
    scale = str(scale).lower()
    if scale == "4m":
        return 512, 8, 2
    if scale == "100m":
        return 768, 12, 8
    return 768, 12, 4


class SiglipResidualTransformer(nn.Module):
    """Transformer safety predictor matching the training-side checkpoint format."""

    def __init__(
        self,
        *,
        token_dim: int,
        action_dim: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.action_dim = int(action_dim)

        self.z_proj = nn.Linear(self.token_dim, d_model)
        self.a_proj = nn.Linear(self.action_dim, d_model)
        self.time_embed = nn.Embedding(512, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.token_dim)

    def forward(self, z_hist: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z_hist.ndim == 2:
            z_hist = z_hist.unsqueeze(0).unsqueeze(0)
            squeeze = True
        elif z_hist.ndim == 3:
            z_hist = z_hist.unsqueeze(1)
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(0)

        if z_hist.ndim != 4:
            raise ValueError(f"z_hist must be (B,H,L,D)/(B,L,D)/(L,D), got {tuple(z_hist.shape)}")
        if a_t.ndim != 2:
            raise ValueError(f"a_t must be (B,A)/(A), got {tuple(a_t.shape)}")
        if z_hist.shape[0] != a_t.shape[0]:
            raise ValueError("Batch mismatch between z_hist and a_t")
        if z_hist.shape[-1] != self.token_dim:
            raise ValueError(f"token_dim mismatch: got {z_hist.shape[-1]}, expected {self.token_dim}")
        if a_t.shape[-1] != self.action_dim:
            raise ValueError(f"action_dim mismatch: got {a_t.shape[-1]}, expected {self.action_dim}")

        bsz, hsz, lsz, _ = z_hist.shape
        x = self.z_proj(z_hist)
        h_ids = torch.arange(hsz, device=x.device)
        h_emb = self.time_embed(h_ids).view(1, hsz, 1, -1)
        x = (x + h_emb).reshape(bsz, hsz * lsz, -1)

        a_tok = self.a_proj(a_t).unsqueeze(1)
        x = torch.cat([a_tok, x], dim=1)
        y = self.encoder(x)
        y_tokens = y[:, 1:, :].reshape(bsz, hsz, lsz, -1)
        y_last = self.norm(y_tokens[:, -1, :, :])
        delta = self.out_proj(y_last)
        return delta[0] if squeeze else delta


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class AdaRMSNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=eps)
        self.mlp_norm = RMSNorm(dim, eps=eps)
        self.to_mod = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 4),
        )
        nn.init.zeros_(self.to_mod[-1].weight)
        nn.init.zeros_(self.to_mod[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_mod(cond).chunk(4, dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rope_dim = cos.shape[-1]
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rope = (x_rope * cos) + (_rotate_half(x_rope) * sin)
    return torch.cat([x_rope, x_pass], dim=-1)


class RotarySelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, rope_theta: float = 10000.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim must be divisible by n_heads, got dim={dim} n_heads={n_heads}")

        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(dim // n_heads)
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got head_dim={self.head_dim}")

        self.rope_theta = float(rope_theta)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.head_dim // 2
        freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = self.rope_theta ** (-freq_seq / max(1, half_dim))
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype=dtype).view(1, 1, seq_len, self.head_dim)
        sin = emb.sin().to(dtype=dtype).view(1, 1, seq_len, self.head_dim)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        cos, sin = self._rope_cache(seq_len, x.device, q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn = attn.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.dim)
        return self.proj(attn)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        ada_rmsnorm_eps: float = 1e-6,
    ):
        super().__init__()
        self.adarms = AdaRMSNorm(dim, cond_dim, eps=ada_rmsnorm_eps)
        self.attn = RotarySelfAttention(dim, n_heads=n_heads, dropout=dropout, rope_theta=rope_theta)
        self.mlp = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        attn_scale, attn_gate, mlp_scale, mlp_gate = self.adarms(cond)
        x_attn = self.adarms.attn_norm(x) * (1.0 + attn_scale.unsqueeze(1))
        x = x + attn_gate.unsqueeze(1) * self.attn(x_attn)
        x_mlp = self.adarms.mlp_norm(x) * (1.0 + mlp_scale.unsqueeze(1))
        x = x + mlp_gate.unsqueeze(1) * self.mlp(x_mlp)
        return x


class SiglipDiTTransformer(nn.Module):
    """DiT safety predictor matching the training-side checkpoint format."""

    def __init__(
        self,
        *,
        token_dim: int,
        action_dim: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 4,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        ada_rmsnorm_eps: float = 1e-6,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)

        self.z_proj = nn.Linear(self.token_dim, d_model)
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=d_model,
                    cond_dim=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    ada_rmsnorm_eps=ada_rmsnorm_eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=ada_rmsnorm_eps)
        self.out_proj = nn.Linear(d_model, self.token_dim)

    def forward(self, z_hist: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z_hist.ndim == 2:
            z_hist = z_hist.unsqueeze(0).unsqueeze(0)
            squeeze = True
        elif z_hist.ndim == 3:
            z_hist = z_hist.unsqueeze(1)
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(0)

        if z_hist.ndim != 4:
            raise ValueError(f"z_hist must be (B,H,L,D)/(B,L,D)/(L,D), got {tuple(z_hist.shape)}")
        if a_t.ndim != 2:
            raise ValueError(f"a_t must be (B,A)/(A), got {tuple(a_t.shape)}")
        if z_hist.shape[0] != a_t.shape[0]:
            raise ValueError("Batch mismatch between z_hist and a_t")
        if z_hist.shape[-1] != self.token_dim:
            raise ValueError(f"token_dim mismatch: got {z_hist.shape[-1]}, expected {self.token_dim}")
        if a_t.shape[-1] != self.action_dim:
            raise ValueError(f"action_dim mismatch: got {a_t.shape[-1]}, expected {self.action_dim}")

        bsz, hsz, lsz, _ = z_hist.shape
        x = self.z_proj(z_hist).reshape(bsz, hsz * lsz, self.d_model)
        cond = self.action_proj(a_t)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x).reshape(bsz, hsz, lsz, self.d_model)
        delta = self.out_proj(x[:, -1, :, :])
        return delta[0] if squeeze else delta


def _find_model_dir(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path.parent

    if (model_path / "config.json").is_file():
        return model_path

    if (model_path / "run_config.json").is_file():
        return model_path

    candidates = sorted(
        p.parent for p in model_path.rglob("config.json") if p.parent.is_dir() and p.parent != model_path
    )
    if not candidates:
        candidates = sorted(
            p.parent for p in model_path.rglob("run_config.json") if p.parent.is_dir() and p.parent != model_path
        )
    if not candidates:
        raise FileNotFoundError(f"No config.json or run_config.json found under safety model path: {model_path}")
    return candidates[0]


def _pick_checkpoint(model_dir: Path) -> Path:
    candidates = [
        "best_cosine_model.pt",
        "best_mse_model.pt",
        "best_model.pt",
    ]
    for name in candidates:
        path = model_dir / name
        if path.is_file():
            return path
    pt_files = sorted(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint found in {model_dir}")
    return pt_files[0]


class SiglipDynamicsPredictor:
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.model_dir = _find_model_dir(Path(model_path).expanduser().resolve())
        self.checkpoint_path = _pick_checkpoint(self.model_dir)
        self.device = torch.device(device)
        self.model_type = "mlp"

        config_path = self.model_dir / "config.json"
        run_config_path = self.model_dir / "run_config.json"
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                cfg_json = json.load(f)

            model_cfg_json = cfg_json.get("model", {})
            custom_widths = tuple(int(x) for x in model_cfg_json.get("custom_widths", [2048, 2048, 2048, 2048]))
            self.model_cfg = SafetyMLPConfig(
                token_dim=int(model_cfg_json["token_dim"]),
                action_dim=int(model_cfg_json["action_dim"]),
                action_embed_dim=int(model_cfg_json.get("action_embed_dim", 256)),
                dropout=float(model_cfg_json.get("dropout", 0.0)),
                scale=str(model_cfg_json.get("scale", "20m")),
                custom_widths=custom_widths,
            )
            self.model_type = str(model_cfg_json.get("model_type", cfg_json.get("model_type", "mlp"))).lower()
            self.rope_theta = float(model_cfg_json.get("rope_theta", cfg_json.get("rope_theta", 10000.0)))
            self.ada_rmsnorm_eps = float(
                model_cfg_json.get("ada_rmsnorm_eps", cfg_json.get("ada_rmsnorm_eps", 1e-6))
            )
            self.h_window = int(cfg_json.get("h_window", 1))
            self.k_step = int(cfg_json.get("k_step", 1))
        elif run_config_path.is_file():
            with open(run_config_path, encoding="utf-8") as f:
                run_cfg = json.load(f)

            args_json = run_cfg.get("args", {})
            self.model_type = str(run_cfg.get("model_type", args_json.get("model_type", "mlp"))).lower()
            self.model_cfg = SafetyMLPConfig(
                token_dim=int(run_cfg["token_dim"]),
                action_dim=int(run_cfg["action_dim"]),
                action_embed_dim=int(args_json.get("action_embed_dim", 256)),
                dropout=float(args_json.get("dropout", 0.0)),
                scale=str(args_json.get("scale", "20m")),
            )
            self.rope_theta = float(args_json.get("rope_theta", run_cfg.get("rope_theta", 10000.0)))
            self.ada_rmsnorm_eps = float(
                args_json.get("ada_rmsnorm_eps", run_cfg.get("ada_rmsnorm_eps", 1e-6))
            )
            self.h_window = int(run_cfg.get("h_window", 1))
            self.k_step = int(run_cfg.get("k_step", 1))
        else:
            raise FileNotFoundError(f"Missing config.json or run_config.json in {self.model_dir}")

        obj = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict: dict[str, Any]
        if isinstance(obj, dict) and "state_dict" in obj:
            state_dict = obj["state_dict"]
            train_cfg = obj.get("train_cfg", {})
            if isinstance(train_cfg, dict):
                self.h_window = int(train_cfg.get("h_window", self.h_window))
                self.k_step = int(train_cfg.get("k_step", self.k_step))
            self.h_window = int(obj.get("h_window", self.h_window))
            self.k_step = int(obj.get("k_step", self.k_step))
            self.model_type = str(obj.get("model_type", self.model_type)).lower()
        elif isinstance(obj, dict):
            state_dict = obj
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

        if self.h_window != 1:
            raise ValueError(
                f"Only h_window=1 is supported in this stage, but checkpoint has h_window={self.h_window}."
            )

        d_model, n_heads, n_layers = _arch_from_scale(self.model_cfg.scale)
        if self.model_type == "mlp":
            self.model = SiglipResidualMLP(self.model_cfg).to(self.device)
        elif self.model_type == "transformer":
            self.model = SiglipResidualTransformer(
                token_dim=int(self.model_cfg.token_dim),
                action_dim=int(self.model_cfg.action_dim),
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=float(self.model_cfg.dropout),
            ).to(self.device)
        elif self.model_type == "dit":
            self.model = SiglipDiTTransformer(
                token_dim=int(self.model_cfg.token_dim),
                action_dim=int(self.model_cfg.action_dim),
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=float(self.model_cfg.dropout),
                rope_theta=float(self.rope_theta),
                ada_rmsnorm_eps=float(self.ada_rmsnorm_eps),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported safety model_type={self.model_type}")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @property
    def token_dim(self) -> int:
        return self.model_cfg.token_dim

    @property
    def action_dim(self) -> int:
        return self.model_cfg.action_dim

    @torch.no_grad()
    def predict_delta_z(self, z_t: np.ndarray, action_t: np.ndarray) -> np.ndarray:
        z = torch.as_tensor(z_t, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action_t, dtype=torch.float32, device=self.device)
        pred = self.model(z, a)
        return pred.detach().to("cpu", dtype=torch.float32).numpy()
