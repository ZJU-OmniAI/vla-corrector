from __future__ import annotations

import torch
from torch import nn

from .config import SiglipMLPConfig


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
    """
    Input:
      - z_t: (B,L,D) or (L,D)
      - a_t: (B,A) or (A)
    Output:
      - delta_z: same rank as z_t
    """

    def __init__(self, cfg: SiglipMLPConfig):
        super().__init__()
        self.cfg = cfg

        widths = cfg.layer_widths()
        if len(widths) < 2:
            raise ValueError("Need at least 2 widths for SiglipResidualMLP.")

        in_dim = cfg.token_dim + cfg.action_embed_dim
        self.action_proj = nn.Linear(cfg.action_dim, cfg.action_embed_dim)
        self.in_proj = nn.Linear(in_dim, widths[0])

        self.blocks = nn.ModuleList([_MLPBlock(w, cfg.dropout) for w in widths])

        transitions = []
        for i in range(len(widths) - 1):
            if widths[i] != widths[i + 1]:
                # Dimension changes: project with a non-linear transition.
                transitions.append(
                    nn.Sequential(nn.LayerNorm(widths[i]), nn.Linear(widths[i], widths[i + 1]), nn.SiLU())
                )
            else:
                # Same dimension: identity keeps the residual stream intact.
                transitions.append(nn.Identity())
        self.transitions = nn.ModuleList(transitions)

        self.out_norm = nn.LayerNorm(widths[-1])
        self.out_proj = nn.Linear(widths[-1], cfg.token_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(0)
            squeeze_batch = True
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(0)

        if z_t.ndim != 3:
            raise ValueError(f"z_t must be (B,L,D) or (L,D), got {tuple(z_t.shape)}")
        if a_t.ndim != 2:
            raise ValueError(f"a_t must be (B,A) or (A), got {tuple(a_t.shape)}")
        if z_t.shape[0] != a_t.shape[0]:
            raise ValueError(f"Batch mismatch: z_t batch={z_t.shape[0]}, a_t batch={a_t.shape[0]}")
        if z_t.shape[-1] != self.cfg.token_dim:
            raise ValueError(f"token_dim mismatch: got {z_t.shape[-1]}, expected {self.cfg.token_dim}")
        if a_t.shape[-1] != self.cfg.action_dim:
            raise ValueError(f"action_dim mismatch: got {a_t.shape[-1]}, expected {self.cfg.action_dim}")

        bsz, lsz, _ = z_t.shape
        a_emb = self.action_proj(a_t)  # (B,C)
        a_emb = a_emb[:, None, :].expand(bsz, lsz, -1)

        x = torch.cat([z_t, a_emb], dim=-1)
        x = self.in_proj(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.out_norm(x)
        delta_z = self.out_proj(x)
        if squeeze_batch:
            return delta_z[0]
        return delta_z
