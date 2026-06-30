from __future__ import annotations

import torch
from torch import nn


class SiglipResidualTransformer(nn.Module):
    """
    Input:
      - z_hist: (B,H,L,D) or (B,L,D) or (L,D)
      - a_t:    (B,A) or (A)
    Output:
      - delta_z: (B,L,D) or (L,D)
    """

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
        self.token_dim = token_dim
        self.action_dim = action_dim

        self.z_proj = nn.Linear(token_dim, d_model)
        self.a_proj = nn.Linear(action_dim, d_model)
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
        self.out_proj = nn.Linear(d_model, token_dim)

    def forward(self, z_hist: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z_hist.ndim == 2:
            z_hist = z_hist.unsqueeze(0).unsqueeze(0)  # [1,1,L,D]
            squeeze = True
        elif z_hist.ndim == 3:
            z_hist = z_hist.unsqueeze(1)  # [B,1,L,D]
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
        x = self.z_proj(z_hist)  # [B,H,L,C]

        # Add per-history-step embedding to preserve temporal order.
        h_ids = torch.arange(hsz, device=x.device)
        h_emb = self.time_embed(h_ids).view(1, hsz, 1, -1)
        x = x + h_emb

        x = x.reshape(bsz, hsz * lsz, -1)

        # action token as global condition
        a_tok = self.a_proj(a_t).unsqueeze(1)  # [B,1,C]
        x = torch.cat([a_tok, x], dim=1)

        y = self.encoder(x)
        # drop action token, keep last frame token outputs only
        y_tokens = y[:, 1:, :].reshape(bsz, hsz, lsz, -1)
        y_last = y_tokens[:, -1, :, :]  # [B,L,C]
        y_last = self.norm(y_last)
        delta = self.out_proj(y_last)  # [B,L,D]

        if squeeze:
            return delta[0]
        return delta

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
