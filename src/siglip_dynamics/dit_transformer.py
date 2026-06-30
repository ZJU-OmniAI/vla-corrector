from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class AdaRMSNorm(nn.Module):
    """
    Action-conditioned RMSNorm modulation.

    Returns four modulation tensors per layer (separate gates for attn and MLP):
      - attn_scale: scale for attention pre-norm
      - attn_gate: residual gate for the attention branch
      - mlp_scale:  scale for MLP pre-norm
      - mlp_gate:   residual gate for the MLP branch
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=eps)
        self.mlp_norm = RMSNorm(dim, eps=eps)
        # Linear first, then SiLU — preserves all sign information from cond.
        self.to_mod = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 4),  # attn_scale, attn_gate, mlp_scale, mlp_gate
        )
        nn.init.zeros_(self.to_mod[-1].weight)
        nn.init.zeros_(self.to_mod[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_scale, attn_gate, mlp_scale, mlp_gate = self.to_mod(cond).chunk(4, dim=-1)
        return attn_scale, attn_gate, mlp_scale, mlp_gate


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = torch.stack((-x2, x1), dim=-1)
    return y.flatten(start_dim=-2)


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

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
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
        hidden_dim = dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
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
        rope_theta: float = 10000.0,
        ada_rmsnorm_eps: float = 1e-6,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.d_model = d_model

        self.z_proj = nn.Linear(token_dim, d_model)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
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
        self.out_proj = nn.Linear(d_model, token_dim)

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
        x = x[:, -1, :, :]
        delta = self.out_proj(x)

        if squeeze:
            return delta[0]
        return delta

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
