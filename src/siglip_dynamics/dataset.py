from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SiglipDynamicsDataset(Dataset):
    """
    Expected keys in torch file:
      - Z_t:    (N,L,D) or (N,D)
      - Z_next: (N,L,D) or (N,D)
      - A_t:    (N,A)
    """

    def __init__(
        self,
        data_path: str,
        *,
        k_step: int = 10,
        h_window: int = 1,
        max_samples: int = 0,
    ):
        self.data_path = Path(data_path)
        self.k_step = int(k_step)
        self.h_window = int(h_window)
        self.max_samples = int(max_samples)
        self.mode = ""
        self.pairs: list[tuple[int, int]] = []

        if self.k_step < 1:
            raise ValueError(f"k_step must be >= 1, got {self.k_step}")
        if self.h_window < 1:
            raise ValueError(f"h_window must be >= 1, got {self.h_window}")

        logger.info("Loading dataset from %s", data_path)
        if self.data_path.is_dir() and (self.data_path / "metadata.json").exists():
            self._load_quantized_frame_cache()
        else:
            self._load_legacy_triplets()

    def _load_legacy_triplets(self) -> None:
        if self.h_window != 1:
            raise ValueError("legacy_triplets format does not support h_window > 1")

        data = torch.load(str(self.data_path), weights_only=False)

        self.z_t = data["Z_t"].float()
        self.z_next = data["Z_next"].float()
        self.a_t = data["A_t"].float()

        if self.z_t.shape != self.z_next.shape:
            raise ValueError(f"Z_t/Z_next shape mismatch: {self.z_t.shape} vs {self.z_next.shape}")
        if self.z_t.shape[0] != self.a_t.shape[0]:
            raise ValueError(f"Sample count mismatch: Z={self.z_t.shape[0]} A={self.a_t.shape[0]}")

        self.delta_z = self.z_next - self.z_t
        self.mode = "legacy_triplets"
        logger.info("Loaded legacy triplet dataset: %d samples", self.z_t.shape[0])

    def _load_quantized_frame_cache(self) -> None:
        with open(self.data_path / "metadata.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.z_q = np.load(self.data_path / "z_q.npy", mmap_mode="r")  # int8 [N,L,D]
        self.z_scale = np.load(self.data_path / "z_scale.npy", mmap_mode="r")  # float16 [N,L]
        self.a_t_np = np.load(self.data_path / "actions.npy", mmap_mode="r")  # float16/float32 [N,A]
        self.ep_idx = np.load(self.data_path / "episode_index.npy", mmap_mode="r")  # int32 [N]

        if self.z_q.shape[0] != self.a_t_np.shape[0] or self.z_q.shape[0] != self.ep_idx.shape[0]:
            raise ValueError(
                f"Frame count mismatch: z_q={self.z_q.shape[0]} actions={self.a_t_np.shape[0]} ep={self.ep_idx.shape[0]}"
            )
        if self.z_scale.shape[:2] != self.z_q.shape[:2]:
            raise ValueError(f"z_scale shape mismatch: z_scale={self.z_scale.shape} z_q={self.z_q.shape}")

        # Build valid (t, t+k) pairs within same episode.
        n = int(self.z_q.shape[0])
        i = 0
        while i < n:
            ep = int(self.ep_idx[i])
            j = i + 1
            while j < n and int(self.ep_idx[j]) == ep:
                j += 1
            hist_start = i + (self.h_window - 1)
            for t in range(hist_start, j - self.k_step):
                self.pairs.append((t, t + self.k_step))
                if self.max_samples > 0 and len(self.pairs) >= self.max_samples:
                    break
            if self.max_samples > 0 and len(self.pairs) >= self.max_samples:
                break
            i = j

        if not self.pairs:
            raise RuntimeError("No valid (t, t+k) pairs found in quantized frame cache.")

        self.mode = "quantized_frames"
        logger.info(
            "Loaded quantized frame cache: frames=%d pairs=%d shape(L,D)=(%d,%d) k=%d h=%d",
            self.z_q.shape[0],
            len(self.pairs),
            self.z_q.shape[1],
            self.z_q.shape[2],
            self.k_step,
            self.h_window,
        )

    @staticmethod
    def _dequantize(z_q: np.ndarray, z_scale: np.ndarray) -> np.ndarray:
        # z_q: [...,L,D] int8, z_scale: [...,L] float16 -> z: [...,L,D] float32
        scale = z_scale.astype(np.float32)[..., None]
        return z_q.astype(np.float32) * scale

    def __len__(self) -> int:
        if self.mode == "legacy_triplets":
            return int(self.z_t.shape[0])
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.mode == "legacy_triplets":
            z_t = self.z_t[idx]
            return {
                "z_t": z_t,
                "z_hist": z_t.unsqueeze(0),
                "a_t": self.a_t[idx],
                "delta_z": self.delta_z[idx],
            }

        t, t_next = self.pairs[idx]
        hist_start = t - self.h_window + 1
        z_hist_np = self._dequantize(self.z_q[hist_start : t + 1], self.z_scale[hist_start : t + 1])
        z_t_np = z_hist_np[-1]
        z_next_np = self._dequantize(self.z_q[t_next], self.z_scale[t_next])
        a_t_np = np.asarray(self.a_t_np[t], dtype=np.float32)
        delta_np = z_next_np - z_t_np
        return {
            "z_t": torch.from_numpy(z_t_np),
            "z_hist": torch.from_numpy(z_hist_np),
            "a_t": torch.from_numpy(a_t_np),
            "delta_z": torch.from_numpy(delta_np),
        }
