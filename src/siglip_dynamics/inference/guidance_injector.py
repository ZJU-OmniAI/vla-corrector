from __future__ import annotations

import torch
import torch.nn.functional as F


class GuidanceInjector:
    """Guidance注入逻辑（使用cosine loss）"""

    def __init__(self, guidance_scale: float = 1.0):
        self.guidance_scale = guidance_scale
        self.target_delta_z: torch.Tensor | None = None
        self.z_baseline: torch.Tensor | None = None

    def set_target(self, z_current: torch.Tensor, delta_z_pred: torch.Tensor):
        """熔断触发时设置guidance目标"""
        self.z_baseline = z_current.clone()
        self.target_delta_z = delta_z_pred.clone()

    def clear_target(self):
        """清除guidance目标"""
        self.target_delta_z = None
        self.z_baseline = None

    def compute_guidance_loss(self, z_current: torch.Tensor) -> torch.Tensor | None:
        """
        计算guidance损失（使用cosine loss）

        参考OpenPI实现：
        target_delta = guidance_delta_z_right_t - guidance_delta_z_t
        loss = (1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1)).mean()
        """
        if self.target_delta_z is None:
            return None

        # 计算当前的delta_z
        delta_z_current = z_current - self.z_baseline

        # Flatten
        pred_flat = delta_z_current.reshape(delta_z_current.shape[0], -1)
        target_flat = self.target_delta_z.reshape(self.target_delta_z.shape[0], -1)

        # Cosine loss
        loss = (1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1)).mean()

        return loss * self.guidance_scale
