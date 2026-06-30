from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F


class BreakerState(Enum):
    NORMAL = "normal"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


class CircuitBreaker:
    """熔断检测器：计算余弦误差，动态阈值判断，状态机管理"""

    def __init__(self, k_mad: float = 3.0, window_size: int = 100, cooldown_steps: int = 5):
        self.k_mad = k_mad
        self.window_size = window_size
        self.cooldown_steps = cooldown_steps

        self.error_history = deque(maxlen=window_size)
        self.state = BreakerState.NORMAL
        self.cooldown_counter = 0

    def compute_error(self, delta_z_pred: torch.Tensor, delta_z_real: torch.Tensor) -> float:
        """余弦误差: E = 1 - cos(pred, real)"""
        pred_flat = delta_z_pred.flatten()
        real_flat = delta_z_real.flatten()
        cos_sim = F.cosine_similarity(pred_flat.unsqueeze(0), real_flat.unsqueeze(0), dim=1)
        return (1.0 - cos_sim).item()

    def get_dynamic_threshold(self) -> float:
        """动态阈值: median + k*MAD"""
        if len(self.error_history) < 10:
            return float("inf")
        errors = np.array(self.error_history)
        median = np.median(errors)
        mad = np.median(np.abs(errors - median))
        return median + self.k_mad * mad

    def check(self, error: float) -> bool:
        """检查是否触发熔断"""
        self.error_history.append(error)

        if self.state == BreakerState.COOLDOWN:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.state = BreakerState.NORMAL
            return False

        threshold = self.get_dynamic_threshold()
        if error > threshold:
            self.state = BreakerState.TRIGGERED
            self.cooldown_counter = self.cooldown_steps
            return True

        return False

    def reset(self):
        """重置状态"""
        self.error_history.clear()
        self.state = BreakerState.NORMAL
        self.cooldown_counter = 0
