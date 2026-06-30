from __future__ import annotations

from collections import deque
from pathlib import Path

import torch
from torch import nn

from ..config import ModelType
from ..MLP import SiglipResidualMLP
from ..transformer import SiglipResidualTransformer
from ..dit_transformer import SiglipDiTTransformer


class SafetyModuleLoader:
    """懒加载safety模型，管理历史窗口，预测delta_z"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str | ModelType,
        token_dim: int,
        action_dim: int,
        h_window: int = 1,
        device: str = "cuda",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = ModelType(model_type) if isinstance(model_type, str) else model_type
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.h_window = h_window
        self.device = device

        self._model: nn.Module | None = None
        self._z_history: deque = deque(maxlen=h_window)

    def load(self):
        """懒加载模型"""
        if self._model is not None:
            return

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        # 处理不同的checkpoint格式
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        if self.model_type == ModelType.MLP:
            from ..config import SiglipMLPConfig
            cfg = SiglipMLPConfig(token_dim=self.token_dim, action_dim=self.action_dim)
            self._model = SiglipResidualMLP(cfg)
        elif self.model_type == ModelType.TRANSFORMER:
            self._model = SiglipResidualTransformer(
                token_dim=self.token_dim,
                action_dim=self.action_dim,
            )
        elif self.model_type == ModelType.DIT:
            self._model = SiglipDiTTransformer(
                token_dim=self.token_dim,
                action_dim=self.action_dim,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

    def predict_delta_z(self, z_t: torch.Tensor, action_normalized: torch.Tensor) -> torch.Tensor:
        """
        预测delta_z
        Args:
            z_t: [L,D] 当前视觉特征
            action_normalized: [A] 归一化动作
        Returns:
            delta_z_pred: [L,D] 预测的k步残差
        """
        self.load()
        self._z_history.append(z_t.clone())

        with torch.no_grad():
            if self.model_type == ModelType.MLP:
                return self._model(z_t.unsqueeze(0), action_normalized.unsqueeze(0))[0]
            else:
                z_hist = torch.stack(list(self._z_history), dim=0).unsqueeze(0)
                return self._model(z_hist, action_normalized.unsqueeze(0))[0]

    def reset_history(self):
        """重置历史窗口"""
        self._z_history.clear()
