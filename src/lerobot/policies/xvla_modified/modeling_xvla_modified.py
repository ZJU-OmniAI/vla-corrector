#!/usr/bin/env python

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.policies.xvla.modeling_xvla import XVLAModel, XVLAPolicy
from lerobot.utils.constants import ACTION

from .configuration_xvla_modified import XVLAModifiedConfig


class XVLAModelModified(XVLAModel):
    """XVLA core model with optional safety-guided denoising updates."""

    def __init__(self, config, *, florence_config, proprio_dim: int):
        super().__init__(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self.safety_predictor: nn.Module | None = None
        self.safety_initialized = False
        self.safety_model_action_dim = 7
        self.safety_model_type = "mlp"
        self.safety_guidance_eta = 0.0
        self._guidance_action_truncate_warned = False
        self._last_guidance_stats: dict[str, Any] = {}

    def set_safety_predictor(
        self,
        predictor: nn.Module | None,
        *,
        action_dim: int,
        model_type: str = "mlp",
        eta: float = 0.0,
    ) -> None:
        self.safety_predictor = predictor
        self.safety_initialized = predictor is not None
        self.safety_model_action_dim = int(max(1, action_dim))
        self.safety_model_type = str(model_type).lower()
        self.safety_guidance_eta = float(eta)
        if predictor is not None:
            predictor.eval()
            for p in predictor.parameters():
                p.requires_grad_(False)

    def get_last_guidance_stats(self) -> dict[str, Any]:
        return dict(self._last_guidance_stats)

    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
        **kwargs,
    ) -> torch.Tensor:
        guidance_replan = bool(kwargs.pop("guidance_replan", False))
        guidance_current_z_t = kwargs.pop("guidance_current_z_t", None)
        guidance_current_z_hist_t = kwargs.pop("guidance_current_z_hist_t", None)
        guidance_current_delta_z_t = kwargs.pop("guidance_current_delta_z_t", None)
        guidance_current_delta_z_right = kwargs.pop("guidance_current_delta_z_right", None)
        guidance_apply_every = kwargs.pop("guidance_apply_every", None)
        guidance_apply_steps = kwargs.pop("guidance_apply_steps", None)
        guidance_loss_objective = kwargs.pop("guidance_loss_objective", None)
        guidance_eta_override = kwargs.pop("guidance_eta", None)
        noise = kwargs.pop("noise", None)

        self.eval()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        with torch.no_grad():
            enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action

        if noise is None:
            x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        else:
            x1 = noise.to(device=proprio.device, dtype=target_dtype)
            if x1.shape != (batch_size, self.chunk_size, action_dim):
                raise ValueError(
                    f"noise shape mismatch: got {tuple(x1.shape)} expect {(batch_size, self.chunk_size, action_dim)}"
                )

        action = torch.zeros_like(x1)

        objective_raw = str(guidance_loss_objective or "attract_delta_z_correction").strip().lower()
        objective_aliases = {
            "legacy": "repel_delta_z_error",
            "repel": "repel_delta_z_error",
            "correction": "attract_delta_z_correction",
            "new": "attract_delta_z_correction",
            "attract": "attract_delta_z_correction",
        }
        objective = objective_aliases.get(objective_raw, objective_raw)
        if objective not in {"repel_delta_z_error", "attract_delta_z_correction"}:
            logging.warning(
                "[GUIDE] unknown guidance_loss_objective=%s; fallback to attract_delta_z_correction.",
                objective_raw,
            )
            objective = "attract_delta_z_correction"

        if guidance_apply_every is None:
            guidance_apply_every = guidance_apply_steps if guidance_apply_steps is not None else 3
        guidance_apply_every = max(1, int(guidance_apply_every))

        steps = max(1, int(steps))
        scheduled_steps = int((steps + guidance_apply_every - 1) // guidance_apply_every)

        eta = float(self.safety_guidance_eta if guidance_eta_override is None else guidance_eta_override)
        use_history_model = str(self.safety_model_type).lower() in {"transformer", "dit"}

        guidance_loss_values: list[float] = []
        guidance_gnorm_values: list[float] = []
        guidance_delta_v_norm_values: list[float] = []
        guidance_v_norm_pre_values: list[float] = []
        guidance_target_norm_values: list[float] = []
        guidance_pred_norm_values: list[float] = []
        guidance_applied_steps = 0

        guidance_stats: dict[str, Any] = {
            "guidance_requested": bool(guidance_replan),
            "guidance_enabled": False,
            "configured_apply_every": int(guidance_apply_every),
            "configured_apply_steps": int(scheduled_steps),
            "scheduled_steps": int(scheduled_steps),
            "applied_steps": 0,
            "eta": float(eta),
            "loss_objective": str(objective),
            "avg_loss": float("nan"),
            "avg_g_norm": float("nan"),
            "avg_delta_v_norm": float("nan"),
            "avg_v_norm_pre": float("nan"),
            "avg_target_norm": float("nan"),
            "avg_pred_norm": float("nan"),
        }

        def _as_batch_tensor(value, *, expected_ndim: int) -> Tensor | None:
            if value is None:
                return None
            out = torch.as_tensor(value, dtype=torch.float32, device=proprio.device)
            if out.ndim == expected_ndim - 1:
                out = out.unsqueeze(0)
            return out

        guidance_enabled = bool(guidance_replan)
        guidance_z_t = None
        guidance_z_hist_t = None
        guidance_delta_z_t = None
        guidance_delta_z_right_t = None

        if guidance_enabled:
            if eta <= 0.0:
                logging.warning("[GUIDE] disabled: eta<=0 (eta=%.6f).", eta)
                guidance_enabled = False
            elif not self.safety_initialized or self.safety_predictor is None:
                logging.warning("[GUIDE] guidance requested but safety predictor is not initialized.")
                guidance_enabled = False
            elif (guidance_current_z_t is None and guidance_current_z_hist_t is None) or guidance_current_delta_z_t is None:
                logging.warning(
                    "[GUIDE] missing guidance payload: z_t_none=%s z_hist_none=%s delta_z_t_none=%s",
                    guidance_current_z_t is None,
                    guidance_current_z_hist_t is None,
                    guidance_current_delta_z_t is None,
                )
                guidance_enabled = False
            else:
                if guidance_current_z_hist_t is not None:
                    guidance_z_hist_t = _as_batch_tensor(guidance_current_z_hist_t, expected_ndim=4)
                if guidance_current_z_t is not None:
                    guidance_z_t = _as_batch_tensor(guidance_current_z_t, expected_ndim=3)
                guidance_delta_z_t = _as_batch_tensor(guidance_current_delta_z_t, expected_ndim=3)
                guidance_delta_z_right_t = _as_batch_tensor(guidance_current_delta_z_right, expected_ndim=3)

                if use_history_model:
                    if guidance_z_hist_t is None and guidance_z_t is not None:
                        guidance_z_hist_t = guidance_z_t.unsqueeze(1)
                    if guidance_z_hist_t is not None and guidance_z_hist_t.shape[0] == 1 and batch_size > 1:
                        guidance_z_hist_t = guidance_z_hist_t.expand(batch_size, -1, -1, -1)
                else:
                    if guidance_z_t is None and guidance_z_hist_t is not None:
                        guidance_z_t = guidance_z_hist_t[:, -1, :, :]
                    if guidance_z_t is not None and guidance_z_t.shape[0] == 1 and batch_size > 1:
                        guidance_z_t = guidance_z_t.expand(batch_size, -1, -1)

                if guidance_delta_z_t is not None and guidance_delta_z_t.shape[0] == 1 and batch_size > 1:
                    guidance_delta_z_t = guidance_delta_z_t.expand(batch_size, -1, -1)
                if guidance_delta_z_right_t is not None and guidance_delta_z_right_t.shape[0] == 1 and batch_size > 1:
                    guidance_delta_z_right_t = guidance_delta_z_right_t.expand(batch_size, -1, -1)

                if guidance_delta_z_t is None:
                    guidance_enabled = False
                elif objective == "attract_delta_z_correction" and guidance_delta_z_right_t is None:
                    logging.warning("[GUIDE] objective attract_delta_z_correction requires guidance_current_delta_z_right.")
                    guidance_enabled = False
                elif not use_history_model and guidance_z_t is None:
                    guidance_enabled = False
                elif use_history_model and guidance_z_hist_t is None:
                    guidance_enabled = False

        guidance_stats["guidance_enabled"] = bool(guidance_enabled)
        guidance_z_input = guidance_z_hist_t if use_history_model else guidance_z_t

        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            with torch.no_grad():
                action = self.transformer(
                    domain_id=domain_id,
                    action_with_noise=x_t_m,
                    proprio=proprio_m,
                    t=t,
                    **enc,
                )

            step_idx = steps - i
            if (
                guidance_enabled
                and guidance_z_input is not None
                and guidance_delta_z_t is not None
                and (step_idx % guidance_apply_every == 0)
            ):
                v_in = action.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    a_t_guidance = v_in[:, 0, :]
                    if a_t_guidance.shape[-1] > self.safety_model_action_dim:
                        if not self._guidance_action_truncate_warned:
                            logging.info(
                                "[GUIDE] truncate action for safety model: %d -> %d",
                                a_t_guidance.shape[-1],
                                self.safety_model_action_dim,
                            )
                            self._guidance_action_truncate_warned = True
                        a_t_guidance = a_t_guidance[:, : self.safety_model_action_dim]
                    elif a_t_guidance.shape[-1] < self.safety_model_action_dim:
                        logging.warning(
                            "[GUIDE] action dim too short for safety model: got=%d expect>=%d",
                            a_t_guidance.shape[-1],
                            self.safety_model_action_dim,
                        )
                        a_t_guidance = None

                    if a_t_guidance is not None:
                        delta_z_pred = self.safety_predictor(guidance_z_input, a_t_guidance)
                        if delta_z_pred.shape == guidance_delta_z_t.shape:
                            pred_flat = delta_z_pred.reshape(delta_z_pred.shape[0], -1)
                            if objective == "repel_delta_z_error":
                                target_flat = guidance_delta_z_t.reshape(guidance_delta_z_t.shape[0], -1)
                                loss_danger = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
                            else:
                                target_delta = guidance_delta_z_right_t - guidance_delta_z_t
                                target_flat = target_delta.reshape(target_delta.shape[0], -1)
                                loss_danger = (1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1)).mean()

                            loss_danger.backward()
                            g_v = v_in.grad
                            if g_v is not None:
                                delta_v = eta * g_v
                                guidance_loss_values.append(float(loss_danger.detach().item()))
                                guidance_gnorm_values.append(float(g_v.detach().norm().item()))
                                guidance_delta_v_norm_values.append(float(delta_v.detach().norm().item()))
                                guidance_v_norm_pre_values.append(float(v_in.detach().norm().item()))
                                guidance_target_norm_values.append(
                                    float(target_flat.detach().norm(dim=-1).mean().item())
                                )
                                guidance_pred_norm_values.append(
                                    float(pred_flat.detach().norm(dim=-1).mean().item())
                                )
                                guidance_applied_steps += 1
                                action = (v_in - delta_v).detach()
                        else:
                            logging.warning(
                                "[GUIDE] shape mismatch: pred=%s target=%s",
                                tuple(delta_z_pred.shape),
                                tuple(guidance_delta_z_t.shape),
                            )

        guidance_stats["applied_steps"] = int(guidance_applied_steps)
        if guidance_loss_values:
            guidance_stats["avg_loss"] = float(np.mean(guidance_loss_values))
            guidance_stats["avg_g_norm"] = float(np.mean(guidance_gnorm_values))
            guidance_stats["avg_delta_v_norm"] = float(np.mean(guidance_delta_v_norm_values))
            guidance_stats["avg_v_norm_pre"] = float(np.mean(guidance_v_norm_pre_values))
            guidance_stats["avg_target_norm"] = float(np.mean(guidance_target_norm_values))
            guidance_stats["avg_pred_norm"] = float(np.mean(guidance_pred_norm_values))

        self._last_guidance_stats = guidance_stats
        return self.action_space.postprocess(action)


class XVLAModifiedPolicy(XVLAPolicy):
    """XVLA policy wrapper with optional safety-guidance APIs."""

    config_class = XVLAModifiedConfig
    name = "xvla_modified"

    def __init__(self, config: XVLAModifiedConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLAModelModified(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self.reset()

    def set_safety_predictor(
        self,
        predictor: nn.Module | None,
        *,
        action_dim: int,
        model_type: str = "mlp",
        eta: float | None = None,
    ) -> None:
        if eta is None:
            eta = float(getattr(self.config, "safety_guidance_eta", 0.0))
        self.model.set_safety_predictor(
            predictor,
            action_dim=int(action_dim),
            model_type=str(model_type),
            eta=float(eta),
        )

    def _ensure_image_feature_keys(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        expected_keys = list(self.config.image_features.keys())
        if not expected_keys:
            return batch
        if any(key in batch for key in expected_keys):
            return batch

        fallback_key = "observation.image"
        if fallback_key not in batch:
            return batch

        src = batch[fallback_key]
        out = dict(batch)
        for idx, key in enumerate(expected_keys):
            if key in out:
                continue
            out[key] = src if idx == 0 else torch.zeros_like(src)
        return out

    @torch.no_grad()
    def encode_siglip_only(self, batch: dict[str, Tensor]) -> np.ndarray:
        self.eval()
        batch = self._ensure_image_feature_keys(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        inputs = self._build_model_inputs(batch)

        image_input = inputs["image_input"].to(dtype=self.model._get_target_dtype())
        image_mask = inputs["image_mask"]
        batch_size, num_views = image_input.shape[:2]

        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = image_input.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.model.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)

        z_t = image_features[:, 0, :, :][0].detach().to("cpu", dtype=torch.float32).numpy()
        return np.asarray(z_t, dtype=np.float32)

    @torch.no_grad()
    def predict_delta_z_only(self, z_t: np.ndarray, raw_action_t: np.ndarray) -> np.ndarray:
        if not self.model.safety_initialized or self.model.safety_predictor is None:
            raise RuntimeError("Safety predictor is not initialized in xvla_modified policy.")
        z = torch.as_tensor(z_t, dtype=torch.float32, device=self.config.device)
        a = torch.as_tensor(raw_action_t, dtype=torch.float32, device=self.config.device)
        pred = self.model.safety_predictor(z, a)
        return pred.detach().to("cpu", dtype=torch.float32).numpy()

    @staticmethod
    def _compute_compare_stats(guided_actions: Tensor, baseline_actions: Tensor, eps: float = 1e-6) -> dict[str, Any]:
        guided = guided_actions.detach().to(dtype=torch.float32)
        baseline = baseline_actions.detach().to(dtype=torch.float32)

        delta = torch.linalg.vector_norm(guided - baseline, dim=-1)
        baseline_norm = torch.linalg.vector_norm(baseline, dim=-1)
        ratio = delta / (baseline_norm + float(eps))

        return {
            "compare_baseline_enabled": True,
            "action_delta_l2_mean": float(delta.mean().item()),
            "action_delta_l2_max": float(delta.max().item()),
            "action_delta_l2_over_baseline_mean": float(ratio.mean().item()),
            "action_delta_first7": delta[0, :7].detach().cpu().tolist() if delta.ndim == 2 and delta.shape[0] > 0 else [],
        }

    @staticmethod
    def _compute_raw_compare_stats(guided_raw: Tensor, baseline_raw: Tensor) -> dict[str, Any]:
        guided = guided_raw.detach().to(dtype=torch.float32)
        baseline = baseline_raw.detach().to(dtype=torch.float32)
        delta = torch.linalg.vector_norm(guided - baseline, dim=-1)
        return {
            "raw_action_delta_l2_mean": float(delta.mean().item()),
            "raw_action_delta_l2_max": float(delta.max().item()),
        }

    @torch.no_grad()
    def predict_action_chunk_with_aux(
        self,
        batch: dict[str, Tensor],
        *,
        guidance_replan: bool,
        guidance_current_z_t: np.ndarray | None,
        guidance_current_z_hist_t: np.ndarray | None,
        guidance_current_delta_z_t: np.ndarray | None,
        guidance_current_delta_z_right: np.ndarray | None,
        guidance_apply_every: int,
        guidance_loss_objective: str,
        guidance_compare_baseline: bool,
    ) -> dict[str, Any]:
        self.eval()
        batch = self._ensure_image_feature_keys(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        inputs = self._build_model_inputs(batch)
        bsize = int(inputs["input_ids"].shape[0])
        action_dim = int(self.model.dim_action)
        target_dtype = self.model._get_target_dtype()
        shared_noise = torch.randn(
            bsize,
            self.config.chunk_size,
            action_dim,
            device=inputs["proprio"].device,
            dtype=target_dtype,
        )

        guided_raw = self.model.generate_actions(
            **inputs,
            steps=self.config.num_denoising_steps,
            guidance_replan=bool(guidance_replan),
            guidance_current_z_t=guidance_current_z_t,
            guidance_current_z_hist_t=guidance_current_z_hist_t,
            guidance_current_delta_z_t=guidance_current_delta_z_t,
            guidance_current_delta_z_right=guidance_current_delta_z_right,
            guidance_apply_every=int(guidance_apply_every),
            guidance_loss_objective=str(guidance_loss_objective),
            noise=shared_noise.clone(),
        )

        guidance_stats = self.model.get_last_guidance_stats() if guidance_replan else {}

        baseline_raw = None
        if guidance_replan and guidance_compare_baseline:
            baseline_raw = self.model.generate_actions(
                **inputs,
                steps=self.config.num_denoising_steps,
                guidance_replan=False,
                noise=shared_noise.clone(),
            )
            env_action_dim = self.config.output_features[ACTION].shape[0]
            compare_stats = self._compute_compare_stats(
                guided_raw[:, :, :env_action_dim],
                baseline_raw[:, :, :env_action_dim],
            )
            raw_compare_stats = self._compute_raw_compare_stats(guided_raw, baseline_raw)
            guidance_stats = {**guidance_stats, **compare_stats, **raw_compare_stats}
            self.model._last_guidance_stats = dict(guidance_stats)

        env_action_dim = self.config.output_features[ACTION].shape[0]
        actions = guided_raw[:, :, :env_action_dim]

        n_action_steps = int(self.config.n_action_steps)
        if n_action_steps <= 0:
            raise ValueError(f"config.n_action_steps must be > 0, got {n_action_steps}")
        if actions.shape[1] < n_action_steps:
            raise ValueError(
                f"policy predicts only {actions.shape[1]} steps, but n_action_steps={n_action_steps}."
            )

        out: dict[str, Any] = {
            "actions": actions,
            "raw_actions": guided_raw,
            "guidance_stats": guidance_stats,
        }
        if baseline_raw is not None:
            out["baseline_raw_actions"] = baseline_raw

        return out
