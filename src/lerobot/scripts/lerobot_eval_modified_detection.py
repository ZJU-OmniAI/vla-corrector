#!/usr/bin/env python

import csv
import dataclasses
import datetime as dt
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
from tqdm import trange, tqdm

from lerobot import envs, policies  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05_modified.configuration_pi05_modified import PI05ModifiedConfig
from lerobot.policies.pi05_modified.modeling_pi05_modified import PI05ModifiedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla_modified.configuration_smolvla_modified import SmolVLAModifiedConfig
from lerobot.policies.smolvla_modified.modeling_smolvla_modified import SmolVLAModifiedPolicy
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla_modified.configuration_xvla_modified import XVLAModifiedConfig
from lerobot.policies.xvla_modified.modeling_xvla_modified import XVLAModifiedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.safety.siglip_dynamics_mlp import SiglipDynamicsPredictor
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, inside_slurm


def _maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as exc:
        logging.warning("matplotlib unavailable, skip curve visualizations: %s", exc)
        return None


def _safe_slug(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out[:120] if out else "task"


def _to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _safe_nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _safe_nansum(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    return float(np.nansum(arr))


def _safe_binary_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) == 0 or len(ys) == 0 or len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda"] = None
    else:
        state["cuda"] = None
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    cpu_state = state.get("cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    if torch.cuda.is_available():
        cuda_state = state.get("cuda")
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass


def _flatten_feature(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={img.shape}")
    img = img.astype(np.uint8, copy=False)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _render_vec_env(env: gym.vector.VectorEnv) -> np.ndarray:
    if isinstance(env, gym.vector.SyncVectorEnv):
        frame = env.envs[0].render()
    else:
        frame = env.call("render")[0]
    return np.asarray(frame, dtype=np.uint8)


def _extract_success(info: dict[str, Any]) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            value = final_info.get("is_success")
            if isinstance(value, np.ndarray):
                if value.size > 0:
                    return bool(value[0])
            elif isinstance(value, list):
                if value:
                    return bool(value[0])
            elif value is not None:
                return bool(value)
        elif isinstance(final_info, list) and final_info:
            first = final_info[0]
            if isinstance(first, dict):
                return bool(first.get("is_success", False))
    value = info.get("is_success", False)
    if isinstance(value, np.ndarray):
        return bool(value[0]) if value.size > 0 else False
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    if not fieldnames:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_recovery_latencies(
    et_values: list[float], thresh_values: list[float], meltdown_flags: list[bool]
) -> list[int]:
    n = len(et_values)
    latencies: list[int] = []
    for i in range(n):
        if not meltdown_flags[i]:
            continue
        for j in range(i + 1, n):
            e = et_values[j]
            th = thresh_values[j]
            if np.isfinite(e) and np.isfinite(th) and e <= th:
                latencies.append(j - i)
                break
    return latencies


def _make_episode_summary(
    *,
    task_group: str,
    task_id: int,
    task_description: str,
    episode_idx: int,
    success: bool,
    t_env: int,
    et_values: list[float],
    thresh_values: list[float],
    meltdown_flags: list[bool],
    episode_error: str,
) -> dict[str, Any]:
    et_np = np.asarray(et_values, dtype=np.float32)
    th_np = np.asarray(thresh_values, dtype=np.float32)
    valid_mask = np.isfinite(et_np) & np.isfinite(th_np)
    valid_et = et_np[valid_mask]
    valid_th = th_np[valid_mask]

    meltdown_count = int(sum(bool(x) for x in meltdown_flags))
    valid_count = int(valid_mask.sum())
    recovery_latencies = _compute_recovery_latencies(et_values, thresh_values, meltdown_flags)
    first_meltdown = -1
    for i, is_md in enumerate(meltdown_flags):
        if is_md:
            first_meltdown = i
            break

    def _maybe_stat(arr: np.ndarray, fn, default=float("nan")) -> float:
        if arr.size == 0:
            return default
        return float(fn(arr))

    return {
        "task_group": task_group,
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "success": bool(success),
        "env_steps_executed": int(t_env),
        "num_frames": int(len(et_values)),
        "num_et_valid": valid_count,
        "meltdown_count": meltdown_count,
        "meltdown_per_100_steps": float(100.0 * meltdown_count / max(1, t_env)),
        "exceed_ratio_valid": float(meltdown_count / valid_count) if valid_count > 0 else float("nan"),
        "et_mean": _maybe_stat(valid_et, np.mean),
        "et_std": _maybe_stat(valid_et, np.std),
        "et_max": _maybe_stat(valid_et, np.max),
        "et_p95": _maybe_stat(valid_et, lambda x: np.percentile(x, 95)),
        "threshold_mean": _maybe_stat(valid_th, np.mean),
        "threshold_max": _maybe_stat(valid_th, np.max),
        "first_meltdown_frame": int(first_meltdown),
        "recovery_count": int(len(recovery_latencies)),
        "recovery_rate_after_meltdown": (
            float(len(recovery_latencies) / meltdown_count) if meltdown_count > 0 else float("nan")
        ),
        "recovery_steps_mean": float(np.mean(recovery_latencies)) if recovery_latencies else float("nan"),
        "episode_error": episode_error,
    }


def _save_et_curve_plot(
    plt,
    out_path: Path,
    et_values: list[float],
    thresh_values: list[float],
    release_thresh_values: list[float],
    meltdown_flags: list[bool],
    title: str,
    *,
    frame_indices: list[int] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_indices is not None and len(frame_indices) >= len(et_values):
        x = np.asarray(frame_indices[: len(et_values)], dtype=np.int32)
    else:
        x = np.arange(len(et_values), dtype=np.int32)

    et = np.asarray(et_values, dtype=np.float32)
    th_on = np.asarray(thresh_values, dtype=np.float32)
    th_off = np.asarray(release_thresh_values, dtype=np.float32)
    md = np.asarray(meltdown_flags, dtype=bool)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=130)
    ax.plot(x, et, label="E_t", linewidth=1.2)
    ax.plot(x, th_on, label="threshold_on (Ton)", linewidth=1.2)
    ax.plot(x, th_off, label="threshold_off (Toff)", linewidth=1.2, linestyle="--")
    if md.any():
        ax.scatter(x[md], et[md], s=12, c="red", label="meltdown", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_sync_video_with_curve(
    plt,
    out_path: Path,
    replay_images: list[np.ndarray],
    et_values: list[float],
    thresh_values: list[float],
    release_thresh_values: list[float],
    meltdown_flags: list[bool],
    *,
    fps: int,
    title: str,
    frame_indices: list[int] | None = None,
    t_env_values: list[int] | None = None,
) -> None:
    if not replay_images:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames = len(replay_images)
    if frame_indices is not None and len(frame_indices) > 0:
        if len(frame_indices) >= num_frames:
            x = np.asarray(frame_indices[:num_frames], dtype=np.int32)
        else:
            start = int(frame_indices[0])
            x = start + np.arange(num_frames, dtype=np.int32)
    else:
        x = np.arange(num_frames, dtype=np.int32)

    t_env_plot = np.full((num_frames,), np.nan, dtype=np.float32)
    if t_env_values:
        n_tenv = min(num_frames, len(t_env_values))
        if n_tenv > 0:
            t_env_plot[:n_tenv] = np.asarray(t_env_values[:n_tenv], dtype=np.float32)

    et = np.full((num_frames,), np.nan, dtype=np.float32)
    th_on = np.full((num_frames,), np.nan, dtype=np.float32)
    th_off = np.full((num_frames,), np.nan, dtype=np.float32)
    md = np.zeros((num_frames,), dtype=bool)
    n_et = min(num_frames, len(et_values))
    n_th_on = min(num_frames, len(thresh_values))
    n_th_off = min(num_frames, len(release_thresh_values))
    n_md = min(num_frames, len(meltdown_flags))
    if n_et > 0:
        et[:n_et] = np.asarray(et_values[:n_et], dtype=np.float32)
    if n_th_on > 0:
        th_on[:n_th_on] = np.asarray(thresh_values[:n_th_on], dtype=np.float32)
    if n_th_off > 0:
        th_off[:n_th_off] = np.asarray(release_thresh_values[:n_th_off], dtype=np.float32)
    if n_md > 0:
        md[:n_md] = np.asarray(meltdown_flags[:n_md], dtype=bool)

    h, w = replay_images[0].shape[:2]
    panel_w = max(512, int(w * 1.6))
    total_w = w + panel_w
    if total_w % 16 != 0:
        panel_w += 16 - (total_w % 16)
    panel_h = h
    fig_w = panel_w / 100.0
    fig_h = panel_h / 100.0

    with imageio.get_writer(out_path, fps=fps, macro_block_size=16) as writer:
        for i in range(num_frames):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
            ax.plot(x, et, label="E_t", linewidth=1.2)
            ax.plot(x, th_on, label="threshold_on (Ton)", linewidth=1.2)
            ax.plot(x, th_off, label="threshold_off (Toff)", linewidth=1.2, linestyle="--")
            if md.any():
                ax.scatter(x[md], et[md], s=12, c="red", label="meltdown", zorder=4)
            current_x = x[i] if i < len(x) else i
            ax.axvline(current_x, color="black", linestyle="--", linewidth=1.0, label="current")

            cur_e = et[i] if i < len(et) else float("nan")
            cur_t_on = th_on[i] if i < len(th_on) else float("nan")
            cur_t_off = th_off[i] if i < len(th_off) else float("nan")
            cur_md = bool(md[i]) if i < len(md) else False
            cur_frame = int(x[i]) if i < len(x) else i
            cur_tenv = t_env_plot[i] if i < len(t_env_plot) else float("nan")
            if np.isfinite(cur_tenv):
                title_suffix = (
                    f"frame_idx={cur_frame}  t_env={int(cur_tenv)}  "
                    f"E_t={cur_e:.4f}  Ton={cur_t_on:.4f}  Toff={cur_t_off:.4f}  meltdown={cur_md}"
                )
            else:
                title_suffix = (
                    f"frame_idx={cur_frame}  "
                    f"E_t={cur_e:.4f}  Ton={cur_t_on:.4f}  Toff={cur_t_off:.4f}  meltdown={cur_md}"
                )
            ax.set_title(f"{title}\n{title_suffix}")
            ax.set_xlabel("frame_idx")
            ax.set_ylabel("value")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.canvas.draw()
            panel_rgba = np.asarray(fig.canvas.buffer_rgba())
            panel = panel_rgba[..., :3].astype(np.uint8)
            plt.close(fig)

            panel = _resize_with_pad(panel, panel_h, panel_w)
            left = _resize_with_pad(np.asarray(replay_images[i], dtype=np.uint8), panel.shape[0], w)
            combined = np.concatenate([left, panel], axis=1)
            combined = np.require(np.ascontiguousarray(combined), dtype=np.uint8, requirements=["C", "A"])
            writer.append_data(combined)


def _save_side_by_side_video(
    out_path: Path,
    left_images: list[np.ndarray],
    right_images: list[np.ndarray],
    *,
    fps: int,
) -> None:
    if not left_images or not right_images:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the full duration of the longer rollout.
    # When one side ends earlier, hold its last frame.
    n = max(len(left_images), len(right_images))
    if n <= 0:
        return

    left0 = np.asarray(left_images[0], dtype=np.uint8)
    h, w = left0.shape[:2]
    left_last = max(0, len(left_images) - 1)
    right_last = max(0, len(right_images) - 1)
    with imageio.get_writer(out_path, fps=fps, macro_block_size=16) as writer:
        for i in range(n):
            left = _resize_with_pad(np.asarray(left_images[min(i, left_last)], dtype=np.uint8), h, w)
            right = _resize_with_pad(np.asarray(right_images[min(i, right_last)], dtype=np.uint8), h, w)
            combined = np.concatenate([left, right], axis=1)
            combined = np.require(np.ascontiguousarray(combined), dtype=np.uint8, requirements=["C", "A"])
            writer.append_data(combined)


def _save_task_level_plots(plt, out_dir: Path, task_rows: list[dict[str, Any]]) -> None:
    if not task_rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [f'{r["task_group"]}:{r["task_id"]}' for r in task_rows]
    success = [float(r.get("success_rate", float("nan"))) * 100.0 for r in task_rows]
    meltdowns = [float(r.get("avg_meltdown_count", float("nan"))) for r in task_rows]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=130)
    ax.bar(names, success)
    ax.set_title("Task Success Rate (%)")
    ax.set_xlabel("task")
    ax.set_ylabel("success %")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "task_success_rate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=130)
    ax.bar(names, meltdowns)
    ax.set_title("Task Avg Meltdown Count per Episode")
    ax.set_xlabel("task")
    ax.set_ylabel("avg meltdown count")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "task_avg_meltdown_count.png")
    plt.close(fig)


@dataclass
class RobustThresholdState:
    window_size: int = 15
    base_noise_floor: float = 0.25
    bootstrap_initial_threshold: float = 10.0
    bootstrap_queue_size: int = 15
    bootstrap_sigma_scale: float = 1.0
    ewma_alpha: float = 0.35
    k_on: float = 3.0
    k_off: float = 2.0
    trigger_margin: float = 0.02
    jump_trigger: float = 0.06
    trigger_consecutive_steps: int = 5
    reset_consecutive_steps: int = 5
    cooldown_steps: int = 0
    hard_retrigger_margin: float = 0.08
    hard_retrigger_consecutive_steps: int = 3

    bootstrap_et_queue: list[float] = field(default_factory=list)
    bootstrapped: bool = False
    clean_history: list[float] = field(default_factory=list)
    dynamic_threshold: list[float] = field(default_factory=list)
    dynamic_release_threshold: list[float] = field(default_factory=list)
    ewma_prev: float | None = None
    alert_active: bool = False
    over_count: int = 0
    under_count: int = 0
    hard_over_count: int = 0
    cooldown_remaining: int = 0


def robust_threshold_step(current_et: float, state: RobustThresholdState) -> tuple[float, bool]:
    if not np.isfinite(current_et):
        state.dynamic_threshold.append(float("nan"))
        state.dynamic_release_threshold.append(float("nan"))
        return float("nan"), False

    if state.ewma_prev is None or not np.isfinite(state.ewma_prev):
        ebar = float(current_et)
        jump = 0.0
    else:
        alpha = float(np.clip(state.ewma_alpha, 1e-6, 1.0))
        ebar = alpha * float(current_et) + (1.0 - alpha) * float(state.ewma_prev)
        jump = abs(ebar - float(state.ewma_prev))
    state.ewma_prev = float(ebar)

    if not state.bootstrapped:
        thresh = float(state.bootstrap_initial_threshold)
        state.bootstrap_et_queue.append(float(ebar))
        if len(state.bootstrap_et_queue) > int(state.bootstrap_queue_size):
            state.bootstrap_et_queue = state.bootstrap_et_queue[-int(state.bootstrap_queue_size) :]

        if len(state.bootstrap_et_queue) >= int(state.bootstrap_queue_size):
            q = np.asarray(state.bootstrap_et_queue, dtype=np.float32)
            med = float(np.median(q))
            mad = float(np.median(np.abs(q - med)))
            mad = max(mad, 1e-6)
            ton = max(state.base_noise_floor, med + float(state.k_on) * mad)
            toff = max(state.base_noise_floor, med + float(state.k_off) * mad)
            if toff > ton:
                toff = ton
            thresh = ton
            state.bootstrapped = True
            state.clean_history = list(float(x) for x in q.tolist())
            state.dynamic_release_threshold.append(float(toff))
        else:
            state.dynamic_release_threshold.append(float(thresh))

        state.dynamic_threshold.append(thresh)
        return thresh, False

    recent_history = state.clean_history[-max(2, int(state.window_size)) :]
    rh = np.asarray(recent_history, dtype=np.float32)
    med = float(np.median(rh))
    mad = float(np.median(np.abs(rh - med)))
    mad = max(mad, 1e-6)
    ton = max(state.base_noise_floor, med + float(state.k_on) * mad)
    toff = max(state.base_noise_floor, med + float(state.k_off) * mad)
    if toff > ton:
        toff = ton

    state.dynamic_threshold.append(float(ton))
    state.dynamic_release_threshold.append(float(toff))

    cond_over = bool(ebar > (ton + float(state.trigger_margin)))
    cond_jump = bool(jump > float(state.jump_trigger) and ebar > ton)
    cond_hard = bool(ebar > (ton + float(state.hard_retrigger_margin)))

    state.over_count = state.over_count + 1 if cond_over else 0
    state.hard_over_count = state.hard_over_count + 1 if cond_hard else 0
    state.under_count = state.under_count + 1 if ebar < toff else 0

    if state.alert_active and state.under_count >= int(state.reset_consecutive_steps):
        state.alert_active = False
        state.over_count = 0
        state.hard_over_count = 0

    should_trigger = False
    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1
        if state.hard_over_count >= int(state.hard_retrigger_consecutive_steps):
            should_trigger = True
    else:
        if cond_jump or state.over_count >= int(state.trigger_consecutive_steps):
            should_trigger = True

    if should_trigger:
        state.alert_active = True
        state.cooldown_remaining = max(0, int(state.cooldown_steps))
        state.over_count = 0
        state.hard_over_count = 0
        state.under_count = 0

    if state.alert_active:
        state.clean_history.append(float(min(ebar, ton)))
    else:
        state.clean_history.append(float(ebar))

    return float(ton), bool(should_trigger)


@dataclass
class EvalModifiedDetectionConfig:
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int = 1000
    rename_map: dict[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False

    safety_model_path: str = ""
    safety_k: int = 0
    safety_history_frames: int = 1
    threshold_window_size: int = 15
    threshold_base_noise_floor: float = 0.25
    threshold_bootstrap_initial_threshold: float = 10.0
    threshold_bootstrap_queue_size: int = 15
    threshold_bootstrap_sigma_scale: float = 1.0
    threshold_ewma_alpha: float = 0.35
    threshold_k_on: float = 3.0
    threshold_k_off: float = 2.0
    threshold_trigger_margin: float = 0.02
    threshold_jump_trigger: float = 0.06
    threshold_trigger_consecutive_steps: int = 5
    threshold_reset_consecutive_steps: int = 5
    threshold_hard_retrigger_margin: float = 0.08
    threshold_hard_retrigger_consecutive_steps: int = 3
    meltdown_cooldown_steps: int = 0
    guidance_eta: float = 0.0
    guidance_apply_every: int = 3
    guidance_loss_objective: str = "attract_delta_z_correction"
    guidance_compare_baseline: bool = True
    meltdown_use_guidance_replan: bool = True
    benchmark_ogg_timing: bool = False

    video_fps: int = 10
    save_analysis: bool = True
    compute_baseline_metrics: bool = True
    save_raw_video: bool = True
    save_sync_video_with_curve: bool = True
    save_baseline_sync_video_with_curve: bool = True
    save_baseline_et_curve_plot: bool = True
    save_baseline_step_csv: bool = True
    save_guidance_dual_video: bool = True
    save_et_curve_plot: bool = True
    save_episode_step_csv: bool = True
    save_guidance_event_csv: bool = True
    save_episode_npz: bool = True
    save_summary_csv: bool = True
    save_summary_json: bool = True
    save_task_plots: bool = True
    max_visualizations_per_task: int = 0
    max_visualizations_total: int = 0

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        else:
            raise ValueError("This script requires --policy.path=<checkpoint_dir_or_repo>.")

        if not self.job_name:
            self.job_name = f"{self.env.type}_{self.policy.type}_modified_detection"

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

        if not self.safety_model_path:
            raise ValueError(
                "Missing safety model path. Please set --safety_model_path=/path/to/your/model_dir_or_parent_dir."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _extract_z_t_from_pi05(policy: PI05Policy, model_input: dict[str, Any]) -> np.ndarray:
    images, img_masks = policy._preprocess_images(model_input)  # noqa: SLF001
    token_blocks = []
    for image, mask in zip(images, img_masks, strict=False):
        use_block = True
        if isinstance(mask, torch.Tensor) and mask.ndim >= 1:
            use_block = bool(mask[0].item())
        if not use_block:
            continue
        token_blocks.append(policy.model.paligemma_with_expert.embed_image(image))

    if not token_blocks:
        token_blocks.append(policy.model.paligemma_with_expert.embed_image(images[0]))

    z_t = torch.cat(token_blocks, dim=1)[0].detach().to("cpu", dtype=torch.float32).numpy()
    return np.asarray(z_t, dtype=np.float32)


def _build_model_input(
    *,
    observation: Any,
    env: gym.vector.VectorEnv,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    obs_tensor = preprocess_observation(observation)
    obs_tensor = add_envs_task(env, obs_tensor)
    obs_tensor = env_preprocessor(obs_tensor)
    return preprocessor(obs_tensor)


def _action_to_env_numpy(
    *,
    action_tensor: torch.Tensor,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> np.ndarray:
    action = postprocessor(action_tensor)
    action_transition = {ACTION: action}
    action_transition = env_postprocessor(action_transition)
    return action_transition[ACTION].to("cpu").numpy()


GUIDANCE_EVENT_COLUMNS = [
    "task_group",
    "task_id",
    "episode_idx",
    "guidance_request_idx",
    "frame_idx",
    "t_env",
    "applied_steps",
    "scheduled_steps",
    "configured_apply_every",
    "loss_objective",
    "avg_loss",
    "avg_target_norm",
    "avg_pred_norm",
    "avg_g_norm",
    "avg_eta_g_norm",
    "avg_v_norm_pre",
    "infer_ms",
    "action_delta_l2_mean",
    "action_delta_l2_max",
    "action_delta_l2_over_baseline_mean",
    "raw_action_delta_l2_mean",
    "raw_action_delta_l2_max",
    "action_delta_first7",
]


@parser.wrap()
def eval_modified_detection(cfg: EvalModifiedDetectionConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.eval.batch_size != 1:
        raise ValueError(
            "This modified detection script currently supports --eval.batch_size=1 only "
            "(to keep per-step safety-state alignment deterministic)."
        )

    logging.info("Loading safety predictor from %s", cfg.safety_model_path)
    predictor = SiglipDynamicsPredictor(cfg.safety_model_path, device=str(device))
    logging.info(
        "Loaded safety predictor | type=%s action_dim=%d k_step=%d h_window=%d dir=%s",
        getattr(predictor, "model_type", "mlp"),
        int(predictor.action_dim),
        int(predictor.k_step),
        int(predictor.h_window),
        predictor.model_dir,
    )
    safety_k = int(cfg.safety_k) if int(cfg.safety_k) > 0 else int(predictor.k_step)
    if safety_k <= 0:
        raise ValueError(f"Invalid safety_k={safety_k}")
    if int(cfg.safety_history_frames) != 1:
        raise ValueError("Current stage supports safety_history_frames=1 only.")
    logging.info(
        "Meltdown replan mode: %s",
        "guidance-corrected" if bool(cfg.meltdown_use_guidance_replan) else "normal(no-guidance)",
    )

    logging.info("Building environments.")
    envs_map = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    analysis_enabled = bool(cfg.save_analysis)
    if not analysis_enabled:
        logging.info("Analysis outputs are disabled via --save_analysis=false.")
    save_raw_video = bool(cfg.save_raw_video)
    save_sync_video_with_curve = analysis_enabled and bool(cfg.save_sync_video_with_curve)
    save_baseline_sync_video_with_curve = analysis_enabled and bool(cfg.save_baseline_sync_video_with_curve)
    save_baseline_et_curve_plot = analysis_enabled and bool(cfg.save_baseline_et_curve_plot)
    save_baseline_step_csv = analysis_enabled and bool(cfg.save_baseline_step_csv)
    save_guidance_dual_video = analysis_enabled and bool(cfg.save_guidance_dual_video)
    save_et_curve_plot = analysis_enabled and bool(cfg.save_et_curve_plot)
    save_episode_step_csv = analysis_enabled and bool(cfg.save_episode_step_csv)
    save_guidance_event_csv = analysis_enabled and bool(cfg.save_guidance_event_csv)
    save_episode_npz = analysis_enabled and bool(cfg.save_episode_npz)
    save_summary_csv = bool(cfg.save_summary_csv)
    save_summary_json = bool(cfg.save_summary_json)
    save_task_plots = analysis_enabled and bool(cfg.save_task_plots)
    baseline_metrics_enabled = analysis_enabled and bool(cfg.compute_baseline_metrics)

    dual_compare_enabled = bool(
        baseline_metrics_enabled
        or save_guidance_dual_video
        or save_baseline_sync_video_with_curve
        or save_baseline_et_curve_plot
        or save_baseline_step_csv
    )
    envs_map_baseline = (
        make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
            trust_remote_code=cfg.trust_remote_code,
        )
        if dual_compare_enabled
        else None
    )

    logging.info("Building policy.")
    if isinstance(cfg.policy, PI05Config):
        base_policy_kwargs = {field.name: getattr(cfg.policy, field.name) for field in dataclasses.fields(cfg.policy)}
        modified_policy_cfg = PI05ModifiedConfig(**base_policy_kwargs)
        modified_policy_cfg.pretrained_path = cfg.policy.pretrained_path
        modified_policy_cfg.device = cfg.policy.device
        modified_policy_cfg.safety_guidance_eta = float(cfg.guidance_eta)
        policy = PI05ModifiedPolicy.from_pretrained(
            pretrained_name_or_path=str(cfg.policy.pretrained_path),
            config=modified_policy_cfg,
        )
    elif isinstance(cfg.policy, SmolVLAConfig):
        base_policy_kwargs = {field.name: getattr(cfg.policy, field.name) for field in dataclasses.fields(cfg.policy)}
        modified_policy_cfg = SmolVLAModifiedConfig(**base_policy_kwargs)
        modified_policy_cfg.pretrained_path = cfg.policy.pretrained_path
        modified_policy_cfg.device = cfg.policy.device
        modified_policy_cfg.safety_guidance_eta = float(cfg.guidance_eta)
        policy = SmolVLAModifiedPolicy.from_pretrained(
            pretrained_name_or_path=str(cfg.policy.pretrained_path),
            config=modified_policy_cfg,
        )
    elif isinstance(cfg.policy, XVLAConfig):
        base_policy_kwargs = {field.name: getattr(cfg.policy, field.name) for field in dataclasses.fields(cfg.policy)}
        modified_policy_cfg = XVLAModifiedConfig(**base_policy_kwargs)
        modified_policy_cfg.pretrained_path = cfg.policy.pretrained_path
        modified_policy_cfg.device = cfg.policy.device
        modified_policy_cfg.safety_guidance_eta = float(cfg.guidance_eta)
        policy = XVLAModifiedPolicy.from_pretrained(
            pretrained_name_or_path=str(cfg.policy.pretrained_path),
            config=modified_policy_cfg,
        )
    else:
        raise TypeError(
            "This stage supports pi05/smolvla/xvla checkpoints only. "
            f"Got: {type(cfg.policy).__name__}."
        )

    policy.set_safety_predictor(
        predictor.model,
        action_dim=int(predictor.action_dim),
        model_type=str(getattr(predictor, "model_type", "mlp")),
        eta=float(cfg.guidance_eta),
    )
    policy.eval()
    required_methods = ("encode_siglip_only", "predict_action_chunk_with_aux", "set_safety_predictor")
    missing_methods = [name for name in required_methods if not hasattr(policy, name)]
    if missing_methods:
        raise TypeError(
            f"Loaded policy does not provide required modified-detection methods {missing_methods}. "
            f"Got: {type(policy).__name__}."
        )
    action_queue_fill_steps = int(policy.config.n_action_steps)
    if action_queue_fill_steps <= 0:
        raise ValueError(f"policy.n_action_steps must be > 0, got {action_queue_fill_steps}")
    if int(policy.config.chunk_size) < action_queue_fill_steps:
        raise ValueError(
            f"policy.n_action_steps ({action_queue_fill_steps}) cannot exceed policy.chunk_size ({policy.config.chunk_size})."
        )
    logging.info(
        "Action queue fill size is driven by policy.n_action_steps=%d (chunk_size=%d).",
        action_queue_fill_steps,
        int(policy.config.chunk_size),
    )

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=modified_policy_cfg,
        pretrained_path=modified_policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=modified_policy_cfg)

    raw_video_dir = output_dir / "videos"
    analysis_root = output_dir / "analysis"
    et_plot_dir = analysis_root / "episode_et_plots"
    sync_video_dir = analysis_root / "sync_videos"
    et_plot_baseline_dir = analysis_root / "episode_et_plots_baseline"
    sync_video_baseline_dir = analysis_root / "sync_videos_baseline"
    sync_dual_video_dir = analysis_root / "sync_videos_dual"
    step_csv_dir = analysis_root / "episode_steps"
    step_csv_baseline_dir = analysis_root / "episode_steps_baseline"
    guidance_event_csv_dir = analysis_root / "episode_guidance_events"
    npz_dir = analysis_root / "episode_npz"
    summary_dir = (analysis_root / "summary") if analysis_enabled else (output_dir / "summary")
    summary_plots_dir = summary_dir / "plots"
    if save_raw_video:
        raw_video_dir.mkdir(parents=True, exist_ok=True)
    if analysis_enabled:
        analysis_root.mkdir(parents=True, exist_ok=True)
    if save_summary_csv or save_summary_json or save_task_plots:
        summary_dir.mkdir(parents=True, exist_ok=True)
    need_main_frames = bool(save_raw_video or save_sync_video_with_curve or save_guidance_dual_video)
    need_baseline_frames = bool(save_guidance_dual_video or save_baseline_sync_video_with_curve)

    needs_mpl = (
        save_et_curve_plot
        or save_sync_video_with_curve
        or save_baseline_et_curve_plot
        or save_baseline_sync_video_with_curve
        or save_task_plots
    )
    plt = _maybe_import_matplotlib() if needs_mpl else None

    total_episodes = 0
    total_successes = 0
    total_baseline_episodes = 0
    total_baseline_successes = 0
    global_episode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_group_rows: list[dict[str, Any]] = []
    visualized_total = 0
    disable_progress = inside_slurm()
    total_task_count = int(sum(len(group_envs) for group_envs in envs_map.values()))
    total_episode_target = int(total_task_count * int(cfg.eval.n_episodes))
    episode_prog = tqdm(
        total=total_episode_target,
        desc="Stepping through eval episodes",
        disable=disable_progress,
    )

    try:
        for task_group, group_envs in envs_map.items():
            group_task_rows: list[dict[str, Any]] = []
            for task_id, env in group_envs.items():
                env_baseline = (
                    envs_map_baseline[task_group][task_id]
                    if envs_map_baseline is not None and task_group in envs_map_baseline and task_id in envs_map_baseline[task_group]
                    else None
                )
                try:
                    task_desc_list = env.call("task_description")
                    task_description = str(task_desc_list[0]) if task_desc_list else f"{task_group}_{task_id}"
                except Exception:
                    task_description = f"{task_group}_{task_id}"

                task_episodes = 0
                task_successes = 0
                task_baseline_episodes = 0
                task_baseline_successes = 0
                task_episode_rows: list[dict[str, Any]] = []
                visualized_task = 0

                max_steps = int(env.call("_max_episode_steps")[0])

                for episode_idx in range(cfg.eval.n_episodes):
                    if not disable_progress:
                        episode_prog.set_description(f"Eval episodes | {task_group}:{task_id}")
                    step_prog = trange(
                        max_steps,
                        desc=f"Running rollout {task_group}:{task_id} ep={episode_idx}",
                        disable=disable_progress,
                        leave=False,
                    )
                    policy.reset()
                    episode_error = ""
                    try:
                        seed = cfg.seed + total_episodes if cfg.seed is not None else None
                        reset_seed = [seed] if seed is not None else None
                        observation, _info = env.reset(seed=reset_seed)
                        observation_baseline = None
                        if env_baseline is not None:
                            observation_baseline, _info_baseline = env_baseline.reset(seed=reset_seed)
                    except Exception as exc:
                        episode_error = str(exc)
                        observation = None
                        observation_baseline = None

                    done = False
                    success = False
                    baseline_success = False
                    baseline_done = False
                    t_env = 0
                    baseline_t_env = 0

                    action_plan: deque[torch.Tensor] = deque()
                    raw_action_plan: deque[np.ndarray] = deque()
                    baseline_action_plan: deque[torch.Tensor] = deque()
                    baseline_raw_action_plan: deque[np.ndarray] = deque()

                    replay_images: list[np.ndarray] = []
                    replay_images_baseline: list[np.ndarray] = []
                    et_values: list[float] = []
                    thresh_values: list[float] = []
                    thresh_release_values: list[float] = []
                    meltdown_flags: list[bool] = []
                    t_env_values: list[int] = []
                    reward_values: list[float] = []
                    guidance_event_rows: list[dict[str, Any]] = []
                    guidance_action_delta_by_frame: dict[int, float] = {}
                    guidance_request_count = 0
                    guidance_effective_count = 0
                    guidance_applied_steps_total = 0
                    guidance_loss_values: list[float] = []
                    guidance_gnorm_values: list[float] = []
                    guidance_delta_v_norm_values: list[float] = []
                    guidance_v_norm_pre_values: list[float] = []
                    guidance_target_norm_values: list[float] = []
                    guidance_pred_norm_values: list[float] = []
                    guidance_infer_ms_values: list[float] = []
                    guidance_action_delta_l2_values: list[float] = []
                    guidance_action_delta_l2_max_values: list[float] = []
                    guidance_action_delta_ratio_values: list[float] = []
                    guidance_raw_action_delta_l2_values: list[float] = []
                    guidance_raw_action_delta_l2_max_values: list[float] = []
                    action_chunk_infer_requests = 0
                    action_chunk_infer_time_ms_total = 0.0
                    benchmark_ogg_no_ogg_ms_total = 0.0
                    benchmark_ogg_with_ogg_ms_total = 0.0
                    benchmark_ogg_paired_replan_requests = 0

                    et_values_baseline: list[float] = []
                    thresh_values_baseline: list[float] = []
                    thresh_release_values_baseline: list[float] = []
                    meltdown_flags_baseline: list[bool] = []
                    t_env_values_baseline: list[int] = []

                    z_history: deque[tuple[int, np.ndarray]] = deque(maxlen=max(safety_k + 1, 2))
                    delta_z_pred_queue: deque[tuple[int, np.ndarray]] = deque(maxlen=safety_k + 2)
                    z_history_baseline: deque[tuple[int, np.ndarray]] = deque(maxlen=max(safety_k + 1, 2))
                    delta_z_pred_queue_baseline: deque[tuple[int, np.ndarray]] = deque(maxlen=safety_k + 2)

                    threshold_state = RobustThresholdState(
                        window_size=int(cfg.threshold_window_size),
                        base_noise_floor=float(cfg.threshold_base_noise_floor),
                        bootstrap_initial_threshold=float(cfg.threshold_bootstrap_initial_threshold),
                        bootstrap_queue_size=(
                            int(cfg.threshold_bootstrap_queue_size)
                            if int(cfg.threshold_bootstrap_queue_size) > 0
                            else int(cfg.threshold_window_size)
                        ),
                        bootstrap_sigma_scale=float(cfg.threshold_bootstrap_sigma_scale),
                        ewma_alpha=float(cfg.threshold_ewma_alpha),
                        k_on=float(cfg.threshold_k_on),
                        k_off=float(cfg.threshold_k_off),
                        trigger_margin=float(cfg.threshold_trigger_margin),
                        jump_trigger=float(cfg.threshold_jump_trigger),
                        trigger_consecutive_steps=int(cfg.threshold_trigger_consecutive_steps),
                        reset_consecutive_steps=int(cfg.threshold_reset_consecutive_steps),
                        cooldown_steps=max(0, int(cfg.meltdown_cooldown_steps)),
                        hard_retrigger_margin=float(cfg.threshold_hard_retrigger_margin),
                        hard_retrigger_consecutive_steps=int(cfg.threshold_hard_retrigger_consecutive_steps),
                    )
                    threshold_state_baseline = RobustThresholdState(
                        window_size=int(cfg.threshold_window_size),
                        base_noise_floor=float(cfg.threshold_base_noise_floor),
                        bootstrap_initial_threshold=float(cfg.threshold_bootstrap_initial_threshold),
                        bootstrap_queue_size=(
                            int(cfg.threshold_bootstrap_queue_size)
                            if int(cfg.threshold_bootstrap_queue_size) > 0
                            else int(cfg.threshold_window_size)
                        ),
                        bootstrap_sigma_scale=float(cfg.threshold_bootstrap_sigma_scale),
                        ewma_alpha=float(cfg.threshold_ewma_alpha),
                        k_on=float(cfg.threshold_k_on),
                        k_off=float(cfg.threshold_k_off),
                        trigger_margin=float(cfg.threshold_trigger_margin),
                        jump_trigger=float(cfg.threshold_jump_trigger),
                        trigger_consecutive_steps=int(cfg.threshold_trigger_consecutive_steps),
                        reset_consecutive_steps=int(cfg.threshold_reset_consecutive_steps),
                        cooldown_steps=max(0, int(cfg.meltdown_cooldown_steps)),
                        hard_retrigger_margin=float(cfg.threshold_hard_retrigger_margin),
                        hard_retrigger_consecutive_steps=int(cfg.threshold_hard_retrigger_consecutive_steps),
                    )

                    while not done and t_env < max_steps and observation is not None:
                        try:
                            if need_main_frames:
                                frame = _render_vec_env(env)
                                replay_images.append(frame)
                            baseline_step_active = bool(
                                env_baseline is not None and observation_baseline is not None and not baseline_done
                            )
                            if need_baseline_frames and baseline_step_active:
                                replay_images_baseline.append(_render_vec_env(env_baseline))

                            step_e_t = float("nan")
                            step_thresh = float("nan")
                            step_thresh_release = float("nan")
                            step_meltdown = False
                            step_e_t_baseline = float("nan")
                            step_thresh_baseline = float("nan")
                            step_thresh_release_baseline = float("nan")
                            step_meltdown_baseline = False
                            guidance_replan = False
                            guidance_current_z_t = None
                            guidance_current_z_hist_t = None
                            guidance_current_delta_z_t = None
                            guidance_current_delta_z_right = None
                            guidance_replan_candidate = False

                            model_input = _build_model_input(
                                observation=observation,
                                env=env,
                                env_preprocessor=env_preprocessor,
                                preprocessor=preprocessor,
                            )
                            z_t_vec = np.asarray(policy.encode_siglip_only(model_input), dtype=np.float32)
                            z_history.append((t_env, z_t_vec))

                            model_input_baseline = None
                            z_t_vec_baseline = None
                            if baseline_step_active:
                                model_input_baseline = _build_model_input(
                                    observation=observation_baseline,
                                    env=env_baseline,
                                    env_preprocessor=env_preprocessor,
                                    preprocessor=preprocessor,
                                )
                                z_t_vec_baseline = np.asarray(policy.encode_siglip_only(model_input_baseline), dtype=np.float32)
                                z_history_baseline.append((t_env, z_t_vec_baseline))

                            if t_env >= safety_k and len(delta_z_pred_queue) > 0:
                                target_t, delta_z_pred = delta_z_pred_queue[0]
                                if target_t == t_env:
                                    z_t_minus_k = None
                                    target_hist_t = t_env - safety_k
                                    for hist_t, hist_z in z_history:
                                        if hist_t == target_hist_t:
                                            z_t_minus_k = hist_z
                                            break
                                    if z_t_minus_k is None:
                                        logging.warning(
                                            "Missing aligned z_history for t_env=%d (target=%d); dropping one queued delta_z_pred.",
                                            t_env,
                                            target_hist_t,
                                        )
                                        delta_z_pred_queue.popleft()
                                    else:
                                        delta_z_real = z_t_vec - z_t_minus_k
                                        if np.linalg.norm(delta_z_real) < 1e-4:
                                            delta_z_real = delta_z_real + (
                                                np.random.randn(*delta_z_real.shape).astype(np.float32) * 1e-4
                                            )

                                        delta_z_real_flat = _flatten_feature(delta_z_real)
                                        delta_z_pred_flat = _flatten_feature(delta_z_pred)
                                        cos_denom = np.linalg.norm(delta_z_real_flat) * np.linalg.norm(delta_z_pred_flat)
                                        cos_sim = (
                                            float(np.dot(delta_z_real_flat, delta_z_pred_flat) / cos_denom)
                                            if cos_denom > 0
                                            else 0.0
                                        )
                                        step_e_t = 1.0 - cos_sim
                                        step_thresh, raw_step_meltdown = robust_threshold_step(step_e_t, threshold_state)
                                        if threshold_state.dynamic_release_threshold:
                                            step_thresh_release = _to_float_or_nan(
                                                threshold_state.dynamic_release_threshold[-1]
                                            )
                                        step_meltdown = bool(raw_step_meltdown)
                                        if step_meltdown:
                                            logging.warning(
                                                "[MELTDOWN] task=%s:%d ep=%d t_env=%d | E_t=%.4f > thresh=%.4f",
                                                task_group,
                                                task_id,
                                                episode_idx,
                                                t_env,
                                                step_e_t,
                                                step_thresh,
                                            )
                                            guidance_replan_candidate = True
                                            guidance_replan = bool(cfg.meltdown_use_guidance_replan)
                                            guidance_current_z_t = np.asarray(z_t_vec, dtype=np.float32)
                                            guidance_current_delta_z_t = np.asarray(delta_z_real, dtype=np.float32)
                                            guidance_current_delta_z_right = np.asarray(delta_z_pred, dtype=np.float32)
                                            action_plan.clear()
                                            raw_action_plan.clear()
                                        delta_z_pred_queue.popleft()

                            if (
                                env_baseline is not None
                                and observation_baseline is not None
                                and not baseline_done
                                and t_env >= safety_k
                                and len(delta_z_pred_queue_baseline) > 0
                            ):
                                target_t_b, delta_z_pred_b = delta_z_pred_queue_baseline[0]
                                if target_t_b == t_env:
                                    z_t_minus_k_b = None
                                    target_hist_t_b = t_env - safety_k
                                    for hist_t_b, hist_z_b in z_history_baseline:
                                        if hist_t_b == target_hist_t_b:
                                            z_t_minus_k_b = hist_z_b
                                            break
                                    if z_t_minus_k_b is None or z_t_vec_baseline is None:
                                        delta_z_pred_queue_baseline.popleft()
                                    else:
                                        delta_z_real_b = z_t_vec_baseline - z_t_minus_k_b
                                        if np.linalg.norm(delta_z_real_b) < 1e-4:
                                            delta_z_real_b = delta_z_real_b + (
                                                np.random.randn(*delta_z_real_b.shape).astype(np.float32) * 1e-4
                                            )
                                        delta_z_real_flat_b = _flatten_feature(delta_z_real_b)
                                        delta_z_pred_flat_b = _flatten_feature(delta_z_pred_b)
                                        cos_denom_b = np.linalg.norm(delta_z_real_flat_b) * np.linalg.norm(delta_z_pred_flat_b)
                                        cos_sim_b = (
                                            float(np.dot(delta_z_real_flat_b, delta_z_pred_flat_b) / cos_denom_b)
                                            if cos_denom_b > 0
                                            else 0.0
                                        )
                                        step_e_t_baseline = 1.0 - cos_sim_b
                                        step_thresh_baseline, raw_step_meltdown_baseline = robust_threshold_step(
                                            step_e_t_baseline,
                                            threshold_state_baseline,
                                        )
                                        if threshold_state_baseline.dynamic_release_threshold:
                                            step_thresh_release_baseline = _to_float_or_nan(
                                                threshold_state_baseline.dynamic_release_threshold[-1]
                                            )
                                        step_meltdown_baseline = bool(raw_step_meltdown_baseline)
                                        delta_z_pred_queue_baseline.popleft()

                            if not action_plan:
                                def _infer_once(
                                    *,
                                    run_guidance_replan: bool,
                                    compare_baseline: bool,
                                ) -> tuple[dict[str, Any], float]:
                                    infer_start_local = dt.datetime.now()
                                    resp_local = policy.predict_action_chunk_with_aux(
                                        model_input,
                                        guidance_replan=bool(run_guidance_replan),
                                        guidance_current_z_t=(
                                            guidance_current_z_t if bool(run_guidance_replan) else None
                                        ),
                                        guidance_current_z_hist_t=(
                                            guidance_current_z_hist_t if bool(run_guidance_replan) else None
                                        ),
                                        guidance_current_delta_z_t=(
                                            guidance_current_delta_z_t if bool(run_guidance_replan) else None
                                        ),
                                        guidance_current_delta_z_right=(
                                            guidance_current_delta_z_right if bool(run_guidance_replan) else None
                                        ),
                                        guidance_apply_every=int(cfg.guidance_apply_every),
                                        guidance_loss_objective=str(cfg.guidance_loss_objective),
                                        guidance_compare_baseline=bool(compare_baseline),
                                    )
                                    infer_end_local = dt.datetime.now()
                                    infer_ms_local = float((infer_end_local - infer_start_local).total_seconds() * 1000.0)
                                    return resp_local, infer_ms_local

                                infer_resp: dict[str, Any]
                                infer_ms: float
                                selected_guidance_replan = bool(guidance_replan)
                                has_replan_payload = bool(
                                    guidance_replan_candidate
                                    and guidance_current_z_t is not None
                                    and guidance_current_delta_z_t is not None
                                    and guidance_current_delta_z_right is not None
                                )

                                if bool(cfg.benchmark_ogg_timing) and has_replan_payload:
                                    rng_before = _capture_rng_state()
                                    infer_resp, infer_ms = _infer_once(
                                        run_guidance_replan=selected_guidance_replan,
                                        compare_baseline=bool(cfg.guidance_compare_baseline),
                                    )
                                    rng_after_actual = _capture_rng_state()
                                    _restore_rng_state(rng_before)
                                    _shadow_resp, shadow_ms = _infer_once(
                                        run_guidance_replan=(not selected_guidance_replan),
                                        compare_baseline=False,
                                    )
                                    _restore_rng_state(rng_after_actual)

                                    benchmark_ogg_paired_replan_requests += 1
                                    if selected_guidance_replan:
                                        benchmark_ogg_with_ogg_ms_total += float(infer_ms)
                                        benchmark_ogg_no_ogg_ms_total += float(shadow_ms)
                                    else:
                                        benchmark_ogg_no_ogg_ms_total += float(infer_ms)
                                        benchmark_ogg_with_ogg_ms_total += float(shadow_ms)

                                    logging.info(
                                        "[BENCH_OGG] task=%s:%d ep=%d t_env=%d paired_replan=%d selected=%s ms_selected=%.3f ms_other=%.3f",
                                        task_group,
                                        task_id,
                                        episode_idx,
                                        t_env,
                                        benchmark_ogg_paired_replan_requests,
                                        "with_ogg" if selected_guidance_replan else "without_ogg",
                                        infer_ms,
                                        shadow_ms,
                                    )
                                else:
                                    infer_resp, infer_ms = _infer_once(
                                        run_guidance_replan=selected_guidance_replan,
                                        compare_baseline=bool(cfg.guidance_compare_baseline),
                                    )
                                    benchmark_ogg_no_ogg_ms_total += float(infer_ms)
                                    benchmark_ogg_with_ogg_ms_total += float(infer_ms)

                                action_chunk_infer_requests += 1
                                action_chunk_infer_time_ms_total += infer_ms

                                action_chunk = infer_resp["actions"]
                                raw_action_chunk = infer_resp.get("raw_actions", action_chunk)
                                if not isinstance(action_chunk, torch.Tensor):
                                    action_chunk = torch.as_tensor(action_chunk, dtype=torch.float32, device=device)
                                if not isinstance(raw_action_chunk, torch.Tensor):
                                    raw_action_chunk = torch.as_tensor(raw_action_chunk, dtype=torch.float32, device=device)
                                if action_chunk.shape[1] < int(action_queue_fill_steps):
                                    raise ValueError(
                                        "Action chunk shorter than policy.n_action_steps: "
                                        f"{action_chunk.shape[1]} < {action_queue_fill_steps}."
                                    )
                                if raw_action_chunk.shape[1] < int(action_queue_fill_steps):
                                    raise ValueError(
                                        "Raw action chunk shorter than policy.n_action_steps: "
                                        f"{raw_action_chunk.shape[1]} < {action_queue_fill_steps}."
                                    )
                                for i in range(int(action_queue_fill_steps)):
                                    action_plan.append(action_chunk[:, i, :])
                                    raw_action_plan.append(
                                        np.asarray(raw_action_chunk[0, i, :].detach().to("cpu", dtype=torch.float32).numpy())
                                    )

                                if guidance_replan:
                                    guidance_request_count += 1
                                    gstats = infer_resp.get("guidance_stats", {})
                                    if isinstance(gstats, dict):
                                        applied_steps = int(_to_float_or_nan(gstats.get("applied_steps")))
                                        guidance_applied_steps_total += max(0, applied_steps)
                                        if applied_steps > 0:
                                            guidance_effective_count += 1
                                        guidance_loss_values.append(_to_float_or_nan(gstats.get("avg_loss")))
                                        guidance_gnorm_values.append(_to_float_or_nan(gstats.get("avg_g_norm")))
                                        guidance_delta_v_norm_values.append(_to_float_or_nan(gstats.get("avg_delta_v_norm")))
                                        guidance_v_norm_pre_values.append(_to_float_or_nan(gstats.get("avg_v_norm_pre")))
                                        guidance_target_norm_values.append(_to_float_or_nan(gstats.get("avg_target_norm")))
                                        guidance_pred_norm_values.append(_to_float_or_nan(gstats.get("avg_pred_norm")))
                                        guidance_infer_ms_values.append(_to_float_or_nan(infer_ms))

                                        compare_action_delta_l2 = _to_float_or_nan(gstats.get("action_delta_l2_mean"))
                                        compare_action_delta_l2_max = _to_float_or_nan(gstats.get("action_delta_l2_max"))
                                        compare_action_delta_ratio = _to_float_or_nan(
                                            gstats.get("action_delta_l2_over_baseline_mean")
                                        )
                                        compare_raw_action_delta_l2 = _to_float_or_nan(
                                            gstats.get("raw_action_delta_l2_mean")
                                        )
                                        compare_raw_action_delta_l2_max = _to_float_or_nan(
                                            gstats.get("raw_action_delta_l2_max")
                                        )
                                        if np.isfinite(compare_action_delta_l2):
                                            guidance_action_delta_l2_values.append(compare_action_delta_l2)
                                            guidance_action_delta_by_frame[int(t_env)] = compare_action_delta_l2
                                        if np.isfinite(compare_action_delta_l2_max):
                                            guidance_action_delta_l2_max_values.append(compare_action_delta_l2_max)
                                        if np.isfinite(compare_action_delta_ratio):
                                            guidance_action_delta_ratio_values.append(compare_action_delta_ratio)
                                        if np.isfinite(compare_raw_action_delta_l2):
                                            guidance_raw_action_delta_l2_values.append(compare_raw_action_delta_l2)
                                        if np.isfinite(compare_raw_action_delta_l2_max):
                                            guidance_raw_action_delta_l2_max_values.append(compare_raw_action_delta_l2_max)
                                        guidance_event_rows.append(
                                            {
                                                "task_group": task_group,
                                                "task_id": int(task_id),
                                                "episode_idx": int(episode_idx),
                                                "guidance_request_idx": int(guidance_request_count),
                                                "frame_idx": int(t_env),
                                                "t_env": int(t_env),
                                                "applied_steps": int(applied_steps),
                                                "scheduled_steps": _to_float_or_nan(gstats.get("scheduled_steps")),
                                                "configured_apply_every": _to_float_or_nan(
                                                    gstats.get("configured_apply_every")
                                                ),
                                                "loss_objective": str(gstats.get("loss_objective", "n/a")),
                                                "avg_loss": _to_float_or_nan(gstats.get("avg_loss")),
                                                "avg_target_norm": _to_float_or_nan(gstats.get("avg_target_norm")),
                                                "avg_pred_norm": _to_float_or_nan(gstats.get("avg_pred_norm")),
                                                "avg_g_norm": _to_float_or_nan(gstats.get("avg_g_norm")),
                                                "avg_eta_g_norm": _to_float_or_nan(gstats.get("avg_delta_v_norm")),
                                                "avg_v_norm_pre": _to_float_or_nan(gstats.get("avg_v_norm_pre")),
                                                "infer_ms": _to_float_or_nan(infer_ms),
                                                "action_delta_l2_mean": compare_action_delta_l2,
                                                "action_delta_l2_max": compare_action_delta_l2_max,
                                                "action_delta_l2_over_baseline_mean": compare_action_delta_ratio,
                                                "raw_action_delta_l2_mean": compare_raw_action_delta_l2,
                                                "raw_action_delta_l2_max": compare_raw_action_delta_l2_max,
                                                "action_delta_first7": json.dumps(gstats.get("action_delta_first7", [])),
                                            }
                                        )
                                        logging.info(
                                            "[GUIDE] task=%s:%d ep=%d t_env=%d req=%d applied=%d/%s every=%s obj=%s "
                                            "loss=%s g_norm=%s eta_g_norm=%s v_norm_pre=%s target_norm=%s pred_norm=%s delta_l2=%s",
                                            task_group,
                                            task_id,
                                            episode_idx,
                                            t_env,
                                            guidance_request_count,
                                            applied_steps,
                                            gstats.get("scheduled_steps"),
                                            gstats.get("configured_apply_every"),
                                            gstats.get("loss_objective"),
                                            gstats.get("avg_loss"),
                                            gstats.get("avg_g_norm"),
                                            gstats.get("avg_delta_v_norm"),
                                            gstats.get("avg_v_norm_pre"),
                                            gstats.get("avg_target_norm"),
                                            gstats.get("avg_pred_norm"),
                                            gstats.get("action_delta_l2_mean"),
                                        )

                            if (
                                env_baseline is not None
                                and observation_baseline is not None
                                and not baseline_done
                                and model_input_baseline is not None
                                and not baseline_action_plan
                            ):
                                infer_resp_baseline = policy.predict_action_chunk_with_aux(
                                    model_input_baseline,
                                    guidance_replan=False,
                                    guidance_current_z_t=None,
                                    guidance_current_z_hist_t=None,
                                    guidance_current_delta_z_t=None,
                                    guidance_current_delta_z_right=None,
                                    guidance_apply_every=int(cfg.guidance_apply_every),
                                    guidance_loss_objective=str(cfg.guidance_loss_objective),
                                    guidance_compare_baseline=False,
                                )
                                baseline_chunk = infer_resp_baseline["actions"]
                                baseline_raw_chunk = infer_resp_baseline.get("raw_actions", baseline_chunk)
                                if not isinstance(baseline_chunk, torch.Tensor):
                                    baseline_chunk = torch.as_tensor(baseline_chunk, dtype=torch.float32, device=device)
                                if not isinstance(baseline_raw_chunk, torch.Tensor):
                                    baseline_raw_chunk = torch.as_tensor(
                                        baseline_raw_chunk, dtype=torch.float32, device=device
                                    )
                                if baseline_chunk.shape[1] < int(action_queue_fill_steps):
                                    raise ValueError(
                                        "Baseline action chunk shorter than policy.n_action_steps: "
                                        f"{baseline_chunk.shape[1]} < {action_queue_fill_steps}."
                                    )
                                for i in range(int(action_queue_fill_steps)):
                                    baseline_action_plan.append(baseline_chunk[:, i, :])
                                    baseline_raw_action_plan.append(
                                        np.asarray(
                                            baseline_raw_chunk[0, i, :].detach().to("cpu", dtype=torch.float32).numpy()
                                        )
                                    )

                            action_tensor = action_plan.popleft()
                            raw_action_t = raw_action_plan.popleft()
                            baseline_action_tensor = baseline_action_plan.popleft() if baseline_action_plan else action_tensor
                            baseline_raw_action_t = (
                                baseline_raw_action_plan.popleft() if baseline_raw_action_plan else raw_action_t
                            )

                            action_np = _action_to_env_numpy(
                                action_tensor=action_tensor,
                                postprocessor=postprocessor,
                                env_postprocessor=env_postprocessor,
                            )
                            action_np_baseline = _action_to_env_numpy(
                                action_tensor=baseline_action_tensor,
                                postprocessor=postprocessor,
                                env_postprocessor=env_postprocessor,
                            )

                            raw_action_for_safety = np.asarray(raw_action_t, dtype=np.float32).reshape(-1)
                            if raw_action_for_safety.size > predictor.action_dim:
                                raw_action_for_safety = raw_action_for_safety[: predictor.action_dim]
                            elif raw_action_for_safety.size < predictor.action_dim:
                                raise ValueError(
                                    f"raw action too short for safety predictor: got {raw_action_for_safety.size}, "
                                    f"expected >= {predictor.action_dim}"
                                )
                            delta_z_pred_vec = predictor.predict_delta_z(z_t_vec, raw_action_for_safety)
                            delta_z_pred_queue.append((t_env + safety_k, np.asarray(delta_z_pred_vec, dtype=np.float32)))

                            observation, reward, terminated, truncated, info = env.step(action_np)
                            done = bool(terminated[0] or truncated[0])
                            success = success or _extract_success(info)
                            if env_baseline is not None and observation_baseline is not None and not baseline_done:
                                observation_baseline, _reward_b, terminated_b, truncated_b, info_b = env_baseline.step(
                                    action_np_baseline
                                )
                                baseline_t_env += 1
                                baseline_success = baseline_success or _extract_success(info_b)
                                baseline_done = bool(terminated_b[0] or truncated_b[0])

                            if (
                                env_baseline is not None
                                and z_t_vec_baseline is not None
                                and observation_baseline is not None
                                and not baseline_done
                            ):
                                raw_action_for_safety_b = np.asarray(baseline_raw_action_t, dtype=np.float32).reshape(-1)
                                if raw_action_for_safety_b.size > predictor.action_dim:
                                    raw_action_for_safety_b = raw_action_for_safety_b[: predictor.action_dim]
                                elif raw_action_for_safety_b.size < predictor.action_dim:
                                    raise ValueError(
                                        f"baseline raw action too short for safety predictor: got {raw_action_for_safety_b.size}, "
                                        f"expected >= {predictor.action_dim}"
                                    )
                                delta_z_pred_vec_b = predictor.predict_delta_z(z_t_vec_baseline, raw_action_for_safety_b)
                                delta_z_pred_queue_baseline.append(
                                    (t_env + safety_k, np.asarray(delta_z_pred_vec_b, dtype=np.float32))
                                )

                            et_values.append(_to_float_or_nan(step_e_t))
                            thresh_values.append(_to_float_or_nan(step_thresh))
                            thresh_release_values.append(_to_float_or_nan(step_thresh_release))
                            meltdown_flags.append(bool(step_meltdown))
                            t_env_values.append(int(t_env))
                            reward_values.append(float(reward[0]))
                            if baseline_step_active:
                                et_values_baseline.append(_to_float_or_nan(step_e_t_baseline))
                                thresh_values_baseline.append(_to_float_or_nan(step_thresh_baseline))
                                thresh_release_values_baseline.append(_to_float_or_nan(step_thresh_release_baseline))
                                meltdown_flags_baseline.append(bool(step_meltdown_baseline))
                                t_env_values_baseline.append(int(t_env))

                            t_env += 1
                            step_prog.update(1)
                            if not disable_progress:
                                running_success_rate = (
                                    100.0 * float(total_successes + (1 if success else 0)) / float(max(1, total_episodes + 1))
                                )
                                step_prog.set_postfix(
                                    {
                                        "running_success_rate": f"{running_success_rate:.1f}%",
                                        "infer_req": int(action_chunk_infer_requests),
                                        "meltdowns": int(sum(meltdown_flags)),
                                    }
                                )
                        except Exception as exc:
                            episode_error = str(exc)
                            logging.exception(
                                "Episode crashed | task=%s:%d ep=%d", task_group, task_id, episode_idx
                            )
                            break
                    step_prog.close()

                    task_episodes += 1
                    total_episodes += 1
                    if success:
                        task_successes += 1
                        total_successes += 1
                    baseline_enabled_for_ep = bool(env_baseline is not None)
                    if baseline_enabled_for_ep:
                        task_baseline_episodes += 1
                        total_baseline_episodes += 1
                        if baseline_success:
                            task_baseline_successes += 1
                            total_baseline_successes += 1

                    # Baseline arrays are produced from independent no-guidance rollout when enabled.

                    suffix = "success" if success else "failure"
                    task_segment = _safe_slug(task_description)
                    episode_tag = f"task{task_id:03d}_ep{episode_idx:03d}_{suffix}_{task_segment}"

                    summary_row = _make_episode_summary(
                        task_group=task_group,
                        task_id=task_id,
                        task_description=task_description,
                        episode_idx=episode_idx,
                        success=success,
                        t_env=t_env,
                        et_values=et_values,
                        thresh_values=thresh_values,
                        meltdown_flags=meltdown_flags,
                        episode_error=episode_error,
                    )
                    baseline_summary_row = (
                        _make_episode_summary(
                            task_group=task_group,
                            task_id=task_id,
                            task_description=task_description,
                            episode_idx=episode_idx,
                            success=baseline_success,
                            t_env=baseline_t_env,
                            et_values=et_values_baseline,
                            thresh_values=thresh_values_baseline,
                            meltdown_flags=meltdown_flags_baseline,
                            episode_error=episode_error,
                        )
                        if baseline_enabled_for_ep
                        else None
                    )
                    summary_row.update(
                        {
                            "guidance_requests": int(guidance_request_count),
                            "guidance_effective_requests": int(guidance_effective_count),
                            "guidance_total_applied_steps": int(guidance_applied_steps_total),
                            "guidance_avg_loss": _safe_nanmean(guidance_loss_values),
                            "guidance_avg_target_norm": _safe_nanmean(guidance_target_norm_values),
                            "guidance_avg_pred_norm": _safe_nanmean(guidance_pred_norm_values),
                            "guidance_avg_g_norm": _safe_nanmean(guidance_gnorm_values),
                            "guidance_avg_eta_g_norm": _safe_nanmean(guidance_delta_v_norm_values),
                            "guidance_avg_v_norm_pre": _safe_nanmean(guidance_v_norm_pre_values),
                            "guidance_avg_infer_ms": _safe_nanmean(guidance_infer_ms_values),
                            "guidance_compare_enabled": int(bool(cfg.guidance_compare_baseline)),
                            "guidance_compare_event_count": int(len(guidance_action_delta_l2_values)),
                            "guidance_action_delta_l2_mean": _safe_nanmean(guidance_action_delta_l2_values),
                            "guidance_action_delta_l2_max": _safe_nanmax(guidance_action_delta_l2_max_values),
                            "guidance_action_delta_over_baseline_mean": _safe_nanmean(guidance_action_delta_ratio_values),
                            "guidance_raw_action_delta_l2_mean": _safe_nanmean(guidance_raw_action_delta_l2_values),
                            "guidance_raw_action_delta_l2_max": _safe_nanmax(guidance_raw_action_delta_l2_max_values),
                            "action_chunk_infer_requests": int(action_chunk_infer_requests),
                            "action_chunk_infer_time_ms": float(action_chunk_infer_time_ms_total),
                            "action_chunk_infer_time_sec": float(action_chunk_infer_time_ms_total / 1000.0),
                            "benchmark_ogg_timing_enabled": int(bool(cfg.benchmark_ogg_timing)),
                            "benchmark_ogg_paired_replan_requests": int(benchmark_ogg_paired_replan_requests),
                            "benchmark_infer_time_total_without_ogg_ms": float(benchmark_ogg_no_ogg_ms_total),
                            "benchmark_infer_time_total_with_ogg_ms": float(benchmark_ogg_with_ogg_ms_total),
                            "benchmark_infer_time_total_without_ogg_sec": float(benchmark_ogg_no_ogg_ms_total / 1000.0),
                            "benchmark_infer_time_total_with_ogg_sec": float(benchmark_ogg_with_ogg_ms_total / 1000.0),
                            "benchmark_infer_time_delta_with_minus_without_sec": float(
                                (benchmark_ogg_with_ogg_ms_total - benchmark_ogg_no_ogg_ms_total) / 1000.0
                            ),
                            "baseline_metrics_enabled": int(baseline_enabled_for_ep),
                            "baseline_success": (
                                int(bool(baseline_summary_row["success"]))
                                if baseline_summary_row is not None
                                else float("nan")
                            ),
                            "baseline_env_steps_executed": (
                                int(baseline_summary_row["env_steps_executed"])
                                if baseline_summary_row is not None
                                else float("nan")
                            ),
                            "baseline_num_et_valid": (
                                int(baseline_summary_row["num_et_valid"])
                                if baseline_summary_row is not None
                                else float("nan")
                            ),
                            "baseline_meltdown_count": (
                                int(baseline_summary_row["meltdown_count"])
                                if baseline_summary_row is not None
                                else float("nan")
                            ),
                            "baseline_meltdown_per_100_steps": (
                                float(baseline_summary_row["meltdown_per_100_steps"])
                                if baseline_summary_row is not None
                                else float("nan")
                            ),
                            "baseline_recovery_rate_after_meltdown": (
                                float(baseline_summary_row["recovery_rate_after_meltdown"])
                                if baseline_summary_row is not None
                                else float("nan")
                            ),
                            "baseline_first_meltdown_frame": (
                                int(baseline_summary_row["first_meltdown_frame"])
                                if baseline_summary_row is not None
                                else float("nan")
                            ),
                        }
                    )
                    task_episode_rows.append(summary_row)
                    global_episode_rows.append(summary_row)

                    can_visualize = True
                    if cfg.max_visualizations_per_task > 0 and visualized_task >= cfg.max_visualizations_per_task:
                        can_visualize = False
                    if cfg.max_visualizations_total > 0 and visualized_total >= cfg.max_visualizations_total:
                        can_visualize = False
                    if can_visualize:
                        visualized_task += 1
                        visualized_total += 1

                    if not replay_images_baseline:
                        replay_images_baseline = [np.asarray(img, dtype=np.uint8) for img in replay_images]
                    if not et_values_baseline:
                        et_values_baseline = list(et_values)
                        thresh_values_baseline = list(thresh_values)
                        thresh_release_values_baseline = list(thresh_release_values)
                        meltdown_flags_baseline = list(meltdown_flags)
                        t_env_values_baseline = list(t_env_values)

                    vis_trim_steps_raw = max(0, int(safety_k))
                    replay_images_vis_raw = replay_images[vis_trim_steps_raw:] if vis_trim_steps_raw > 0 else replay_images
                    replay_images_baseline_vis_raw = (
                        replay_images_baseline[vis_trim_steps_raw:] if vis_trim_steps_raw > 0 else replay_images_baseline
                    )

                    vis_trim_steps_curve = max(0, int(safety_k) + max(0, int(cfg.threshold_bootstrap_queue_size)))
                    replay_images_vis_curve = (
                        replay_images[vis_trim_steps_curve:] if vis_trim_steps_curve > 0 else replay_images
                    )
                    replay_images_baseline_vis_curve = (
                        replay_images_baseline[vis_trim_steps_curve:]
                        if vis_trim_steps_curve > 0
                        else replay_images_baseline
                    )
                    et_values_vis = et_values[vis_trim_steps_curve:] if vis_trim_steps_curve > 0 else et_values
                    thresh_values_vis = (
                        thresh_values[vis_trim_steps_curve:] if vis_trim_steps_curve > 0 else thresh_values
                    )
                    thresh_release_values_vis = (
                        thresh_release_values[vis_trim_steps_curve:]
                        if vis_trim_steps_curve > 0
                        else thresh_release_values
                    )
                    meltdown_flags_vis = (
                        meltdown_flags[vis_trim_steps_curve:] if vis_trim_steps_curve > 0 else meltdown_flags
                    )

                    et_values_baseline_vis = (
                        et_values_baseline[vis_trim_steps_curve:] if vis_trim_steps_curve > 0 else et_values_baseline
                    )
                    thresh_values_baseline_vis = (
                        thresh_values_baseline[vis_trim_steps_curve:]
                        if vis_trim_steps_curve > 0
                        else thresh_values_baseline
                    )
                    thresh_release_values_baseline_vis = (
                        thresh_release_values_baseline[vis_trim_steps_curve:]
                        if vis_trim_steps_curve > 0
                        else thresh_release_values_baseline
                    )
                    meltdown_flags_baseline_vis = (
                        meltdown_flags_baseline[vis_trim_steps_curve:]
                        if vis_trim_steps_curve > 0
                        else meltdown_flags_baseline
                    )
                    frame_indices_vis_curve = list(
                        range(vis_trim_steps_curve, vis_trim_steps_curve + len(et_values_vis))
                    )
                    t_env_values_vis_curve = (
                        t_env_values[vis_trim_steps_curve:] if vis_trim_steps_curve > 0 else t_env_values
                    )
                    frame_indices_baseline_vis_curve = list(
                        range(vis_trim_steps_curve, vis_trim_steps_curve + len(et_values_baseline_vis))
                    )
                    t_env_values_baseline_vis_curve = (
                        t_env_values_baseline[vis_trim_steps_curve:]
                        if vis_trim_steps_curve > 0
                        else t_env_values_baseline
                    )

                    if save_raw_video and can_visualize and replay_images_vis_raw:
                        imageio.mimwrite(
                            raw_video_dir / f"{episode_tag}.mp4",
                            [np.asarray(x, dtype=np.uint8) for x in replay_images_vis_raw],
                            fps=cfg.video_fps,
                        )
                    if (
                        save_guidance_dual_video
                        and can_visualize
                        and replay_images_vis_raw
                        and replay_images_baseline_vis_raw
                    ):
                        _save_side_by_side_video(
                            sync_dual_video_dir / f"{episode_tag}.mp4",
                            replay_images_vis_raw,
                            replay_images_baseline_vis_raw,
                            fps=cfg.video_fps,
                        )
                    if save_et_curve_plot and can_visualize and plt is not None:
                        _save_et_curve_plot(
                            plt,
                            et_plot_dir / f"{episode_tag}.png",
                            et_values=et_values_vis,
                            thresh_values=thresh_values_vis,
                            release_thresh_values=thresh_release_values_vis,
                            meltdown_flags=meltdown_flags_vis,
                            title=f"{task_description} | episode={episode_idx} | success={success}",
                            frame_indices=frame_indices_vis_curve,
                        )
                    if save_sync_video_with_curve and can_visualize and plt is not None:
                        _save_sync_video_with_curve(
                            plt,
                            sync_video_dir / f"{episode_tag}.mp4",
                            replay_images=replay_images_vis_curve,
                            et_values=et_values_vis,
                            thresh_values=thresh_values_vis,
                            release_thresh_values=thresh_release_values_vis,
                            meltdown_flags=meltdown_flags_vis,
                            fps=cfg.video_fps,
                            title=f"{task_description} | ep={episode_idx}",
                            frame_indices=frame_indices_vis_curve,
                            t_env_values=t_env_values_vis_curve,
                        )
                    if save_baseline_et_curve_plot and can_visualize and plt is not None:
                        _save_et_curve_plot(
                            plt,
                            et_plot_baseline_dir / f"{episode_tag}.png",
                            et_values=et_values_baseline_vis,
                            thresh_values=thresh_values_baseline_vis,
                            release_thresh_values=thresh_release_values_baseline_vis,
                            meltdown_flags=meltdown_flags_baseline_vis,
                            title=f"{task_description} | ep={episode_idx} | baseline(no-guidance)",
                            frame_indices=frame_indices_baseline_vis_curve,
                        )
                    if save_baseline_sync_video_with_curve and can_visualize and plt is not None:
                        _save_sync_video_with_curve(
                            plt,
                            sync_video_baseline_dir / f"{episode_tag}.mp4",
                            replay_images=replay_images_baseline_vis_curve,
                            et_values=et_values_baseline_vis,
                            thresh_values=thresh_values_baseline_vis,
                            release_thresh_values=thresh_release_values_baseline_vis,
                            meltdown_flags=meltdown_flags_baseline_vis,
                            fps=cfg.video_fps,
                            title=f"{task_description} | ep={episode_idx} | baseline(no-guidance)",
                            frame_indices=frame_indices_baseline_vis_curve,
                            t_env_values=t_env_values_baseline_vis_curve,
                        )
                    if save_episode_step_csv:
                        step_rows = []
                        for i in range(len(et_values)):
                            e_val = et_values[i]
                            th_val = thresh_values[i]
                            step_rows.append(
                                {
                                    "frame_idx": i,
                                    "t_env": t_env_values[i] if i < len(t_env_values) else i,
                                    "reward": reward_values[i] if i < len(reward_values) else float("nan"),
                                    "e_t": e_val,
                                    "threshold": th_val,
                                    "threshold_on": th_val,
                                    "threshold_off": (
                                        thresh_release_values[i] if i < len(thresh_release_values) else float("nan")
                                    ),
                                    "is_meltdown": int(bool(meltdown_flags[i])),
                                    "is_et_valid": int(np.isfinite(e_val) and np.isfinite(th_val)),
                                    "is_over_threshold": int(
                                        np.isfinite(e_val) and np.isfinite(th_val) and e_val > th_val
                                    ),
                                }
                            )
                        _write_rows_csv(step_csv_dir / f"{episode_tag}.csv", step_rows)
                    if save_baseline_step_csv:
                        baseline_rows = []
                        for i in range(len(et_values_baseline)):
                            e_val = et_values_baseline[i]
                            th_val = thresh_values_baseline[i]
                            baseline_rows.append(
                                {
                                    "frame_idx": i,
                                    "t_env": t_env_values_baseline[i] if i < len(t_env_values_baseline) else i,
                                    "e_t": e_val,
                                    "threshold": th_val,
                                    "threshold_on": th_val,
                                    "threshold_off": (
                                        thresh_release_values_baseline[i]
                                        if i < len(thresh_release_values_baseline)
                                        else float("nan")
                                    ),
                                    "is_meltdown": int(bool(meltdown_flags_baseline[i])),
                                    "is_et_valid": int(np.isfinite(e_val) and np.isfinite(th_val)),
                                    "is_over_threshold": int(
                                        np.isfinite(e_val) and np.isfinite(th_val) and e_val > th_val
                                    ),
                                    "is_baseline_no_guidance": 1,
                                }
                            )
                        _write_rows_csv(step_csv_baseline_dir / f"{episode_tag}.csv", baseline_rows)
                    if save_guidance_event_csv:
                        _write_rows_csv(
                            guidance_event_csv_dir / f"{episode_tag}.csv",
                            guidance_event_rows,
                            fieldnames=GUIDANCE_EVENT_COLUMNS,
                        )
                    if save_episode_npz:
                        npz_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            npz_dir / f"{episode_tag}.npz",
                            et=np.asarray(et_values, dtype=np.float32),
                            threshold=np.asarray(thresh_values, dtype=np.float32),
                            threshold_on=np.asarray(thresh_values, dtype=np.float32),
                            threshold_off=np.asarray(thresh_release_values, dtype=np.float32),
                            meltdown=np.asarray(meltdown_flags, dtype=np.int8),
                            t_env=np.asarray(t_env_values, dtype=np.int32),
                            reward=np.asarray(reward_values, dtype=np.float32),
                        )

                    logging.info(
                        "Episode done | task=%s:%d ep=%d success=%s | total_success=%d/%d",
                        task_group,
                        task_id,
                        episode_idx,
                        success,
                        total_successes,
                        total_episodes,
                    )
                    logging.info(
                        "[INFER_EP] task=%s:%d ep=%d env_steps=%d infer_chunk_requests=%d infer_time_ms=%.3f",
                        task_group,
                        task_id,
                        episode_idx,
                        t_env,
                        action_chunk_infer_requests,
                        action_chunk_infer_time_ms_total,
                    )
                    logging.info(
                        "[GUIDE_SUMMARY] task=%s:%d ep=%d req=%d eff=%d steps=%d "
                        "loss=%s g_norm=%s eta_g_norm=%s v_norm_pre=%s delta_l2=%s",
                        task_group,
                        task_id,
                        episode_idx,
                        guidance_request_count,
                        guidance_effective_count,
                        guidance_applied_steps_total,
                        _safe_nanmean(guidance_loss_values),
                        _safe_nanmean(guidance_gnorm_values),
                        _safe_nanmean(guidance_delta_v_norm_values),
                        _safe_nanmean(guidance_v_norm_pre_values),
                        _safe_nanmean(guidance_action_delta_l2_values),
                    )
                    episode_prog.update(1)
                    if not disable_progress:
                        running_success_rate = 100.0 * float(total_successes) / float(max(1, total_episodes))
                        episode_prog.set_postfix({"running_success_rate": f"{running_success_rate:.1f}%"})

                task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else float("nan")
                task_baseline_success_rate = (
                    float(task_baseline_successes) / float(task_baseline_episodes)
                    if task_baseline_episodes > 0
                    else float("nan")
                )
                task_meltdown_avg = (
                    float(np.mean([r["meltdown_count"] for r in task_episode_rows]))
                    if task_episode_rows
                    else float("nan")
                )
                task_et_valid_avg = (
                    float(np.mean([r["num_et_valid"] for r in task_episode_rows]))
                    if task_episode_rows
                    else float("nan")
                )
                avg_env_steps_per_episode = (
                    _safe_nanmean([float(r.get("env_steps_executed", float("nan"))) for r in task_episode_rows])
                    if task_episode_rows
                    else float("nan")
                )
                avg_action_chunk_infer_requests_per_episode = (
                    _safe_nanmean([float(r.get("action_chunk_infer_requests", float("nan"))) for r in task_episode_rows])
                    if task_episode_rows
                    else float("nan")
                )
                successful_episode_env_steps_total = (
                    float(
                        np.sum(
                            [
                                float(r.get("env_steps_executed", 0.0))
                                for r in task_episode_rows
                                if bool(r.get("success", False))
                            ]
                        )
                    )
                    if task_episode_rows
                    else float("nan")
                )
                all_episode_env_steps_total = (
                    float(np.sum([float(r.get("env_steps_executed", 0.0)) for r in task_episode_rows]))
                    if task_episode_rows
                    else float("nan")
                )
                infer_time_total_seconds = (
                    float(np.sum([float(r.get("action_chunk_infer_time_sec", 0.0)) for r in task_episode_rows]))
                    if task_episode_rows
                    else float("nan")
                )
                benchmark_infer_time_total_without_ogg_seconds = (
                    float(
                        np.sum(
                            [
                                float(r.get("benchmark_infer_time_total_without_ogg_sec", 0.0))
                                for r in task_episode_rows
                            ]
                        )
                    )
                    if task_episode_rows
                    else float("nan")
                )
                benchmark_infer_time_total_with_ogg_seconds = (
                    float(
                        np.sum(
                            [float(r.get("benchmark_infer_time_total_with_ogg_sec", 0.0)) for r in task_episode_rows]
                        )
                    )
                    if task_episode_rows
                    else float("nan")
                )
                benchmark_infer_time_delta_with_minus_without_seconds = (
                    float(benchmark_infer_time_total_with_ogg_seconds - benchmark_infer_time_total_without_ogg_seconds)
                    if np.isfinite(benchmark_infer_time_total_with_ogg_seconds)
                    and np.isfinite(benchmark_infer_time_total_without_ogg_seconds)
                    else float("nan")
                )
                benchmark_ogg_paired_replan_requests_total = (
                    int(np.sum([int(r.get("benchmark_ogg_paired_replan_requests", 0)) for r in task_episode_rows]))
                    if task_episode_rows
                    else 0
                )
                success_over_success_episode_env_steps = (
                    float(task_successes) / successful_episode_env_steps_total
                    if np.isfinite(successful_episode_env_steps_total) and successful_episode_env_steps_total > 0
                    else float("nan")
                )
                success_over_all_episode_env_steps = (
                    float(task_successes) / all_episode_env_steps_total
                    if np.isfinite(all_episode_env_steps_total) and all_episode_env_steps_total > 0
                    else float("nan")
                )
                success_over_infer_seconds = (
                    float(task_successes) / infer_time_total_seconds
                    if np.isfinite(infer_time_total_seconds) and infer_time_total_seconds > 0
                    else float("nan")
                )
                baseline_avg_meltdown_count = (
                    _safe_nanmean([float(r.get("baseline_meltdown_count", float("nan"))) for r in task_episode_rows])
                    if task_episode_rows
                    else float("nan")
                )
                baseline_avg_meltdown_per_100_steps = (
                    _safe_nanmean(
                        [float(r.get("baseline_meltdown_per_100_steps", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan")
                )
                baseline_avg_recovery_rate_after_meltdown = (
                    _safe_nanmean(
                        [float(r.get("baseline_recovery_rate_after_meltdown", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan")
                )
                corr_meltdown_count_vs_success = (
                    _safe_binary_correlation(
                        [float(r.get("meltdown_count", float("nan"))) for r in task_episode_rows],
                        [1.0 if bool(r.get("success", False)) else 0.0 for r in task_episode_rows],
                    )
                    if task_episode_rows
                    else float("nan")
                )
                baseline_corr_meltdown_count_vs_success = (
                    _safe_binary_correlation(
                        [float(r.get("baseline_meltdown_count", float("nan"))) for r in task_episode_rows],
                        [float(r.get("baseline_success", float("nan"))) for r in task_episode_rows],
                    )
                    if task_episode_rows
                    else float("nan")
                )
                task_row = {
                    "task_group": task_group,
                    "task_id": task_id,
                    "task_description": task_description,
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "success_rate": task_success_rate,
                    "baseline_episodes": task_baseline_episodes,
                    "baseline_successes": task_baseline_successes,
                    "baseline_success_rate": task_baseline_success_rate,
                    "avg_env_steps_per_episode": avg_env_steps_per_episode,
                    "avg_action_chunk_infer_requests_per_episode": avg_action_chunk_infer_requests_per_episode,
                    "successful_episode_env_steps_total": successful_episode_env_steps_total,
                    "all_episode_env_steps_total": all_episode_env_steps_total,
                    "infer_time_total_seconds": infer_time_total_seconds,
                    "benchmark_infer_time_total_without_ogg_seconds": benchmark_infer_time_total_without_ogg_seconds,
                    "benchmark_infer_time_total_with_ogg_seconds": benchmark_infer_time_total_with_ogg_seconds,
                    "benchmark_infer_time_delta_with_minus_without_seconds": (
                        benchmark_infer_time_delta_with_minus_without_seconds
                    ),
                    "benchmark_ogg_paired_replan_requests_total": benchmark_ogg_paired_replan_requests_total,
                    "success_over_success_episode_env_steps": success_over_success_episode_env_steps,
                    "success_over_all_episode_env_steps": success_over_all_episode_env_steps,
                    "success_over_infer_seconds": success_over_infer_seconds,
                    "avg_meltdown_count": task_meltdown_avg,
                    "avg_valid_et_count": task_et_valid_avg,
                    "avg_meltdown_per_100_steps": (
                        float(np.mean([r["meltdown_per_100_steps"] for r in task_episode_rows]))
                        if task_episode_rows
                        else float("nan")
                    ),
                    "avg_recovery_rate_after_meltdown": (
                        _safe_nanmean([float(r["recovery_rate_after_meltdown"]) for r in task_episode_rows])
                        if task_episode_rows
                        else float("nan")
                    ),
                    "avg_et_mean": (
                        _safe_nanmean([float(r["et_mean"]) for r in task_episode_rows])
                        if task_episode_rows
                        else float("nan")
                    ),
                    "avg_et_p95": (
                        _safe_nanmean([float(r["et_p95"]) for r in task_episode_rows])
                        if task_episode_rows
                        else float("nan")
                    ),
                    "avg_guidance_requests": _safe_nanmean(
                        [float(r.get("guidance_requests", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_effective_requests": _safe_nanmean(
                        [float(r.get("guidance_effective_requests", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_total_applied_steps": _safe_nanmean(
                        [float(r.get("guidance_total_applied_steps", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_loss": _safe_nanmean(
                        [float(r.get("guidance_avg_loss", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_g_norm": _safe_nanmean(
                        [float(r.get("guidance_avg_g_norm", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_eta_g_norm": _safe_nanmean(
                        [float(r.get("guidance_avg_eta_g_norm", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_v_norm_pre": _safe_nanmean(
                        [float(r.get("guidance_avg_v_norm_pre", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_compare_events": _safe_nanmean(
                        [float(r.get("guidance_compare_event_count", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "avg_guidance_action_delta_l2": _safe_nanmean(
                        [float(r.get("guidance_action_delta_l2_mean", float("nan"))) for r in task_episode_rows]
                    )
                    if task_episode_rows
                    else float("nan"),
                    "baseline_avg_meltdown_count": baseline_avg_meltdown_count,
                    "baseline_avg_meltdown_per_100_steps": baseline_avg_meltdown_per_100_steps,
                    "baseline_avg_recovery_rate_after_meltdown": baseline_avg_recovery_rate_after_meltdown,
                    "corr_meltdown_count_vs_success": corr_meltdown_count_vs_success,
                    "baseline_corr_meltdown_count_vs_success": baseline_corr_meltdown_count_vs_success,
                }
                task_rows.append(task_row)
                group_task_rows.append(task_row)
                logging.info(
                    "Task summary | %s:%d success_rate=%.4f avg_meltdown=%.4f",
                    task_group,
                    task_id,
                    task_success_rate,
                    task_meltdown_avg,
                )
                logging.info(
                    "[TASK_METRIC] %s:%d avg_env_steps/ep=%s avg_infer_req/ep=%s "
                    "succ/succ_ep_steps=%s succ/all_ep_steps=%s succ/infer_sec=%s",
                    task_group,
                    task_id,
                    task_row["avg_env_steps_per_episode"],
                    task_row["avg_action_chunk_infer_requests_per_episode"],
                    task_row["success_over_success_episode_env_steps"],
                    task_row["success_over_all_episode_env_steps"],
                    task_row["success_over_infer_seconds"],
                )
                logging.info(
                    "[GUIDE_TASK] %s:%d avg_req=%s avg_eff=%s avg_steps=%s avg_loss=%s "
                    "avg_g_norm=%s avg_eta_g_norm=%s avg_v_norm_pre=%s avg_delta_l2=%s",
                    task_group,
                    task_id,
                    task_row["avg_guidance_requests"],
                    task_row["avg_guidance_effective_requests"],
                    task_row["avg_guidance_total_applied_steps"],
                    task_row["avg_guidance_loss"],
                    task_row["avg_guidance_g_norm"],
                    task_row["avg_guidance_eta_g_norm"],
                    task_row["avg_guidance_v_norm_pre"],
                    task_row["avg_guidance_action_delta_l2"],
                )
                logging.info(
                    "[BASELINE_TASK] %s:%d succ_rate=%s avg_meltdown=%s avg_meltdown_per_100=%s corr(md,success)=%s",
                    task_group,
                    task_id,
                    task_row["baseline_success_rate"],
                    task_row["baseline_avg_meltdown_count"],
                    task_row["baseline_avg_meltdown_per_100_steps"],
                    task_row["baseline_corr_meltdown_count_vs_success"],
                )

            group_row = {
                "task_group": task_group,
                "num_tasks": len(group_task_rows),
                "group_avg_env_steps_per_episode": _safe_nanmean(
                    [float(r.get("avg_env_steps_per_episode", float("nan"))) for r in group_task_rows]
                )
                if group_task_rows
                else float("nan"),
                "group_avg_action_chunk_infer_requests_per_episode": _safe_nanmean(
                    [float(r.get("avg_action_chunk_infer_requests_per_episode", float("nan"))) for r in group_task_rows]
                )
                if group_task_rows
                else float("nan"),
                "group_avg_success_over_success_episode_env_steps": _safe_nanmean(
                    [float(r.get("success_over_success_episode_env_steps", float("nan"))) for r in group_task_rows]
                )
                if group_task_rows
                else float("nan"),
                "group_avg_success_over_all_episode_env_steps": _safe_nanmean(
                    [float(r.get("success_over_all_episode_env_steps", float("nan"))) for r in group_task_rows]
                )
                if group_task_rows
                else float("nan"),
                "group_avg_success_over_infer_seconds": _safe_nanmean(
                    [float(r.get("success_over_infer_seconds", float("nan"))) for r in group_task_rows]
                )
                if group_task_rows
                else float("nan"),
            }
            task_group_rows.append(group_row)
            logging.info(
                "[TASK_GROUP_METRIC] group=%s num_tasks=%d "
                "avg(metric1_env_steps/ep)=%s avg(metric2_infer_req/ep)=%s "
                "avg(metric3_succ/succ_ep_steps)=%s avg(metric4_succ/all_ep_steps)=%s "
                "avg(metric5_succ/infer_sec)=%s",
                task_group,
                group_row["num_tasks"],
                group_row["group_avg_env_steps_per_episode"],
                group_row["group_avg_action_chunk_infer_requests_per_episode"],
                group_row["group_avg_success_over_success_episode_env_steps"],
                group_row["group_avg_success_over_all_episode_env_steps"],
                group_row["group_avg_success_over_infer_seconds"],
            )

        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else float("nan")
        total_baseline_success_rate = (
            float(total_baseline_successes) / float(total_baseline_episodes)
            if total_baseline_episodes > 0
            else float("nan")
        )
        logging.info("Total success rate: %s", total_success_rate)
        logging.info("Total episodes: %d", total_episodes)
        logging.info("Baseline total success rate: %s", total_baseline_success_rate)
        logging.info("Baseline total episodes: %d", total_baseline_episodes)
        global_guidance_requests = _safe_nanmean(
            [float(r.get("guidance_requests", float("nan"))) for r in global_episode_rows]
        )
        global_guidance_effective = _safe_nanmean(
            [float(r.get("guidance_effective_requests", float("nan"))) for r in global_episode_rows]
        )
        global_guidance_steps = _safe_nanmean(
            [float(r.get("guidance_total_applied_steps", float("nan"))) for r in global_episode_rows]
        )
        global_guidance_loss = _safe_nanmean(
            [float(r.get("guidance_avg_loss", float("nan"))) for r in global_episode_rows]
        )
        global_guidance_g_norm = _safe_nanmean(
            [float(r.get("guidance_avg_g_norm", float("nan"))) for r in global_episode_rows]
        )
        global_guidance_eta_g_norm = _safe_nanmean(
            [float(r.get("guidance_avg_eta_g_norm", float("nan"))) for r in global_episode_rows]
        )
        global_guidance_v_norm_pre = _safe_nanmean(
            [float(r.get("guidance_avg_v_norm_pre", float("nan"))) for r in global_episode_rows]
        )
        global_guidance_delta_l2 = _safe_nanmean(
            [float(r.get("guidance_action_delta_l2_mean", float("nan"))) for r in global_episode_rows]
        )
        logging.info(
            "[GUIDE_GLOBAL] avg_req=%s avg_eff=%s avg_steps=%s avg_loss=%s "
            "avg_g_norm=%s avg_eta_g_norm=%s avg_v_norm_pre=%s avg_delta_l2=%s",
            global_guidance_requests,
            global_guidance_effective,
            global_guidance_steps,
            global_guidance_loss,
            global_guidance_g_norm,
            global_guidance_eta_g_norm,
            global_guidance_v_norm_pre,
            global_guidance_delta_l2,
        )
        global_corr_meltdown_count_vs_success = _safe_binary_correlation(
            [float(r.get("meltdown_count", float("nan"))) for r in global_episode_rows],
            [1.0 if bool(r.get("success", False)) else 0.0 for r in global_episode_rows],
        )
        baseline_global_avg_meltdown_count = _safe_nanmean(
            [float(r.get("baseline_meltdown_count", float("nan"))) for r in global_episode_rows]
        )
        baseline_global_avg_meltdown_per_100_steps = _safe_nanmean(
            [float(r.get("baseline_meltdown_per_100_steps", float("nan"))) for r in global_episode_rows]
        )
        baseline_global_avg_recovery_rate_after_meltdown = _safe_nanmean(
            [float(r.get("baseline_recovery_rate_after_meltdown", float("nan"))) for r in global_episode_rows]
        )
        baseline_corr_meltdown_count_vs_success = _safe_binary_correlation(
            [float(r.get("baseline_meltdown_count", float("nan"))) for r in global_episode_rows],
            [float(r.get("baseline_success", float("nan"))) for r in global_episode_rows],
        )
        global_action_chunk_infer_requests_total = (
            int(np.sum([int(r.get("action_chunk_infer_requests", 0)) for r in global_episode_rows]))
            if global_episode_rows
            else 0
        )
        global_benchmark_infer_time_total_without_ogg_sec = (
            float(
                np.sum(
                    [float(r.get("benchmark_infer_time_total_without_ogg_sec", 0.0)) for r in global_episode_rows]
                )
            )
            if global_episode_rows
            else float("nan")
        )
        global_benchmark_infer_time_total_with_ogg_sec = (
            float(np.sum([float(r.get("benchmark_infer_time_total_with_ogg_sec", 0.0)) for r in global_episode_rows]))
            if global_episode_rows
            else float("nan")
        )
        global_benchmark_infer_time_delta_with_minus_without_sec = (
            float(global_benchmark_infer_time_total_with_ogg_sec - global_benchmark_infer_time_total_without_ogg_sec)
            if np.isfinite(global_benchmark_infer_time_total_with_ogg_sec)
            and np.isfinite(global_benchmark_infer_time_total_without_ogg_sec)
            else float("nan")
        )
        global_benchmark_ogg_paired_replan_requests_total = (
            int(np.sum([int(r.get("benchmark_ogg_paired_replan_requests", 0)) for r in global_episode_rows]))
            if global_episode_rows
            else 0
        )
        global_action_chunk_infer_time_ms_total = (
            float(np.sum([float(r.get("action_chunk_infer_time_ms", 0.0)) for r in global_episode_rows]))
            if global_episode_rows
            else float("nan")
        )
        global_action_chunk_infer_time_sec_total = (
            float(global_action_chunk_infer_time_ms_total / 1000.0)
            if np.isfinite(global_action_chunk_infer_time_ms_total)
            else float("nan")
        )
        global_all_episode_env_steps_total = (
            int(np.sum([int(r.get("env_steps_executed", 0)) for r in global_episode_rows]))
            if global_episode_rows
            else 0
        )
        global_avg_action_chunk_infer_requests_per_episode = (
            _safe_nanmean([float(r.get("action_chunk_infer_requests", float("nan"))) for r in global_episode_rows])
            if global_episode_rows
            else float("nan")
        )
        global_avg_action_chunk_infer_time_ms_per_episode = (
            _safe_nanmean([float(r.get("action_chunk_infer_time_ms", float("nan"))) for r in global_episode_rows])
            if global_episode_rows
            else float("nan")
        )
        global_avg_action_chunk_infer_time_ms_per_request = (
            float(global_action_chunk_infer_time_ms_total / global_action_chunk_infer_requests_total)
            if np.isfinite(global_action_chunk_infer_time_ms_total) and global_action_chunk_infer_requests_total > 0
            else float("nan")
        )
        global_avg_action_chunk_infer_time_ms_per_env_step = (
            float(global_action_chunk_infer_time_ms_total / global_all_episode_env_steps_total)
            if np.isfinite(global_action_chunk_infer_time_ms_total) and global_all_episode_env_steps_total > 0
            else float("nan")
        )
        global_success_over_infer_seconds = (
            float(total_successes) / global_action_chunk_infer_time_sec_total
            if np.isfinite(global_action_chunk_infer_time_sec_total) and global_action_chunk_infer_time_sec_total > 0
            else float("nan")
        )
        logging.info(
            "[BASELINE_GLOBAL] succ_rate=%s avg_meltdown=%s avg_meltdown_per_100=%s corr(md,success)=%s",
            total_baseline_success_rate,
            baseline_global_avg_meltdown_count,
            baseline_global_avg_meltdown_per_100_steps,
            baseline_corr_meltdown_count_vs_success,
        )
        logging.info(
            "[INFER_GLOBAL] requests=%d infer_time_ms=%.3f infer_time_sec=%.3f "
            "avg_ms_per_request=%s avg_req_per_episode=%s avg_ms_per_env_step=%s success_per_infer_sec=%s",
            global_action_chunk_infer_requests_total,
            global_action_chunk_infer_time_ms_total,
            global_action_chunk_infer_time_sec_total,
            global_avg_action_chunk_infer_time_ms_per_request,
            global_avg_action_chunk_infer_requests_per_episode,
            global_avg_action_chunk_infer_time_ms_per_env_step,
            global_success_over_infer_seconds,
        )
        logging.info(
            "[BENCH_OGG_GLOBAL] enabled=%s paired_replan_requests=%d without_ogg_sec=%s with_ogg_sec=%s delta_with_minus_without_sec=%s",
            bool(cfg.benchmark_ogg_timing),
            global_benchmark_ogg_paired_replan_requests_total,
            global_benchmark_infer_time_total_without_ogg_sec,
            global_benchmark_infer_time_total_with_ogg_sec,
            global_benchmark_infer_time_delta_with_minus_without_sec,
        )

        if save_summary_csv:
            _write_rows_csv(summary_dir / "episode_summary.csv", global_episode_rows)
            _write_rows_csv(summary_dir / "task_summary.csv", task_rows)
            _write_rows_csv(summary_dir / "task_group_summary.csv", task_group_rows)

        if save_task_plots and plt is not None:
            _save_task_level_plots(plt, summary_plots_dir, task_rows)

        if save_summary_json:
            global_meltdown_mean = (
                float(np.mean([r["meltdown_count"] for r in global_episode_rows]))
                if global_episode_rows
                else float("nan")
            )
            global_json = {
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "total_success_rate": total_success_rate,
                "global_avg_meltdown_count_per_episode": global_meltdown_mean,
                "global_avg_meltdown_per_100_steps": (
                    float(np.mean([r["meltdown_per_100_steps"] for r in global_episode_rows]))
                    if global_episode_rows
                    else float("nan")
                ),
                "global_avg_recovery_rate_after_meltdown": (
                    _safe_nanmean([float(r["recovery_rate_after_meltdown"]) for r in global_episode_rows])
                    if global_episode_rows
                    else float("nan")
                ),
                "corr_meltdown_count_vs_success": global_corr_meltdown_count_vs_success,
                "baseline_total_episodes": total_baseline_episodes,
                "baseline_total_successes": total_baseline_successes,
                "baseline_total_success_rate": total_baseline_success_rate,
                "baseline_global_avg_meltdown_count_per_episode": baseline_global_avg_meltdown_count,
                "baseline_global_avg_meltdown_per_100_steps": baseline_global_avg_meltdown_per_100_steps,
                "baseline_global_avg_recovery_rate_after_meltdown": baseline_global_avg_recovery_rate_after_meltdown,
                "baseline_corr_meltdown_count_vs_success": baseline_corr_meltdown_count_vs_success,
                "global_all_episode_env_steps_total": global_all_episode_env_steps_total,
                "global_action_chunk_infer_requests_total": global_action_chunk_infer_requests_total,
                "global_action_chunk_infer_time_ms_total": global_action_chunk_infer_time_ms_total,
                "global_action_chunk_infer_time_sec_total": global_action_chunk_infer_time_sec_total,
                "global_avg_action_chunk_infer_requests_per_episode": (
                    global_avg_action_chunk_infer_requests_per_episode
                ),
                "global_avg_action_chunk_infer_time_ms_per_episode": (
                    global_avg_action_chunk_infer_time_ms_per_episode
                ),
                "global_avg_action_chunk_infer_time_ms_per_request": (
                    global_avg_action_chunk_infer_time_ms_per_request
                ),
                "global_avg_action_chunk_infer_time_ms_per_env_step": (
                    global_avg_action_chunk_infer_time_ms_per_env_step
                ),
                "global_success_over_infer_seconds": global_success_over_infer_seconds,
                "benchmark_ogg_timing_enabled": int(bool(cfg.benchmark_ogg_timing)),
                "global_benchmark_ogg_paired_replan_requests_total": global_benchmark_ogg_paired_replan_requests_total,
                "global_benchmark_infer_time_total_without_ogg_sec": global_benchmark_infer_time_total_without_ogg_sec,
                "global_benchmark_infer_time_total_with_ogg_sec": global_benchmark_infer_time_total_with_ogg_sec,
                "global_benchmark_infer_time_delta_with_minus_without_sec": (
                    global_benchmark_infer_time_delta_with_minus_without_sec
                ),
                "global_avg_guidance_requests_per_episode": global_guidance_requests,
                "global_avg_guidance_effective_requests_per_episode": global_guidance_effective,
                "global_avg_guidance_applied_steps_per_episode": global_guidance_steps,
                "global_avg_guidance_loss": global_guidance_loss,
                "global_avg_guidance_g_norm": global_guidance_g_norm,
                "global_avg_guidance_eta_g_norm": global_guidance_eta_g_norm,
                "global_avg_guidance_v_norm_pre": global_guidance_v_norm_pre,
                "global_avg_guidance_action_delta_l2": global_guidance_delta_l2,
                "task_group_rows": task_group_rows,
                "safety_model": {
                    "model_dir": str(predictor.model_dir),
                    "checkpoint_path": str(predictor.checkpoint_path),
                    "token_dim": predictor.token_dim,
                    "action_dim": predictor.action_dim,
                    "k_step_from_model": predictor.k_step,
                    "effective_safety_k": safety_k,
                },
                "paths": {
                    "raw_video_dir": str(raw_video_dir),
                    "analysis_root": str(analysis_root),
                    "episode_et_plot_dir": str(et_plot_dir),
                    "sync_video_dir": str(sync_video_dir),
                    "episode_et_plot_baseline_dir": str(et_plot_baseline_dir),
                    "sync_video_baseline_dir": str(sync_video_baseline_dir),
                    "sync_dual_video_dir": str(sync_dual_video_dir),
                    "episode_step_csv_dir": str(step_csv_dir),
                    "episode_step_csv_baseline_dir": str(step_csv_baseline_dir),
                    "episode_guidance_event_csv_dir": str(guidance_event_csv_dir),
                    "episode_npz_dir": str(npz_dir),
                    "summary_dir": str(summary_dir),
                },
                "task_rows": task_rows,
            }
            with open(summary_dir / "global_summary.json", "w", encoding="utf-8") as f:
                json.dump(global_json, f, ensure_ascii=False, indent=2)

    finally:
        episode_prog.close()
        close_envs(envs_map)
        if envs_map_baseline is not None:
            close_envs(envs_map_baseline)

    logging.info("Modified detection eval finished.")


def main() -> None:
    init_logging()
    register_third_party_plugins()
    eval_modified_detection()


if __name__ == "__main__":
    main()
