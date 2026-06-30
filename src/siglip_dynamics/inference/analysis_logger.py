from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class AnalysisLogger:
    """记录分析数据，生成E_t曲线、同步视频、baseline对比"""

    def __init__(self, output_dir: str | Path, save_videos: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_videos = save_videos

        # MyIdea数据
        self.error_history = []
        self.threshold_history = []
        self.meltdown_flags = []
        self.frames = []
        self.timestamps = []

        # Baseline数据
        self.baseline_error_history = []
        self.baseline_threshold_history = []
        self.baseline_meltdown_flags = []
        self.baseline_frames = []

    def log_step(self, t: int, error: float | None, threshold: float, is_meltdown: bool, frame=None):
        """记录MyIdea每一步的数据"""
        self.error_history.append(error if error is not None else float("nan"))
        self.threshold_history.append(threshold)
        self.meltdown_flags.append(is_meltdown)
        self.timestamps.append(t)
        if frame is not None and self.save_videos:
            self.frames.append(frame)

    def log_baseline_step(self, error: float | None, threshold: float, is_meltdown: bool, frame=None):
        """记录Baseline每一步的数据"""
        self.baseline_error_history.append(error if error is not None else float("nan"))
        self.baseline_threshold_history.append(threshold)
        self.baseline_meltdown_flags.append(is_meltdown)
        if frame is not None and self.save_videos:
            self.baseline_frames.append(frame)

    def _compute_stats(self, error_history, meltdown_flags):
        valid_errors = [e for e in error_history if not np.isnan(e)]
        return {
            "mean_error": float(np.mean(valid_errors)) if valid_errors else float("nan"),
            "max_error": float(np.max(valid_errors)) if valid_errors else float("nan"),
            "num_meltdowns": int(sum(meltdown_flags)),
            "meltdown_rate": float(sum(meltdown_flags) / len(meltdown_flags)) if meltdown_flags else 0.0,
        }

    def generate_analysis(self, episode_id: int) -> dict:
        """生成完整分析报告"""
        import json

        stats = {
            "myidea": self._compute_stats(self.error_history, self.meltdown_flags),
            "baseline": self._compute_stats(self.baseline_error_history, self.baseline_meltdown_flags),
        }

        # 保存原始数据为JSON
        data = {
            "episode_id": episode_id,
            "error_history": self.error_history,
            "threshold_history": self.threshold_history,
            "meltdown_flags": self.meltdown_flags,
            "timestamps": self.timestamps,
            "stats": stats
        }

        json_path = self.output_dir / f"episode_{episode_id}_data.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ 分析数据已保存: {json_path}")

        # 尝试生成图表（如果有matplotlib）
        try:
            self._plot_error_curve(episode_id, "myidea", self.error_history, self.threshold_history, self.meltdown_flags)
            self._plot_error_curve(episode_id, "baseline", self.baseline_error_history, self.baseline_threshold_history, self.baseline_meltdown_flags)
        except ImportError:
            print("⚠ matplotlib未安装，跳过图表生成")

        return stats

    def _plot_error_curve(self, episode_id, prefix, error_history, threshold_history, meltdown_flags):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps[:len(error_history)], error_history, "b-", label="E_t")
        plt.plot(self.timestamps[:len(threshold_history)], threshold_history, "r--", label="Threshold")

        meltdown_t = [self.timestamps[i] for i, flag in enumerate(meltdown_flags) if flag and i < len(self.timestamps)]
        meltdown_e = [error_history[i] for i, flag in enumerate(meltdown_flags) if flag and i < len(error_history)]
        if meltdown_t:
            plt.scatter(meltdown_t, meltdown_e, c="red", s=100, marker="x", label="Meltdown")

        plt.xlabel("Time Step")
        plt.ylabel("Error (1 - cosine similarity)")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / f"{prefix}_episode_{episode_id}_error_curve.png")
        plt.close()

    def _generate_sync_video(self, episode_id, prefix, frames, error_history, threshold_history, meltdown_flags):
        if not frames:
            return
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(self.output_dir / f"{prefix}_episode_{episode_id}_sync.mp4"), fourcc, 30.0, (1280, 720))
        for t, frame in enumerate(frames):
            left = cv2.resize(frame, (640, 720))
            right = self._render_error_plot(t, error_history, threshold_history, meltdown_flags)
            combined = np.hstack([left, right])
            out.write(combined)
        out.release()

    def _generate_dual_comparison_video(self, episode_id):
        if not self.frames or not self.baseline_frames:
            return
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(self.output_dir / f"dual_comparison_episode_{episode_id}.mp4"), fourcc, 30.0, (1280, 720))
        max_frames = max(len(self.frames), len(self.baseline_frames))
        for t in range(max_frames):
            left = self.frames[min(t, len(self.frames) - 1)]
            right = self.baseline_frames[min(t, len(self.baseline_frames) - 1)]
            left = cv2.resize(left, (640, 720))
            right = cv2.resize(right, (640, 720))
            cv2.putText(left, "MyIdea", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(right, "Baseline", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            combined = np.hstack([left, right])
            out.write(combined)
        out.release()

    def _render_error_plot(self, t, error_history, threshold_history, meltdown_flags):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2

        fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=100)
        ax.plot(error_history[:t + 1], "b-", label="E_t")
        ax.plot(threshold_history[:t + 1], "r--", label="Threshold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Error")
        ax.legend()
        ax.grid(True)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def reset(self):
        """重置所有数据"""
        self.error_history.clear()
        self.threshold_history.clear()
        self.meltdown_flags.clear()
        self.frames.clear()
        self.timestamps.clear()
        self.baseline_error_history.clear()
        self.baseline_threshold_history.clear()
        self.baseline_meltdown_flags.clear()
        self.baseline_frames.clear()
