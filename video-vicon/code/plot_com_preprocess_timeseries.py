"""绘制 CoM 平滑前后时序信号对比图。

输入 JSON 中 source_com 表示平滑前 CoM，com 表示平滑后 CoM。
脚本按 trial 输出 x/y 两个方向的 raw vs preprocessed 曲线。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# 输入(含 source_com 与 com)与输出图目录。
SMOOTH_JSON = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_com-preprocessed\results\keypoints_and_com_smooth.json"
)
OUTPUT_DIR = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_com-preprocessed\preprocess_plots"
)
VIDEO_FPS = 29.996


def load_records_by_trial(json_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """读取 JSON，按 {试验: {帧号: 记录}} 双层字典组织。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON records in {json_path}")

    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in data:
        if not isinstance(record, dict):
            continue
        trial = record.get("trial")
        frame = record.get("frame")
        if trial is None or frame is None:
            continue
        grouped[str(trial)][int(frame)] = record
    return dict(grouped)


def com_xy(record: dict[str, Any], key: str) -> tuple[float, float]:
    """从记录中取指定键(source_com 或 com)的 (x, y)；缺失返回 (nan, nan)。"""
    com = record.get(key)
    if not isinstance(com, dict):
        return np.nan, np.nan
    try:
        return float(com["com_x"]), float(com["com_y"])
    except (KeyError, TypeError, ValueError):
        return np.nan, np.nan


def plot_com_trial(
    trial: str,
    frames: list[int],
    records: dict[int, dict[str, Any]],
    output_path: Path,
) -> None:
    """绘制单个 trial 的 CoM x/y 时序对比图。"""
    # 时间轴(以首帧为 0)。
    time_s = np.array([(frame - frames[0]) / VIDEO_FPS for frame in frames], dtype=float)
    # source_com 是平滑前的 CoM，com 是当前 JSON 中已经写回的平滑后 CoM。
    raw_xy = np.array([com_xy(records[frame], "source_com") for frame in frames], dtype=float)
    smooth_xy = np.array([com_xy(records[frame], "com") for frame in frames], dtype=float)

    # 从首帧记录读取平滑方法/窗口，用作图例标签。
    smoothing = records[frames[0]].get("com_smoothing", {})
    if isinstance(smoothing, dict):
        method = smoothing.get("method", "smoothed")
        window = smoothing.get("window_frames")
        smooth_label = f"{method} ({window} frames)" if window else str(method)
    else:
        smooth_label = "smoothed"

    # 上下两子图分别画 x、y 方向的 raw vs 平滑曲线。
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis_index, axis_label in enumerate(["x", "y"]):
        ax = axes[axis_index]
        ax.plot(time_s, raw_xy[:, axis_index], color="#64748B", linewidth=1.4, label="raw")
        ax.plot(time_s, smooth_xy[:, axis_index], color="#2563EB", linewidth=1.6, label=smooth_label)
        ax.set_ylabel(f"CoM {axis_label} (px)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{trial} CoM time-series: raw vs preprocessed CoM")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    """主流程：按试验逐个绘制 CoM 平滑前后对比图。"""
    records_by_trial = load_records_by_trial(SMOOTH_JSON)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for trial in sorted(records_by_trial):
        frames = sorted(records_by_trial[trial])
        if not frames:
            continue
        # 每个试验一个子目录、一张对比图。
        trial_dir = OUTPUT_DIR / trial
        plot_com_trial(
            trial,
            frames,
            records_by_trial[trial],
            trial_dir / f"{trial}_com_raw_vs_preprocessed.png",
        )
        written += 1

    print(f"Input: {SMOOTH_JSON}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Trials: {len(records_by_trial)}")
    print(f"Plots written: {written}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
