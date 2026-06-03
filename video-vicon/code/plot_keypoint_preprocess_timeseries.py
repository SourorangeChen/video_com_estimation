"""绘制 keypoints 预处理前后的时序信号对比图。

每个 trial 输出三张图：
1. CoM raw vs keypoints-preprocessed 后重算 CoM；
2. 17 个关键点 x 坐标 raw vs preprocessed；
3. 17 个关键点 y 坐标 raw vs preprocessed。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# 原始与预处理后两个 JSON，以及输出图目录。
RAW_JSON = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json"
)
PREPROCESSED_JSON = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_keypoints-preprocessed\results\keypoints_and_com_preprocessed.json"
)
OUTPUT_DIR = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_keypoints-preprocessed\preprocess_plots"
)
VIDEO_FPS = 29.996

# COCO-17 关键点名称(索引顺序)。
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_trial_and_frame(image_path: str) -> tuple[str, int] | None:
    """从图片路径用正则解析出 (试验名, 帧号)；不匹配返回 None。"""
    match = re.search(r"Video_(?P<trial>.+?)_Trajectory.*?frame_(?P<frame>\d+)", image_path)
    if not match:
        return None
    return match.group("trial"), int(match.group("frame"))


def load_records_by_trial(json_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """读取 JSON，按 {试验: {帧号: 记录}} 组织。

    优先用 image 路径解析试验/帧号；解析不出则回退到记录里的 trial/frame 字段。
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON records in {json_path}")

    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in data:
        if not isinstance(record, dict):
            continue
        parsed = parse_trial_and_frame(str(record.get("image", "")))
        if parsed is None:
            trial = record.get("trial")
            frame = record.get("frame")
            if trial is None or frame is None:
                continue
            parsed = str(trial), int(frame)
        trial, frame = parsed
        grouped[trial][frame] = record
    return dict(grouped)


def keypoint_xy(record: dict[str, Any], keypoint_index: int) -> tuple[float, float]:
    """取某帧第 keypoint_index 个关键点的 (x, y)；缺失返回 (nan, nan)。"""
    keypoints = record.get("keypoints")
    if not isinstance(keypoints, list) or keypoint_index >= len(keypoints):
        return np.nan, np.nan
    point = keypoints[keypoint_index]
    if not isinstance(point, list) or len(point) < 2:
        return np.nan, np.nan
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return np.nan, np.nan


def com_xy(record: dict[str, Any]) -> tuple[float, float]:
    """取某帧 CoM 的 (x, y)；缺失返回 (nan, nan)。"""
    com = record.get("com")
    if not isinstance(com, dict):
        return np.nan, np.nan
    try:
        return float(com["com_x"]), float(com["com_y"])
    except (KeyError, TypeError, ValueError):
        return np.nan, np.nan


def common_trial_frames(
    raw_by_trial: dict[str, dict[int, dict[str, Any]]],
    processed_by_trial: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, list[int]]:
    """求每个试验中 raw 与 preprocessed 都存在的公共帧(排序后)。"""
    # 两侧都有的试验。
    trials = sorted(set(raw_by_trial) & set(processed_by_trial))
    return {
        # 两侧都有的帧号取交集并排序。
        trial: sorted(set(raw_by_trial[trial]) & set(processed_by_trial[trial]))
        for trial in trials
    }


def plot_com_trial(
    trial: str,
    frames: list[int],
    raw_records: dict[int, dict[str, Any]],
    processed_records: dict[int, dict[str, Any]],
    output_path: Path,
) -> None:
    """绘制单个 trial 的 CoM x/y 时序对比。"""
    time_s = np.array([(frame - frames[0]) / VIDEO_FPS for frame in frames], dtype=float)
    raw_xy = np.array([com_xy(raw_records[frame]) for frame in frames], dtype=float)
    processed_xy = np.array([com_xy(processed_records[frame]) for frame in frames], dtype=float)

    # 上下两子图：x、y 方向各画 raw vs 预处理曲线。
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis_index, axis_label in enumerate(["x", "y"]):
        ax = axes[axis_index]
        # raw 来自原始 PCT 输出，processed 来自滤波后的 keypoints 重算 CoM。
        ax.plot(time_s, raw_xy[:, axis_index], color="#64748B", linewidth=1.5, label="raw")
        ax.plot(time_s, processed_xy[:, axis_index], color="#2563EB", linewidth=1.5, label="median + S-G")
        ax.set_ylabel(f"CoM {axis_label} (px)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{trial} CoM time-series: raw vs preprocessed keypoints")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_keypoint_axis_trial(
    trial: str,
    frames: list[int],
    raw_records: dict[int, dict[str, Any]],
    processed_records: dict[int, dict[str, Any]],
    axis_index: int,
    output_path: Path,
) -> None:
    """绘制某个坐标轴(x 或 y)上 17 个关键点的 raw vs 预处理时序(5×4 子图网格)。"""
    time_s = np.array([(frame - frames[0]) / VIDEO_FPS for frame in frames], dtype=float)
    coord_label = "x" if axis_index == 0 else "y"

    # 5 行 4 列(20 个格子)放 17 个关键点，多余格子隐藏。
    fig, axes = plt.subplots(5, 4, figsize=(18, 15), sharex=True)
    flat_axes = axes.ravel()
    for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
        ax = flat_axes[keypoint_index]
        # 该关键点在所选轴上的 raw 与预处理序列。
        raw_values = np.array([
            keypoint_xy(raw_records[frame], keypoint_index)[axis_index]
            for frame in frames
        ], dtype=float)
        processed_values = np.array([
            keypoint_xy(processed_records[frame], keypoint_index)[axis_index]
            for frame in frames
        ], dtype=float)
        ax.plot(time_s, raw_values, color="#64748B", linewidth=1.0, label="raw")
        ax.plot(time_s, processed_values, color="#2563EB", linewidth=1.0, label="median + S-G")
        ax.set_title(f"{keypoint_index:02d} {keypoint_name}", fontsize=9)
        ax.set_ylabel(f"{coord_label} px", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    # 隐藏多余的空子图。
    for ax in flat_axes[len(KEYPOINT_NAMES):]:
        ax.axis("off")
    # 仅第一个子图显示图例；最后一行子图加横轴标签。
    flat_axes[0].legend(loc="best", fontsize=8)
    for ax in flat_axes[-4:]:
        ax.set_xlabel("time (s)", fontsize=8)

    fig.suptitle(f"{trial} keypoint {coord_label}-coordinate: raw vs preprocessed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    """主流程：对每个试验输出 CoM 对比图与 x/y 两张关键点对比图。"""
    raw_by_trial = load_records_by_trial(RAW_JSON)
    processed_by_trial = load_records_by_trial(PREPROCESSED_JSON)
    # 取每个试验两侧公共帧。
    frames_by_trial = common_trial_frames(raw_by_trial, processed_by_trial)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for trial, frames in frames_by_trial.items():
        if not frames:
            continue
        trial_dir = OUTPUT_DIR / trial
        trial_dir.mkdir(parents=True, exist_ok=True)
        # 图 1：CoM 对比。
        plot_com_trial(
            trial,
            frames,
            raw_by_trial[trial],
            processed_by_trial[trial],
            trial_dir / f"{trial}_com_raw_vs_preprocessed.png",
        )
        # 图 2：关键点 x 坐标对比。
        plot_keypoint_axis_trial(
            trial,
            frames,
            raw_by_trial[trial],
            processed_by_trial[trial],
            0,
            trial_dir / f"{trial}_keypoints_x_raw_vs_preprocessed.png",
        )
        # 图 3：关键点 y 坐标对比。
        plot_keypoint_axis_trial(
            trial,
            frames,
            raw_by_trial[trial],
            processed_by_trial[trial],
            1,
            trial_dir / f"{trial}_keypoints_y_raw_vs_preprocessed.png",
        )
        written += 3

    print(f"Raw input: {RAW_JSON}")
    print(f"Preprocessed input: {PREPROCESSED_JSON}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Trials: {len(frames_by_trial)}")
    print(f"Plots written: {written}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
