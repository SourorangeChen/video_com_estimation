"""用 keypoints 预处理后的视频指标，绘制 WT02 前 100 帧的视频侧 2D 运动学图。

复用 plot_wt02_video_metrics_2d 的渲染函数，但输入换成
metrics_keypoints_preprocessed/ 下的视频/ Vicon 指标 CSV 与预处理后关键点 JSON。
视频帧按时间最近邻匹配到 Vicon 帧(仅用于在文件名中标注时间差)。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

# 复用：时间最近邻匹配、Vicon 运动学加载、视频侧渲染函数。
from plot_wt02_aligned_video_vicon_2d import match_video_rows_to_vicon_rows
from plot_wt02_kinematics_3d import load_trial_kinematics
from plot_wt02_video_metrics_2d import (
    load_keypoint_rows,
    limits_for_selection,
    render_frame,
    render_overview,
)


TRIAL = "WT02"
FRAME_COUNT = 100
ROOT = Path(r"H:\COM\video-vicon")
# 输入：预处理后的视频/ Vicon 指标 CSV 与关键点 JSON；输出：视频帧图目录。
VIDEO_METRIC_CSV = ROOT / "validation" / "metrics_keypoints_preprocessed" / "video_com_metric.csv"
VICON_METRIC_CSV = ROOT / "validation" / "metrics_keypoints_preprocessed" / "vicon_com_metric.csv"
KEYPOINTS_JSON = (
    ROOT
    / "data"
    / "Chenzixuan"
    / "Video"
    / "video_keypoints-preprocessed"
    / "results"
    / "keypoints_and_com_preprocessed.json"
)
OUTPUT_DIR = ROOT / "validation" / "metrics_keypoints_preprocessed" / "WT02_first100_video_metrics"


def load_numeric_metric_rows(csv_path: Path) -> list[dict[str, Any]]:
    """读取指标 CSV；trial/frame 之外的列尽量转 float(转不动则保留原值)。"""
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted: dict[str, Any] = {
                "trial": row["trial"],
                "frame": int(row["frame"]),
            }
            for key, value in row.items():
                if key in {"trial", "frame"}:
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    # 非数值列(如方法名)保留字符串。
                    converted[key] = value
            rows.append(converted)
    return rows


def build_video_selection() -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    """构造前 100 帧的视频 selection 与对应的 Vicon 行(按时间最近邻匹配)。"""
    # 只取目标试验的视频/ Vicon 行。
    video_rows = [
        row for row in load_numeric_metric_rows(VIDEO_METRIC_CSV)
        if row["trial"] == TRIAL
    ]
    vicon_rows = [
        row for row in load_trial_kinematics(VICON_METRIC_CSV, TRIAL)
        if row["trial"] == TRIAL
    ]
    video_rows.sort(key=lambda row: row["time_s"])
    vicon_rows.sort(key=lambda row: row["time_s"])

    # 时间最近邻匹配。
    matched_rows = match_video_rows_to_vicon_rows(video_rows, vicon_rows, FRAME_COUNT)
    if len(matched_rows) < FRAME_COUNT:
        raise RuntimeError(f"Only found {len(matched_rows)} aligned frames for {TRIAL}")

    # 补齐每帧的关键点(绘图需要)。
    keypoint_rows = load_keypoint_rows(KEYPOINTS_JSON, TRIAL)
    video_selection: list[tuple[dict[str, Any], dict[str, Any]]] = []
    matched_vicon_rows: list[dict[str, Any]] = []
    for video_row, vicon_row in matched_rows:
        keypoint_row = keypoint_rows.get(video_row["frame"])
        # 缺关键点则跳过该对。
        if keypoint_row is None:
            continue
        video_selection.append((video_row, keypoint_row))
        matched_vicon_rows.append(vicon_row)

    if len(video_selection) < FRAME_COUNT:
        raise RuntimeError(f"Only found {len(video_selection)} video frames with preprocessed keypoints")
    return video_selection[:FRAME_COUNT], matched_vicon_rows[:FRAME_COUNT]


def main() -> int:
    """主流程：构造对齐帧 -> 出总览图 -> 逐帧出视频 2D 图(文件名含时间差)。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_selection, matched_vicon_rows = build_video_selection()
    video_limits = limits_for_selection(video_selection)

    # 总览图。
    render_overview(
        video_selection,
        video_limits,
        OUTPUT_DIR / f"WT02_aligned_video_first{FRAME_COUNT}_overview_com_xcom_path.png",
    )

    # 逐帧出图，文件名标注视频/ Vicon 帧号与时间差(ms)。
    for index, ((video_metric, video_keypoints), vicon_metric) in enumerate(
        zip(video_selection, matched_vicon_rows)
    ):
        video_t = float(video_metric["time_s"])
        vicon_t = float(vicon_metric["time_s"])
        output_stem = (
            f"idx_{index + 1:02d}_video_{video_metric['frame']:04d}"
            f"_vicon_{vicon_metric['frame']:04d}_dt_{abs(video_t - vicon_t) * 1000:.1f}ms"
        )
        render_frame(
            video_metric,
            video_keypoints,
            video_limits,
            OUTPUT_DIR / f"WT02_aligned_video_{output_stem}.png",
            index,
            len(video_selection),
        )

    print(f"Output: {OUTPUT_DIR}")
    print(f"Rendered video frames: {len(video_selection)}")
    print(
        f"First pair: video frame {video_selection[0][0]['frame']} t={video_selection[0][0]['time_s']:.3f}s, "
        f"vicon frame {matched_vicon_rows[0]['frame']} t={matched_vicon_rows[0]['time_s']:.3f}s"
    )
    print(
        f"Last pair: video frame {video_selection[-1][0]['frame']} t={video_selection[-1][0]['time_s']:.3f}s, "
        f"vicon frame {matched_vicon_rows[-1]['frame']} t={matched_vicon_rows[-1]['time_s']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
