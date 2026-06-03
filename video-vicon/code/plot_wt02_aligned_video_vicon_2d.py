"""把 WT02 的视频与 Vicon 按时间对齐，逐帧并排出图(视频 2D + Vicon Y-Z 2D)，共 100 对。

做法：分别读取视频与 Vicon 的逐帧运动学，按"时间最近邻"把视频帧匹配到 Vicon 帧，
再分别复用 plot_wt02_video_metrics_2d 与 plot_wt02_kinematics_3d 中的渲染函数出图。
两个输出目录、CSV 路径等均可通过环境变量覆盖(便于在测试中重定向)。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# 复用 Vicon 侧的加载/配对/渲染函数。
from plot_wt02_kinematics_3d import (
    VICON_CSV,
    load_trial_kinematics,
    parse_trajectories,
    render_frame_yz_2d,
    render_overview_yz_2d,
    select_matching_frames,
    yz_limits_for_selection,
)
# 复用视频侧的加载/渲染函数(OUTPUT_CSV 重命名为 VIDEO_METRIC_CSV)。
from plot_wt02_video_metrics_2d import (
    KEYPOINTS_JSON,
    OUTPUT_CSV as VIDEO_METRIC_CSV,
    load_keypoint_rows,
    load_metric_rows,
    limits_for_selection as video_limits_for_selection,
    render_frame as render_video_frame,
    render_overview as render_video_overview,
)


TRIAL = "WT02"
FRAME_COUNT = 100
# Vicon 运动学 CSV：默认新版，缺失则回退旧版；均可被环境变量覆盖。
DEFAULT_VICON_METRIC_CSV = Path(r"H:\COM\video-vicon\validation\vicon_com_metric.csv")
FALLBACK_VICON_METRIC_CSV = Path(r"H:\COM\video-vicon\validation\vicon_com_kinematics.csv")
VICON_METRIC_CSV = Path(os.environ.get(
    "VICON_METRIC_CSV",
    str(DEFAULT_VICON_METRIC_CSV if DEFAULT_VICON_METRIC_CSV.exists() else FALLBACK_VICON_METRIC_CSV),
))
# 视频运动学 CSV(同样可被环境变量覆盖)。
VIDEO_METRIC_CSV = Path(os.environ.get("VIDEO_METRIC_CSV", str(VIDEO_METRIC_CSV)))
# 两个输出目录(视频帧 / Vicon Y-Z 帧)。
OUTPUT_VIDEO_DIR = Path(os.environ.get(
    "ALIGNED_VIDEO_OUTPUT_DIR",
    r"H:\COM\video-vicon\validation\WT02_first50_aligned_video_metrics_2d",
))
OUTPUT_VICON_DIR = Path(os.environ.get(
    "ALIGNED_VICON_OUTPUT_DIR",
    r"H:\COM\video-vicon\validation\WT02_first50_aligned_vicon_metrics_yz_2d",
))


def match_video_rows_to_vicon_rows(
    video_rows: list[dict[str, Any]],
    vicon_rows: list[dict[str, Any]],
    limit: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """把每个视频帧按时间最近邻匹配到一个 Vicon 帧，最多返回 limit 对。"""
    if not vicon_rows:
        return []
    # Vicon 各帧时间，及其有效时间区间。
    vicon_times = np.array([row["time_s"] for row in vicon_rows], dtype=float)
    t_min = float(vicon_times.min())
    t_max = float(vicon_times.max())

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for video_row in video_rows:
        video_t = float(video_row["time_s"])
        # 视频时间超出 Vicon 覆盖范围的帧跳过。
        if video_t < t_min or video_t > t_max:
            continue
        # 找时间最接近的 Vicon 帧。
        nearest_idx = int(np.argmin(np.abs(vicon_times - video_t)))
        matched.append((video_row, vicon_rows[nearest_idx]))
        if len(matched) == limit:
            break
    return matched


def build_aligned_selections():
    """构造对齐后的视频与 Vicon 两套 selection(各含 FRAME_COUNT 对)。

    返回 (video_selection, vicon_selection)，元素分别为
    (视频指标行, 视频关键点行) 与 (Vicon 指标行, Vicon 轨迹帧)。
    """
    # 只取目标试验的视频/ Vicon 运动学行。
    video_rows = [
        row for row in load_metric_rows(VIDEO_METRIC_CSV)
        if row["trial"] == TRIAL
    ]
    vicon_rows = [
        row for row in load_trial_kinematics(VICON_METRIC_CSV, TRIAL)
        if row["trial"] == TRIAL
    ]
    # 按时间排序后再做最近邻匹配。
    video_rows.sort(key=lambda row: row["time_s"])
    vicon_rows.sort(key=lambda row: row["time_s"])

    matched_rows = match_video_rows_to_vicon_rows(video_rows, vicon_rows, FRAME_COUNT)
    if len(matched_rows) < FRAME_COUNT:
        raise RuntimeError(f"Only found {len(matched_rows)} aligned frames for {TRIAL}")

    # 加载关键点与 Vicon 轨迹，用于补齐绘图所需的几何数据。
    video_keypoints = load_keypoint_rows(KEYPOINTS_JSON, TRIAL)
    _, _, _, _, trajectory_frames = parse_trajectories(VICON_CSV)
    trajectory_by_frame = {frame["frame"]: frame for frame in trajectory_frames}

    video_selection = []
    vicon_selection = []
    for video_row, vicon_row in matched_rows:
        # 找到该视频帧的关键点与该 Vicon 帧的轨迹点。
        video_keypoint_row = video_keypoints.get(video_row["frame"])
        vicon_trajectory_row = trajectory_by_frame.get(vicon_row["frame"])
        # 任一缺失则跳过该对。
        if video_keypoint_row is None or vicon_trajectory_row is None:
            continue
        video_selection.append((video_row, video_keypoint_row))
        vicon_selection.append((vicon_row, vicon_trajectory_row))

    # 补齐几何数据后数量仍不足则报错。
    if len(video_selection) < FRAME_COUNT or len(vicon_selection) < FRAME_COUNT:
        raise RuntimeError(
            f"Only found {len(video_selection)} aligned frame pairs with keypoints/trajectories for {TRIAL}"
        )
    return video_selection[:FRAME_COUNT], vicon_selection[:FRAME_COUNT]


def main() -> int:
    """主流程：对齐后先各出一张总览图，再逐对输出视频帧与 Vicon Y-Z 帧。"""
    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VICON_DIR.mkdir(parents=True, exist_ok=True)
    video_selection, vicon_selection = build_aligned_selections()

    # 分别计算视频与 Vicon 的绘图范围。
    video_limits = video_limits_for_selection(video_selection)
    vicon_limits = yz_limits_for_selection(vicon_selection)

    # 各出一张总览图。
    render_video_overview(
        video_selection,
        video_limits,
        OUTPUT_VIDEO_DIR / "WT02_aligned_video_first50_overview_com_xcom_path.png",
    )
    render_overview_yz_2d(
        vicon_selection,
        vicon_limits,
        OUTPUT_VICON_DIR / "WT02_aligned_vicon_first50_overview_yz_com_xcom_path.png",
    )

    # 逐对(视频帧, Vicon 帧)并排出图，文件名含两者帧号与时间差(ms)。
    for index, ((video_metric, video_keypoints), (vicon_metric, vicon_traj)) in enumerate(
        zip(video_selection, vicon_selection)
    ):
        video_t = video_metric["time_s"]
        vicon_t = vicon_metric["time_s"]
        output_stem = (
            f"idx_{index + 1:02d}_video_{video_metric['frame']:04d}"
            f"_vicon_{vicon_metric['frame']:04d}_dt_{abs(video_t - vicon_t) * 1000:.1f}ms"
        )
        # 视频侧 2D 帧。
        render_video_frame(
            video_metric,
            video_keypoints,
            video_limits,
            OUTPUT_VIDEO_DIR / f"WT02_aligned_video_{output_stem}.png",
            index,
            len(video_selection),
        )
        # Vicon 侧 Y-Z 2D 帧。
        render_frame_yz_2d(
            vicon_metric,
            vicon_traj,
            vicon_limits,
            OUTPUT_VICON_DIR / f"WT02_aligned_vicon_{output_stem}.png",
            index,
            len(vicon_selection),
        )

    print(f"Video output: {OUTPUT_VIDEO_DIR}")
    print(f"Vicon output: {OUTPUT_VICON_DIR}")
    print(f"Aligned pairs: {len(video_selection)}")
    print(
        f"First pair: video frame {video_selection[0][0]['frame']} t={video_selection[0][0]['time_s']:.3f}s, "
        f"vicon frame {vicon_selection[0][0]['frame']} t={vicon_selection[0][0]['time_s']:.3f}s"
    )
    print(
        f"Last pair: video frame {video_selection[-1][0]['frame']} t={video_selection[-1][0]['time_s']:.3f}s, "
        f"vicon frame {vicon_selection[-1][0]['frame']} t={vicon_selection[-1][0]['time_s']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
