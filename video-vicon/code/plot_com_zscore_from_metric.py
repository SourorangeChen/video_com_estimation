"""直接从运动学指标 CSV(而非原始 JSON/CSV)做视频 vs Vicon CoM 的 z 分数对比。

与 validate_com_normalized.py 思路一致，但输入是已算好的 video_com_metric*.csv 与
vicon_com_*.csv。复用 validate_com_normalized 中的信号处理函数(去漂移/z-score/相关/互相关)。
默认使用平滑版视频指标。各输入/输出路径均可通过环境变量覆盖。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import matplotlib

# 无界面后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 复用验证脚本中的信号处理与指标函数，保证算法一致。
from validate_com_normalized import (
    compute_metrics,
    compute_xcorr,
    remove_linear_drift,
    zscore_normalize,
)


VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
# 输入：视频(默认平滑版)与 Vicon 指标 CSV；输出：相关性 CSV 与 z 分数图目录。均可被环境变量覆盖。
VIDEO_METRIC_CSV = Path(os.environ.get(
    "VIDEO_METRIC_CSV",
    str(VALIDATION_DIR / "video_com_metric_smoothed.csv"),
))
VICON_METRIC_CSV = Path(os.environ.get(
    "VICON_METRIC_CSV",
    str(VALIDATION_DIR / "vicon_com_kinematics.csv"),
))
OUTPUT_CSV = Path(os.environ.get(
    "COM_CORRELATION_OUTPUT",
    str(VALIDATION_DIR / "com_correlation_smoothed.csv"),
))
OUTPUT_DIR = Path(os.environ.get(
    "COM_ZSCORE_OUTPUT_DIR",
    str(VALIDATION_DIR / "com_zscore_smoothed"),
))
VIDEO_FPS = 29.996


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    """读取 CSV；除 trial 列外，其余列统一转 float。"""
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key == "trial":
                continue
            row[key] = float(value)
    return rows


def group_by_trial(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 trial 分组，并在组内按时间排序。"""
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["trial"], []).append(row)
    # 每个试验内部按时间排序。
    for trial_rows in result.values():
        trial_rows.sort(key=lambda row: row["time_s"])
    return result


def build_axis_signals(
    video_rows: list[dict[str, Any]],
    vicon_rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """构造水平/竖直两个轴的视频与 Vicon 信号(Vicon 插值到视频时间并做方向对齐)。"""
    video_t = np.array([row["time_s"] for row in video_rows], dtype=float)
    vicon_t = np.array([row["time_s"] for row in vicon_rows], dtype=float)

    vicon_y = np.array([row["com_y_m"] for row in vicon_rows], dtype=float)
    vicon_z = np.array([row["com_z_m"] for row in vicon_rows], dtype=float)

    return {
        # 水平：video com_x_m ↔ −Vicon Y(插值到视频时间)。
        "horizontal": {
            "video": np.array([row["com_x_m"] for row in video_rows], dtype=float),
            "vicon": -np.interp(video_t, vicon_t, vicon_y),
        },
        # 竖直：video com_y_m_up(向上为正) ↔ Vicon Z(向上为正，无需翻转)。
        "vertical": {
            "video": np.array([row["com_y_m_up"] for row in video_rows], dtype=float),
            "vicon": np.interp(video_t, vicon_t, vicon_z),
        },
    }


def plot_zscore(
    time_s: np.ndarray,
    video_z: np.ndarray,
    vicon_z: np.ndarray,
    lag_frames: int,
    title: str,
    output_path: Path,
) -> None:
    """按互相关滞后对齐后，绘制视频 vs Vicon 的 z 分数叠加图。"""
    # 依据 lag 的正负裁剪两路信号以对齐。
    shift = abs(lag_frames)
    if lag_frames < 0:
        # Vicon 领先：裁视频尾、Vicon 头。
        video_plot = video_z[:-shift] if shift > 0 else video_z
        vicon_plot = vicon_z[shift:] if shift > 0 else vicon_z
        time_plot = time_s[:-shift] if shift > 0 else time_s
    elif lag_frames > 0:
        # 视频领先：裁视频头、Vicon 尾。
        video_plot = video_z[shift:]
        vicon_plot = vicon_z[:-shift]
        time_plot = time_s[shift:]
    else:
        # 无滞后。
        video_plot = video_z
        vicon_plot = vicon_z
        time_plot = time_s

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    ax.plot(time_plot, video_plot, color="#1f77b4", linewidth=1.6, label="video z-score")
    ax.plot(time_plot, vicon_plot, color="#d62728", linewidth=1.6, label="Vicon z-score")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def process_trial(
    trial: str,
    video_rows: list[dict[str, Any]],
    vicon_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """对单个试验做两轴 z 分数对比，出图并返回指标行。"""
    # 数据太少则跳过。
    if len(video_rows) < 5 or len(vicon_rows) < 5:
        return []

    # 仅保留落在 Vicon 时间覆盖范围内的视频帧。
    video_t_all = np.array([row["time_s"] for row in video_rows], dtype=float)
    vicon_t = np.array([row["time_s"] for row in vicon_rows], dtype=float)
    mask = (video_t_all >= vicon_t.min()) & (video_t_all <= vicon_t.max())
    clipped_video_rows = [row for row, keep in zip(video_rows, mask) if keep]
    if len(clipped_video_rows) < 5:
        return []

    video_t = np.array([row["time_s"] for row in clipped_video_rows], dtype=float)
    # 构造两轴信号(含插值与方向对齐)。
    axes = build_axis_signals(clipped_video_rows, vicon_rows)

    result_rows: list[dict[str, Any]] = []
    for axis_name, signals in axes.items():
        # 去线性漂移 + z-score。
        video_z = zscore_normalize(remove_linear_drift(signals["video"]))
        vicon_z = zscore_normalize(remove_linear_drift(signals["vicon"]))
        # 相关性与互相关滞后。
        r, p_value, nrmse = compute_metrics(video_z, vicon_z)
        xcorr_peak, lag_frames, lag_ms = compute_xcorr(video_z, vicon_z, VIDEO_FPS)

        # 出 z 分数对齐图。
        output_path = output_dir / f"{trial}_{axis_name}_zscore.png"
        plot_zscore(
            video_t,
            video_z,
            vicon_z,
            lag_frames,
            f"{trial} CoM {axis_name}: video vs Vicon z-score | r={r:.3f}",
            output_path,
        )

        # 记录该轴指标。
        result_rows.append({
            "trial": trial,
            "axis": axis_name,
            "pearson_r": round(r, 4),
            "p_value": f"{p_value:.4e}",
            "nrmse": round(nrmse, 4),
            "n_frames": len(video_t),
            "xcorr_peak_r": xcorr_peak,
            "xcorr_lag_frames": lag_frames,
            "xcorr_lag_ms": lag_ms,
            "source_video_metric": str(VIDEO_METRIC_CSV),
        })
    return result_rows


def write_correlation(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """把相关性结果写入 CSV。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "axis", "pearson_r", "p_value", "nrmse", "n_frames",
        "xcorr_peak_r", "xcorr_lag_frames", "xcorr_lag_ms", "source_video_metric",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """主流程：按试验分组 -> 逐试验对比 -> 汇总相关性 CSV 与 z 分数图。"""
    video_by_trial = group_by_trial(load_rows(VIDEO_METRIC_CSV))
    vicon_by_trial = group_by_trial(load_rows(VICON_METRIC_CSV))
    rows: list[dict[str, Any]] = []
    for trial in sorted(video_by_trial):
        # 只处理两边都有的试验。
        if trial not in vicon_by_trial:
            continue
        rows.extend(process_trial(trial, video_by_trial[trial], vicon_by_trial[trial], OUTPUT_DIR))
    write_correlation(rows, OUTPUT_CSV)
    print(f"Correlation CSV: {OUTPUT_CSV}")
    print(f"Z-score plots: {OUTPUT_DIR}")
    print(f"Rows: {len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
