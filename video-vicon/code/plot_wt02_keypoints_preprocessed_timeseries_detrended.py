"""WT02(关键点预处理版)视频 vs Vicon 的运动学时间序列对比，含去趋势/互相关/滞后校正。

基于 metrics_keypoints_preprocessed/ 下的视频与 Vicon 指标 CSV：
    1. 按时间最近邻匹配视频帧与 Vicon 帧(前 100 对)；
    2. 计算各量相对首帧的偏移(CoM/xCoM 的水平/竖直分量)与逐区间位移/速度；
    3. 导出对比 CSV，并绘制多组图：原始对比、去趋势对比、互相关、按互相关滞后对齐后的对比。

轴对齐：video x ↔ −Vicon Y；video y_up ↔ Vicon Z。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

# 无界面后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

# 复用时间最近邻匹配与 Vicon 运动学加载。
from plot_wt02_aligned_video_vicon_2d import match_video_rows_to_vicon_rows
from plot_wt02_kinematics_3d import load_trial_kinematics


TRIAL = "WT02"
FRAME_COUNT = 100
VIDEO_FPS = 29.996

ROOT = Path(r"H:\COM\video-vicon")
METRIC_DIR = ROOT / "validation" / "metrics_keypoints_preprocessed"
VIDEO_METRIC_CSV = METRIC_DIR / "video_com_metric.csv"
VICON_METRIC_CSV = METRIC_DIR / "vicon_com_metric.csv"
OUTPUT_DIR = METRIC_DIR / "metrics_timeseries_validation_WT02_first100"
OUTPUT_CSV = OUTPUT_DIR / "metrics_WT02_First100.csv"
TITLE_SUFFIX = "WT02_First100"    # 图标题/CSV 列名的统一后缀


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
                    converted[key] = value
            rows.append(converted)
    return rows


def build_matched_rows() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """读取视频/ Vicon 指标并按时间最近邻匹配出前 FRAME_COUNT 对。"""
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
    matched = match_video_rows_to_vicon_rows(video_rows, vicon_rows, FRAME_COUNT)
    if len(matched) < FRAME_COUNT:
        raise RuntimeError(f"Only found {len(matched)} aligned frames for {TRIAL}")
    return matched[:FRAME_COUNT]


def build_detrended_rows(matched: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    """把每对帧整理为对比行：各量相对首帧的偏移 + 逐区间位移/速度。"""
    video_times = np.array([video["time_s"] for video, _ in matched], dtype=float)
    video_com_x_px = np.array([video["com_x_px"] for video, _ in matched], dtype=float)

    rows: list[dict[str, Any]] = []
    # 首帧各量作为偏移基准。
    first_video, first_vicon = matched[0]
    first_video_com_horizontal = float(first_video["com_x_m"])
    first_video_com_vertical = float(first_video["com_y_m_up"])
    first_vicon_com_y = float(first_vicon["com_y_m"])
    first_vicon_com_z = float(first_vicon["com_z_m"])
    first_video_xcom_horizontal = float(first_video["xcom_x_m"])
    first_video_xcom_vertical = float(first_video["xcom_y_m_up"])
    first_vicon_xcom_y = float(first_vicon["xcom_y_m"])
    first_vicon_xcom_z = float(first_vicon["xcom_z_m"])

    for idx, (video_row, vicon_row) in enumerate(matched):
        if idx == 0:
            # 首帧无上一帧，逐区间位移/速度记 0。
            vicon_displacement_yz_m = 0.0
            vicon_vcom_yz_m_s = 0.0
        else:
            # Vicon 在 Y-Z 平面相对上一帧的位移与速度。
            _, prev_vicon = matched[idx - 1]
            dy = float(vicon_row["com_y_m"]) - float(prev_vicon["com_y_m"])
            dz = float(vicon_row["com_z_m"]) - float(prev_vicon["com_z_m"])
            dt = float(vicon_row["time_s"]) - float(prev_vicon["time_s"])
            vicon_displacement_yz_m = math.hypot(dy, dz)
            vicon_vcom_yz_m_s = vicon_displacement_yz_m / dt if dt > 0.0 else 0.0

        # 各量相对首帧的偏移(Vicon 水平方向按轴对齐取负)。
        video_com_horizontal_delta_m = float(video_row["com_x_m"]) - first_video_com_horizontal
        video_com_vertical_delta_m = float(video_row["com_y_m_up"]) - first_video_com_vertical
        vicon_com_horizontal_delta_m = -(float(vicon_row["com_y_m"]) - first_vicon_com_y)
        vicon_com_vertical_delta_m = float(vicon_row["com_z_m"]) - first_vicon_com_z
        video_xcom_horizontal_delta_m = float(video_row["xcom_x_m"]) - first_video_xcom_horizontal
        video_xcom_vertical_delta_m = float(video_row["xcom_y_m_up"]) - first_video_xcom_vertical
        vicon_xcom_horizontal_delta_m = -(float(vicon_row["xcom_y_m"]) - first_vicon_xcom_y)
        vicon_xcom_vertical_delta_m = float(vicon_row["xcom_z_m"]) - first_vicon_xcom_z

        rows.append({
            "index": idx + 1,
            "video_frame": int(video_row["frame"]),
            "vicon_frame": int(vicon_row["frame"]),
            "video_time_s": float(video_row["time_s"]),
            "vicon_time_s": float(vicon_row["time_s"]),
            "time_delta_ms": abs(float(video_row["time_s"]) - float(vicon_row["time_s"])) * 1000.0,
            "video_com_x_px_raw_preprocessed": float(video_com_x_px[idx]),
            "video_displacement_m": float(video_row["displacement_m"]),
            "vicon_displacement_yz_m": float(vicon_displacement_yz_m),
            "video_vcom_m_s": float(video_row["velocity_m_s"]),
            "vicon_vcom_yz_m_s": float(vicon_vcom_yz_m_s),
            "video_com_horizontal_delta_m": float(video_com_horizontal_delta_m),
            "vicon_com_horizontal_delta_m": float(vicon_com_horizontal_delta_m),
            "video_com_vertical_delta_m": float(video_com_vertical_delta_m),
            "vicon_com_vertical_delta_m": float(vicon_com_vertical_delta_m),
            "video_l_m": float(video_row["l_m"]),
            "vicon_l_m": float(vicon_row["com_z_m"]),
            "video_xcom_horizontal_delta_m": float(video_xcom_horizontal_delta_m),
            "vicon_xcom_horizontal_delta_m": float(vicon_xcom_horizontal_delta_m),
            "video_xcom_vertical_delta_m": float(video_xcom_vertical_delta_m),
            "vicon_xcom_vertical_delta_m": float(vicon_xcom_vertical_delta_m),
        })
    return rows


def write_rows(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """把行列表写入 CSV(列名取自首行键)。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把内部对比行重命名为带 TITLE_SUFFIX 后缀的对外列名(用于导出 CSV)。"""
    return [
        {
            f"index | {TITLE_SUFFIX}": row["index"],
            f"video_frame | {TITLE_SUFFIX}": row["video_frame"],
            f"vicon_frame | {TITLE_SUFFIX}": row["vicon_frame"],
            f"video_time_s | {TITLE_SUFFIX}": row["video_time_s"],
            f"vicon_time_s | {TITLE_SUFFIX}": row["vicon_time_s"],
            f"dt_ms | {TITLE_SUFFIX}": row["time_delta_ms"],
            f"video_com_x_delta_m | {TITLE_SUFFIX}": row["video_com_horizontal_delta_m"],
            f"vicon_com_x_delta_m | {TITLE_SUFFIX}": row["vicon_com_horizontal_delta_m"],
            f"video_com_y_delta_m | {TITLE_SUFFIX}": row["video_com_vertical_delta_m"],
            f"vicon_com_y_delta_m | {TITLE_SUFFIX}": row["vicon_com_vertical_delta_m"],
            f"video_displacement_m | {TITLE_SUFFIX}": row["video_displacement_m"],
            f"vicon_displacement_m | {TITLE_SUFFIX}": row["vicon_displacement_yz_m"],
            f"video_velocity_m_s | {TITLE_SUFFIX}": row["video_vcom_m_s"],
            f"vicon_velocity_m_s | {TITLE_SUFFIX}": row["vicon_vcom_yz_m_s"],
            f"video_l_m | {TITLE_SUFFIX}": row["video_l_m"],
            f"vicon_l_m | {TITLE_SUFFIX}": row["vicon_l_m"],
            f"video_xcom_x_delta_m | {TITLE_SUFFIX}": row["video_xcom_horizontal_delta_m"],
            f"vicon_xcom_x_delta_m | {TITLE_SUFFIX}": row["vicon_xcom_horizontal_delta_m"],
            f"video_xcom_y_delta_m | {TITLE_SUFFIX}": row["video_xcom_vertical_delta_m"],
            f"vicon_xcom_y_delta_m | {TITLE_SUFFIX}": row["vicon_xcom_vertical_delta_m"],
        }
        for row in rows
    ]


def plot_two_series(rows: list[dict[str, Any]], y_video: str, y_vicon: str, ylabel: str, title: str, output_path: Path) -> None:
    """绘制视频 vs Vicon 单一标量的时间序列对比(单图)。"""
    t = [row["video_time_s"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.plot(t, [row[y_video] for row in rows], color="#1f77b4", linewidth=1.8, label="video")
    ax.plot(t, [row[y_vicon] for row in rows], color="#d62728", linewidth=1.8, label="Vicon")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_xcom(rows: list[dict[str, Any]], output_path: Path) -> None:
    """绘制 xCoM 偏移的水平/竖直分量对比(上下两子图)。"""
    t = [row["video_time_s"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    # 上：水平 xCoM。
    axes[0].plot(t, [row["video_xcom_horizontal_delta_m"] for row in rows], color="#1f77b4", label="video xCoM x")
    axes[0].plot(t, [row["vicon_xcom_horizontal_delta_m"] for row in rows], color="#d62728", label="Vicon xCoM x")
    axes[0].set_ylabel("xCoM x delta (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    # 下：竖直 xCoM。
    axes[1].plot(t, [row["video_xcom_vertical_delta_m"] for row in rows], color="#1f77b4", label="video xCoM y")
    axes[1].plot(t, [row["vicon_xcom_vertical_delta_m"] for row in rows], color="#d62728", label="Vicon xCoM y")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("xCoM y delta (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"xcom | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_com(rows: list[dict[str, Any]], output_path: Path) -> None:
    """绘制 CoM 偏移的水平/竖直分量对比(上下两子图)。"""
    t = [row["video_time_s"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    # 上：水平 CoM。
    axes[0].plot(t, [row["video_com_horizontal_delta_m"] for row in rows], color="#1f77b4", label="video CoM x")
    axes[0].plot(t, [row["vicon_com_horizontal_delta_m"] for row in rows], color="#d62728", label="Vicon CoM x")
    axes[0].set_ylabel("CoM x delta (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    # 下：竖直 CoM。
    axes[1].plot(t, [row["video_com_vertical_delta_m"] for row in rows], color="#1f77b4", label="video CoM y")
    axes[1].plot(t, [row["vicon_com_vertical_delta_m"] for row in rows], color="#d62728", label="Vicon CoM y")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("CoM y delta (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"com | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_com_detrended(rows: list[dict[str, Any]], output_path: Path) -> None:
    """绘制去线性趋势后的 CoM 偏移对比(突出周期性摆动，去掉整体漂移)。"""
    t = [row["video_time_s"] for row in rows]
    # 对四条曲线分别做线性去趋势。
    video_x = detrend(np.array([row["video_com_horizontal_delta_m"] for row in rows], dtype=float), type="linear")
    vicon_x = detrend(np.array([row["vicon_com_horizontal_delta_m"] for row in rows], dtype=float), type="linear")
    video_y = detrend(np.array([row["video_com_vertical_delta_m"] for row in rows], dtype=float), type="linear")
    vicon_y = detrend(np.array([row["vicon_com_vertical_delta_m"] for row in rows], dtype=float), type="linear")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    axes[0].plot(t, video_x, color="#1f77b4", label="video CoM x")
    axes[0].plot(t, vicon_x, color="#d62728", label="Vicon CoM x")
    axes[0].set_ylabel("CoM x detrended (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, video_y, color="#1f77b4", label="video CoM y")
    axes[1].plot(t, vicon_y, color="#d62728", label="Vicon CoM y")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("CoM y detrended (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"com_detrended | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_xcom_detrended(rows: list[dict[str, Any]], output_path: Path) -> None:
    """绘制去线性趋势后的 xCoM 偏移对比。"""
    t = [row["video_time_s"] for row in rows]
    # 对四条曲线分别去趋势。
    video_x = detrend(np.array([row["video_xcom_horizontal_delta_m"] for row in rows], dtype=float), type="linear")
    vicon_x = detrend(np.array([row["vicon_xcom_horizontal_delta_m"] for row in rows], dtype=float), type="linear")
    video_y = detrend(np.array([row["video_xcom_vertical_delta_m"] for row in rows], dtype=float), type="linear")
    vicon_y = detrend(np.array([row["vicon_xcom_vertical_delta_m"] for row in rows], dtype=float), type="linear")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    axes[0].plot(t, video_x, color="#1f77b4", label="video xCoM x")
    axes[0].plot(t, vicon_x, color="#d62728", label="Vicon xCoM x")
    axes[0].set_ylabel("xCoM x detrended (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, video_y, color="#1f77b4", label="video xCoM y")
    axes[1].plot(t, vicon_y, color="#d62728", label="Vicon xCoM y")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("xCoM y detrended (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"xcom_detrended | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def zscore(values: np.ndarray) -> np.ndarray:
    """z-score 归一化(零均值、单位标准差)；常数信号报错。"""
    centered = values - np.mean(values)
    std = np.std(centered)
    if std == 0.0:
        raise ValueError("Cannot z-score a constant signal")
    return centered / std


def compute_xcorr(video_values: np.ndarray, vicon_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """对两路信号去趋势+z-score 后做互相关，返回 (滞后数组, 相关数组, 峰值滞后, 峰值相关)。"""
    # 先去趋势再 z-score，使比较聚焦于波形而非幅度/漂移。
    video_z = zscore(detrend(video_values.astype(float), type="linear"))
    vicon_z = zscore(detrend(vicon_values.astype(float), type="linear"))
    # 全模式互相关并按长度归一化。
    corr = np.correlate(video_z, vicon_z, mode="full") / len(video_z)
    lags = np.arange(-len(video_z) + 1, len(video_z))
    peak_idx = int(np.argmax(corr))
    return lags, corr, int(lags[peak_idx]), float(corr[peak_idx])


def lag_correct_series(
    video_values: np.ndarray,
    vicon_values: np.ndarray,
    lag_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按给定滞后裁剪对齐两路信号，返回 (保留索引, 对齐后视频, 对齐后Vicon)。"""
    if lag_frames > 0:
        # 视频领先：裁视频头、Vicon 尾。
        indices = np.arange(lag_frames, len(video_values))
        return indices, video_values[lag_frames:], vicon_values[:-lag_frames]
    if lag_frames < 0:
        # Vicon 领先：裁视频尾、Vicon 头。
        shift = abs(lag_frames)
        indices = np.arange(0, len(video_values) - shift)
        return indices, video_values[:-shift], vicon_values[shift:]
    # 无滞后。
    indices = np.arange(len(video_values))
    return indices, video_values, vicon_values


def plot_metric_xcorr(
    rows: list[dict[str, Any]],
    metric_name: str,
    video_x_key: str,
    vicon_x_key: str,
    video_y_key: str,
    vicon_y_key: str,
    output_path: Path,
) -> None:
    """绘制某指标(水平/竖直)的互相关函数曲线，并标注峰值滞后。"""
    video_x = np.array([row[video_x_key] for row in rows], dtype=float)
    vicon_x = np.array([row[vicon_x_key] for row in rows], dtype=float)
    video_y = np.array([row[video_y_key] for row in rows], dtype=float)
    vicon_y = np.array([row[vicon_y_key] for row in rows], dtype=float)

    # 分别求水平、竖直方向的互相关。
    lags_x, corr_x, peak_lag_x, peak_corr_x = compute_xcorr(video_x, vicon_x)
    lags_y, corr_y, peak_lag_y, peak_corr_y = compute_xcorr(video_y, vicon_y)
    # 峰值滞后换算为毫秒。
    lag_ms_x = peak_lag_x / VIDEO_FPS * 1000.0
    lag_ms_y = peak_lag_y / VIDEO_FPS * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    # 上：水平方向互相关曲线，竖虚线标峰值滞后。
    axes[0].plot(lags_x, corr_x, color="#1f77b4", linewidth=1.8)
    axes[0].axvline(peak_lag_x, color="#d62728", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("xcorr")
    axes[0].set_title(f"{metric_name} x: peak={peak_corr_x:.3f}, lag={peak_lag_x} frames ({lag_ms_x:.1f} ms)")
    axes[0].grid(True, alpha=0.3)

    # 下：竖直方向互相关曲线。
    axes[1].plot(lags_y, corr_y, color="#1f77b4", linewidth=1.8)
    axes[1].axvline(peak_lag_y, color="#d62728", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Lag (video frames)")
    axes[1].set_ylabel("xcorr")
    axes[1].set_title(f"{metric_name} y: peak={peak_corr_y:.3f}, lag={peak_lag_y} frames ({lag_ms_y:.1f} ms)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{metric_name}_xcorr | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_lag_corrected(
    rows: list[dict[str, Any]],
    metric_name: str,
    video_x_key: str,
    vicon_x_key: str,
    video_y_key: str,
    vicon_y_key: str,
    output_path: Path,
) -> None:
    """按互相关峰值滞后对齐后，绘制某指标(去趋势)的视频 vs Vicon 对比。"""
    time_s = np.array([row["video_time_s"] for row in rows], dtype=float)
    video_x_raw = np.array([row[video_x_key] for row in rows], dtype=float)
    vicon_x_raw = np.array([row[vicon_x_key] for row in rows], dtype=float)
    video_y_raw = np.array([row[video_y_key] for row in rows], dtype=float)
    vicon_y_raw = np.array([row[vicon_y_key] for row in rows], dtype=float)

    # 先求两方向各自的最优滞后。
    _, _, peak_lag_x, peak_corr_x = compute_xcorr(video_x_raw, vicon_x_raw)
    _, _, peak_lag_y, peak_corr_y = compute_xcorr(video_y_raw, vicon_y_raw)

    # 再对原始偏移去趋势(绘图用)。
    video_x = detrend(video_x_raw, type="linear")
    vicon_x = detrend(vicon_x_raw, type="linear")
    video_y = detrend(video_y_raw, type="linear")
    vicon_y = detrend(vicon_y_raw, type="linear")

    # 按各自滞后裁剪对齐。
    x_indices, video_x_lagged, vicon_x_lagged = lag_correct_series(video_x, vicon_x, peak_lag_x)
    y_indices, video_y_lagged, vicon_y_lagged = lag_correct_series(video_y, vicon_y, peak_lag_y)
    lag_ms_x = peak_lag_x / VIDEO_FPS * 1000.0
    lag_ms_y = peak_lag_y / VIDEO_FPS * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=False)
    # 上：水平方向，对齐后叠加。
    axes[0].plot(time_s[x_indices], video_x_lagged, color="#1f77b4", linewidth=1.8, label=f"video {metric_name} x")
    axes[0].plot(time_s[x_indices], vicon_x_lagged, color="#d62728", linewidth=1.8, label=f"Vicon {metric_name} x")
    axes[0].set_ylabel(f"{metric_name} x detrended (m)")
    axes[0].set_title(f"x lag={peak_lag_x} frames ({lag_ms_x:.1f} ms), peak={peak_corr_x:.3f}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 下：竖直方向，对齐后叠加。
    axes[1].plot(time_s[y_indices], video_y_lagged, color="#1f77b4", linewidth=1.8, label=f"video {metric_name} y")
    axes[1].plot(time_s[y_indices], vicon_y_lagged, color="#d62728", linewidth=1.8, label=f"Vicon {metric_name} y")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(f"{metric_name} y detrended (m)")
    axes[1].set_title(f"y lag={peak_lag_y} frames ({lag_ms_y:.1f} ms), peak={peak_corr_y:.3f}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{metric_name}_detrended_lag | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    """主流程：匹配帧 -> 计算对比行 -> 清空并写出 CSV -> 绘制全部对比图。"""
    matched = build_matched_rows()
    rows = build_detrended_rows(matched)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 清空输出目录内的旧文件，避免残留。
    for existing_file in OUTPUT_DIR.iterdir():
        if existing_file.is_file():
            existing_file.unlink()
    # 重命名列名后导出 CSV。
    output_rows = build_output_rows(rows)
    write_rows(output_rows, OUTPUT_CSV)

    # ── CoM 系列图：原始、去趋势、互相关、滞后对齐 ──
    plot_com(rows, OUTPUT_DIR / "com_WT02_First100.png")
    plot_com_detrended(rows, OUTPUT_DIR / "com_detrended_WT02_First100.png")
    plot_metric_xcorr(
        rows,
        "com",
        "video_com_horizontal_delta_m",
        "vicon_com_horizontal_delta_m",
        "video_com_vertical_delta_m",
        "vicon_com_vertical_delta_m",
        OUTPUT_DIR / "com_xcorr_WT02_First100.png",
    )
    plot_metric_lag_corrected(
        rows,
        "com",
        "video_com_horizontal_delta_m",
        "vicon_com_horizontal_delta_m",
        "video_com_vertical_delta_m",
        "vicon_com_vertical_delta_m",
        OUTPUT_DIR / "com_lag_WT02_First100.png",
    )
    # ── 标量对比图：位移、速度、摆长 l ──
    plot_two_series(
        rows,
        "video_displacement_m",
        "vicon_displacement_yz_m",
        "Displacement over video-frame interval (m)",
        f"displacement | {TITLE_SUFFIX}",
        OUTPUT_DIR / "displacement_WT02_First100.png",
    )
    plot_two_series(
        rows,
        "video_vcom_m_s",
        "vicon_vcom_yz_m_s",
        "CoM velocity magnitude (m/s)",
        f"velocity | {TITLE_SUFFIX}",
        OUTPUT_DIR / "velocity_WT02_First100.png",
    )
    plot_two_series(
        rows,
        "video_l_m",
        "vicon_l_m",
        "l (m)",
        f"l | {TITLE_SUFFIX}",
        OUTPUT_DIR / "l_WT02_First100.png",
    )
    # ── xCoM 系列图：原始、去趋势、互相关、滞后对齐 ──
    plot_xcom(rows, OUTPUT_DIR / "xcom_WT02_First100.png")
    plot_xcom_detrended(rows, OUTPUT_DIR / "xcom_detrended_WT02_First100.png")
    plot_metric_xcorr(
        rows,
        "xcom",
        "video_xcom_horizontal_delta_m",
        "vicon_xcom_horizontal_delta_m",
        "video_xcom_vertical_delta_m",
        "vicon_xcom_vertical_delta_m",
        OUTPUT_DIR / "xcom_xcorr_WT02_First100.png",
    )
    plot_metric_lag_corrected(
        rows,
        "xcom",
        "video_xcom_horizontal_delta_m",
        "vicon_xcom_horizontal_delta_m",
        "video_xcom_vertical_delta_m",
        "vicon_xcom_vertical_delta_m",
        OUTPUT_DIR / "xcom_lag_WT02_First100.png",
    )

    print(f"Output: {OUTPUT_DIR}")
    print(f"Rows: {len(rows)}")
    print(
        f"First pair: video frame {rows[0]['video_frame']} t={rows[0]['video_time_s']:.3f}s, "
        f"vicon frame {rows[0]['vicon_frame']} t={rows[0]['vicon_time_s']:.3f}s"
    )
    print(
        f"Last pair: video frame {rows[-1]['video_frame']} t={rows[-1]['video_time_s']:.3f}s, "
        f"vicon frame {rows[-1]['vicon_frame']} t={rows[-1]['vicon_time_s']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
