from __future__ import annotations

r"""WT02 全部重叠帧的视频/Vicon 指标时序对比图。

输入来自 metrics_keypoints_preprocessed 下的 video_com_metric.csv 和 vicon_com_metric.csv。
时间对齐采用当前确认的方法：以 Vicon 时间轴为目标，在相邻视频帧之间线性插值 video 指标。
输出只写入 H:\COM\temp，不覆盖 validation 目录。
"""

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

# 无界面后端，便于批量出图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend
from scipy.stats import pearsonr


TRIAL = "WT02"
ROOT = Path(r"H:\COM\video-vicon")
METRIC_DIR = ROOT / "validation" / "metrics_keypoints_preprocessed"
VIDEO_METRIC_CSV = METRIC_DIR / "video_com_metric.csv"
VICON_METRIC_CSV = METRIC_DIR / "vicon_com_metric.csv"
# 输出目录与两份 CSV(时序数据、相关性汇总)，均写到 temp。
OUTPUT_DIR = Path(r"H:\COM\temp\WT02_allframes_metrics_timeseries")
TIMESERIES_CSV = OUTPUT_DIR / "WT02_allframes_metrics_timeseries.csv"
CORRELATION_CSV = OUTPUT_DIR / "WT02_allframes_metric_correlations.csv"
TITLE_SUFFIX = "WT02_AllFrames"      # 图标题/文件名统一后缀


def load_rows(csv_path: Path, trial: str) -> list[dict[str, Any]]:
    """读取一个 trial 的 metric CSV，除 trial 外尽量转成数值。"""
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只保留目标试验。
            if row.get("trial") != trial:
                continue
            converted: dict[str, Any] = {"trial": row["trial"]}
            for key, value in row.items():
                if key == "trial":
                    continue
                try:
                    # frame 转 int，其余列转 float。
                    if key == "frame":
                        converted[key] = int(float(value))
                    else:
                        converted[key] = float(value)
                except (TypeError, ValueError):
                    # 非数值列保留原字符串。
                    converted[key] = value
            rows.append(converted)
    # 按时间排序，保证后续插值/差分正确。
    rows.sort(key=lambda item: float(item["time_s"]))
    return rows


def interp_video(video_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """把视频指标插值到 Vicon 时间点。"""
    return np.interp(target_t, video_t, values)


def build_timeseries_rows(video_rows: list[dict[str, Any]], vicon_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造 Vicon 时间轴上的 video/Vicon 对比数据。"""
    # 视频时间轴与 Vicon 全部时间。
    video_t = np.array([row["time_s"] for row in video_rows], dtype=float)
    vicon_t_all = np.array([row["time_s"] for row in vicon_rows], dtype=float)
    # 仅取落在视频时间范围内的 Vicon 帧作为目标时间点。
    overlap_mask = (vicon_t_all >= video_t.min()) & (vicon_t_all <= video_t.max())
    target_t = vicon_t_all[overlap_mask]
    if len(target_t) < 5:
        raise RuntimeError(f"{TRIAL}: fewer than 5 overlapping Vicon frames")

    # 对应的 Vicon 重叠帧记录，以及插值出的视频帧号(仅供参考/对照)。
    vicon_overlap = [row for row, keep in zip(vicon_rows, overlap_mask) if keep]
    video_frame_interp = interp_video(video_t, np.array([row["frame"] for row in video_rows], dtype=float), target_t)

    # video 指标插值到 Vicon 时间点。
    video_velocity = interp_video(video_t, np.array([row["velocity_m_s"] for row in video_rows], dtype=float), target_t)
    video_l = interp_video(video_t, np.array([row["l_m"] for row in video_rows], dtype=float), target_t)
    video_com_x = interp_video(video_t, np.array([row["com_x_m"] for row in video_rows], dtype=float), target_t)
    video_com_y = interp_video(video_t, np.array([row["com_y_m_up"] for row in video_rows], dtype=float), target_t)
    video_xcom_x = interp_video(video_t, np.array([row["xcom_x_m"] for row in video_rows], dtype=float), target_t)
    video_xcom_y = interp_video(video_t, np.array([row["xcom_y_m_up"] for row in video_rows], dtype=float), target_t)

    # Vicon 前视图约定：horizontal = -Y，vertical = Z。
    vicon_com_x = -np.array([row["com_y_m"] for row in vicon_overlap], dtype=float)
    vicon_com_y = np.array([row["com_z_m"] for row in vicon_overlap], dtype=float)
    vicon_xcom_x = -np.array([row["xcom_y_m"] for row in vicon_overlap], dtype=float)
    vicon_xcom_y = np.array([row["xcom_z_m"] for row in vicon_overlap], dtype=float)
    # Vicon 速度取 Y-Z 平面合速度。
    vicon_velocity = np.sqrt(
        np.array([row["velocity_y_m_s"] for row in vicon_overlap], dtype=float) ** 2
        + np.array([row["velocity_z_m_s"] for row in vicon_overlap], dtype=float) ** 2
    )
    vicon_l = np.array([row["com_z_m"] for row in vicon_overlap], dtype=float)

    # CoM/xCoM 使用相对首个重叠时间点的 delta，避免视频/Vicon 原点不同导致不可比。
    video_com_x_delta = video_com_x - video_com_x[0]
    video_com_y_delta = video_com_y - video_com_y[0]
    vicon_com_x_delta = vicon_com_x - vicon_com_x[0]
    vicon_com_y_delta = vicon_com_y - vicon_com_y[0]
    video_xcom_x_delta = video_xcom_x - video_xcom_x[0]
    video_xcom_y_delta = video_xcom_y - video_xcom_y[0]
    vicon_xcom_x_delta = vicon_xcom_x - vicon_xcom_x[0]
    vicon_xcom_y_delta = vicon_xcom_y - vicon_xcom_y[0]

    # 逐个重叠帧组装对比行。
    rows: list[dict[str, Any]] = []
    for idx, vicon_row in enumerate(vicon_overlap):
        rows.append({
            "index": idx + 1,
            "time_s": float(target_t[idx]),
            "video_frame_interp": float(video_frame_interp[idx]),
            "vicon_frame": int(vicon_row["frame"]),
            "video_velocity_m_s": float(video_velocity[idx]),
            "vicon_velocity_yz_m_s": float(vicon_velocity[idx]),
            "video_l_m": float(video_l[idx]),
            "vicon_l_m": float(vicon_l[idx]),
            "video_com_x_delta_m": float(video_com_x_delta[idx]),
            "vicon_com_x_delta_m": float(vicon_com_x_delta[idx]),
            "video_com_y_delta_m": float(video_com_y_delta[idx]),
            "vicon_com_y_delta_m": float(vicon_com_y_delta[idx]),
            "video_xcom_x_delta_m": float(video_xcom_x_delta[idx]),
            "vicon_xcom_x_delta_m": float(vicon_xcom_x_delta[idx]),
            "video_xcom_y_delta_m": float(video_xcom_y_delta[idx]),
            "vicon_xcom_y_delta_m": float(vicon_xcom_y_delta[idx]),
        })
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """把行列表写入 CSV(列名取自首行键)。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def zscore(values: np.ndarray) -> np.ndarray:
    """z-score 归一化(零均值、单位标准差)；常数信号报错。"""
    centered = values - np.mean(values)
    std = np.std(centered)
    if std == 0.0:
        raise ValueError("cannot z-score a constant signal")
    return centered / std


def compute_xcorr(video_values: np.ndarray, vicon_values: np.ndarray, sample_rate_hz: float) -> tuple[float, int, float]:
    """对 detrend+zscore 后的信号做互相关，lag>0 表示 Vicon leads video。"""
    # 先去趋势再 z-score，使比较聚焦波形而非幅度/漂移。
    video_z = zscore(detrend(video_values.astype(float), type="linear"))
    vicon_z = zscore(detrend(vicon_values.astype(float), type="linear"))
    # 全模式互相关并按长度归一化。
    corr = np.correlate(video_z, vicon_z, mode="full") / len(video_z)
    lags = np.arange(-len(video_z) + 1, len(video_z))
    # 取相关峰值对应的滞后(帧 + 毫秒)。
    peak_idx = int(np.argmax(corr))
    lag_frames = int(lags[peak_idx])
    lag_ms = lag_frames / sample_rate_hz * 1000.0
    return float(corr[peak_idx]), lag_frames, float(lag_ms)


def lag_correct_series(video_values: np.ndarray, vicon_values: np.ndarray, lag_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 xcorr lag 裁剪两路信号，返回保留索引和对齐后的两路值。"""
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


def correlation_row(rows: list[dict[str, Any]], metric: str, video_key: str, vicon_key: str, sample_rate_hz: float) -> dict[str, Any]:
    """计算单个指标序列的 raw Pearson 以及 detrend+zscore 后的相关性/nRMSE/xcorr。"""
    video = np.array([row[video_key] for row in rows], dtype=float)
    vicon = np.array([row[vicon_key] for row in rows], dtype=float)

    # 原始(未处理)Pearson 相关与显著性。
    raw_r, raw_p = pearsonr(video, vicon)
    # 去趋势 + z-score 后的相关、nRMSE。
    video_z = zscore(detrend(video, type="linear"))
    vicon_z = zscore(detrend(vicon, type="linear"))
    dz_r, dz_p = pearsonr(video_z, vicon_z)
    nrmse = float(np.sqrt(np.mean((video_z - vicon_z) ** 2)))
    # 互相关峰值与滞后。
    xcorr_peak, lag_frames, lag_ms = compute_xcorr(video, vicon, sample_rate_hz)
    # 原始尺度下的 RMSE(带量纲)。
    rmse_raw = float(np.sqrt(np.mean((video - vicon) ** 2)))

    return {
        "metric": metric,
        "video_column": video_key,
        "vicon_column": vicon_key,
        "n_frames": len(rows),
        "pearson_r_raw": round(float(raw_r), 6),
        "p_value_raw": f"{float(raw_p):.6e}",
        "rmse_raw": rmse_raw,
        "pearson_r_detrended_zscore": round(float(dz_r), 6),
        "p_value_detrended_zscore": f"{float(dz_p):.6e}",
        "nrmse_detrended_zscore": nrmse,
        "xcorr_peak_r": round(xcorr_peak, 6),
        "xcorr_lag_frames": lag_frames,
        "xcorr_lag_ms": lag_ms,
        "time_axis": "Vicon time_s",
        "video_interpolation": "linear video metric to Vicon time",
    }


def write_correlations(rows: list[dict[str, Any]], sample_rate_hz: float) -> None:
    """对速度/l/CoM/xCoM 各指标计算相关性并写出汇总 CSV。"""
    # 每个 spec = (指标名, 视频列, Vicon列)。
    specs = [
        ("velocity", "video_velocity_m_s", "vicon_velocity_yz_m_s"),
        ("l", "video_l_m", "vicon_l_m"),
        ("com_x_delta", "video_com_x_delta_m", "vicon_com_x_delta_m"),
        ("com_y_delta", "video_com_y_delta_m", "vicon_com_y_delta_m"),
        ("xcom_x_delta", "video_xcom_x_delta_m", "vicon_xcom_x_delta_m"),
        ("xcom_y_delta", "video_xcom_y_delta_m", "vicon_xcom_y_delta_m"),
    ]
    corr_rows = [correlation_row(rows, *spec, sample_rate_hz) for spec in specs]
    write_csv(corr_rows, CORRELATION_CSV)


def plot_scalar(rows: list[dict[str, Any]], video_key: str, vicon_key: str, ylabel: str, title: str, output_path: Path) -> None:
    """绘制单一标量(如速度、l)的 video vs Vicon 时间序列对比。"""
    t = np.array([row["time_s"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    ax.plot(t, [row[video_key] for row in rows], color="#1f77b4", linewidth=1.5, label="video")
    ax.plot(t, [row[vicon_key] for row in rows], color="#d62728", linewidth=1.5, label="Vicon")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pair(rows: list[dict[str, Any]], metric: str, x_keys: tuple[str, str], y_keys: tuple[str, str], output_path: Path) -> None:
    """绘制某指标(CoM/xCoM)水平/竖直分量的 video vs Vicon 对比(上下两子图)。"""
    t = np.array([row["time_s"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)
    # 上：水平(x)分量。
    axes[0].plot(t, [row[x_keys[0]] for row in rows], color="#1f77b4", label=f"video {metric} x")
    axes[0].plot(t, [row[x_keys[1]] for row in rows], color="#d62728", label=f"Vicon {metric} x")
    axes[0].set_ylabel(f"{metric} x delta (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    # 下：竖直(y)分量。
    axes[1].plot(t, [row[y_keys[0]] for row in rows], color="#1f77b4", label=f"video {metric} y")
    axes[1].plot(t, [row[y_keys[1]] for row in rows], color="#d62728", label=f"Vicon {metric} y")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(f"{metric} y delta (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"{metric} | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pair_detrended(rows: list[dict[str, Any]], metric: str, x_keys: tuple[str, str], y_keys: tuple[str, str], output_path: Path) -> None:
    """绘制去线性趋势后的某指标水平/竖直分量对比(突出周期摆动)。"""
    t = np.array([row["time_s"] for row in rows], dtype=float)
    # 对四条曲线分别做线性去趋势。
    vx = detrend(np.array([row[x_keys[0]] for row in rows], dtype=float), type="linear")
    cx = detrend(np.array([row[x_keys[1]] for row in rows], dtype=float), type="linear")
    vy = detrend(np.array([row[y_keys[0]] for row in rows], dtype=float), type="linear")
    cy = detrend(np.array([row[y_keys[1]] for row in rows], dtype=float), type="linear")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)
    axes[0].plot(t, vx, color="#1f77b4", label=f"video {metric} x")
    axes[0].plot(t, cx, color="#d62728", label=f"Vicon {metric} x")
    axes[0].set_ylabel(f"{metric} x detrended (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, vy, color="#1f77b4", label=f"video {metric} y")
    axes[1].plot(t, cy, color="#d62728", label=f"Vicon {metric} y")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(f"{metric} y detrended (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"{metric}_detrended | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pair_lag_corrected(
    rows: list[dict[str, Any]],
    metric: str,
    x_keys: tuple[str, str],
    y_keys: tuple[str, str],
    sample_rate_hz: float,
    output_path: Path,
) -> None:
    """按互相关峰值滞后对齐后，绘制某指标(去趋势)水平/竖直分量的对比。"""
    t = np.array([row["time_s"] for row in rows], dtype=float)
    # 原始(未去趋势)序列，用于求最优滞后。
    vx_raw = np.array([row[x_keys[0]] for row in rows], dtype=float)
    cx_raw = np.array([row[x_keys[1]] for row in rows], dtype=float)
    vy_raw = np.array([row[y_keys[0]] for row in rows], dtype=float)
    cy_raw = np.array([row[y_keys[1]] for row in rows], dtype=float)

    # 分别求水平、竖直方向的最优滞后。
    peak_x, lag_x, lag_ms_x = compute_xcorr(vx_raw, cx_raw, sample_rate_hz)
    peak_y, lag_y, lag_ms_y = compute_xcorr(vy_raw, cy_raw, sample_rate_hz)

    # 去趋势后用于绘图。
    vx = detrend(vx_raw, type="linear")
    cx = detrend(cx_raw, type="linear")
    vy = detrend(vy_raw, type="linear")
    cy = detrend(cy_raw, type="linear")
    # 按各自滞后裁剪对齐。
    x_indices, vx_lag, cx_lag = lag_correct_series(vx, cx, lag_x)
    y_indices, vy_lag, cy_lag = lag_correct_series(vy, cy, lag_y)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=False)
    # 上：水平分量，对齐后叠加，标题标注滞后与峰值。
    axes[0].plot(t[x_indices], vx_lag, color="#1f77b4", label=f"video {metric} x")
    axes[0].plot(t[x_indices], cx_lag, color="#d62728", label=f"Vicon {metric} x")
    axes[0].set_title(f"x lag={lag_x} Vicon frames ({lag_ms_x:.1f} ms), peak={peak_x:.3f}")
    axes[0].set_ylabel(f"{metric} x detrended (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    # 下：竖直分量。
    axes[1].plot(t[y_indices], vy_lag, color="#1f77b4", label=f"video {metric} y")
    axes[1].plot(t[y_indices], cy_lag, color="#d62728", label=f"Vicon {metric} y")
    axes[1].set_title(f"y lag={lag_y} Vicon frames ({lag_ms_y:.1f} ms), peak={peak_y:.3f}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(f"{metric} y detrended (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"{metric}_detrended_lag | {TITLE_SUFFIX}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def clear_output_dir() -> None:
    """清空 temp 输出目录(带 workspace 边界校验，拒绝删除工作区外路径)。"""
    workspace = Path(r"H:\COM").resolve()
    target = OUTPUT_DIR.resolve()
    # 安全校验：目标必须位于工作区目录之下。
    if not str(target).startswith(str(workspace) + "\\"):
        raise RuntimeError(f"Refusing to clear outside workspace: {target}")
    # 确保目录存在后，自底向上删除文件与空目录。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(OUTPUT_DIR.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def main() -> int:
    """主流程：读取两侧指标 -> 估计 Vicon 采样率 -> 构造时序 -> 写 CSV -> 绘图。"""
    video_rows = load_rows(VIDEO_METRIC_CSV, TRIAL)
    vicon_rows = load_rows(VICON_METRIC_CSV, TRIAL)
    if not video_rows or not vicon_rows:
        raise RuntimeError(f"Missing metric rows for {TRIAL}")
    # 用 Vicon 相邻时间差的中位数估计采样率(供 xcorr 把滞后帧换算成毫秒)。
    vicon_dt = np.median(np.diff(np.array([row["time_s"] for row in vicon_rows], dtype=float)))
    sample_rate_hz = 1.0 / float(vicon_dt)

    # 构造对齐时序、清空输出目录，写出时序与相关性两份 CSV。
    rows = build_timeseries_rows(video_rows, vicon_rows)
    clear_output_dir()
    write_csv(rows, TIMESERIES_CSV)
    write_correlations(rows, sample_rate_hz)

    # 标量对比：速度、摆长 l。
    plot_scalar(rows, "video_velocity_m_s", "vicon_velocity_yz_m_s", "Velocity magnitude (m/s)", f"velocity | {TITLE_SUFFIX}", OUTPUT_DIR / "velocity_WT02_AllFrames.png")
    plot_scalar(rows, "video_l_m", "vicon_l_m", "l (m)", f"l | {TITLE_SUFFIX}", OUTPUT_DIR / "l_WT02_AllFrames.png")
    # CoM 系列：原始、去趋势、滞后对齐。
    plot_pair(rows, "com", ("video_com_x_delta_m", "vicon_com_x_delta_m"), ("video_com_y_delta_m", "vicon_com_y_delta_m"), OUTPUT_DIR / "com_WT02_AllFrames.png")
    plot_pair_detrended(rows, "com", ("video_com_x_delta_m", "vicon_com_x_delta_m"), ("video_com_y_delta_m", "vicon_com_y_delta_m"), OUTPUT_DIR / "com_detrended_WT02_AllFrames.png")
    plot_pair_lag_corrected(rows, "com", ("video_com_x_delta_m", "vicon_com_x_delta_m"), ("video_com_y_delta_m", "vicon_com_y_delta_m"), sample_rate_hz, OUTPUT_DIR / "com_lag_WT02_AllFrames.png")
    # xCoM 系列：原始、去趋势、滞后对齐。
    plot_pair(rows, "xcom", ("video_xcom_x_delta_m", "vicon_xcom_x_delta_m"), ("video_xcom_y_delta_m", "vicon_xcom_y_delta_m"), OUTPUT_DIR / "xcom_WT02_AllFrames.png")
    plot_pair_detrended(rows, "xcom", ("video_xcom_x_delta_m", "vicon_xcom_x_delta_m"), ("video_xcom_y_delta_m", "vicon_xcom_y_delta_m"), OUTPUT_DIR / "xcom_detrended_WT02_AllFrames.png")
    plot_pair_lag_corrected(rows, "xcom", ("video_xcom_x_delta_m", "vicon_xcom_x_delta_m"), ("video_xcom_y_delta_m", "vicon_xcom_y_delta_m"), sample_rate_hz, OUTPUT_DIR / "xcom_lag_WT02_AllFrames.png")

    print(f"Output: {OUTPUT_DIR}")
    print(f"Rows: {len(rows)}")
    print(f"Sample rate: {sample_rate_hz:.3f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
