"""基于 metric CSV 做 CoM 相关性分析的实验脚本。

这个脚本采用“视频 CoM 插值到 Vicon 时间轴”的方案，主要用于和旧的
Vicon->video_t 方法做对照。当前正式复刻旧 z-score 图的方法见
validate_com_keypoints_preprocessed_zscore.py。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

# 无界面后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

# 复用验证脚本中的信号处理与指标函数，保证算法一致。
from validate_com_normalized import (
    compute_metrics,
    compute_xcorr,
    remove_linear_drift,
    zscore_normalize,
)


ROOT = Path(r"H:\COM\video-vicon")
METRIC_DIR = ROOT / "validation" / "metrics_keypoints_preprocessed"
VIDEO_METRIC_CSV = METRIC_DIR / "video_com_metric.csv"
VICON_METRIC_CSV = METRIC_DIR / "vicon_com_metric.csv"
OUTPUT_DIR = ROOT / "validation" / "com_keypoints_preprocessed"
OUTPUT_CSV = OUTPUT_DIR / "com_correlation.csv"
PLOT_DIR = OUTPUT_DIR / "com_z-score_detrend_xcorr"
VICON_RATE_HZ = 250.0          # Vicon 采样率(本方案以 Vicon 时间轴为目标)
# 相关性前对米制 CoM 再平滑用的滤波参数。
COM_MEDIAN_WINDOW = 3
COM_SAVGOL_WINDOW = 15
COM_SAVGOL_POLYORDER = 2


def load_rows(csv_path: Path, required_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """读取 CSV 中所需字段(trial 保留字符串，其余转 float)；缺字段则报错。"""
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 先校验所需列是否齐全。
        missing = [field for field in required_fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} missing fields: {missing}")
        for row in reader:
            parsed: dict[str, Any] = {"trial": row["trial"]}
            for field in required_fields:
                if field == "trial":
                    continue
                parsed[field] = float(row[field])
            rows.append(parsed)
    return rows


def group_by_trial(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 trial 分组，组内按时间排序。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["trial"], []).append(row)
    for trial_rows in grouped.values():
        trial_rows.sort(key=lambda row: row["time_s"])
    return grouped


def interpolate_video_to_vicon_time(
    vicon_t: np.ndarray,
    video_t: np.ndarray,
    video_values: np.ndarray,
) -> np.ndarray:
    """把视频数值插值到 Vicon 时间点上(本方案目标时间轴为 Vicon)。"""
    return np.interp(vicon_t, video_t, video_values)


def odd_window_for_length(preferred_window: int, signal_length: int) -> int | None:
    """返回不超过序列长度的奇数窗口；序列过短返回 None。"""
    if signal_length < 3:
        return None
    window = min(preferred_window, signal_length)
    if window % 2 == 0:
        window -= 1
    return window if window >= 3 else None


def filter_com_signal(values: np.ndarray) -> np.ndarray:
    """相关性前对米制 CoM 序列再做 median + S-G 平滑，减少尺度抖动带来的噪声。"""
    filtered = values.astype(float).copy()
    # 中值滤波。
    median_window = odd_window_for_length(COM_MEDIAN_WINDOW, len(filtered))
    if median_window is not None:
        filtered = median_filter(filtered, size=median_window, mode="nearest")

    # Savitzky-Golay 平滑(窗口需大于多项式阶数)。
    sg_window = odd_window_for_length(COM_SAVGOL_WINDOW, len(filtered))
    if sg_window is not None and sg_window > COM_SAVGOL_POLYORDER:
        filtered = savgol_filter(
            filtered,
            window_length=sg_window,
            polyorder=COM_SAVGOL_POLYORDER,
            mode="interp",
        )
    return filtered


def build_warning(
    total_video_rows: int,
    total_vicon_rows: int,
    retained_video_rows: int,
    retained_vicon_rows: int,
) -> str:
    """若时间重叠后保留比例不足 80%，生成相应告警字符串。"""
    warnings: list[str] = []
    if total_video_rows and retained_video_rows / total_video_rows < 0.8:
        warnings.append("video overlap retained <80%")
    if total_vicon_rows and retained_vicon_rows / total_vicon_rows < 0.8:
        warnings.append("vicon overlap retained <80%")
    return "; ".join(warnings)


def build_axis_signals(
    video_rows: list[dict[str, Any]],
    vicon_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], str]:
    """构造同一时间轴上的 video/Vicon 信号；这里的目标时间轴是 Vicon 时间轴。"""
    video_t_all = np.array([row["time_s"] for row in video_rows], dtype=float)
    vicon_t_all = np.array([row["time_s"] for row in vicon_rows], dtype=float)

    # 取两者时间重叠区间。
    t_min = max(float(video_t_all.min()), float(vicon_t_all.min()))
    t_max = min(float(video_t_all.max()), float(vicon_t_all.max()))

    video_mask = (video_t_all >= t_min) & (video_t_all <= t_max)
    vicon_mask = (vicon_t_all >= t_min) & (vicon_t_all <= t_max)
    # 目标时间轴 = 重叠区间内的 Vicon 时间点。
    target_t = vicon_t_all[vicon_mask]
    if len(target_t) < 5:
        raise ValueError("fewer than 5 overlapping Vicon time points")

    # 视频 CoM(米)先平滑，再插值到 Vicon 时间。
    video_x = filter_com_signal(np.array([row["com_x_m"] for row in video_rows], dtype=float))
    video_y = filter_com_signal(np.array([row["com_y_m_up"] for row in video_rows], dtype=float))

    # Vicon CoM 取重叠区间。
    vicon_y = np.array([row["com_y_m"] for row in vicon_rows], dtype=float)[vicon_mask]
    vicon_z = np.array([row["com_z_m"] for row in vicon_rows], dtype=float)[vicon_mask]

    # 重叠保留比例告警。
    warning = build_warning(
        len(video_rows),
        len(vicon_rows),
        int(video_mask.sum()),
        int(vicon_mask.sum()),
    )

    # 方向对齐：水平 video_x ↔ −Vicon Y；竖直 video_y_up ↔ Vicon Z。
    return target_t, {
        "Horizontal": {
            "video": interpolate_video_to_vicon_time(target_t, video_t_all, video_x),
            "vicon": -vicon_y,
        },
        "Vertical": {
            "video": interpolate_video_to_vicon_time(target_t, video_t_all, video_y),
            "vicon": vicon_z,
        },
    }, warning


def plot_zscore(
    time_s: np.ndarray,
    video_z: np.ndarray,
    vicon_z: np.ndarray,
    lag_frames: int,
    title: str,
    output_path: Path,
) -> None:
    """按互相关滞后对齐后绘制 z 分数叠加图。"""
    # 依据 lag 正负裁剪信号以对齐。
    shift = abs(lag_frames)
    if lag_frames < 0:
        video_plot = video_z[:-shift] if shift else video_z
        vicon_plot = vicon_z[shift:] if shift else vicon_z
        time_plot = time_s[:-shift] if shift else time_s
    elif lag_frames > 0:
        video_plot = video_z[shift:]
        vicon_plot = vicon_z[:-shift]
        time_plot = time_s[shift:]
    else:
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
) -> list[dict[str, Any]]:
    """对单个试验做两轴相关性分析并出图，返回结果行。"""
    # 数据太少则跳过。
    if len(video_rows) < 5 or len(vicon_rows) < 5:
        return []

    try:
        target_t, axes, warning = build_axis_signals(video_rows, vicon_rows)
    except ValueError as exc:
        # 重叠不足等情况：返回一行带错误说明的占位结果。
        return [{
            "trial": trial,
            "axis": "x",
            "pearson_r": "",
            "p_value": "",
            "nrmse": "",
            "n_frames": 0,
            "xcorr_peak_r": "",
            "xcorr_lag_frames": "",
            "xcorr_lag_ms": "",
            "warning": str(exc),
        }]

    rows: list[dict[str, Any]] = []
    for axis_name, signals in axes.items():
        # 去线性漂移 + z-score。
        video_z = zscore_normalize(remove_linear_drift(signals["video"]))
        vicon_z = zscore_normalize(remove_linear_drift(signals["vicon"]))
        # 相关性与互相关(注意 fps 用 Vicon 采样率)。
        r, p_value, nrmse = compute_metrics(video_z, vicon_z)
        xcorr_peak, lag_frames, lag_ms = compute_xcorr(video_z, vicon_z, VICON_RATE_HZ)

        # 出 z 分数对齐图。
        plot_zscore(
            target_t,
            video_z,
            vicon_z,
            lag_frames,
            f"com_{axis_name} | {trial}",
            PLOT_DIR / f"{trial}_{axis_name}.png",
        )
        rows.append({
            "trial": trial,
            "axis": axis_name,
            "pearson_r": round(r, 4),
            "p_value": f"{p_value:.4e}",
            "nrmse": round(nrmse, 4),
            "n_frames": len(target_t),
            "xcorr_peak_r": xcorr_peak,
            "xcorr_lag_frames": lag_frames,
            "xcorr_lag_ms": lag_ms,
            "warning": warning,
        })
    return rows


def clear_output_dir() -> None:
    """清空本脚本的输出目录(带 workspace 边界校验，拒绝删除工作区外路径)。"""
    workspace = ROOT.resolve()
    target = OUTPUT_DIR.resolve()
    # 安全校验：目标必须位于工作区目录之下。
    if not str(target).startswith(str(workspace) + "\\"):
        raise RuntimeError(f"Refusing to clear outside workspace: {target}")
    if not target.exists():
        return
    # 自底向上删除文件与空目录。
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def write_correlation(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """把相关性结果写入 CSV。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "axis", "pearson_r", "p_value", "nrmse", "n_frames",
        "xcorr_peak_r", "xcorr_lag_frames", "xcorr_lag_ms", "warning",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """主流程：清空输出 -> 读取两侧 CSV -> 逐试验相关性分析 -> 汇总 CSV。"""
    clear_output_dir()
    # 各自只读取所需字段。
    video_rows = load_rows(
        VIDEO_METRIC_CSV,
        ("trial", "time_s", "com_x_m", "com_y_m_up"),
    )
    vicon_rows = load_rows(
        VICON_METRIC_CSV,
        ("trial", "time_s", "com_y_m", "com_z_m"),
    )

    video_by_trial = group_by_trial(video_rows)
    vicon_by_trial = group_by_trial(vicon_rows)

    all_rows: list[dict[str, Any]] = []
    for trial in sorted(video_by_trial):
        # 只处理两侧都有的试验。
        if trial not in vicon_by_trial:
            continue
        all_rows.extend(process_trial(trial, video_by_trial[trial], vicon_by_trial[trial]))

    write_correlation(all_rows, OUTPUT_CSV)
    print(f"Correlation CSV: {OUTPUT_CSV}")
    print(f"Z-score plots: {PLOT_DIR}")
    print(f"Rows: {len(all_rows)}")
    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
