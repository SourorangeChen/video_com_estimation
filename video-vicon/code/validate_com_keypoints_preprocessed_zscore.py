from __future__ import annotations

"""对 keypoints 预处理后的 CoM 执行 z-score/detrend/xcorr 验证流程。

当前正式方法：
1. 输入 keypoints 预处理且 CoM 已平滑后的 JSON；
2. 以 Vicon 时间轴为目标，在相邻两个视频帧之间线性插值 video CoM；
3. 坐标方向对齐；
4. detrend -> z-score -> Pearson/nRMSE/xcorr；
5. 按 xcorr lag 裁剪对齐后输出图。
"""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validate_com_normalized import (
    MANIFEST_CSV,
    VICON_CSV_ROOT,
    build_vicon_time_axis,
    compute_metrics,
    compute_xcorr,
    load_manifest,
    parse_vicon_model_outputs,
    remove_linear_drift,
    zscore_normalize,
)


KEYPOINTS_JSON = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_keypoints-preprocessed\results\keypoints_and_com_preprocessed.json"
)
OUTPUT_ROOT = Path(r"H:\COM\video-vicon\validation\com_keypoints_preprocessed")
PLOTS_DIR = OUTPUT_ROOT / "com_z-score_detrend_xcorr"
OUTPUT_CSV = OUTPUT_ROOT / "com_correlation.csv"
VIDEO_FPS = 29.996


def parse_video_com_by_trial(json_path: Path) -> dict[str, list[dict[str, float]]]:
    """从 JSON 中读取当前 com 字段；当前 com 已是 keypoints 预处理 + CoM 平滑后的值。"""
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list JSON records in {json_path}")

    grouped: dict[str, list[dict[str, float]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        trial = record.get("trial")
        frame = record.get("frame")
        com = record.get("com")
        if trial is None or frame is None or not isinstance(com, dict):
            continue
        grouped.setdefault(str(trial), []).append({
            "frame": float(frame),
            "time_s": (float(frame) - 1.0) / VIDEO_FPS,
            "com_x": float(com["com_x"]),
            "com_y": float(com["com_y"]),
        })

    for rows in grouped.values():
        rows.sort(key=lambda row: row["time_s"])
    return grouped


def plot_lag_aligned(
    time_s: np.ndarray,
    video_z: np.ndarray,
    vicon_z: np.ndarray,
    lag_frames: int,
    title: str,
    output_path: Path,
) -> None:
    """按 xcorr lag 裁剪两路 z-score 信号后画叠加图。"""
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

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(time_plot, video_plot, color="tab:blue", linewidth=1.2, label="video (z-score)")
    ax.plot(time_plot, vicon_plot, color="tab:red", linewidth=1.2, alpha=0.8, label="vicon (z-score)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def process_trial(
    trial: str,
    manifest_row: dict[str, Any],
    video_rows: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """处理单个 trial；核心区别是 video CoM 插值到 Vicon 时间点。"""
    vicon_csv = VICON_CSV_ROOT / f"{trial}.csv"
    if not vicon_csv.exists():
        return []

    rate_hz, vicon_frames, _vicon_x_mm, vicon_y_mm, vicon_z_mm = parse_vicon_model_outputs(vicon_csv)
    if len(video_rows) < 5 or len(vicon_frames) < 5:
        return []

    video_t = np.array([row["time_s"] for row in video_rows], dtype=float)
    video_x = np.array([row["com_x"] for row in video_rows], dtype=float)
    video_y = np.array([row["com_y"] for row in video_rows], dtype=float)
    vicon_t = build_vicon_time_axis(
        vicon_frames,
        int(manifest_row["first_frame"]),
        float(manifest_row["trajectory_rate_hz"]),
    )

    overlap_mask = (vicon_t >= video_t.min()) & (vicon_t <= video_t.max())
    target_t = vicon_t[overlap_mask]
    if len(target_t) < 5:
        return []

    # 时间对齐核心：对每个 Vicon 时刻，在相邻两个视频帧之间按时间比例估计 video CoM。
    video_x_interp = np.interp(target_t, video_t, video_x)
    video_y_interp = np.interp(target_t, video_t, video_y)

    # 保持旧 z-score 图的坐标方向约定：horizontal 对 -Vicon Y，vertical 对 -Vicon Z。
    vicon_y_aligned = -np.array(vicon_y_mm, dtype=float)[overlap_mask]
    vicon_z_aligned = -np.array(vicon_z_mm, dtype=float)[overlap_mask]
    missing_pct = (len(vicon_t) - int(overlap_mask.sum())) / len(vicon_t) * 100.0

    rows: list[dict[str, Any]] = []
    for axis_label, video_sig, vicon_sig, suffix in [
        ("horizontal (video_x vs vicon_Y)", video_x_interp, vicon_y_aligned, "x"),
        ("vertical (video_y vs vicon_Z)", video_y_interp, vicon_z_aligned, "z"),
    ]:
        # 两路分别线性去趋势，再 z-score；比较形状和相位，不比较绝对幅值。
        video_z = zscore_normalize(remove_linear_drift(video_sig))
        vicon_z = zscore_normalize(remove_linear_drift(vicon_sig))
        r, p_value, nrmse = compute_metrics(video_z, vicon_z)
        xcorr_peak, lag_frames, lag_ms = compute_xcorr(video_z, vicon_z, fps=rate_hz)

        plot_lag_aligned(
            target_t,
            video_z,
            vicon_z,
            lag_frames,
            f"{trial} - {axis_label} | xcorr_r={xcorr_peak:.3f} lag={lag_ms:.1f}ms",
            PLOTS_DIR / f"{trial}_{suffix}.png",
        )

        rows.append({
            "trial": trial,
            "axis": axis_label,
            "pearson_r": round(r, 4),
            "p_value": f"{p_value:.4e}",
            "nrmse": round(nrmse, 4),
            "n_frames": int(len(target_t)),
            "xcorr_peak_r": xcorr_peak,
            "xcorr_lag_frames": lag_frames,
            "xcorr_lag_ms": lag_ms,
            "warning": f"temporal overlap clipped {missing_pct:.0f}% of Vicon frames" if missing_pct > 20 else "",
        })
    return rows


def clear_plots_dir() -> None:
    """覆盖输出前只清空本次目标图片目录，避免残留旧文件名。"""
    workspace = Path(r"H:\COM").resolve()
    target = PLOTS_DIR.resolve()
    if not str(target).startswith(str(workspace) + "\\"):
        raise RuntimeError(f"Refusing to clear outside workspace: {target}")
    if not target.exists():
        return
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "axis", "pearson_r", "p_value", "nrmse", "n_frames",
        "xcorr_peak_r", "xcorr_lag_frames", "xcorr_lag_ms", "warning",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    manifest = load_manifest(MANIFEST_CSV)
    video_by_trial = parse_video_com_by_trial(KEYPOINTS_JSON)
    clear_plots_dir()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for trial, manifest_row in manifest.items():
        if trial not in video_by_trial:
            continue
        rows = process_trial(trial, manifest_row, video_by_trial[trial])
        all_rows.extend(rows)

    write_summary_csv(all_rows)
    print(f"Input: {KEYPOINTS_JSON}")
    print(f"Correlation CSV: {OUTPUT_CSV}")
    print(f"Z-score plots: {PLOTS_DIR}")
    print(f"Rows: {len(all_rows)}")
    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
