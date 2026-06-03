from __future__ import annotations

"""临时重画 CoM z-score 对比图：video CoM 插值到 Vicon 时间轴。

输入使用 keypoints 预处理且 CoM 已平滑后的 JSON。处理步骤除插值方向外，
保持与旧 z-score 流程一致：detrend -> z-score -> Pearson/nRMSE/xcorr -> lag 对齐画图。
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
OUTPUT_DIR = Path(r"H:\COM\temp\com_zscore_video_to_vicon_time")
OUTPUT_CSV = OUTPUT_DIR / "com_correlation_video_to_vicon_time.csv"
VIDEO_FPS = 29.996


def parse_video_com_by_trial(json_path: Path) -> dict[str, list[dict[str, float]]]:
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

    # 核心修正：以 Vicon 时间点为目标，在相邻视频帧之间线性插值 video CoM。
    video_x_interp = np.interp(target_t, video_t, video_x)
    video_y_interp = np.interp(target_t, video_t, video_y)

    vicon_y_aligned = -np.array(vicon_y_mm, dtype=float)[overlap_mask]
    vicon_z_aligned = -np.array(vicon_z_mm, dtype=float)[overlap_mask]
    missing_pct = (len(vicon_t) - int(overlap_mask.sum())) / len(vicon_t) * 100.0

    rows: list[dict[str, Any]] = []
    for axis_label, video_sig, vicon_sig, suffix in [
        ("horizontal (video_x vs vicon_Y)", video_x_interp, vicon_y_aligned, "x"),
        ("vertical (video_y vs vicon_Z)", video_y_interp, vicon_z_aligned, "z"),
    ]:
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
            OUTPUT_DIR / f"{trial}_{suffix}.png",
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


def clear_output() -> None:
    workspace = Path(r"H:\COM").resolve()
    target = OUTPUT_DIR.resolve()
    if not str(target).startswith(str(workspace) + "\\"):
        raise RuntimeError(f"Refusing to clear outside workspace: {target}")
    if not target.exists():
        return
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trial", "axis", "pearson_r", "p_value", "nrmse", "n_frames",
        "xcorr_peak_r", "xcorr_lag_frames", "xcorr_lag_ms", "warning",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    manifest = load_manifest(MANIFEST_CSV)
    video_by_trial = parse_video_com_by_trial(KEYPOINTS_JSON)
    clear_output()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for trial, manifest_row in manifest.items():
        if trial not in video_by_trial:
            continue
        rows.extend(process_trial(trial, manifest_row, video_by_trial[trial]))

    write_csv(rows)
    print(f"Input: {KEYPOINTS_JSON}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Rows: {len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
