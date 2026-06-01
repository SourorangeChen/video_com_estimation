from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validate_com_normalized import (
    compute_metrics,
    compute_xcorr,
    remove_linear_drift,
    zscore_normalize,
)


VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
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
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key == "trial":
                continue
            row[key] = float(value)
    return rows


def group_by_trial(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["trial"], []).append(row)
    for trial_rows in result.values():
        trial_rows.sort(key=lambda row: row["time_s"])
    return result


def build_axis_signals(
    video_rows: list[dict[str, Any]],
    vicon_rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    video_t = np.array([row["time_s"] for row in video_rows], dtype=float)
    vicon_t = np.array([row["time_s"] for row in vicon_rows], dtype=float)

    vicon_y = np.array([row["com_y_m"] for row in vicon_rows], dtype=float)
    vicon_z = np.array([row["com_z_m"] for row in vicon_rows], dtype=float)

    return {
        "horizontal": {
            "video": np.array([row["com_x_m"] for row in video_rows], dtype=float),
            "vicon": -np.interp(video_t, vicon_t, vicon_y),
        },
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
    shift = abs(lag_frames)
    if lag_frames < 0:
        video_plot = video_z[:-shift] if shift > 0 else video_z
        vicon_plot = vicon_z[shift:] if shift > 0 else vicon_z
        time_plot = time_s[:-shift] if shift > 0 else time_s
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
    output_dir: Path,
) -> list[dict[str, Any]]:
    if len(video_rows) < 5 or len(vicon_rows) < 5:
        return []

    video_t_all = np.array([row["time_s"] for row in video_rows], dtype=float)
    vicon_t = np.array([row["time_s"] for row in vicon_rows], dtype=float)
    mask = (video_t_all >= vicon_t.min()) & (video_t_all <= vicon_t.max())
    clipped_video_rows = [row for row, keep in zip(video_rows, mask) if keep]
    if len(clipped_video_rows) < 5:
        return []

    video_t = np.array([row["time_s"] for row in clipped_video_rows], dtype=float)
    axes = build_axis_signals(clipped_video_rows, vicon_rows)

    result_rows: list[dict[str, Any]] = []
    for axis_name, signals in axes.items():
        video_z = zscore_normalize(remove_linear_drift(signals["video"]))
        vicon_z = zscore_normalize(remove_linear_drift(signals["vicon"]))
        r, p_value, nrmse = compute_metrics(video_z, vicon_z)
        xcorr_peak, lag_frames, lag_ms = compute_xcorr(video_z, vicon_z, VIDEO_FPS)

        output_path = output_dir / f"{trial}_{axis_name}_zscore.png"
        plot_zscore(
            video_t,
            video_z,
            vicon_z,
            lag_frames,
            f"{trial} CoM {axis_name}: video vs Vicon z-score | r={r:.3f}",
            output_path,
        )

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
    video_by_trial = group_by_trial(load_rows(VIDEO_METRIC_CSV))
    vicon_by_trial = group_by_trial(load_rows(VICON_METRIC_CSV))
    rows: list[dict[str, Any]] = []
    for trial in sorted(video_by_trial):
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
