from __future__ import annotations

"""Export CoM signals after detrend and before z-score normalization.

This follows the current keypoints-preprocessed CoM z-score validation flow:
video CoM is linearly interpolated to the Vicon time axis, axes are direction
aligned, then both signals are linearly detrended. No source data or validation
outputs are modified.
"""

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"H:\COM")
CODE_DIR = ROOT / "video-vicon" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from validate_com_keypoints_preprocessed_zscore import (  # noqa: E402
    KEYPOINTS_JSON,
    VIDEO_FPS,
    parse_video_com_by_trial,
)
from validate_com_normalized import (  # noqa: E402
    MANIFEST_CSV,
    VICON_CSV_ROOT,
    build_vicon_time_axis,
    load_manifest,
    parse_vicon_model_outputs,
    remove_linear_drift,
)


OUTPUT_DIR = ROOT / "temp" / "com_keypoints_preprocessed_detrend"
DETAIL_CSV = OUTPUT_DIR / "com_detrend_pre_zscore_samples.csv"
SUMMARY_CSV = OUTPUT_DIR / "com_detrend_pre_zscore_summary.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"


def linear_trend(original: np.ndarray, detrended: np.ndarray) -> np.ndarray:
    """Return the removed linear trend, matching scipy.signal.detrend output."""
    return original - detrended


def plot_detrended_pair(
    time_s: np.ndarray,
    video_det: np.ndarray,
    vicon_det: np.ndarray,
    trial: str,
    axis_suffix: str,
    axis_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.plot(time_s, video_det, color="tab:blue", linewidth=1.2, label="video detrended (px)")
    ax.plot(time_s, vicon_det, color="tab:red", linewidth=1.2, alpha=0.85, label="Vicon detrended (mm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("detrended value before z-score")
    ax.set_title(f"{trial} {axis_label}: detrend result before z-score")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def process_trial(
    trial: str,
    manifest_row: dict[str, Any],
    video_rows: list[dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vicon_csv = VICON_CSV_ROOT / f"{trial}.csv"
    if not vicon_csv.exists() or len(video_rows) < 5:
        return [], []

    rate_hz, vicon_frames, _vicon_x_mm, vicon_y_mm, vicon_z_mm = parse_vicon_model_outputs(vicon_csv)
    if len(vicon_frames) < 5:
        return [], []

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
        return [], []

    target_vicon_frames = np.array(vicon_frames, dtype=int)[overlap_mask]
    video_x_interp = np.interp(target_t, video_t, video_x)
    video_y_interp = np.interp(target_t, video_t, video_y)
    vicon_y_aligned = -np.array(vicon_y_mm, dtype=float)[overlap_mask]
    vicon_z_aligned = -np.array(vicon_z_mm, dtype=float)[overlap_mask]

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    axes = [
        ("horizontal", "video_x vs -vicon_Y", "x", video_x_interp, vicon_y_aligned),
        ("vertical", "video_y vs -vicon_Z", "z", video_y_interp, vicon_z_aligned),
    ]
    for axis_name, axis_label, suffix, video_sig, vicon_sig in axes:
        video_det = remove_linear_drift(video_sig)
        vicon_det = remove_linear_drift(vicon_sig)
        video_trend = linear_trend(video_sig, video_det)
        vicon_trend = linear_trend(vicon_sig, vicon_det)

        plot_detrended_pair(
            target_t,
            video_det,
            vicon_det,
            trial,
            suffix,
            axis_label,
            PLOTS_DIR / f"{trial}_{suffix}_detrend_pre_zscore.png",
        )

        for idx, time_s in enumerate(target_t):
            detail_rows.append({
                "trial": trial,
                "axis": axis_name,
                "axis_mapping": axis_label,
                "sample_index": idx,
                "time_s": f"{float(time_s):.9f}",
                "vicon_frame": int(target_vicon_frames[idx]),
                "video_interp_raw": f"{float(video_sig[idx]):.9f}",
                "video_linear_trend": f"{float(video_trend[idx]):.9f}",
                "video_detrended_pre_zscore": f"{float(video_det[idx]):.9f}",
                "vicon_aligned_raw": f"{float(vicon_sig[idx]):.9f}",
                "vicon_linear_trend": f"{float(vicon_trend[idx]):.9f}",
                "vicon_detrended_pre_zscore": f"{float(vicon_det[idx]):.9f}",
                "video_unit": "px",
                "vicon_unit": "mm",
                "target_time_axis": "Vicon time_s",
                "vicon_rate_hz": f"{float(rate_hz):.6f}",
                "video_fps": f"{float(VIDEO_FPS):.6f}",
            })

        summary_rows.append({
            "trial": trial,
            "axis": axis_name,
            "axis_mapping": axis_label,
            "n_samples": len(target_t),
            "target_time_axis": "Vicon time_s",
            "video_detrended_mean": f"{float(video_det.mean()):.9f}",
            "video_detrended_std": f"{float(video_det.std()):.9f}",
            "vicon_detrended_mean": f"{float(vicon_det.mean()):.9f}",
            "vicon_detrended_std": f"{float(vicon_det.std()):.9f}",
            "video_unit": "px",
            "vicon_unit": "mm",
        })
    return detail_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST_CSV)
    video_by_trial = parse_video_com_by_trial(KEYPOINTS_JSON)

    all_detail_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    for trial, manifest_row in manifest.items():
        if trial not in video_by_trial:
            continue
        detail_rows, summary_rows = process_trial(trial, manifest_row, video_by_trial[trial])
        all_detail_rows.extend(detail_rows)
        all_summary_rows.extend(summary_rows)

    detail_fields = [
        "trial",
        "axis",
        "axis_mapping",
        "sample_index",
        "time_s",
        "vicon_frame",
        "video_interp_raw",
        "video_linear_trend",
        "video_detrended_pre_zscore",
        "vicon_aligned_raw",
        "vicon_linear_trend",
        "vicon_detrended_pre_zscore",
        "video_unit",
        "vicon_unit",
        "target_time_axis",
        "vicon_rate_hz",
        "video_fps",
    ]
    summary_fields = [
        "trial",
        "axis",
        "axis_mapping",
        "n_samples",
        "target_time_axis",
        "video_detrended_mean",
        "video_detrended_std",
        "vicon_detrended_mean",
        "vicon_detrended_std",
        "video_unit",
        "vicon_unit",
    ]
    write_csv(DETAIL_CSV, all_detail_rows, detail_fields)
    write_csv(SUMMARY_CSV, all_summary_rows, summary_fields)

    print(f"Input JSON: {KEYPOINTS_JSON}")
    print(f"Detail CSV: {DETAIL_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Plots: {PLOTS_DIR}")
    print(f"Trials/axes: {len(all_summary_rows)}")
    print(f"Samples: {len(all_detail_rows)}")
    return 0 if all_detail_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
