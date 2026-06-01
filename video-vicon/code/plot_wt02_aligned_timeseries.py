from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_wt02_aligned_video_vicon_2d import FRAME_COUNT, build_aligned_selections


OUTPUT_DIR = Path(os.environ.get(
    "ALIGNED_TIMESERIES_OUTPUT_DIR",
    r"H:\COM\video-vicon\validation\WT02_first100_aligned_timeseries",
))
OUTPUT_CSV = OUTPUT_DIR / "WT02_aligned_100_metric_comparison.csv"


def build_interval_comparison_rows(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (video_row, vicon_row) in enumerate(matched):
        if idx == 0:
            vicon_displacement_yz = 0.0
            vicon_vcom_yz = 0.0
            video_xcom_horizontal_delta = 0.0
            video_xcom_vertical_delta = 0.0
            vicon_xcom_horizontal_delta = 0.0
            vicon_xcom_vertical_delta = 0.0
        else:
            prev_video, prev_vicon = matched[idx - 1]
            dy = vicon_row["com_y_m"] - prev_vicon["com_y_m"]
            dz = vicon_row["com_z_m"] - prev_vicon["com_z_m"]
            dt = vicon_row["time_s"] - prev_vicon["time_s"]
            vicon_displacement_yz = math.hypot(dy, dz)
            vicon_vcom_yz = vicon_displacement_yz / dt if dt > 0 else 0.0
            video_xcom_horizontal_delta = video_row["xcom_x_m"] - matched[0][0]["xcom_x_m"]
            video_xcom_vertical_delta = video_row["xcom_y_m_up"] - matched[0][0]["xcom_y_m_up"]
            # video x aligns with -Vicon Y; video y_up aligns with Vicon Z.
            vicon_xcom_horizontal_delta = -(vicon_row["xcom_y_m"] - matched[0][1]["xcom_y_m"])
            vicon_xcom_vertical_delta = vicon_row["xcom_z_m"] - matched[0][1]["xcom_z_m"]

        rows.append({
            "index": idx + 1,
            "video_frame": int(video_row["frame"]),
            "vicon_frame": int(vicon_row["frame"]),
            "video_time_s": video_row["time_s"],
            "vicon_time_s": vicon_row["time_s"],
            "time_delta_ms": abs(video_row["time_s"] - vicon_row["time_s"]) * 1000.0,
            "video_displacement_m": video_row["displacement_m"],
            "vicon_displacement_yz_m": vicon_displacement_yz,
            "video_vcom_m_s": video_row["velocity_m_s"],
            "vicon_vcom_yz_m_s": vicon_vcom_yz,
            "video_l_m": video_row["l_m"],
            "vicon_l_m": vicon_row["com_z_m"],
            "video_xcom_horizontal_delta_m": video_xcom_horizontal_delta,
            "vicon_xcom_horizontal_delta_m": vicon_xcom_horizontal_delta,
            "video_xcom_vertical_delta_m": video_xcom_vertical_delta,
            "vicon_xcom_vertical_delta_m": vicon_xcom_vertical_delta,
        })
    return rows


def write_rows(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_two_series(rows, y_video, y_vicon, ylabel, title, output_path: Path) -> None:
    t = [row["video_time_s"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.plot(t, [row[y_video] for row in rows], color="#1f77b4", linewidth=1.8, label="video")
    ax.plot(t, [row[y_vicon] for row in rows], color="#d62728", linewidth=1.8, label="Vicon Y-Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_xcom(rows, output_path: Path) -> None:
    t = [row["video_time_s"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    axes[0].plot(t, [row["video_xcom_horizontal_delta_m"] for row in rows], color="#1f77b4", label="video xCoM horizontal")
    axes[0].plot(t, [row["vicon_xcom_horizontal_delta_m"] for row in rows], color="#d62728", label="Vicon xCoM horizontal")
    axes[0].set_ylabel("Horizontal xCoM delta (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, [row["video_xcom_vertical_delta_m"] for row in rows], color="#1f77b4", label="video xCoM vertical")
    axes[1].plot(t, [row["vicon_xcom_vertical_delta_m"] for row in rows], color="#d62728", label="Vicon xCoM vertical")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Vertical xCoM delta (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle("WT02 aligned xCoM time series, first 100 paired frames")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    video_selection, vicon_selection = build_aligned_selections()
    matched = [(video_selection[i][0], vicon_selection[i][0]) for i in range(FRAME_COUNT)]
    rows = build_interval_comparison_rows(matched)
    write_rows(rows, OUTPUT_CSV)
    plot_two_series(
        rows,
        "video_displacement_m",
        "vicon_displacement_yz_m",
        "Displacement over video-frame interval (m)",
        "WT02 aligned displacement, first 100 paired frames",
        OUTPUT_DIR / "WT02_aligned_displacement_video_vs_vicon.png",
    )
    plot_two_series(
        rows,
        "video_vcom_m_s",
        "vicon_vcom_yz_m_s",
        "CoM velocity magnitude (m/s)",
        "WT02 aligned vCoM, first 100 paired frames",
        OUTPUT_DIR / "WT02_aligned_vcom_video_vs_vicon.png",
    )
    plot_two_series(
        rows,
        "video_l_m",
        "vicon_l_m",
        "l (m)",
        "WT02 aligned l, first 100 paired frames",
        OUTPUT_DIR / "WT02_aligned_l_video_vs_vicon.png",
    )
    plot_xcom(rows, OUTPUT_DIR / "WT02_aligned_xcom_video_vs_vicon.png")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
