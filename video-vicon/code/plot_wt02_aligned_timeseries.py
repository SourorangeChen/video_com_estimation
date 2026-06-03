"""WT02 视频 vs Vicon 运动学的时间序列对比(前 100 对对齐帧)。

复用 plot_wt02_aligned_video_vicon_2d.build_aligned_selections() 得到对齐帧对，
逐区间计算位移/速度/摆长 l/xCoM 偏移，写出对比 CSV 并绘制 4 张对比图。

xCoM 偏移的轴对齐：video x ↔ −Vicon Y；video y_up ↔ Vicon Z。
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

# 无界面后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用对齐帧对的构造逻辑与帧数常量。
from plot_wt02_aligned_video_vicon_2d import FRAME_COUNT, build_aligned_selections


# 输出目录与对比 CSV(目录可由环境变量覆盖)。
OUTPUT_DIR = Path(os.environ.get(
    "ALIGNED_TIMESERIES_OUTPUT_DIR",
    r"H:\COM\video-vicon\validation\WT02_first100_aligned_timeseries",
))
OUTPUT_CSV = OUTPUT_DIR / "WT02_aligned_100_metric_comparison.csv"


def build_interval_comparison_rows(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """对每对(视频行, Vicon行)计算逐区间对比量，组装为结果行列表。"""
    rows: list[dict[str, Any]] = []
    for idx, (video_row, vicon_row) in enumerate(matched):
        if idx == 0:
            # 首帧没有"上一帧"，所有区间量(位移/速度/xCoM 偏移)记为 0。
            vicon_displacement_yz = 0.0
            vicon_vcom_yz = 0.0
            video_xcom_horizontal_delta = 0.0
            video_xcom_vertical_delta = 0.0
            vicon_xcom_horizontal_delta = 0.0
            vicon_xcom_vertical_delta = 0.0
        else:
            # 取上一对帧。
            prev_video, prev_vicon = matched[idx - 1]
            # Vicon 在 Y-Z 平面上相对上一帧的位移与时间间隔。
            dy = vicon_row["com_y_m"] - prev_vicon["com_y_m"]
            dz = vicon_row["com_z_m"] - prev_vicon["com_z_m"]
            dt = vicon_row["time_s"] - prev_vicon["time_s"]
            vicon_displacement_yz = math.hypot(dy, dz)
            # 由位移与 dt 得到速度(dt<=0 时记 0 以防除零)。
            vicon_vcom_yz = vicon_displacement_yz / dt if dt > 0 else 0.0
            # 视频 xCoM 相对首帧的偏移。
            video_xcom_horizontal_delta = video_row["xcom_x_m"] - matched[0][0]["xcom_x_m"]
            video_xcom_vertical_delta = video_row["xcom_y_m_up"] - matched[0][0]["xcom_y_m_up"]
            # 轴对齐：video x ↔ −Vicon Y；video y_up ↔ Vicon Z。
            vicon_xcom_horizontal_delta = -(vicon_row["xcom_y_m"] - matched[0][1]["xcom_y_m"])
            vicon_xcom_vertical_delta = vicon_row["xcom_z_m"] - matched[0][1]["xcom_z_m"]

        rows.append({
            "index": idx + 1,
            "video_frame": int(video_row["frame"]),
            "vicon_frame": int(vicon_row["frame"]),
            "video_time_s": video_row["time_s"],
            "vicon_time_s": vicon_row["time_s"],
            # 配对的时间误差(ms)。
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
    """把对比结果行写入 CSV(列名取自首行的键)。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_two_series(rows, y_video, y_vicon, ylabel, title, output_path: Path) -> None:
    """绘制视频 vs Vicon 两条时间序列曲线(横轴为视频时间)。"""
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
    """绘制 xCoM 偏移的水平/竖直两个分量随时间的对比(上下两子图)。"""
    t = [row["video_time_s"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    # 上子图：水平方向 xCoM 偏移。
    axes[0].plot(t, [row["video_xcom_horizontal_delta_m"] for row in rows], color="#1f77b4", label="video xCoM horizontal")
    axes[0].plot(t, [row["vicon_xcom_horizontal_delta_m"] for row in rows], color="#d62728", label="Vicon xCoM horizontal")
    axes[0].set_ylabel("Horizontal xCoM delta (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    # 下子图：竖直方向 xCoM 偏移。
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
    """主流程：取对齐帧对 -> 计算区间对比 -> 写 CSV -> 画 4 张对比图。"""
    video_selection, vicon_selection = build_aligned_selections()
    # 取每对的指标行(去掉关键点/轨迹部分)。
    matched = [(video_selection[i][0], vicon_selection[i][0]) for i in range(FRAME_COUNT)]
    rows = build_interval_comparison_rows(matched)
    write_rows(rows, OUTPUT_CSV)
    # 位移对比。
    plot_two_series(
        rows,
        "video_displacement_m",
        "vicon_displacement_yz_m",
        "Displacement over video-frame interval (m)",
        "WT02 aligned displacement, first 100 paired frames",
        OUTPUT_DIR / "WT02_aligned_displacement_video_vs_vicon.png",
    )
    # CoM 速度对比。
    plot_two_series(
        rows,
        "video_vcom_m_s",
        "vicon_vcom_yz_m_s",
        "CoM velocity magnitude (m/s)",
        "WT02 aligned vCoM, first 100 paired frames",
        OUTPUT_DIR / "WT02_aligned_vcom_video_vs_vicon.png",
    )
    # 摆长 l 对比。
    plot_two_series(
        rows,
        "video_l_m",
        "vicon_l_m",
        "l (m)",
        "WT02 aligned l, first 100 paired frames",
        OUTPUT_DIR / "WT02_aligned_l_video_vs_vicon.png",
    )
    # xCoM 偏移对比(水平+竖直)。
    plot_xcom(rows, OUTPUT_DIR / "WT02_aligned_xcom_video_vs_vicon.png")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
