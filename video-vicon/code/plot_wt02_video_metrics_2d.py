"""绘制 WT02 试验视频侧的 2D CoM 运动学：COCO 关键点 + CoM/xCoM/位移/速度向量，前 50 帧。

与 plot_wt02_kinematics_3d.py 对应，但这里是视频 2D 像素坐标(原点左上、Y 向下)：
    - 数据来源：video_com_metric.csv(逐帧像素与物理量) + keypoints_and_com.json(关键点)；
    - 输出逐帧图与一张 CoM/xCoM 轨迹总览图。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

# 无界面后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# 复用 video_com_metric 中的输入 JSON 与输出 CSV 路径常量。
from video_com_metric import KEYPOINTS_JSON, OUTPUT_CSV


TRIAL = "WT02"
FRAME_COUNT = 50
OUTPUT_DIR = Path(r"H:\COM\video-vicon\validation\WT02_first50_video_metrics_2d")
VELOCITY_VECTOR_SECONDS = 0.05    # 速度向量按 0.05 秒缩放绘制


def add_side_legend(ax, loc: str, fontsize: int | None = None) -> None:
    """在已有图例基础上追加"左侧(绿)/右侧(红)"颜色说明。"""
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color="#2b8a3e", linewidth=2.0),
        Line2D([0], [0], color="#c92a2a", linewidth=2.0),
    ])
    labels.extend(["left side", "right side"])
    ax.legend(handles, labels, loc=loc, fontsize=fontsize)

# COCO-17 骨架连线(关键点索引对)：躯干/四肢/头面部。
COCO_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def load_metric_rows(csv_path: Path) -> list[dict[str, Any]]:
    """读取视频运动学 CSV，trial/frame 之外的列统一转 float。"""
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted: dict[str, Any] = {"trial": row["trial"], "frame": int(row["frame"])}
            for key, value in row.items():
                if key in {"trial", "frame"}:
                    continue
                converted[key] = float(value)
            rows.append(converted)
    return rows


def load_keypoint_rows(keypoints_json: Path, trial: str) -> dict[int, dict[str, Any]]:
    """读取某试验的关键点记录，按帧号建立索引。"""
    records = json.loads(keypoints_json.read_text(encoding="utf-8"))
    result: dict[int, dict[str, Any]] = {}
    prefix = f"Video_{trial}_Trajectory"
    for entry in records:
        if not isinstance(entry, dict):
            continue
        # 仅保留该试验的记录。
        image = entry.get("image", "")
        if not image.startswith(prefix):
            continue
        try:
            frame = int(Path(image).stem.split("_")[-1])
        except ValueError:
            continue
        result[frame] = {
            "frame": frame,
            "image": image,
            "keypoints": entry.get("keypoints", []),
        }
    return result


def select_trial_frames(
    metric_rows: list[dict[str, Any]],
    keypoint_rows: dict[int, dict[str, Any]],
    trial: str,
    limit: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """把指标行与关键点行按帧号配对，最多取 limit 对。"""
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in metric_rows:
        # 只处理目标试验。
        if row["trial"] != trial:
            continue
        keypoint_row = keypoint_rows.get(row["frame"])
        # 没有对应关键点则跳过。
        if keypoint_row is None:
            continue
        selected.append((row, keypoint_row))
        if len(selected) == limit:
            break
    return selected


def valid_keypoints(keypoints: Any) -> dict[int, tuple[float, float]]:
    """从关键点列表中提取有效的 (x, y) 点，返回 {索引: (x, y)}。"""
    points: dict[int, tuple[float, float]] = {}
    if not isinstance(keypoints, list):
        return points
    for idx, point in enumerate(keypoints):
        # 跳过结构异常或坐标缺失的点。
        if not isinstance(point, list) or len(point) < 2:
            continue
        if point[0] is None or point[1] is None:
            continue
        try:
            points[idx] = (float(point[0]), float(point[1]))
        except (TypeError, ValueError):
            continue
    return points


def limits_for_selection(selection: list[tuple[dict[str, Any], dict[str, Any]]]):
    """根据所选帧的关键点与 CoM/xCoM 像素坐标计算绘图范围(各留 12% 边距)。"""
    xs: list[float] = []
    ys: list[float] = []
    for metric, keypoint_row in selection:
        # 纳入所有有效关键点。
        for x, y in valid_keypoints(keypoint_row["keypoints"]).values():
            xs.append(x)
            ys.append(y)
        # 纳入 CoM 与 xCoM 像素坐标。
        xs.extend([metric["com_x_px"], metric["xcom_x_px"]])
        ys.extend([metric["com_y_px"], metric["xcom_y_px"]])

    x_margin = (max(xs) - min(xs)) * 0.12
    y_margin = (max(ys) - min(ys)) * 0.12
    return (min(xs) - x_margin, max(xs) + x_margin), (min(ys) - y_margin, max(ys) + y_margin)


def render_frame(
    metric: dict[str, Any],
    keypoint_row: dict[str, Any],
    limits,
    output_path: Path,
    index: int,
    total: int,
) -> None:
    """渲染单帧的视频 2D 图(关键点骨架 + CoM/xCoM/位移/速度)。"""
    points = valid_keypoints(keypoint_row["keypoints"])
    fig, ax = plt.subplots(figsize=(7.5, 9), dpi=150)

    # 画 COCO 骨架连线：含左侧索引→绿，含右侧索引→红，其余灰。
    for a, b in COCO_EDGES:
        if a in points and b in points:
            xa, ya = points[a]
            xb, yb = points[b]
            color = "#4a4a4a"
            if a in {5, 7, 9, 11, 13, 15} or b in {5, 7, 9, 11, 13, 15}:
                color = "#2b8a3e"
            if a in {6, 8, 10, 12, 14, 16} or b in {6, 8, 10, 12, 14, 16}:
                color = "#c92a2a"
            ax.plot([xa, xb], [ya, yb], color=color, linewidth=2.0, alpha=0.85)

    # 画关键点散点。
    if points:
        xy = np.array(list(points.values()))
        ax.scatter(xy[:, 0], xy[:, 1], s=18, c="#111111", label="video keypoints", zorder=3)

    # 取 CoM、xCoM、位移(均为像素)。
    com = np.array([metric["com_x_px"], metric["com_y_px"]])
    xcom = np.array([metric["xcom_x_px"], metric["xcom_y_px"]])
    displacement = np.array([metric["displacement_x_px"], metric["displacement_y_px"]])
    # 速度从"米/秒(向上为正)"换算为"像素/秒(向下为正)"：x 直接乘 ppm，y 需取负号。
    velocity_px_s = np.array([
        metric["velocity_x_m_s"] * metric["pixels_per_meter"],
        -metric["velocity_y_m_s_up"] * metric["pixels_per_meter"],
    ])

    # CoM(蓝点)、xCoM(橙叉)及连线。
    ax.scatter(*com, s=80, c="#1f77b4", label="CoM", zorder=5)
    ax.scatter(*xcom, s=100, c="#ff7f0e", marker="x", linewidths=2.6, label="xCoM", zorder=6)
    ax.plot([com[0], xcom[0]], [com[1], xcom[1]], color="#ff7f0e", linewidth=1.2)

    # 位移箭头(从上一位置指向当前 CoM)。
    disp_start = com - displacement
    ax.arrow(
        disp_start[0],
        disp_start[1],
        displacement[0],
        displacement[1],
        color="#9467bd",
        width=1.4,
        length_includes_head=True,
        label="displacement",
        zorder=4,
    )
    # 速度箭头(按 0.05s 缩放)。
    ax.arrow(
        com[0],
        com[1],
        velocity_px_s[0] * VELOCITY_VECTOR_SECONDS,
        velocity_px_s[1] * VELOCITY_VECTOR_SECONDS,
        color="#17a2b8",
        width=1.4,
        length_includes_head=True,
        label="velocity x 0.05s",
        zorder=4,
    )

    # 左上角标注各标量(含像素/米比例 ppm)。
    metric_text = (
        f"disp = {metric['displacement_m']:.4f} m\n"
        f"vel = {metric['velocity_m_s']:.3f} m/s\n"
        f"ppm = {metric['pixels_per_meter']:.1f} px/m\n"
        f"l = {metric['l_m']:.3f} m\n"
        f"omega0 = {metric['omega0_rad_s']:.3f} rad/s"
    )
    ax.text(0.02, 0.98, metric_text, transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    # 像素坐标 Y 向下，反转 Y 轴使显示符合图像方向。
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("video X (px)")
    ax.set_ylabel("video Y (px, downward)")
    ax.set_title(f"WT02 video front/back metrics | Frame {metric['frame']} | {index + 1}/{total}")
    ax.grid(True, alpha=0.25)
    add_side_legend(ax, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_overview(selection: list[tuple[dict[str, Any], dict[str, Any]]], limits, output_path: Path) -> None:
    """渲染总览图：首帧骨架做背景 + CoM/xCoM 整段像素轨迹。"""
    fig, ax = plt.subplots(figsize=(7.5, 9), dpi=150)
    # 首帧骨架做淡灰背景。
    first_points = valid_keypoints(selection[0][1]["keypoints"])
    for a, b in COCO_EDGES:
        if a in first_points and b in first_points:
            xa, ya = first_points[a]
            xb, yb = first_points[b]
            ax.plot([xa, xb], [ya, yb], color="#b0b0b0", linewidth=1.2, alpha=0.55)

    # CoM 与 xCoM 整段轨迹及起止点。
    com_path = np.array([[metric["com_x_px"], metric["com_y_px"]] for metric, _ in selection])
    xcom_path = np.array([[metric["xcom_x_px"], metric["xcom_y_px"]] for metric, _ in selection])
    ax.plot(com_path[:, 0], com_path[:, 1], color="#1f77b4", linewidth=2.2, label="CoM path")
    ax.plot(xcom_path[:, 0], xcom_path[:, 1], color="#ff7f0e", linewidth=2.0, label="xCoM path")
    ax.scatter(com_path[0, 0], com_path[0, 1], c="#1f77b4", s=65, label="start")
    ax.scatter(com_path[-1, 0], com_path[-1, 1], c="#d62728", s=65, label="end")

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("video X (px)")
    ax.set_ylabel("video Y (px, downward)")
    ax.set_title("WT02 video first 50 frames | CoM and xCoM paths")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    """主流程：配对前 50 帧，先出总览图，再逐帧出 2D 图。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = load_metric_rows(OUTPUT_CSV)
    keypoint_rows = load_keypoint_rows(KEYPOINTS_JSON, TRIAL)
    selection = select_trial_frames(metric_rows, keypoint_rows, TRIAL, FRAME_COUNT)
    if len(selection) < FRAME_COUNT:
        raise RuntimeError(f"Only found {len(selection)} matching frames for {TRIAL}")

    limits = limits_for_selection(selection)
    # 先出总览图。
    render_overview(selection, limits, OUTPUT_DIR / "WT02_video_first50_overview_com_xcom_path.png")
    # 逐帧出图。
    for index, (metric, keypoint_row) in enumerate(selection):
        output_path = OUTPUT_DIR / f"WT02_video_frame_{index + 1:02d}_frame_{metric['frame']:04d}.png"
        render_frame(metric, keypoint_row, limits, output_path, index, len(selection))

    print(f"Output: {OUTPUT_DIR}")
    print(f"Rendered frames: {len(selection)}")
    print(f"First video frame: {selection[0][0]['frame']}")
    print(f"Last video frame: {selection[-1][0]['frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
