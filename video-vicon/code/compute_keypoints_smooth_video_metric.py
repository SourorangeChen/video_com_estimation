"""从 keypoints 预处理后的 CoM 计算视频侧运动学指标。

该脚本读取 keypoints_and_com_preprocessed.json，先对视频 CoM 像素坐标做
centered moving average，再用肩宽 0.34 m 估计 pixels_per_meter，并计算
CoM 位移、速度、摆长 l、omega0 和 xCoM。

与 video_com_metric.py 的区别：此处用"肩宽"做像素->米标定(而非鼻-踝身高)，
且输入是已对关键点做过中值+SG 平滑的 JSON。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# 复用 video_com_metric 中的平滑窗口常量、路径解析与平滑函数。
from video_com_metric import SMOOTHING_WINDOW, parse_trial_and_frame, smooth_signal


# 输入(预处理后关键点 JSON) 与 输出 CSV。
INPUT_JSON = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_keypoints-preprocessed\results\keypoints_and_com_preprocessed.json"
)
OUTPUT_DIR = Path(r"H:\COM\video-vicon\validation\metrics_keypoints_preprocessed")
OUTPUT_CSV = OUTPUT_DIR / "video_com_metric.csv"

VIDEO_FPS = 29.996
SHOULDER_WIDTH_M = 0.34     # 假设的左右肩真实间距(米)，用于像素->米标定
GRAVITY_M_S2 = 9.81

# 用到的 COCO 关键点索引。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# 记录在输出中的预处理元信息(便于溯源)。
KEYPOINT_PREPROCESSING_METHOD = "median_filter_then_savitzky_golay"
MEDIAN_WINDOW = 3
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2
COM_SMOOTHING_METHOD = "moving_average"
COM_SMOOTHING_WINDOW = SMOOTHING_WINDOW


def keypoint_xy(keypoints: Any, index: int) -> tuple[float, float] | None:
    """安全地取第 index 个关键点的 (x, y)；结构异常或缺失返回 None。"""
    if not isinstance(keypoints, list) or index >= len(keypoints):
        return None
    point = keypoints[index]
    if not isinstance(point, list) or len(point) < 2:
        return None
    if point[0] is None or point[1] is None:
        return None
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None


def load_video_trials(keypoints_json: Path) -> dict[str, list[dict[str, Any]]]:
    """读取预处理 JSON，按试验聚合每帧的 CoM、双肩、双踝坐标与肩宽。"""
    records = json.loads(keypoints_json.read_text(encoding="utf-8"))
    trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in records:
        if not isinstance(entry, dict):
            continue
        # 优先用 image 路径解析试验/帧号；解析不出则回退到记录里的 trial/frame 字段。
        parsed = parse_trial_and_frame(str(entry.get("image", "")))
        if parsed is None:
            trial = entry.get("trial")
            frame = entry.get("frame")
            if trial is None or frame is None:
                continue
            parsed = str(trial), int(frame)
        trial, frame = parsed

        com = entry.get("com")
        keypoints = entry.get("keypoints")
        if not isinstance(com, dict):
            continue

        # 取双肩与双踝坐标(用于标定与地面线)。
        left_shoulder = keypoint_xy(keypoints, LEFT_SHOULDER)
        right_shoulder = keypoint_xy(keypoints, RIGHT_SHOULDER)
        left_ankle = keypoint_xy(keypoints, LEFT_ANKLE)
        right_ankle = keypoint_xy(keypoints, RIGHT_ANKLE)
        if left_shoulder is None or right_shoulder is None or left_ankle is None or right_ankle is None:
            continue

        # 肩宽像素 = 双肩欧氏距离。
        shoulder_width_px = float(np.hypot(
            left_shoulder[0] - right_shoulder[0],
            left_shoulder[1] - right_shoulder[1],
        ))
        # 肩宽非正则该帧无法标定，跳过。
        if shoulder_width_px <= 0.0:
            continue

        trials[trial].append({
            "frame": int(frame),
            "com_x_px": float(com["com_x"]),
            "com_y_px": float(com["com_y"]),
            "left_shoulder_x_px": left_shoulder[0],
            "left_shoulder_y_px": left_shoulder[1],
            "right_shoulder_x_px": right_shoulder[0],
            "right_shoulder_y_px": right_shoulder[1],
            "shoulder_width_px": shoulder_width_px,
            "left_ankle_y_px": left_ankle[1],
            "right_ankle_y_px": right_ankle[1],
        })
    return dict(trials)


def compute_video_com_metrics(
    trial: str,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """计算单个 trial 的视频 CoM 运动学指标。"""
    frame_numbers = [entry["frame"] for entry in entries]
    raw_com_x_px = np.array([entry["com_x_px"] for entry in entries], dtype=float)
    raw_com_y_px = np.array([entry["com_y_px"] for entry in entries], dtype=float)
    # 这里再做一次 CoM 像素级平滑，降低 CoM 帧间抖动对速度和 xCoM 的放大。
    com_x_px = smooth_signal(raw_com_x_px, COM_SMOOTHING_WINDOW)
    com_y_px = smooth_signal(raw_com_y_px, COM_SMOOTHING_WINDOW)
    shoulder_width_px = np.array([entry["shoulder_width_px"] for entry in entries], dtype=float)
    left_ankle_y_px = np.array([entry["left_ankle_y_px"] for entry in entries], dtype=float)
    right_ankle_y_px = np.array([entry["right_ankle_y_px"] for entry in entries], dtype=float)

    # 至少两帧才能算速度。
    if len(frame_numbers) < 2:
        raise ValueError("at least two frames are required to compute velocity")
    if np.any(shoulder_width_px <= 0.0):
        raise ValueError("shoulder-width scaling requires positive shoulder width for every frame")

    # 时间轴以首帧为 t=0。
    time_s = np.array([(frame - frame_numbers[0]) / VIDEO_FPS for frame in frame_numbers], dtype=float)
    # 当前尺度假设：左右肩真实距离为 0.34 m，每帧独立估计 pixels_per_meter。
    pixels_per_meter = shoulder_width_px / SHOULDER_WIDTH_M
    # 视频 y 轴向下，因此左右踝中 y 更大的点更接近画面下方，作为地面近似。
    ground_y_px = np.maximum(left_ankle_y_px, right_ankle_y_px)
    # 摆长 l：CoM 到地面的像素高度，换算为米。
    l_px = ground_y_px - com_y_px
    l_m = l_px / pixels_per_meter
    if np.any(l_m <= 0.0):
        raise ValueError("xCoM requires positive CoM-to-ground height for every frame")

    # CoM 换算为米；竖直向上为正(取负)。
    com_x_m = com_x_px / pixels_per_meter
    com_y_m_up = -com_y_px / pixels_per_meter
    com_m = np.column_stack([com_x_m, com_y_m_up])

    # 逐帧像素位移(首帧补 0)。
    displacement_px = np.column_stack([
        np.r_[0.0, np.diff(com_x_px)],
        np.r_[0.0, np.diff(com_y_px)],
    ])
    # 位移换算为米(竖直分量取反)。
    displacement_m = np.column_stack([
        displacement_px[:, 0] / pixels_per_meter,
        -displacement_px[:, 1] / pixels_per_meter,
    ])
    displacement_mag_m = np.linalg.norm(displacement_m, axis=1)

    # 速度 = 位移 × 帧率；ω₀ = sqrt(g/l)；xCoM = CoM + 速度/ω₀。
    velocity_m_s = displacement_m * VIDEO_FPS
    velocity_mag_m_s = np.linalg.norm(velocity_m_s, axis=1)
    omega0 = np.sqrt(GRAVITY_M_S2 / l_m)
    xcom_m = com_m + velocity_m_s / omega0[:, np.newaxis]
    # xCoM 换算回像素(竖直分量再次取反)。
    xcom_x_px = xcom_m[:, 0] * pixels_per_meter
    xcom_y_px = -xcom_m[:, 1] * pixels_per_meter

    # 组装逐帧结果(含原始/平滑 CoM 与各预处理元信息)。
    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        rows.append({
            "trial": trial,
            "frame": int(entry["frame"]),
            "time_s": float(time_s[idx]),
            "pixels_per_meter": float(pixels_per_meter[idx]),
            "scale_method": "shoulder_width",
            "reference_shoulder_width_m": SHOULDER_WIDTH_M,
            "shoulder_width_px": float(shoulder_width_px[idx]),
            "left_shoulder_x_px": float(entry["left_shoulder_x_px"]),
            "left_shoulder_y_px": float(entry["left_shoulder_y_px"]),
            "right_shoulder_x_px": float(entry["right_shoulder_x_px"]),
            "right_shoulder_y_px": float(entry["right_shoulder_y_px"]),
            "ground_y_px": float(ground_y_px[idx]),
            "com_x_px": float(com_x_px[idx]),
            "com_y_px": float(com_y_px[idx]),
            "raw_com_x_px": float(raw_com_x_px[idx]),
            "raw_com_y_px": float(raw_com_y_px[idx]),
            "com_x_m": float(com_x_m[idx]),
            "com_y_m_up": float(com_y_m_up[idx]),
            "l_px": float(l_px[idx]),
            "l_m": float(l_m[idx]),
            "displacement_x_px": float(displacement_px[idx, 0]),
            "displacement_y_px": float(displacement_px[idx, 1]),
            "displacement_x_m": float(displacement_m[idx, 0]),
            "displacement_y_m_up": float(displacement_m[idx, 1]),
            "displacement_m": float(displacement_mag_m[idx]),
            "velocity_x_m_s": float(velocity_m_s[idx, 0]),
            "velocity_y_m_s_up": float(velocity_m_s[idx, 1]),
            "velocity_m_s": float(velocity_mag_m_s[idx]),
            "omega0_rad_s": float(omega0[idx]),
            "xcom_x_m": float(xcom_m[idx, 0]),
            "xcom_y_m_up": float(xcom_m[idx, 1]),
            "xcom_x_px": float(xcom_x_px[idx]),
            "xcom_y_px": float(xcom_y_px[idx]),
            "keypoint_preprocessing_method": KEYPOINT_PREPROCESSING_METHOD,
            "keypoint_median_window": MEDIAN_WINDOW,
            "keypoint_savgol_window": SAVGOL_WINDOW,
            "keypoint_savgol_polyorder": SAVGOL_POLYORDER,
            "com_smoothing_method": COM_SMOOTHING_METHOD,
            "com_smoothing_window": COM_SMOOTHING_WINDOW,
        })
    return rows


def compute_all_video_metrics(keypoints_json: Path) -> list[dict[str, Any]]:
    """对所有试验计算指标(试验内按帧号排序，不足两帧跳过)。"""
    all_rows: list[dict[str, Any]] = []
    trials = load_video_trials(keypoints_json)
    for trial in sorted(trials):
        entries = sorted(trials[trial], key=lambda item: item["frame"])
        if len(entries) < 2:
            continue
        all_rows.extend(compute_video_com_metrics(trial, entries))
    return all_rows


def write_metric_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """把逐帧指标写入 CSV(固定列顺序)。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "frame", "time_s",
        "pixels_per_meter", "scale_method", "reference_shoulder_width_m", "shoulder_width_px",
        "left_shoulder_x_px", "left_shoulder_y_px",
        "right_shoulder_x_px", "right_shoulder_y_px",
        "ground_y_px",
        "com_x_px", "com_y_px", "raw_com_x_px", "raw_com_y_px", "com_x_m", "com_y_m_up",
        "l_px", "l_m",
        "displacement_x_px", "displacement_y_px",
        "displacement_x_m", "displacement_y_m_up", "displacement_m",
        "velocity_x_m_s", "velocity_y_m_s_up", "velocity_m_s",
        "omega0_rad_s",
        "xcom_x_m", "xcom_y_m_up", "xcom_x_px", "xcom_y_px",
        "keypoint_preprocessing_method",
        "keypoint_median_window",
        "keypoint_savgol_window",
        "keypoint_savgol_polyorder",
        "com_smoothing_method",
        "com_smoothing_window",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """主流程：计算全部试验指标并写出 CSV。"""
    rows = compute_all_video_metrics(INPUT_JSON)
    write_metric_csv(rows, OUTPUT_CSV)
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows: {len(rows)}")
    print(f"Scale: shoulder_width_px / {SHOULDER_WIDTH_M} m")
    print(f"CoM smoothing: {COM_SMOOTHING_METHOD}, window={COM_SMOOTHING_WINDOW}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
