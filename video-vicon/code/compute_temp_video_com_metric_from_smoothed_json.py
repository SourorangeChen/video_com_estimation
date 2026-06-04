from __future__ import annotations

r"""从 keypoints 预处理且 CoM 已平滑后的 JSON 重新计算视频 CoM 指标。

这是 temp 专用脚本：只读取当前 JSON 的 com 字段作为最终 CoM，不再做第二次
CoM 平滑。输出仅写入 H:\COM\temp，避免覆盖 validation 中已有结果。
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# 输入(已平滑 CoM 的预处理 JSON)与输出 CSV(写到 temp 目录)。
INPUT_JSON = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_keypoints-preprocessed\results\keypoints_and_com_preprocessed.json"
)
OUTPUT_CSV = Path(r"H:\COM\temp\video_com_metric_keypoints_preprocessed_com_smoothed.csv")

VIDEO_FPS = 29.996          # 视频帧率(Hz)
SHOULDER_WIDTH_M = 0.34     # 假设的左右肩真实间距(米)，用于像素->米标定
GRAVITY_M_S2 = 9.81         # 重力加速度(m/s²)

# 用到的 COCO 关键点索引。
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def keypoint_xy(keypoints: Any, index: int) -> tuple[float, float] | None:
    """读取一个 COCO keypoint 的 x/y 坐标；缺失或非法则返回 None。"""
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


def com_xy(record: dict[str, Any], field: str) -> tuple[float, float] | None:
    """读取 com/source_com 中的 CoM 像素坐标。"""
    com = record.get(field)
    if not isinstance(com, dict):
        return None
    try:
        return float(com["com_x"]), float(com["com_y"])
    except (KeyError, TypeError, ValueError):
        return None


def load_video_trials(json_path: Path) -> dict[str, list[dict[str, Any]]]:
    """按 trial 读取可用于 metric 计算的帧记录。"""
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list JSON records in {json_path}")

    trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue

        # 取试验名、帧号，以及平滑后(com)与平滑前(source_com)两份 CoM。
        trial = record.get("trial")
        frame = record.get("frame")
        keypoints = record.get("keypoints")
        smoothed_com = com_xy(record, "com")
        raw_com = com_xy(record, "source_com")
        # 任一关键字段缺失则跳过该帧。
        if trial is None or frame is None or smoothed_com is None or raw_com is None:
            continue

        # 取双肩与双踝(用于标定与地面线)。
        left_shoulder = keypoint_xy(keypoints, LEFT_SHOULDER)
        right_shoulder = keypoint_xy(keypoints, RIGHT_SHOULDER)
        left_ankle = keypoint_xy(keypoints, LEFT_ANKLE)
        right_ankle = keypoint_xy(keypoints, RIGHT_ANKLE)
        if left_shoulder is None or right_shoulder is None or left_ankle is None or right_ankle is None:
            continue

        # 肩宽像素 = 双肩欧氏距离；非正则无法标定，跳过。
        shoulder_width_px = float(np.hypot(
            left_shoulder[0] - right_shoulder[0],
            left_shoulder[1] - right_shoulder[1],
        ))
        if shoulder_width_px <= 0.0:
            continue

        # 取预处理/平滑元信息(若缺失则用空字典兜底)，随帧记录一并保存。
        keypoint_preprocessing = record.get("keypoint_preprocessing") or {}
        com_smoothing = record.get("com_smoothing") or {}

        trials[str(trial)].append({
            "frame": int(frame),
            "com_x_px": smoothed_com[0],
            "com_y_px": smoothed_com[1],
            "raw_com_x_px": raw_com[0],
            "raw_com_y_px": raw_com[1],
            "left_shoulder_x_px": left_shoulder[0],
            "left_shoulder_y_px": left_shoulder[1],
            "right_shoulder_x_px": right_shoulder[0],
            "right_shoulder_y_px": right_shoulder[1],
            "shoulder_width_px": shoulder_width_px,
            "left_ankle_y_px": left_ankle[1],
            "right_ankle_y_px": right_ankle[1],
            "keypoint_preprocessing_method": keypoint_preprocessing.get("method", ""),
            "keypoint_median_window": keypoint_preprocessing.get("median_window", ""),
            "keypoint_savgol_window": keypoint_preprocessing.get("savgol_window", ""),
            "keypoint_savgol_polyorder": keypoint_preprocessing.get("savgol_polyorder", ""),
            "com_smoothing_method": com_smoothing.get("method", ""),
            "com_smoothing_window": com_smoothing.get("window_frames", ""),
        })
    return dict(trials)


def compute_video_com_metrics(trial: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """使用当前 JSON 的平滑 CoM 计算单个 trial 的逐帧视频指标。"""
    # 按帧号排序，至少两帧才能算速度。
    entries = sorted(entries, key=lambda item: item["frame"])
    frame_numbers = [entry["frame"] for entry in entries]
    if len(frame_numbers) < 2:
        return []

    # 抽取各序列为数组(平滑 CoM、原始 CoM、肩宽、双踝 y)。
    com_x_px = np.array([entry["com_x_px"] for entry in entries], dtype=float)
    com_y_px = np.array([entry["com_y_px"] for entry in entries], dtype=float)
    raw_com_x_px = np.array([entry["raw_com_x_px"] for entry in entries], dtype=float)
    raw_com_y_px = np.array([entry["raw_com_y_px"] for entry in entries], dtype=float)
    shoulder_width_px = np.array([entry["shoulder_width_px"] for entry in entries], dtype=float)
    left_ankle_y_px = np.array([entry["left_ankle_y_px"] for entry in entries], dtype=float)
    right_ankle_y_px = np.array([entry["right_ankle_y_px"] for entry in entries], dtype=float)

    if np.any(shoulder_width_px <= 0.0):
        raise ValueError(f"{trial}: shoulder-width scaling requires positive shoulder width")

    # 时间轴(以首帧为 0)与逐帧像素/米标定(肩宽 / 0.34m)。
    time_s = np.array([(frame - frame_numbers[0]) / VIDEO_FPS for frame in frame_numbers], dtype=float)
    pixels_per_meter = shoulder_width_px / SHOULDER_WIDTH_M
    # 地面 y 取左右踝中更靠下者；摆长 l = (地面 - CoM) 的像素高度换算为米。
    ground_y_px = np.maximum(left_ankle_y_px, right_ankle_y_px)
    l_px = ground_y_px - com_y_px
    l_m = l_px / pixels_per_meter
    if np.any(l_m <= 0.0):
        raise ValueError(f"{trial}: xCoM requires positive CoM-to-ground height")

    # CoM 换算为米；竖直向上为正(取负号)。
    com_x_m = com_x_px / pixels_per_meter
    com_y_m_up = -com_y_px / pixels_per_meter

    # 逐帧位移(首帧补 0)，竖直分量取反(向上为正)。
    displacement_x_px = np.r_[0.0, np.diff(com_x_px)]
    displacement_y_px = np.r_[0.0, np.diff(com_y_px)]
    displacement_x_m = displacement_x_px / pixels_per_meter
    displacement_y_m_up = -displacement_y_px / pixels_per_meter
    displacement_m = np.sqrt(displacement_x_m ** 2 + displacement_y_m_up ** 2)

    # 速度 = 位移 × 帧率。
    velocity_x_m_s = displacement_x_m * VIDEO_FPS
    velocity_y_m_s_up = displacement_y_m_up * VIDEO_FPS
    velocity_m_s = np.sqrt(velocity_x_m_s ** 2 + velocity_y_m_s_up ** 2)

    # ω₀ = sqrt(g/l)；xCoM = CoM + 速度/ω₀；再把 xCoM 换算回像素(竖直取反)。
    omega0 = np.sqrt(GRAVITY_M_S2 / l_m)
    xcom_x_m = com_x_m + velocity_x_m_s / omega0
    xcom_y_m_up = com_y_m_up + velocity_y_m_s_up / omega0
    xcom_x_px = xcom_x_m * pixels_per_meter
    xcom_y_px = -xcom_y_m_up * pixels_per_meter

    # 组装逐帧结果(含原始/平滑 CoM 与预处理元信息)。
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
            "displacement_x_px": float(displacement_x_px[idx]),
            "displacement_y_px": float(displacement_y_px[idx]),
            "displacement_x_m": float(displacement_x_m[idx]),
            "displacement_y_m_up": float(displacement_y_m_up[idx]),
            "displacement_m": float(displacement_m[idx]),
            "velocity_x_m_s": float(velocity_x_m_s[idx]),
            "velocity_y_m_s_up": float(velocity_y_m_s_up[idx]),
            "velocity_m_s": float(velocity_m_s[idx]),
            "omega0_rad_s": float(omega0[idx]),
            "xcom_x_m": float(xcom_x_m[idx]),
            "xcom_y_m_up": float(xcom_y_m_up[idx]),
            "xcom_x_px": float(xcom_x_px[idx]),
            "xcom_y_px": float(xcom_y_px[idx]),
            "keypoint_preprocessing_method": entry["keypoint_preprocessing_method"],
            "keypoint_median_window": entry["keypoint_median_window"],
            "keypoint_savgol_window": entry["keypoint_savgol_window"],
            "keypoint_savgol_polyorder": entry["keypoint_savgol_polyorder"],
            "com_smoothing_method": entry["com_smoothing_method"],
            "com_smoothing_window": entry["com_smoothing_window"],
        })
    return rows


def write_metric_csv(rows: list[dict[str, Any]]) -> None:
    """把逐帧视频指标写入 temp 目录的 CSV(固定列顺序)。"""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "frame", "time_s",
        "pixels_per_meter", "scale_method", "reference_shoulder_width_m", "shoulder_width_px",
        "left_shoulder_x_px", "left_shoulder_y_px",
        "right_shoulder_x_px", "right_shoulder_y_px",
        "ground_y_px",
        "com_x_px", "com_y_px",
        "raw_com_x_px", "raw_com_y_px",
        "com_x_m", "com_y_m_up",
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
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """主流程：按试验读取 -> 逐试验计算指标 -> 写出 CSV。"""
    trials = load_video_trials(INPUT_JSON)
    rows: list[dict[str, Any]] = []
    # 按试验名排序后逐个计算并累加。
    for trial in sorted(trials):
        rows.extend(compute_video_com_metrics(trial, trials[trial]))

    write_metric_csv(rows)
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Trials: {len(trials)}")
    print(f"Rows: {len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
