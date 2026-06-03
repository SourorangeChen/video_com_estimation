"""从视频关键点 JSON 计算每帧的视频侧运动学指标(速度、xCoM、摆长 l、自然频率 ω₀)。

由于视频是 2D 像素坐标，需要先用"鼻-踝像素身高 / 参考身高(1.70m)"换算出
每帧的"像素/米"比例(pixels_per_meter)，再把像素量换算为物理量(米、米/秒)。

关键符号约定(像素 Y 向下为正)：
    - com_y_m_up = −com_y_px / pixels_per_meter  (向上为正)
    - 竖直方向的位移/速度分量相应取反号

输出两份 CSV：原始 CoM 版与对 CoM 做滑动平均平滑后的版本。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# 输入关键点 JSON 与输出路径。
KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
OUTPUT_CSV = VALIDATION_DIR / "video_com_metric.csv"
SMOOTHED_OUTPUT_CSV = VALIDATION_DIR / "video_com_metric_smoothed.csv"
VIDEO_FPS = 29.996            # 视频帧率(Hz)
REFERENCE_HEIGHT_M = 1.70     # 受试者参考身高(米)，用于像素->米标定
GRAVITY_M_S2 = 9.81           # 重力加速度(m/s²)
SMOOTHING_WINDOW = 5          # CoM 平滑的滑动窗口长度(必须为正奇数)

# 用到的 COCO 关键点索引。
NOSE = 0
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def smooth_signal(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """对一维信号做简单滑动平均(中心对齐)，两端保留原值。"""
    # 窗口必须是正奇数(保证中心对齐)。
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    # 数据点比窗口还少时不平滑，直接返回副本。
    if len(values) < window:
        return values.astype(float).copy()

    values = values.astype(float)
    half_window = window // 2
    smoothed = values.copy()
    # 均匀核(每个权重 1/window)。
    kernel = np.ones(window) / window
    # 仅对中间部分卷积，两端(各 half_window 个点)保留原值。
    smoothed[half_window:-half_window] = np.convolve(values, kernel, mode="valid")
    return smoothed


def parse_trial_and_frame(image_path: str) -> tuple[str, int] | None:
    """从图片相对路径解析出 (试验名, 帧号)。

    路径首段应形如 "Video_<trial>_Trajectory"，文件名末尾为帧号。
    不符合格式则返回 None。
    """
    path = Path(image_path)
    folder = path.parts[0] if path.parts else ""
    # 校验目录命名格式。
    if not folder.startswith("Video_") or not folder.endswith("_Trajectory"):
        return None
    # 去掉前后缀得到试验名。
    trial = folder.removeprefix("Video_").removesuffix("_Trajectory")
    try:
        # 文件名(如 frame_000123)末段为帧号。
        frame = int(path.stem.split("_")[-1])
    except ValueError:
        return None
    return trial, frame


def keypoint_y(keypoints: Any, index: int) -> float | None:
    """安全地取出第 index 个关键点的 y 坐标；结构异常或缺失返回 None。"""
    if not isinstance(keypoints, list) or index >= len(keypoints):
        return None
    point = keypoints[index]
    if not isinstance(point, list) or len(point) < 2:
        return None
    if point[1] is None:
        return None
    try:
        return float(point[1])
    except (TypeError, ValueError):
        return None


def compute_video_com_metrics(
    trial: str,
    frame_numbers: list[int],
    com_x_px: np.ndarray,
    com_y_px: np.ndarray,
    nose_y_px: np.ndarray,
    left_ankle_y_px: np.ndarray,
    right_ankle_y_px: np.ndarray,
    fps: float = VIDEO_FPS,
    reference_height_m: float = REFERENCE_HEIGHT_M,
) -> list[dict[str, Any]]:
    """对单个试验的逐帧 CoM 计算完整运动学指标。

    步骤：像素->米标定 -> 摆长 l -> 位移/速度 -> ω₀ -> xCoM，最终组装为逐帧字典。
    """
    # 所有输入数组长度必须一致。
    if not (
        len(frame_numbers)
        == len(com_x_px)
        == len(com_y_px)
        == len(nose_y_px)
        == len(left_ankle_y_px)
        == len(right_ankle_y_px)
    ):
        raise ValueError("frame, CoM, and keypoint arrays must have the same length")
    # 至少两帧才能算速度。
    if len(frame_numbers) < 2:
        raise ValueError("at least two frames are required to compute velocity")

    # 时间轴：以首帧为 t=0(按帧号差 / fps)。
    time_s = np.array([(frame - frame_numbers[0]) / fps for frame in frame_numbers], dtype=float)
    # 地面 y 取左右踝中更靠下(像素值更大)者。
    ground_y_px = np.maximum(left_ankle_y_px, right_ankle_y_px)
    # 鼻到踝的像素高度(代表身高的像素跨度)。
    nose_to_ankle_height_px = ground_y_px - nose_y_px
    if np.any(nose_to_ankle_height_px <= 0.0):
        raise ValueError("video scaling requires positive nose-to-ankle height for every frame")

    # 每帧的像素/米比例 = 鼻踝像素高度 / 参考身高。
    pixels_per_meter = nose_to_ankle_height_px / reference_height_m
    # 摆长 l：CoM 到地面的像素高度，再换算为米。
    l_px = ground_y_px - com_y_px
    l_m = l_px / pixels_per_meter
    if np.any(l_m <= 0.0):
        raise ValueError("xCoM requires positive CoM-to-ground height for every frame")

    # CoM 像素坐标换算为米；竖直方向取"向上为正"(故取负号)。
    com_x_m = com_x_px / pixels_per_meter
    com_y_m_up = -com_y_px / pixels_per_meter
    com_m = np.column_stack([com_x_m, com_y_m_up])

    # 逐帧像素位移(首帧补 0)。
    displacement_px = np.column_stack([
        np.r_[0.0, np.diff(com_x_px)],
        np.r_[0.0, np.diff(com_y_px)],
    ])
    # 位移换算为米；竖直分量取反(向上为正)。
    displacement_m = np.column_stack([
        displacement_px[:, 0] / pixels_per_meter,
        -displacement_px[:, 1] / pixels_per_meter,
    ])
    displacement_mag_m = np.linalg.norm(displacement_m, axis=1)

    # 速度 = 逐帧位移 × 帧率。
    velocity_m_s = displacement_m * fps
    velocity_mag_m_s = np.linalg.norm(velocity_m_s, axis=1)
    # 倒立摆自然频率 ω₀ = sqrt(g / l)。
    omega0 = np.sqrt(GRAVITY_M_S2 / l_m)
    # 外推质心 xCoM = CoM + 速度 / ω₀。
    xcom_m = com_m + velocity_m_s / omega0[:, np.newaxis]
    # 把 xCoM 换算回像素坐标(竖直分量再次取反，回到像素向下为正)。
    xcom_x_px = xcom_m[:, 0] * pixels_per_meter
    xcom_y_px = -xcom_m[:, 1] * pixels_per_meter

    # 组装逐帧结果字典。
    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(frame_numbers):
        rows.append({
            "trial": trial,
            "frame": int(frame),
            "time_s": float(time_s[idx]),
            "pixels_per_meter": float(pixels_per_meter[idx]),
            "reference_height_m": float(reference_height_m),
            "nose_to_ankle_height_px": float(nose_to_ankle_height_px[idx]),
            "nose_y_px": float(nose_y_px[idx]),
            "ground_y_px": float(ground_y_px[idx]),
            "com_x_px": float(com_x_px[idx]),
            "com_y_px": float(com_y_px[idx]),
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
        })
    return rows


def load_video_trials(keypoints_json: Path) -> dict[str, list[dict[str, Any]]]:
    """读取关键点 JSON，按试验聚合每帧所需的 CoM 与关键点 y 值。"""
    records = json.loads(keypoints_json.read_text(encoding="utf-8"))
    trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in records:
        if not isinstance(entry, dict):
            continue
        # 解析试验名与帧号。
        parsed = parse_trial_and_frame(entry.get("image", ""))
        if parsed is None:
            continue
        trial, frame = parsed
        com = entry.get("com")
        keypoints = entry.get("keypoints")
        # 必须有 CoM 字典。
        if not isinstance(com, dict):
            continue
        # 取鼻、左右踝的 y(用于标定与地面线)。
        nose_y = keypoint_y(keypoints, NOSE)
        left_ankle_y = keypoint_y(keypoints, LEFT_ANKLE)
        right_ankle_y = keypoint_y(keypoints, RIGHT_ANKLE)
        # 任一关键缺失则跳过该帧。
        if nose_y is None or left_ankle_y is None or right_ankle_y is None:
            continue
        trials[trial].append({
            "frame": frame,
            "com_x_px": float(com["com_x"]),
            "com_y_px": float(com["com_y"]),
            "nose_y_px": nose_y,
            "left_ankle_y_px": left_ankle_y,
            "right_ankle_y_px": right_ankle_y,
        })
    return dict(trials)


def compute_all_video_metrics(keypoints_json: Path, smooth_com: bool = False) -> list[dict[str, Any]]:
    """对所有试验计算运动学指标；smooth_com=True 时先对 CoM 序列做平滑。"""
    all_rows: list[dict[str, Any]] = []
    trials = load_video_trials(keypoints_json)
    # 按试验名排序，保证输出顺序稳定。
    for trial in sorted(trials):
        # 同一试验内按帧号排序。
        entries = sorted(trials[trial], key=lambda item: item["frame"])
        # 不足两帧无法算速度，跳过。
        if len(entries) < 2:
            continue
        com_x_px = np.array([entry["com_x_px"] for entry in entries])
        com_y_px = np.array([entry["com_y_px"] for entry in entries])
        # 可选：对 CoM 做滑动平均以抑制抖动。
        if smooth_com:
            com_x_px = smooth_signal(com_x_px)
            com_y_px = smooth_signal(com_y_px)

        # 计算该试验逐帧指标。
        rows = compute_video_com_metrics(
            trial=trial,
            frame_numbers=[entry["frame"] for entry in entries],
            com_x_px=com_x_px,
            com_y_px=com_y_px,
            nose_y_px=np.array([entry["nose_y_px"] for entry in entries]),
            left_ankle_y_px=np.array([entry["left_ankle_y_px"] for entry in entries]),
            right_ankle_y_px=np.array([entry["right_ankle_y_px"] for entry in entries]),
        )
        # 平滑版额外记录所用窗口长度。
        if smooth_com:
            for row in rows:
                row["com_smoothing_window"] = SMOOTHING_WINDOW
        all_rows.extend(rows)
    return all_rows


def write_video_metric_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """把逐帧视频运动学指标写入 CSV。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # 固定列顺序。
    fieldnames = [
        "trial", "frame", "time_s",
        "pixels_per_meter", "reference_height_m", "nose_to_ankle_height_px",
        "nose_y_px", "ground_y_px",
        "com_x_px", "com_y_px", "com_x_m", "com_y_m_up",
        "l_px", "l_m",
        "displacement_x_px", "displacement_y_px",
        "displacement_x_m", "displacement_y_m_up", "displacement_m",
        "velocity_x_m_s", "velocity_y_m_s_up", "velocity_m_s",
        "omega0_rad_s",
        "xcom_x_m", "xcom_y_m_up", "xcom_x_px", "xcom_y_px",
    ]
    # 若为平滑版(含额外列)，把该列追加到表头。
    if rows and "com_smoothing_window" in rows[0]:
        fieldnames.append("com_smoothing_window")
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """主流程：分别计算原始与平滑两版指标并各写一份 CSV。"""
    rows = compute_all_video_metrics(KEYPOINTS_JSON)
    smoothed_rows = compute_all_video_metrics(KEYPOINTS_JSON, smooth_com=True)
    write_video_metric_csv(rows, OUTPUT_CSV)
    write_video_metric_csv(smoothed_rows, SMOOTHED_OUTPUT_CSV)
    print(f"Output: {OUTPUT_CSV}")
    print(f"Smoothed output: {SMOOTHED_OUTPUT_CSV}")
    print(f"Rows: {len(rows)}")
    print(f"Smoothed rows: {len(smoothed_rows)}")
    return 0 if rows and smoothed_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
