"""对原始 COCO-17 keypoints 做预处理，并用预处理后的 keypoints 重算 CoM。

处理顺序是按 trial 分组，对每个关键点的 x/y 坐标分别做：
缺失值插值 -> median filter -> Savitzky-Golay filter。
最后保留原始 CoM 到 source_com，并把重算后的 CoM 写回 com。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


# 输入(原始关键点+CoM) 与 输出(预处理后) 的路径。
INPUT_JSON = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json"
)
OUTPUT_ROOT = Path(
    r"H:\COM\video-vicon\data\Chenzixuan\Video\video_keypoints-preprocessed"
)
OUTPUT_RESULTS_DIR = OUTPUT_ROOT / "results"
OUTPUT_JSON = OUTPUT_RESULTS_DIR / "keypoints_and_com_preprocessed.json"

# 中值滤波窗口、Savitzky-Golay 窗口与多项式阶数。
MEDIAN_WINDOW = 3
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2

# 7 段人体测量学模型(与 calculate_com.py 一致)：
# (段名, 近端锚点, 远端锚点, 质量占比, 近->远质心比例)。
SEGMENTS = [
    ("trunk_head_neck", (11, 12), (5, 6), 0.578, 0.660),
    ("left_total_arm", 5, 9, 0.050, 0.530),
    ("right_total_arm", 6, 10, 0.050, 0.530),
    ("left_foot_and_leg", 13, 15, 0.061, 0.606),
    ("right_foot_and_leg", 14, 16, 0.061, 0.606),
    ("left_thigh", 11, 13, 0.100, 0.433),
    ("right_thigh", 12, 14, 0.100, 0.433),
]


def parse_trial_and_frame(image_path: str) -> tuple[str, int] | None:
    """从图片路径用正则解析出 (试验名, 帧号)；不匹配返回 None。"""
    match = re.search(r"Video_(?P<trial>.+?)_Trajectory.*?frame_(?P<frame>\d+)", image_path)
    if not match:
        return None
    return match.group("trial"), int(match.group("frame"))


def get_keypoint_xy(keypoints: Any, index: int) -> tuple[float, float] | None:
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


def midpoint(keypoints: Any, indices: tuple[int, int]) -> tuple[float, float] | None:
    """计算两个关键点的中点；任一缺失返回 None。"""
    first = get_keypoint_xy(keypoints, indices[0])
    second = get_keypoint_xy(keypoints, indices[1])
    if first is None or second is None:
        return None
    return (0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1]))


def segment_anchor(keypoints: Any, anchor: int | tuple[int, int]) -> tuple[float, float] | None:
    """取段锚点：元组取中点，单索引取该关键点。"""
    if isinstance(anchor, tuple):
        return midpoint(keypoints, anchor)
    return get_keypoint_xy(keypoints, anchor)


def compute_frame_com(keypoints: Any) -> dict[str, float] | None:
    """用 7 段模型计算单帧质心(像素)；任一所需关键点缺失返回 None。"""
    weighted_x = 0.0
    weighted_y = 0.0
    total_weight = 0.0

    for _, proximal_anchor, distal_anchor, weight, ratio in SEGMENTS:
        # 取该段近端/远端锚点。
        proximal = segment_anchor(keypoints, proximal_anchor)
        distal = segment_anchor(keypoints, distal_anchor)
        if proximal is None or distal is None:
            return None

        # 段质心 = 近端 + ratio*(远端-近端)，按段质量加权累加。
        segment_x = proximal[0] + ratio * (distal[0] - proximal[0])
        segment_y = proximal[1] + ratio * (distal[1] - proximal[1])
        weighted_x += weight * segment_x
        weighted_y += weight * segment_y
        total_weight += weight

    if total_weight == 0.0:
        return None
    # 加权平均得全身质心，保留 4 位小数。
    return {
        "com_x": round(weighted_x / total_weight, 4),
        "com_y": round(weighted_y / total_weight, 4),
    }


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    """对一维序列中的缺失值(非有限)做线性插值，端点用最近有效值延拓。"""
    result = values.astype(float).copy()
    valid = np.isfinite(result)
    # 全部有效：无需插值。
    if valid.all():
        return result
    # 全部无效：无法插值。
    if not valid.any():
        return result
    # 只有一个有效值：用它填满整段。
    if valid.sum() == 1:
        result[~valid] = result[valid][0]
        return result

    # 一般情况：用有效点对缺失点做线性插值(np.interp 端点会自动平延)。
    indices = np.arange(len(result), dtype=float)
    result[~valid] = np.interp(indices[~valid], indices[valid], result[valid])
    return result


def odd_window_for_length(preferred: int, length: int, min_window: int = 3) -> int | None:
    """根据序列长度返回一个不超过它的奇数窗口；过短则返回 None。"""
    # 序列比最小窗口还短，无法滤波。
    if length < min_window:
        return None
    # 取 preferred 与序列长度(取奇数)中的较小者。
    window = min(preferred, length if length % 2 == 1 else length - 1)
    if window < min_window:
        return None
    return window


def preprocess_signal(values: np.ndarray) -> np.ndarray:
    """处理单个关键点的单个坐标序列，例如 left_ankle_x 的全 trial 时序。"""
    # 第一步：插值补齐缺失值。
    signal = interpolate_missing(values)
    # 仍含非有限值(如全缺失)则直接返回，不再滤波。
    if not np.isfinite(signal).all():
        return signal

    # 第二步：中值滤波去除尖刺(窗口随序列长度自适应)。
    median_window = odd_window_for_length(MEDIAN_WINDOW, len(signal))
    if median_window is not None:
        signal = median_filter(signal, size=median_window, mode="nearest")

    # 第三步：Savitzky-Golay 多项式平滑(需窗口 > 多项式阶数)。
    sg_window = odd_window_for_length(SAVGOL_WINDOW, len(signal), min_window=SAVGOL_POLYORDER + 2)
    if sg_window is not None and sg_window > SAVGOL_POLYORDER:
        signal = savgol_filter(signal, window_length=sg_window, polyorder=SAVGOL_POLYORDER, mode="interp")
    return signal


def extract_xy_array(records: list[dict[str, Any]]) -> np.ndarray:
    """把若干帧记录的关键点抽取为形状 (帧数, 17, 2) 的数组，缺失填 NaN。"""
    xy = np.full((len(records), 17, 2), np.nan, dtype=float)
    for frame_index, record in enumerate(records):
        keypoints = record.get("keypoints")
        if not isinstance(keypoints, list):
            continue
        # 逐关键点填入 x、y(最多 17 个)。
        for keypoint_index in range(min(17, len(keypoints))):
            point = keypoints[keypoint_index]
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                xy[frame_index, keypoint_index, 0] = float(point[0])
                xy[frame_index, keypoint_index, 1] = float(point[1])
            except (TypeError, ValueError):
                continue
    return xy


def preprocess_trial_keypoints(records: list[dict[str, Any]]) -> np.ndarray:
    """对一个试验内全部帧的关键点逐点逐轴做时序预处理，返回同形状数组。"""
    xy = extract_xy_array(records)
    processed = xy.copy()
    # 对 17 个关键点 × 2 个坐标轴各自做时序滤波。
    for keypoint_index in range(17):
        for axis in range(2):
            processed[:, keypoint_index, axis] = preprocess_signal(xy[:, keypoint_index, axis])
    return processed


def update_keypoints(record: dict[str, Any], processed_xy: np.ndarray) -> list[Any]:
    """把某帧的关键点 x/y 替换为预处理值(置信度等其它字段保持不变)。"""
    source_keypoints = record.get("keypoints")
    if not isinstance(source_keypoints, list):
        return []

    # 深拷贝原关键点，仅覆盖 x、y 两个分量。
    keypoints = deepcopy(source_keypoints)
    for keypoint_index in range(min(17, len(keypoints))):
        point = keypoints[keypoint_index]
        if not isinstance(point, list) or len(point) < 2:
            continue
        x = processed_xy[keypoint_index, 0]
        y = processed_xy[keypoint_index, 1]
        # 仅当预处理值有效时才覆盖。
        if np.isfinite(x) and np.isfinite(y):
            point[0] = float(x)
            point[1] = float(y)
    return keypoints


def build_preprocessed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成新的 JSON 记录：keypoints 替换为平滑值，CoM 基于平滑 keypoints 重算。"""
    # indexed_records 记录原始顺序；grouped 按试验分组以便逐试验处理。
    indexed_records: list[tuple[int, str, int, dict[str, Any]]] = []
    grouped: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for source_index, record in enumerate(records):
        parsed = parse_trial_and_frame(str(record.get("image", "")))
        if parsed is None:
            continue
        trial, frame = parsed
        indexed_records.append((source_index, trial, frame, record))
        grouped[trial].append((source_index, frame, record))

    # 以原始索引为键收集输出记录，便于最后还原原顺序。
    output_by_source_index: dict[int, dict[str, Any]] = {}
    for trial in sorted(grouped):
        # 试验内按帧号排序后做时序预处理。
        trial_items = sorted(grouped[trial], key=lambda item: item[1])
        trial_records = [item[2] for item in trial_items]
        processed_xy = preprocess_trial_keypoints(trial_records)

        # 为该试验预建输出图片目录。
        (OUTPUT_RESULTS_DIR / f"Video_{trial}_pred").mkdir(parents=True, exist_ok=True)
        for row_index, (source_index, frame, source_record) in enumerate(trial_items):
            # 在源记录副本上写入试验名、帧号、原始 CoM、平滑关键点。
            output_record = deepcopy(source_record)
            output_record["trial"] = trial
            output_record["frame"] = frame
            output_record["source_com"] = deepcopy(source_record.get("com"))
            output_record["keypoints"] = update_keypoints(source_record, processed_xy[row_index])
            # CoM 必须基于平滑后的 keypoints 重算，不能沿用原始 CoM。
            output_record["com"] = compute_frame_com(output_record["keypoints"])
            # 记录预处理参数，便于溯源。
            output_record["keypoint_preprocessing"] = {
                "method": "median_filter_then_savitzky_golay",
                "median_window": MEDIAN_WINDOW,
                "savgol_window": SAVGOL_WINDOW,
                "savgol_polyorder": SAVGOL_POLYORDER,
                "grouping": "trial",
                "coordinates": ["x", "y"],
                "score_policy": "preserved_from_source",
                "missing_value_policy": "linear_interpolation_with_endpoint_extension",
            }
            output_by_source_index[source_index] = output_record

    # 按原始记录顺序输出。
    return [output_by_source_index[index] for index, *_ in indexed_records]


def main() -> int:
    """主流程：读取原始 JSON -> 预处理并重算 CoM -> 写出预处理后 JSON。"""
    records = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list JSON records in {INPUT_JSON}")

    preprocessed = build_preprocessed_records(records)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(preprocessed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 打印汇总信息。
    trials = sorted({record["trial"] for record in preprocessed if "trial" in record})
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_JSON}")
    print(f"Records: {len(preprocessed)}")
    print(f"Trials: {', '.join(trials)}")
    print(f"Median window: {MEDIAN_WINDOW}")
    print(f"Savitzky-Golay window/polyorder: {SAVGOL_WINDOW}/{SAVGOL_POLYORDER}")
    return 0 if preprocessed else 1


if __name__ == "__main__":
    raise SystemExit(main())
