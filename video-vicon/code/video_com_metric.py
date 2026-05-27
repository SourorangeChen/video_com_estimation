from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
OUTPUT_CSV = VALIDATION_DIR / "video_com_metric.csv"
VIDEO_FPS = 29.996
REFERENCE_HEIGHT_M = 1.70
GRAVITY_M_S2 = 9.81

NOSE = 0
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def parse_trial_and_frame(image_path: str) -> tuple[str, int] | None:
    path = Path(image_path)
    folder = path.parts[0] if path.parts else ""
    if not folder.startswith("Video_") or not folder.endswith("_Trajectory"):
        return None
    trial = folder.removeprefix("Video_").removesuffix("_Trajectory")
    try:
        frame = int(path.stem.split("_")[-1])
    except ValueError:
        return None
    return trial, frame


def keypoint_y(keypoints: Any, index: int) -> float | None:
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
    if not (
        len(frame_numbers)
        == len(com_x_px)
        == len(com_y_px)
        == len(nose_y_px)
        == len(left_ankle_y_px)
        == len(right_ankle_y_px)
    ):
        raise ValueError("frame, CoM, and keypoint arrays must have the same length")
    if len(frame_numbers) < 2:
        raise ValueError("at least two frames are required to compute velocity")

    time_s = np.array([(frame - frame_numbers[0]) / fps for frame in frame_numbers], dtype=float)
    ground_y_px = np.maximum(left_ankle_y_px, right_ankle_y_px)
    nose_to_ankle_height_px = ground_y_px - nose_y_px
    if np.any(nose_to_ankle_height_px <= 0.0):
        raise ValueError("video scaling requires positive nose-to-ankle height for every frame")

    pixels_per_meter = nose_to_ankle_height_px / reference_height_m
    l_px = ground_y_px - com_y_px
    l_m = l_px / pixels_per_meter
    if np.any(l_m <= 0.0):
        raise ValueError("xCoM requires positive CoM-to-ground height for every frame")

    com_x_m = com_x_px / pixels_per_meter
    com_y_m_up = -com_y_px / pixels_per_meter
    com_m = np.column_stack([com_x_m, com_y_m_up])

    displacement_px = np.column_stack([
        np.r_[0.0, np.diff(com_x_px)],
        np.r_[0.0, np.diff(com_y_px)],
    ])
    displacement_m = np.column_stack([
        displacement_px[:, 0] / pixels_per_meter,
        -displacement_px[:, 1] / pixels_per_meter,
    ])
    displacement_mag_m = np.linalg.norm(displacement_m, axis=1)

    velocity_m_s = np.gradient(com_m, time_s, axis=0)
    velocity_mag_m_s = np.linalg.norm(velocity_m_s, axis=1)
    omega0 = np.sqrt(GRAVITY_M_S2 / l_m)
    xcom_m = com_m + velocity_m_s / omega0[:, np.newaxis]
    xcom_x_px = xcom_m[:, 0] * pixels_per_meter
    xcom_y_px = -xcom_m[:, 1] * pixels_per_meter

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
    records = json.loads(keypoints_json.read_text(encoding="utf-8"))
    trials: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in records:
        if not isinstance(entry, dict):
            continue
        parsed = parse_trial_and_frame(entry.get("image", ""))
        if parsed is None:
            continue
        trial, frame = parsed
        com = entry.get("com")
        keypoints = entry.get("keypoints")
        if not isinstance(com, dict):
            continue
        nose_y = keypoint_y(keypoints, NOSE)
        left_ankle_y = keypoint_y(keypoints, LEFT_ANKLE)
        right_ankle_y = keypoint_y(keypoints, RIGHT_ANKLE)
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


def compute_all_video_metrics(keypoints_json: Path) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    trials = load_video_trials(keypoints_json)
    for trial in sorted(trials):
        entries = sorted(trials[trial], key=lambda item: item["frame"])
        if len(entries) < 2:
            continue
        all_rows.extend(
            compute_video_com_metrics(
                trial=trial,
                frame_numbers=[entry["frame"] for entry in entries],
                com_x_px=np.array([entry["com_x_px"] for entry in entries]),
                com_y_px=np.array([entry["com_y_px"] for entry in entries]),
                nose_y_px=np.array([entry["nose_y_px"] for entry in entries]),
                left_ankle_y_px=np.array([entry["left_ankle_y_px"] for entry in entries]),
                right_ankle_y_px=np.array([entry["right_ankle_y_px"] for entry in entries]),
            )
        )
    return all_rows


def write_video_metric_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = compute_all_video_metrics(KEYPOINTS_JSON)
    write_video_metric_csv(rows, OUTPUT_CSV)
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows: {len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
