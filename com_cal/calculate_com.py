from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None


CAMERA_DATA_ROOT = Path(r"H:\Camera_data")
EXCEL_PATH = CAMERA_DATA_ROOT / "视频标注box.xlsx"


SEGMENTS = [
    ("trunk_head_neck", (11, 12), (5, 6), 0.578, 0.660),
    ("left_total_arm", 5, 9, 0.050, 0.530),
    ("right_total_arm", 6, 10, 0.050, 0.530),
    ("left_foot_and_leg", 13, 15, 0.061, 0.606),
    ("right_foot_and_leg", 14, 16, 0.061, 0.606),
    ("left_thigh", 11, 13, 0.100, 0.433),
    ("right_thigh", 12, 14, 0.100, 0.433),
]


@dataclass(frozen=True)
class SegmentMeta:
    session_name: str
    video_name: str
    video_stem: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class SegmentSource:
    json_path: Path
    segment_dir: Path


def normalize_video_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).lower()


def parse_int(value: Any) -> int:
    if value is None:
        raise ValueError("Expected integer-like value, got None")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text:
        raise ValueError("Expected integer-like value, got empty string")
    return int(float(text))


def print_excel_preview(sheet_name: str, rows: list[dict[str, Any]]) -> None:
    print(f"[Excel预览] sheet={sheet_name}")
    if not rows:
        print("列名: []")
        print("前3行: []")
        return

    columns = list(rows[0].keys())
    print(f"列名: {columns}")
    print("前3行:")
    for row in rows[:3]:
        print(json.dumps(row, ensure_ascii=False))


def load_segment_metadata(patient_name: str) -> list[SegmentMeta]:
    workbook = load_workbook(EXCEL_PATH, read_only=True, data_only=True)
    try:
        if patient_name not in workbook.sheetnames:
            raise ValueError(f"Excel中不存在sheet: {patient_name}")

        sheet = workbook[patient_name]
        rows_iter = sheet.iter_rows(values_only=True)
        header_row = next(rows_iter)
        if header_row is None:
            raise ValueError(f"sheet为空: {patient_name}")

        headers = [str(cell).strip() if cell is not None else "" for cell in header_row]
        data_rows: list[dict[str, Any]] = []
        for values in rows_iter:
            row = {headers[idx]: values[idx] for idx in range(len(headers))}
            if all(value is None for value in row.values()):
                continue
            data_rows.append(row)

        print_excel_preview(patient_name, data_rows)

        required_columns = ["文件夹名称", "视频文件名", "开始帧号", "结束帧号"]
        missing_columns = [column for column in required_columns if column not in headers]
        if missing_columns:
            raise ValueError(f"Excel缺少必要列: {missing_columns}")

        metadata: list[SegmentMeta] = []
        for row in data_rows:
            session_name = str(row["文件夹名称"]).strip()
            video_name = str(row["视频文件名"]).strip()
            if not session_name or not video_name or video_name.lower() == "none":
                continue

            metadata.append(
                SegmentMeta(
                    session_name=session_name,
                    video_name=video_name,
                    video_stem=Path(video_name).stem,
                    start_frame=parse_int(row["开始帧号"]),
                    end_frame=parse_int(row["结束帧号"]),
                )
            )

        return metadata
    finally:
        workbook.close()


def extract_session_segment_info(segment_dir: Path) -> tuple[str, int, int]:
    base_name = segment_dir.name
    if base_name.endswith("_skeleton_msk"):
        base_name = base_name[: -len("_skeleton_msk")]

    match = re.match(r"^(?P<video>.+)_(?P<start>\d+)_(?P<end>\d+)$", base_name)
    if not match:
        raise ValueError(f"无法从目录名解析视频与帧号: {segment_dir}")

    return (
        normalize_video_name(match.group("video")),
        int(match.group("start")),
        int(match.group("end")),
    )


def build_segment_json_index(patient_dir: Path) -> dict[tuple[str, str, int, int], SegmentSource]:
    index: dict[tuple[str, str, int, int], SegmentSource] = {}
    json_paths = sorted(patient_dir.glob("*/**/all_frames_ankle_data.json"))
    for json_path in json_paths:
        session_name = json_path.parent.parent.name
        normalized_video, start_frame, end_frame = extract_session_segment_info(json_path.parent)
        index[(session_name, normalized_video, start_frame, end_frame)] = SegmentSource(
            json_path=json_path,
            segment_dir=json_path.parent,
        )
    return index


def print_json_preview(first_json_path: Path) -> None:
    data = json.loads(first_json_path.read_text(encoding="utf-8"))
    preview = data[:2] if isinstance(data, list) else data
    print(f"[JSON预览] {first_json_path}")
    print(json.dumps(preview, ensure_ascii=False, indent=2)[:4000])


def get_keypoint_xy(keypoints: Any, index: int) -> tuple[float, float] | None:
    if not isinstance(keypoints, list) or index >= len(keypoints):
        return None

    point = keypoints[index]
    if not isinstance(point, list) or len(point) < 2:
        return None

    x = point[0]
    y = point[1]
    if x is None or y is None:
        return None

    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def midpoint(keypoints: Any, indices: tuple[int, int]) -> tuple[float, float] | None:
    first = get_keypoint_xy(keypoints, indices[0])
    second = get_keypoint_xy(keypoints, indices[1])
    if first is None or second is None:
        return None
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def segment_anchor(keypoints: Any, anchor: int | tuple[int, int]) -> tuple[float, float] | None:
    if isinstance(anchor, tuple):
        return midpoint(keypoints, anchor)
    return get_keypoint_xy(keypoints, anchor)


def compute_frame_com(keypoints: Any) -> dict[str, float] | None:
    weighted_x = 0.0
    weighted_y = 0.0
    total_weight = 0.0

    for _, proximal_anchor, distal_anchor, weight, ratio in SEGMENTS:
        proximal = segment_anchor(keypoints, proximal_anchor)
        distal = segment_anchor(keypoints, distal_anchor)
        if proximal is None or distal is None:
            return None

        segment_x = proximal[0] + ratio * (distal[0] - proximal[0])
        segment_y = proximal[1] + ratio * (distal[1] - proximal[1])

        weighted_x += weight * segment_x
        weighted_y += weight * segment_y
        total_weight += weight

    if total_weight == 0:
        return None

    return {
        "com_x": round(weighted_x / total_weight, 4),
        "com_y": round(weighted_y / total_weight, 4),
    }


def compute_segment_output(meta: SegmentMeta, json_path: Path) -> dict[str, Any]:
    frame_entries = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(frame_entries, list):
        raise ValueError(f"JSON根结构不是list: {json_path}")

    frames: dict[str, dict[str, float] | None] = {}
    for entry in frame_entries:
        if not isinstance(entry, dict):
            continue

        frame_number = entry.get("frame")
        if frame_number is None:
            continue

        try:
            frame_key = str(parse_int(frame_number))
        except ValueError:
            continue

        frames[frame_key] = compute_frame_com(entry.get("keypoints"))

    return {
        "video": meta.video_stem,
        "start_frame": meta.start_frame,
        "end_frame": meta.end_frame,
        "frames": frames,
    }


def draw_com_overlays(segment_dir: Path, frames: dict[str, dict[str, float] | None]) -> None:
    if cv2 is None:
        raise RuntimeError("缺少cv2，请先安装opencv-python后再运行绘图功能")

    output_dir = segment_dir / "com_overlay_frames"
    output_dir.mkdir(exist_ok=True)

    for frame_key, com in frames.items():
        source_image = segment_dir / f"frame_{int(frame_key):06d}.jpg"
        if not source_image.exists():
            continue

        image = cv2.imread(str(source_image))
        if image is None:
            continue

        if com is not None:
            center = (int(round(com["com_x"])), int(round(com["com_y"])))
            cv2.circle(image, center, 6, (0, 0, 255), -1)
            cv2.circle(image, center, 12, (255, 255, 255), 2)
            cv2.putText(
                image,
                f"COM ({com['com_x']:.1f}, {com['com_y']:.1f})",
                (center[0] + 10, max(center[1] - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        output_image = output_dir / source_image.name
        cv2.imwrite(str(output_image), image)


def main() -> int:
    if len(sys.argv) != 2:
        print("用法: python calculate_com.py 3_MsSu")
        return 1

    patient_name = sys.argv[1]
    patient_dir = CAMERA_DATA_ROOT / patient_name
    if not patient_dir.exists():
        print(f"患者目录不存在: {patient_dir}")
        return 1

    metadata = load_segment_metadata(patient_name)
    segment_index = build_segment_json_index(patient_dir)
    if segment_index:
        first_segment_source = next(iter(segment_index.values()))
        print_json_preview(first_segment_source.json_path)

    missing_segments: list[str] = []
    for meta in metadata:
        normalized_video = normalize_video_name(meta.video_stem)
        segment_source = segment_index.get((meta.session_name, normalized_video, meta.start_frame, meta.end_frame))
        if segment_source is None:
            missing_segments.append(
                f"{patient_name}/{meta.session_name}/{meta.video_stem}_{meta.start_frame}_{meta.end_frame}"
            )
            continue

        output = compute_segment_output(meta, segment_source.json_path)
        output_name = f"{meta.video_stem}_{meta.start_frame}_{meta.end_frame}_COM.json"
        output_path = patient_dir / meta.session_name / output_name
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        draw_com_overlays(segment_source.segment_dir, output["frames"])
        print(f"[完成] {patient_name} / {meta.session_name} / {meta.video_stem}_{meta.start_frame}_{meta.end_frame} → {output_name}")

    if missing_segments:
        print("[警告] 以下segment未找到对应的all_frames_ankle_data.json:")
        for item in missing_segments:
            print(f"  - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
