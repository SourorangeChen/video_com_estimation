"""把预处理后的 keypoints 和 CoM 画回视频帧图片。

输出图用于目视检查 keypoints 平滑和 CoM 重算后的位置是否合理：
黄色线段为 COCO 骨架，紫色点为 keypoints，蓝色点为 CoM。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# 视频数据根目录、预处理后 JSON、输出根目录。
VIDEO_ROOT = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video")
PREPROCESSED_JSON = (
    VIDEO_ROOT
    / "video_keypoints-preprocessed"
    / "results"
    / "keypoints_and_com_preprocessed.json"
)
OUTPUT_ROOT = VIDEO_ROOT / "video_keypoints-preprocessed" / "results"

# 绘图配色(RGB)。
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 关键点圆点半径、CoM 圆点半径、骨架线宽、CoM 文字标签。
POINT_RADIUS = 7
COM_RADIUS = 10
LINE_WIDTH = 6
LABEL = "preprocessed"

# COCO-17 骨架连线(关键点索引对)。
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


def load_font(size: int = 24) -> ImageFont.ImageFont:
    """加载标签字体；找不到则回退到 PIL 默认字体。"""
    for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def valid_xy(point: Any) -> tuple[float, float] | None:
    """从单个关键点中安全地取出 (x, y)；异常或缺失返回 None。"""
    if not isinstance(point, list) or len(point) < 2:
        return None
    if point[0] is None or point[1] is None:
        return None
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None


def resolve_source_image(trial: str, frame: int) -> Path | None:
    """定位某试验某帧的源底图：优先用原始帧，其次用 PCT 标注帧；都没有返回 None。"""
    # 优先：原始抽帧图。
    raw_frame = (
        VIDEO_ROOT
        / "video_trial_pic"
        / f"Video_{trial}_pic"
        / "raw_frames"
        / f"frame_{frame:06d}.jpg"
    )
    if raw_frame.exists():
        return raw_frame

    # 次选：PCT 推理标注帧。
    pct_frame = (
        VIDEO_ROOT
        / "Video_keypoint-com"
        / "results"
        / f"Video_{trial}_pred"
        / "video_pct_pred"
        / f"frame_{frame:06d}_pct.jpg"
    )
    if pct_frame.exists():
        return pct_frame
    return None


def draw_circle(draw: ImageDraw.ImageDraw, xy: tuple[float, float], radius: int, fill: tuple[int, int, int]) -> None:
    """以 xy 为中心画一个实心圆。"""
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[float, float],
    font: ImageFont.ImageFont,
) -> None:
    """在 xy 旁绘制带白底蓝框的文字标签。"""
    x, y = xy
    text_x = int(x + 14)
    text_y = int(y - 18)
    # 先按文字外接框画白底蓝边的矩形，再写文字，保证可读性。
    bbox = draw.textbbox((text_x, text_y), text, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=WHITE,
        outline=BLUE,
    )
    draw.text((text_x, text_y), text, fill=BLUE, font=font)


def draw_record(record: dict[str, Any], source_image: Path, output_image: Path, font: ImageFont.ImageFont) -> None:
    """把单帧记录的骨架、关键点与 CoM 画到底图上并保存。"""
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    keypoints = record.get("keypoints")
    if isinstance(keypoints, list):
        # 先画骨架连线，再画关键点，避免点被线遮住。
        xy_by_index = {idx: valid_xy(point) for idx, point in enumerate(keypoints)}
        # 画骨架(仅当两端都有效)。
        for start, end in COCO_EDGES:
            start_xy = xy_by_index.get(start)
            end_xy = xy_by_index.get(end)
            if start_xy is not None and end_xy is not None:
                draw.line((start_xy[0], start_xy[1], end_xy[0], end_xy[1]), fill=YELLOW, width=LINE_WIDTH)
        # 画关键点(紫色)。
        for xy in xy_by_index.values():
            if xy is not None:
                draw_circle(draw, xy, POINT_RADIUS, MAGENTA)

    com = record.get("com")
    if isinstance(com, dict):
        try:
            com_xy = float(com["com_x"]), float(com["com_y"])
        except (KeyError, TypeError, ValueError):
            com_xy = None
        if com_xy is not None:
            # CoM 使用更醒目的蓝色圆点，便于和人体关键点区分。
            draw_circle(draw, com_xy, COM_RADIUS, BLUE)
            draw_label(draw, LABEL, com_xy, font)

    # 确保目录存在后保存(高质量 JPEG)。
    output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image, quality=95)


def main() -> int:
    """主流程：遍历预处理记录，逐帧把骨架/关键点/CoM 画回底图。"""
    records = json.loads(PREPROCESSED_JSON.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list JSON records in {PREPROCESSED_JSON}")

    font = load_font()
    written = 0          # 成功写出的图片数
    missing_source = 0   # 找不到源底图的帧数
    for record in records:
        trial = record.get("trial")
        frame = record.get("frame")
        if trial is None or frame is None:
            continue
        trial = str(trial)
        frame = int(frame)
        # 定位源底图，缺失则计数并跳过。
        source_image = resolve_source_image(trial, frame)
        if source_image is None:
            missing_source += 1
            continue
        # 输出到该试验的 video_keypoints_com_pred 子目录。
        output_image = (
            OUTPUT_ROOT
            / f"Video_{trial}_pred"
            / "video_keypoints_com_pred"
            / f"frame_{frame:06d}_pct.jpg"
        )
        draw_record(record, source_image, output_image, font)
        written += 1

    print(f"Input: {PREPROCESSED_JSON}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Images written: {written}")
    print(f"Missing source images: {missing_source}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
