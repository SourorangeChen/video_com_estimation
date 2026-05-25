from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parent
PCT_RESULTS_ROOT = WORKSPACE_ROOT / "Video" / "Video_Keypoint" / "pct_results"
KEYPOINTS_JSON = PCT_RESULTS_ROOT / "keypoints.json"
CALCULATE_COM_DIR = Path(r"H:\COM")

BLUE = (0, 0, 255)


def load_com_function():
    sys.path.insert(0, str(CALCULATE_COM_DIR))
    from calculate_com import compute_frame_com

    return compute_frame_com


def load_font(size: int = 18) -> ImageFont.ImageFont:
    for font_path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def resolve_source_image(relative_image: str) -> Path | None:
    listed_path = PCT_RESULTS_ROOT / relative_image
    if listed_path.exists():
        return listed_path

    suffix_path = listed_path.with_name(f"{listed_path.stem}_pct{listed_path.suffix}")
    if suffix_path.exists():
        return suffix_path

    matches = sorted(listed_path.parent.glob(f"{listed_path.stem}*{listed_path.suffix}"))
    return matches[0] if matches else None


def output_path_for(source_image: Path) -> Path:
    video_dir = source_image.parent.parent
    return video_dir / "com_estimated_frames" / source_image.name


def label_position(center_x: int, center_y: int, image_width: int, image_height: int) -> tuple[int, int]:
    x = min(center_x + 10, max(image_width - 95, 0))
    y = center_y - 22
    if y < 5:
        y = min(center_y + 10, max(image_height - 24, 0))
    return max(x, 0), max(y, 0)


def draw_estimated_com(source_image: Path, output_image: Path, com: dict[str, float], font: ImageFont.ImageFont) -> None:
    with Image.open(source_image) as original:
        image = original.convert("RGB")

    draw = ImageDraw.Draw(image)
    center = (int(round(com["com_x"])), int(round(com["com_y"])))
    radius = 6
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=BLUE,
    )
    draw.text(label_position(center[0], center[1], image.width, image.height), "estimated", fill=BLUE, font=font)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image, quality=95)


def main() -> int:
    if not KEYPOINTS_JSON.exists():
        print(f"keypoints.json not found: {KEYPOINTS_JSON}")
        return 1

    compute_frame_com = load_com_function()
    font = load_font()
    entries = json.loads(KEYPOINTS_JSON.read_text(encoding="utf-8"))

    stats: Counter[str] = Counter()
    per_video: Counter[str] = Counter()

    for entry in entries:
        if not isinstance(entry, dict):
            stats["bad_entry"] += 1
            continue

        source_image = resolve_source_image(str(entry.get("image", "")))
        if source_image is None:
            stats["missing_image"] += 1
            continue

        com = compute_frame_com(entry.get("keypoints"))
        if com is None:
            stats["missing_com"] += 1
            continue

        output_image = output_path_for(source_image)
        draw_estimated_com(source_image, output_image, com, font)

        stats["annotated"] += 1
        per_video[source_image.parent.parent.name] += 1

    print(f"annotated: {stats['annotated']}")
    print(f"missing_image: {stats['missing_image']}")
    print(f"missing_com: {stats['missing_com']}")
    if stats["bad_entry"]:
        print(f"bad_entry: {stats['bad_entry']}")

    print("per_video:")
    for video_name, count in sorted(per_video.items()):
        print(f"  {video_name}: {count}")

    return 0 if stats["annotated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
