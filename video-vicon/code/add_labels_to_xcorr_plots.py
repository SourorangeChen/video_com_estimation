"""
Add keypoint-to-Vicon mapping labels to all cross-correlation PNGs in
video-vicon/validation/keypoints_xcorr/.

Reads each image, adds a subtitle line below the existing title, and
overwrites the file in-place.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Same mapping as validate_keypoints_normalized.py ──────────────────────
COCO17_TO_VICON: dict[str, str] = {
    "nose":           "Vicon: (LFHD + RFHD) / 2",
    "left_eye":       "Vicon: LFHD",
    "right_eye":      "Vicon: RFHD",
    "left_ear":       "Vicon: LFHD",
    "right_ear":      "Vicon: RFHD",
    "left_shoulder":  "Vicon: LSHO",
    "right_shoulder": "Vicon: RSHO",
    "left_elbow":     "Vicon: LELB",
    "right_elbow":    "Vicon: RELB",
    "left_wrist":     "Vicon: (LWRA + LWRB) / 2",
    "right_wrist":    "Vicon: (RWRA + RWRB) / 2",
    "left_hip":       "Vicon: LASI",
    "right_hip":      "Vicon: RASI",
    "left_knee":      "Vicon: LKNE",
    "right_knee":     "Vicon: RKNE",
    "left_ankle":     "Vicon: LANK",
    "right_ankle":    "Vicon: RANK",
}

XCORR_DIR = Path(__file__).parent / "keypoints_xcorr"

# How many pixels to extend canvas at the bottom for the label
LABEL_HEIGHT = 34
FONT_SIZE = 18
BG_COLOR = (255, 255, 255)       # white strip
TEXT_COLOR = (50, 50, 50)        # dark grey


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def annotate_image(img_path: Path, label_text: str) -> None:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Create new canvas with extra strip at bottom
    new_img = Image.new("RGB", (w, h + LABEL_HEIGHT), BG_COLOR)
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    font = get_font(FONT_SIZE)

    # Centre the label in the new strip
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w = bbox[2] - bbox[0]
    x = (w - text_w) // 2
    y = h + (LABEL_HEIGHT - (bbox[3] - bbox[1])) // 2

    draw.text((x, y), label_text, fill=TEXT_COLOR, font=font)
    new_img.save(img_path)


def main() -> None:
    if not XCORR_DIR.exists():
        print(f"Directory not found: {XCORR_DIR}")
        return

    processed = 0
    for folder in sorted(XCORR_DIR.iterdir()):
        if not folder.is_dir():
            continue
        # folder name like  kp11_left_hip  →  kp_name = "left_hip"
        m = re.match(r"kp\d+_(.+)", folder.name)
        if not m:
            continue
        kp_name = m.group(1)
        vicon_label = COCO17_TO_VICON.get(kp_name)
        if vicon_label is None:
            print(f"  [SKIP] unknown keypoint: {kp_name}")
            continue

        label_text = f"Video: {kp_name}   |   {vicon_label}"

        for img_path in sorted(folder.glob("*.png")):
            annotate_image(img_path, label_text)
            processed += 1
            print(f"  annotated: {img_path.relative_to(XCORR_DIR.parent)}")

    print(f"\nDone — {processed} images annotated.")


if __name__ == "__main__":
    main()
