"""为 video-vicon/validation/keypoints_xcorr/ 下所有互相关 PNG 添加"关键点→Vicon"映射标签。

对每张图：在底部扩出一条白边，在白边上居中绘制该关键点对应的 Vicon 标记说明，
然后**就地覆盖**原文件(故重复运行会叠加白边，注意只运行一次)。
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── 与 validate_keypoints_normalized.py 中相同的映射 ──────────────────────
# 关键点名 -> 其对应的 Vicon 标记说明文字。
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

# 待处理图片所在目录(相对本脚本)。
XCORR_DIR = Path(__file__).parent / "keypoints_xcorr"

# 底部白边高度(像素)，以及标签字体大小与配色。
LABEL_HEIGHT = 34
FONT_SIZE = 18
BG_COLOR = (255, 255, 255)       # 白色条带
TEXT_COLOR = (50, 50, 50)        # 深灰文字


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """依次尝试几种常见 Windows 字体；都不可用时回退到 PIL 默认字体。"""
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
    """在图片底部加一条白边并居中写入 label_text，覆盖保存原文件。"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # 新建比原图高 LABEL_HEIGHT 的白底画布，并把原图贴到顶部。
    new_img = Image.new("RGB", (w, h + LABEL_HEIGHT), BG_COLOR)
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    font = get_font(FONT_SIZE)

    # 计算文字尺寸，使其在底部白边内水平、垂直居中。
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w = bbox[2] - bbox[0]
    x = (w - text_w) // 2
    y = h + (LABEL_HEIGHT - (bbox[3] - bbox[1])) // 2

    # 绘制文字并就地覆盖保存。
    draw.text((x, y), label_text, fill=TEXT_COLOR, font=font)
    new_img.save(img_path)


def main() -> None:
    """遍历每个关键点子目录，给其中所有 PNG 加上对应的映射标签。"""
    if not XCORR_DIR.exists():
        print(f"Directory not found: {XCORR_DIR}")
        return

    processed = 0
    for folder in sorted(XCORR_DIR.iterdir()):
        # 只处理子目录。
        if not folder.is_dir():
            continue
        # 目录名形如 "kp11_left_hip" → 提取关键点名 "left_hip"。
        m = re.match(r"kp\d+_(.+)", folder.name)
        if not m:
            continue
        kp_name = m.group(1)
        # 查映射，未知关键点跳过。
        vicon_label = COCO17_TO_VICON.get(kp_name)
        if vicon_label is None:
            print(f"  [SKIP] unknown keypoint: {kp_name}")
            continue

        # 组装标签文字："Video: <关键点>   |   Vicon: <标记>"。
        label_text = f"Video: {kp_name}   |   {vicon_label}"

        # 给该目录下所有 PNG 加标签。
        for img_path in sorted(folder.glob("*.png")):
            annotate_image(img_path, label_text)
            processed += 1
            print(f"  annotated: {img_path.relative_to(XCORR_DIR.parent)}")

    print(f"\nDone — {processed} images annotated.")


if __name__ == "__main__":
    main()
