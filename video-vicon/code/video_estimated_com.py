"""读取 PCT 关键点 JSON，逐帧计算质心(CoM)并在图片上绘制蓝色质心点。

流程概述：
    1. 动态加载 calculate_com.compute_frame_com(7 段人体测量学质心模型)。
    2. 读取 keypoints.json(每条记录含 image 路径与 keypoints)。
    3. 对每条记录：定位源图片 -> 计算质心 -> 在图上画蓝点并标注 "estimated"。
    4. 统计标注成功/缺图/缺质心数量，并按视频分类汇总后打印。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# 以本脚本所在目录为工作根目录，推导出 PCT 结果目录与关键点 JSON 路径。
WORKSPACE_ROOT = Path(__file__).resolve().parent
PCT_RESULTS_ROOT = WORKSPACE_ROOT / "Video" / "Video_Keypoint" / "pct_results"
KEYPOINTS_JSON = PCT_RESULTS_ROOT / "keypoints.json"
# calculate_com.py 所在目录(用于动态导入质心计算函数)。
CALCULATE_COM_DIR = Path(r"H:\COM")

# 质心标记使用的颜色(RGB 蓝色)。
BLUE = (0, 0, 255)


def load_com_function():
    """把 calculate_com.py 所在目录加入 sys.path 并导入 compute_frame_com。"""
    sys.path.insert(0, str(CALCULATE_COM_DIR))
    from calculate_com import compute_frame_com

    return compute_frame_com


def load_font(size: int = 18) -> ImageFont.ImageFont:
    """加载用于标注文字的字体；找不到系统字体则回退到 PIL 默认字体。"""
    # 依次尝试常见的 Windows 字体。
    for font_path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    # 都不存在时使用内置位图字体。
    return ImageFont.load_default()


def resolve_source_image(relative_image: str) -> Path | None:
    """根据 JSON 中记录的相对图片路径定位真实源图片文件。

    依次尝试：原路径 -> 加 "_pct" 后缀的路径 -> 同名前缀的模糊匹配。
    都找不到则返回 None。
    """
    # 1) 直接按记录路径查找。
    listed_path = PCT_RESULTS_ROOT / relative_image
    if listed_path.exists():
        return listed_path

    # 2) 尝试带 "_pct" 后缀的文件名(PCT 结果图命名)。
    suffix_path = listed_path.with_name(f"{listed_path.stem}_pct{listed_path.suffix}")
    if suffix_path.exists():
        return suffix_path

    # 3) 用同名前缀做模糊匹配，取排序后的第一个。
    matches = sorted(listed_path.parent.glob(f"{listed_path.stem}*{listed_path.suffix}"))
    return matches[0] if matches else None


def output_path_for(source_image: Path) -> Path:
    """根据源图片路径推导其质心标注图的输出路径。

    输出到该视频目录下的 com_estimated_frames/ 子目录，文件名保持不变。
    """
    # 源图片的上上级目录视为该视频的根目录。
    video_dir = source_image.parent.parent
    return video_dir / "com_estimated_frames" / source_image.name


def label_position(center_x: int, center_y: int, image_width: int, image_height: int) -> tuple[int, int]:
    """计算 "estimated" 文字标签的位置，尽量避免超出图像边界。"""
    # 默认放在质心点右侧，但不超过右边界(留出约 95px 文字宽度)。
    x = min(center_x + 10, max(image_width - 95, 0))
    # 默认放在质心点上方。
    y = center_y - 22
    # 若太靠顶部，则改放到质心点下方(并避免超出底边)。
    if y < 5:
        y = min(center_y + 10, max(image_height - 24, 0))
    # 保证坐标非负。
    return max(x, 0), max(y, 0)


def draw_estimated_com(source_image: Path, output_image: Path, com: dict[str, float], font: ImageFont.ImageFont) -> None:
    """在源图片上绘制质心蓝点与 "estimated" 文字，并保存到输出路径。"""
    # 打开图片并统一转为 RGB 模式。
    with Image.open(source_image) as original:
        image = original.convert("RGB")

    draw = ImageDraw.Draw(image)
    # 质心像素坐标(四舍五入取整)。
    center = (int(round(com["com_x"])), int(round(com["com_y"])))
    radius = 6
    # 画一个填充蓝色的圆点表示质心。
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=BLUE,
    )
    # 在合适位置标注 "estimated" 文字。
    draw.text(label_position(center[0], center[1], image.width, image.height), "estimated", fill=BLUE, font=font)

    # 确保输出目录存在后保存(高质量 JPEG)。
    output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image, quality=95)


def main() -> int:
    """主流程：遍历关键点 JSON，逐帧绘制质心标注并统计结果。"""
    # 关键点 JSON 不存在则直接退出。
    if not KEYPOINTS_JSON.exists():
        print(f"keypoints.json not found: {KEYPOINTS_JSON}")
        return 1

    # 加载质心计算函数与标注字体，并读取所有关键点记录。
    compute_frame_com = load_com_function()
    font = load_font()
    entries = json.loads(KEYPOINTS_JSON.read_text(encoding="utf-8"))

    # stats: 各类处理结果计数；per_video: 每个视频成功标注的帧数。
    stats: Counter[str] = Counter()
    per_video: Counter[str] = Counter()

    for entry in entries:
        # 跳过结构异常的记录。
        if not isinstance(entry, dict):
            stats["bad_entry"] += 1
            continue

        # 定位源图片，找不到则计入 missing_image。
        source_image = resolve_source_image(str(entry.get("image", "")))
        if source_image is None:
            stats["missing_image"] += 1
            continue

        # 计算质心，关键点不全时返回 None，计入 missing_com。
        com = compute_frame_com(entry.get("keypoints"))
        if com is None:
            stats["missing_com"] += 1
            continue

        # 绘制并保存质心标注图。
        output_image = output_path_for(source_image)
        draw_estimated_com(source_image, output_image, com, font)

        # 累计成功标注数，并按视频名分类计数。
        stats["annotated"] += 1
        per_video[source_image.parent.parent.name] += 1

    # 打印总体统计。
    print(f"annotated: {stats['annotated']}")
    print(f"missing_image: {stats['missing_image']}")
    print(f"missing_com: {stats['missing_com']}")
    if stats["bad_entry"]:
        print(f"bad_entry: {stats['bad_entry']}")

    # 打印每个视频的标注数量(按视频名排序)。
    print("per_video:")
    for video_name, count in sorted(per_video.items()):
        print(f"  {video_name}: {count}")

    # 只要有任意一帧标注成功就返回 0(成功)，否则返回 1。
    return 0 if stats["annotated"] else 1


if __name__ == "__main__":
    # 以 main() 返回值作为进程退出码。
    raise SystemExit(main())
