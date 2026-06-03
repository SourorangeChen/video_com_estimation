#!/usr/bin/env python
"""对文件夹中的每张图片批量运行 PCT 姿态关键点推理。

流程概述：
    1. 解析命令行参数(输入/输出目录、模型配置与权重、设备等)。
    2. 在输入目录下查找所有图片。
    3. 动态加载 PCT 仓库及其依赖(mmdet/mmpose/torch 等)。
    4. 初始化人体检测模型(mmdet)与 PCT 姿态模型。
    5. 逐张图片：先检测人体框，再做 top-down 姿态估计，得到 17 个关键点。
    6. 在图片上绘制骨架并保存，同时把关键点记录汇总写入 JSON。

说明：mmdet/mmpose/torch 等重型依赖只有在真正推理时才导入，
因此本模块在没有这些依赖的环境下也能被测试导入(测试只用到纯函数部分)。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable


# 支持处理的图片扩展名集合(统一用小写比较)。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 各类默认路径：PCT 仓库根目录、输入图片目录、结果输出目录。
DEFAULT_PCT_ROOT = Path(r"H:\PCT")
DEFAULT_INPUT_DIR = Path(r"H:\VICON\Chenzixuan\Video\Video_ViconTrial_PCT")
DEFAULT_OUTPUT_DIR = Path(r"H:\VICON\Chenzixuan\Video\Video_Keypoint\pct_results")


def np_array(value: Any) -> Any:
    """在 numpy 可用时返回 numpy 数组；测试环境无 numpy 时退化为 _ListLike。"""
    try:
        import numpy as np

        return np.array(value)
    except Exception:
        # 没有 numpy(如测试环境)时，返回一个仅提供 tolist() 的轻量替身。
        return _ListLike(value)


def np_scalar(value: Any) -> Any:
    """在 numpy 可用时返回 float32 标量；否则原样返回。"""
    try:
        import numpy as np

        return np.float32(value)
    except Exception:
        return value


class _ListLike:
    """numpy 缺失时的占位对象，只需支持 .tolist() 以便后续 JSON 序列化。"""

    def __init__(self, value: Any):
        self.value = value

    def tolist(self) -> Any:
        return self.value


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    """在输入目录中查找所有图片文件并按路径排序返回。

    recursive=True 时递归查找子目录。
    """
    # 递归用 "**/*"，非递归用 "*"。
    pattern = "**/*" if recursive else "*"
    # 仅保留扩展名在白名单中的普通文件。
    images = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    # 用小写 posix 路径排序，保证跨平台结果稳定。
    return sorted(images, key=lambda path: path.as_posix().lower())


def build_output_image_path(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    recursive: bool,
) -> Path:
    """为某张输入图片计算其骨架结果图的输出路径。

    递归模式下保留相对于输入根目录的子目录结构。
    """
    if recursive:
        # 保留输入图片相对输入根目录的父级目录层级。
        relative_parent = image_path.parent.relative_to(input_root)
        output_parent = output_root / relative_parent
    else:
        output_parent = output_root
    # 输出图命名为 "<原文件名>_pct.jpg"。
    return output_parent / f"{image_path.stem}_pct.jpg"


def _to_jsonable(value: Any) -> Any:
    """把含 numpy 数组/标量的嵌套结构递归转换为可 JSON 序列化的原生类型。"""
    # numpy 数组 -> list。
    if hasattr(value, "tolist"):
        return value.tolist()
    # numpy 标量 -> python 标量。
    if hasattr(value, "item"):
        return value.item()
    # 字典：递归处理每个值。
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    # 列表/元组：递归处理每个元素。
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    # 其余原生类型原样返回。
    return value


def pose_results_to_records(image_name: str, pose_results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """把一张图的姿态估计结果转换为可写入 JSON 的记录列表。

    每检测到一个人就生成一条记录，person_id 按检测顺序编号。
    """
    records = []
    for person_id, result in enumerate(pose_results):
        records.append({
            "image": image_name,
            "person_id": person_id,
            # bbox/score/keypoints 可能是 numpy 类型，统一转为可序列化类型。
            "bbox": _to_jsonable(result.get("bbox")),
            "bbox_score": _to_jsonable(result.get("bbox_score")),
            "keypoints": _to_jsonable(result.get("keypoints")),
        })
    return records


def load_runtime_dependencies(pct_root: Path) -> dict[str, Any]:
    """动态加载 PCT 推理所需的全部重型依赖，返回一个依赖字典。

    把 PCT 仓库加入 sys.path 后再导入其内部模块(如 models.build_posenet)。
    """
    pct_root = pct_root.resolve()
    # PCT 仓库不存在则无法推理。
    if not pct_root.exists():
        raise FileNotFoundError(f"PCT root not found: {pct_root}")

    # 将 PCT 仓库根目录插到 sys.path 最前，确保能 import 其内部 models 包。
    sys.path.insert(0, str(pct_root))

    import cv2
    import mmcv
    import numpy as np
    import torch
    from mmcv.runner import load_checkpoint
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, process_mmdet_results
    from mmpose.datasets import DatasetInfo
    from models import build_posenet

    # 把所有依赖打包成字典传递，避免到处 import。
    return {
        "cv2": cv2,
        "mmcv": mmcv,
        "np": np,
        "torch": torch,
        "load_checkpoint": load_checkpoint,
        "inference_detector": inference_detector,
        "init_detector": init_detector,
        "inference_top_down_pose_model": inference_top_down_pose_model,
        "process_mmdet_results": process_mmdet_results,
        "DatasetInfo": DatasetInfo,
        "build_posenet": build_posenet,
    }


def init_pct_pose_model(runtime: dict[str, Any], config_path: Path, checkpoint_path: Path, device: str) -> Any:
    """根据配置文件与权重初始化 PCT 姿态模型并切换到推理模式。"""
    mmcv = runtime["mmcv"]
    # 从 .py 配置文件加载模型配置。
    config = mmcv.Config.fromfile(str(config_path))
    # 推理阶段无需加载主干网络的预训练权重(下面会整体加载 checkpoint)。
    config.model.pretrained = None

    # 按配置搭建网络结构。
    model = runtime["build_posenet"](config.model)
    # 加载训练好的权重(先映射到 CPU，随后再 .to(device))。
    runtime["load_checkpoint"](model, str(checkpoint_path), map_location="cpu")
    # 把配置挂回模型，后续推理 API 需要用到。
    model.cfg = config
    # 转移到目标设备并设为评估模式(关闭 dropout/BN 更新)。
    model.to(device)
    model.eval()
    return model


def get_dataset_info(runtime: dict[str, Any], pose_model: Any) -> tuple[str, Any]:
    """从姿态模型配置中读取数据集类型与 DatasetInfo(关键点定义)。"""
    # 数据集类型字符串(如 'TopDownCocoDataset')。
    dataset = pose_model.cfg.data["test"]["type"]
    # 数据集的关键点/骨架元信息。
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        # 缺失时给出弃用告警，并返回 None(旧版配置可能没有此字段)。
        warnings.warn("dataset_info is missing from pose config.", DeprecationWarning)
        return dataset, None
    return dataset, runtime["DatasetInfo"](dataset_info)


# COCO-17 骨架连线定义：每对为 (起点关键点索引, 终点关键点索引)。
# 依次为：左腿、右腿、左臂、躯干与右臂、头面部。
SKELETON = [
    (15, 13), (13, 11), (11, 5),
    (12, 14), (14, 16), (12, 6),
    (9, 7), (7, 5), (5, 6), (6, 8), (8, 10),
    (3, 1), (1, 2), (1, 0), (0, 2), (2, 4),
]


def draw_pose_result(runtime: dict[str, Any], image_path: Path, pose_results: list[dict[str, Any]], out_file: Path, thickness: int) -> None:
    """在原图上绘制每个人的骨架连线与关键点圆点，并保存结果图。"""
    cv2 = runtime["cv2"]
    np = runtime["np"]

    # 读取图像(忽略 EXIF 方向，保证像素坐标与关键点一致)。
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # 骨架连线用黄色，关键点用品红色(BGR 顺序)。
    line_color = (0, 255, 255)
    point_color = (255, 0, 255)
    for result in pose_results:
        # 把关键点整理为 (17, N) 形状，前两列为 x、y。
        keypoints = np.array(result["keypoints"]).reshape(17, -1)
        # 逐条骨架连线绘制。
        for start, end in SKELETON:
            pt1 = tuple(keypoints[start, :2].astype(int))
            pt2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(image, pt1, pt2, line_color, max(1, thickness * 2), lineType=cv2.LINE_AA)
        # 逐个关键点画实心圆。
        for x, y, *_ in keypoints:
            cv2.circle(image, (int(x), int(y)), max(2, thickness * 2), point_color, -1, lineType=cv2.LINE_AA)

    # 确保输出目录存在后写出结果图。
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), image)


def process_images(args: argparse.Namespace) -> int:
    """主处理流程：加载模型，逐张图片推理、绘制并汇总关键点 JSON。"""
    # 规范化各路径为绝对路径。
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    pct_root = args.pct_root.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # 收集待处理图片。
    images = find_images(input_dir, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    # 加载依赖并初始化检测模型与姿态模型。
    runtime = load_runtime_dependencies(pct_root)
    det_model = runtime["init_detector"](
        str(args.det_config),
        str(args.det_checkpoint),
        device=args.device.lower(),
    )
    pose_model = init_pct_pose_model(runtime, args.pose_config, args.pose_checkpoint, args.device.lower())
    # 读取数据集类型与关键点定义。
    dataset, dataset_info = get_dataset_info(runtime, pose_model)

    all_records = []
    total = len(images)
    # 逐张图片处理(index 从 1 开始，便于打印进度)。
    for index, image_path in enumerate(images, start=1):
        out_file = build_output_image_path(image_path, input_dir, output_dir, recursive=args.recursive)
        # 若启用跳过已存在结果，且输出图已存在，则跳过该图。
        if args.skip_existing and out_file.exists():
            print(f"[{index}/{total}] skip existing {image_path}")
            continue

        print(f"[{index}/{total}] processing {image_path}")
        # 第一步：人体检测，得到所有候选框。
        mmdet_results = runtime["inference_detector"](det_model, str(image_path))
        # 仅保留指定类别(默认 person)的检测框。
        person_results = runtime["process_mmdet_results"](mmdet_results, args.det_cat_id)
        # 第二步：在每个人体框内做 top-down 姿态估计。
        pose_results, _ = runtime["inference_top_down_pose_model"](
            pose_model,
            str(image_path),
            person_results,
            bbox_thr=args.bbox_thr,    # 框置信度阈值
            format="xyxy",             # 框格式为 (x1,y1,x2,y2)
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        # 绘制并保存骨架结果图。
        draw_pose_result(runtime, image_path, pose_results, out_file, args.thickness)
        # 记录时用相对路径(递归)或文件名(非递归)作为图片标识。
        relative_image = str(image_path.relative_to(input_dir)) if args.recursive else image_path.name
        all_records.extend(pose_results_to_records(relative_image, pose_results))

    # 把所有人的关键点记录汇总写入 JSON。
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoint_json = output_dir / args.json_name
    keypoint_json.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(all_records)} person records to {keypoint_json}")
    return 0


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run mmdet + PCT on every image in a folder.",
        # 在帮助信息中显示各参数默认值。
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 输入图片目录与输出目录。
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder containing input images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for PCT result images and JSON.")
    # PCT 仓库根目录。
    parser.add_argument("--pct-root", type=Path, default=DEFAULT_PCT_ROOT, help="PCT repository root.")
    # 人体检测模型配置与权重。
    parser.add_argument("--det-config", type=Path, default=DEFAULT_PCT_ROOT / "vis_tools" / "cascade_rcnn_x101_64x4d_fpn_coco.py")
    parser.add_argument("--det-checkpoint", type=Path, default=DEFAULT_PCT_ROOT / "vis_tools" / "cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth")
    # PCT 姿态模型配置与权重。
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_PCT_ROOT / "configs" / "pct_base_classifier.py")
    parser.add_argument("--pose-checkpoint", type=Path, default=DEFAULT_PCT_ROOT / "weights" / "pct" / "swin_base.pth")
    # 推理设备。
    parser.add_argument("--device", default="cuda:0", help="Inference device, for example cuda:0 or cpu.")
    # 是否递归处理子目录(一对互斥开关，默认递归)。
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Process images in nested folders too.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only process images directly inside input-dir.")
    # 是否跳过已有输出图。
    parser.add_argument("--skip-existing", action="store_true", help="Skip images whose output JPG already exists.")
    # mmdet 中 person 类别的 id。
    parser.add_argument("--det-cat-id", type=int, default=1, help="mmdet category id for person.")
    # 检测置信度阈值。
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Detection confidence threshold.")
    # 骨架线条粗细。
    parser.add_argument("--thickness", type=int, default=2, help="Skeleton drawing thickness.")
    # 输出 JSON 文件名。
    parser.add_argument("--json-name", default="keypoints.json", help="Output JSON filename.")
    return parser.parse_args()


def main() -> int:
    """命令行入口：解析参数并执行图片处理。"""
    return process_images(parse_args())


if __name__ == "__main__":
    # 以 main() 返回值作为进程退出码。
    raise SystemExit(main())
