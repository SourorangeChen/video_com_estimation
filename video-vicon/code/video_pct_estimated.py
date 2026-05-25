#!/usr/bin/env python
"""Batch PCT keypoint inference for every image in a folder."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_PCT_ROOT = Path(r"H:\PCT")
DEFAULT_INPUT_DIR = Path(r"H:\VICON\Chenzixuan\Video\Video_ViconTrial_PCT")
DEFAULT_OUTPUT_DIR = Path(r"H:\VICON\Chenzixuan\Video\Video_Keypoint\pct_results")


def np_array(value: Any) -> Any:
    """Return a numpy array when numpy is available; used by tests too."""
    try:
        import numpy as np

        return np.array(value)
    except Exception:
        return _ListLike(value)


def np_scalar(value: Any) -> Any:
    """Return a numpy scalar when numpy is available; used by tests too."""
    try:
        import numpy as np

        return np.float32(value)
    except Exception:
        return value


class _ListLike:
    def __init__(self, value: Any):
        self.value = value

    def tolist(self) -> Any:
        return self.value


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.as_posix().lower())


def build_output_image_path(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    recursive: bool,
) -> Path:
    if recursive:
        relative_parent = image_path.parent.relative_to(input_root)
        output_parent = output_root / relative_parent
    else:
        output_parent = output_root
    return output_parent / f"{image_path.stem}_pct.jpg"


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def pose_results_to_records(image_name: str, pose_results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for person_id, result in enumerate(pose_results):
        records.append({
            "image": image_name,
            "person_id": person_id,
            "bbox": _to_jsonable(result.get("bbox")),
            "bbox_score": _to_jsonable(result.get("bbox_score")),
            "keypoints": _to_jsonable(result.get("keypoints")),
        })
    return records


def load_runtime_dependencies(pct_root: Path) -> dict[str, Any]:
    pct_root = pct_root.resolve()
    if not pct_root.exists():
        raise FileNotFoundError(f"PCT root not found: {pct_root}")

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
    mmcv = runtime["mmcv"]
    config = mmcv.Config.fromfile(str(config_path))
    config.model.pretrained = None

    model = runtime["build_posenet"](config.model)
    runtime["load_checkpoint"](model, str(checkpoint_path), map_location="cpu")
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def get_dataset_info(runtime: dict[str, Any], pose_model: Any) -> tuple[str, Any]:
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn("dataset_info is missing from pose config.", DeprecationWarning)
        return dataset, None
    return dataset, runtime["DatasetInfo"](dataset_info)


SKELETON = [
    (15, 13), (13, 11), (11, 5),
    (12, 14), (14, 16), (12, 6),
    (9, 7), (7, 5), (5, 6), (6, 8), (8, 10),
    (3, 1), (1, 2), (1, 0), (0, 2), (2, 4),
]


def draw_pose_result(runtime: dict[str, Any], image_path: Path, pose_results: list[dict[str, Any]], out_file: Path, thickness: int) -> None:
    cv2 = runtime["cv2"]
    np = runtime["np"]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    line_color = (0, 255, 255)
    point_color = (255, 0, 255)
    for result in pose_results:
        keypoints = np.array(result["keypoints"]).reshape(17, -1)
        for start, end in SKELETON:
            pt1 = tuple(keypoints[start, :2].astype(int))
            pt2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(image, pt1, pt2, line_color, max(1, thickness * 2), lineType=cv2.LINE_AA)
        for x, y, *_ in keypoints:
            cv2.circle(image, (int(x), int(y)), max(2, thickness * 2), point_color, -1, lineType=cv2.LINE_AA)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), image)


def process_images(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    pct_root = args.pct_root.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = find_images(input_dir, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    runtime = load_runtime_dependencies(pct_root)
    det_model = runtime["init_detector"](
        str(args.det_config),
        str(args.det_checkpoint),
        device=args.device.lower(),
    )
    pose_model = init_pct_pose_model(runtime, args.pose_config, args.pose_checkpoint, args.device.lower())
    dataset, dataset_info = get_dataset_info(runtime, pose_model)

    all_records = []
    total = len(images)
    for index, image_path in enumerate(images, start=1):
        out_file = build_output_image_path(image_path, input_dir, output_dir, recursive=args.recursive)
        if args.skip_existing and out_file.exists():
            print(f"[{index}/{total}] skip existing {image_path}")
            continue

        print(f"[{index}/{total}] processing {image_path}")
        mmdet_results = runtime["inference_detector"](det_model, str(image_path))
        person_results = runtime["process_mmdet_results"](mmdet_results, args.det_cat_id)
        pose_results, _ = runtime["inference_top_down_pose_model"](
            pose_model,
            str(image_path),
            person_results,
            bbox_thr=args.bbox_thr,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        draw_pose_result(runtime, image_path, pose_results, out_file, args.thickness)
        relative_image = str(image_path.relative_to(input_dir)) if args.recursive else image_path.name
        all_records.extend(pose_results_to_records(relative_image, pose_results))

    output_dir.mkdir(parents=True, exist_ok=True)
    keypoint_json = output_dir / args.json_name
    keypoint_json.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(all_records)} person records to {keypoint_json}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mmdet + PCT on every image in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder containing input images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for PCT result images and JSON.")
    parser.add_argument("--pct-root", type=Path, default=DEFAULT_PCT_ROOT, help="PCT repository root.")
    parser.add_argument("--det-config", type=Path, default=DEFAULT_PCT_ROOT / "vis_tools" / "cascade_rcnn_x101_64x4d_fpn_coco.py")
    parser.add_argument("--det-checkpoint", type=Path, default=DEFAULT_PCT_ROOT / "vis_tools" / "cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth")
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_PCT_ROOT / "configs" / "pct_base_classifier.py")
    parser.add_argument("--pose-checkpoint", type=Path, default=DEFAULT_PCT_ROOT / "weights" / "pct" / "swin_base.pth")
    parser.add_argument("--device", default="cuda:0", help="Inference device, for example cuda:0 or cpu.")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="Process images in nested folders too.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only process images directly inside input-dir.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images whose output JPG already exists.")
    parser.add_argument("--det-cat-id", type=int, default=1, help="mmdet category id for person.")
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Detection confidence threshold.")
    parser.add_argument("--thickness", type=int, default=2, help="Skeleton drawing thickness.")
    parser.add_argument("--json-name", default="keypoints.json", help="Output JSON filename.")
    return parser.parse_args()


def main() -> int:
    return process_images(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
