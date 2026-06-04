from __future__ import annotations

r"""临时输出 WT02 前 100 帧视频侧指标图，包含人体 keypoints。

复用正式脚本 plot_wt02_keypoints_smooth_video_metrics.py 的输入与绘图逻辑，
只把输出目录改到 H:\COM\temp，避免覆盖 validation 目录。
"""

import importlib.util
from pathlib import Path


# 被复用的正式脚本路径，以及本临时脚本的独立输出目录(写到 temp，避免覆盖 validation)。
SOURCE_SCRIPT = Path(r"H:\COM\video-vicon\code\plot_wt02_keypoints_smooth_video_metrics.py")
OUTPUT_DIR = Path(r"H:\COM\temp\WT02_first100_video_metrics_keypoints")


def load_source_module():
    """用 importlib 按文件路径动态加载正式脚本作为模块，以便复用其函数与变量。"""
    # 根据源脚本路径创建模块规格(spec)。
    spec = importlib.util.spec_from_file_location("wt02_video_metrics_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {SOURCE_SCRIPT}")
    # 由 spec 创建空模块对象，再执行其代码完成导入。
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_output_dir(output_dir: Path) -> None:
    """清空临时输出目录(带 workspace 边界校验，拒绝删除工作区外路径)。"""
    workspace = Path(r"H:\COM").resolve()
    target = output_dir.resolve()
    # 安全校验：目标必须位于工作区目录之下，避免误删。
    if not str(target).startswith(str(workspace) + "\\"):
        raise RuntimeError(f"Refusing to clear outside workspace: {target}")
    # 确保目录存在，然后自底向上删除其中的文件与空目录。
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def main() -> int:
    """主流程：加载正式脚本 -> 把其输出目录重定向到 temp -> 清空后运行。"""
    # 动态加载正式绘图脚本模块。
    module = load_source_module()
    # 覆盖其 OUTPUT_DIR，使结果写入 temp 目录而非 validation。
    module.OUTPUT_DIR = OUTPUT_DIR
    # 清空 temp 输出目录后，调用正式脚本的 main() 执行绘图。
    clear_output_dir(OUTPUT_DIR)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
