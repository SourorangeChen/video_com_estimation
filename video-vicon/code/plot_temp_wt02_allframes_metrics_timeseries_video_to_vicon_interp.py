from __future__ import annotations

r"""WT02 全部重叠帧指标对比：video 指标线性插值到 Vicon 时间轴。

这是临时输出版本，结果只写入:
H:\COM\temp\WT02_allframes_metrics_timeseries_video_to_vicon_interp
"""

import importlib.util
from pathlib import Path


# 被复用的源时序脚本路径，以及本变体(video→Vicon 插值)的独立输出目录。
SOURCE_SCRIPT = Path(r"H:\COM\video-vicon\code\plot_temp_wt02_allframes_metrics_timeseries.py")
OUTPUT_DIR = Path(r"H:\COM\temp\WT02_allframes_metrics_timeseries_video_to_vicon_interp")


def load_source_module():
    """用 importlib 按文件路径动态加载源时序脚本作为模块，以便复用其逻辑。"""
    # 根据源脚本路径创建模块规格(spec)。
    spec = importlib.util.spec_from_file_location("wt02_allframes_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {SOURCE_SCRIPT}")
    # 由 spec 创建空模块对象，再执行其代码完成导入。
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """主流程：加载源脚本 -> 把输出目录与两份 CSV 路径重定向到本变体 -> 运行。"""
    # 动态加载源时序脚本模块。
    module = load_source_module()
    # 覆盖输出目录与 CSV 路径，使本变体结果独立保存(与原脚本互不覆盖)。
    module.OUTPUT_DIR = OUTPUT_DIR
    module.TIMESERIES_CSV = OUTPUT_DIR / "WT02_allframes_metrics_timeseries_video_to_vicon_interp.csv"
    module.CORRELATION_CSV = OUTPUT_DIR / "WT02_allframes_metric_correlations_video_to_vicon_interp.csv"
    # 调用源脚本的 main() 执行实际计算与绘图。
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
