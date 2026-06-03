"""从 Vicon CSV 中解析 Trajectories(标记点轨迹)并逐帧渲染 3D 骨架 PNG。

流程概述：
    1. parse_trajectories(): 定位 CSV 中的 "Trajectories" 区段，读取采样率、
       标记名、轴标签、单位，再逐帧读取各标记的 (x, y, z) 坐标。
    2. equal_3d_limits(): 统计所有帧所有点的范围，计算等比例的 3D 坐标轴范围。
    3. render_frame(): 对每一帧，按 SKELETON_EDGES 连线绘制骨架并保存为 PNG。

Vicon CSV 为多区段格式(Devices / Model Outputs / Trajectories)，本脚本只处理
Trajectories 区段；解析方式见 parse_trajectories()。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

# 使用无界面后端 Agg，便于在无显示环境(服务器)中批量出图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 以本脚本所在目录为根，定位输入 CSV 与输出目录。
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Chenzixuan_20260505_test" / "WT02.csv"
OUTPUT_DIR = ROOT / "VICON_Trajectory_WT02"


# 骨架连线定义：每对为需要连线的两个 Vicon 标记名。
# 依次包含：头部方框、躯干、骨盆、左臂、右臂、左腿、右腿。
SKELETON_EDGES = [
    ("LFHD", "RFHD"),
    ("RFHD", "RBHD"),
    ("RBHD", "LBHD"),
    ("LBHD", "LFHD"),
    ("LFHD", "LBHD"),
    ("RFHD", "RBHD"),
    ("C7", "CLAV"),
    ("CLAV", "STRN"),
    ("STRN", "RBAK"),
    ("RBAK", "C7"),
    ("LSHO", "C7"),
    ("RSHO", "C7"),
    ("LSHO", "CLAV"),
    ("RSHO", "CLAV"),
    ("LASI", "RASI"),
    ("RASI", "RPSI"),
    ("RPSI", "LPSI"),
    ("LPSI", "LASI"),
    ("LASI", "LSHO"),
    ("RASI", "RSHO"),
    ("LPSI", "C7"),
    ("RPSI", "C7"),
    ("LSHO", "LUPA"),
    ("LUPA", "LELB"),
    ("LELB", "LFRM"),
    ("LFRM", "LWRA"),
    ("LFRM", "LWRB"),
    ("LWRA", "LWRB"),
    ("LWRA", "LFIN"),
    ("LWRB", "LFIN"),
    ("RSHO", "RUPA"),
    ("RUPA", "RELB"),
    ("RELB", "RFRM"),
    ("RFRM", "RWRA"),
    ("RFRM", "RWRB"),
    ("RWRA", "RWRB"),
    ("RWRA", "RFIN"),
    ("RWRB", "RFIN"),
    ("LASI", "LTHI"),
    ("LTHI", "LKNE"),
    ("LKNE", "LTIB"),
    ("LTIB", "LANK"),
    ("LANK", "LHEE"),
    ("LANK", "LTOE"),
    ("LHEE", "LTOE"),
    ("RASI", "RTHI"),
    ("RTHI", "RKNE"),
    ("RKNE", "RTIB"),
    ("RTIB", "RANK"),
    ("RANK", "RHEE"),
    ("RANK", "RTOE"),
    ("RHEE", "RTOE"),
]


def short_marker_name(label: str) -> str:
    """去掉标记名中 "受试者:标记" 形式的前缀，只保留冒号后的标记名。"""
    return label.split(":", 1)[-1].strip()


def parse_trajectories(csv_path: Path):
    """解析 Vicon CSV 的 Trajectories 区段。

    返回 (采样率Hz, 单位, 轴标签, 标记名列表, 逐帧数据)。
    每帧数据为 {"frame", "subframe", "points": {标记名: (x,y,z)}}。
    """
    # 用 utf-8-sig 读取以兼容 BOM；按行切分。
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    # 定位 "Trajectories" 区段标题行的索引。
    start = next(i for i, line in enumerate(lines) if line.strip() == "Trajectories")
    # 区段结构(相对 start 的偏移)：
    #   +1 采样率, +2 标记名行, +3 轴标签行, +4 单位行, +5 起 数据行。
    rate_hz = float(lines[start + 1].strip())
    marker_row = next(csv.reader([lines[start + 2]]))
    axis_row = next(csv.reader([lines[start + 3]]))
    unit_row = next(csv.reader([lines[start + 4]]))

    # 解析标记名及其所在列：前两列为 Frame/Sub Frame，标记数据每 3 列(X/Y/Z)一组。
    markers: list[tuple[str, int]] = []
    col = 2
    while col < len(marker_row):
        raw_name = marker_row[col].strip()
        if raw_name:
            markers.append((short_marker_name(raw_name), col))
        col += 3

    # 逐行读取数据帧。
    frames = []
    for line in lines[start + 5 :]:
        # 跳过空行。
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        # 数据行首列必须是帧号(数字)；遇到非数据行(如下一区段)即停止。
        if len(row) < 5 or not row[0].strip().isdigit():
            break
        points = {}
        for marker, marker_col in markers:
            try:
                # 取该标记的连续 3 列作为 (x, y, z)。
                xyz_raw = row[marker_col : marker_col + 3]
            except IndexError:
                continue
            # 缺列或含空值(该帧此标记缺失)则跳过该标记。
            if len(xyz_raw) != 3 or any(not value.strip() for value in xyz_raw):
                continue
            try:
                points[marker] = tuple(float(value) for value in xyz_raw)
            except ValueError:
                continue
        frames.append(
            {
                "frame": int(row[0]),       # 帧号
                "subframe": int(row[1]),    # 子帧号
                "points": points,           # 该帧各标记坐标
            }
        )

    # 单位通常在单位行第 3 列(索引 2)，缺失时默认 mm。
    units = unit_row[2].strip() if len(unit_row) > 2 else "mm"
    # 轴标签取第 3~5 列(X/Y/Z)。
    axes = axis_row[2:5]
    return rate_hz, units, axes, [marker for marker, _ in markers], frames


def equal_3d_limits(frames):
    """根据所有帧所有点的坐标范围，计算等比例(立方体)的 3D 坐标轴范围。"""
    xs, ys, zs = [], [], []
    # 汇总全部点的 x/y/z。
    for frame in frames:
        for x, y, z in frame["points"].values():
            xs.append(x)
            ys.append(y)
            zs.append(z)
    # 取三轴各自范围的中心点。
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    # 半径取三轴跨度的最大值的一半，保证三轴等比例。
    radius = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
    # 留出 8% 边距。
    radius *= 1.08
    # 以中心 ± 半径构造三轴范围。
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def render_frame(frame, limits, index: int, total: int, rate_hz: float, units: str):
    """渲染单帧的 3D 骨架图并保存为 PNG。

    注意：标题中的相对时间引用了模块级全局 frames(取首帧帧号作为时间原点)。
    """
    points = frame["points"]
    # 新建 3D 画布。
    fig = plt.figure(figsize=(7, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    # 逐条骨架连线绘制(仅当两端标记在本帧都存在时)。
    for a, b in SKELETON_EDGES:
        if a in points and b in points:
            xa, ya, za = points[a]
            xb, yb, zb = points[b]
            # 默认蓝色;左侧肢体绿色,右侧肢体红色(便于区分左右)。
            color = "#1f77b4"
            if a.startswith("L") or b.startswith("L"):
                color = "#2ca02c"
            if a.startswith("R") or b.startswith("R"):
                color = "#d62728"
            ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=2.2, alpha=0.92)

    # 把所有标记点画成黑色散点。
    if points:
        xs, ys, zs = zip(*points.values())
        ax.scatter(xs, ys, zs, s=14, c="#111111", depthshade=False)

    # 设置等比例坐标范围与立方体盒子比例。
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    # 固定视角(俯仰 16°、方位 -78°)以保证各帧视角一致。
    ax.view_init(elev=16, azim=-78)
    ax.set_xlabel(f"X ({units})")
    ax.set_ylabel(f"Y ({units})")
    ax.set_zlabel(f"Z ({units})")
    # 标题含帧号、进度与相对首帧的时间(全局 frames[0] 为时间原点)。
    ax.set_title(
        f"WT02 Trajectories | Frame {frame['frame']} | {index + 1}/{total} | t={(frame['frame'] - frames[0]['frame']) / rate_hz:.3f}s",
        pad=18,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    # 输出文件名按帧号 4 位补零，保存后关闭画布释放内存。
    output = OUTPUT_DIR / f"WT02_frame_{frame['frame']:04d}.png"
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    # 确保输出目录存在。
    OUTPUT_DIR.mkdir(exist_ok=True)
    # 解析轨迹并计算统一的坐标轴范围。
    rate_hz, units, axes, markers, frames = parse_trajectories(CSV_PATH)
    limits = equal_3d_limits(frames)

    # 打印基本信息供核对。
    print(f"CSV: {CSV_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Rate: {rate_hz:g} Hz, units: {units}, markers: {len(markers)}, frames: {len(frames)}")
    print(f"First frame: {frames[0]['frame']}, last frame: {frames[-1]['frame']}")

    # 逐帧渲染，每 100 帧或最后一帧打印一次进度。
    for index, frame in enumerate(frames):
        render_frame(frame, limits, index, len(frames), rate_hz, units)
        if (index + 1) % 100 == 0 or index + 1 == len(frames):
            print(f"Rendered {index + 1}/{len(frames)}")
