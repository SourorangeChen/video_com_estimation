from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Chenzixuan_20260505_test" / "WT02.csv"
OUTPUT_DIR = ROOT / "VICON_Trajectory_WT02"


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
    return label.split(":", 1)[-1].strip()


def parse_trajectories(csv_path: Path):
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "Trajectories")
    rate_hz = float(lines[start + 1].strip())
    marker_row = next(csv.reader([lines[start + 2]]))
    axis_row = next(csv.reader([lines[start + 3]]))
    unit_row = next(csv.reader([lines[start + 4]]))

    markers: list[tuple[str, int]] = []
    col = 2
    while col < len(marker_row):
        raw_name = marker_row[col].strip()
        if raw_name:
            markers.append((short_marker_name(raw_name), col))
        col += 3

    frames = []
    for line in lines[start + 5 :]:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if len(row) < 5 or not row[0].strip().isdigit():
            break
        points = {}
        for marker, marker_col in markers:
            try:
                xyz_raw = row[marker_col : marker_col + 3]
            except IndexError:
                continue
            if len(xyz_raw) != 3 or any(not value.strip() for value in xyz_raw):
                continue
            try:
                points[marker] = tuple(float(value) for value in xyz_raw)
            except ValueError:
                continue
        frames.append(
            {
                "frame": int(row[0]),
                "subframe": int(row[1]),
                "points": points,
            }
        )

    units = unit_row[2].strip() if len(unit_row) > 2 else "mm"
    axes = axis_row[2:5]
    return rate_hz, units, axes, [marker for marker, _ in markers], frames


def equal_3d_limits(frames):
    xs, ys, zs = [], [], []
    for frame in frames:
        for x, y, z in frame["points"].values():
            xs.append(x)
            ys.append(y)
            zs.append(z)
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    radius = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
    radius *= 1.08
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def render_frame(frame, limits, index: int, total: int, rate_hz: float, units: str):
    points = frame["points"]
    fig = plt.figure(figsize=(7, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    for a, b in SKELETON_EDGES:
        if a in points and b in points:
            xa, ya, za = points[a]
            xb, yb, zb = points[b]
            color = "#1f77b4"
            if a.startswith("L") or b.startswith("L"):
                color = "#2ca02c"
            if a.startswith("R") or b.startswith("R"):
                color = "#d62728"
            ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=2.2, alpha=0.92)

    if points:
        xs, ys, zs = zip(*points.values())
        ax.scatter(xs, ys, zs, s=14, c="#111111", depthshade=False)

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-78)
    ax.set_xlabel(f"X ({units})")
    ax.set_ylabel(f"Y ({units})")
    ax.set_zlabel(f"Z ({units})")
    ax.set_title(
        f"WT02 Trajectories | Frame {frame['frame']} | {index + 1}/{total} | t={(frame['frame'] - frames[0]['frame']) / rate_hz:.3f}s",
        pad=18,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output = OUTPUT_DIR / f"WT02_frame_{frame['frame']:04d}.png"
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    rate_hz, units, axes, markers, frames = parse_trajectories(CSV_PATH)
    limits = equal_3d_limits(frames)

    print(f"CSV: {CSV_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Rate: {rate_hz:g} Hz, units: {units}, markers: {len(markers)}, frames: {len(frames)}")
    print(f"First frame: {frames[0]['frame']}, last frame: {frames[-1]['frame']}")

    for index, frame in enumerate(frames):
        render_frame(frame, limits, index, len(frames), rate_hz, units)
        if (index + 1) % 100 == 0 or index + 1 == len(frames):
            print(f"Rendered {index + 1}/{len(frames)}")
