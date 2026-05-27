from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vicon_skeleton_drawing import SKELETON_EDGES, parse_trajectories


ROOT = Path(r"H:\COM\video-vicon")
VICON_CSV = ROOT / "data" / "Chenzixuan" / "Vicon" / "rawdata" / "Chenzixuan_20260505_test" / "WT02.csv"
KINEMATICS_CSV = ROOT / "validation" / "vicon_com_metric.csv"
LEGACY_KINEMATICS_CSV = ROOT / "validation" / "vicon_com_kinematics.csv"
OUTPUT_DIR = ROOT / "validation" / "WT02_first50_kinematics_3d"
OUTPUT_2D_DIR = ROOT / "validation" / "WT02_first50_kinematics_yz_2d"
TRIAL = "WT02"
FRAME_COUNT = 50
VELOCITY_VECTOR_SECONDS = 0.05


def load_trial_kinematics(csv_path: Path, trial: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["trial"] != trial:
                continue
            converted: dict[str, Any] = {"trial": row["trial"], "frame": int(row["frame"])}
            for key, value in row.items():
                if key in {"trial", "frame"}:
                    continue
                converted[key] = float(value)
            rows.append(converted)
    return rows


def trajectory_points_to_meters(frame: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    return {
        marker: (xyz[0] / 1000.0, xyz[1] / 1000.0, xyz[2] / 1000.0)
        for marker, xyz in frame["points"].items()
    }


def yz_point(point_xyz: tuple[float, float, float] | np.ndarray) -> tuple[float, float]:
    return float(point_xyz[1]), float(point_xyz[2])


def select_matching_frames(
    kinematics_rows: list[dict[str, Any]],
    trajectory_frames: list[dict[str, Any]],
    limit: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    trajectory_by_frame = {frame["frame"]: frame for frame in trajectory_frames}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in kinematics_rows:
        trajectory_frame = trajectory_by_frame.get(row["frame"])
        if trajectory_frame is None or not trajectory_frame["points"]:
            continue
        selected.append((row, trajectory_frame))
        if len(selected) == limit:
            break
    return selected


def equal_limits_for_selection(selection: list[tuple[dict[str, Any], dict[str, Any]]]):
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for kin, trajectory_frame in selection:
        for x, y, z in trajectory_points_to_meters(trajectory_frame).values():
            xs.append(x)
            ys.append(y)
            zs.append(z)
        for prefix in ("com", "xcom"):
            xs.append(kin[f"{prefix}_x_m"])
            ys.append(kin[f"{prefix}_y_m"])
            zs.append(kin[f"{prefix}_z_m"])

    center = np.array([
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    ])
    radius = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2.0
    radius *= 1.12
    return tuple((center[axis] - radius, center[axis] + radius) for axis in range(3))


def draw_vector(ax, origin, vector, color: str, label: str, linewidth: float = 2.0):
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        vector[0],
        vector[1],
        vector[2],
        color=color,
        linewidth=linewidth,
        arrow_length_ratio=0.18,
        label=label,
    )


def yz_limits_for_selection(selection: list[tuple[dict[str, Any], dict[str, Any]]]):
    ys: list[float] = []
    zs: list[float] = []
    for kin, trajectory_frame in selection:
        for point in trajectory_points_to_meters(trajectory_frame).values():
            y, z = yz_point(point)
            ys.append(y)
            zs.append(z)
        for prefix in ("com", "xcom"):
            ys.append(kin[f"{prefix}_y_m"])
            zs.append(kin[f"{prefix}_z_m"])

    y_margin = (max(ys) - min(ys)) * 0.08
    z_margin = (max(zs) - min(zs)) * 0.08
    return (min(ys) - y_margin, max(ys) + y_margin), (min(zs) - z_margin, max(zs) + z_margin)


def render_frame_yz_2d(
    kin: dict[str, Any],
    trajectory_frame: dict[str, Any],
    limits,
    output_path: Path,
    index: int,
    total: int,
) -> None:
    points = trajectory_points_to_meters(trajectory_frame)
    fig, ax = plt.subplots(figsize=(7.5, 8), dpi=150)

    for a, b in SKELETON_EDGES:
        if a in points and b in points:
            ya, za = yz_point(points[a])
            yb, zb = yz_point(points[b])
            color = "#4a4a4a"
            if a.startswith("L") or b.startswith("L"):
                color = "#2b8a3e"
            if a.startswith("R") or b.startswith("R"):
                color = "#c92a2a"
            ax.plot([ya, yb], [za, zb], color=color, linewidth=1.7, alpha=0.85)

    if points:
        yz = np.array([yz_point(point) for point in points.values()])
        ax.scatter(yz[:, 0], yz[:, 1], s=16, c="#111111", label="Vicon keypoints")

    com = np.array([kin["com_x_m"], kin["com_y_m"], kin["com_z_m"]])
    xcom = np.array([kin["xcom_x_m"], kin["xcom_y_m"], kin["xcom_z_m"]])
    displacement = np.array([
        kin["displacement_x_m"],
        kin["displacement_y_m"],
        kin["displacement_z_m"],
    ])
    velocity = np.array([
        kin["velocity_x_m_s"],
        kin["velocity_y_m_s"],
        kin["velocity_z_m_s"],
    ])

    com_yz = yz_point(com)
    xcom_yz = yz_point(xcom)
    ax.scatter(*com_yz, s=75, c="#1f77b4", label="CoM", zorder=5)
    ax.scatter(*xcom_yz, s=95, c="#ff7f0e", marker="x", linewidths=2.5, label="xCoM", zorder=6)
    ax.plot([com_yz[0], xcom_yz[0]], [com_yz[1], xcom_yz[1]], color="#ff7f0e", linewidth=1.2)

    disp_start = yz_point(com - displacement)
    disp_vec = yz_point(displacement)
    vel_vec = yz_point(velocity * VELOCITY_VECTOR_SECONDS)
    ax.arrow(disp_start[0], disp_start[1], disp_vec[0], disp_vec[1],
             color="#9467bd", width=0.002, length_includes_head=True, label="displacement")
    ax.arrow(com_yz[0], com_yz[1], vel_vec[0], vel_vec[1],
             color="#17a2b8", width=0.002, length_includes_head=True, label="velocity x 0.05s")

    metric_text = (
        f"disp = {kin['displacement_m']:.4f} m\n"
        f"vel = {kin['velocity_m_s']:.3f} m/s\n"
        f"omega0 = {kin['omega0_rad_s']:.3f} rad/s\n"
        f"xCoM-CoM = {np.linalg.norm(xcom - com):.3f} m"
    )
    ax.text(0.02, 0.98, metric_text, transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"WT02 front/back Y-Z view | Frame {kin['frame']} | {index + 1}/{total}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_overview_yz_2d(selection: list[tuple[dict[str, Any], dict[str, Any]]], limits, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 8), dpi=150)
    first_points = trajectory_points_to_meters(selection[0][1])
    for a, b in SKELETON_EDGES:
        if a in first_points and b in first_points:
            ya, za = yz_point(first_points[a])
            yb, zb = yz_point(first_points[b])
            ax.plot([ya, yb], [za, zb], color="#b0b0b0", linewidth=1.1, alpha=0.55)

    com_path = np.array([[kin["com_y_m"], kin["com_z_m"]] for kin, _ in selection])
    xcom_path = np.array([[kin["xcom_y_m"], kin["xcom_z_m"]] for kin, _ in selection])
    ax.plot(com_path[:, 0], com_path[:, 1], color="#1f77b4", linewidth=2.2, label="CoM path")
    ax.plot(xcom_path[:, 0], xcom_path[:, 1], color="#ff7f0e", linewidth=2.0, label="xCoM path")
    ax.scatter(com_path[0, 0], com_path[0, 1], c="#1f77b4", s=65, label="start")
    ax.scatter(com_path[-1, 0], com_path[-1, 1], c="#d62728", s=65, label="end")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("WT02 first 50 matched Vicon frames | Y-Z front/back view")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_frame(
    kin: dict[str, Any],
    trajectory_frame: dict[str, Any],
    limits,
    output_path: Path,
    index: int,
    total: int,
) -> None:
    points = trajectory_points_to_meters(trajectory_frame)
    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for a, b in SKELETON_EDGES:
        if a in points and b in points:
            xa, ya, za = points[a]
            xb, yb, zb = points[b]
            color = "#4a4a4a"
            if a.startswith("L") or b.startswith("L"):
                color = "#2b8a3e"
            if a.startswith("R") or b.startswith("R"):
                color = "#c92a2a"
            ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=1.5, alpha=0.82)

    if points:
        xs, ys, zs = zip(*points.values())
        ax.scatter(xs, ys, zs, s=14, c="#111111", depthshade=False, label="Vicon keypoints")

    com = np.array([kin["com_x_m"], kin["com_y_m"], kin["com_z_m"]])
    xcom = np.array([kin["xcom_x_m"], kin["xcom_y_m"], kin["xcom_z_m"]])
    displacement = np.array([
        kin["displacement_x_m"],
        kin["displacement_y_m"],
        kin["displacement_z_m"],
    ])
    velocity = np.array([
        kin["velocity_x_m_s"],
        kin["velocity_y_m_s"],
        kin["velocity_z_m_s"],
    ])

    ax.scatter(*com, s=70, c="#1f77b4", depthshade=False, label="CoM")
    ax.scatter(*xcom, s=90, c="#ff7f0e", marker="x", linewidths=2.5, depthshade=False, label="xCoM")
    ax.plot([com[0], xcom[0]], [com[1], xcom[1]], [com[2], xcom[2]], color="#ff7f0e", linewidth=1.2, alpha=0.75)

    draw_vector(ax, com - displacement, displacement, "#9467bd", "displacement")
    draw_vector(ax, com, velocity * VELOCITY_VECTOR_SECONDS, "#17a2b8", "velocity x 0.05s")

    metric_text = (
        f"disp = {kin['displacement_m']:.4f} m\n"
        f"vel = {kin['velocity_m_s']:.3f} m/s\n"
        f"omega0 = {kin['omega0_rad_s']:.3f} rad/s\n"
        f"xCoM-CoM = {np.linalg.norm(xcom - com):.3f} m"
    )
    ax.text2D(0.02, 0.92, metric_text, transform=ax.transAxes, fontsize=10, va="top")

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-72)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"WT02 Vicon CoM Metrics Example 2 | Frame {kin['frame']} | {index + 1}/{total}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_overview(selection: list[tuple[dict[str, Any], dict[str, Any]]], limits, output_path: Path) -> None:
    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    first_points = trajectory_points_to_meters(selection[0][1])
    for a, b in SKELETON_EDGES:
        if a in first_points and b in first_points:
            xa, ya, za = first_points[a]
            xb, yb, zb = first_points[b]
            ax.plot([xa, xb], [ya, yb], [za, zb], color="#b0b0b0", linewidth=1.1, alpha=0.55)

    com_path = np.array([[kin["com_x_m"], kin["com_y_m"], kin["com_z_m"]] for kin, _ in selection])
    xcom_path = np.array([[kin["xcom_x_m"], kin["xcom_y_m"], kin["xcom_z_m"]] for kin, _ in selection])
    ax.plot(com_path[:, 0], com_path[:, 1], com_path[:, 2], color="#1f77b4", linewidth=2.2, label="CoM path")
    ax.plot(xcom_path[:, 0], xcom_path[:, 1], xcom_path[:, 2], color="#ff7f0e", linewidth=2.0, label="xCoM path")
    ax.scatter(com_path[0, 0], com_path[0, 1], com_path[0, 2], c="#1f77b4", s=65, label="start")
    ax.scatter(com_path[-1, 0], com_path[-1, 1], com_path[-1, 2], c="#d62728", s=65, label="end")

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-72)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("WT02 first 50 matched Vicon frames | CoM and xCoM paths")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_2D_DIR.mkdir(parents=True, exist_ok=True)
    kinematics_csv = KINEMATICS_CSV if KINEMATICS_CSV.exists() else LEGACY_KINEMATICS_CSV
    kinematics_rows = load_trial_kinematics(kinematics_csv, TRIAL)
    _, _, _, _, trajectory_frames = parse_trajectories(VICON_CSV)
    selection = select_matching_frames(kinematics_rows, trajectory_frames, FRAME_COUNT)
    if len(selection) < FRAME_COUNT:
        raise RuntimeError(f"Only found {len(selection)} matching frames for {TRIAL}")

    limits = equal_limits_for_selection(selection)
    yz_limits = yz_limits_for_selection(selection)
    render_overview(selection, limits, OUTPUT_DIR / "WT02_first50_overview_com_xcom_path.png")
    render_overview_yz_2d(selection, yz_limits, OUTPUT_2D_DIR / "WT02_first50_overview_yz_com_xcom_path.png")
    for index, (kin, trajectory_frame) in enumerate(selection):
        output_path = OUTPUT_DIR / f"WT02_example2_frame_{index + 1:02d}_vicon_{kin['frame']:04d}.png"
        render_frame(kin, trajectory_frame, limits, output_path, index, len(selection))
        output_2d_path = OUTPUT_2D_DIR / f"WT02_example2_yz_frame_{index + 1:02d}_vicon_{kin['frame']:04d}.png"
        render_frame_yz_2d(kin, trajectory_frame, yz_limits, output_2d_path, index, len(selection))

    print(f"Output: {OUTPUT_DIR}")
    print(f"2D Y-Z output: {OUTPUT_2D_DIR}")
    print(f"Rendered frames: {len(selection)}")
    print(f"First Vicon frame: {selection[0][0]['frame']}")
    print(f"Last Vicon frame: {selection[-1][0]['frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
