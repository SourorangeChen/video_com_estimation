from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import detrend

# Machine-specific paths for this research project (H:\COM on Windows)
VICON_CSV_ROOT = Path(r"H:\COM\video-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test")
KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json")
MANIFEST_CSV   = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_trial\Video_ViconTrial_manifest.csv")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
VIDEO_FPS      = 29.996
GRAVITY_M_S2   = 9.81


def remove_linear_drift(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend (zero-drift) from signal."""
    return detrend(signal, type="linear")


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    std = signal.std()
    if std == 0.0:
        raise ValueError("zero std: signal is constant, cannot z-score normalize")
    return (signal - signal.mean()) / std


def compute_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    # Returns (pearson_r, p_value, nrmse). Inputs are expected to be z-score normalized;
    # RMSE on z-scored signals is dimensionless (in units of std), i.e. normalized RMSE.
    r, p = pearsonr(a, b)
    nrmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return float(r), float(p), nrmse


def compute_xcorr(a: np.ndarray, b: np.ndarray, fps: float = VIDEO_FPS) -> tuple[float, int, float]:
    """Cross-correlation: returns (peak_r, lag_frames, lag_ms).
    lag_frames > 0 means b leads a (b occurs earlier than a).
    """
    n = len(a)
    corr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    corr /= n * a.std() * b.std()
    lags = np.arange(-(n - 1), n)
    peak_idx = int(np.argmax(corr))
    lag_frames = int(lags[peak_idx])
    lag_ms = round(lag_frames / fps * 1000, 1)
    return round(float(corr[peak_idx]), 4), lag_frames, lag_ms


def parse_vicon_model_outputs(
    csv_path: Path,
) -> tuple[float, list[int], list[float], list[float], list[float]]:
    """Parse CentreOfMass (mm) from the 'Model Outputs' section of a Vicon CSV export."""
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "Model Outputs"),
        None,
    )
    if start is None:
        raise ValueError(f"'Model Outputs' section not found in {csv_path}")

    rate_hz = float(lines[start + 1].strip())
    names = next(csv.reader([lines[start + 2]]))
    units = next(csv.reader([lines[start + 4]]))

    com_col = next(
        (i for i, name in enumerate(names) if name.strip().endswith(":CentreOfMass")),
        None,
    )
    if com_col is None:
        raise ValueError(f"':CentreOfMass' column not found in Model Outputs of {csv_path}")

    # Verify CentreOfMass unit is mm (Vicon Model Outputs should always export in mm)
    com_unit = units[com_col].strip() if com_col < len(units) else ""
    if com_unit and com_unit.lower() != "mm":
        raise ValueError(f"Expected CentreOfMass unit 'mm', got '{com_unit}' in {csv_path}")

    frame_numbers: list[int] = []
    com_x: list[float] = []
    com_y: list[float] = []
    com_z: list[float] = []

    for line in lines[start + 5:]:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if not row or not row[0].strip().isdigit():
            break
        try:
            x_str = row[com_col].strip()
            y_str = row[com_col + 1].strip()
            z_str = row[com_col + 2].strip()
        except IndexError:
            continue
        if not x_str or not y_str or not z_str:
            continue
        frame_numbers.append(int(row[0]))
        com_x.append(float(x_str))
        com_y.append(float(y_str))
        com_z.append(float(z_str))

    return rate_hz, frame_numbers, com_x, com_y, com_z


def parse_video_com(
    keypoints_json: Path,
    trial_name: str,
) -> tuple[list[int], list[float], list[float]]:
    """Return (frame_numbers, com_x_px, com_y_px) for one trial from keypoints.json."""
    folder_prefix = f"Video_{trial_name}_Trajectory"
    records: list[Any] = json.loads(keypoints_json.read_text(encoding="utf-8"))

    frame_numbers: list[int] = []
    com_x: list[float] = []
    com_y: list[float] = []

    for entry in records:
        if not isinstance(entry, dict):
            continue
        image: str = entry.get("image", "")
        if not image.startswith(folder_prefix):
            continue
        com = entry.get("com")
        if not isinstance(com, dict):
            continue
        stem = Path(image).stem  # "frame_000001"
        try:
            frame_num = int(stem.split("_")[-1])
        except ValueError:
            continue
        frame_numbers.append(frame_num)
        com_x.append(float(com["com_x"]))
        com_y.append(float(com["com_y"]))

    return frame_numbers, com_x, com_y


def load_manifest(manifest_csv: Path) -> dict[str, dict[str, Any]]:
    """Load trial timing info from the manifest CSV. Returns dict keyed by trial name."""
    result: dict[str, dict[str, Any]] = {}
    with manifest_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial = row["trial"].strip()
            result[trial] = {
                "first_frame": int(row["first_frame"]),
                "last_frame": int(row["last_frame"]),
                "trajectory_rate_hz": float(row["trajectory_rate_hz"]),
            }
    return result


def build_video_time_axis(frame_numbers: list[int], fps: float = VIDEO_FPS) -> np.ndarray:
    """Convert video frame numbers (1-indexed) to relative time in seconds (frame 1 → t=0)."""
    return np.array([(fn - 1) / fps for fn in frame_numbers])


def build_vicon_time_axis(
    frame_numbers: list[int], first_frame: int, rate_hz: float
) -> np.ndarray:
    """Convert Vicon frame numbers to relative time in seconds (first_frame → t=0)."""
    return np.array([(fn - first_frame) / rate_hz for fn in frame_numbers])


def interpolate_vicon_to_video(
    video_t: np.ndarray, vicon_t: np.ndarray, vicon_vals: np.ndarray
) -> np.ndarray:
    """Linearly interpolate Vicon values onto video time points."""
    # vicon_t must be monotonically increasing (np.interp requirement)
    return np.interp(video_t, vicon_t, vicon_vals)


def compute_com_kinematics(
    trial: str,
    frame_numbers: list[int],
    time_s: np.ndarray,
    com_x_mm: np.ndarray,
    com_y_mm: np.ndarray,
    com_z_mm: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute Vicon CoM displacement, velocity, and xCoM.

    Vicon exports CoM in millimeters. Kinematic outputs use meters and seconds.
    Displacement is frame-to-frame: r(t_i) - r(t_{i-1}); the first frame is zero.
    The inverted-pendulum length l(t) is the per-frame CoM height, i.e. com_z_m.
    """
    if not (len(frame_numbers) == len(time_s) == len(com_x_mm) == len(com_y_mm) == len(com_z_mm)):
        raise ValueError("frame, time, and CoM arrays must have the same length")
    if len(frame_numbers) < 2:
        raise ValueError("at least two frames are required to compute velocity")

    com_m = np.column_stack([com_x_mm, com_y_mm, com_z_mm]).astype(float) / 1000.0
    if np.any(com_m[:, 2] <= 0.0):
        raise ValueError("xCoM requires positive CoM height for every frame")

    displacement = np.vstack([np.zeros(3), np.diff(com_m, axis=0)])
    displacement_mag = np.linalg.norm(displacement, axis=1)
    velocity = np.gradient(com_m, time_s, axis=0)
    velocity_mag = np.linalg.norm(velocity, axis=1)
    omega0 = np.sqrt(GRAVITY_M_S2 / com_m[:, 2])
    xcom = com_m + velocity / omega0[:, np.newaxis]

    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(frame_numbers):
        rows.append({
            "trial": trial,
            "frame": int(frame),
            "time_s": float(time_s[idx]),
            "com_x_m": float(com_m[idx, 0]),
            "com_y_m": float(com_m[idx, 1]),
            "com_z_m": float(com_m[idx, 2]),
            "displacement_x_m": float(displacement[idx, 0]),
            "displacement_y_m": float(displacement[idx, 1]),
            "displacement_z_m": float(displacement[idx, 2]),
            "displacement_m": float(displacement_mag[idx]),
            "velocity_x_m_s": float(velocity[idx, 0]),
            "velocity_y_m_s": float(velocity[idx, 1]),
            "velocity_z_m_s": float(velocity[idx, 2]),
            "velocity_m_s": float(velocity_mag[idx]),
            "omega0_rad_s": float(omega0[idx]),
            "xcom_x_m": float(xcom[idx, 0]),
            "xcom_y_m": float(xcom[idx, 1]),
            "xcom_z_m": float(xcom[idx, 2]),
        })
    return rows


def compute_trial_kinematics(
    trial: str,
    manifest_row: dict[str, Any],
    vicon_csv: Path,
) -> list[dict[str, Any]]:
    """Compute Vicon CoM kinematics for one trial."""
    _, vicon_frames, vicon_cx, vicon_cy, vicon_cz = parse_vicon_model_outputs(vicon_csv)
    if not vicon_frames:
        print(f"[{trial}] WARNING: no Vicon COM records found for kinematics, skipping")
        return []

    time_s = build_vicon_time_axis(
        vicon_frames,
        manifest_row["first_frame"],
        manifest_row["trajectory_rate_hz"],
    )
    return compute_com_kinematics(
        trial=trial,
        frame_numbers=vicon_frames,
        time_s=time_s,
        com_x_mm=np.array(vicon_cx),
        com_y_mm=np.array(vicon_cy),
        com_z_mm=np.array(vicon_cz),
    )


def plot_comparison(
    t: np.ndarray,
    video_z: np.ndarray,
    vicon_z: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save a side-by-side z-score overlay plot of video vs Vicon COM."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, video_z, color="tab:blue", linewidth=1.2, label="video (z-score)")
    ax.plot(t, vicon_z, color="tab:red", linewidth=1.2, label="vicon (z-score)", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_xcorr(
    a: np.ndarray,
    b: np.ndarray,
    fps: float,
    peak_r: float,
    lag_frames: int,
    lag_ms: float,
    title: str,
    output_path: Path,
) -> None:
    """Save cross-correlation function plot with peak marked."""
    n = len(a)
    corr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    corr /= n * a.std() * b.std()
    lags = np.arange(-(n - 1), n)
    lags_ms = lags / fps * 1000

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags_ms, corr, color="tab:purple", linewidth=1.2)
    ax.axvline(lag_ms, color="tab:orange", linewidth=1.2, linestyle="--",
               label=f"peak lag = {lag_ms:.1f} ms ({lag_frames} frames)")
    ax.scatter([lag_ms], [peak_r], color="tab:orange", zorder=5, s=60)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Lag (ms)  [positive = video leads Vicon]")
    ax.set_ylabel("Cross-correlation r")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write validation summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial", "axis", "pearson_r", "p_value", "nrmse", "n_frames",
                  "xcorr_peak_r", "xcorr_lag_frames", "xcorr_lag_ms", "warning"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_kinematics_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write per-frame Vicon CoM kinematics rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "frame", "time_s",
        "com_x_m", "com_y_m", "com_z_m",
        "displacement_x_m", "displacement_y_m", "displacement_z_m", "displacement_m",
        "velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s", "velocity_m_s",
        "omega0_rad_s",
        "xcom_x_m", "xcom_y_m", "xcom_z_m",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_trial(
    trial: str,
    manifest_row: dict[str, Any],
    keypoints_json: Path,
    vicon_csv: Path,
    plots_dir: Path,
) -> list[dict[str, Any]]:
    """Run normalized COM comparison for one trial. Returns list of result rows."""
    # --- 视频 COM ---
    video_frames, video_cx, video_cy = parse_video_com(keypoints_json, trial)
    if not video_frames:
        print(f"[{trial}] WARNING: no video COM records found, skipping")
        return []

    # --- Vicon COM ---
    _, vicon_frames, vicon_cx, vicon_cy, vicon_cz = parse_vicon_model_outputs(vicon_csv)
    if not vicon_frames:
        print(f"[{trial}] WARNING: no Vicon COM records found, skipping")
        return []

    first_frame = manifest_row["first_frame"]
    rate_hz = manifest_row["trajectory_rate_hz"]

    video_t = build_video_time_axis(video_frames)
    vicon_t = build_vicon_time_axis(vicon_frames, first_frame, rate_hz)

    # Restrict to overlapping time window
    t_min = max(video_t[0], vicon_t[0])
    t_max = min(video_t[-1], vicon_t[-1])
    mask = (video_t >= t_min) & (video_t <= t_max)
    video_t = video_t[mask]
    video_cx_arr = np.array(video_cx)[mask]
    video_cy_arr = np.array(video_cy)[mask]

    if len(video_t) < 5:
        print(f"[{trial}] WARNING: fewer than 5 overlapping frames, skipping")
        return []

    vicon_cy_arr = np.array(vicon_cy)
    vicon_cz_arr = np.array(vicon_cz)
    vicon_t_arr = np.array(vicon_t)

    # Interpolate Vicon to video time points
    vicon_y_interp = interpolate_vicon_to_video(video_t, vicon_t_arr, vicon_cy_arr)
    vicon_z_interp = interpolate_vicon_to_video(video_t, vicon_t_arr, vicon_cz_arr)

    # Sign alignment:
    # - video x positive is opposite to Vicon Y positive → flip Y
    # - video y increases downward (pixel), Vicon Z increases upward → flip Z
    vicon_y_aligned = -vicon_y_interp
    vicon_z_aligned = -vicon_z_interp

    rows: list[dict[str, Any]] = []
    missing_pct = (mask.size - mask.sum()) / mask.size * 100

    for axis_label, vid_sig, vic_sig, plot_suffix in [
        ("horizontal (video_x vs vicon_Y)", video_cx_arr, vicon_y_aligned, "x"),
        ("vertical (video_y vs vicon_Z)", video_cy_arr, vicon_z_aligned, "z"),
    ]:
        warning = f"temporal overlap clipped {missing_pct:.0f}% of frames" if missing_pct > 20 else ""

        try:
            vid_z = zscore_normalize(remove_linear_drift(vid_sig))
            vic_z = zscore_normalize(remove_linear_drift(vic_sig))
        except ValueError as exc:
            print(f"[{trial}][{axis_label}] zscore failed: {exc}")
            continue

        r, p, nrmse = compute_metrics(vid_z, vic_z)
        xcorr_peak, xcorr_lag_frames, xcorr_lag_ms = compute_xcorr(vid_z, vic_z)

        # Lag-aligned overlay: shift signals by xcorr lag before plotting
        shift = abs(xcorr_lag_frames)
        if xcorr_lag_frames < 0:
            # Vicon leads video: align by trimming start of vicon, end of video
            vid_z_aligned = vid_z[:-shift] if shift > 0 else vid_z
            vic_z_aligned = vic_z[shift:] if shift > 0 else vic_z
            t_aligned = video_t[:-shift] if shift > 0 else video_t
        elif xcorr_lag_frames > 0:
            # Video leads Vicon: align by trimming start of video, end of vicon
            vid_z_aligned = vid_z[shift:]
            vic_z_aligned = vic_z[:-shift]
            t_aligned = video_t[shift:]
        else:
            vid_z_aligned, vic_z_aligned, t_aligned = vid_z, vic_z, video_t

        aligned_plot_path = plots_dir / f"{trial}_{plot_suffix}.png"
        plot_comparison(
            t_aligned,
            vid_z_aligned,
            vic_z_aligned,
            title=f"{trial} — {axis_label} | xcorr_r={xcorr_peak:.3f} lag={xcorr_lag_ms:.1f}ms",
            output_path=aligned_plot_path,
        )

        rows.append({
            "trial": trial,
            "axis": axis_label,
            "pearson_r": round(r, 4),
            "p_value": f"{p:.4e}",
            "nrmse": round(nrmse, 4),
            "n_frames": int(len(video_t)),
            "xcorr_peak_r": xcorr_peak,
            "xcorr_lag_frames": xcorr_lag_frames,
            "xcorr_lag_ms": xcorr_lag_ms,
            "warning": warning,
        })
        print(f"[{trial}][{plot_suffix}] r={r:.3f}, nRMSE={nrmse:.3f}, xcorr_peak={xcorr_peak:.3f}, lag={xcorr_lag_frames}f({xcorr_lag_ms}ms), n={len(video_t)}")

    return rows


def main() -> int:
    manifest = load_manifest(MANIFEST_CSV)
    plots_dir = VALIDATION_DIR / "plots_aligned"
    all_rows: list[dict[str, Any]] = []
    all_kinematics_rows: list[dict[str, Any]] = []

    for trial, manifest_row in manifest.items():
        vicon_csv = VICON_CSV_ROOT / f"{trial}.csv"
        if not vicon_csv.exists():
            print(f"[{trial}] WARNING: CSV not found at {vicon_csv}, skipping")
            continue
        all_kinematics_rows.extend(compute_trial_kinematics(trial, manifest_row, vicon_csv))
        rows = process_trial(trial, manifest_row, KEYPOINTS_JSON, vicon_csv, plots_dir)
        all_rows.extend(rows)

    summary_path = VALIDATION_DIR / "summary.csv"
    write_summary_csv(all_rows, summary_path)
    kinematics_path = VALIDATION_DIR / "vicon_com_kinematics.csv"
    write_kinematics_csv(all_kinematics_rows, kinematics_path)
    print(f"\nSummary saved to {summary_path}")
    print(f"Vicon COM kinematics saved to {kinematics_path}")
    print(f"Plots saved to {plots_dir}")
    return 0 if all_rows and all_kinematics_rows else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
