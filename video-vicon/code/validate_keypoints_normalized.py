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

# ── Paths ──────────────────────────────────────────────────────────────────
VICON_CSV_ROOT = Path(r"H:\COM\video-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test")
KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json")
MANIFEST_CSV   = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_trial\Video_ViconTrial_manifest.csv")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
VIDEO_FPS      = 29.996

# ── COCO-17 → Vicon Trajectories marker mapping ────────────────────────────
# Tuple element: (coco_idx, coco_name, vicon_marker)
# vicon_marker can be a str (single marker) or tuple of str (average them).
COCO17_TO_VICON: list[tuple[int, str, Any]] = [
    (0,  "nose",           ("LFHD", "RFHD")),   # midpoint of two front-head markers
    (1,  "left_eye",       "LFHD"),
    (2,  "right_eye",      "RFHD"),
    (3,  "left_ear",       "LFHD"),
    (4,  "right_ear",      "RFHD"),
    (5,  "left_shoulder",  "LSHO"),
    (6,  "right_shoulder", "RSHO"),
    (7,  "left_elbow",     "LELB"),
    (8,  "right_elbow",    "RELB"),
    (9,  "left_wrist",     ("LWRA", "LWRB")),    # midpoint of two wrist markers
    (10, "right_wrist",    ("RWRA", "RWRB")),
    (11, "left_hip",       "LASI"),
    (12, "right_hip",      "RASI"),
    (13, "left_knee",      "LKNE"),
    (14, "right_knee",     "RKNE"),
    (15, "left_ankle",     "LANK"),
    (16, "right_ankle",    "RANK"),
]


# ── Signal processing helpers ───────────────────────────────────────────────
def remove_linear_drift(signal: np.ndarray) -> np.ndarray:
    return detrend(signal, type="linear")


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    std = signal.std()
    if std == 0.0:
        raise ValueError("zero std: signal is constant")
    return (signal - signal.mean()) / std


def compute_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    r, p = pearsonr(a, b)
    nrmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return float(r), float(p), nrmse


def compute_xcorr(a: np.ndarray, b: np.ndarray, fps: float = VIDEO_FPS) -> tuple[float, int, float]:
    """Returns (peak_r, lag_frames, lag_ms). lag<0 means b leads a."""
    n = len(a)
    corr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    corr /= n * a.std() * b.std()
    lags = np.arange(-(n - 1), n)
    peak_idx = int(np.argmax(corr))
    lag_frames = int(lags[peak_idx])
    lag_ms = round(lag_frames / fps * 1000, 1)
    return round(float(corr[peak_idx]), 4), lag_frames, lag_ms


# ── Time axis builders ──────────────────────────────────────────────────────
def build_video_time_axis(frame_numbers: list[int], fps: float = VIDEO_FPS) -> np.ndarray:
    return np.array([(fn - 1) / fps for fn in frame_numbers])


def build_vicon_time_axis(frame_numbers: list[int], first_frame: int, rate_hz: float) -> np.ndarray:
    return np.array([(fn - first_frame) / rate_hz for fn in frame_numbers])


def interpolate_to_video(video_t: np.ndarray, vicon_t: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return np.interp(video_t, vicon_t, vals)


# ── Manifest ────────────────────────────────────────────────────────────────
def load_manifest(manifest_csv: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with manifest_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trial = row["trial"].strip()
            result[trial] = {
                "first_frame": int(row["first_frame"]),
                "last_frame":  int(row["last_frame"]),
                "trajectory_rate_hz": float(row["trajectory_rate_hz"]),
            }
    return result


# ── Vicon Trajectories parser ───────────────────────────────────────────────
def parse_vicon_trajectories(
    csv_path: Path,
) -> tuple[float, list[int], dict[str, tuple[list[float], list[float], list[float]]]]:
    """Parse the Trajectories section of a Vicon CSV.

    Returns (rate_hz, frame_numbers, markers)
    where markers maps marker_name → (X_list, Y_list, Z_list).
    """
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "Trajectories"),
        None,
    )
    if start is None:
        raise ValueError(f"'Trajectories' section not found in {csv_path}")

    rate_hz = float(lines[start + 1].strip())
    name_row = next(csv.reader([lines[start + 2]]))

    # Marker names appear at columns 2, 5, 8, … (every 3 cols, after Frame & SubFrame)
    marker_cols: dict[str, int] = {}
    for col_idx, cell in enumerate(name_row):
        cell = cell.strip()
        if cell and ":" in cell:
            marker = cell.split(":")[-1]   # strip subject prefix
            marker_cols[marker] = col_idx

    # Pre-allocate accumulators
    markers: dict[str, tuple[list[float], list[float], list[float]]] = {
        m: ([], [], []) for m in marker_cols
    }
    frame_numbers: list[int] = []

    for line in lines[start + 5:]:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if not row or not row[0].strip().isdigit():
            break
        frame_numbers.append(int(row[0]))
        for marker, col in marker_cols.items():
            try:
                xs = row[col].strip()
                ys = row[col + 1].strip()
                zs = row[col + 2].strip()
            except IndexError:
                xs = ys = zs = ""
            mx, my, mz = markers[marker]
            mx.append(float(xs) if xs else float("nan"))
            my.append(float(ys) if ys else float("nan"))
            mz.append(float(zs) if zs else float("nan"))

    return rate_hz, frame_numbers, markers


# ── Video keypoints parser ──────────────────────────────────────────────────
def parse_video_keypoints(
    keypoints_json: Path,
    trial_name: str,
) -> tuple[list[int], list[list[float]], list[list[float]], list[list[float]]]:
    """Return (frame_numbers, kp_x[17][T], kp_y[17][T], kp_conf[17][T])."""
    folder_prefix = f"Video_{trial_name}_Trajectory"
    records: list[Any] = json.loads(keypoints_json.read_text(encoding="utf-8"))

    frame_numbers: list[int] = []
    kp_x: list[list[float]] = [[] for _ in range(17)]
    kp_y: list[list[float]] = [[] for _ in range(17)]
    kp_conf: list[list[float]] = [[] for _ in range(17)]

    for entry in records:
        if not isinstance(entry, dict):
            continue
        image: str = entry.get("image", "")
        if not image.startswith(folder_prefix):
            continue
        kps = entry.get("keypoints")
        if not isinstance(kps, list) or len(kps) < 17:
            continue
        stem = Path(image).stem
        try:
            frame_num = int(stem.split("_")[-1])
        except ValueError:
            continue
        frame_numbers.append(frame_num)
        for i in range(17):
            kp_x[i].append(float(kps[i][0]))
            kp_y[i].append(float(kps[i][1]))
            kp_conf[i].append(float(kps[i][2]))

    return frame_numbers, kp_x, kp_y, kp_conf


# ── Vicon marker → Y/Z arrays (with NaN interpolation) ─────────────────────
def get_vicon_marker_yz(
    marker_spec: Any,
    markers: dict[str, tuple[list[float], list[float], list[float]]],
    vicon_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get Y and Z arrays for a marker spec (str or tuple for midpoint).
    NaN gaps are linearly interpolated over valid samples.
    """
    def _get_yz(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in markers:
            raise KeyError(f"Vicon marker '{name}' not found")
        _, my, mz = markers[name]
        return np.array(my, dtype=float), np.array(mz, dtype=float)

    def _interp_nan(arr: np.ndarray) -> np.ndarray:
        nans = np.isnan(arr)
        if nans.all():
            return arr
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        return arr

    if isinstance(marker_spec, (list, tuple)):
        ys, zs = zip(*[_get_yz(m) for m in marker_spec])
        y = np.mean(np.stack(ys), axis=0)
        z = np.mean(np.stack(zs), axis=0)
    else:
        y, z = _get_yz(marker_spec)

    return _interp_nan(y), _interp_nan(z)


# ── Plot ────────────────────────────────────────────────────────────────────
def plot_aligned_comparison(
    t: np.ndarray,
    video_z: np.ndarray,
    vicon_z: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, video_z, color="tab:blue", linewidth=1.2, label="video (z-score)")
    ax.plot(t, vicon_z, color="tab:red",  linewidth=1.2, label="vicon (z-score)", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Per-trial processing ────────────────────────────────────────────────────
def process_trial(
    trial: str,
    manifest_row: dict[str, Any],
    keypoints_json: Path,
    vicon_csv: Path,
    plots_root: Path,
) -> list[dict[str, Any]]:

    # --- Video keypoints ---
    video_frames, vid_kp_x, vid_kp_y, vid_kp_conf = parse_video_keypoints(keypoints_json, trial)
    if not video_frames:
        print(f"[{trial}] WARNING: no video keypoint records, skipping")
        return []

    # --- Vicon trajectories ---
    rate_hz, vicon_frames, vicon_markers = parse_vicon_trajectories(vicon_csv)
    if not vicon_frames:
        print(f"[{trial}] WARNING: no Vicon trajectory records, skipping")
        return []

    first_frame = manifest_row["first_frame"]
    vicon_rate  = manifest_row["trajectory_rate_hz"]

    video_t = build_video_time_axis(video_frames)
    vicon_t = build_vicon_time_axis(vicon_frames, first_frame, vicon_rate)

    # Overlapping time window
    t_min = max(video_t[0], vicon_t[0])
    t_max = min(video_t[-1], vicon_t[-1])
    mask = (video_t >= t_min) & (video_t <= t_max)
    video_t_clip = video_t[mask]

    if len(video_t_clip) < 5:
        print(f"[{trial}] WARNING: fewer than 5 overlapping frames, skipping")
        return []

    missing_pct = (mask.size - mask.sum()) / mask.size * 100
    vicon_t_arr = np.array(vicon_t)

    rows: list[dict[str, Any]] = []

    for coco_idx, kp_name, vicon_spec in COCO17_TO_VICON:
        # --- Video signal ---
        vid_x_full = np.array(vid_kp_x[coco_idx])[mask]
        vid_y_full = np.array(vid_kp_y[coco_idx])[mask]

        # --- Vicon Y and Z for this marker ---
        try:
            vic_y_full, vic_z_full = get_vicon_marker_yz(vicon_spec, vicon_markers, vicon_t_arr)
        except KeyError as e:
            print(f"[{trial}][{kp_name}] {e}, skipping")
            continue

        vic_y_interp = interpolate_to_video(video_t_clip, vicon_t_arr, vic_y_full)
        vic_z_interp = interpolate_to_video(video_t_clip, vicon_t_arr, vic_z_full)

        # Sign alignment (same as COM):
        # video_x positive ↔ Vicon_Y negative → flip Y
        # video_y increases downward ↔ Vicon_Z increases upward → flip Z
        vic_y_aligned = -vic_y_interp
        vic_z_aligned = -vic_z_interp

        warning = f"temporal overlap clipped {missing_pct:.0f}% of frames" if missing_pct > 20 else ""

        for axis_label, vid_sig, vic_sig, axis_tag in [
            ("horizontal (video_x vs vicon_Y)", vid_x_full, vic_y_aligned, "x"),
            ("vertical (video_y vs vicon_Z)",   vid_y_full, vic_z_aligned, "z"),
        ]:
            try:
                vid_z = zscore_normalize(remove_linear_drift(vid_sig))
                vic_z = zscore_normalize(remove_linear_drift(vic_sig))
            except ValueError as exc:
                print(f"[{trial}][{kp_name}][{axis_tag}] zscore failed: {exc}")
                continue

            r, p, nrmse = compute_metrics(vid_z, vic_z)
            xcorr_peak, xcorr_lag_frames, xcorr_lag_ms = compute_xcorr(vid_z, vic_z)

            # Lag-aligned plot
            shift = abs(xcorr_lag_frames)
            if xcorr_lag_frames < 0 and shift > 0:
                vid_za = vid_z[:-shift]
                vic_za = vic_z[shift:]
                t_plot = video_t_clip[:-shift]
            elif xcorr_lag_frames > 0:
                vid_za = vid_z[shift:]
                vic_za = vic_z[:-shift]
                t_plot = video_t_clip[shift:]
            else:
                vid_za, vic_za, t_plot = vid_z, vic_z, video_t_clip

            plot_path = plots_root / f"kp{coco_idx:02d}_{kp_name}" / f"{trial}_{axis_tag}.png"
            plot_aligned_comparison(
                t_plot, vid_za, vic_za,
                title=f"{trial} | {kp_name} {axis_label} | xcorr_r={xcorr_peak:.3f} lag={xcorr_lag_ms:.1f}ms",
                output_path=plot_path,
            )

            rows.append({
                "trial":             trial,
                "kp_idx":            coco_idx,
                "kp_name":           kp_name,
                "vicon_marker":      str(vicon_spec),
                "axis":              axis_label,
                "pearson_r":         round(r, 4),
                "p_value":           f"{p:.4e}",
                "nrmse":             round(nrmse, 4),
                "n_frames":          int(len(video_t_clip)),
                "xcorr_peak_r":      xcorr_peak,
                "xcorr_lag_frames":  xcorr_lag_frames,
                "xcorr_lag_ms":      xcorr_lag_ms,
                "warning":           warning,
            })

        print(f"[{trial}][kp{coco_idx:02d} {kp_name}] "
              f"x: r={rows[-2]['pearson_r']:.3f} xcorr={rows[-2]['xcorr_peak_r']:.3f}  "
              f"z: r={rows[-1]['pearson_r']:.3f} xcorr={rows[-1]['xcorr_peak_r']:.3f}")

    return rows


# ── Summary CSV ─────────────────────────────────────────────────────────────
def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "kp_idx", "kp_name", "vicon_marker", "axis",
        "pearson_r", "p_value", "nrmse", "n_frames",
        "xcorr_peak_r", "xcorr_lag_frames", "xcorr_lag_ms", "warning",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    manifest  = load_manifest(MANIFEST_CSV)
    plots_root = VALIDATION_DIR / "keypoint_plots"
    all_rows: list[dict[str, Any]] = []

    for trial, manifest_row in manifest.items():
        vicon_csv = VICON_CSV_ROOT / f"{trial}.csv"
        if not vicon_csv.exists():
            print(f"[{trial}] WARNING: CSV not found, skipping")
            continue
        rows = process_trial(trial, manifest_row, KEYPOINTS_JSON, vicon_csv, plots_root)
        all_rows.extend(rows)

    summary_path = VALIDATION_DIR / "keypoint_summary.csv"
    write_summary_csv(all_rows, summary_path)
    print(f"\nSummary → {summary_path}")
    print(f"Plots   → {plots_root}")
    return 0 if all_rows else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
