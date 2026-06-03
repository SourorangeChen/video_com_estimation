"""COCO-17 关键点(视频)与 Vicon Trajectories 标记点的归一化对比验证脚本。

与 validate_com_normalized.py 流程一致，但对象是 17 个关键点而非整体 CoM：
对每个试验、每个关键点、每个轴(水平 x↔Vicon_Y、竖直 y↔Vicon_Z)，做
时间对齐 -> 方向对齐 -> 去线性漂移 -> z-score -> 相关性/互相关 -> lag 对齐出图。

每个 COCO 关键点通过 COCO17_TO_VICON 映射到对应的 Vicon 标记(单点或两点中点)。
"""

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

# ── 路径配置 ────────────────────────────────────────────────────────────────
VICON_CSV_ROOT = Path(r"H:\COM\video-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test")
KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json")
MANIFEST_CSV   = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_trial\Video_ViconTrial_manifest.csv")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
VIDEO_FPS      = 29.996

# ── COCO-17 → Vicon Trajectories 标记映射 ──────────────────────────────────
# 元素含义：(coco_idx, coco_name, vicon_marker)
# vicon_marker 为字符串表示单个标记，元组表示取多个标记的平均(中点)。
COCO17_TO_VICON: list[tuple[int, str, Any]] = [
    (0,  "nose",           ("LFHD", "RFHD")),   # 两个前额标记的中点
    (1,  "left_eye",       "LFHD"),
    (2,  "right_eye",      "RFHD"),
    (3,  "left_ear",       "LFHD"),
    (4,  "right_ear",      "RFHD"),
    (5,  "left_shoulder",  "LSHO"),
    (6,  "right_shoulder", "RSHO"),
    (7,  "left_elbow",     "LELB"),
    (8,  "right_elbow",    "RELB"),
    (9,  "left_wrist",     ("LWRA", "LWRB")),    # 两个腕部标记的中点
    (10, "right_wrist",    ("RWRA", "RWRB")),
    (11, "left_hip",       "LASI"),
    (12, "right_hip",      "RASI"),
    (13, "left_knee",      "LKNE"),
    (14, "right_knee",     "RKNE"),
    (15, "left_ankle",     "LANK"),
    (16, "right_ankle",    "RANK"),
]


# ── 信号处理辅助函数 ─────────────────────────────────────────────────────────
def remove_linear_drift(signal: np.ndarray) -> np.ndarray:
    """去除信号的线性趋势(基线漂移)。"""
    return detrend(signal, type="linear")


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """z-score 归一化(零均值、单位标准差)；常数信号会报错。"""
    std = signal.std()
    if std == 0.0:
        raise ValueError("zero std: signal is constant")
    return (signal - signal.mean()) / std


def compute_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """返回 (pearson_r, p_value, nrmse)；输入应为已 z-score 的信号。"""
    r, p = pearsonr(a, b)
    nrmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return float(r), float(p), nrmse


def compute_xcorr(a: np.ndarray, b: np.ndarray, fps: float = VIDEO_FPS) -> tuple[float, int, float]:
    """互相关：返回 (峰值r, 滞后帧数, 滞后毫秒)。lag<0 表示 b 领先 a。"""
    n = len(a)
    # 去均值后做全模式互相关，并用标准差归一化到 [-1,1]。
    corr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    corr /= n * a.std() * b.std()
    lags = np.arange(-(n - 1), n)
    # 取相关峰值对应的滞后。
    peak_idx = int(np.argmax(corr))
    lag_frames = int(lags[peak_idx])
    lag_ms = round(lag_frames / fps * 1000, 1)
    return round(float(corr[peak_idx]), 4), lag_frames, lag_ms


# ── 时间轴构造 ──────────────────────────────────────────────────────────────
def build_video_time_axis(frame_numbers: list[int], fps: float = VIDEO_FPS) -> np.ndarray:
    """视频帧号(从1开始) -> 相对时间(秒)，第1帧为 t=0。"""
    return np.array([(fn - 1) / fps for fn in frame_numbers])


def build_vicon_time_axis(frame_numbers: list[int], first_frame: int, rate_hz: float) -> np.ndarray:
    """Vicon 帧号 -> 相对时间(秒)，first_frame 为 t=0。"""
    return np.array([(fn - first_frame) / rate_hz for fn in frame_numbers])


def interpolate_to_video(video_t: np.ndarray, vicon_t: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """把 Vicon 数值线性插值到视频时间点上。"""
    return np.interp(video_t, vicon_t, vals)


# ── manifest 读取 ────────────────────────────────────────────────────────────
def load_manifest(manifest_csv: Path) -> dict[str, dict[str, Any]]:
    """读取各试验的时序信息(起始/结束帧、采样率)。"""
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


# ── Vicon Trajectories 解析 ─────────────────────────────────────────────────
def parse_vicon_trajectories(
    csv_path: Path,
) -> tuple[float, list[int], dict[str, tuple[list[float], list[float], list[float]]]]:
    """解析 Vicon CSV 的 Trajectories 区段。

    返回 (采样率Hz, 帧号列表, markers)，
    其中 markers 为 {标记名: (X列表, Y列表, Z列表)}。
    """
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    # 定位 Trajectories 区段标题。
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "Trajectories"),
        None,
    )
    if start is None:
        raise ValueError(f"'Trajectories' section not found in {csv_path}")

    rate_hz = float(lines[start + 1].strip())
    name_row = next(csv.reader([lines[start + 2]]))

    # 标记名出现在第 2,5,8,… 列(在 Frame 与 SubFrame 之后，每 3 列一组)。
    marker_cols: dict[str, int] = {}
    for col_idx, cell in enumerate(name_row):
        cell = cell.strip()
        if cell and ":" in cell:
            marker = cell.split(":")[-1]   # 去掉 "受试者:" 前缀
            marker_cols[marker] = col_idx

    # 为每个标记预分配 (X,Y,Z) 三个累加列表。
    markers: dict[str, tuple[list[float], list[float], list[float]]] = {
        m: ([], [], []) for m in marker_cols
    }
    frame_numbers: list[int] = []

    # 逐行读取数据帧。
    for line in lines[start + 5:]:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        # 首列非数字表示数据区结束。
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
            # 缺失值记为 NaN，后续再插值填补。
            mx.append(float(xs) if xs else float("nan"))
            my.append(float(ys) if ys else float("nan"))
            mz.append(float(zs) if zs else float("nan"))

    return rate_hz, frame_numbers, markers


# ── 视频关键点解析 ──────────────────────────────────────────────────────────
def parse_video_keypoints(
    keypoints_json: Path,
    trial_name: str,
) -> tuple[list[int], list[list[float]], list[list[float]], list[list[float]]]:
    """返回 (帧号, kp_x[17][T], kp_y[17][T], kp_conf[17][T])。

    即 17 个关键点各自的 x、y 序列与置信度序列。
    """
    folder_prefix = f"Video_{trial_name}_Trajectory"
    records: list[Any] = json.loads(keypoints_json.read_text(encoding="utf-8"))

    frame_numbers: list[int] = []
    # 17 个关键点，每个对应一个时间序列列表。
    kp_x: list[list[float]] = [[] for _ in range(17)]
    kp_y: list[list[float]] = [[] for _ in range(17)]
    kp_conf: list[list[float]] = [[] for _ in range(17)]

    for entry in records:
        if not isinstance(entry, dict):
            continue
        # 仅保留属于该试验的记录。
        image: str = entry.get("image", "")
        if not image.startswith(folder_prefix):
            continue
        kps = entry.get("keypoints")
        # 关键点必须是含至少 17 项的列表。
        if not isinstance(kps, list) or len(kps) < 17:
            continue
        stem = Path(image).stem
        try:
            frame_num = int(stem.split("_")[-1])
        except ValueError:
            continue
        frame_numbers.append(frame_num)
        # 把每个关键点的 x/y/置信度分别追加到对应序列。
        for i in range(17):
            kp_x[i].append(float(kps[i][0]))
            kp_y[i].append(float(kps[i][1]))
            kp_conf[i].append(float(kps[i][2]))

    return frame_numbers, kp_x, kp_y, kp_conf


# ── Vicon 标记 → Y/Z 数组(含 NaN 插值) ─────────────────────────────────────
def get_vicon_marker_yz(
    marker_spec: Any,
    markers: dict[str, tuple[list[float], list[float], list[float]]],
    vicon_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """获取某标记规格(单个名或多名中点)对应的 Y、Z 数组。

    缺失(NaN)样本会用有效样本做线性插值填补。
    """
    def _get_yz(name: str) -> tuple[np.ndarray, np.ndarray]:
        """取单个标记的 Y、Z 数组。"""
        if name not in markers:
            raise KeyError(f"Vicon marker '{name}' not found")
        _, my, mz = markers[name]
        return np.array(my, dtype=float), np.array(mz, dtype=float)

    def _interp_nan(arr: np.ndarray) -> np.ndarray:
        """用线性插值填补数组中的 NaN 缺口。"""
        nans = np.isnan(arr)
        # 全为 NaN 时无法插值，原样返回。
        if nans.all():
            return arr
        x = np.arange(len(arr))
        # 用非 NaN 处的值插值出 NaN 处的值。
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        return arr

    # 元组/列表表示多个标记取平均(中点)。
    if isinstance(marker_spec, (list, tuple)):
        ys, zs = zip(*[_get_yz(m) for m in marker_spec])
        y = np.mean(np.stack(ys), axis=0)
        z = np.mean(np.stack(zs), axis=0)
    else:
        y, z = _get_yz(marker_spec)

    return _interp_nan(y), _interp_nan(z)


# ── 绘图 ──────────────────────────────────────────────────────────────────────
def plot_aligned_comparison(
    t: np.ndarray,
    video_z: np.ndarray,
    vicon_z: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """绘制视频 vs Vicon 的 z 分数叠加对比图并保存。"""
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


# ── 单试验处理 ────────────────────────────────────────────────────────────────
def process_trial(
    trial: str,
    manifest_row: dict[str, Any],
    keypoints_json: Path,
    vicon_csv: Path,
    plots_root: Path,
) -> list[dict[str, Any]]:
    """对单个试验的全部关键点逐一做归一化对比，返回结果行列表。"""

    # --- 视频关键点 ---
    video_frames, vid_kp_x, vid_kp_y, vid_kp_conf = parse_video_keypoints(keypoints_json, trial)
    if not video_frames:
        print(f"[{trial}] WARNING: no video keypoint records, skipping")
        return []

    # --- Vicon 轨迹 ---
    rate_hz, vicon_frames, vicon_markers = parse_vicon_trajectories(vicon_csv)
    if not vicon_frames:
        print(f"[{trial}] WARNING: no Vicon trajectory records, skipping")
        return []

    first_frame = manifest_row["first_frame"]
    vicon_rate  = manifest_row["trajectory_rate_hz"]

    # 构造两路时间轴。
    video_t = build_video_time_axis(video_frames)
    vicon_t = build_vicon_time_axis(vicon_frames, first_frame, vicon_rate)

    # 仅保留时间重叠窗口。
    t_min = max(video_t[0], vicon_t[0])
    t_max = min(video_t[-1], vicon_t[-1])
    mask = (video_t >= t_min) & (video_t <= t_max)
    video_t_clip = video_t[mask]

    # 重叠帧太少则跳过。
    if len(video_t_clip) < 5:
        print(f"[{trial}] WARNING: fewer than 5 overlapping frames, skipping")
        return []

    # 被裁剪掉的帧占比(用于告警)。
    missing_pct = (mask.size - mask.sum()) / mask.size * 100
    vicon_t_arr = np.array(vicon_t)

    rows: list[dict[str, Any]] = []

    # 遍历每个 COCO 关键点及其对应的 Vicon 标记。
    for coco_idx, kp_name, vicon_spec in COCO17_TO_VICON:
        # --- 视频信号(裁剪到重叠窗口) ---
        vid_x_full = np.array(vid_kp_x[coco_idx])[mask]
        vid_y_full = np.array(vid_kp_y[coco_idx])[mask]

        # --- 该标记的 Vicon Y、Z ---
        try:
            vic_y_full, vic_z_full = get_vicon_marker_yz(vicon_spec, vicon_markers, vicon_t_arr)
        except KeyError as e:
            print(f"[{trial}][{kp_name}] {e}, skipping")
            continue

        # 插值到视频时间点。
        vic_y_interp = interpolate_to_video(video_t_clip, vicon_t_arr, vic_y_full)
        vic_z_interp = interpolate_to_video(video_t_clip, vicon_t_arr, vic_z_full)

        # 方向对齐(与 CoM 相同)：
        # video_x 为正 ↔ Vicon_Y 为负 → 翻转 Y
        # video_y 向下为正 ↔ Vicon_Z 向上为正 → 翻转 Z
        vic_y_aligned = -vic_y_interp
        vic_z_aligned = -vic_z_interp

        warning = f"temporal overlap clipped {missing_pct:.0f}% of frames" if missing_pct > 20 else ""

        # 分别处理水平与竖直两个轴。
        for axis_label, vid_sig, vic_sig, axis_tag in [
            ("horizontal (video_x vs vicon_Y)", vid_x_full, vic_y_aligned, "x"),
            ("vertical (video_y vs vicon_Z)",   vid_y_full, vic_z_aligned, "z"),
        ]:
            try:
                # 去线性漂移后 z-score 归一化。
                vid_z = zscore_normalize(remove_linear_drift(vid_sig))
                vic_z = zscore_normalize(remove_linear_drift(vic_sig))
            except ValueError as exc:
                print(f"[{trial}][{kp_name}][{axis_tag}] zscore failed: {exc}")
                continue

            # 相关性与互相关。
            r, p, nrmse = compute_metrics(vid_z, vic_z)
            xcorr_peak, xcorr_lag_frames, xcorr_lag_ms = compute_xcorr(vid_z, vic_z)

            # 按 xcorr 滞后裁剪信号以绘制对齐图。
            shift = abs(xcorr_lag_frames)
            if xcorr_lag_frames < 0 and shift > 0:
                # Vicon 领先：裁 Vicon 头、视频尾。
                vid_za = vid_z[:-shift]
                vic_za = vic_z[shift:]
                t_plot = video_t_clip[:-shift]
            elif xcorr_lag_frames > 0:
                # 视频领先：裁视频头、Vicon 尾。
                vid_za = vid_z[shift:]
                vic_za = vic_z[:-shift]
                t_plot = video_t_clip[shift:]
            else:
                vid_za, vic_za, t_plot = vid_z, vic_z, video_t_clip

            # 每个关键点一个子目录，每试验每轴一张图。
            plot_path = plots_root / f"kp{coco_idx:02d}_{kp_name}" / f"{trial}_{axis_tag}.png"
            plot_aligned_comparison(
                t_plot, vid_za, vic_za,
                title=f"{trial} | {kp_name} {axis_label} | xcorr_r={xcorr_peak:.3f} lag={xcorr_lag_ms:.1f}ms",
                output_path=plot_path,
            )

            # 记录该关键点该轴的指标。
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

        # 打印该关键点 x/z 两轴的概要(rows 末两行即本关键点的 x 与 z 结果)。
        print(f"[{trial}][kp{coco_idx:02d} {kp_name}] "
              f"x: r={rows[-2]['pearson_r']:.3f} xcorr={rows[-2]['xcorr_peak_r']:.3f}  "
              f"z: r={rows[-1]['pearson_r']:.3f} xcorr={rows[-1]['xcorr_peak_r']:.3f}")

    return rows


# ── 汇总 CSV ─────────────────────────────────────────────────────────────────
def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """把每试验每关键点每轴的验证结果写入 CSV。"""
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


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main() -> int:
    """遍历所有试验，输出关键点验证汇总 CSV 与每关键点对齐图。"""
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
