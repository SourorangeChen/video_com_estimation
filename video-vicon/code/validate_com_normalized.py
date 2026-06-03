"""视频质心(CoM)与 Vicon 金标准质心的归一化对比验证脚本。

核心思想：视频 CoM 是 2D 像素坐标、Vicon CoM 是 3D 毫米坐标，量纲与尺度都不同，
无法直接逐点比较。因此本脚本对每个试验、每个轴，统一做如下处理后再比较"形状/节律"：

    1. 时间对齐：把高频 Vicon(约 250Hz)线性插值到低频视频(约 30Hz)时间点；
    2. 方向对齐：翻转 Vicon 的 Y/Z 轴使其与视频坐标方向一致；
    3. 去线性漂移(detrend)：去除基线缓慢漂移；
    4. z-score 归一化：零均值、单位方差，使像素与毫米可比；
    5. 计算 Pearson r 与 nRMSE；
    6. 互相关(xcorr)寻找最优时间滞后(lag)；
    7. 按 lag 对齐后绘制叠加图。

此外还会从 Vicon CSV 计算每帧 CoM 运动学量(位移、速度、ω₀、xCoM)并导出 CSV。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
# 使用无界面后端，便于批量出图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import detrend

# 本研究项目在 Windows(H:\COM)下的机器相关路径。
VICON_CSV_ROOT = Path(r"H:\COM\video-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test")
KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints_and_com.json")
MANIFEST_CSV   = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_trial\Video_ViconTrial_manifest.csv")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
VIDEO_FPS      = 29.996      # 视频帧率(Hz)
GRAVITY_M_S2   = 9.81        # 重力加速度(m/s²)，用于计算倒立摆自然频率 ω₀


def remove_linear_drift(signal: np.ndarray) -> np.ndarray:
    """去除信号的线性趋势(零漂)。"""
    return detrend(signal, type="linear")


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """对信号做 z-score 归一化(零均值、单位标准差)。"""
    std = signal.std()
    # 常数信号标准差为 0，无法归一化。
    if std == 0.0:
        raise ValueError("zero std: signal is constant, cannot z-score normalize")
    return (signal - signal.mean()) / std


def compute_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """计算两路信号的 (pearson_r, p_value, nrmse)。

    输入应为已 z-score 归一化的信号；在 z 分数上算的 RMSE 是无量纲的(以标准差为单位)，
    即归一化 RMSE(nRMSE)。
    """
    # 皮尔逊相关系数与显著性 p 值。
    r, p = pearsonr(a, b)
    # 归一化均方根误差。
    nrmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return float(r), float(p), nrmse


def compute_xcorr(a: np.ndarray, b: np.ndarray, fps: float = VIDEO_FPS) -> tuple[float, int, float]:
    """互相关：返回 (峰值相关 r, 滞后帧数, 滞后毫秒)。

    lag_frames > 0 表示 b 领先 a(b 比 a 更早发生)。
    """
    n = len(a)
    # 去均值后做全模式互相关。
    corr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    # 用样本数与两信号标准差归一化，使相关值落在 [-1, 1]。
    corr /= n * a.std() * b.std()
    # 对应每个相关值的滞后量(从 -(n-1) 到 n-1)。
    lags = np.arange(-(n - 1), n)
    # 取相关最大处作为最优滞后。
    peak_idx = int(np.argmax(corr))
    lag_frames = int(lags[peak_idx])
    # 把滞后帧数换算为毫秒。
    lag_ms = round(lag_frames / fps * 1000, 1)
    return round(float(corr[peak_idx]), 4), lag_frames, lag_ms


def parse_vicon_model_outputs(
    csv_path: Path,
) -> tuple[float, list[int], list[float], list[float], list[float]]:
    """从 Vicon CSV 导出文件的 'Model Outputs' 区段解析 CentreOfMass(单位 mm)。"""
    # utf-8-sig 兼容 BOM。
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    # 定位 "Model Outputs" 区段标题行。
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "Model Outputs"),
        None,
    )
    if start is None:
        raise ValueError(f"'Model Outputs' section not found in {csv_path}")

    # 区段结构：+1 采样率，+2 列名行，+4 单位行，+5 起 数据行。
    rate_hz = float(lines[start + 1].strip())
    names = next(csv.reader([lines[start + 2]]))
    units = next(csv.reader([lines[start + 4]]))

    # 找到列名以 ":CentreOfMass" 结尾的那一列(其后连续 3 列为 X/Y/Z)。
    com_col = next(
        (i for i, name in enumerate(names) if name.strip().endswith(":CentreOfMass")),
        None,
    )
    if com_col is None:
        raise ValueError(f"':CentreOfMass' column not found in Model Outputs of {csv_path}")

    # 校验 CentreOfMass 单位必须为 mm(Vicon Model Outputs 应始终以 mm 导出)。
    com_unit = units[com_col].strip() if com_col < len(units) else ""
    if com_unit and com_unit.lower() != "mm":
        raise ValueError(f"Expected CentreOfMass unit 'mm', got '{com_unit}' in {csv_path}")

    frame_numbers: list[int] = []
    com_x: list[float] = []
    com_y: list[float] = []
    com_z: list[float] = []

    # 逐行读取数据。
    for line in lines[start + 5:]:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        # 首列非数字表示数据区结束(进入下一区段)。
        if not row or not row[0].strip().isdigit():
            break
        try:
            # 取 CoM 的连续三列。
            x_str = row[com_col].strip()
            y_str = row[com_col + 1].strip()
            z_str = row[com_col + 2].strip()
        except IndexError:
            continue
        # 任一坐标缺失则跳过该帧。
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
    """从 keypoints.json 中读取某试验的视频 CoM，返回 (帧号, com_x_px, com_y_px)。"""
    # 该试验图片路径的前缀。
    folder_prefix = f"Video_{trial_name}_Trajectory"
    records: list[Any] = json.loads(keypoints_json.read_text(encoding="utf-8"))

    frame_numbers: list[int] = []
    com_x: list[float] = []
    com_y: list[float] = []

    for entry in records:
        if not isinstance(entry, dict):
            continue
        # 仅保留属于该试验的记录。
        image: str = entry.get("image", "")
        if not image.startswith(folder_prefix):
            continue
        com = entry.get("com")
        if not isinstance(com, dict):
            continue
        # 从文件名(如 "frame_000001")末尾解析帧号。
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
    """从 manifest CSV 读取各试验的时序信息，返回以试验名为键的字典。"""
    result: dict[str, dict[str, Any]] = {}
    with manifest_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial = row["trial"].strip()
            result[trial] = {
                "first_frame": int(row["first_frame"]),               # Vicon 该试验起始帧号
                "last_frame": int(row["last_frame"]),                 # Vicon 该试验结束帧号
                "trajectory_rate_hz": float(row["trajectory_rate_hz"]),  # 轨迹采样率
            }
    return result


def build_video_time_axis(frame_numbers: list[int], fps: float = VIDEO_FPS) -> np.ndarray:
    """把视频帧号(从 1 开始)转换为相对时间(秒)，第 1 帧对应 t=0。"""
    return np.array([(fn - 1) / fps for fn in frame_numbers])


def build_vicon_time_axis(
    frame_numbers: list[int], first_frame: int, rate_hz: float
) -> np.ndarray:
    """把 Vicon 帧号转换为相对时间(秒)，first_frame 对应 t=0。"""
    return np.array([(fn - first_frame) / rate_hz for fn in frame_numbers])


def interpolate_vicon_to_video(
    video_t: np.ndarray, vicon_t: np.ndarray, vicon_vals: np.ndarray
) -> np.ndarray:
    """把 Vicon 数值线性插值到视频时间点上。"""
    # 当前 z-score 验证采用旧流程：把高频 Vicon 信号重采样到低频视频时间点。
    # vicon_t 必须单调递增(np.interp 的要求)。
    return np.interp(video_t, vicon_t, vicon_vals)


def compute_com_kinematics(
    trial: str,
    frame_numbers: list[int],
    time_s: np.ndarray,
    com_x_mm: np.ndarray,
    com_y_mm: np.ndarray,
    com_z_mm: np.ndarray,
) -> list[dict[str, Any]]:
    """计算 Vicon CoM 的位移、速度与 xCoM。

    Vicon 以毫米导出 CoM，本函数运动学输出统一用米和秒。
    位移为逐帧差分 r(t_i) - r(t_{i-1})，首帧为零。
    倒立摆长度 l(t) 取每帧 CoM 高度，即 com_z_m。
    """
    # 各数组长度必须一致。
    if not (len(frame_numbers) == len(time_s) == len(com_x_mm) == len(com_y_mm) == len(com_z_mm)):
        raise ValueError("frame, time, and CoM arrays must have the same length")
    # 至少两帧才能算速度。
    if len(frame_numbers) < 2:
        raise ValueError("at least two frames are required to compute velocity")

    # 堆叠为 (N,3) 并由毫米转米。
    com_m = np.column_stack([com_x_mm, com_y_mm, com_z_mm]).astype(float) / 1000.0
    # xCoM 需要正的 CoM 高度(z>0)。
    if np.any(com_m[:, 2] <= 0.0):
        raise ValueError("xCoM requires positive CoM height for every frame")

    # 逐帧位移：首帧补零，其余为相邻帧差分。
    displacement = np.vstack([np.zeros(3), np.diff(com_m, axis=0)])
    displacement_mag = np.linalg.norm(displacement, axis=1)
    # 速度用 np.gradient(对非均匀时间也适用)。
    velocity = np.gradient(com_m, time_s, axis=0)
    velocity_mag = np.linalg.norm(velocity, axis=1)
    # 倒立摆自然频率 ω₀ = sqrt(g / 高度)。
    omega0 = np.sqrt(GRAVITY_M_S2 / com_m[:, 2])
    # 外推质心 xCoM = CoM + 速度 / ω₀(Hof 稳定裕度模型)。
    xcom = com_m + velocity / omega0[:, np.newaxis]

    # 把逐帧结果组织为字典列表。
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
    """计算单个试验的 Vicon CoM 运动学量。"""
    # 解析该试验的 Vicon CoM。
    _, vicon_frames, vicon_cx, vicon_cy, vicon_cz = parse_vicon_model_outputs(vicon_csv)
    if not vicon_frames:
        print(f"[{trial}] WARNING: no Vicon COM records found for kinematics, skipping")
        return []

    # 构造 Vicon 时间轴。
    time_s = build_vicon_time_axis(
        vicon_frames,
        manifest_row["first_frame"],
        manifest_row["trajectory_rate_hz"],
    )
    # 计算并返回逐帧运动学。
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
    """绘制视频 vs Vicon CoM 的 z 分数叠加对比图并保存。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    # 视频(蓝)与 Vicon(红)两条 z 分数曲线。
    ax.plot(t, video_z, color="tab:blue", linewidth=1.2, label="video (z-score)")
    ax.plot(t, vicon_z, color="tab:red", linewidth=1.2, label="vicon (z-score)", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    # 确保目录存在后保存并关闭。
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
    """绘制互相关函数曲线并标注峰值位置。"""
    n = len(a)
    # 重新计算互相关曲线(与 compute_xcorr 一致)。
    corr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    corr /= n * a.std() * b.std()
    lags = np.arange(-(n - 1), n)
    # 把滞后量换算为毫秒作为横轴。
    lags_ms = lags / fps * 1000

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags_ms, corr, color="tab:purple", linewidth=1.2)
    # 竖虚线标注峰值滞后。
    ax.axvline(lag_ms, color="tab:orange", linewidth=1.2, linestyle="--",
               label=f"peak lag = {lag_ms:.1f} ms ({lag_frames} frames)")
    # 在峰值处画点。
    ax.scatter([lag_ms], [peak_r], color="tab:orange", zorder=5, s=60)
    # 零滞后参考线。
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
    """把验证汇总结果写入 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 固定列顺序。
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
    """把每帧 Vicon CoM 运动学结果写入 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 固定列顺序。
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
    """对单个试验做归一化 CoM 对比，返回结果行列表。"""
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

    # 分别构造视频与 Vicon 的时间轴。
    video_t = build_video_time_axis(video_frames)
    vicon_t = build_vicon_time_axis(vicon_frames, first_frame, rate_hz)

    # 仅保留两者时间重叠的窗口。
    t_min = max(video_t[0], vicon_t[0])
    t_max = min(video_t[-1], vicon_t[-1])
    mask = (video_t >= t_min) & (video_t <= t_max)
    video_t = video_t[mask]
    video_cx_arr = np.array(video_cx)[mask]
    video_cy_arr = np.array(video_cy)[mask]

    # 重叠帧太少则跳过该试验。
    if len(video_t) < 5:
        print(f"[{trial}] WARNING: fewer than 5 overlapping frames, skipping")
        return []

    vicon_cy_arr = np.array(vicon_cy)
    vicon_cz_arr = np.array(vicon_cz)
    vicon_t_arr = np.array(vicon_t)

    # 时间对齐：Vicon 250 Hz -> video 约 30 Hz，插值后两路信号长度等于视频帧数。
    vicon_y_interp = interpolate_vicon_to_video(video_t, vicon_t_arr, vicon_cy_arr)
    vicon_z_interp = interpolate_vicon_to_video(video_t, vicon_t_arr, vicon_cz_arr)

    # 坐标方向对齐：
    # - video x 与 Vicon Y 正方向相反，所以翻转 Y；
    # - video y 像素坐标向下为正，Vicon Z 向上为正，所以翻转 Z。
    vicon_y_aligned = -vicon_y_interp
    vicon_z_aligned = -vicon_z_interp

    rows: list[dict[str, Any]] = []
    # 被时间重叠裁剪掉的帧占比(用于告警)。
    missing_pct = (mask.size - mask.sum()) / mask.size * 100

    # 分别处理水平轴(x↔Y)与竖直轴(y↔Z)两组信号。
    for axis_label, vid_sig, vic_sig, plot_suffix in [
        ("horizontal (video_x vs vicon_Y)", video_cx_arr, vicon_y_aligned, "x"),
        ("vertical (video_y vs vicon_Z)", video_cy_arr, vicon_z_aligned, "z"),
    ]:
        # 裁剪比例过高时给出告警。
        warning = f"temporal overlap clipped {missing_pct:.0f}% of frames" if missing_pct > 20 else ""

        try:
            # 先去掉线性漂移，再做 z-score；后续比较形状/节律而不是绝对数值。
            vid_z = zscore_normalize(remove_linear_drift(vid_sig))
            vic_z = zscore_normalize(remove_linear_drift(vic_sig))
        except ValueError as exc:
            print(f"[{trial}][{axis_label}] zscore failed: {exc}")
            continue

        # 计算相关性指标与互相关滞后。
        r, p, nrmse = compute_metrics(vid_z, vic_z)
        xcorr_peak, xcorr_lag_frames, xcorr_lag_ms = compute_xcorr(vid_z, vic_z)

        # 根据 xcorr 找到的 lag 裁剪两路信号，使图中显示的是 lag 对齐后的叠加结果。
        shift = abs(xcorr_lag_frames)
        if xcorr_lag_frames < 0:
            # Vicon 领先视频：裁掉 Vicon 开头、视频结尾以对齐。
            vid_z_aligned = vid_z[:-shift] if shift > 0 else vid_z
            vic_z_aligned = vic_z[shift:] if shift > 0 else vic_z
            t_aligned = video_t[:-shift] if shift > 0 else video_t
        elif xcorr_lag_frames > 0:
            # 视频领先 Vicon：裁掉视频开头、Vicon 结尾以对齐。
            vid_z_aligned = vid_z[shift:]
            vic_z_aligned = vic_z[:-shift]
            t_aligned = video_t[shift:]
        else:
            # 无滞后，原样使用。
            vid_z_aligned, vic_z_aligned, t_aligned = vid_z, vic_z, video_t

        # 绘制并保存 lag 对齐后的叠加图。
        aligned_plot_path = plots_dir / f"{trial}_{plot_suffix}.png"
        plot_comparison(
            t_aligned,
            vid_z_aligned,
            vic_z_aligned,
            title=f"{trial} — {axis_label} | xcorr_r={xcorr_peak:.3f} lag={xcorr_lag_ms:.1f}ms",
            output_path=aligned_plot_path,
        )

        # 汇总该轴的指标。
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
    """主流程：遍历 manifest 中所有试验，输出验证汇总、运动学 CSV 与对齐图。"""
    manifest = load_manifest(MANIFEST_CSV)
    plots_dir = VALIDATION_DIR / "plots_aligned"
    all_rows: list[dict[str, Any]] = []
    all_kinematics_rows: list[dict[str, Any]] = []

    # 逐个试验处理。
    for trial, manifest_row in manifest.items():
        vicon_csv = VICON_CSV_ROOT / f"{trial}.csv"
        # 缺少对应 Vicon CSV 则跳过。
        if not vicon_csv.exists():
            print(f"[{trial}] WARNING: CSV not found at {vicon_csv}, skipping")
            continue
        # 累加该试验的 Vicon 运动学与 CoM 对比结果。
        all_kinematics_rows.extend(compute_trial_kinematics(trial, manifest_row, vicon_csv))
        rows = process_trial(trial, manifest_row, KEYPOINTS_JSON, vicon_csv, plots_dir)
        all_rows.extend(rows)

    # 写出汇总 CSV 与运动学 CSV。
    summary_path = VALIDATION_DIR / "summary.csv"
    write_summary_csv(all_rows, summary_path)
    kinematics_path = VALIDATION_DIR / "vicon_com_kinematics.csv"
    write_kinematics_csv(all_kinematics_rows, kinematics_path)
    print(f"\nSummary saved to {summary_path}")
    print(f"Vicon COM kinematics saved to {kinematics_path}")
    print(f"Plots saved to {plots_dir}")
    # 两类结果都有才算成功。
    return 0 if all_rows and all_kinematics_rows else 1


if __name__ == "__main__":
    import sys
    # 以 main() 返回值作为进程退出码。
    raise SystemExit(main())
