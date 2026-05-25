# COM 归一化验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写 `validate_com_normalized.py`，将视频估算 COM 与 Vicon 金标准 COM 做 z-score 归一化后对比，输出每个试验的叠加曲线图和汇总统计 CSV。

**Architecture:** 脚本分为独立的纯函数模块（解析、对齐、归一化、指标、绘图）和一个 `main()` 入口。所有模块均有单元测试。视频 COM 已预计算在 `keypoints.json` 的 `"com"` 字段中，无需重算。Vicon COM 从 CSV `Model Outputs` 部分的 `*:CentreOfMass` 列读取。

**Tech Stack:** Python 3.x, numpy, scipy, matplotlib，标准库 csv/json/pathlib

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `video-vicon/code/validate_com_normalized.py` | 主脚本，含所有模块函数和 `main()` |
| `video-vicon/code/tests/test_validate_com_normalized.py` | 单元测试 |
| `video-vicon/validation/plots/<trial>_x.png` | 水平轴叠加曲线图（运行时生成） |
| `video-vicon/validation/plots/<trial>_z.png` | 垂直轴叠加曲线图（运行时生成） |
| `video-vicon/validation/summary.csv` | 汇总统计表（运行时生成） |

---

## 数据路径常量（供脚本使用）

```python
VICON_CSV_ROOT = Path(r"H:\COM\video-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test")
KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints.json")
MANIFEST_CSV   = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_trial\Video_ViconTrial_manifest.csv")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
VIDEO_FPS      = 29.996
```

---

## Task 1: 核心数学工具函数

**Files:**

- Create: `video-vicon/code/validate_com_normalized.py`
- Create: `video-vicon/code/tests/test_validate_com_normalized.py`

- [ ] **Step 1: 写失败测试**

创建 `H:\COM\video-vicon\code\tests\test_validate_com_normalized.py`：

```python
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validate_com_normalized import zscore_normalize, compute_metrics


def test_zscore_normalize_basic():
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore_normalize(sig)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std() - 1.0) < 1e-10


def test_zscore_normalize_constant_raises():
    sig = np.array([3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="zero std"):
        zscore_normalize(sig)


def test_compute_metrics_perfect_match():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r, p, nrmse = compute_metrics(a, a)
    assert abs(r - 1.0) < 1e-10
    assert abs(nrmse) < 1e-10


def test_compute_metrics_anti_correlation():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = -a
    r, p, nrmse = compute_metrics(a, b)
    assert r < -0.99


def test_compute_metrics_known_nrmse():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    r, p, nrmse = compute_metrics(a, b)
    assert abs(nrmse - 1.0) < 1e-10
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd H:\COM\video-vicon\code
python -m pytest tests/test_validate_com_normalized.py::test_zscore_normalize_basic -v
```

期望：`ImportError: No module named 'validate_com_normalized'`

- [ ] **Step 3: 实现 zscore_normalize 和 compute_metrics**

创建 `H:\COM\video-vicon\code\validate_com_normalized.py`：

```python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr

VICON_CSV_ROOT = Path(r"H:\COM\video-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test")
KEYPOINTS_JSON = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints.json")
MANIFEST_CSV   = Path(r"H:\COM\video-vicon\data\Chenzixuan\Video\Video_trial\Video_ViconTrial_manifest.csv")
VALIDATION_DIR = Path(r"H:\COM\video-vicon\validation")
VIDEO_FPS      = 29.996


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    std = signal.std()
    if std == 0.0:
        raise ValueError("zero std: signal is constant, cannot z-score normalize")
    return (signal - signal.mean()) / std


def compute_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    r, p = pearsonr(a, b)
    nrmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return float(r), float(p), nrmse
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_validate_com_normalized.py -v
```

期望：5 passed

- [ ] **Step 5: Commit**

```bash
git add video-vicon/code/validate_com_normalized.py video-vicon/code/tests/test_validate_com_normalized.py
git commit -m "feat: add zscore_normalize and compute_metrics with tests"
```

---

## Task 2: Vicon CSV 解析

**Files:**
- Modify: `video-vicon/code/validate_com_normalized.py`
- Modify: `video-vicon/code/tests/test_validate_com_normalized.py`

Vicon CSV `Model Outputs` 结构（实测 WT02.csv）：
- 第 0 行：`Model Outputs`
- 第 1 行：`250`（采样率）
- 第 2 行：名称行，含 `Chenzixuan:CentreOfMass` 在列 2（X）、3（Y）、4（Z）
- 第 3 行：轴名称行（X,Y,Z,...）
- 第 4 行：单位行（mm,mm,mm,...）
- 第 5+ 行：数据行 `frame_num,0,X,Y,Z,...`（某些帧的 COM 列可能为空，需跳过）

- [ ] **Step 1: 写失败测试**

在 `test_validate_com_normalized.py` 末尾追加：

```python
def test_parse_vicon_model_outputs_basic():
    csv_text = "\n\nModel Outputs\n250\n,,Subject:CentreOfMass,,,\nFrame,Sub Frame,X,Y,Z\n,,mm,mm,mm\n1,0,100.0,200.0,300.0\n2,0,101.0,201.0,301.0\n3,0,,,\n4,0,103.0,203.0,303.0\n"
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_text)
        tmp = f.name
    try:
        from validate_com_normalized import parse_vicon_model_outputs
        rate, frames, cx, cy, cz = parse_vicon_model_outputs(Path(tmp))
        assert rate == 250.0
        assert frames == [1, 2, 4]
        assert abs(cx[0] - 100.0) < 1e-6
        assert abs(cy[1] - 201.0) < 1e-6
        assert abs(cz[2] - 303.0) < 1e-6
    finally:
        os.unlink(tmp)


def test_parse_vicon_model_outputs_missing_section():
    from validate_com_normalized import parse_vicon_model_outputs
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("no model outputs here\n")
        tmp = f.name
    try:
        with pytest.raises(ValueError, match="Model Outputs"):
            parse_vicon_model_outputs(Path(tmp))
    finally:
        os.unlink(tmp)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_validate_com_normalized.py::test_parse_vicon_model_outputs_basic -v
```

期望：`ImportError: cannot import name 'parse_vicon_model_outputs'`

- [ ] **Step 3: 实现 parse_vicon_model_outputs**

在 `validate_com_normalized.py` 中追加：

```python
def parse_vicon_model_outputs(
    csv_path: Path,
) -> tuple[float, list[int], list[float], list[float], list[float]]:
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "Model Outputs"),
        None,
    )
    if start is None:
        raise ValueError(f"'Model Outputs' section not found in {csv_path}")

    rate_hz = float(lines[start + 1].strip())
    names = next(csv.reader([lines[start + 2]]))

    com_col = next(
        (i for i, name in enumerate(names) if name.strip().endswith(":CentreOfMass")),
        None,
    )
    if com_col is None:
        raise ValueError(f"':CentreOfMass' column not found in Model Outputs of {csv_path}")

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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_validate_com_normalized.py -v
```

期望：7 passed

- [ ] **Step 5: 用真实 WT02.csv 验证输出（手动检查）**

```bash
python -c "
from pathlib import Path
from validate_com_normalized import parse_vicon_model_outputs
rate, frames, cx, cy, cz = parse_vicon_model_outputs(Path(r'H:\COM\video-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test\WT02.csv'))
print('rate:', rate)
print('frames count:', len(frames))
print('first frame:', frames[0], 'last frame:', frames[-1])
print('first COM (X,Y,Z):', cx[0], cy[0], cz[0])
"
```

期望：rate=250.0, frames count > 0, COM 值在合理范围内（mm）

- [ ] **Step 6: Commit**

```bash
git add video-vicon/code/validate_com_normalized.py video-vicon/code/tests/test_validate_com_normalized.py
git commit -m "feat: add parse_vicon_model_outputs with tests"
```

---

## Task 3: 视频 COM 解析

**Files:**
- Modify: `video-vicon/code/validate_com_normalized.py`
- Modify: `video-vicon/code/tests/test_validate_com_normalized.py`

`keypoints.json` 每条记录格式：
```json
{
  "image": "Video_WT02_Trajectory\\raw_frames\\frame_000001.jpg",
  "com": {"com_x": 631.2061, "com_y": 875.556}
}
```
试验文件夹前缀规则：`WT02` → `Video_WT02_Trajectory`，`WTFAST11` → `Video_WTFAST11_Trajectory`

帧号从文件名 `frame_NNNNNN.jpg` 提取，从 1 开始。

- [ ] **Step 1: 写失败测试**

追加到测试文件：

```python
def test_parse_video_com_basic():
    import json, tempfile, os
    records = [
        {"image": "Video_WT02_Trajectory\\raw_frames\\frame_000001.jpg", "com": {"com_x": 100.0, "com_y": 200.0}},
        {"image": "Video_WT02_Trajectory\\raw_frames\\frame_000002.jpg", "com": {"com_x": 101.0, "com_y": 201.0}},
        {"image": "Video_WT06_Trajectory\\raw_frames\\frame_000001.jpg", "com": {"com_x": 999.0, "com_y": 999.0}},
        {"image": "Video_WT02_Trajectory\\raw_frames\\frame_000003.jpg", "com": None},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(records, f)
        tmp = f.name
    try:
        from validate_com_normalized import parse_video_com
        frame_nums, com_x, com_y = parse_video_com(Path(tmp), "WT02")
        assert frame_nums == [1, 2]
        assert abs(com_x[0] - 100.0) < 1e-6
        assert abs(com_y[1] - 201.0) < 1e-6
    finally:
        os.unlink(tmp)


def test_parse_video_com_null_skipped():
    import json, tempfile, os
    records = [
        {"image": "Video_WT02_Trajectory\\raw_frames\\frame_000005.jpg", "com": None},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(records, f)
        tmp = f.name
    try:
        from validate_com_normalized import parse_video_com
        frame_nums, com_x, com_y = parse_video_com(Path(tmp), "WT02")
        assert frame_nums == []
    finally:
        os.unlink(tmp)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_validate_com_normalized.py::test_parse_video_com_basic -v
```

期望：`ImportError: cannot import name 'parse_video_com'`

- [ ] **Step 3: 实现 parse_video_com**

追加到 `validate_com_normalized.py`：

```python
def parse_video_com(
    keypoints_json: Path,
    trial_name: str,
) -> tuple[list[int], list[float], list[float]]:
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_validate_com_normalized.py -v
```

期望：9 passed

- [ ] **Step 5: Commit**

```bash
git add video-vicon/code/validate_com_normalized.py video-vicon/code/tests/test_validate_com_normalized.py
git commit -m "feat: add parse_video_com with tests"
```

---

## Task 4: manifest 加载、时间轴构建、插值

**Files:**
- Modify: `video-vicon/code/validate_com_normalized.py`
- Modify: `video-vicon/code/tests/test_validate_com_normalized.py`

manifest CSV 列：`trial,first_frame,last_frame,frames,trajectory_rate_hz,capture_start,trajectory_start_time,video_offset_sec,duration_sec,output`

时间轴约定：
- 视频：`t = (frame_num - 1) / VIDEO_FPS`（frame_num 从 1 开始，frame 1 → t=0）
- Vicon：`t = (frame_num - first_frame) / trajectory_rate_hz`（first_frame → t=0）

- [ ] **Step 1: 写失败测试**

追加到测试文件：

```python
def test_load_manifest_basic():
    import tempfile, os
    content = "trial,first_frame,last_frame,frames,trajectory_rate_hz,capture_start,trajectory_start_time,video_offset_sec,duration_sec,output\nWT02,256,1249,994,250.0,2026-05-05 08:27:26.144,2026-05-05 08:27:27.164,26.164,3.976,output.mov\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(content)
        tmp = f.name
    try:
        from validate_com_normalized import load_manifest
        manifest = load_manifest(Path(tmp))
        assert "WT02" in manifest
        assert manifest["WT02"]["first_frame"] == 256
        assert manifest["WT02"]["trajectory_rate_hz"] == 250.0
    finally:
        os.unlink(tmp)


def test_build_video_time_axis():
    from validate_com_normalized import build_video_time_axis
    t = build_video_time_axis([1, 2, 3], fps=10.0)
    np.testing.assert_allclose(t, [0.0, 0.1, 0.2], atol=1e-10)


def test_build_vicon_time_axis():
    from validate_com_normalized import build_vicon_time_axis
    t = build_vicon_time_axis([256, 257, 258], first_frame=256, rate_hz=250.0)
    np.testing.assert_allclose(t, [0.0, 0.004, 0.008], atol=1e-10)


def test_interpolate_vicon_to_video():
    from validate_com_normalized import interpolate_vicon_to_video
    vicon_t = np.array([0.0, 0.1, 0.2, 0.3])
    vicon_vals = np.array([0.0, 10.0, 20.0, 30.0])
    video_t = np.array([0.05, 0.15, 0.25])
    result = interpolate_vicon_to_video(video_t, vicon_t, vicon_vals)
    np.testing.assert_allclose(result, [5.0, 15.0, 25.0], atol=1e-10)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_validate_com_normalized.py::test_load_manifest_basic tests/test_validate_com_normalized.py::test_build_video_time_axis -v
```

期望：ImportError

- [ ] **Step 3: 实现这四个函数**

追加到 `validate_com_normalized.py`：

```python
def load_manifest(manifest_csv: Path) -> dict[str, dict[str, Any]]:
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
    return np.array([(fn - 1) / fps for fn in frame_numbers])


def build_vicon_time_axis(
    frame_numbers: list[int], first_frame: int, rate_hz: float
) -> np.ndarray:
    return np.array([(fn - first_frame) / rate_hz for fn in frame_numbers])


def interpolate_vicon_to_video(
    video_t: np.ndarray, vicon_t: np.ndarray, vicon_vals: np.ndarray
) -> np.ndarray:
    return np.interp(video_t, vicon_t, vicon_vals)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_validate_com_normalized.py -v
```

期望：13 passed

- [ ] **Step 5: Commit**

```bash
git add video-vicon/code/validate_com_normalized.py video-vicon/code/tests/test_validate_com_normalized.py
git commit -m "feat: add manifest loader, time axis builders, and interpolation with tests"
```

---

## Task 5: 绘图和 summary CSV 输出

**Files:**
- Modify: `video-vicon/code/validate_com_normalized.py`

不需要单元测试（纯 I/O 副作用）；在 Task 6 集成运行时目视验证。

- [ ] **Step 1: 实现 plot_comparison 和 write_summary_csv**

先在 `validate_com_normalized.py` **文件顶部** `import numpy` 之后添加：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

然后在文件末尾追加函数体：

```python
def plot_comparison(
    t: np.ndarray,
    video_z: np.ndarray,
    vicon_z: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
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


def write_summary_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial", "axis", "pearson_r", "p_value", "nrmse", "n_frames", "warning"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 2: Commit**

```bash
git add video-vicon/code/validate_com_normalized.py
git commit -m "feat: add plot_comparison and write_summary_csv"
```

---

## Task 6: main 管道整合

**Files:**
- Modify: `video-vicon/code/validate_com_normalized.py`

- [ ] **Step 1: 实现 process_trial 和 main**

追加到 `validate_com_normalized.py`：

```python
def process_trial(
    trial: str,
    manifest_row: dict[str, Any],
    keypoints_json: Path,
    vicon_csv: Path,
    plots_dir: Path,
) -> list[dict[str, Any]]:
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

    # 限定时间范围到两侧重叠区间
    t_min = max(video_t[0], vicon_t[0])
    t_max = min(video_t[-1], vicon_t[-1])
    mask = (video_t >= t_min) & (video_t <= t_max)
    video_t = video_t[mask]
    video_cx_arr = np.array(video_cx)[mask]
    video_cy_arr = np.array(video_cy)[mask]

    if len(video_t) < 5:
        print(f"[{trial}] WARNING: fewer than 5 overlapping frames, skipping")
        return []

    vicon_cx_arr = np.array(vicon_cx)
    vicon_cy_arr = np.array(vicon_cy)
    vicon_cz_arr = np.array(vicon_cz)
    vicon_t_arr = np.array(vicon_t)

    # 插值 Vicon 到视频时间点
    vicon_y_interp = interpolate_vicon_to_video(video_t, vicon_t_arr, vicon_cy_arr)
    vicon_z_interp = interpolate_vicon_to_video(video_t, vicon_t_arr, vicon_cz_arr)

    # 符号对齐（视频 y 向下，Vicon Z 向上）
    vicon_z_aligned = -vicon_z_interp

    rows: list[dict[str, Any]] = []

    for axis_label, vid_sig, vic_sig, plot_suffix in [
        ("horizontal (video_x vs vicon_Y)", video_cx_arr, vicon_y_interp, "x"),
        ("vertical (video_y vs vicon_Z)", video_cy_arr, vicon_z_aligned, "z"),
    ]:
        warning = ""
        missing_pct = (mask.size - mask.sum()) / mask.size * 100
        if missing_pct > 20:
            warning = f"video COM missing {missing_pct:.0f}% of frames"

        try:
            vid_z = zscore_normalize(vid_sig)
            vic_z = zscore_normalize(vic_sig)
        except ValueError as exc:
            print(f"[{trial}][{axis_label}] zscore failed: {exc}")
            continue

        r, p, nrmse = compute_metrics(vid_z, vic_z)

        plot_path = plots_dir / f"{trial}_{plot_suffix}.png"
        plot_comparison(
            video_t,
            vid_z,
            vic_z,
            title=f"{trial} — {axis_label} | r={r:.3f} nRMSE={nrmse:.3f}",
            output_path=plot_path,
        )

        rows.append({
            "trial": trial,
            "axis": axis_label,
            "pearson_r": round(r, 4),
            "p_value": f"{p:.4e}",
            "nrmse": round(nrmse, 4),
            "n_frames": len(video_t),
            "warning": warning,
        })
        print(f"[{trial}][{plot_suffix}] r={r:.3f}, nRMSE={nrmse:.3f}, n={len(video_t)}")

    return rows


def main() -> int:
    manifest = load_manifest(MANIFEST_CSV)
    plots_dir = VALIDATION_DIR / "plots"
    all_rows: list[dict[str, Any]] = []

    for trial, manifest_row in manifest.items():
        vicon_csv = VICON_CSV_ROOT / f"{trial}.csv"
        if not vicon_csv.exists():
            print(f"[{trial}] WARNING: CSV not found at {vicon_csv}, skipping")
            continue
        rows = process_trial(trial, manifest_row, KEYPOINTS_JSON, vicon_csv, plots_dir)
        all_rows.extend(rows)

    summary_path = VALIDATION_DIR / "summary.csv"
    write_summary_csv(all_rows, summary_path)
    print(f"\nSummary saved to {summary_path}")
    print(f"Plots saved to {plots_dir}")
    return 0 if all_rows else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
```

- [ ] **Step 2: 运行所有单元测试确认不受影响**

```bash
python -m pytest tests/test_validate_com_normalized.py -v
```

期望：13 passed

- [ ] **Step 3: 端到端运行**

```bash
cd H:\COM\video-vicon\code
python validate_com_normalized.py
```

期望输出示例：
```
[WT02][x] r=0.xxx, nRMSE=0.xxx, n=xxx
[WT02][z] r=0.xxx, nRMSE=0.xxx, n=xxx
...
Summary saved to H:\COM\video-vicon\validation\summary.csv
Plots saved to H:\COM\video-vicon\validation\plots
```

- [ ] **Step 4: 检查 summary.csv 和图表**

```bash
type H:\COM\video-vicon\validation\summary.csv
dir H:\COM\video-vicon\validation\plots
```

期望：13 个试验 × 2 轴 = 最多 26 行；26 个 PNG 文件

- [ ] **Step 5: Commit**

```bash
git add video-vicon/code/validate_com_normalized.py
git commit -m "feat: add process_trial and main pipeline for COM normalized validation"
```

---

## 运行命令汇总

```bash
# 运行全部单元测试
cd H:\COM\video-vicon\code
python -m pytest tests/test_validate_com_normalized.py -v

# 运行单个测试
python -m pytest tests/test_validate_com_normalized.py::test_zscore_normalize_basic -v

# 端到端验证
python validate_com_normalized.py

# 安装依赖（如缺少）
pip install numpy scipy matplotlib
```

