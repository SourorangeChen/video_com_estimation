# COM 验证方案一：归一化趋势比较 — 设计规格

**日期**：2026-05-22  
**阶段**：可行性验证（B 阶段），为后续论文级验证（A 阶段）铺路

---

## 目标

对每个 Vicon 试验，将视频估算的 COM 时间序列与 Vicon COM 时间序列做归一化后比较，输出相关系数、归一化 RMSE 和叠加曲线图，判断视频 COM 能否追踪 Vicon 金标准的运动趋势。

---

## 数据来源

| 数据 | 路径 | 格式 | 采样率 |
|---|---|---|---|
| 视频 COM | `video-vicon/data/Chenzixuan/Video/Video_keypoint-com/results/keypoints.json` | JSON 列表，每条记录含 `"com": {"com_x": float, "com_y": float}`（已预计算） | ~30 fps（不均匀） |
| Vicon COM | `video-vicon/data/Chenzixuan/Vicon/rawdata/Chenzixuan_20260505_test/<trial>.csv` | CSV `Model Outputs` 部分，含 COM XYZ 列 | 250 Hz |
| 试验时间信息 | `video-vicon/data/Chenzixuan/Video/Video_trial/Video_ViconTrial_manifest.csv` | CSV，含 `video_offset_sec`、`trajectory_start_time`、`trajectory_rate_hz` | — |

试验列表：WT02、WT06、WT07、WT08、WT09、WT10、WTFAST11、WTFAST14、WTFAST18、WTFAST19、WTFAST20、WTFAST21、WTFAST22

---

## 坐标轴对应关系

相机朝向 Vicon X 轴负方向（前后视角）：

| 视频轴 | Vicon 轴 | 说明 |
|---|---|---|
| COM_x（像素，水平，向右为正） | COM_Y（mm，ML 内外侧） | 直接对应 |
| COM_y（像素，垂直，向下为正） | COM_Z（mm，垂直，向上为正） | **符号取反**后对应 |

Vicon X 轴（AP 前后）为深度方向，视频无法直接观测，本方案不比较该轴。

---

## 处理流程

### 1. 时间对齐

manifest 中的 `trajectory_start_time` 已经是经过 9.754s 校正的绝对时间，视频试验片段和 Vicon 轨迹共享同一起点，因此直接使用**试验内相对时间（t=0 为试验开始）**：

- 视频帧相对时间 = `frame_index_in_trial_video / fps`
  - `frame_index` 从 0 开始（cut 视频的第几帧）
  - `fps` 使用原始视频标称值 29.996 fps（iPhone 15 Pro 元数据）
- Vicon 帧相对时间 = `(frame_number - first_frame) / trajectory_rate_hz`
  - `first_frame` 和 `trajectory_rate_hz` 均来自 manifest

### 2. 视频 COM 读取

- 读取 `keypoints.json`，按试验名（`image` 字段路径的第一级目录）过滤记录
- 直接读取每条记录的 `"com"` 字段（`com_x`、`com_y`，像素坐标）
- `"com"` 为 `null` 的帧跳过；从 `image` 文件名（`frame_NNNNNN.jpg`）提取帧号

### 3. Vicon COM 提取

- 解析 CSV，定位 `Model Outputs` 部分
- 找到 COM 相关列（预期列名含 `CentreOfMass` 或类似，需运行时确认）
- 提取 `(frame_number, COM_X, COM_Y, COM_Z)`，转换为时间轴
- 如果没有则记作0

### 4. 重采样到公共时间轴

- 以视频帧时间点为基准时间轴（稀疏，~30fps）
- 用线性插值将 Vicon COM_Y 和 COM_Z 插值到视频帧时间点
- 不反向上采样视频，避免引入视频端的人工插值误差

### 5. 符号对齐

- `vicon_com_z_aligned = -vicon_com_z_interpolated`（翻转垂直轴方向）

### 6. 逐试验 z-score 归一化

对每个试验，各信号独立归一化：

```
z = (x - mean(x)) / std(x)
```

归一化对象：`video_com_x`、`video_com_y`、`vicon_com_y_aligned`、`vicon_com_z_aligned`

不跨试验合并归一化。

### 7. 计算指标

对每个试验，分别针对水平轴（x/Y）和垂直轴（y/Z）计算：

- **Pearson r**：`scipy.stats.pearsonr`
- **归一化 RMSE**：`sqrt(mean((video_z - vicon_z)^2))`（z-score 后，单位为标准差）

### 8. 输出

每个试验输出：
- 两张叠加曲线图（水平轴、垂直轴各一张），保存至 `video-vicon/validation/plots/<trial>_x.png` 和 `<trial>_z.png`
- 一份汇总 CSV，列：`trial, axis, pearson_r, p_value, nrmse, n_frames`，保存至 `video-vicon/validation/summary.csv`

---

## 模块划分

| 模块 | 职责 |
|---|---|
| `parse_vicon_com(csv_path)` | 解析 CSV Model Outputs，返回 `(frame_numbers, com_x, com_y, com_z)` |
| `parse_video_com(keypoints_json, trial_name)` | 过滤 keypoints.json，直接读取 `com` 字段，返回 `(timestamps, com_x_px, com_y_px)` |
| `align_and_resample(video_t, video_com, vicon_t, vicon_com)` | 时间对齐 + 线性插值，返回公共时间轴上的两组信号 |
| `zscore_normalize(signal)` | 单信号 z-score |
| `compute_metrics(a, b)` | 返回 `(pearson_r, p_value, nrmse)` |
| `plot_comparison(t, video_z, vicon_z, title, output_path)` | 保存叠加曲线图 |
| `main()` | 遍历所有试验，调用上述模块，写 summary.csv |

新脚本路径：`video-vicon/code/validate_com_normalized.py`

---

## 已知风险

| 风险 | 处理方式 |
|---|---|
| Vicon CSV Model Outputs 的 COM 列名不确定 | 运行时打印列名预览，手动确认后硬编码 |
| 视频帧率不均匀 | 从帧时间戳差值估算实际 fps，不假设固定 30fps |
| WT02 文件夹命名不一致 | manifest 中 trial 名与 keypoints.json 中 image 路径的匹配需要容错处理 |
| 部分帧 `"com"` 为 null | 跳过缺失帧，记录缺失比例；若某试验缺失 > 20% 则标记警告 |
