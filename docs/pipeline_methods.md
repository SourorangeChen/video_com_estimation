# 视频-Vicon 步态分析验证管道：工作流与方法说明

## 1. 研究目标

用 **iPhone 视频 + PCT 姿态估计** 计算的重心（CoM）和步态运动学指标，能否替代 Vicon 动作捕捉系统进行步态与平衡分析？

本管道的任务是对两路信号进行时间对齐、信号处理、相关性验证，并可视化对比结果。

- **受试者**：Chenzixuan
- **采集日期**：2026-05-05
- **金标准**：Vicon（250 Hz，反光标记点动捕）
- **待验证方法**：iPhone 视频（~30 Hz，PCT 姿态估计）

---

## 2. 数据采集

### 2.1 设备与原始数据

| 来源 | 设备 | 原始数据 | 采样率 |
|---|---|---|---|
| 视频 | iPhone | `IMG_8467.MOV`（311 秒完整录像） | ~29.996 fps |
| 动捕 | Vicon | `WT02.csv`、`WT06.csv` … `WTFAST22.csv` | 250 Hz |

### 2.2 试验编号

- **正常速度步行**：WT02、WT06、WT07、WT08、WT09、WT10
- **快速步行**：WTFAST11、WTFAST14、WTFAST18、WTFAST19、WTFAST20、WTFAST21、WTFAST22

### 2.3 时间同步

两套系统独立计时，需要一个锚点对齐：

| 系统 | 绝对时刻 |
|---|---|
| 视频 | 2026-05-05 08:27:27.700 |
| Vicon | 2026-05-05 08:27:37.454 |

**偏移量**：Vicon 比视频快 **9.754 秒**。

`.xcp` 文件（Vicon 捕捉配置）已修正为对齐后的时间，原始备份存于 `xcp_backup_before_video_time_alignment/`。

每个试验的起止时刻记录在 `Video_ViconTrial_manifest.csv`：

```
trial, first_frame, last_frame, trajectory_rate_hz
WT02, 294, 513, 250.0
...
```

`first_frame` 是该试验在 Vicon 帧序列中的起始帧号，用于建立 Vicon 时间轴的 t=0 锚点。

---

## 3. 视频处理流程

### 3.1 视频切段

用 FFmpeg 按 Vicon 试验时序将完整 MOV 切割为每试验独立视频片段，存入 `Video_trial/`。

### 3.2 PCT 姿态估计

脚本：`video_pct_estimated.py`

```
每段试验视频
  → [mmdet] 人体检测，输出 bounding box
  → [PCT]   17 个 COCO 关键点坐标估计
  → 每帧输出关键点列表 + 已计算的 CoM 像素坐标
  → 汇总写入 keypoints_and_com.json
```

输出 JSON 格式（每条记录对应一帧）：

```json
{
  "image": "Video_WT02_Trajectory/frame_000001.jpg",
  "keypoints": [[x0,y0], [x1,y1], ..., [x16,y16]],
  "com": {"com_x": 412.3, "com_y": 580.1}
}
```

### 3.3 CoM 计算（7 段人体模型）

函数：`compute_frame_com()`，定义于 `com_cal/calculate_com.py`。

将人体分为 7 个刚体段，按各段质量比例加权平均关键点坐标，得到 2D 重心：

| 段 | 关键点 | 质量比 | 质心位置比（近端→远端） |
|---|---|---|---|
| 躯干+头+颈 | 肩、髋、鼻 | 57.8% | 0.660 |
| 左/右全臂 | 肩、肘、腕 | 各 5.0% | 0.530 |
| 左/右小腿+足 | 膝、踝 | 各 6.1% | 0.606 |
| 左/右大腿 | 髋、膝 | 各 10.0% | 0.433 |

任意一个必需关键点缺失则该帧返回 `None`。

---

## 4. Vicon 数据解析

### 4.1 CSV 格式

Vicon 导出的 CSV 是多段结构，包含三个 section：

```
Devices          ← 模拟设备数据（本项目不用）
Model Outputs    ← 生物力学模型输出，含 :CentreOfMass（mm）
Trajectories     ← 各反光标记点三维坐标（mm）
```

解析方式：在文本中定位 section 标题行，跳过固定行数的表头（rate、名称、轴、单位），从第 +5 行开始读数据。

### 4.2 重心数据（金标准）

从 `Model Outputs` 段读取 `:CentreOfMass` 列（X/Y/Z，单位 mm）。

这是 Vicon 的生物力学模型直接输出，是本项目的**金标准 CoM**。

### 4.3 坐标系

相机面向 Vicon 的 **−X 方向**（正面视角），因此：

| 方向 | 视频坐标 | Vicon 坐标 | 转换 |
|---|---|---|---|
| 左右（ML） | `video_x`，向右为正 | `Vicon_Y`，向右为负 | `video_x ↔ −Vicon_Y` |
| 垂直 | `video_y`，向下为正 | `Vicon_Z`，向上为正 | `video_y ↔ −Vicon_Z` |

---

## 5. 视频运动学指标计算（video_com_metric.py）

从 `keypoints_and_com.json` 逐试验、逐帧计算以下指标，结果写入 `validation/video_com_metric.csv`。

### 5.1 时间轴

```
time_s[i] = (frame_numbers[i] - frame_numbers[0]) / fps
```

以该试验第一帧为 t=0，fps = 29.996。

### 5.2 像素→米缩放因子（每帧独立）

```
ground_y_px         = max(left_ankle_y_px, right_ankle_y_px)
nose_to_ankle_px    = ground_y_px - nose_y_px         ← 鼻尖到脚踝的像素身高
pixels_per_meter[i] = nose_to_ankle_px[i] / 1.70      ← 参考身高 1.70 m
```

**设计原因**：步行中受试者距相机距离变化，透视缩放比不固定，每帧独立计算可消除这一误差。鼻-踝距离近似等于站立身高，以 1.70 m 为参考值。

### 5.3 摆长 l（倒立摆支撑腿长度）

```
l_px[i] = ground_y_px[i] - com_y_px[i]      ← 重心到地面的像素距离
l_m[i]  = l_px[i] / pixels_per_meter[i]     ← 换算成米
```

**物理意义**：倒立摆模型将人体简化为一根以脚踝为支点的杆，l 是杆长（重心离地高度）。

### 5.4 CoM 位置（米制，向上为正）

```
com_x_m[i]    = com_x_px[i] / pixels_per_meter[i]
com_y_m_up[i] = -com_y_px[i] / pixels_per_meter[i]    ← 翻转：像素 Y 向下 → 物理 Y 向上
```

### 5.5 位移（帧间有限差分）

```
displacement_x_px[i]   = com_x_px[i] - com_x_px[i-1]     (第 0 帧 = 0)
displacement_y_px[i]   = com_y_px[i] - com_y_px[i-1]
displacement_x_m[i]    = displacement_x_px[i] / pixels_per_meter[i]
displacement_y_m_up[i] = -displacement_y_px[i] / pixels_per_meter[i]
displacement_m[i]      = sqrt(displacement_x_m² + displacement_y_m_up²)   ← 合位移
```

使用 `np.diff`（向后差分），第 0 帧补零。

### 5.6 速度

```
velocity_x_m_s[i]   = displacement_x_m[i] * fps
velocity_y_m_s_up[i] = displacement_y_m_up[i] * fps
velocity_m_s[i]     = sqrt(vx² + vy²)
```

等价于一阶向后差分乘以帧率，第 0 帧速度为 0。

### 5.7 自然频率 ω₀

```
omega0[i] = sqrt(9.81 / l_m[i])    ← 单位：rad/s
```

倒立摆的固有角频率，l 越大（人越高）则 ω₀ 越小。

### 5.8 外推重心 xCoM（Hof 2005）

```
xcom_x_m[i]    = com_x_m[i]    + velocity_x_m_s[i]   / omega0[i]
xcom_y_m_up[i] = com_y_m_up[i] + velocity_y_m_s_up[i] / omega0[i]
```

**物理意义**：`velocity / omega0` 是速度折算的位置偏移量。xCoM 是当前运动状态下，若不再主动施力，身体最终会"倒向"的预测落点。`xCoM − CoM` 即稳定裕度（Margin of Stability）。

还原为像素坐标（可视化用）：

```
xcom_x_px[i] = xcom_x_m[i] * pixels_per_meter[i]
xcom_y_px[i] = -xcom_y_m_up[i] * pixels_per_meter[i]
```

---

## 6. Vicon 运动学指标计算（validate_com_normalized.py）

从 Vicon CSV 的 `:CentreOfMass` 解析 X/Y/Z（mm），计算后写入 `validation/vicon_com_metric.csv`。

### 6.1 时间轴

```
time_s[i] = (frame_numbers[i] - first_frame) / 250.0
```

以 manifest 中该试验的 `first_frame` 为 t=0。

### 6.2 单位换算

```
com_m = [com_x_mm, com_y_mm, com_z_mm] / 1000.0    ← mm → m
```

### 6.3 位移（帧间有限差分）

```
displacement[i] = com_m[i] - com_m[i-1]    (第 0 帧 = [0,0,0])
displacement_m[i] = norm(displacement[i])   ← 三维合位移
```

与视频相同，使用 `np.diff`，第 0 帧补零。

### 6.4 速度（中心差分）⚠️ 与视频不同

```python
velocity = np.gradient(com_m, time_s, axis=0)
```

`np.gradient` 对中间点使用**中心差分**：

```
velocity[i] ≈ (com_m[i+1] - com_m[i-1]) / (time_s[i+1] - time_s[i-1])
```

端点使用单侧差分。比视频的向后差分更精确（误差 O(Δt²) vs O(Δt)），Vicon 250 Hz 数据密集、噪声小，适合用中心差分。

### 6.5 摆长 l

```
l = com_z_m    ← 直接用 Z 轴高度（Vicon Z 向上，Z=0 是地板）
```

Vicon 不需要像视频那样用身高换算像素，Z 坐标本身就是重心离地高度（米）。

### 6.6 ω₀ 和 xCoM

```
omega0[i] = sqrt(9.81 / com_z_m[i])
xcom[i]   = com_m[i] + velocity[i] / omega0[i]    ← 三维向量运算
```

公式与视频完全相同，但 Vicon 是三维的（X/Y/Z 三分量）。

---

## 7. 相关性验证管道（validate_com_normalized.py / validate_keypoints_normalized.py）

### 7.1 总体流程

以下步骤对每个试验、每个轴（水平 x、垂直 z）独立执行：

```
视频 CoM 序列（~30 Hz）      Vicon CoM 序列（250 Hz）
         ↓                           ↓
     建时间轴                     建时间轴
         ↓                           ↓
      截取重叠时间窗口 ──────────────┘
         ↓
  np.interp 线性插值：Vicon 250 Hz → 视频 30 Hz 时间点
         ↓
  符号对齐（翻转 Vicon Y、Z）
         ↓
  线性去趋势（scipy.signal.detrend）
         ↓
  Z-score 归一化
         ↓
  ┌── Pearson r + p 值
  ├── nRMSE
  └── xcorr（互相关）→ peak_r、lag_frames、lag_ms
```

### 7.2 时间重叠窗口截取

```python
t_min = max(video_t[0], vicon_t[0])
t_max = min(video_t[-1], vicon_t[-1])
mask  = (video_t >= t_min) & (video_t <= t_max)
```

只保留两路信号都有数据的时间段，截掉比例 > 20% 时在 CSV 中记录 warning。

### 7.3 插值（Vicon → 视频时间轴）

```python
vicon_interp = np.interp(video_t, vicon_t, vicon_vals)
```

**线性插值**（`np.interp`）：对每个视频时间点，在其左右最近的两个 Vicon 帧之间做线性内插。

插值方向选择 Vicon→视频的原因：
- Vicon 250 Hz，视频 30 Hz，Vicon 在任意视频时间点附近都有足够近的样本（最大误差 2 ms）
- 反向（视频→250 Hz）需要从稀疏信号向密集时间轴扩展，无意义
- 插值后两路信号长度相同，才能逐点计算 Pearson r

### 7.4 线性去趋势

```python
signal_detrended = scipy.signal.detrend(signal, type="linear")
```

对信号做最小二乘线性拟合，然后减去拟合直线。

**目的**：步行时受试者从 A 走到 B，重心有整体位移漂移（视频像素坐标随人移动，Vicon 世界坐标也在变化），去掉这个线性趋势后，信号只保留步频节律振荡，这才是待验证的核心内容。

### 7.5 Z-score 归一化

```python
signal_z = (signal - signal.mean()) / signal.std()
```

消除量纲差异（视频单位：像素；Vicon 单位：毫米），两路信号均变为"以各自标准差为单位的偏离"，只比较**形状和节律**。

### 7.6 Pearson r

```python
r, p = scipy.stats.pearsonr(a, b)
```

标准线性相关系数，范围 \[-1, 1\]。由于输入已 Z-score（均值 0、标准差 1），等价于：

```
r = (1/n) * Σ a[i] * b[i]
```

p 值为双尾检验，零假设：两信号无线性相关。

### 7.7 nRMSE（归一化均方根误差）

```python
nrmse = sqrt(mean((a - b)**2))
```

因为输入已经 Z-score（单位为标准差），此处的 RMSE 本身即为归一化值：
- nRMSE = 0：两信号完全一致
- nRMSE = 1：误差相当于一个标准差

### 7.8 互相关 xcorr（时移估计）

```python
n    = len(a)
corr = np.correlate(a - mean(a), b - mean(b), mode="full")
corr /= n * std(a) * std(b)
lags = arange(-(n-1), n)        # 共 2n-1 个时移值
lag_frames = lags[argmax(corr)]
lag_ms     = lag_frames / fps * 1000
```

`np.correlate(a, b, mode="full")` 滑动计算所有时移下的互相关：

```
corr[k] = Σ_t  a[t] * b[t + k]
```

归一化后，峰值等价于最优时移下的 Pearson r。

**lag 符号约定**：
- `lag_frames > 0`：b 比 a 超前（b 先发生）
- `lag_frames < 0`：b 比 a 滞后（即 Vicon 超前视频，b=Vicon）

`xcorr_peak_r` 反映排除同步残差后两信号的最大相关上限；`xcorr_lag_ms` 量化残余时间偏差。

---

## 8. 关键点级别验证（validate_keypoints_normalized.py）

对 17 个 COCO 关键点逐一验证，流程与 CoM 验证相同，但信号来源不同：

- **视频侧**：`keypoints_and_com.json` 中每帧每个关键点的像素坐标
- **Vicon 侧**：`Trajectories` 段中对应反光标记点的三维坐标

### COCO-17 → Vicon 标记点映射

| COCO 关键点 | Vicon 标记点 |
|---|---|
| nose | (LFHD + RFHD) / 2 |
| left_eye | LFHD |
| right_eye | RFHD |
| left_ear | LFHD |
| right_ear | RFHD |
| left_shoulder | LSHO |
| right_shoulder | RSHO |
| left_elbow | LELB |
| right_elbow | RELB |
| left_wrist | (LWRA + LWRB) / 2 |
| right_wrist | (RWRA + RWRB) / 2 |
| left_hip | LASI |
| right_hip | RASI |
| left_knee | LKNE |
| right_knee | RKNE |
| left_ankle | LANK |
| right_ankle | RANK |

验证轴：水平（`video_x` vs `−Vicon_Y`）和垂直（`video_y` vs `−Vicon_Z`）。

---

## 9. 运动学对比可视化

### 9.1 单路可视化

| 脚本 | 输入 | 输出 |
|---|---|---|
| `plot_wt02_kinematics_3d.py` | `vicon_com_metric.csv` + `WT02.csv` | 3D 骨架 + CoM/xCoM 向量（前 50 帧）|
| `plot_wt02_video_metrics_2d.py` | `video_com_metric.csv` + `keypoints_and_com.json` | 2D 关键点 + CoM/xCoM 向量（前 50 帧）|

每帧图像标注：位移向量、速度向量、l、ω₀、xCoM-CoM 距离（稳定裕度）。

### 9.2 时间对齐并排可视化

`plot_wt02_aligned_video_vicon_2d.py`：

1. 从两个 metric CSV 分别载入 WT02 的所有帧
2. 按时间戳匹配：对每个视频帧，在 Vicon 时间轴上找最近的帧（`argmin |t_video - t_vicon|`）
3. 生成 100 对匹配帧的并排图像（左：视频 2D，右：Vicon YZ 2D）
4. 文件名编码时间偏差，如 `dt_1.3ms`

### 9.3 运动学时间序列对比

`plot_wt02_aligned_timeseries.py`：对 100 对匹配帧，逐指标绘制 video vs Vicon 时间序列：

| 对比图 | Video 信号 | Vicon 信号 |
|---|---|---|
| 位移 | `displacement_m`（向后差分） | `hypot(Δy, Δz)`（帧间 YZ 面位移） |
| CoM 速度 | `velocity_m_s` | `Δdisplacement / Δt` |
| 摆长 l | `l_m`（踝-CoM 换算） | `com_z_m`（Vicon Z 高度） |
| xCoM delta | `xcom_x_m − xcom_x_m[0]` | `−(xcom_y_m − xcom_y_m[0])` |

坐标轴对齐：`video_x ↔ −Vicon_Y`，`video_y_up ↔ Vicon_Z`。

---

## 10. 验证结果输出

### 10.1 CSV 文件

| 文件 | 内容 |
|---|---|
| `validation/summary.csv` | 每试验每轴的 CoM 相关性指标（r、p、nRMSE、xcorr lag） |
| `validation/keypoint_summary.csv` | 每试验每关键点每轴的相关性指标 |
| `validation/video_com_metric.csv` | 所有试验每帧的视频运动学指标 |
| `validation/vicon_com_metric.csv` | 所有试验每帧的 Vicon CoM 运动学指标 |
| `validation/WT02_first100_aligned_timeseries/WT02_aligned_100_metric_comparison.csv` | WT02 前 100 对匹配帧的运动学指标对比 |

### 10.2 图像目录

| 目录 | 内容 |
|---|---|
| `validation/plots_aligned/` | CoM xcorr 对齐后的 z-score 叠加图（每试验每轴一张） |
| `validation/keypoints_xcorr/kpNN_<name>/` | 各关键点 xcorr 对齐图，附 COCO→Vicon 映射标注 |
| `validation/WT02_first50_kinematics_3d/` | Vicon 3D 骨架 + CoM/xCoM 向量，前 50 帧 |
| `validation/WT02_first50_kinematics_yz_2d/` | Vicon YZ 正视图，前 50 帧 |
| `validation/WT02_first50_video_metrics_2d/` | 视频 2D 关键点 + CoM/xCoM，前 50 帧 |
| `validation/WT02_first100_video_metrics_2d/` | 时间对齐后视频侧 100 帧 |
| `validation/WT02_first100_vicon_metrics_yz_2d/` | 时间对齐后 Vicon 侧 100 帧 |
| `validation/WT02_first100_aligned_timeseries/` | 运动学时间序列对比图（4 张） |

---

## 11. 数据流全图

```
iPhone MOV（311s）
   ↓ FFmpeg 切段（按 manifest first/last_frame）
Video_trial/ 每试验视频片段
   ↓ video_pct_estimated.py（mmdet + PCT）
keypoints_and_com.json
   ├── validate_com_normalized.py ─────────────────────────────────┐
   │      ↑ 也读 Vicon CSV（Model Outputs :CentreOfMass）          │
   │      → summary.csv（相关性指标）                               │
   │      → plots_aligned/（z-score 叠加图）                        │
   │      → vicon_com_metric.csv（Vicon 运动学）                    │
   │                                                                │
   ├── validate_keypoints_normalized.py                            │
   │      ↑ 也读 Vicon CSV（Trajectories 段，17 个标记点）          │
   │      → keypoint_summary.csv                                   │
   │      → keypoints_xcorr/kpNN_<name>/                          │
   │      ↓ add_labels_to_xcorr_plots.py（标注映射关系）            │
   │                                                                │
   └── video_com_metric.py                                         │
          → video_com_metric.csv（视频运动学）                      │
                                                                    │
video_com_metric.csv + vicon_com_metric.csv ←──────────────────────┘
   ├── plot_wt02_kinematics_3d.py   → WT02_first50_kinematics_3d/ + _yz_2d/
   ├── plot_wt02_video_metrics_2d.py → WT02_first50_video_metrics_2d/
   ├── plot_wt02_aligned_video_vicon_2d.py → WT02_first100_{video,vicon}_metrics_*/
   └── plot_wt02_aligned_timeseries.py → WT02_first100_aligned_timeseries/
```

---

## 12. 当前局限与待改进点

| 问题 | 当前处理 | 备注 |
|---|---|---|
| 像素-米换算误差 | 每帧用鼻-踝身高估算缩放因子 | 透视形变、身体倾斜时误差增大 |
| 速度估计方法不一致 | 视频用向后差分，Vicon 用中心差分 | 统一为中心差分可改善比较 |
| 空间绝对比较 | 仅做归一化趋势比较（z-score） | 像素级比较需要验证过的相机标定投影矩阵 |
| 运动学验证范围 | 仅 WT02 前 100 帧 | 待扩展到全部试验 |
| 参考身高固定 | 1.70 m（写死） | 应从受试者信息读取 |
