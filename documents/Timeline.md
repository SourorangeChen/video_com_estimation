# COM Project Timeline

项目：视频 2D CoM 与 Vicon gold standard 对比验证  
工作区：`H:\COM`  
主目录：`video-vicon/`

## 记录方式

每周按以下结构记录：

- **本周目标**：这一周主要解决什么问题。
- **本周完成**：实际完成的代码、数据处理、分析或可视化。
- **关键产出**：CSV、JSON、图片、PPT、Markdown 或其他可复用文件。
- **主要发现**：结果说明了什么，哪里表现好，哪里仍有问题。
- **下一步**：下一周建议继续推进的内容。

---

## 2026-06-01 至 2026-06-04：keypoints-preprocessed 分支、时间对齐与周报整理

### 本周目标

完善当前 COM validation pipeline，使视频端 CoM 指标、Vicon 指标、时间对齐、相关性分析和结果展示形成一条完整可解释的流程。

### 本周完成

- 视频端加入了关键点预处理流程。
  - 按 trial 分组处理 COCO-17 keypoints。
  - 对缺失关键点做线性插值。
  - 使用 median filter 去除尖刺。
  - 使用 Savitzky-Golay filter 平滑关键点轨迹。
- 视频端加入了 CoM 平滑处理。
  - 基于预处理后的 keypoints 重新计算 CoM。
  - 对 CoM 像素坐标做 centered moving average 平滑。
  - 保留 `source_com`，用于追踪 CoM smoothing 前的结果。
- 优化了 CoM 相关性分析前的时间对齐方法。
  - 不再直接按局部行号或近似 frame index 比较。
  - 先裁剪视频与 Vicon 的重叠时间窗口。
  - 将 video CoM / video metric 线性插值到 Vicon `time_s`。
  - 插值后 video 和 Vicon 使用相同的 Vicon-time samples。
- 明确区分了两类时间对齐。
  - 逐帧可视化：使用 nearest-time Vicon frame，仅用于 side-by-side 图和文件名配对。
  - 相关性分析：使用 video -> Vicon time axis 的线性插值。
- 重新生成了 CoM 相关性分析结果。
  - 对 video CoM 和 Vicon CoM 做 detrend。
  - 做 z-score normalization。
  - 计算 Pearson r、nRMSE、xcorr peak、xcorr lag。
  - xcorr lag 以 Vicon frame 为单位，并按 250 Hz 换算为毫秒。
- 重新计算了 CoM 相关指标。
  - video 端计算 scale、displacement、velocity、`l`、xCoM。
  - Vicon 端计算 CoM displacement、velocity、`l`、xCoM。
  - 视频速度改为基于 frame-to-frame displacement，而不是对绝对 `com_px / pixels_per_meter(t)` 求导。
- 重新生成了 video 和 Vicon 的时序信号对比。
  - WT02 all-frame metric comparison 使用 Vicon 时间轴。
  - 对比 velocity、`l`、CoM displacement、xCoM displacement。
  - 同时生成 raw、detrended、lag-corrected 等图像结果。
- 生成了 WT02 对齐可视化结果。
  - 视频端生成 first100 overlay / path overview。
  - Vicon 端生成 Y-Z projection 对齐图。
  - 文件名中保留 video frame、Vicon frame 和 residual time gap。

### 关键产出

- Active video metric：
  `video-vicon/validation/metrics_keypoints_preprocessed/video_com_metric.csv`

- Active Vicon metric：
  `video-vicon/validation/metrics_keypoints_preprocessed/vicon_com_metric.csv`

- CoM 相关性结果：
  `video-vicon/validation/com_keypoints_preprocessed/com_correlation.csv`

- CoM z-score / detrend / xcorr 图：
  `video-vicon/validation/com_keypoints_preprocessed/com_z-score_detrend_xcorr/`

- WT02 all-frame metric comparison：
  `video-vicon/validation/metrics_keypoints_preprocessed/metrics_timeseries_validation_WT02/`

- WT02 视频对齐图：
  `video-vicon/validation/metrics_keypoints_preprocessed/video_metrics_WT02_first100/`

- WT02 Vicon 对齐图：
  `video-vicon/validation/metrics_keypoints_preprocessed/vicon_metrics_WT02_first100/`

- PPT 周报：
  `temp/weekly_report/COM_weekly_report_2026-06-04.pptx`

- Markdown 方法论周报：
  `temp/COM_methodology_weekly_report.md`

### 主要发现

- 当前 active pipeline 已覆盖 13 个 trial。
- 视频端 metric 行数为 1028，Vicon metric 行数为 8001。
- CoM shape correlation 在多数 trial 中表现可用。
  - horizontal 平均 Pearson r 约为 0.766。
  - vertical 平均 Pearson r 约为 0.773。
- WT02 中，`xcom_y_delta` 在 detrend + z-score 后表现最好。
  - Pearson r 约为 0.835。
  - xcorr lag 接近 0，约为 4 ms。
- velocity 仍然较弱。
  - 主要原因可能是关键点抖动、每帧 shoulder-width scale 变化、ground_y 估计不稳定。
- 视频指标仍然是 2D projection metric，不能直接解释为完整 3D biomechanical metric。

### 下一步

- 对比 raw video CoM 与 smoothed video CoM，评估 CoM smoothing window 的影响。
- 对比 raw `pixels_per_meter` 与 smoothed `pixels_per_meter`，检查 shoulder-width scale 抖动。
- 测试更稳定的 ground estimate，避免 `max(left_ankle_y, right_ankle_y)` 在步行中频繁换脚。
- 将 Vicon 指标聚合到视频帧间隔后，再比较 displacement 和 velocity。
- 保持 raw 与 filtered 输出分离，所有派生结果写入独立目录，方便审计和回溯。



