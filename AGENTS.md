# AGENTS.md

Guidance for agents working in this repository.

## Project

This is a biomechanics research validation pipeline for comparing video-derived 2D center-of-mass (CoM) estimates against Vicon CoM as the gold standard.

Active workspace:

```text
H:\COM
```

Use the full Anaconda Python path:

```powershell
D:\Program\Anaconda3\python.exe <script>
```

## Key Data

Current active data layout is under `video-vicon/`:

```text
video-vicon/
  code/
  data/Chenzixuan/
    Vicon/rawdata/Chenzixuan_20260505_test/*.csv
    Video/Video_trial/Video_ViconTrial_manifest.csv
    Video/Video_keypoint-com/results/keypoints_and_com.json
  validation/
```

Do not overwrite raw Vicon CSVs, raw videos, raw frames, or `.xcp` files.

## Coordinate And Timeline Rules

There are two coordinate systems:

```text
Video: 2D pixels, origin top-left, y increases downward
Vicon: 3D lab coordinates in mm, z is vertical
```

There are multiple timelines:

```text
Original iPhone video timeline
Cut trial video timeline
Vicon trial frame timeline
```

For aligned video/Vicon examples, use relative `time_s`, not local row index. In WT02, video frame 1 and Vicon frame 294 are not the same moment. The aligned first pair used in the current plots is:

```text
video frame 6,  t=0.167s  <->  Vicon frame 298, t=0.168s
```

Aligned-frame filenames encode this pairing and the residual time gap, e.g.
`WT02_aligned_video_idx_01_video_0006_vicon_0298_dt_1.3ms.png`.

When comparing displacement or velocity, compare over the same time interval. Vicon is 250 Hz; video is about 29.996 Hz. Do not compare Vicon 250 Hz frame-to-frame displacement directly against video 30 Hz frame-to-frame displacement.

## Current Metric Scripts

All scripts are standalone. Run from the repo root with the full Anaconda path. The
plotting scripts import from each other, so run them from `video-vicon\code` (or keep
that directory importable) — they share data loaders.

### 1. Vicon CoM kinematics

```powershell
D:\Program\Anaconda3\python.exe video-vicon\code\validate_com_normalized.py
```

Reads the raw Vicon CSVs + manifest + keypoints JSON. Writes:

```text
video-vicon/validation/summary.csv               ← per-trial/per-axis correlation metrics
video-vicon/validation/vicon_com_kinematics.csv  ← Vicon CoM kinematics (mm→m, displacement, velocity, xCoM)
```

Note the filename: the plotting scripts read `vicon_com_metric.csv` first and fall back
to `vicon_com_kinematics.csv`. If only `vicon_com_kinematics.csv` exists, that fallback is
used automatically — no rename needed. Treat `vicon_com_metric.csv` /
`vicon_com_kinematics.csv` as the canonical Vicon kinematics table.

### 2. Legacy video CoM kinematics

```powershell
D:\Program\Anaconda3\python.exe video-vicon\code\video_com_metric.py
```

Reads the legacy `keypoints_and_com.json`. Writes:

```text
video-vicon/validation/video_com_metric.csv
```

Only frames whose `image` folder matches `Video_<trial>_Trajectory` and that have a
`com` dict plus valid nose + both-ankle y are included.

### 2a. Active keypoints-preprocessed video CoM kinematics

The current active branch uses the keypoints-preprocessed JSON, not the legacy raw
`Video_keypoint-com` JSON:

```text
video-vicon/data/Chenzixuan/Video/video_keypoints-preprocessed/results/keypoints_and_com_preprocessed.json
```

The JSON `com` field is the current input CoM: it is recomputed from preprocessed
COCO-17 keypoints and then smoothed with a centered moving average. `source_com`
keeps the CoM before CoM smoothing but after keypoint preprocessing.

```powershell
D:\Program\Anaconda3\python.exe video-vicon\code\compute_keypoints_smooth_video_metric.py
```

Writes:

```text
video-vicon/validation/metrics_keypoints_preprocessed/video_com_metric.csv
```

Only write to the user-requested output directory. Disable or ignore legacy/default
side outputs when running one-off wrappers.

### 3. WT02 Vicon kinematics frames (3D + Y-Z 2D)

```powershell
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_kinematics_3d.py
```

Reads the Vicon kinematics CSV + raw WT02 trajectory markers. Writes per-frame skeletons:

```text
video-vicon/validation/WT02_first50_kinematics_3d/
video-vicon/validation/WT02_first50_kinematics_yz_2d/
```

### 4. WT02 video metric frames (2D)

```powershell
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_video_metrics_2d.py
```

Reads `video_com_metric.csv` + keypoints JSON. Writes:

```text
video-vicon/validation/WT02_first50_video_metrics_2d/
```

### 5. Aligned WT02 video/Vicon 2D example plots

```powershell
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_aligned_video_vicon_2d.py
```

Pairs the first `FRAME_COUNT` (currently 100) video frames to their nearest-time Vicon
frame, then renders matched video and Vicon Y-Z frames side by side. Writes:

```text
video-vicon/validation/WT02_first50_aligned_video_metrics_2d/
video-vicon/validation/WT02_first50_aligned_vicon_metrics_yz_2d/
```

### 6. Aligned WT02 time-series comparison

```powershell
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_aligned_timeseries.py
```

Uses the same aligned pairs to plot video vs Vicon Y-Z displacement, vCoM, `l`, and xCoM
over time, plus a comparison CSV. Writes:

```text
video-vicon/validation/WT02_first100_aligned_timeseries/
  WT02_aligned_100_metric_comparison.csv
  WT02_aligned_displacement_video_vs_vicon.png
  WT02_aligned_vcom_video_vs_vicon.png
  WT02_aligned_l_video_vs_vicon.png
  WT02_aligned_xcom_video_vs_vicon.png
```

### Other scripts

```text
validate_keypoints_normalized.py  ← COCO-17 vs Vicon Trajectories correlation metrics
add_labels_to_xcorr_plots.py      ← post-process xcorr alignment plots with annotations
vicon_skeleton_drawing.py         ← raw Vicon 3D skeleton frames from CSV
video_pct_estimated.py            ← mmdet + PCT pose inference (GPU, H:\PCT)
video_estimated_com.py            ← draw video CoM overlay frames
```

### Time alignment (current approach)

There are two different alignment methods in use. Do not conflate them.

For first-100 frame visual examples, aligned plots pair each video frame to its
**nearest-time** Vicon frame (`np.argmin(|vicon_time - video_time|)`), bounded to the
Vicon time range. This does not interpolate values; it only chooses the nearest Vicon
frame for file names and side-by-side display.

For the current CoM z-score correlation analysis, the direction is **video -> Vicon
time axis**. First crop to the overlapping time window. For every Vicon time point
`t_vicon`, estimate the video value between the adjacent video frames:

```text
video(t_vicon) = video0 + (video1 - video0) * (t_vicon - t0) / (t1 - t0)
```

After this interpolation, video and Vicon have the same Vicon-time samples. xcorr lag
is therefore reported in Vicon frames and converted with 250 Hz (`lag_ms = lag_frames /
250 * 1000`).

## Vicon Metrics

Vicon CoM is parsed from the Vicon CSV `Model Outputs` section, column ending in `:CentreOfMass`, in millimeters.

Calculations convert to meters:

```text
com_m = [com_x_mm, com_y_mm, com_z_mm] / 1000
```

Displacement is relative to the previous Vicon frame:

```text
dr_i = r_i - r_(i-1)
dr_0 = [0, 0, 0]
displacement_m = norm(dr_i)
```

Velocity is currently calculated with numerical differentiation over Vicon time:

```text
velocity = gradient(com_m, time_s)
velocity_m_s = norm(velocity)
```

For xCoM:

```text
l(t) = com_z_m(t)
omega0(t) = sqrt(9.81 / l(t))
xCoM(t) = CoM(t) + velocity(t) / omega0(t)
```

This assumes Vicon ground height is `Z = 0`.

For front/back 2D Vicon plots, use the Y-Z projection. The current Vicon Y-Z plots mirror the horizontal Y axis to visually match the video walking direction:

```text
Y (m, mirrored)
Z (m)
```

## Video Metrics

Video metrics are 2D projection metrics, not true 3D metrics.

Legacy inputs come from:

```text
video-vicon/data/Chenzixuan/Video/Video_keypoint-com/results/keypoints_and_com.json
```

The active keypoints-preprocessed branch uses:

```text
video-vicon/data/Chenzixuan/Video/video_keypoints-preprocessed/results/keypoints_and_com_preprocessed.json
```

Current preprocessing and smoothing:

```text
keypoints: trial-wise median window 3, then Savitzky-Golay window 7 / polyorder 2
CoM: recomputed from preprocessed keypoints, then centered moving average window 5
```

The current video scale estimate uses shoulder width per frame:

```text
reference_shoulder_width_m = 0.34
shoulder_width_px = norm(left_shoulder - right_shoulder)
pixels_per_meter = shoulder_width_px / 0.34
```

Ground height still uses the lower of the two ankle keypoints in image coordinates:

```text
ground_y_px = max(left_ankle_y_px, right_ankle_y_px)
```

Video displacement is relative to the previous video frame:

```text
dx_px = com_x_px_i - com_x_px_(i-1)
dy_px = com_y_px_i - com_y_px_(i-1)

displacement_x_m = dx_px / pixels_per_meter_i
displacement_y_m_up = -dy_px / pixels_per_meter_i
displacement_m = sqrt(displacement_x_m^2 + displacement_y_m_up^2)
```

Video velocity is based on the frame-to-frame displacement, not on differentiating absolute `com_px / pixels_per_meter(t)`:

```text
velocity_x_m_s = displacement_x_m * VIDEO_FPS
velocity_y_m_s_up = displacement_y_m_up * VIDEO_FPS
velocity_m_s = sqrt(velocity_x_m_s^2 + velocity_y_m_s_up^2)
```

The previous absolute-coordinate differentiation approach was wrong because frame-varying `pixels_per_meter(t)` rescales the image origin and creates artificial velocity.

Video xCoM:

```text
l_px = ground_y_px - com_y_px
l_m = l_px / pixels_per_meter
omega0 = sqrt(9.81 / l_m)

com_x_m = com_x_px / pixels_per_meter
com_y_m_up = -com_y_px / pixels_per_meter

xcom_x_m = com_x_m + velocity_x_m_s / omega0
xcom_y_m_up = com_y_m_up + velocity_y_m_s_up / omega0

xcom_x_px = xcom_x_m * pixels_per_meter
xcom_y_px = -xcom_y_m_up * pixels_per_meter
```

## Known Issues With Video Metrics

The video metrics remain noisier than Vicon. Do not treat them as final validated biomechanical metrics without further preprocessing.

Known causes:

- `pixels_per_meter(t)` still changes frame to frame because it uses detected shoulder width.
- `ground_y_px = max(left_ankle_y, right_ankle_y)` can switch between left and right foot during walking.
- Shoulder and ankle keypoint jitter affects scale, `l`, velocity, and xCoM.
- Velocity amplifies small CoM/keypoint errors because it differentiates position over a 30 Hz time step.
- Video is a 2D projection; Vicon is 3D. For front/back view, compare video X/Y only against Vicon Y/Z projection, not Vicon full 3D velocity.

Recommended next investigation before changing formulas:

```text
raw video CoM vs smoothed video CoM
raw pixels_per_meter vs smoothed pixels_per_meter
video frame-to-frame metrics vs Vicon metrics aggregated over matching video-frame intervals
```

Potential improvements:

- Test alternative CoM smoothing windows before velocity/xCoM.
- Smooth `pixels_per_meter(t)` with a median or low-pass filter.
- Use a stance-foot or smoothed ground estimate instead of raw `max(left_ankle_y, right_ankle_y)`.
- Keep raw and filtered video metric CSVs separate for auditability.

## Plot Color Legend

Current visualizations use:

```text
green: left side
red: right side
black dots: keypoints/markers
blue dot: CoM
orange x: xCoM
purple: displacement
cyan: velocity x 0.05s
```

## Tests

Run focused tests after metric or plot changes:

```powershell
D:\Program\Anaconda3\python.exe -m pytest video-vicon/code/tests -q
```

At minimum, run the tests for any touched script.
