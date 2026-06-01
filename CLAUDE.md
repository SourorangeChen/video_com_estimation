# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a **biomechanics research validation pipeline** — not a software product. The goal is to validate video-based 2D center-of-mass (CoM) estimates (from iPhone + PCT pose estimation) against Vicon motion capture as a gold standard for gait and balance analysis. The pipeline has expanded to also compute and compare full kinematic metrics: CoM velocity, extrapolated CoM (xCoM), pendulum length (l), and natural frequency (ω₀).

Subject: Chenzixuan. Data lives in `video-vicon/data/Chenzixuan/`.

## Python Runtime

**Always use the full Anaconda path** — `python` alone is blocked by shell hooks:

```powershell
D:\Program\Anaconda3\python.exe <script>
```

## Running Scripts

All scripts are standalone Python files. Run from the repo root:

```powershell
# PCT keypoint inference (requires GPU + mmdet + PCT at H:\PCT)
D:\Program\Anaconda3\python.exe video-vicon\code\video_pct_estimated.py --input-dir <path> --output-dir <path>

# Draw video CoM overlays from keypoints_and_com.json
D:\Program\Anaconda3\python.exe video-vicon\code\video_estimated_com.py

# Draw Vicon 3D skeleton frames from CSV
D:\Program\Anaconda3\python.exe video-vicon\code\vicon_skeleton_drawing.py

# Validate CoM (video vs Vicon gold-standard):
#   outputs validation/summary.csv + validation/plots_aligned/ PNGs
#   also outputs validation/vicon_com_metric.csv (per-frame Vicon CoM kinematics)
D:\Program\Anaconda3\python.exe video-vicon\code\validate_com_normalized.py

# Validate COCO-17 keypoints (video vs Vicon Trajectories):
#   outputs validation/keypoint_summary.csv + validation/keypoint_plots/{kp_name}/{trial}_{axis}.png
D:\Program\Anaconda3\python.exe video-vicon\code\validate_keypoints_normalized.py

# Annotate keypoints_xcorr PNGs with COCO→Vicon mapping labels (overwrites in-place)
D:\Program\Anaconda3\python.exe video-vicon\code\add_labels_to_xcorr_plots.py

# Compute per-frame video kinematic metrics (vel, xCoM, l, ω₀) from keypoints_and_com.json
#   outputs validation/video_com_metric.csv
D:\Program\Anaconda3\python.exe video-vicon\code\video_com_metric.py

# Plot WT02 Vicon CoM kinematics: 3D skeleton + CoM/xCoM vectors, first 50 frames
#   reads validation/vicon_com_metric.csv + WT02.csv trajectories
#   outputs validation/WT02_first50_kinematics_3d/ + validation/WT02_first50_kinematics_yz_2d/
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_kinematics_3d.py

# Plot WT02 video CoM metrics: 2D keypoints + CoM/xCoM vectors, first 50 frames
#   reads validation/video_com_metric.csv + keypoints_and_com.json
#   outputs validation/WT02_first50_video_metrics_2d/
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_video_metrics_2d.py

# Plot WT02 time-aligned video vs Vicon side-by-side 2D frames (100 paired frames)
#   reads both metric CSVs, aligns by time, outputs paired frame PNGs
#   outputs validation/WT02_first100_video_metrics_2d/ + validation/WT02_first100_vicon_metrics_yz_2d/
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_aligned_video_vicon_2d.py

# Plot WT02 time-series comparison of video vs Vicon kinematics (100 paired frames)
#   reads both metric CSVs via build_aligned_selections()
#   outputs validation/WT02_first100_aligned_timeseries/ (CSV + 4 PNGs)
D:\Program\Anaconda3\python.exe video-vicon\code\plot_wt02_aligned_timeseries.py
```

## Architecture

### Data Flow

```
iPhone MOV → [FFmpeg cut by Vicon trial timing] → per-trial video segments
                                                         ↓
                                              [video_pct_estimated.py]
                                        mmdet person detection + PCT pose
                                                         ↓
                                         keypoints_and_com.json (COCO-style)
                                           image paths: Video_{trial}_Trajectory/
                                                         ↓
                         ┌───────────────────────────────┴──────────────────────────────┐
              [video_estimated_com.py]          [validate_keypoints_normalized.py]   [video_com_metric.py]
          compute_frame_com() → blue dot     COCO-17 vs Vicon Trajectories         vel, xCoM, l, ω₀
          video_com_pred/ images             keypoint_plots/{kp_name}/{trial}_{axis}.png  video_com_metric.csv
                                             keypoint_summary.csv
                         └───────────────────────────────┬──────────────────────────────┘
                                                         ↓
                                           [validate_com_normalized.py]
                                   video CoM vs Vicon Model Outputs CentreOfMass
                                          plots_aligned/ PNGs
                                          summary.csv
                                          vicon_com_metric.csv  ← per-frame Vicon kinematics

Vicon CSV (multi-section) → [vicon_skeleton_drawing.py] → 3D skeleton PNGs

vicon_com_metric.csv + WT02.csv → [plot_wt02_kinematics_3d.py] → WT02_first50_kinematics_3d/ + _yz_2d/
video_com_metric.csv + keypoints_and_com.json → [plot_wt02_video_metrics_2d.py] → WT02_first50_video_metrics_2d/
both metric CSVs → [plot_wt02_aligned_video_vicon_2d.py] → WT02_first100_video_metrics_2d/ + _vicon_metrics_yz_2d/
both metric CSVs → [plot_wt02_aligned_timeseries.py] → WT02_first100_aligned_timeseries/ (CSV + 4 PNGs)
```

### 7-Segment Anthropometric Model (`compute_frame_com`)

Defined in `com_drawing/calculate_com.py` (and also `calculate_com.py` at repo root). Uses COCO keypoint indices. Returns `None` if any required keypoint is missing. Segments and their mass fractions:

| Segment | Mass fraction | Proximal ratio |
|---|---|---|
| trunk + head + neck | 0.578 | 0.660 |
| left/right total arm | 0.050 | 0.530 |
| left/right foot+leg | 0.061 | 0.606 |
| left/right thigh | 0.100 | 0.433 |

### Kinematic Metrics Model

Computed in `video_com_metric.py` (video) and `validate_com_normalized.py` (Vicon). Key quantities:

- **pixels_per_meter** (video only): nose-to-ankle pixel height / `REFERENCE_HEIGHT_M` (1.70 m), computed per-frame
- **l**: CoM height above ground — video: `(ground_y_px − com_y_px) / pixels_per_meter`; Vicon: `com_z_m`
- **ω₀**: `sqrt(g / l)` — natural frequency of inverted pendulum
- **xCoM** (extrapolated CoM): `CoM_pos + velocity / ω₀` — Hof et al. margin-of-stability model
- **velocity**: video uses finite differences × fps; Vicon uses `np.gradient`

Sign conventions for video: pixel Y increases downward; `com_y_m_up = −com_y_px / pixels_per_meter`; displacement/velocity upward components are sign-flipped accordingly.

### Script Responsibilities

| Script | Inputs | Outputs | Path assumptions |
|---|---|---|---|
| `video_pct_estimated.py` | image folders | `keypoints_and_com.json` + annotated JPGs | defaults to `H:\VICON\...` — pass explicit `--input-dir`/`--output-dir` |
| `video_estimated_com.py` | `keypoints_and_com.json` + raw frames | `video_com_pred/` images | `PCT_RESULTS_ROOT` set relative to `__file__` |
| `vicon_skeleton_drawing.py` | Vicon CSV | PNG frames per frame | `CSV_PATH` hardcoded to `ROOT/Chenzixuan_20260505_test/WT02.csv` |
| `validate_com_normalized.py` | `keypoints_and_com.json` + Vicon CSV + manifest | `summary.csv` + `plots_aligned/` PNGs + `vicon_com_metric.csv` | paths hardcoded to `H:\COM\video-vicon\...` |
| `validate_keypoints_normalized.py` | `keypoints_and_com.json` + Vicon CSV + manifest | `keypoint_summary.csv` + `keypoint_plots/{kp_name}/{trial}_{axis}.png` | paths hardcoded to `H:\COM\video-vicon\...` |
| `add_labels_to_xcorr_plots.py` | `validation/keypoints_xcorr/` PNGs | same PNGs annotated in-place | path relative to `__file__` |
| `video_com_metric.py` | `keypoints_and_com.json` | `validation/video_com_metric.csv` | paths hardcoded to `H:\COM\video-vicon\...` |
| `plot_wt02_kinematics_3d.py` | `vicon_com_metric.csv` + `WT02.csv` | `WT02_first50_kinematics_3d/` + `WT02_first50_kinematics_yz_2d/` | paths hardcoded to `H:\COM\video-vicon\...` |
| `plot_wt02_video_metrics_2d.py` | `video_com_metric.csv` + `keypoints_and_com.json` | `WT02_first50_video_metrics_2d/` | paths hardcoded to `H:\COM\video-vicon\...` |
| `plot_wt02_aligned_video_vicon_2d.py` | both metric CSVs + keypoints JSON + WT02.csv | `WT02_first100_video_metrics_2d/` + `WT02_first100_vicon_metrics_yz_2d/` | paths hardcoded to `H:\COM\video-vicon\...` |
| `plot_wt02_aligned_timeseries.py` | (delegates to `build_aligned_selections()`) | `WT02_first100_aligned_timeseries/` CSV + 4 PNGs | paths hardcoded to `H:\COM\video-vicon\...` |

### Data Layout

```
video-vicon/
  code/
    validate_com_normalized.py          ← video CoM vs Vicon CentreOfMass; writes vicon_com_metric.csv
    validate_keypoints_normalized.py    ← COCO-17 keypoints vs Vicon Trajectories
    vicon_skeleton_drawing.py
    video_pct_estimated.py
    video_estimated_com.py
    video_com_metric.py                 ← per-frame video kinematic metrics (vel, xCoM, l, ω₀)
    plot_wt02_kinematics_3d.py          ← WT02 Vicon 3D/YZ skeleton + CoM/xCoM vectors
    plot_wt02_video_metrics_2d.py       ← WT02 video 2D keypoints + CoM/xCoM vectors
    plot_wt02_aligned_video_vicon_2d.py ← WT02 time-aligned video+Vicon side-by-side 2D frames
    plot_wt02_aligned_timeseries.py     ← WT02 video vs Vicon kinematic time-series comparison
    add_labels_to_xcorr_plots.py        ← annotates keypoints_xcorr/ PNGs with mapping labels
    tests/
  data/Chenzixuan/
    Vicon/
      rawdata/Chenzixuan_20260505_test/ ← Vicon CSV files (WT02.csv, WT06.csv, …)
        xcp_backup_before_video_time_alignment/
    Video/
      Rawvideo/                         ← original iPhone MOV
      Video_trial/                      ← per-trial cut video segments + manifest CSV
        Video_ViconTrial_manifest.csv   ← trial → first_frame, last_frame, trajectory_rate_hz
      Video_trial_pic/                  ← frames extracted from trial videos
      Video_keypoint-com/
        results/
          keypoints.json                ← legacy (1028 records, all person_id=0)
          keypoints_and_com.json        ← current; image paths use Video_{trial}_Trajectory/
          <Video_WTXX_pred>/
            video_pct_pred/             ← raw frames extracted from trial video
            video_com_pred/             ← frames with CoM overlay drawn
  validation/
    summary.csv                         ← per-trial CoM validation metrics (Pearson r, nRMSE, xcorr)
    keypoint_summary.csv                ← per-trial per-keypoint validation metrics
    com_correlation.csv                 ← updated/renamed version of summary.csv
    keypoints_correlation.csv           ← updated/renamed version of keypoint_summary.csv
    video_com_metric.csv                ← per-frame video kinematic metrics (all trials)
    vicon_com_metric.csv                ← per-frame Vicon CoM kinematics (all trials)
    plots_aligned/                      ← CoM lag-aligned z-score overlay plots ({trial}_{axis}.png)
    com_z-score_detrend/                ← CoM z-score detrended overlay plots (newer run)
    com_z-score_detrend_xcorr/          ← CoM xcorr-aligned plots (newer run)
    keypoint_plots/                     ← per-keypoint validation plots (older)
    keypoints_xcorr/                    ← per-keypoint xcorr-aligned plots (newer run)
      kpNN_<name>/                      ← one plot per trial per axis, annotated with mapping label
    WT02_first50_kinematics_3d/         ← Vicon 3D skeleton + CoM/xCoM per frame
    WT02_first50_kinematics_yz_2d/      ← Vicon YZ front-view 2D per frame
    WT02_first50_video_metrics_2d/      ← video 2D keypoints + CoM/xCoM per frame
    WT02_first100_video_metrics_2d/     ← time-aligned video 2D frames (100 pairs)
    WT02_first100_vicon_metrics_yz_2d/  ← time-aligned Vicon YZ frames (100 pairs)
    WT02_first100_aligned_timeseries/   ← time-series CSV + 4 comparison PNGs
```

Trial naming: `WT02`, `WT06`–`WT10`, `WTFAST11`, `WTFAST14`, `WTFAST18`–`WTFAST22`.  
All trial result folders follow the naming `Video_WTXX_pred/`.

## Critical Constraints

**Two coordinate systems — never mix without explicit conversion:**
- Video CoM: 2D pixel coordinates (origin top-left, Y increases downward)
- Vicon CoM: 3D world coordinates in millimeters

**Three timelines — always label which one you're on:**
1. Original iPhone video timeline (full 311 s recording)
2. Cut trial video timeline (per-trial segments)
3. Vicon trial frame timeline (250 Hz, frame numbers from CSV)

**Synchronization anchor:**
- Video `2026-05-05 08:27:27.700` ↔ Vicon `2026-05-05 08:27:37.454`
- Offset: Vicon is 9.754 s ahead of video
- `.xcp` files in `rawdata/` already have corrected capture times; backups are in `xcp_backup_before_video_time_alignment/`

**Vicon CSV format:** Multi-section (Devices / Model Outputs / Trajectories). `parse_trajectories()` in `vicon_skeleton_drawing.py` shows the correct parsing approach — skip to the `Trajectories` section header, then read rate, marker names, axis labels, units, then data rows starting at offset +5.

**Do not overwrite:** raw frames, raw Vicon CSV files, or `.xcp` files without backing up first.

**keypoints_and_com.json image path format:** Current format uses `Video_{trial}_Trajectory/{stem}_{frame:04d}.jpg`. The legacy format `Video_{trial}_pred/video_pct_pred/` is no longer used.

**vicon_com_metric.csv columns:** `trial, frame, time_s, com_x_m, com_y_m, com_z_m, displacement_{x,y,z}_m, displacement_m, velocity_{x,y,z}_m_s, velocity_m_s, omega0_rad_s, xcom_{x,y,z}_m`

**video_com_metric.csv columns:** `trial, frame, time_s, pixels_per_meter, reference_height_m, nose_to_ankle_height_px, nose_y_px, ground_y_px, com_{x,y}_px, com_x_m, com_y_m_up, l_px, l_m, displacement_{x,y}_px, displacement_{x,y_m_up}_m, displacement_m, velocity_{x,y_m_s_up}_m_s, velocity_m_s, omega0_rad_s, xcom_{x,y}_m, xcom_{x,y}_px`

### COCO-17 → Vicon Marker Mapping

Defined in `validate_keypoints_normalized.py`. Used for keypoint-level validation.

| COCO keypoint | Vicon marker |
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

Sign alignment: `video_x ↔ −Vicon_Y`，`video_y ↔ −Vicon_Z`（video Y 向下，Vicon Z 向上）。

## Validation Pipeline (validate_com_normalized.py / validate_keypoints_normalized.py)

Both scripts follow the same processing steps per trial per axis:

1. **Time alignment** — Vicon 250Hz linearly interpolated onto video ~30Hz time points (`np.interp`)
2. **Sign alignment** — `vicon_Y_aligned = -vicon_Y`，`vicon_Z_aligned = -vicon_Z`
3. **Linear detrend** — `scipy.signal.detrend(type="linear")` removes baseline drift from both signals
4. **Z-score normalize** — zero mean, unit std; makes pixel and mm units comparable
5. **Pearson r + nRMSE** — correlation and normalized error on z-scored signals
6. **xcorr** — `np.correlate(full)` finds optimal lag; `lag<0` means Vicon leads video
7. **Lag-aligned plot** — signals shifted by xcorr lag before plotting overlay

Outputs per signal pair: `pearson_r`, `p_value`, `nrmse`, `n_frames`, `xcorr_peak_r`, `xcorr_lag_frames`, `xcorr_lag_ms`.

## Kinematic Comparison Pipeline

`plot_wt02_aligned_timeseries.py` compares these metrics between video and Vicon for WT02:

| Metric | Video | Vicon |
|---|---|---|
| displacement per interval | `displacement_m` (finite diff) | `hypot(Δy, Δz)` between adjacent Vicon frames |
| CoM velocity | `velocity_m_s` | `Δdisplacement / Δt` |
| l (pendulum length) | `l_m` (CoM height above ankle) | `com_z_m` |
| xCoM horizontal delta | `xcom_x_m − xcom_x_m[0]` | `−(xcom_y_m − xcom_y_m[0])` (axis flip) |
| xCoM vertical delta | `xcom_y_m_up − xcom_y_m_up[0]` | `xcom_z_m − xcom_z_m[0]` |

Axis alignment for xCoM comparison: `video_x ↔ −Vicon_Y`, `video_y_up ↔ Vicon_Z`.

## Open Research Questions

1. ~~Where is Vicon gold-standard CoM?~~ → Resolved: parsed from `Model Outputs` section column `:CentreOfMass` (mm).
2. ~~Which Vicon axis maps to video X/Y?~~ → Resolved: camera faces Vicon −X direction (front view); `video_x ↔ Vicon_Y` (ML), `video_y ↔ Vicon_Z` (vertical), both signs flipped.
3. Spatial comparison strategy: normalized 1D trend comparison (current approach) is the safe first pass; pixel-level comparison requires a validated Vicon→video projection.
4. Kinematic metric comparison (xCoM, velocity, l): time-series comparison implemented in `plot_wt02_aligned_timeseries.py` for WT02 first 100 frames — extend to all trials pending result review.
