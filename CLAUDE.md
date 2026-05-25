# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a **biomechanics research validation pipeline** — not a software product. The goal is to validate video-based 2D center-of-mass (CoM) estimates (from iPhone + PCT pose estimation) against Vicon motion capture as a gold standard for gait and balance analysis.

Subject: Chenzixuan. Data lives in `com-vicon/data/Chenzixuan/`.

## Running Scripts

All scripts are standalone Python files. Run from the repo root:

```powershell
# PCT keypoint inference (requires GPU + mmdet + PCT at H:\PCT)
python com-vicon\code\video_pct_estimated.py --input-dir <path> --output-dir <path>

# Draw video CoM overlays from keypoints.json
python com-vicon\code\video_estimated_com.py

# Draw Vicon 3D skeleton frames from CSV
python com-vicon\code\vicon_skeleton_drawing.py

# Compute CoM from ankle JSON (separate patient dataset H:\Camera_data)
python calculate_com.py <patient_name>

# Plot CoM x-t and y-t timeseries (for H:\Camera_data patients)
python plot_com_timeseries.py <patient_name>
```

Core reusable function (import directly rather than running the CLI):
```python
from calculate_com import compute_frame_com  # takes COCO keypoints list, returns {"com_x": float, "com_y": float} or None
```

## Architecture

### Data Flow

```
iPhone MOV → [FFmpeg cut by Vicon trial timing] → per-trial video segments
                                                         ↓
                                              [video_pct_estimated.py]
                                        mmdet person detection + PCT pose
                                                         ↓
                                              keypoints.json (COCO-style)
                                                         ↓
                                              [video_estimated_com.py]
                                        imports compute_frame_com() → draws blue dot
                                                         ↓
                                            com_estimated_frames/ images

Vicon CSV (multi-section) → [vicon_skeleton_drawing.py] → 3D skeleton PNGs
                           → [future validation script] → CoM time-series comparison
```

### 7-Segment Anthropometric Model (`compute_frame_com`)

Defined in `com_drawing/calculate_com.py` (and also `calculate_com.py` at repo root). Uses COCO keypoint indices. Returns `None` if any required keypoint is missing. Segments and their mass fractions:

| Segment | Mass fraction | Proximal ratio |
|---|---|---|
| trunk + head + neck | 0.578 | 0.660 |
| left/right total arm | 0.050 | 0.530 |
| left/right foot+leg | 0.061 | 0.606 |
| left/right thigh | 0.100 | 0.433 |

### Script Responsibilities

| Script | Inputs | Outputs | Path assumptions |
|---|---|---|---|
| `video_pct_estimated.py` | image folders | `keypoints.json` + annotated JPGs | defaults to `H:\VICON\...` — pass explicit `--input-dir`/`--output-dir` |
| `video_estimated_com.py` | `keypoints.json` + raw frames | `com_estimated_frames/` images | `PCT_RESULTS_ROOT` is set relative to `__file__` — needs path fix for current data layout |
| `vicon_skeleton_drawing.py` | Vicon CSV | PNG frames per frame | `CSV_PATH` hardcoded to `ROOT/Chenzixuan_20260505_test/WT02.csv` — update before use |
| `calculate_com.py` (CLI) | `H:\Camera_data` Excel + JSON | `_COM.json` + overlay images | hardcoded to `H:\Camera_data` — only the `compute_frame_com()` function is portable |
| `plot_com_timeseries.py` (CLI) | `_COM.json` | `_x_t.png`, `_y_t.png` | hardcoded to `H:\Camera_data` |

### Data Layout

```
com-vicon/data/Chenzixuan/
  Vicon/
    rawdata/Chenzixuan_20260505_test/   ← Vicon CSV files (WT02.csv, WT06.csv, …)
      xcp_backup_before_video_time_alignment/
  Video/
    Rawvideo/                           ← original iPhone MOV
    Video_trial/                        ← per-trial cut video segments + manifest CSV
    Video_trial_pic/                    ← frames extracted from trial videos
    Video_keypoint-com/
      results/
        keypoints.json                  ← 1028 person records, all person_id=0
        <Video_WTXX_Trajectory>/
          raw_frames/
          com_estimated_frames/
```

Trial naming: `WT02`, `WT06`–`WT10`, `WTFAST11`, `WTFAST14`, `WTFAST18`–`WTFAST22`.  
WT02 uses older folder names `video_pct_estimated/` and `video_com_estimated/` instead of the standard `raw_frames/` and `com_estimated_frames/`.

## Critical Constraints

**Two coordinate systems — never mix without explicit conversion:**
- Video CoM: 2D pixel coordinates (origin top-left)
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

## Open Research Questions

1. Where is Vicon gold-standard CoM? Check `Model Outputs` section of CSV, or compute from markers + anthropometric segments.
2. Which camera view (front/back vs. side) determines which Vicon axis to compare against video X or Y.
3. Spatial comparison strategy: normalized 1D trend comparison is the safe first pass; pixel-level comparison requires a validated Vicon→video projection.
