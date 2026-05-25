# COM / Vicon Project Context for Agents

## Project Goal

This project is about validating a video-based center of mass (CoM) estimation method against Vicon-derived CoM as the gold standard.

The intended validation question is:

> Can CoM estimated from 2D human keypoints in video approximate Vicon CoM well enough for gait and balance analysis?

The current work should therefore be treated as a validation pipeline, not only as visualization.

## Current Workspace

Root workspace:

```text
H:\COM
```

Important files:

```text
H:\COM\calculate_com.py
H:\COM\plot_com_timeseries.py
H:\COM\COM.pptx
H:\COM\com-vicon\code\video_pct_estimated.py
H:\COM\com-vicon\code\video_estimated_com.py
H:\COM\com-vicon\code\vicon_skeleton_drawing.py
H:\COM\com-vicon\data\Chenzixuan
```

The workspace is not currently a Git repository.

## Research Background

The slide deck `COM.pptx` describes a balance assessment pipeline:

```text
human keypoints -> preprocessing -> CoM trajectory
segment pose + anthropometric model -> CoM metrics
```

Relevant downstream metrics include:

- CoM displacement
- CoM velocity
- extrapolated CoM (xCoM)
- base of support (BoS)
- margin of stability (MoS)
- angular momentum around CoM

Current code mainly implements the CoM trajectory step using a 7-segment anthropometric model.

## Video Data

The original iPhone MOV in the earlier working directory was:

```text
H:\VICON\Chenzixuan\Video\IMG_8467.MOV
```

Important metadata found previously:

- Device: Apple iPhone 15 Pro
- Resolution: `1920 x 1080`
- Display rotation: `-90 degrees`, so visually portrait
- Effective viewing size: about `1080 x 1920`
- Codec: HEVC / H.265
- Nominal frame rate: `30 fps`
- Average frame rate: about `29.996 fps`
- Duration: about `311.028333 s`
- Apple local creation date: `2026-05-05T08:27:01+0800`

Important caveat:

Apple/QuickTime timestamps are only reliable to about second-level precision here. They are useful for rough absolute timing, but not sufficient for millisecond/frame-accurate Vicon synchronization.

## Vicon / Video Synchronization

The earlier sessions established a manual timing anchor:

```text
Video timestamp:  2026-05-05 08:27:27.700
Vicon timestamp:  2026-05-05 08:27:37.454
Offset:           Vicon is 9.754 s ahead of video
```

Based on this, `.xcp` Capture `START_TIME` and `END_TIME` values were shifted backward by `9.754 s` in the earlier Vicon workspace. A backup directory was created before editing:

```text
H:\VICON\Chenzixuan\Chenzixuan_20260505_test\xcp_backup_before_video_time_alignment
```

In the current copied data under `H:\COM`, the analogous backup path is:

```text
H:\COM\com-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test\xcp_backup_before_video_time_alignment
```

Example corrected WT02 Capture time:

```text
Original WT02 Capture START: 08:27:35.898
Corrected WT02 Capture START: 08:27:26.144
```

## Trial Video Segments

Using corrected `.xcp` Capture times and CSV `Trajectories` frame ranges, 13 Vicon trial video segments were previously cut from the original iPhone video.

Manifest:

```text
H:\COM\com-vicon\data\Chenzixuan\Video\Video_trial\Video_ViconTrial_manifest.csv
```

Known trials:

```text
WT02
WT06
WT07
WT08
WT09
WT10
WTFAST11
WTFAST14
WTFAST18
WTFAST19
WTFAST20
WTFAST21
WTFAST22
```

The manifest columns include:

```text
trial, first_frame, last_frame, frames, trajectory_rate_hz,
capture_start, trajectory_start_time, video_offset_sec,
duration_sec, output
```

Example WT02:

```text
trial: WT02
first_frame: 256
last_frame: 1249
frames: 994
trajectory_rate_hz: 250.0
capture_start: 2026-05-05 08:27:26.144
trajectory_start_time: 2026-05-05 08:27:27.164
video_offset_sec: 26.164
duration_sec: 3.976
```

## Vicon CSV Structure

Vicon CSV files are multi-section exports, not simple single tables.

For WT02 and similar trials, sections include:

```text
Devices
Model Outputs
Trajectories
```

`Devices`:

- Usually `1000 Hz`
- Force plates FP1/FP2
- Force: `Fx, Fy, Fz`, unit `N`
- Moment: `Mx, My, Mz`, unit `N.mm`
- Center of pressure: `Cx, Cy, Cz`, unit `mm`
- Raw voltage pins
- Has `Sub Frame` values because force data is 1000 Hz while motion trajectories are 250 Hz

`Trajectories`:

- Usually `250 Hz`
- Marker 3D coordinates in `mm`
- Plug-in Gait marker names such as `LFHD`, `RFHD`, `C7`, `LSHO`, `LASI`, `LANK`, `LHEE`, `LTOE`, `RANK`, `RHEE`, `RTOE`

WT02 complete trial notes from earlier analysis:

- Full trial appears to be frame `1` to `1466` according to `WT02.history`
- Exported CSV `Trajectories` range is only frame `256` to `1249`
- CSV range therefore contains `994` frames

## Existing Python Scripts

### `calculate_com.py`

Path:

```text
H:\COM\calculate_com.py
```

Purpose:

- Defines the 7-segment model used to compute video-estimated CoM from 2D keypoints.
- Main reusable function is `compute_frame_com(keypoints)`.
- Uses COCO-style keypoint indices.
- Returns:

```python
{"com_x": float, "com_y": float}
```

Important issue:

The script's CLI path assumptions point to:

```text
H:\Camera_data
```

So the CLI flow is for a separate patient dataset, not directly for the current `com-vicon\data\Chenzixuan` layout. For this project, other scripts import `compute_frame_com()` rather than running the whole CLI.

### `plot_com_timeseries.py`

Path:

```text
H:\COM\plot_com_timeseries.py
```

Purpose:

- Reads `_COM.json` files produced by `calculate_com.py`.
- Plots video CoM x-time and y-time curves.

Also assumes:

```text
H:\Camera_data
```

### `video_pct_estimated.py`

Path:

```text
H:\COM\com-vicon\code\video_pct_estimated.py
```

Purpose:

- Batch runs mmdet + PCT on images.
- Produces skeleton/keypoint result images and `keypoints.json`.

Default paths currently point to older/external paths:

```text
H:\PCT
H:\VICON\Chenzixuan\Video\Video_ViconTrial_PCT
H:\VICON\Chenzixuan\Video\Video_Keypoint\pct_results
```

Use explicit arguments or update paths before reusing.

### `video_estimated_com.py`

Path:

```text
H:\COM\com-vicon\code\video_estimated_com.py
```

Purpose:

- Imports `compute_frame_com()` from `H:\COM\calculate_com.py`.
- Reads PCT `keypoints.json`.
- Draws a blue CoM point and blue text `estimated` on each image.

Important issue:

Its default `PCT_RESULTS_ROOT` is based on `WORKSPACE_ROOT / "Video" / "Video_Keypoint" / "pct_results"`, where `WORKSPACE_ROOT` is the script directory. In the current copied data, the actual results are under:

```text
H:\COM\com-vicon\data\Chenzixuan\Video\Video_keypoint-com\results
```

So this script may need path adjustment before direct execution.

### `vicon_skeleton_drawing.py`

Path:

```text
H:\COM\com-vicon\code\vicon_skeleton_drawing.py
```

Purpose:

- Parses a Vicon CSV `Trajectories` section.
- Draws a 3D skeleton image for each frame.
- Uses marker edge definitions for Plug-in Gait FullBody.

Current default path is probably stale:

```text
ROOT / "Chenzixuan_20260505_test" / "WT02.csv"
```

The actual copied data is under:

```text
H:\COM\com-vicon\data\Chenzixuan\Vicon\rawdata\Chenzixuan_20260505_test
```

## Existing Generated / Derived Data

PCT keypoint result JSON:

```text
H:\COM\com-vicon\data\Chenzixuan\Video\Video_keypoint-com\results\keypoints.json
```

This file contains 1028 person records. Earlier inspection found:

- all records have `person_id = 0`
- no missing keypoints
- trial counts:

```text
Video_WT02_Trajectory      120
Video_WT06_Trajectory      105
Video_WT07_Trajectory      104
Video_WT08_Trajectory      117
Video_WT09_Trajectory       93
Video_WT10_Trajectory       87
Video_WTFAST11_Trajectory   48
Video_WTFAST14_Trajectory   54
Video_WTFAST18_Trajectory   57
Video_WTFAST19_Trajectory   57
Video_WTFAST20_Trajectory   58
Video_WTFAST21_Trajectory   64
Video_WTFAST22_Trajectory   64
```

Most trial folders contain:

```text
raw_frames
com_estimated_frames
```

WT02 is slightly different in the copied data: it has older folder names such as:

```text
video_pct_estimated
video_com_estimated
```

## Important Conceptual Boundary

Do not directly subtract video CoM from Vicon CoM without spatial alignment.

Video-estimated CoM is:

```text
2D pixel coordinate
```

Vicon CoM is expected to be:

```text
3D world coordinate in mm
```

To compare them quantitatively, the project needs either:

1. A camera calibration / projection from Vicon 3D world to video pixels.
2. A simpler first-pass validation using normalized 1D time-series trends.

Recommended first-pass validation:

- Align timestamps.
- Resample/interpolate Vicon CoM to video frame timestamps.
- Compare normalized trajectories rather than raw units.
- For frontal/back views, compare the appropriate lateral/vertical projected component.
- For side views, compare AP/vertical projected components.

Only draw Vicon CoM as a red point on video if a valid Vicon-to-video projection or manual spatial mapping has been established. Otherwise the red point would look precise but be scientifically unreliable.

## Open Research Questions

These questions define the parts of the project that are not fully settled yet:

1. Where exactly is Vicon gold-standard CoM stored?
   - Is it present in CSV `Model Outputs`?
   - Or must it be calculated from Vicon markers and anthropometric segments?

2. Which camera view is being validated?
   - Front/back view?
   - Side view?
   - The phrase "前后视频" may mean front/back camera view, not AP direction.

3. What spatial comparison is acceptable for the current research stage?
   - normalized 1D trajectory trend
   - 2D projected pixel error
   - full 3D comparison after reconstruction

4. Should WT02's older folder naming be normalized to match the other trials?

## Practical Warnings

- Many scripts contain hard-coded absolute paths that point to `H:\VICON`, `H:\Camera_data`, or `H:\PCT`.
- The current active workspace is `H:\COM`, and the copied Chenzixuan data is under `H:\COM\com-vicon\data\Chenzixuan`.
- Do not assume `H:\VICON` exists in future sessions.
- Do not overwrite raw frames or raw Vicon files.
- If modifying `.xcp` files, always back them up first.
- If using video timestamps for synchronization, document uncertainty. Apple metadata is not a millisecond-accurate clock sync source.

## Quick Mental Model

The project has three timelines:

```text
Original iPhone video timeline
Cut trial video timeline
Vicon trial frame timeline
```

The project also has two coordinate systems:

```text
Video image pixels
Vicon lab coordinates in millimeters
```

Most bugs will come from mixing these timelines or coordinate systems without labeling the conversion.
