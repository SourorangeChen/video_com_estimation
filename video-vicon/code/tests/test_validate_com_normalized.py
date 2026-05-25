import numpy as np
import pytest
from pathlib import Path
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
