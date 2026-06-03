"""validate_com_normalized.py 的单元测试：信号处理、CSV/JSON 解析、时间轴与运动学。"""

import numpy as np
import pytest
from pathlib import Path
from validate_com_normalized import zscore_normalize, compute_metrics


def test_zscore_normalize_basic():
    """z-score 后应满足均值≈0、标准差≈1。"""
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore_normalize(sig)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std() - 1.0) < 1e-10


def test_zscore_normalize_constant_raises():
    """常数信号标准差为 0，z-score 应抛出 ValueError。"""
    sig = np.array([3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="zero std"):
        zscore_normalize(sig)


def test_compute_metrics_perfect_match():
    """信号与自身比较：相关系数应为 1，nRMSE 应为 0。"""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r, p, nrmse = compute_metrics(a, a)
    assert abs(r - 1.0) < 1e-10
    assert abs(nrmse) < 1e-10


def test_compute_metrics_anti_correlation():
    """信号与其相反数比较：应为强负相关。"""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = -a
    r, p, nrmse = compute_metrics(a, b)
    assert r < -0.99


def test_compute_metrics_known_nrmse():
    """常数 0 与常数 1 比较：nRMSE 应等于 1。"""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    r, p, nrmse = compute_metrics(a, b)
    assert abs(nrmse - 1.0) < 1e-10


def test_parse_vicon_model_outputs_basic():
    """解析 Model Outputs 区段：采样率、帧号(应跳过空值帧)与 X/Y/Z 取值正确。"""
    # 构造一个带 Model Outputs 区段的 CSV 文本，第 3 帧 X/Y/Z 为空应被跳过。
    csv_text = "\n\nModel Outputs\n250\n,,Subject:CentreOfMass,,,\nFrame,Sub Frame,X,Y,Z\n,,mm,mm,mm\n1,0,100.0,200.0,300.0\n2,0,101.0,201.0,301.0\n3,0,,,\n4,0,103.0,203.0,303.0\n"
    import tempfile, os
    # 写入临时 CSV。
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_text)
        tmp = f.name
    try:
        from validate_com_normalized import parse_vicon_model_outputs
        rate, frames, cx, cy, cz = parse_vicon_model_outputs(Path(tmp))
        assert rate == 250.0
        # 第 3 帧因坐标为空被跳过，仅保留 1/2/4。
        assert frames == [1, 2, 4]
        assert abs(cx[0] - 100.0) < 1e-6
        assert abs(cy[1] - 201.0) < 1e-6
        assert abs(cz[2] - 303.0) < 1e-6
    finally:
        os.unlink(tmp)


def test_parse_vicon_model_outputs_missing_section():
    """CSV 中缺少 Model Outputs 区段时应抛出 ValueError。"""
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
    """从 JSON 解析指定试验的视频 CoM：应只取该试验且 com 非空的帧。"""
    import json, tempfile, os
    records = [
        {"image": "Video_WT02_Trajectory\\raw_frames\\frame_000001.jpg", "com": {"com_x": 100.0, "com_y": 200.0}},
        {"image": "Video_WT02_Trajectory\\raw_frames\\frame_000002.jpg", "com": {"com_x": 101.0, "com_y": 201.0}},
        # 不同试验(WT06)应被过滤掉。
        {"image": "Video_WT06_Trajectory\\raw_frames\\frame_000001.jpg", "com": {"com_x": 999.0, "com_y": 999.0}},
        # com 为 None 的帧应被跳过。
        {"image": "Video_WT02_Trajectory\\raw_frames\\frame_000003.jpg", "com": None},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(records, f)
        tmp = f.name
    try:
        from validate_com_normalized import parse_video_com
        frame_nums, com_x, com_y = parse_video_com(Path(tmp), "WT02")
        # 仅保留 WT02 且 com 非空的帧 1、2。
        assert frame_nums == [1, 2]
        assert abs(com_x[0] - 100.0) < 1e-6
        assert abs(com_y[1] - 201.0) < 1e-6
    finally:
        os.unlink(tmp)


def test_parse_video_com_null_skipped():
    """当唯一记录的 com 为 None 时，应返回空帧列表。"""
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
    """解析 manifest CSV：应正确读出 first_frame 与 trajectory_rate_hz。"""
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
    """视频帧号 [1,2,3] 在 10fps 下应映射为 [0,0.1,0.2] 秒。"""
    from validate_com_normalized import build_video_time_axis
    t = build_video_time_axis([1, 2, 3], fps=10.0)
    np.testing.assert_allclose(t, [0.0, 0.1, 0.2], atol=1e-10)


def test_build_vicon_time_axis():
    """Vicon 帧号相对 first_frame、按 250Hz 应映射为 [0,0.004,0.008] 秒。"""
    from validate_com_normalized import build_vicon_time_axis
    t = build_vicon_time_axis([256, 257, 258], first_frame=256, rate_hz=250.0)
    np.testing.assert_allclose(t, [0.0, 0.004, 0.008], atol=1e-10)


def test_interpolate_vicon_to_video():
    """线性插值：在区间中点处应得到相邻值的平均。"""
    from validate_com_normalized import interpolate_vicon_to_video
    vicon_t = np.array([0.0, 0.1, 0.2, 0.3])
    vicon_vals = np.array([0.0, 10.0, 20.0, 30.0])
    video_t = np.array([0.05, 0.15, 0.25])
    result = interpolate_vicon_to_video(video_t, vicon_t, vicon_vals)
    np.testing.assert_allclose(result, [5.0, 15.0, 25.0], atol=1e-10)


def test_compute_com_kinematics_uses_step_displacement_velocity_and_frame_height_for_xcom():
    """验证 Vicon 运动学：逐帧位移、梯度速度、ω₀(用每帧高度)与 xCoM 的计算。"""
    from validate_com_normalized import compute_com_kinematics

    rows = compute_com_kinematics(
        trial="WT_TEST",
        frame_numbers=[10, 11, 12],
        time_s=np.array([0.0, 1.0, 2.0]),
        com_x_mm=np.array([1000.0, 2000.0, 3000.0]),
        com_y_mm=np.array([0.0, 0.0, 0.0]),
        com_z_mm=np.array([1000.0, 1000.0, 4000.0]),
    )

    # 基本字段透传。
    assert rows[0]["trial"] == "WT_TEST"
    assert rows[0]["frame"] == 10
    assert rows[2]["time_s"] == 2.0

    # 逐帧位移(米)：首帧 0，其后为相邻帧差(1m、1m)。
    np.testing.assert_allclose(
        [row["displacement_x_m"] for row in rows],
        [0.0, 1.0, 1.0],
        atol=1e-10,
    )
    # 位移模长：第三帧含 z 方向 3m → sqrt(1²+3²)=sqrt(10)。
    np.testing.assert_allclose(
        [row["displacement_m"] for row in rows],
        [0.0, 1.0, np.sqrt(10.0)],
        atol=1e-10,
    )
    # 速度用 np.gradient：x 恒为 1m/s。
    np.testing.assert_allclose(
        [row["velocity_x_m_s"] for row in rows],
        [1.0, 1.0, 1.0],
        atol=1e-10,
    )
    # z 方向梯度速度：[0, 1.5, 3.0]。
    np.testing.assert_allclose(
        [row["velocity_z_m_s"] for row in rows],
        [0.0, 1.5, 3.0],
        atol=1e-10,
    )

    # ω₀ = sqrt(g / 高度z)，高度为 [1,1,4] m。
    expected_omega0 = np.sqrt(9.81 / np.array([1.0, 1.0, 4.0]))
    np.testing.assert_allclose(
        [row["omega0_rad_s"] for row in rows],
        expected_omega0,
        atol=1e-10,
    )
    # xCoM = CoM + 速度/ω₀(水平 x)。
    np.testing.assert_allclose(
        [row["xcom_x_m"] for row in rows],
        np.array([1.0, 2.0, 3.0]) + (1.0 / expected_omega0),
        atol=1e-10,
    )
    # xCoM(竖直 z)。
    np.testing.assert_allclose(
        [row["xcom_z_m"] for row in rows],
        np.array([1.0, 1.0, 4.0]) + (np.array([0.0, 1.5, 3.0]) / expected_omega0),
        atol=1e-10,
    )


def test_compute_com_kinematics_rejects_non_positive_frame_height():
    """任一帧 CoM 高度非正时，xCoM 计算应抛出 ValueError。"""
    from validate_com_normalized import compute_com_kinematics

    with pytest.raises(ValueError, match="positive CoM height"):
        compute_com_kinematics(
            trial="WT_TEST",
            frame_numbers=[1, 2],
            time_s=np.array([0.0, 1.0]),
            com_x_mm=np.array([0.0, 1.0]),
            com_y_mm=np.array([0.0, 1.0]),
            # 第二帧高度为 0，触发校验失败。
            com_z_mm=np.array([1000.0, 0.0]),
        )
