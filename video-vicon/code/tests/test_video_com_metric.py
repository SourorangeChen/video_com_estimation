"""video_com_metric.py 的单元测试：逐帧像素标定、运动学量与平滑函数。"""

import numpy as np
import pytest


def test_compute_video_com_metrics_uses_framewise_pixel_per_meter():
    """验证逐帧 pixels_per_meter、摆长 l、位移、速度、ω₀、xCoM 的计算是否正确。"""
    from video_com_metric import compute_video_com_metrics

    rows = compute_video_com_metrics(
        trial="WT_TEST",
        frame_numbers=[1, 2, 3],
        com_x_px=np.array([100.0, 110.0, 130.0]),
        com_y_px=np.array([200.0, 190.0, 170.0]),
        nose_y_px=np.array([100.0, 100.0, 100.0]),
        left_ankle_y_px=np.array([270.0, 440.0, 440.0]),
        right_ankle_y_px=np.array([260.0, 430.0, 430.0]),
        fps=10.0,
        reference_height_m=1.70,
    )

    # 像素/米 = 鼻踝高度 / 1.70；这里三帧鼻踝高度为 170/340/340 → 100/200/200。
    np.testing.assert_allclose(
        [row["pixels_per_meter"] for row in rows],
        [100.0, 200.0, 200.0],
        atol=1e-10,
    )
    # 摆长 l = (地面y - CoM_y) / ppm。
    np.testing.assert_allclose(
        [row["l_m"] for row in rows],
        [0.7, 1.25, 1.35],
        atol=1e-10,
    )
    # 水平位移(米)：首帧 0，其后按像素差/ppm。
    np.testing.assert_allclose(
        [row["displacement_x_m"] for row in rows],
        [0.0, 0.05, 0.10],
        atol=1e-10,
    )
    # 竖直位移(向上为正)。
    np.testing.assert_allclose(
        [row["displacement_y_m_up"] for row in rows],
        [0.0, 0.05, 0.10],
        atol=1e-10,
    )

    # 速度 = 位移 × fps(=10)。
    expected_vx = np.array([0.0, 0.5, 1.0])
    expected_vy = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose([row["velocity_x_m_s"] for row in rows], expected_vx, atol=1e-10)
    np.testing.assert_allclose([row["velocity_y_m_s_up"] for row in rows], expected_vy, atol=1e-10)

    # ω₀ = sqrt(g/l)，xCoM = CoM + 速度/ω₀。
    expected_omega0 = np.sqrt(9.81 / np.array([0.7, 1.25, 1.35]))
    np.testing.assert_allclose([row["omega0_rad_s"] for row in rows], expected_omega0, atol=1e-10)
    np.testing.assert_allclose(
        [row["xcom_x_m"] for row in rows],
        np.array([1.0, 0.55, 0.65]) + expected_vx / expected_omega0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        [row["xcom_y_m_up"] for row in rows],
        np.array([-2.0, -0.95, -0.85]) + expected_vy / expected_omega0,
        atol=1e-10,
    )


def test_compute_video_com_metrics_rejects_invalid_scale_height():
    """鼻踝高度非正(无法标定)时应抛出 ValueError。"""
    from video_com_metric import compute_video_com_metrics

    with pytest.raises(ValueError, match="positive nose-to-ankle"):
        compute_video_com_metrics(
            trial="WT_TEST",
            frame_numbers=[1, 2],
            com_x_px=np.array([100.0, 110.0]),
            com_y_px=np.array([200.0, 190.0]),
            # 首帧鼻与踝同高 → 鼻踝高度为 0，触发校验失败。
            nose_y_px=np.array([100.0, 100.0]),
            left_ankle_y_px=np.array([100.0, 270.0]),
            right_ankle_y_px=np.array([100.0, 260.0]),
            fps=10.0,
            reference_height_m=1.70,
        )


def test_smooth_signal_centered_moving_average_preserves_edges():
    """验证滑动平均仅平滑中间、两端保留原值的行为。"""
    from video_com_metric import smooth_signal

    # 窗口 3 时，中间 3 个点变为邻域均值(9/3=3)，首尾两端保留原值 0。
    result = smooth_signal(np.array([0.0, 0.0, 9.0, 0.0, 0.0]), window=3)

    np.testing.assert_allclose(result, [0.0, 3.0, 3.0, 3.0, 0.0], atol=1e-10)
