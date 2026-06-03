"""plot_com_zscore_from_metric.py 的单元测试：CoM 轴对齐方向是否正确。"""

import numpy as np


def test_aligned_zscore_axis_mapping_uses_video_x_to_negative_vicon_y_and_video_y_up_to_vicon_z():
    """验证轴对齐：水平 video_x ↔ −Vicon_Y；竖直 video_y_up ↔ Vicon_Z。"""
    from plot_com_zscore_from_metric import build_axis_signals

    video_rows = [
        {"time_s": 0.0, "com_x_m": 1.0, "com_y_m_up": 2.0},
        {"time_s": 1.0, "com_x_m": 2.0, "com_y_m_up": 3.0},
    ]
    vicon_rows = [
        # Vicon_Y 取负后应与 video_x 同号(此处 -(-1)=1, -(-2)=2)。
        {"time_s": 0.0, "com_y_m": -1.0, "com_z_m": 2.0},
        {"time_s": 1.0, "com_y_m": -2.0, "com_z_m": 3.0},
    ]

    axes = build_axis_signals(video_rows, vicon_rows)

    # 水平：video 原值；Vicon 取 −Y 后变为 [1,2]。
    np.testing.assert_allclose(axes["horizontal"]["video"], [1.0, 2.0])
    np.testing.assert_allclose(axes["horizontal"]["vicon"], [1.0, 2.0])
    # 竖直：video_y_up 原值；Vicon 取 Z(不翻转)。
    np.testing.assert_allclose(axes["vertical"]["video"], [2.0, 3.0])
    np.testing.assert_allclose(axes["vertical"]["vicon"], [2.0, 3.0])
