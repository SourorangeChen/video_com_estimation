import numpy as np


def test_aligned_zscore_axis_mapping_uses_video_x_to_negative_vicon_y_and_video_y_up_to_vicon_z():
    from plot_com_zscore_from_metric import build_axis_signals

    video_rows = [
        {"time_s": 0.0, "com_x_m": 1.0, "com_y_m_up": 2.0},
        {"time_s": 1.0, "com_x_m": 2.0, "com_y_m_up": 3.0},
    ]
    vicon_rows = [
        {"time_s": 0.0, "com_y_m": -1.0, "com_z_m": 2.0},
        {"time_s": 1.0, "com_y_m": -2.0, "com_z_m": 3.0},
    ]

    axes = build_axis_signals(video_rows, vicon_rows)

    np.testing.assert_allclose(axes["horizontal"]["video"], [1.0, 2.0])
    np.testing.assert_allclose(axes["horizontal"]["vicon"], [1.0, 2.0])
    np.testing.assert_allclose(axes["vertical"]["video"], [2.0, 3.0])
    np.testing.assert_allclose(axes["vertical"]["vicon"], [2.0, 3.0])
