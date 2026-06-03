"""plot_wt02_aligned_timeseries.py 的单元测试：逐区间位移/速度/摆长对比量。"""


def test_build_interval_comparison_rows_uses_matched_vicon_interval():
    """首帧区间量应为 0；后续帧应基于相邻 Vicon 帧计算位移/速度，并正确透传 l。"""
    from plot_wt02_aligned_timeseries import build_interval_comparison_rows

    matched = [
        (
            {"frame": 1, "time_s": 0.0, "displacement_m": 0.0, "velocity_m_s": 0.0, "l_m": 1.0, "xcom_x_m": 1.0, "xcom_y_m_up": 2.0},
            {"frame": 10, "time_s": 0.0, "com_y_m": 0.0, "com_z_m": 1.0, "xcom_y_m": 0.0, "xcom_z_m": 1.0},
        ),
        (
            {"frame": 2, "time_s": 0.1, "displacement_m": 0.2, "velocity_m_s": 2.0, "l_m": 1.1, "xcom_x_m": 1.2, "xcom_y_m_up": 2.2},
            {"frame": 20, "time_s": 0.1, "com_y_m": 0.3, "com_z_m": 1.4, "xcom_y_m": 0.4, "xcom_z_m": 1.5},
        ),
    ]

    rows = build_interval_comparison_rows(matched)

    # 首帧无上一帧，位移与速度均为 0。
    assert rows[0]["vicon_displacement_yz_m"] == 0.0
    assert rows[0]["vicon_vcom_yz_m_s"] == 0.0
    # 第二帧 Vicon 位移 = hypot(Δy=0.3, Δz=0.4) = 0.5。
    assert round(rows[1]["vicon_displacement_yz_m"], 6) == 0.5
    # 速度 = 位移 0.5 / Δt 0.1 = 5.0。
    assert round(rows[1]["vicon_vcom_yz_m_s"], 6) == 5.0
    # 摆长 l：视频取 l_m，Vicon 取 com_z_m。
    assert rows[1]["video_l_m"] == 1.1
    assert rows[1]["vicon_l_m"] == 1.4
