"""plot_wt02_kinematics_3d.py 的单元测试：帧配对、YZ 投影与 Y 轴镜像。"""


def test_select_matching_frames_uses_only_frames_with_metrics_and_points():
    """配对应只保留同时有运动学行与非空标记点的帧，并尊重数量上限。"""
    from plot_wt02_kinematics_3d import select_matching_frames

    kinematics_rows = [
        {"frame": 10, "com_x_m": 1.0},
        {"frame": 11, "com_x_m": 2.0},
        {"frame": 12, "com_x_m": 3.0},
    ]
    trajectory_frames = [
        {"frame": 9, "points": {"A": (0.0, 0.0, 0.0)}},
        {"frame": 10, "points": {"A": (1.0, 0.0, 0.0)}},
        {"frame": 12, "points": {"A": (2.0, 0.0, 0.0)}},
    ]

    selected = select_matching_frames(kinematics_rows, trajectory_frames, limit=2)

    # 帧 11 无对应轨迹被跳过；limit=2 故只取 10、12。
    assert [item[0]["frame"] for item in selected] == [10, 12]
    assert selected[0][1]["points"]["A"] == (1.0, 0.0, 0.0)


def test_yz_point_returns_y_and_z_only():
    """yz_point 应丢弃 x，只返回 (y, z)。"""
    from plot_wt02_kinematics_3d import yz_point

    assert yz_point((1.0, 2.0, 3.0)) == (2.0, 3.0)


def test_mirrored_y_limits_reverse_horizontal_axis():
    """mirrored_y_limits 应翻转 Y 轴上下限、保持 Z 轴不变。"""
    from plot_wt02_kinematics_3d import mirrored_y_limits

    assert mirrored_y_limits(((-0.2, 0.5), (0.0, 1.8))) == ((0.5, -0.2), (0.0, 1.8))
