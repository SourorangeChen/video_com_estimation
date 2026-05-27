def test_select_matching_frames_uses_only_frames_with_metrics_and_points():
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

    assert [item[0]["frame"] for item in selected] == [10, 12]
    assert selected[0][1]["points"]["A"] == (1.0, 0.0, 0.0)


def test_yz_point_returns_y_and_z_only():
    from plot_wt02_kinematics_3d import yz_point

    assert yz_point((1.0, 2.0, 3.0)) == (2.0, 3.0)
