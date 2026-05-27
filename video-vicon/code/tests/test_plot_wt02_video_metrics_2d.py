def test_select_trial_frames_matches_metrics_to_keypoints():
    from plot_wt02_video_metrics_2d import select_trial_frames

    metric_rows = [
        {"trial": "WT02", "frame": 1},
        {"trial": "WT02", "frame": 2},
        {"trial": "WT06", "frame": 1},
    ]
    keypoint_rows = {
        1: {"frame": 1, "keypoints": [[1.0, 2.0]]},
        3: {"frame": 3, "keypoints": [[3.0, 4.0]]},
    }

    selected = select_trial_frames(metric_rows, keypoint_rows, "WT02", 50)

    assert len(selected) == 1
    assert selected[0][0]["frame"] == 1
    assert selected[0][1]["keypoints"] == [[1.0, 2.0]]
