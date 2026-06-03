"""plot_wt02_video_metrics_2d.py 的单元测试：指标行与关键点的帧配对。"""


def test_select_trial_frames_matches_metrics_to_keypoints():
    """应只保留指定试验、且有对应关键点的帧。"""
    from plot_wt02_video_metrics_2d import select_trial_frames

    metric_rows = [
        {"trial": "WT02", "frame": 1},
        {"trial": "WT02", "frame": 2},
        # 不同试验应被过滤。
        {"trial": "WT06", "frame": 1},
    ]
    keypoint_rows = {
        1: {"frame": 1, "keypoints": [[1.0, 2.0]]},
        3: {"frame": 3, "keypoints": [[3.0, 4.0]]},
    }

    selected = select_trial_frames(metric_rows, keypoint_rows, "WT02", 50)

    # WT02 中只有帧 1 有对应关键点(帧 2 无关键点被跳过)。
    assert len(selected) == 1
    assert selected[0][0]["frame"] == 1
    assert selected[0][1]["keypoints"] == [[1.0, 2.0]]
