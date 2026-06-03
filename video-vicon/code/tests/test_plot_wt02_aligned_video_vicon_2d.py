"""plot_wt02_aligned_video_vicon_2d.py 的单元测试：视频帧到 Vicon 帧的时间最近邻匹配。"""


def test_match_video_rows_to_nearest_vicon_rows_uses_first_overlap():
    """超出 Vicon 时间范围的视频帧应被丢弃，其余取时间最近的 Vicon 帧。"""
    from plot_wt02_aligned_video_vicon_2d import match_video_rows_to_vicon_rows

    video_rows = [
        # 前两帧(0.00、0.10)早于 Vicon 起始时间 0.15，应被丢弃。
        {"trial": "WT02", "frame": 1, "time_s": 0.00},
        {"trial": "WT02", "frame": 2, "time_s": 0.10},
        {"trial": "WT02", "frame": 3, "time_s": 0.20},
        {"trial": "WT02", "frame": 4, "time_s": 0.30},
    ]
    vicon_rows = [
        {"trial": "WT02", "frame": 100, "time_s": 0.15},
        {"trial": "WT02", "frame": 101, "time_s": 0.19},
        {"trial": "WT02", "frame": 102, "time_s": 0.31},
    ]

    matched = match_video_rows_to_vicon_rows(video_rows, vicon_rows, limit=2)

    # 帧3(0.20)最近 Vicon 0.19→101；帧4(0.30)最近 Vicon 0.31→102。
    assert [(video["frame"], vicon["frame"]) for video, vicon in matched] == [(3, 101), (4, 102)]
