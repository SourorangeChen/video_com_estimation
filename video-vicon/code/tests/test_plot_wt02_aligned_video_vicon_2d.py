def test_match_video_rows_to_nearest_vicon_rows_uses_first_overlap():
    from plot_wt02_aligned_video_vicon_2d import match_video_rows_to_vicon_rows

    video_rows = [
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

    assert [(video["frame"], vicon["frame"]) for video, vicon in matched] == [(3, 101), (4, 102)]
