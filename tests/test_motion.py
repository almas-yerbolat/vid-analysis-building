from app.pipeline.motion import motion_curve


def test_moving_video_scores_higher_than_static(static_video, moving_video):
    static = [s for _, s in motion_curve(static_video)]
    moving = [s for _, s in motion_curve(moving_video)]
    assert len(static) >= 3
    assert sum(moving) / len(moving) > sum(static) / len(static)


def test_timestamps_monotonic(moving_video):
    ts = [t for t, _ in motion_curve(moving_video)]
    assert ts == sorted(ts)
    assert ts[-1] <= 13.0
