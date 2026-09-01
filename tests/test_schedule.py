from app.pipeline.schedule import scene_cuts, schedule_keyframes


def test_baseline_every_5s_when_no_motion():
    curve = [(t / 4, 0.1) for t in range(1, 480)]  # 120 s of near-zero motion
    ks = schedule_keyframes(120.0, curve, [])
    assert [r for _, r in ks] == ["baseline"] * len(ks)
    assert 23 <= len(ks) <= 25  # ~one per 5 s


def test_fast_interval_densified():
    # calm everywhere except 20–25 s
    curve = [(t / 4.0, 8.0 if 20 <= t / 4.0 < 25 else 0.1) for t in range(1, 240)]
    ks = schedule_keyframes(60.0, curve, [])
    dense = [t for t, r in ks if r == "fast_motion"]
    assert dense, "fast interval must add fast_motion frames"
    assert all(20 <= t < 25 for t in dense)


def test_scene_cut_adds_frame_and_dedup():
    ks = schedule_keyframes(30.0, [(t / 4, 0.1) for t in range(1, 120)], [12.3])
    assert any(r == "scene_cut" and abs(t - 12.5) < 0.5 for t, r in ks)
    ts = [t for t, _ in ks]
    assert len(ts) == len(set(ts)) and ts == sorted(ts)


def test_scene_cuts_detected_on_hard_cuts(cut_video, static_video):
    from app.pipeline.motion import motion_curve
    assert len(scene_cuts(motion_curve(cut_video))) >= 1  # 3 palette segments → ≥1 cut
    assert scene_cuts(motion_curve(static_video)) == []   # no cuts in a static clip


def test_scene_cuts_needs_enough_samples():
    assert scene_cuts([(0.25, 99.0)]) == []


def test_photo_like_zero_duration():
    assert schedule_keyframes(0.4, [], []) == [(0.0, "baseline")]
