import numpy as np

from app.config import settings


def scene_cuts(curve: list[tuple[float, float]]) -> list[float]:
    """Hard cuts as spikes in the motion curve.

    The curve already holds frame-to-frame differences (§4.1 step 2); a separate
    scene-detection pass would decode the video again to rediscover the same numbers.
    A cut has to clear both a relative bar (`cut_ratio` × the median difference) and an
    absolute one (`cut_floor`), so a near-static clip whose median is ~0 yields no cuts.
    """
    scores = [s for _, s in curve]
    if len(scores) < 3:
        return []
    threshold = max(float(np.median(scores)) * settings.cut_ratio, settings.cut_floor)
    return [t for t, s in curve if s > threshold and t > 0.0]


def schedule_keyframes(
    duration_s: float,
    curve: list[tuple[float, float]],
    cuts: list[float],
) -> list[tuple[float, str]]:
    if duration_s <= settings.baseline_interval_s:
        picked = {0.0: "baseline"}
    else:
        picked = {}
    scores = np.array([s for _, s in curve]) if curve else np.array([0.0])
    t_fast = float(np.percentile(scores, settings.t_fast_percentile)) if curve else float("inf")

    t = 0.0
    while t < duration_s:
        end = min(t + settings.baseline_interval_s, duration_s)
        window = [s for ts, s in curve if t <= ts < end]
        fast = bool(window) and float(np.mean(window)) > t_fast and max(window) > 0.5
        step = settings.dense_interval_s if fast else settings.baseline_interval_s
        tt = t
        while tt < end:
            key = round(tt, 2)
            reason = "fast_motion" if fast and key != round(t, 2) else "baseline"
            picked.setdefault(key, reason)
            tt += step
        t += settings.baseline_interval_s

    for c in cuts:
        key = round(min(c + 0.2, max(duration_s - 0.05, 0.0)), 2)
        picked.setdefault(key, "scene_cut")

    return sorted(picked.items())
