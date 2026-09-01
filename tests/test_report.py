from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.aggregate import MergedFinding
from app.models import Base, Finding, FindingFrame, Frame, Report, Video
from app.report import build_report


def test_build_report_shape_and_persistence():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = Session(engine)
    v = Video(filename="a.mp4", media_key="k", duration_s=120.0)
    s.add(v); s.flush()
    frames = []
    for ts in (0, 5000, 30000):
        f = Frame(video_id=v.id, ts_ms=ts, media_key=f"f/{ts}.jpg",
                  thumb_key=f"f/{ts}_t.jpg", width=1568, height=882)
        s.add(f); frames.append(f)
    s.flush()

    merged = [MergedFinding(
        category="тб_от", subtype="отсутствие_каски", severity="high",
        title="Рабочие без касок", comment="Зафиксировано на кадрах 00:00–00:30. …",
        confidence=0.85,
        evidence=[(30000, "два рабочих", [{"label": "рабочий", "box_2d": [1, 2, 300, 400]}]),
                  (0, "один рабочий", [])],
    )]
    stage = {"primary": "каркас", "secondary": [], "confidence": 0.9, "evidence_ts": [0]}
    report = build_report(s, v, frames, 0, 1, merged, stage,
                          [{"type": "башенный_кран", "max_count": 1, "evidence_ts": 5000}],
                          [{"from_ms": 0, "to_ms": 30000, "activity": "Монтаж"}],
                          "Резюме.", frames_extracted=5, frames_analyzed=3)

    assert report["video_id"] == v.id
    assert report["stage"]["primary"] == "каркас"
    assert report["stats"] == {"critical": 0, "high": 1, "medium": 0, "low": 0}
    ev = report["findings"][0]["evidence"]
    assert ev[0]["ts_ms"] == 30000 and ev[0]["boxes"][0]["box_2d"] == [1, 2, 300, 400]
    assert ev[0]["full_url"].startswith("/api/frames/frm_")
    assert report["equipment"][0]["evidence_frame"] == frames[1].id
    assert report["meta"]["frames_analyzed"] == 3
    assert report["meta"]["coverage_pct"] == 100

    s.commit()
    assert s.execute(select(Finding)).scalars().one().severity == "high"
    assert len(s.execute(select(FindingFrame)).scalars().all()) == 2
    assert s.execute(select(Report)).scalars().one().report_json["stats"]["high"] == 1


def test_duplicate_nearest_frame_evidence_does_not_violate_composite_pk():
    """Two evidence timestamps that both fall short of any exact Frame row can both
    resolve to the SAME nearest Frame. finding_frames has a composite PK of
    (finding_id, frame_id), so inserting two FindingFrame rows for that pair must not
    happen — the second (lower-confidence) evidence entry should be dropped instead."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = Session(engine)
    v = Video(filename="a.mp4", media_key="k", duration_s=60.0)
    s.add(v); s.flush()
    frames = []
    for ts in (0, 30000):
        f = Frame(video_id=v.id, ts_ms=ts, media_key=f"f/{ts}.jpg",
                  thumb_key=f"f/{ts}_t.jpg", width=1568, height=882)
        s.add(f); frames.append(f)
    s.flush()

    # 10000 and 14999 both sit nearer to frame ts=0 than to frame ts=30000.
    merged = [MergedFinding(
        category="тб_от", subtype="отсутствие_каски", severity="high",
        title="Рабочие без касок", comment="…", confidence=0.85,
        evidence=[(10000, "первый", []), (14999, "второй", [])],
    )]
    stage = {"primary": "каркас", "secondary": [], "confidence": 0.9, "evidence_ts": []}
    report = build_report(s, v, frames, 0, 1, merged, stage, [], [], "Резюме.",
                          frames_extracted=2, frames_analyzed=2)

    ev = report["findings"][0]["evidence"]
    assert len(ev) == 1
    assert ev[0]["comment"] == "первый"

    s.commit()
    assert len(s.execute(select(FindingFrame)).scalars().all()) == 1
