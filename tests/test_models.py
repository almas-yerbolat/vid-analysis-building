from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base, Video, Frame, Finding, FindingFrame, Report


def make_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_video_frame_finding_roundtrip():
    s = make_session()
    v = Video(filename="a.mp4", media_key="videos/x/a.mp4")
    s.add(v)
    s.flush()
    assert v.id.startswith("vid_")
    assert v.status == "uploaded"

    f = Frame(video_id=v.id, ts_ms=5000, media_key="frames/x/5000.jpg",
              thumb_key="frames/x/5000_thumb.jpg", width=1568, height=882,
              motion_score=1.2, phash=123, selected_reason="baseline")
    s.add(f)
    s.flush()
    assert f.id.startswith("frm_")

    fnd = Finding(video_id=v.id, category="тб_от", subtype="отсутствие_каски",
                  severity="high", title="t", comment="c", confidence=0.8)
    s.add(fnd)
    s.flush()
    s.add(FindingFrame(finding_id=fnd.id, frame_id=f.id, frame_comment="c2",
                       boxes=[{"label": "рабочий без каски", "box_2d": [412, 806, 471, 843]}]))
    s.add(Report(video_id=v.id, report_json={"stats": {}}, summary_ru="s"))
    s.commit()

    got = s.get(FindingFrame, (fnd.id, f.id))
    assert got.boxes[0]["box_2d"] == [412, 806, 471, 843]
