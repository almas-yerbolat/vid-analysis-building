import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import storage
from app.models import Base, Frame, Report, Video
from app.pipeline.run import run_pipeline
from app.vlm.client import FakeVlmClient
from tests.conftest import make_video


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    engine = create_engine("sqlite://", poolclass=None)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def test_end_to_end_video(session_factory, tmp_path):
    path = make_video(tmp_path / "clip.mp4", seconds=12, moving=True)
    with session_factory() as s:
        v = Video(filename="clip.mp4",
                  media_key=storage.save_upload("videos/x/clip.mp4", open(path, "rb")))
        s.add(v); s.commit()
        vid = v.id

    run_pipeline(vid, session_factory=session_factory, client=FakeVlmClient())

    with session_factory() as s:
        v = s.get(Video, vid)
        assert v.status == "done" and v.progress_pct == 100
        assert v.duration_s > 10
        frames = s.execute(select(Frame).where(Frame.video_id == vid)).scalars().all()
        assert len(frames) >= 1
        report = s.execute(select(Report).where(Report.video_id == vid)).scalars().one()
        assert report.report_json["stage"]["primary"] == "каркас"
        assert report.report_json["summary_ru"]


def test_pipeline_failure_sets_status(session_factory):
    with session_factory() as s:
        v = Video(filename="nope.mp4", media_key="videos/none/nope.mp4")
        s.add(v); s.commit()
        vid = v.id
    run_pipeline(vid, session_factory=session_factory, client=FakeVlmClient())
    with session_factory() as s:
        v = s.get(Video, vid)
        assert v.status == "failed" and v.error
