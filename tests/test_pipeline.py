import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app import storage
from app.models import Analysis, Base, Frame, Report, Video
from app.pipeline import run as pipeline_run
from app.pipeline.run import run_pipeline
from app.vlm.client import FakeVlmClient
from tests.conftest import make_video


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    engine = create_engine("sqlite://", poolclass=None)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def recording_session_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    engine = create_engine("sqlite://", poolclass=None)
    Base.metadata.create_all(engine)
    transitions = []

    class RecordingSession(Session):
        def commit(self):
            super().commit()
            transitions.extend(
                (video.status, video.progress_pct, video.progress_note)
                for video in self.identity_map.values()
                if isinstance(video, Video)
            )

    return sessionmaker(engine, class_=RecordingSession, expire_on_commit=False), transitions


def test_end_to_end_video(session_factory, tmp_path):
    path = make_video(tmp_path / "clip.mp4", seconds=12, moving=True)
    with session_factory() as s:
        with open(path, "rb") as upload:
            media_key = storage.save_upload("videos/x/clip.mp4", upload)
        v = Video(filename="clip.mp4", media_key=media_key)
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


def test_end_to_end_photo(session_factory, tmp_path):
    image = np.full((120, 160, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path = tmp_path / "photo.jpg"
    path.write_bytes(encoded.tobytes())
    with session_factory() as s:
        with open(path, "rb") as upload:
            media_key = storage.save_upload("videos/x/photo.jpg", upload)
        v = Video(filename="photo.jpg", media_key=media_key, is_photo=True)
        s.add(v); s.commit()
        vid = v.id

    run_pipeline(vid, session_factory=session_factory, client=FakeVlmClient())

    with session_factory() as s:
        v = s.get(Video, vid)
        frame = s.execute(select(Frame).where(Frame.video_id == vid)).scalars().one()
        report = s.execute(select(Report).where(Report.video_id == vid)).scalars().one()
        assert (v.status, v.progress_pct, v.duration_s) == ("done", 100, 0.0)
        assert (frame.ts_ms, frame.selected_reason) == (0, "photo")
        assert report.report_json["meta"]["frames_extracted"] == 1


def test_pipeline_persists_progress_through_each_analysis_batch(recording_session_factory, tmp_path):
    session_factory, transitions = recording_session_factory
    path = make_video(tmp_path / "clip.mp4", seconds=12, moving=True)
    with session_factory() as s:
        with open(path, "rb") as upload:
            media_key = storage.save_upload("videos/x/clip.mp4", upload)
        v = Video(filename="clip.mp4", media_key=media_key)
        s.add(v); s.commit()
        vid = v.id
    transitions.clear()

    run_pipeline(vid, session_factory=session_factory, client=FakeVlmClient())

    with session_factory() as s:
        batches = s.execute(select(Analysis).where(Analysis.video_id == vid)).scalars().all()
    analyzing = [(pct, note) for status, pct, note in transitions if status == "analyzing"]
    assert len(batches) >= 2
    assert analyzing[0][0] == 40
    assert [pct for pct, _ in analyzing[1:]] == [
        40 + round(50 * done / len(batches)) for done in range(1, len(batches) + 1)
    ]
    assert [(status, pct) for status, pct, _ in transitions if status != "analyzing"] == [
        ("probing", 5), ("sampling", 10), ("sampling", 25),
        ("aggregating", 92), ("done", 100),
    ]


def test_pipeline_failure_sets_status(session_factory):
    with session_factory() as s:
        v = Video(filename="nope.mp4", media_key="videos/none/nope.mp4")
        s.add(v); s.commit()
        vid = v.id
    run_pipeline(vid, session_factory=session_factory, client=FakeVlmClient())
    with session_factory() as s:
        v = s.get(Video, vid)
        assert v.status == "failed" and v.error


def test_default_client_setup_failure_sets_status(session_factory, monkeypatch):
    with session_factory() as s:
        v = Video(filename="clip.mp4", media_key="videos/x/clip.mp4")
        s.add(v); s.commit()
        vid = v.id

    def fail_client_setup():
        raise RuntimeError("invalid Gemini credentials")

    monkeypatch.setattr(pipeline_run, "get_client", fail_client_setup)
    run_pipeline(vid, session_factory=session_factory)

    with session_factory() as s:
        v = s.get(Video, vid)
        assert (v.status, v.error) == ("failed", "invalid Gemini credentials")
