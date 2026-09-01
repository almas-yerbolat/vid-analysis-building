from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app import storage
from app.models import Analysis, Base, Frame, Video
from app.vlm.analyze import analyze_frames
from app.vlm.client import FakeVlmClient


class FlakyClient(FakeVlmClient):
    def __init__(self):
        self.calls = 0

    def analyze_batch(self, *a, **k):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return super().analyze_batch(*a, **k)


def setup(tmp_path, monkeypatch, n_frames):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = Session(engine)
    v = Video(filename="a.mp4", media_key="videos/v/a.mp4")
    s.add(v); s.flush()
    frames = []
    for i in range(n_frames):
        key = storage.save_bytes(f"frames/{v.id}/{i * 5000}.jpg", b"fakejpeg")
        f = Frame(video_id=v.id, ts_ms=i * 5000, media_key=key, thumb_key=key,
                  width=100, height=100)
        s.add(f); frames.append(f)
    s.flush()
    return s, v, frames


def test_batches_of_four_and_persists(tmp_path, monkeypatch):
    s, v, frames = setup(tmp_path, monkeypatch, 6)
    progress = []
    result = analyze_frames(s, v, frames, FakeVlmClient(),
                            on_progress=lambda d, t: progress.append((d, t)))
    assert len(result) == 6 and [r.ts_ms for r in result] == sorted(r.ts_ms for r in result)
    rows = s.execute(select(Analysis)).scalars().all()
    assert len(rows) == 2  # 6 frames → batches of 4 + 2
    assert all(r.status == "ok" for r in rows)
    assert progress[-1] == (2, 2)


def test_retry_once_then_succeed(tmp_path, monkeypatch):
    s, v, frames = setup(tmp_path, monkeypatch, 2)
    client = FlakyClient()
    result = analyze_frames(s, v, frames, client)
    assert client.calls == 2 and len(result) == 2


def test_double_failure_flags_batch(tmp_path, monkeypatch):
    class DeadClient(FakeVlmClient):
        def analyze_batch(self, *a, **k):
            raise RuntimeError("dead")

    s, v, frames = setup(tmp_path, monkeypatch, 2)
    result = analyze_frames(s, v, frames, DeadClient())
    assert result == []
    row = s.execute(select(Analysis)).scalars().one()
    assert row.status == "failed"
