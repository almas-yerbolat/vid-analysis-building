from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app import storage
from app.models import Analysis, Base, Frame, Video
from app.report import build_report
from app.vlm.analyze import analyze_frames
from app.vlm.client import FakeVlmClient, VlmResult
from app.vlm.schema import BatchResponse


class FlakyClient(FakeVlmClient):
    def __init__(self):
        self.calls = 0

    def analyze_batch(self, *a, **k):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return super().analyze_batch(*a, **k)


def setup(tmp_path, monkeypatch, n_frames, first_ts_ms=0):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = Session(engine)
    v = Video(filename="a.mp4", media_key="videos/v/a.mp4")
    s.add(v); s.flush()
    frames = []
    for i in range(n_frames):
        ts = first_ts_ms + i * 5000
        key = storage.save_bytes(f"frames/{v.id}/{ts}.jpg", b"fakejpeg")
        f = Frame(video_id=v.id, ts_ms=ts, media_key=key, thumb_key=key,
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


class DirtyClient(FakeVlmClient):
    """Returns output the sanitizers (BatchResponse.clean / validate_boxes) will alter,
    so raw_response snapshotting can be tested against real mutation."""

    def analyze_batch(self, images, ts_ms, low_quality, project_name) -> VlmResult:
        parsed = BatchResponse.model_validate({"frames": [{
            "ts_ms": ts_ms[0], "stage": "каркас", "stage_confidence": 0.9,
            "activity": "Монтаж опалубки", "equipment": [],
            "findings": [
                {  # category/subtype mismatch: dropped whole by clean()
                    "category": "экология_клининг", "subtype": "отсутствие_каски",
                    "severity": "high", "comment": "мусор на площадке", "confidence": 0.7,
                    "boxes": [],
                },
                {  # valid finding; boxes altered/dropped by validate_boxes
                    "category": "тб_от", "subtype": "отсутствие_каски", "severity": "high",
                    "comment": "Рабочий без каски.", "confidence": 0.8,
                    "boxes": [
                        {"label": "рабочий", "box_2d": [-50, 900, 500, 1200]},  # clipped
                        {"label": "мелкий объект", "box_2d": [500, 500, 505, 510]},  # too small
                    ],
                },
            ],
        }]})
        return VlmResult(parsed=parsed, raw_text=parsed.model_dump_json(), model="dirty")


def test_raw_response_keeps_untouched_model_output(tmp_path, monkeypatch):
    s, v, frames = setup(tmp_path, monkeypatch, 1)
    result = analyze_frames(s, v, frames, DirtyClient())

    # returned FrameAnalysis carries only the sanitized survivors.
    assert len(result) == 1
    survivors = result[0].findings
    assert len(survivors) == 1
    assert (survivors[0].category, survivors[0].subtype) == ("тб_от", "отсутствие_каски")
    assert [b.box_2d for b in survivors[0].boxes] == [[0, 900, 500, 1000]]

    # raw_response must still hold both findings and the original, unclipped boxes.
    row = s.execute(select(Analysis)).scalars().one()
    raw_findings = row.raw_response["frames"][0]["findings"]
    assert len(raw_findings) == 2
    pairs = {(f["category"], f["subtype"]) for f in raw_findings}
    assert ("экология_клининг", "отсутствие_каски") in pairs
    assert ("тб_от", "отсутствие_каски") in pairs
    valid = next(f for f in raw_findings if f["category"] == "тб_от")
    assert [b["box_2d"] for b in valid["boxes"]] == [[-50, 900, 500, 1200], [500, 500, 505, 510]]


class SecondsClient(FakeVlmClient):
    """Answers ts_ms in seconds instead of milliseconds — the classic unit slip."""

    def analyze_batch(self, images, ts_ms, low_quality, project_name) -> VlmResult:
        return super().analyze_batch(images, [ts // 1000 for ts in ts_ms],
                                     low_quality, project_name)


class DriftingClient(FakeVlmClient):
    """Answers plausible-but-inexact timestamps, as a model transcribing them will."""

    def analyze_batch(self, images, ts_ms, low_quality, project_name) -> VlmResult:
        return super().analyze_batch(images, [ts + 300 for ts in ts_ms],
                                     low_quality, project_name)


def test_implausible_timestamps_are_dropped_not_silently_mismatched(tmp_path, monkeypatch):
    """A seconds-for-milliseconds answer lands far outside the batch's own span, so the
    analyses are dropped rather than joined onto whatever frame happens to be nearest
    (report.py's ref() has no distance bound of its own)."""
    s, v, frames = setup(tmp_path, monkeypatch, 4, first_ts_ms=60000)
    result = analyze_frames(s, v, frames, SecondsClient())
    assert result == []
    assert s.execute(select(Analysis)).scalars().one().status == "ok"  # batch itself was fine


def test_drifting_timestamps_snap_to_the_batch_frames(tmp_path, monkeypatch):
    s, v, frames = setup(tmp_path, monkeypatch, 4)
    result = analyze_frames(s, v, frames, DriftingClient())
    assert [r.ts_ms for r in result] == [f.ts_ms for f in frames]


class HalfDeadClient(FakeVlmClient):
    """Fails every attempt on the batch starting at ts 0, succeeds on the rest."""

    def analyze_batch(self, images, ts_ms, low_quality, project_name) -> VlmResult:
        if ts_ms[0] == 0:
            raise RuntimeError("dead batch")
        return super().analyze_batch(images, ts_ms, low_quality, project_name)


def test_failed_batch_lowers_reported_coverage(tmp_path, monkeypatch):
    """A skipped batch must be visible to the reader: coverage % in meta, and
    frames_analyzed counting only frames that actually got analyzed (spec §5.3)."""
    s, v, frames = setup(tmp_path, monkeypatch, 8)  # two batches of four
    analyses = analyze_frames(s, v, frames, HalfDeadClient())
    assert len(analyses) == 4
    batches_failed = len(s.execute(
        select(Analysis).where(Analysis.status == "failed")).scalars().all())
    assert batches_failed == 1

    stage = {"primary": "каркас", "secondary": [], "confidence": 0.9, "evidence_ts": []}
    report = build_report(s, v, frames, batches_failed, 2, [], stage, [], [], "Резюме.",
                          frames_extracted=8, frames_analyzed=len(analyses))
    assert report["meta"]["coverage_pct"] == 50
    assert report["meta"]["frames_analyzed"] == 4
    assert report["meta"]["frames_analyzed"] != len(frames)
