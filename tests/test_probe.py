import pytest

from app.pipeline.probe import probe


def test_probe_reads_metadata(static_video):
    info = probe(static_video)
    assert info.width == 320 and info.height == 240
    assert 9 <= info.fps <= 11
    assert 11.0 <= info.duration_s <= 13.0


def test_probe_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        probe(str(bad))
