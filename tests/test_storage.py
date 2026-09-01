from app import storage


def test_save_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path))
    key = storage.save_bytes("frames/vid_1/5000.jpg", b"jpegdata")
    assert key == "frames/vid_1/5000.jpg"
    assert storage.read_bytes(key) == b"jpegdata"
    assert storage.path_for(key).exists()


def test_save_upload_streams(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path))
    storage.save_upload("videos/vid_1/a.mp4", io.BytesIO(b"x" * 1000))
    assert storage.read_bytes("videos/vid_1/a.mp4") == b"x" * 1000
