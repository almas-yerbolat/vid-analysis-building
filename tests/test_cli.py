import numpy as np

from app.cli import contact_sheet, run_sample


def test_contact_sheet_grid():
    imgs = [np.full((240, 320, 3), i * 40, np.uint8) for i in range(6)]
    sheet = contact_sheet(imgs, [f"l{i}" for i in range(6)], cols=3)
    assert sheet.shape[1] == 3 * 320 and sheet.shape[0] >= 2 * 240


def test_run_sample_writes_outputs(moving_video, tmp_path, monkeypatch):
    from app import storage

    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    stats = run_sample(moving_video, str(tmp_path / "out"))
    assert (tmp_path / "out" / "contact_sheet.jpg").exists()
    assert stats["kept"] >= 1 and stats["candidates"] >= stats["kept"]
