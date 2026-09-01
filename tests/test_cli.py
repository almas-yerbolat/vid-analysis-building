import numpy as np
import cv2

from app import storage
from app.cli import contact_sheet, run_analyze, run_sample
from app.pipeline.extract import ExtractedFrame
from app.vlm.client import VlmResult
from app.vlm.schema import BatchResponse


def test_contact_sheet_grid():
    imgs = [np.full((240, 320, 3), i * 40, np.uint8) for i in range(6)]
    sheet = contact_sheet(imgs, [f"l{i}" for i in range(6)], cols=3)
    assert sheet.shape[1] == 3 * 320 and sheet.shape[0] >= 2 * 240


def test_run_sample_writes_outputs(moving_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    stats = run_sample(moving_video, str(tmp_path / "out"))
    assert (tmp_path / "out" / "contact_sheet.jpg").exists()
    assert stats["kept"] >= 1 and stats["candidates"] >= stats["kept"]


def test_run_analyze_draws_all_findings_at_the_same_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    frame = ExtractedFrame(
        ts_ms=1000,
        media_key="frames/cli/1000.jpg",
        thumb_key="frames/cli/1000_thumb.jpg",
        width=100,
        height=100,
        motion_score=0,
        phash=0,
        low_quality=False,
        selected_reason="baseline",
    )
    source = storage.path_for(frame.media_key)
    source.parent.mkdir(parents=True)
    assert cv2.imwrite(str(source), np.zeros((100, 100, 3), np.uint8))

    class Client:
        def analyze_batch(self, *args):
            parsed = BatchResponse.model_validate({"frames": [{
                "ts_ms": 1000,
                "stage": "каркас",
                "stage_confidence": 0.9,
                "activity": "Монтаж",
                "equipment": [],
                "findings": [
                    {
                        "category": "тб_от",
                        "subtype": "отсутствие_каски",
                        "severity": "high",
                        "comment": "Первая находка.",
                        "confidence": 0.9,
                        "boxes": [{"label": "первая", "box_2d": [100, 100, 300, 300]}],
                    },
                    {
                        "category": "экология_клининг",
                        "subtype": "свалка_мусора",
                        "severity": "medium",
                        "comment": "Вторая находка.",
                        "confidence": 0.8,
                        "boxes": [{"label": "вторая", "box_2d": [600, 600, 800, 800]}],
                    },
                ],
            }]})
            return VlmResult(parsed=parsed, raw_text="", model="test")

    monkeypatch.setattr("app.cli._sample", lambda _: ([], [frame], 2.0))
    monkeypatch.setattr("app.cli.get_client", Client)

    run_analyze("video.mp4", str(tmp_path / "out"), draw=True)

    annotated = cv2.imread(str(tmp_path / "out" / "annotated" / "1000.jpg"))
    assert annotated[10, 10].sum() > 0  # first finding's box
    assert annotated[60, 60].sum() > 0  # second finding's box
