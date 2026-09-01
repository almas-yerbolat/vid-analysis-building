import numpy as np

from app.vlm.client import FakeVlmClient
from app.vlm.render import draw_boxes


def test_fake_client_shapes():
    c = FakeVlmClient()
    r = c.analyze_batch([b"img1", b"img2"], [0, 5000], [False, False], "Объект")
    assert [f.ts_ms for f in r.parsed.frames] == [0, 5000]
    assert r.parsed.frames[0].findings and not r.parsed.frames[1].findings
    f = r.parsed.frames[0].findings[0]
    assert f.category == "тб_от" and f.boxes[0].box_2d == [400, 400, 600, 600]
    assert c.summarize("prompt").startswith("На объекте")


def test_draw_boxes_changes_pixels():
    img = np.zeros((500, 800, 3), np.uint8)
    out = draw_boxes(img, [{"label": "тест", "box_2d": [100, 100, 400, 400]}], "high")
    assert out.shape == img.shape and out.sum() > 0
