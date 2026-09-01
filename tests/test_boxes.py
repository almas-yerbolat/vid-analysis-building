from app.vlm.boxes import iou, validate_boxes
from app.vlm.schema import Box


def b(y0, x0, y1, x1, label="x"):
    return Box(label=label, box_2d=[y0, x0, y1, x1])


def test_clips_out_of_range():
    out = validate_boxes([b(-50, 900, 500, 1200)])
    assert out[0].box_2d == [0, 900, 500, 1000]


def test_drops_degenerate_and_tiny():
    assert validate_boxes([b(500, 500, 400, 600)]) == []      # y_max <= y_min
    assert validate_boxes([b(500, 500, 505, 510)]) == []      # area 25 < 200 norm²
    assert len(validate_boxes([b(100, 100, 300, 300)])) == 1  # fine


def test_drops_iou_duplicates():
    out = validate_boxes([b(100, 100, 300, 300), b(101, 101, 301, 301), b(600, 600, 800, 800)])
    assert len(out) == 2


def test_iou_disjoint_zero():
    assert iou([0, 0, 10, 10], [500, 500, 600, 600]) == 0.0
