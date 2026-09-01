from app.vlm.schema import Box

MIN_AREA = 200  # 0.02% of 1000x1000 normalized frame
MAX_DUP_IOU = 0.9


def iou(a: list[int], b: list[int]) -> float:
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def validate_boxes(boxes: list[Box]) -> list[Box]:
    kept: list[Box] = []
    for box in boxes:
        y0, x0, y1, x1 = (min(max(v, 0), 1000) for v in box.box_2d)
        if y1 <= y0 or x1 <= x0 or (y1 - y0) * (x1 - x0) < MIN_AREA:
            continue
        clipped = [y0, x0, y1, x1]
        if any(iou(clipped, k.box_2d) > MAX_DUP_IOU for k in kept):
            continue
        kept.append(Box(label=box.label, box_2d=clipped))
    return kept
