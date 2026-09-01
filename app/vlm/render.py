import numpy as np
import cv2

SEVERITY_BGR = {
    "critical": (0, 0, 220),   # red
    "high": (0, 100, 255),     # orange
    "medium": (0, 200, 255),   # yellow
    "low": (180, 180, 180),    # gray
}


def draw_boxes(img: np.ndarray, boxes: list[dict], severity: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    color = SEVERITY_BGR.get(severity, (255, 255, 255))
    for box in boxes:
        y0, x0, y1, x1 = box["box_2d"]
        p0 = (round(x0 / 1000 * w), round(y0 / 1000 * h))
        p1 = (round(x1 / 1000 * w), round(y1 / 1000 * h))
        cv2.rectangle(out, p0, p1, color, max(2, w // 800))
        label = box.get("label", "")
        if label:
            cv2.putText(out, label, (p0[0], max(p0[1] - 6, 12)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
