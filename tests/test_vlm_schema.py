from app.vlm.schema import BatchResponse

SPEC_SAMPLE = {
    "frames": [{
        "ts_ms": 125000,
        "stage": "каркас",
        "stage_confidence": 0.9,
        "activity": "Монтаж опалубки на 6-м этаже",
        "equipment": [{"type": "башенный_кран", "count": 2}],
        "findings": [{
            "category": "тб_от",
            "subtype": "отсутствие_каски",
            "severity": "high",
            "comment": "Два рабочих без касок.",
            "confidence": 0.8,
            "boxes": [{"label": "рабочий без каски", "box_2d": [412, 806, 471, 843]}],
        }],
    }]
}


def test_spec_sample_parses():
    r = BatchResponse.model_validate(SPEC_SAMPLE)
    f = r.frames[0]
    assert f.stage == "каркас"
    assert f.findings[0].boxes[0].box_2d == [412, 806, 471, 843]


def test_clean_drops_mismatched_category_subtype():
    bad = BatchResponse.model_validate({
        "frames": [{**SPEC_SAMPLE["frames"][0],
                    "findings": [{**SPEC_SAMPLE["frames"][0]["findings"][0],
                                  "category": "экология_клининг"}]}]
    })
    cleaned = bad.clean()
    assert cleaned.frames[0].findings == []  # отсутствие_каски is not an ecology subtype


def test_clean_validates_boxes_so_every_consumer_gets_clipped_coordinates():
    """Box validation used to live in app.vlm.analyze, so app/cli.py — which calls
    clean() directly — rendered unclipped, degenerate and duplicate boxes."""
    dirty = BatchResponse.model_validate({
        "frames": [{**SPEC_SAMPLE["frames"][0],
                    "findings": [{**SPEC_SAMPLE["frames"][0]["findings"][0],
                                  "boxes": [
                                      {"label": "вне кадра", "box_2d": [-50, 900, 500, 1200]},
                                      {"label": "дубль", "box_2d": [-40, 901, 499, 1199]},
                                      {"label": "мелкий", "box_2d": [500, 500, 505, 510]},
                                  ]}]}]
    })
    boxes = dirty.clean().frames[0].findings[0].boxes
    assert [b.box_2d for b in boxes] == [[0, 900, 500, 1000]]
