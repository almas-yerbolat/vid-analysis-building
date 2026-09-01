from app.aggregate import (activity_timeline, decide_stage, equipment_inventory,
                           merge_findings)
from app.vlm.schema import BatchResponse

def fa(ts_ms, stage="каркас", conf=0.9, activity="Монтаж", equipment=None, findings=None):
    return BatchResponse.model_validate({"frames": [{
        "ts_ms": ts_ms, "stage": stage, "stage_confidence": conf, "activity": activity,
        "equipment": equipment or [], "findings": findings or [],
    }]}).frames[0]


def helmet(conf=0.8, comment="без каски"):
    return {"category": "тб_от", "subtype": "отсутствие_каски", "severity": "high",
            "comment": comment, "confidence": conf,
            "boxes": [{"label": "рабочий", "box_2d": [100, 100, 300, 300]}]}


def test_merge_within_60s_and_split_beyond():
    analyses = [fa(0, findings=[helmet()]), fa(30000, findings=[helmet(0.9)]),
                fa(120000, findings=[helmet(0.7)])]
    merged = merge_findings(analyses)
    assert len(merged) == 2  # 0+30s merged; 120s separate (gap 90s)
    first = next(m for m in merged if m.evidence[0][0] in (0, 30000))
    assert first.confidence == 0.9 and len(first.evidence) == 2


def test_evidence_capped_at_3_best():
    analyses = [fa(i * 10000, findings=[helmet(0.5 + i * 0.1)]) for i in range(5)]
    merged = merge_findings(analyses)
    assert len(merged) == 1
    assert len(merged[0].evidence) == 3
    assert merged[0].evidence[0][0] == 40000  # highest confidence first


def test_stage_weighted_vote():
    analyses = [fa(0, "каркас", 0.9), fa(5000, "каркас", 0.8), fa(10000, "фундамент", 0.4)]
    stage = decide_stage(analyses)
    assert stage["primary"] == "каркас"
    assert stage["secondary"] == []  # 0.4 / 1.7 < 25%


def test_equipment_max_simultaneous():
    analyses = [fa(0, equipment=[{"type": "самосвал", "count": 1}]),
                fa(5000, equipment=[{"type": "самосвал", "count": 3}])]
    inv = equipment_inventory(analyses)
    assert inv == [{"type": "самосвал", "max_count": 3, "evidence_ts": 5000}]


def test_timeline_merges_identical_activity():
    analyses = [fa(0, activity="Монтаж"), fa(5000, activity="Монтаж"),
                fa(10000, activity="Разгрузка")]
    tl = activity_timeline(analyses)
    assert tl == [{"from_ms": 0, "to_ms": 10000, "activity": "Монтаж"},
                  {"from_ms": 10000, "to_ms": 10000, "activity": "Разгрузка"}]
