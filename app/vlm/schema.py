from typing import Literal

from pydantic import BaseModel, Field

STAGES = ["котлован", "фундамент", "каркас", "кровля", "фасад", "благоустройство"]

SUBTYPES_BY_CATEGORY: dict[str, set[str]] = {
    "нарушения_площадки": {
        "стихийное_складирование", "нет_зус", "нет_ограждения_площадки",
        "нет_маршрутов_техники", "загромождение_проезда",
    },
    "тб_от": {
        "отсутствие_каски", "отсутствие_жилета", "нет_ограждения_перекрытия",
        "открытая_шахта_лифта", "нарушение_установки_лесов",
    },
    "экология_клининг": {"свалка_мусора", "грязная_техника_выезд", "нет_мойки_колес"},
}

SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}

Category = Literal["нарушения_площадки", "тб_от", "экология_клининг"]
Severity = Literal["critical", "high", "medium", "low"]
Stage = Literal["котлован", "фундамент", "каркас", "кровля", "фасад", "благоустройство"]
Subtype = Literal[
    "стихийное_складирование", "нет_зус", "нет_ограждения_площадки",
    "нет_маршрутов_техники", "загромождение_проезда",
    "отсутствие_каски", "отсутствие_жилета", "нет_ограждения_перекрытия",
    "открытая_шахта_лифта", "нарушение_установки_лесов",
    "свалка_мусора", "грязная_техника_выезд", "нет_мойки_колес",
]


class Box(BaseModel):
    label: str
    box_2d: list[int] = Field(min_length=4, max_length=4)  # [y_min, x_min, y_max, x_max] 0–1000


class VlmFinding(BaseModel):
    category: Category
    subtype: Subtype
    severity: Severity
    comment: str
    confidence: float = Field(ge=0.0, le=1.0)
    boxes: list[Box] = []


class EquipmentCount(BaseModel):
    type: str
    count: int = Field(ge=0)


class FrameAnalysis(BaseModel):
    ts_ms: int
    stage: Stage
    stage_confidence: float = Field(ge=0.0, le=1.0)
    activity: str
    equipment: list[EquipmentCount] = []
    findings: list[VlmFinding] = []


class BatchResponse(BaseModel):
    frames: list[FrameAnalysis]

    def clean(self) -> "BatchResponse":
        """Drop findings whose subtype doesn't belong to their category."""
        for frame in self.frames:
            frame.findings = [
                f for f in frame.findings if f.subtype in SUBTYPES_BY_CATEGORY[f.category]
            ]
        return self
