from collections import defaultdict
from dataclasses import dataclass

from app.vlm.client import VlmClient
from app.vlm.prompt import SUMMARY_PROMPT_TEMPLATE
from app.vlm.schema import SEVERITY_RANK, FrameAnalysis

MERGE_WINDOW_MS = 60_000
MAX_EVIDENCE = 3


@dataclass
class MergedFinding:
    category: str
    subtype: str
    severity: str
    title: str
    comment: str
    confidence: float
    evidence: list  # [(ts_ms, frame_comment, boxes_as_dicts)] best-first, ≤3


def merge_findings(analyses: list[FrameAnalysis]) -> list[MergedFinding]:
    by_key = defaultdict(list)  # (category, subtype) -> [(ts_ms, finding)]
    for a in analyses:
        for f in a.findings:
            by_key[(f.category, f.subtype)].append((a.ts_ms, f))

    merged: list[MergedFinding] = []
    for (category, subtype), items in by_key.items():
        items.sort(key=lambda x: x[0])
        group = [items[0]]
        for item in items[1:]:
            if item[0] - group[-1][0] <= MERGE_WINDOW_MS:
                group.append(item)
            else:
                merged.append(_finalize(category, subtype, group))
                group = [item]
        merged.append(_finalize(category, subtype, group))
    merged.sort(key=lambda m: (-SEVERITY_RANK[m.severity], -m.confidence))
    return merged


def _finalize(category, subtype, group) -> MergedFinding:
    best = max(group, key=lambda x: x[1].confidence)
    ranked = sorted(group, key=lambda x: -x[1].confidence)[:MAX_EVIDENCE]
    def mmss(ts):
        m, s = divmod(ts // 1000, 60)
        return f"{m:02d}:{s:02d}"
    span = (f"Зафиксировано на кадре {mmss(group[0][0])}" if len(group) == 1
            else f"Зафиксировано на кадрах {mmss(group[0][0])}–{mmss(group[-1][0])}")
    return MergedFinding(
        category=category, subtype=subtype,
        severity=max((g[1].severity for g in group), key=lambda s: SEVERITY_RANK[s]),
        title=best[1].comment.split(".")[0][:120],
        comment=f"{span}. {best[1].comment}",
        confidence=best[1].confidence,
        evidence=[(ts, f.comment, [b.model_dump() for b in f.boxes]) for ts, f in ranked],
    )


def decide_stage(analyses: list[FrameAnalysis]) -> dict:
    weights = defaultdict(float)
    evidence = defaultdict(list)
    for a in analyses:
        weights[a.stage] += a.stage_confidence
        evidence[a.stage].append(a.ts_ms)
    if not weights:
        return {"primary": None, "secondary": [], "confidence": 0.0, "evidence_ts": []}
    primary = max(weights, key=weights.get)
    total = sum(weights.values())
    secondary = [s for s, w in weights.items()
                 if s != primary and w >= 0.25 * weights[primary]]
    return {
        "primary": primary,
        "secondary": sorted(secondary, key=lambda s: -weights[s]),
        "confidence": round(weights[primary] / total, 2),
        "evidence_ts": evidence[primary][:2],
    }


def equipment_inventory(analyses: list[FrameAnalysis]) -> list[dict]:
    best: dict[str, tuple[int, int]] = {}  # type -> (max_count, ts)
    for a in analyses:
        for eq in a.equipment:
            if eq.count > best.get(eq.type, (0, 0))[0]:
                best[eq.type] = (eq.count, a.ts_ms)
    return [{"type": t, "max_count": c, "evidence_ts": ts}
            for t, (c, ts) in sorted(best.items(), key=lambda kv: -kv[1][0])]


def activity_timeline(analyses: list[FrameAnalysis]) -> list[dict]:
    """Contiguous segments: each ends where the next begins (spec §4.5 shape)."""
    ordered = sorted(analyses, key=lambda a: a.ts_ms)
    segments: list[dict] = []
    for a in ordered:
        if segments and segments[-1]["activity"] == a.activity:
            segments[-1]["to_ms"] = a.ts_ms
            continue
        if segments:
            segments[-1]["to_ms"] = a.ts_ms
        segments.append({"from_ms": a.ts_ms, "to_ms": a.ts_ms, "activity": a.activity})
    return segments


def build_summary(client: VlmClient, stage: dict, equipment: list[dict],
                  timeline: list[dict], findings: list[MergedFinding]) -> str:
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        stage=stage["primary"],
        equipment="; ".join(f"{e['type']}: {e['max_count']}" for e in equipment) or "не выявлена",
        timeline="; ".join(f"{t['from_ms'] // 1000}–{t['to_ms'] // 1000}с: {t['activity']}"
                           for t in timeline) or "нет данных",
        n_findings=len(findings),
        findings="; ".join(f"[{f.severity}] {f.title}" for f in findings) or "не выявлены",
    )
    return client.summarize(prompt)
