from datetime import datetime, timezone

from app.aggregate import MergedFinding
from app.models import Finding, FindingFrame, Frame, Report, Video


def _frame_by_ts(frames: list[Frame]) -> dict[int, Frame]:
    return {f.ts_ms: f for f in frames}


def build_report(session, video: Video, frames: list[Frame], batches_failed: int,
                 merged: list[MergedFinding], stage: dict, equipment: list[dict],
                 timeline: list[dict], summary_ru: str, frames_extracted: int) -> dict:
    by_ts = _frame_by_ts(frames)

    def ref(ts_ms: int) -> Frame | None:
        if ts_ms in by_ts:
            return by_ts[ts_ms]
        return min(by_ts.values(), key=lambda f: abs(f.ts_ms - ts_ms), default=None)

    findings_json = []
    stats = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for m in merged:
        stats[m.severity] += 1
        row = Finding(video_id=video.id, category=m.category, subtype=m.subtype,
                      severity=m.severity, title=m.title, comment=m.comment,
                      confidence=m.confidence)
        session.add(row)
        session.flush()
        evidence_json = []
        seen_frame_ids = set()
        for ts_ms, frame_comment, boxes in m.evidence:
            frame = ref(ts_ms)
            # Nearest-frame fallback can map two distinct timestamps onto the same
            # Frame row; finding_frames' PK is (finding_id, frame_id), so keep only
            # the first (higher-confidence, since evidence is best-first) hit per frame.
            if frame is None or frame.id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame.id)
            session.add(FindingFrame(finding_id=row.id, frame_id=frame.id,
                                     frame_comment=frame_comment, boxes=boxes))
            evidence_json.append({
                "frame_id": frame.id, "ts_ms": frame.ts_ms,
                "thumb_url": f"/api/frames/{frame.id}?thumb=1",
                "full_url": f"/api/frames/{frame.id}",
                "comment": frame_comment, "boxes": boxes,
            })
        findings_json.append({
            "id": row.id, "category": m.category, "subtype": m.subtype,
            "severity": m.severity, "title": m.title, "comment": m.comment,
            "confidence": m.confidence, "status": "unreviewed",
            "evidence": evidence_json,
        })

    stage_frames = [ref(ts) for ts in stage.get("evidence_ts", [])]
    report = {
        "video_id": video.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {"duration_s": video.duration_s, "frames_analyzed": len(frames),
                 "frames_extracted": frames_extracted, "batches_failed": batches_failed},
        "stage": {"primary": stage["primary"], "secondary": stage["secondary"],
                  "confidence": stage["confidence"],
                  "evidence_frames": [f.id for f in stage_frames if f]},
        "summary_ru": summary_ru,
        "equipment": [{"type": e["type"], "max_count": e["max_count"],
                       "evidence_frame": (ref(e["evidence_ts"]).id if ref(e["evidence_ts"]) else None)}
                      for e in equipment],
        "activity_timeline": timeline,
        "findings": findings_json,
        "stats": stats,
    }
    session.add(Report(video_id=video.id, report_json=report, summary_ru=summary_ru))
    session.flush()
    return report
