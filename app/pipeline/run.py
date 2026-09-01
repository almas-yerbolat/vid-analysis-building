import logging

from sqlalchemy import select

from app import storage
from app.aggregate import (activity_timeline, build_summary, decide_stage,
                           equipment_inventory, merge_findings)
from app.db import SessionLocal
from app.models import Analysis, Frame, Video
from app.pipeline.extract import extract_frames, extract_photo
from app.pipeline.motion import motion_curve
from app.pipeline.probe import probe
from app.pipeline.schedule import scene_cuts, schedule_keyframes
from app.report import build_report
from app.vlm.analyze import analyze_frames
from app.vlm.client import VlmClient, get_client

logger = logging.getLogger(__name__)


def _set_status(session, video: Video, status: str, pct: int, note: str = "") -> None:
    video.status, video.progress_pct, video.progress_note = status, pct, note
    session.commit()


def run_pipeline(video_id: str, session_factory=SessionLocal,
                 client: VlmClient | None = None) -> None:
    with session_factory() as session:
        video = session.get(Video, video_id)
        if video is None:
            logger.error("video %s not found", video_id)
            return
        try:
            client = client or get_client()
            path = str(storage.path_for(video.media_key))
            frames_extracted = 0

            if video.is_photo:
                _set_status(session, video, "sampling", 10, "Обработка фото")
                extracted = [extract_photo(storage.read_bytes(video.media_key), video.id)]
                frames_extracted = 1
            else:
                _set_status(session, video, "probing", 5, "Чтение видео")
                info = probe(path)
                video.duration_s, video.fps = info.duration_s, info.fps
                video.width, video.height = info.width, info.height

                _set_status(session, video, "sampling", 10, "Анализ движения")
                motion = motion_curve(path)
                cuts = scene_cuts(motion)
                keyframes = schedule_keyframes(info.duration_s, motion, cuts)
                frames_extracted = len(keyframes)
                _set_status(session, video, "sampling", 25,
                            f"Извлечение кадров: {len(keyframes)}")
                extracted = extract_frames(path, video.id, keyframes, motion)

            frame_rows = [Frame(video_id=video.id, ts_ms=e.ts_ms, media_key=e.media_key,
                                thumb_key=e.thumb_key, width=e.width, height=e.height,
                                motion_score=e.motion_score, phash=e.phash,
                                low_quality=e.low_quality, selected_reason=e.selected_reason)
                          for e in extracted]
            session.add_all(frame_rows)
            session.flush()

            _set_status(session, video, "analyzing", 40, f"Анализ 0/{-(-len(frame_rows) // 4)}")

            def on_progress(done, total):
                _set_status(session, video, "analyzing",
                            40 + round(50 * done / max(total, 1)), f"Анализ {done}/{total}")

            analyses = analyze_frames(session, video, frame_rows, client, on_progress)
            batches_failed = len(session.execute(
                select(Analysis).where(Analysis.video_id == video.id,
                                       Analysis.status == "failed")).scalars().all())

            _set_status(session, video, "aggregating", 92, "Формирование отчёта")
            merged = merge_findings(analyses)
            stage = decide_stage(analyses)
            equipment = equipment_inventory(analyses)
            timeline = activity_timeline(analyses)
            summary = build_summary(client, stage, equipment, timeline, merged)
            build_report(session, video, frame_rows, batches_failed, merged, stage,
                         equipment, timeline, summary, frames_extracted)

            _set_status(session, video, "done", 100, "Готово")
        except Exception as exc:
            logger.exception("pipeline failed for %s", video_id)
            session.rollback()
            video = session.get(Video, video_id)
            video.status, video.error = "failed", str(exc)[:2000]
            session.commit()
