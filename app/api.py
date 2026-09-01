import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import select

from app import storage
from app.db import SessionLocal
from app.models import Frame, Report, Video
from app.pipeline.run import run_pipeline
from app.vlm.client import get_client as get_vlm_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

RUNNING = {"probing", "sampling", "analyzing", "aggregating"}


def _get_video(session, video_id: str) -> Video:
    video = session.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    return video


def _safe_name(name: str | None) -> str:
    """Upload filenames are untrusted and land in storage keys: basename only."""
    return Path(name or "").name or "upload.bin"


@router.post("/videos/upload")
async def upload_video(file: UploadFile = File(...), project_name: str = Form("")):
    with SessionLocal() as session:
        video = Video(filename=_safe_name(file.filename), project_name=project_name, media_key="")
        session.add(video)
        session.flush()
        video.media_key = storage.save_upload(f"videos/{video.id}/{video.filename}", file.file)
        session.commit()
        return {"video_id": video.id}


@router.post("/photos/upload")
async def upload_photo(file: UploadFile = File(...), project_name: str = Form("")):
    with SessionLocal() as session:
        video = Video(
            filename=_safe_name(file.filename), project_name=project_name, media_key="", is_photo=True
        )
        session.add(video)
        session.flush()
        video.media_key = storage.save_upload(f"videos/{video.id}/{video.filename}", file.file)
        session.commit()
        return {"video_id": video.id}


@router.post("/videos/{video_id}/analyze", status_code=202)
async def analyze(video_id: str, background: BackgroundTasks):
    with SessionLocal() as session:
        video = _get_video(session, video_id)
        if video.status in RUNNING:
            raise HTTPException(409, "already processing")
    background.add_task(run_pipeline, video_id, SessionLocal, get_vlm_client())
    return {"video_id": video_id, "status": "queued"}


@router.get("/videos/{video_id}/status")
async def status_stream(video_id: str):
    async def gen():
        while True:
            with SessionLocal() as session:
                video = session.get(Video, video_id)
                if video is None:
                    yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                    return
                payload = {
                    "status": video.status,
                    "progress_pct": video.progress_pct,
                    "progress_note": video.progress_note,
                    "error": video.error,
                }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload["status"] in ("done", "failed"):
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/videos/{video_id}/report")
async def get_report(video_id: str):
    with SessionLocal() as session:
        video = _get_video(session, video_id)
        report = session.execute(
            select(Report).where(Report.video_id == video_id).order_by(Report.created_at.desc())
        ).scalars().first()
        if report is None:
            return JSONResponse({"status": video.status, "error": video.error}, status_code=202)
        return report.report_json


@router.get("/videos")
async def list_videos():
    with SessionLocal() as session:
        videos = session.execute(select(Video).order_by(Video.created_at.desc())).scalars().all()
        return [
            {
                "id": video.id,
                "filename": video.filename,
                "project_name": video.project_name,
                "status": video.status,
                "is_photo": video.is_photo,
                "created_at": video.created_at.isoformat(),
            }
            for video in videos
        ]


@router.get("/frames/{frame_id}")
async def get_frame(frame_id: str, thumb: bool = False):
    with SessionLocal() as session:
        frame = session.get(Frame, frame_id)
        if frame is None:
            raise HTTPException(404, "frame not found")
        key = frame.thumb_key if thumb else frame.media_key
    return Response(storage.read_bytes(key), media_type="image/jpeg")
