import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, BigInteger, Boolean, DateTime, Float, ForeignKey,
                        Integer, String, Text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _make_id(prefix: str):
    def gen() -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"
    return gen


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_make_id("vid"))
    filename: Mapped[str] = mapped_column(String)
    project_name: Mapped[str] = mapped_column(String, default="")
    media_key: Mapped[str] = mapped_column(String)
    is_photo: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    # status ∈ {uploaded, probing, sampling, analyzing, aggregating, done, failed}
    status: Mapped[str] = mapped_column(String, default="uploaded")
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    progress_note: Mapped[str] = mapped_column(String, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Frame(Base):
    __tablename__ = "frames"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_make_id("frm"))
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    ts_ms: Mapped[int] = mapped_column(Integer)
    media_key: Mapped[str] = mapped_column(String)
    thumb_key: Mapped[str] = mapped_column(String)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    motion_score: Mapped[float] = mapped_column(Float, default=0.0)
    phash: Mapped[int] = mapped_column(BigInteger, default=0)  # 63-bit, see quality.phash
    low_quality: Mapped[bool] = mapped_column(Boolean, default=False)
    # selected_reason ∈ {baseline, fast_motion, scene_cut, photo}
    selected_reason: Mapped[str] = mapped_column(String, default="baseline")


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_make_id("ana"))
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    batch_index: Mapped[int] = mapped_column(Integer)
    frame_ids: Mapped[list] = mapped_column(JSON, default=list)
    raw_response: Mapped[dict] = mapped_column(JSON, default=dict)
    model: Mapped[str] = mapped_column(String, default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="ok")  # ok | failed


class Finding(Base):
    __tablename__ = "findings"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_make_id("fnd"))
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    category: Mapped[str] = mapped_column(String)
    subtype: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    comment: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String, default="unreviewed")


class FindingFrame(Base):
    __tablename__ = "finding_frames"
    finding_id: Mapped[str] = mapped_column(ForeignKey("findings.id"), primary_key=True)
    frame_id: Mapped[str] = mapped_column(ForeignKey("frames.id"), primary_key=True)
    frame_comment: Mapped[str] = mapped_column(Text, default="")
    boxes: Mapped[list] = mapped_column(JSON, default=list)  # [{label, box_2d[4]}] 0–1000 norm


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_make_id("rpt"))
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    report_json: Mapped[dict] = mapped_column(JSON)
    summary_ru: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
