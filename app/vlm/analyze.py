import logging
from concurrent.futures import ThreadPoolExecutor

from app import storage
from app.config import settings
from app.models import Analysis, Frame, Video
from app.vlm.client import VlmClient, VlmResult
from app.vlm.schema import FrameAnalysis

logger = logging.getLogger(__name__)

BATCH_SIZE = 4
# A single-frame batch has zero span; without a floor every returned timestamp would
# be dropped unless it matched to the millisecond.
SNAP_TOLERANCE_FLOOR_MS = 1000


def _snap_timestamps(frames: list[FrameAnalysis], batch_ts: list[int]) -> list[FrameAnalysis]:
    """Snap each returned ts_ms onto one of the batch's own timestamps.

    The model echoes ts_ms back as free-form data, and report.py joins evidence to
    Frame rows by nearest timestamp with no distance bound. A model answering in
    seconds, or repeating one timestamp for every image, would therefore collapse
    findings onto the wrong frame and draw boxes over an image that never contained
    the object. The batch's true timestamps are known right here, so snap to them and
    drop anything further away than the batch's own span.
    """
    tolerance = max(max(batch_ts) - min(batch_ts), SNAP_TOLERANCE_FLOOR_MS)
    kept: list[FrameAnalysis] = []
    for frame in frames:
        nearest = min(batch_ts, key=lambda ts: abs(ts - frame.ts_ms))
        gap = abs(nearest - frame.ts_ms)
        if gap > tolerance:
            logger.warning(
                "dropping frame analysis: model ts_ms=%s is %s ms from nearest batch "
                "timestamp %s (tolerance %s ms)", frame.ts_ms, gap, nearest, tolerance)
            continue
        if gap:
            logger.warning("snapping model ts_ms=%s to batch timestamp %s (gap %s ms)",
                           frame.ts_ms, nearest, gap)
            frame.ts_ms = nearest
        kept.append(frame)
    return kept


def _run_batch(video: Video, batch: list[Frame], client: VlmClient) -> VlmResult:
    images = [storage.read_bytes(f.media_key) for f in batch]
    ts = [f.ts_ms for f in batch]
    lq = [f.low_quality for f in batch]
    try:
        # ponytail: bare except retries programming errors (e.g. AttributeError) as if
        # they were flaky API calls; narrow to the SDK's transient-error types if that bites.
        return client.analyze_batch(images, ts, lq, video.project_name)
    except Exception:
        logger.warning("batch failed, retrying once", exc_info=True)
        return client.analyze_batch(images, ts, lq, video.project_name)


def analyze_frames(session, video, frames, client, on_progress=None) -> list[FrameAnalysis]:
    batches = [frames[i:i + BATCH_SIZE] for i in range(0, len(frames), BATCH_SIZE)]
    results: list[FrameAnalysis] = []
    done = 0

    def work(indexed):
        idx, batch = indexed
        try:
            return idx, batch, _run_batch(video, batch, client), None
        except Exception as exc:
            return idx, batch, None, exc

    with ThreadPoolExecutor(max_workers=settings.vlm_concurrency) as pool:
        for idx, batch, result, exc in pool.map(work, enumerate(batches)):
            if exc is not None:
                logger.error("batch %s failed twice: %s", idx, exc)
                session.add(Analysis(video_id=video.id, batch_index=idx,
                                     frame_ids=[f.id for f in batch],
                                     raw_response={"error": str(exc)}, status="failed"))
                done += 1
            else:
                raw_snapshot = result.parsed.model_dump()  # before clean() mutates in place
                parsed = result.parsed.clean()
                results.extend(_snap_timestamps(parsed.frames, [f.ts_ms for f in batch]))
                session.add(Analysis(
                    video_id=video.id, batch_index=idx,
                    frame_ids=[f.id for f in batch],
                    raw_response=raw_snapshot, model=result.model,
                    tokens_in=result.tokens_in, tokens_out=result.tokens_out, status="ok",
                ))
                done += 1
            session.flush()
            if on_progress:
                on_progress(done, len(batches))

    return sorted(results, key=lambda r: r.ts_ms)
