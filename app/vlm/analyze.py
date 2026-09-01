import logging
from concurrent.futures import ThreadPoolExecutor

from app import storage
from app.config import settings
from app.models import Analysis, Frame, Video
from app.vlm.boxes import validate_boxes
from app.vlm.client import VlmClient, VlmResult
from app.vlm.schema import FrameAnalysis

logger = logging.getLogger(__name__)

BATCH_SIZE = 4


def _run_batch(video: Video, batch: list[Frame], client: VlmClient) -> VlmResult:
    images = [storage.read_bytes(f.media_key) for f in batch]
    ts = [f.ts_ms for f in batch]
    lq = [f.low_quality for f in batch]
    try:
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
                parsed = result.parsed.clean()
                for frame_analysis in parsed.frames:
                    for finding in frame_analysis.findings:
                        finding.boxes = validate_boxes(finding.boxes)
                    results.append(frame_analysis)
                session.add(Analysis(
                    video_id=video.id, batch_index=idx,
                    frame_ids=[f.id for f in batch],
                    raw_response=parsed.model_dump(), model=result.model,
                    tokens_in=result.tokens_in, tokens_out=result.tokens_out, status="ok",
                ))
                done += 1
            session.flush()
            if on_progress:
                on_progress(done, len(batches))

    return sorted(results, key=lambda r: r.ts_ms)
