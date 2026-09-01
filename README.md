# Construction Site Video Analysis

Ingests construction-site drone footage, samples frames adaptively, analyzes them with
Gemini on Vertex AI, and produces a structured inspection report in Russian — every
finding localized with bounding boxes and linked to the exact frame it was seen in.

Proof of concept. Backend and pipeline are complete; the web frontend is in progress.

## What it does

A video goes in. What comes out is a report with a construction-stage assessment, an
equipment inventory, an activity timeline, and a list of findings grouped by category and
severity — each one clickable through to its evidence frame with boxes drawn over the
evidence.

Findings use a **closed vocabulary**, so they are groupable and comparable across videos
instead of free text. Three categories, thirteen subtypes:

| Category | Subtypes |
|---|---|
| `нарушения_площадки` | `стихийное_складирование`, `нет_зус`, `нет_ограждения_площадки`, `нет_маршрутов_техники`, `загромождение_проезда` |
| `тб_от` | `отсутствие_каски`, `отсутствие_жилета`, `нет_ограждения_перекрытия`, `открытая_шахта_лифта`, `нарушение_установки_лесов` |
| `экология_клининг` | `свалка_мусора`, `грязная_техника_выезд`, `нет_мойки_колес` |

Severity is `critical` (immediate life-safety), `high` (safety violation, blocked fire
lane), `medium` (unauthorized storage, dirty vehicles), or `low` (cleanliness). Stages are
`котлован`, `фундамент`, `каркас`, `кровля`, `фасад`, `благоустройство`.

## How the pipeline works

```
upload → probe → motion scan → keyframe schedule → extract + filter + dedup
       → VLM analysis (batches of 4, concurrent) → aggregate → report
```

The interesting parts are the ones that keep cost down and evidence honest:

**Adaptive sampling.** One candidate frame every 5 s baseline, densified to every 1.5 s
across intervals where the drone is panning fast, so no scenery is skipped between
keyframes. Scene cuts are detected as spikes in the motion curve the pipeline already
computes — the video is decoded once, not twice.

**Quality filter and deduplication.** A motion-blurred or badly exposed frame is replaced
by the sharpest passing neighbour within ±0.7 s; if none passes, the best available is kept
and flagged `low_quality` so the model is told to lower its confidence rather than guess.
Near-duplicate frames are dropped by perceptual hash, which on a hovering drone removes a
large fraction of candidates — a direct saving on vision-model cost.

**Batching for temporal context.** Frames go to the model four at a time, in sequence, each
labelled with its timestamp. That lets the model reason across frames ("the same truck",
"the crane is actually moving") while keeping requests parallelizable.

**Aggregation.** The same violation appears in many consecutive frames, so findings sharing
a category and subtype within 60 s are merged into one, keeping the best 1–3 frames as
evidence. Construction stage is a confidence-weighted vote; a genuinely mixed site reports
secondary stages rather than silently picking one. Equipment is reported as the **maximum
simultaneously visible count** in any single frame — an honest lower bound, not a sum.

**Bounding boxes.** The model returns coordinates directly, normalized 0–1000 against the
stored frame. They are clipped, de-duplicated by overlap, and degenerate or vanishingly
small boxes are dropped. A finding whose boxes all fail validation falls back to
whole-frame evidence, because a box drawn over the wrong object is worse than no box.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
uv sync
docker compose up -d db
cp .env.example .env
```

No system `ffmpeg` is needed — all video work goes through `opencv-python-headless`.

### Running against the real model

The default provider is a deterministic fake, so everything runs and all tests pass with no
credentials and no network. To use Gemini, put a Vertex AI service-account key at
`creds.json` (git-ignored) and set:

```
VLM_PROVIDER=gemini
```

The GCP project is read from the key itself. Verified working with `gemini-3.6-flash` at
`location=global`.

## Usage

### API

```bash
uv run uvicorn app.main:app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /api/videos/upload` | Multipart upload, returns `video_id` |
| `POST /api/photos/upload` | A photo enters the pipeline as a single frame |
| `POST /api/videos/{id}/analyze` | Starts analysis (202) |
| `GET /api/videos/{id}/status` | SSE stream: stage, percent, note |
| `GET /api/videos/{id}/report` | Report JSON, or 202 while still running |
| `GET /api/videos` | Listing / history |
| `GET /api/frames/{id}` | Frame image (`?thumb=1` for the thumbnail) |

```bash
curl -F "file=@clip.mp4" localhost:8000/api/videos/upload
curl -X POST localhost:8000/api/videos/<id>/analyze
curl localhost:8000/api/videos/<id>/report
```

Pipeline status moves `uploaded → queued → probing → sampling → analyzing → aggregating →
done`, or `failed`.

### CLI (threshold tuning)

The CLI is the tool for calibrating sampling on real footage without touching the database:

```bash
# contact sheet of kept frames, labelled mm:ss + why it was selected
uv run python -m app.cli sample clip.mp4 --out ./spike_out

# full analysis, with boxes burned into the evidence frames
VLM_PROVIDER=gemini uv run python -m app.cli analyze clip.mp4 --draw-boxes
```

`sample` prints candidates, kept, dropped and a per-reason breakdown — eyeball the contact
sheet, adjust a threshold, run again.

### Frontend

A Next.js dashboard lives in `web/` and is a work in progress.

```bash
cd web && npm install && npm run dev
```

## Configuration

Every threshold is an environment-overridable setting in `app/config.py` — this is the
calibration surface, deliberately not hardcoded.

| Setting | Default | What it controls |
|---|---|---|
| `BASELINE_INTERVAL_S` | 5.0 | Baseline seconds between candidate frames |
| `DENSE_INTERVAL_S` | 1.5 | Interval inside fast-motion stretches |
| `T_FAST_PERCENTILE` | 75.0 | Motion percentile that counts as "fast" |
| `CUT_RATIO` / `CUT_FLOOR` | 3.0 / 8.0 | Scene cut must clear both a relative and an absolute bar |
| `BLUR_THRESHOLD` | 100.0 | Variance-of-Laplacian floor |
| `LUMA_MIN` / `LUMA_MAX` | 15 / 240 | Exposure bounds |
| `PHASH_MAX_DISTANCE` | 8 | Below this Hamming distance a frame is a duplicate |
| `NEIGHBOR_WINDOW_S` | 0.7 | Search window for a replacement frame |
| `MAX_FRAME_SIDE` | 1568 | Longest side sent to the model |
| `VLM_CONCURRENCY` | 4 | Batches in flight |
| `VLM_RETRY_DELAY_S` | 2.0 | Backoff before the single retry |

## Tests

```bash
uv run pytest
```

65 tests, no network and no Postgres required — they run on in-memory SQLite against the
fake provider.

## Known limitations

These ship in the report itself, in Russian, rather than being left implicit:

- **PPE detection is unreliable at altitude.** Helmets and vests cannot be judged when a
  person occupies less than roughly 40 px of frame height. The prompt instructs the model to
  lower confidence and skip such judgments rather than guess.
- **Equipment counts are a lower bound**, not unique counts — maximum simultaneously visible
  per frame. Unique counting across a flight needs object tracking, which is out of scope.
- **Analysis is not deterministic.** Re-running the same video can yield a different set of
  findings and different counts.
- **Wheel-wash compliance is properly verified at the gate**, not from the air. The report
  records only visibly dirty vehicles and whether a wash station is in frame.

A failed analysis batch is recorded and skipped rather than aborting the video; the report
publishes `meta.coverage_pct` so partial coverage is visible instead of silent.

## Layout

```
app/
  api.py, main.py        FastAPI routes and app
  config.py              all tunable thresholds
  models.py, db.py       SQLAlchemy models
  storage.py             media storage behind a key-based API (S3 is a config switch later)
  pipeline/
    probe.py             duration, fps, resolution
    motion.py            low-res motion curve
    schedule.py          scene cuts + adaptive keyframe schedule
    quality.py           blur, exposure, perceptual hash
    extract.py           full-res extraction, filtering, dedup
    run.py               orchestrator
  vlm/
    schema.py            enforced response schema + closed vocabulary
    prompt.py            Russian inspection prompt
    client.py            VlmClient protocol, Fake and Gemini implementations
    boxes.py             box validation
    analyze.py           batching, concurrency, retry, persistence
    render.py            box rendering
  aggregate.py           merge findings, decide stage, inventory, timeline, summary
  report.py              the report contract + persistence
web/                     Next.js dashboard (in progress)
docs/                    specification and implementation plans
tests/                   65 tests
```

## Roadmap

S3 and a real task queue in place of the local-disk and in-process equivalents; auth and
RBAC; a human review workflow for findings; a local YOLO PPE detector with crop verification
to fix small-object accuracy; equipment tracking for unique counts; cross-visit comparison;
PDF export.
