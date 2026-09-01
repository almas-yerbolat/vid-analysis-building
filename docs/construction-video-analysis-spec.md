# Construction Site Video Analysis System
## Technical Specification & Implementation Plan

**Version:** 1.1 (bbox overlays added to POC scope) · **Date:** 2026-09-01
**Mode:** Batch (upload video → get report) · **VLM:** Google Vertex AI (Gemini Flash) · **Delivery:** Web app + PDF export
**Scope:** Proof of concept — single organization, single video per run, clips of 1–2 minutes, no audio, **bounding boxes on findings: yes** (§4.3.1)

---

## 1. Overview

The system ingests construction-site videos (drone and ground footage) and photos, intelligently selects representative frames, analyzes them with a cloud Vision-Language Model (VLM), and produces a structured inspection report. Every finding in the report references the exact frame(s) it was detected in; clicking a reference opens the frame in a pull-up viewer with the AI comment and metadata.

### Analysis checklist (per the requirements)

The VLM evaluates each scene against six dimensions:

1. **Стадия строительства** — котлован / фундамент / каркас / кровля / фасад / благоустройство.
2. **Грубые нарушения на площадке** — стихийные места складирования, отсутствие защитно-улавливающих сеток (ЗУС), отсутствие ограждений, отсутствие обозначенных маршрутов для техники, загромождение пожарных и технологических проездов.
3. **Спецтехника и краны** — наличие и подсчёт по категориям (башенные/автокраны, экскаваторы, самосвалы, бетононасосы, погрузчики и т.д.).
4. **Активность** — что происходит в кадре (бетонирование, монтаж опалубки, разгрузка, простой и т.д.).
5. **ТБ и ОТ** — рабочие без касок / сигнальных жилетов; отсутствие защитных ограждений на краях перекрытий; незакрытые шахты лифтов; некорректно установленные леса.
6. **Экология и клининг** — несанкционированные свалки строительного мусора; выезд грязной техники (отсутствие мойки колёс перед выездом на городскую дорогу).

### Output

A report per video containing: an executive summary, construction-stage assessment, equipment inventory, a findings list grouped by category and severity, and a stage/activity timeline. Each finding carries clickable frame references. The report is viewable in the web app and exportable to PDF.

---

## 2. Scope Decisions (confirmed with the client)

- **Language:** reports, findings and PDF in Russian.
- **Input:** drone/ground clips of **1–2 minutes**, no audio track (nothing in the pipeline uses audio); photos supported as single frames. No drone telemetry — sampling relies on visual motion analysis.
- **VLM:** Google **Vertex AI**. Recommended model: current **Gemini Flash** generation with platform-enforced structured JSON output (details in §4.3). Note: "Gemma"/PaliGemma are open-weight models available in Vertex Model Garden — deployable, but a weaker fit for strict structured Russian JSON; keep them as a fallback only if open weights ever become a requirement.
- **Bounding boxes: in scope.** Findings are localized on the frame with boxes drawn over the evidence image (web overlay + burned into the PDF). Implemented with **Gemini's native object-detection output** (normalized `box_2d` coordinates) — no separate detector needed for the POC. Design in §4.3.1. This is the main reason Gemini, not Gemma, is the required model: Gemma-family open models do not produce reliable grounded coordinates.
- **Other POC reductions:** no human review/approval workflow; single tenant, no multi-project RBAC; single video per run, no streaming, no cross-visit comparison. These trims are reflected in §3, §5 and the plan in §7.
- **Volume:** not a constraint at POC stage; cost per video is negligible (§4.6).

---

## 3. System Architecture

```
┌────────────┐   presigned    ┌─────────────┐
│  Web App   │───upload──────▶│ Object store │  (S3-compatible: AWS S3 / MinIO)
│ (Next.js)  │                │ videos/frames│
└─────┬──────┘                └──────┬──────┘
      │ REST + SSE                   │
┌─────▼──────────────┐        ┌──────▼───────────────────────────┐
│  API Backend       │──jobs─▶│  Worker pool (Celery + Redis)    │
│  (FastAPI)         │        │  1. probe & transcode (ffmpeg)   │
│  auth, reports,    │        │  2. adaptive frame sampling      │
│  findings, PDF     │        │  3. quality filter + dedup       │
└─────┬──────────────┘        │  4. VLM batch analysis  ─────────┼──▶ Cloud VLM API
      │                       │  5. aggregation & report build   │
┌─────▼──────┐                └──────────────────────────────────┘
│ PostgreSQL │  videos · frames · findings · reports · users
└────────────┘
```

**Stack summary:** Next.js + TypeScript (frontend) · FastAPI + Python 3.12 (API) · Celery + Redis (job queue) · PostgreSQL (metadata & findings) · S3/MinIO (media) · ffmpeg + OpenCV + PySceneDetect (video processing) · cloud VLM API (analysis) · Playwright (HTML→PDF).

Everything is containerized (Docker Compose for dev, can move to k8s later). The pipeline is a chain of idempotent tasks so a failed stage retries without redoing the whole video.

**POC simplification.** For the POC the same shape collapses into two containers: `api` (FastAPI, also serving the built frontend) and `worker`. For 1–2-minute clips processing takes well under a minute, so a full Celery deployment is overkill — RQ or even FastAPI `BackgroundTasks` with a status table is enough; keep the task chain structure so Celery is a drop-in later. Storage: local disk volume behind a `Storage` interface (S3 becomes a config switch, not a rewrite). Database: PostgreSQL in Compose. Auth: none or a single shared token. The full architecture above is the target if the POC graduates.

---
## 4. AI Pipeline Specification

### 4.1 Frame extraction & adaptive sampling

**Goal:** cover the whole video with the minimum number of frames, densifying where the drone moves/turns fast, and never sending near-duplicate or unusable frames to the VLM.

**Step 1 — Probe & normalize.** `ffprobe` extracts duration, fps, resolution, rotation. If the codec is exotic, transcode once to H.264. Nothing else is re-encoded.

**Step 2 — Low-res motion scan.** Decode the video at ~4 fps, downscaled to 320 px width (cheap, single pass with PyAV/OpenCV). For each consecutive pair compute a **motion score**: mean magnitude of Farneback optical flow (or, cheaper, ECC/global histogram difference). This yields a motion curve `m(t)` for the whole video in seconds of compute, not minutes.

**Step 3 — Adaptive keyframe schedule.**

- Baseline: one candidate frame every **5 s** (as agreed).
- For every 5-second interval where `mean(m) > T_fast` (drone panning/turning), subdivide: sample every **1–2 s** inside that interval, so no scenery is skipped between keyframes.
- Additionally run PySceneDetect `ContentDetector` on the low-res stream; every detected scene cut adds a candidate frame right after the cut (covers hard cuts in edited videos).
- `T_fast` is calibrated once on sample footage (start with the 75th percentile of `m` on a reference video and tune).

**Step 4 — Full-res extraction.** For each scheduled timestamp, extract the frame at source resolution with ffmpeg (`select` filter by timestamp), resize to **1568 px on the longest side** (a good ceiling for cloud VLM vision inputs — larger adds tokens without adding accuracy), save as JPEG q85 to S3 under `frames/{video_id}/{ts_ms}.jpg`.

### 4.2 Quality filter & deduplication

- **Blur filter:** variance of Laplacian below threshold → frame is blurred (motion blur during fast turns). Replace it with the sharpest neighbor within ±0.7 s; if none passes, keep the best available and mark `low_quality=true` so the VLM prompt can note reduced confidence.
- **Exposure filter:** reject frames with mean luma < 15 or > 240 (against sun-flare / near-black frames), same neighbor-replacement logic.
- **Deduplication:** compute perceptual hash (pHash, 64-bit) per frame; if Hamming distance to the previous *kept* frame < 8, drop the frame. On a hovering drone this typically removes 30–50% of baseline frames — direct cost savings.

Every kept frame is stored in the `frames` table with `ts_ms`, `s3_key`, `motion_score`, `phash`, and `selected_reason ∈ {baseline, fast_motion, scene_cut, photo}`.

### 4.3 VLM analysis

**Batching strategy:** frames are sent in **groups of 4 sequential frames per request**, each labeled with its timestamp. Sequential grouping gives the model temporal context ("the same truck across frames", "the crane is actually moving") while keeping requests parallelizable. Groups are processed concurrently by workers with rate-limit-aware backoff.

**Model (Vertex AI):** current-generation **Gemini Flash** via the `google-genai` Python SDK pointed at the project's Vertex endpoint, called with `response_mime_type="application/json"` and a `response_schema` — the JSON schema below is then enforced by the platform, not by prompt discipline alone (see Google's structured-output docs for Vertex). Verify the exact model id against the Vertex model list at implementation time (the Flash line iterates quickly). The pipeline hides the provider behind a `VlmClient` interface, so Claude or another API can be benchmarked without touching the pipeline.

**Option B for the POC — native video input.** Gemini on Vertex accepts video files directly. For 1–2-minute clips it is viable to send the whole video once and ask for timestamped findings, then extract only the referenced frames for the viewer. Less code, but it gives up deterministic frame selection, dedup control, and exact evidence timestamps (model-reported timestamps can drift by a few seconds). The frame-based path (Option A, this section) remains the primary spec because precise clickable evidence frames are a core requirement — but Option B is worth an afternoon benchmark in Step 1 of the plan.

**Prompt design (per batch), summarized:**

- System prompt: role of a construction-site inspection expert; the six-category checklist verbatim (in Russian); strict instruction to answer **only** with JSON matching the schema; instruction to report `confidence` per finding and to avoid guessing when the frame is too far/blurred (`low_quality` flag passed in).
- User content: 4 images + their timestamps + video-level context (project name, previous batch's detected stage — helps consistency).
- Use the provider's structured-output / JSON mode where available; otherwise validate with Pydantic and retry once on schema failure.
- **Localization instruction:** for every finding, return `boxes` — one bounding box per involved object (each worker without a helmet, the pile blocking the lane) in Gemini's native detection format: `box_2d = [y_min, x_min, y_max, x_max]`, normalized to **0–1000** relative to the image, plus a short `label` per box. If the violation is scene-wide and unlocalizable (e.g., «нет обозначенных маршрутов»), `boxes` is an empty list and the evidence is the whole frame.

### 4.3.1 Bounding boxes (Gemini native detection)

Gemini models expose grounded object detection out of the box: prompted for coordinates, they return `box_2d` in the 0–1000 normalized convention above, which the backend converts to pixel space (`x_px = x/1000 · width`). This gives the POC clickable, visual evidence without training or hosting a detector. Rules the pipeline enforces on receipt:

- **Validate & clip:** clip coordinates to [0, 1000]; drop degenerate boxes (`y_max ≤ y_min`, area < 0.02% of frame) and boxes duplicated within IoU > 0.9.
- **Coordinate integrity:** boxes are computed against the exact image sent to the model, so the stored frame (post-resize, §4.1) is the single source of truth; the frontend scales boxes against the rendered image size, never against original video resolution.
- **Honest imprecision:** VLM boxes are approximate (typically off by a few percent, worse on tiny/overlapping objects). Boxes are rendered as *indication*, not measurement: the UI draws them with the finding's severity color and shows `label`; findings whose boxes were all dropped in validation fall back to whole-frame evidence rather than showing garbage.
- **Small-object floor:** the prompt instructs the model not to emit boxes for people smaller than ~1.5% of frame height (unreliable at drone altitude); such cases become frame-level findings with lower confidence.

Post-POC upgrade path unchanged: a local YOLO person/PPE detector supplies precise boxes and the VLM verifies crops (§7 backlog).

**Per-frame JSON schema (VLM output):**

```json
{
  "frames": [
    {
      "ts_ms": 125000,
      "stage": "каркас",
      "stage_confidence": 0.9,
      "activity": "Монтаж опалубки на 6-м этаже, подача бетона автобетононасосом",
      "equipment": [
        {"type": "башенный_кран", "count": 2},
        {"type": "автобетононасос", "count": 1},
        {"type": "самосвал", "count": 3}
      ],
      "findings": [
        {
          "category": "тб_от",
          "subtype": "отсутствие_каски",
          "severity": "high",
          "comment": "На перекрытии 6-го этажа два рабочих без касок (правый край кадра).",
          "confidence": 0.8,
          "boxes": [
            {"label": "рабочий без каски", "box_2d": [412, 806, 471, 843]},
            {"label": "рабочий без каски", "box_2d": [405, 861, 468, 902]}
          ]
        },
        {
          "category": "нарушения_площадки",
          "subtype": "загромождение_проезда",
          "severity": "medium",
          "comment": "Пожарный проезд вдоль южного фасада частично заблокирован поддонами с кирпичом.",
          "confidence": 0.75,
          "boxes": [
            {"label": "поддоны в проезде", "box_2d": [618, 120, 742, 388]}
          ]
        }
      ]
    }
  ]
}
```

**Category / subtype dictionary (closed vocabulary, enforced):**

| category | subtypes |
|---|---|
| `нарушения_площадки` | `стихийное_складирование`, `нет_зус`, `нет_ограждения_площадки`, `нет_маршрутов_техники`, `загромождение_проезда` |
| `тб_от` | `отсутствие_каски`, `отсутствие_жилета`, `нет_ограждения_перекрытия`, `открытая_шахта_лифта`, `нарушение_установки_лесов` |
| `экология_клининг` | `свалка_мусора`, `грязная_техника_выезд`, `нет_мойки_колес` |

A closed vocabulary makes findings groupable, filterable, and comparable across videos; free text lives only in `comment`.

**Severity:** `critical` (immediate life-safety: открытая шахта, работа на краю без ограждения), `high` (ТБ violations, blocked fire lane), `medium` (складирование, грязная техника), `low` (cleanliness/cosmetic).

### 4.4 Aggregation & summary generation

Raw per-frame findings are noisy: the same violation appears in 10 consecutive frames. Aggregation runs after all batches finish:

1. **Finding merge:** group findings with identical `category+subtype` whose frames are within 60 s of each other (and, for drone orbits, whose comments the aggregation model judges to describe the same object) into a single **finding** with a list of evidence frames. The best 1–3 frames (sharpest, highest confidence) become the primary references.
2. **Stage decision:** majority vote of per-frame `stage` weighted by confidence; if the site genuinely shows mixed stages (one block каркас, another фасад), report both with the timeline.
3. **Equipment inventory:** per equipment type, report the **maximum simultaneous count** observed in any single frame (honest lower bound of fleet size) plus the frame reference where the max was seen. True unique counting across the whole flight requires tracking — explicitly out of scope for v1, noted in the report footer.
4. **Summary pass:** one text-only LLM call receives the merged findings + stage + inventory + activity notes and produces the executive summary (RU), a severity breakdown, and recommendations. This is cheap (no images) and gives a coherent narrative instead of stitched fragments.

### 4.5 Final report format (the "response with summary")

This is the contract between backend and frontend/PDF:

```json
{
  "video_id": "…",
  "generated_at": "2026-09-01T10:00:00Z",
  "meta": {"duration_s": 1240, "frames_analyzed": 212, "frames_extracted": 486},
  "stage": {"primary": "каркас", "secondary": ["фундамент"], "confidence": 0.88,
            "evidence_frames": ["frm_017", "frm_142"]},
  "summary_ru": "На объекте ведутся работы стадии «каркас»… Выявлено 2 критических и 5 значимых нарушений…",
  "equipment": [
    {"type": "башенный_кран", "max_count": 2, "evidence_frame": "frm_063"},
    {"type": "экскаватор", "max_count": 1, "evidence_frame": "frm_010"}
  ],
  "activity_timeline": [
    {"from_ms": 0, "to_ms": 300000, "activity": "Облет северного фасада, монтаж опалубки"},
    {"from_ms": 300000, "to_ms": 720000, "activity": "Зона складирования, разгрузка самосвалов"}
  ],
  "findings": [
    {
      "id": "fnd_004",
      "category": "тб_от",
      "subtype": "отсутствие_каски",
      "severity": "high",
      "title": "Рабочие без касок на перекрытии 6-го этажа",
      "comment": "Зафиксировано на кадрах 02:05–02:20…",
      "status": "unreviewed",
      "evidence": [
        {"frame_id": "frm_025", "ts_ms": 125000, "thumb_url": "…", "full_url": "…",
         "comment": "Два рабочих без касок, правый край кадра",
         "boxes": [
           {"label": "рабочий без каски", "box_2d": [412, 806, 471, 843]},
           {"label": "рабочий без каски", "box_2d": [405, 861, 468, 902]}
         ]}
      ]
    }
  ],
  "stats": {"critical": 2, "high": 5, "medium": 4, "low": 3}
}
```

### 4.6 Cost & throughput estimation (formula)

Let `D` = video duration in seconds. Baseline frames ≈ `D/5`; adaptive densification adds ~10–20%; dedup removes ~30–40%. Net frames `F ≈ 0.8 · D/5`. Requests ≈ `F/4`.

Example for the POC, 2-minute video: `F ≈ 0.8 · 120/5 ≈ 19` frames ≈ **5 VLM requests**. Token load per request ≈ 4 images + ~1k prompt + ~1.5k output — at current Gemini Flash pricing this lands in single-digit cents per video, i.e. cost is a non-issue at POC scale (plug exact per-token prices from the Vertex AI pricing page at implementation time). Wall-clock end-to-end (extraction → analysis → report): typically **under a minute** with 4 concurrent requests. The formula still holds if longer videos arrive later.

### 4.7 Known limitations & mitigations

- **Small-object PPE detection from high altitude:** helmets/vests are unreliable for a VLM when a person is < ~40 px tall. Mitigation now: the prompt instructs the model to report `confidence` and skip judgments on tiny figures; report footer states the altitude limitation. Mitigation later (Phase 4): local YOLO person/PPE detector on full-res frames, crops of detected people sent to the VLM for verification.
- **Unique equipment counting** needs multi-object tracking (ByteTrack/OC-SORT) — Phase 4.
- **Wheel-wash control** is best verified at the gate, not from a drone; v1 reports visibly dirty trucks near the exit and the presence/absence of a wash station in frame. A fixed gate camera is the correct long-term source for this check.

---

## 5. Backend Specification

### 5.1 Data model (PostgreSQL)

```
projects   (id, name, address, created_at)
videos     (id, project_id, filename, s3_key, duration_s, resolution, status,
            uploaded_by, created_at)
            status ∈ {uploaded, probing, sampling, analyzing, aggregating, done, failed}
frames     (id, video_id, ts_ms, s3_key, thumb_s3_key, width, height,
            motion_score, phash, low_quality, selected_reason)
analyses   (id, video_id, batch_index, frame_ids[], raw_response jsonb,
            model, tokens_in, tokens_out, status)
findings   (id, video_id, category, subtype, severity, title, comment,
            confidence, status ∈ {unreviewed, confirmed, rejected},
            reviewer_comment, reviewed_by, reviewed_at)
finding_frames (finding_id, frame_id, frame_comment,
                boxes jsonb)   -- evidence links; boxes: [{label, box_2d[4]}] in 0–1000 norm
reports    (id, video_id, report_json jsonb, summary_ru, pdf_s3_key, created_at)
users      (id, email, role ∈ {admin, inspector, viewer}, …)
```

`analyses.raw_response` keeps the untouched VLM output — essential for debugging prompts and re-aggregating without re-paying for vision calls.

### 5.2 API endpoints (FastAPI, REST + SSE)

```
POST   /api/videos/upload-init          → presigned S3 URL(s), video_id   (multipart for >5 GB)
POST   /api/videos/{id}/analyze         → enqueue pipeline, returns job id
GET    /api/videos/{id}/status          → SSE stream: stage + percent (sampling 12%, batch 34/73…)
GET    /api/videos/{id}/report          → report JSON (§4.5)
GET    /api/videos/{id}/report.pdf      → generated PDF (or 202 while rendering)
GET    /api/frames/{id}                 → frame image (full-res or thumb)
GET    /api/videos                      → listing/history
POST   /api/photos/upload               → photos enter the same pipeline as single frames
```

Auth (POC): none or a single shared bearer token. The human-review `PATCH /findings` endpoint and role model are out of POC scope — findings are read-only.

### 5.3 Workers (Celery task chain per video)

```
probe_video → motion_scan → schedule_keyframes → extract_frames
  → filter_and_dedup → fanout: analyze_batch × N (rate-limited group)
  → aggregate_findings → generate_summary → build_report → render_pdf
```

Each task writes its progress to Redis (consumed by the SSE endpoint), is idempotent (keyed by video_id + stage), and retries with exponential backoff. VLM calls have a per-provider rate limiter and a circuit breaker; a batch that fails schema validation retries once with a "fix your JSON" reminder, then is flagged and skipped (report notes coverage %).

### 5.4 PDF export

The report page has a print-optimized route (`/report/{id}/print`); a worker renders it to PDF with Playwright (headless Chromium) — one styling source for web and PDF. The PDF embeds: title page with project/date/stats, executive summary, stage & equipment tables, findings with their primary evidence frames inlined as images and timestamps, appendix with methodology & limitations. Evidence images in the PDF have bounding boxes **burned in server-side** (Pillow: severity-colored rectangle + label chip) so the printed report is self-contained; the annotated variant is cached to S3 next to the clean frame (`…/{ts_ms}_annotated.jpg`).

---

## 6. Frontend Specification (Next.js + TypeScript + Tailwind)

### 6.1 Pages

1. **Upload** — drag-and-drop for videos/photos, project selector, chunked/resumable upload straight to S3 (presigned), then "Analyze" button.
2. **Processing** — live status via SSE: stepper (Извлечение кадров → Отбор → Анализ 34/73 → Сводка) with percent and ETA; list of already-extracted thumbnails appears progressively so the user sees life before the report is ready.
3. **Report** — the core screen, below.
4. **History** — videos per project, status, quick stats, re-open reports.

### 6.2 Report screen

Layout (desktop): left column — executive summary card, stage badge with confidence, severity stat chips (2 critical / 5 high / …), equipment table, activity timeline (horizontal bar with segments; clicking a segment scrolls to that period's findings). Right/main column — findings list.

**Finding card:** severity chip, title, comment, category tag, evidence thumbnails strip (1–3), review controls (Подтвердить / Отклонить / комментарий) if Q3 is confirmed.

**Frame viewer (the "pullable view"):** clicking any thumbnail or inline frame reference opens:

- **mobile:** a bottom sheet, pullable between half and full height (swipe down to dismiss);
- **desktop:** a modal with the same content.

Contents: full-resolution frame (pinch-zoom / scroll-zoom) with an **SVG bounding-box overlay** — boxes from `evidence[].boxes` scaled from 0–1000 space to the rendered image size, colored by severity, with the box `label` shown on tap/hover, and a «Рамки вкл/выкл» toggle; the overlay pans/zooms together with the image (single transformed container). Also: overlay bar with timestamp (`02:05`), category & severity, the **frame-level VLM comment**, prev/next arrows to walk the evidence frames of that finding, and a "показать в контексте" strip of ±2 neighboring extracted frames. Deep-linkable: `/report/{videoId}?frame=frm_025` opens the sheet directly — this is what makes frame references in the summary text clickable (summary text renders `[кадр 02:05]` tokens as links).

**Filters:** by category, severity, review status; full-text search across comments.

### 6.3 Export & sharing

"Скачать PDF" button (polls until worker finishes), plus report JSON export for integrations. Optional share-link with viewer-role token.

---

## 7. Step-by-Step POC Plan (~3 weeks, one engineer)

**Step 1 — Sampling + prompt spike (days 1–3).**
A CLI script, no services: motion scan → adaptive schedule → extraction → blur/exposure filter → pHash dedup, run on 2–3 real clips; hand-tune `T_fast`, pHash and blur thresholds by eyeballing the kept-frame contact sheet. Then call Gemini Flash with `response_schema` on the sampled frames and review output quality with a domain expert (прораб/инспектор); iterate on the checklist prompt and subtype vocabulary. Benchmark **Option B** (native video input, §4.3) on the same clips for comparison. **Validate box quality specifically:** render returned `box_2d` on 30–50 frames as a contact sheet and eyeball localization accuracy — this decides how prominently boxes are presented in the UI (solid "evidence" vs dashed "indication"). **Exit criteria:** stable JSON on every run; the expert agrees with stage classification and with the majority of findings on all test clips; boxes land on the right object in ≥80% of localizable findings.

**Step 2 — Pipeline service (days 4–8).**
Docker Compose: `api` + `worker` + Postgres. Endpoints: upload, analyze, SSE status, report JSON, frame serving. Persist frames/analyses/findings; aggregation + summary pass. **Exit:** upload via API → complete report JSON with evidence frame links, automatically.

**Step 3 — Frontend (days 8–13).**
Upload page → processing screen with SSE progress → report screen: summary card, stage badge, severity chips, equipment table, findings list, and the bottom-sheet/modal frame viewer with deep links (`?frame=…`) and prev/next navigation. Category/severity filters.

**Step 4 — PDF & polish (days 13–15).**
Playwright HTML→PDF with embedded evidence frames; photos-as-input; error and low-confidence states («требует проверки» section for findings with confidence < 0.6); README + demo script.

**Post-POC backlog (if it graduates):** S3 + Celery swap-in, auth/RBAC, human review workflow, local YOLO PPE detector with crop verification (small-object accuracy), equipment tracking for unique counts, cross-visit comparison, gate camera for wheel-wash control.

---

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| VLM misses small PPE violations at altitude | False negatives in ТБ | Confidence gating + Phase-4 detector; state limitation in report |
| VLM bounding boxes imprecise / on wrong object | Misleading visual evidence | Validation & clipping (§4.3.1), small-object floor, "indication not measurement" rendering, Step-1 contact-sheet check; fallback to whole-frame evidence |
| Hallucinated violations | Trust loss | Closed subtype vocabulary, confidence threshold (< 0.6 → "требует проверки" section), human review workflow |
| Cost blow-up on long videos | Budget | Dedup + adaptive sampling + provider batch tier; per-video token cap with alert |
| Provider rate limits | Slow processing | Concurrency limiter, batch API, second provider behind `VlmClient` |
| Large uploads over weak networks | UX failure | Resumable multipart upload direct to S3 |

---

## 9. Prerequisites to Start Step 1

1. **2–3 sample videos** (1–2 min, representative of real footage: at least one with fast drone turns, one with visible workers/equipment).
2. **GCP project** with Vertex AI API enabled and a service account key (roles: Vertex AI User).
3. A domain expert (прораб / инженер по ТБ) available for ~1 hour to review Step-1 output — this single review is the highest-leverage hour in the whole POC.
