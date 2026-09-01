import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app import storage
from app.aggregate import (
    activity_timeline,
    decide_stage,
    equipment_inventory,
    merge_findings,
)
from app.pipeline.extract import extract_frames
from app.pipeline.motion import motion_curve
from app.pipeline.probe import probe
from app.pipeline.schedule import scene_cuts, schedule_keyframes
from app.vlm.client import get_client
from app.vlm.render import draw_boxes


def contact_sheet(images: list[np.ndarray], labels: list[str], cols: int = 5) -> np.ndarray:
    thumbs = []
    for img, label in zip(images, labels):
        h = round(img.shape[0] * 320 / img.shape[1])
        thumb = cv2.resize(img, (320, h))
        cv2.putText(thumb, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        thumbs.append(thumb)
    max_h = max(thumb.shape[0] for thumb in thumbs)
    thumbs = [
        cv2.copyMakeBorder(thumb, 0, max_h - thumb.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for thumb in thumbs
    ]
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        row += [np.zeros_like(thumbs[0])] * (cols - len(row))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def _sample(video_path: str) -> tuple[list, list]:
    info = probe(video_path)
    motion = motion_curve(video_path)
    cuts = scene_cuts(motion)
    keyframes = schedule_keyframes(info.duration_s, motion, cuts)
    extracted = extract_frames(video_path, "cli", keyframes, motion)
    return keyframes, extracted


def run_sample(video_path: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keyframes, extracted = _sample(video_path)
    images, labels = [], []
    for extracted_frame in extracted:
        source = storage.path_for(extracted_frame.media_key)
        img = cv2.imread(str(source))
        shutil.copy(source, out / f"{extracted_frame.ts_ms}.jpg")
        minutes, seconds = divmod(extracted_frame.ts_ms // 1000, 60)
        labels.append(
            f"{minutes:02d}:{seconds:02d} {extracted_frame.selected_reason}"
            + (" LQ" if extracted_frame.low_quality else "")
        )
        images.append(img)
    cv2.imwrite(str(out / "contact_sheet.jpg"), contact_sheet(images, labels))
    stats = {
        "candidates": len(keyframes),
        "kept": len(extracted),
        "dropped": len(keyframes) - len(extracted),
        "reasons": dict(Counter(extracted_frame.selected_reason for extracted_frame in extracted)),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def run_analyze(video_path: str, out_dir: str, draw: bool) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, extracted = _sample(video_path)
    client = get_client()
    analyses = []
    for i in range(0, len(extracted), 4):
        batch = extracted[i:i + 4]
        result = client.analyze_batch(
            [storage.read_bytes(extracted_frame.media_key) for extracted_frame in batch],
            [extracted_frame.ts_ms for extracted_frame in batch],
            [extracted_frame.low_quality for extracted_frame in batch],
            "CLI",
        )
        analyses.extend(result.parsed.clean().frames)
    merged = merge_findings(analyses)
    if draw:
        annotated = out / "annotated"
        annotated.mkdir(exist_ok=True)
        by_ts = {extracted_frame.ts_ms: extracted_frame for extracted_frame in extracted}
        annotated_images = {}
        for merged_finding in merged:
            for ts, _, boxes in merged_finding.evidence:
                if boxes and ts in by_ts:
                    img = annotated_images.get(ts)
                    if img is None:
                        img = cv2.imread(str(storage.path_for(by_ts[ts].media_key)))
                    annotated_images[ts] = draw_boxes(img, boxes, merged_finding.severity)
        for ts, img in annotated_images.items():
            cv2.imwrite(str(annotated / f"{ts}.jpg"), img)
    output = {
        "stage": decide_stage(analyses),
        "equipment": equipment_inventory(analyses),
        "timeline": activity_timeline(analyses),
        "findings": [
            {
                "category": merged_finding.category,
                "subtype": merged_finding.subtype,
                "severity": merged_finding.severity,
                "title": merged_finding.title,
                "confidence": merged_finding.confidence,
                "evidence_ts": [evidence[0] for evidence in merged_finding.evidence],
            }
            for merged_finding in merged
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="Step-1 spike: sampling + VLM analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("video")
    sample_parser.add_argument("--out", default="./spike_out")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("video")
    analyze_parser.add_argument("--out", default="./spike_out")
    analyze_parser.add_argument("--draw-boxes", action="store_true")
    args = parser.parse_args()
    if args.cmd == "sample":
        run_sample(args.video, args.out)
    else:
        run_analyze(args.video, args.out, args.draw_boxes)


if __name__ == "__main__":
    main()
