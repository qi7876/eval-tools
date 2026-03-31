"""Prepared-data cache generation."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional fast path
    import av  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment dependent
    av = None
import cv2
from PIL import Image

from .dataset import dataset_fingerprint, scan_dataset_report, summarize_scan_report
from .protocols import (
    get_protocol,
    original_interval_to_sampled_interval,
    protocol_supports_task,
    sample_frames_for_sample,
    sampled_to_original_mapping,
)
from .schema import PreparedSample, ProtocolSpec, SampleRecord
from .utils import (
    ensure_directory,
    read_json,
    read_jsonl,
    sample_bundle_dir,
    write_json,
    write_jsonl,
)


def decode_selected_frames(video_path: Path, frame_indices: list[int]) -> dict[int, Image.Image]:
    if not frame_indices:
        return {}
    if av is not None:
        return _decode_selected_frames_with_av(video_path, frame_indices)
    return _decode_selected_frames_with_cv2(video_path, frame_indices)


def _decode_selected_frames_with_av(
    video_path: Path,
    frame_indices: list[int],
) -> dict[int, Image.Image]:
    targets = sorted(set(frame_indices))
    results: dict[int, Image.Image] = {}
    target_cursor = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if target_cursor >= len(targets):
                break
            target = targets[target_cursor]
            if frame_index < target:
                continue
            if frame_index == target:
                results[frame_index] = frame.to_image().convert("RGB")
                target_cursor += 1
                continue
            while target_cursor < len(targets) and targets[target_cursor] < frame_index:
                target_cursor += 1
            if target_cursor >= len(targets):
                break
            if targets[target_cursor] == frame_index:
                results[frame_index] = frame.to_image().convert("RGB")
                target_cursor += 1
    missing = [index for index in targets if index not in results]
    if missing:
        raise ValueError(f"{video_path} is missing decoded frames: {missing[:10]}")
    return results


def _decode_selected_frames_with_cv2(
    video_path: Path,
    frame_indices: list[int],
) -> dict[int, Image.Image]:
    targets = sorted(set(frame_indices))
    results: dict[int, Image.Image] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"failed to open video with OpenCV: {video_path}")
    try:
        for target in targets:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(f"{video_path} is missing decoded frame {target}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results[target] = Image.fromarray(rgb)
    finally:
        capture.release()
    return results


def _prepare_tracking_rows(
    tracking_original: list[dict[str, Any]],
    sampled_frames_original: list[int],
    *,
    valid_interval: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    by_original = {
        int(row["frame_original"]): [float(value) for value in row["bbox_mot"]]
        for row in tracking_original
    }
    aligned: list[dict[str, Any]] = []
    for sampled_index, frame_original in enumerate(sampled_frames_original):
        if valid_interval is not None:
            start, end = valid_interval
            if frame_original < start or frame_original > end:
                continue
        bbox = by_original.get(frame_original)
        if bbox is None:
            continue
        aligned.append(
            {
                "frame_sampled": sampled_index,
                "frame_original": frame_original,
                "bbox_mot": bbox,
            }
        )
    return aligned


def _prepare_reference_payload(
    sample: SampleRecord,
    sampled_frames_original: list[int],
) -> dict[str, Any]:
    payload = deepcopy(sample.reference_payload)
    if "segments_original" in payload:
        payload["segments_sampled"] = [
            {
                "start_sampled": original_interval_to_sampled_interval(
                    segment["start_frame"],
                    segment["end_frame"],
                    sampled_frames_original,
                )[0],
                "end_sampled": original_interval_to_sampled_interval(
                    segment["start_frame"],
                    segment["end_frame"],
                    sampled_frames_original,
                )[1],
                "text": segment["text"],
            }
            for segment in payload["segments_original"]
        ]
    if sample.task_name == "Continuous_Actions_Caption":
        payload["tracking_gt_sampled"] = _prepare_tracking_rows(
            payload.get("tracking_original", []),
            sampled_frames_original,
        )
    if sample.task_name == "Spatial_Temporal_Grounding":
        start_frame, end_frame = sample.a_window or tuple(payload["time_window_original"])
        payload["time_window_sampled"] = original_interval_to_sampled_interval(
            start_frame,
            end_frame,
            sampled_frames_original,
        )
        payload["tracking_gt_sampled"] = _prepare_tracking_rows(
            payload.get("tracking_original", []),
            sampled_frames_original,
            valid_interval=(start_frame, end_frame),
        )
    return payload


def build_prepared_sample(
    sample: SampleRecord,
    protocol: ProtocolSpec,
    sampled_frames_original: list[int],
    bundle_dir: Path,
    decoded_frames: dict[int, Image.Image],
) -> PreparedSample:
    frames_dir = ensure_directory(bundle_dir / "frames")
    frame_files: list[str] = []
    for sampled_index, frame_original in enumerate(sampled_frames_original):
        output_name = f"{sampled_index:04d}.jpg"
        output_path = frames_dir / output_name
        decoded_frames[frame_original].save(output_path, format="JPEG", quality=95)
        frame_files.append(str(Path("frames") / output_name))
    prepared = PreparedSample(
        sample_id=sample.sample_id,
        annotation_id=sample.annotation_id,
        video_key=sample.video_key,
        task_name=sample.task_name,
        task_level=sample.task_level,
        protocol_id=protocol.protocol_id,
        question_text=sample.question_text,
        sampled_frames_original=sampled_frames_original,
        sampled_to_original=sampled_to_original_mapping(sampled_frames_original),
        frame_files=frame_files,
        source_video_path=str(sample.source_video_path),
        source_annotation_path=str(sample.source_annotation_path),
        reference_payload=_prepare_reference_payload(sample, sampled_frames_original),
        timestamp_frame=sample.timestamp_frame,
        q_window=sample.q_window,
        a_window=sample.a_window,
        source_tracking_path=(
            str(sample.source_tracking_path) if sample.source_tracking_path is not None else None
        ),
        upstream_annotation_id=sample.upstream_annotation_id,
        metadata={
            "num_sampled_frames": len(sampled_frames_original),
            "video_total_frames": sample.video_metadata.total_frames,
        },
    )
    write_json(bundle_dir / "manifest.json", prepared.to_dict())
    return prepared


def _protocol_stats(prepared_samples: list[PreparedSample]) -> dict[str, Any]:
    by_task: dict[str, int] = defaultdict(int)
    total_frames = 0
    for prepared in prepared_samples:
        by_task[prepared.task_name] += 1
        total_frames += len(prepared.sampled_frames_original)
    return {
        "num_samples": len(prepared_samples),
        "task_counts": dict(sorted(by_task.items())),
        "total_sampled_frames": total_frames,
        "avg_sampled_frames": (total_frames / len(prepared_samples)) if prepared_samples else 0.0,
    }


def _build_video_cache_entries(
    video_path: Path,
    video_records: list[SampleRecord],
    protocol: ProtocolSpec,
    samples_root: Path,
) -> tuple[list[PreparedSample], list[dict[str, Any]]]:
    ordered_records = sorted(video_records, key=lambda record: record.sample_id)
    sample_plan = {
        record.sample_id: sample_frames_for_sample(record, protocol)
        for record in ordered_records
    }
    union_frames = sorted({frame for frames in sample_plan.values() for frame in frames})
    decoded_frames = decode_selected_frames(video_path, union_frames)

    prepared_samples: list[PreparedSample] = []
    index_rows: list[dict[str, Any]] = []
    for record in ordered_records:
        bundle_dir = sample_bundle_dir(samples_root, record.video_key, record.annotation_id)
        ensure_directory(bundle_dir)
        prepared = build_prepared_sample(
            record,
            protocol,
            sample_plan[record.sample_id],
            bundle_dir,
            decoded_frames,
        )
        prepared_samples.append(prepared)
        index_rows.append(
            {
                "sample_id": prepared.sample_id,
                "task_name": prepared.task_name,
                "video_key": prepared.video_key,
                "annotation_id": prepared.annotation_id,
                "manifest_path": str((bundle_dir / "manifest.json").relative_to(samples_root.parent)),
                "frame_count": len(prepared.sampled_frames_original),
            }
        )
    return prepared_samples, index_rows


def build_protocol_cache(
    records: list[SampleRecord],
    protocol: ProtocolSpec,
    prepared_root: Path,
    *,
    data_status: dict[str, Any],
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("prepare-data workers must be >= 1")
    protocol_root = ensure_directory(prepared_root / protocol.protocol_id)
    samples_root = ensure_directory(protocol_root / "samples")
    relevant_records = [
        record for record in records if protocol_supports_task(protocol, record.task_name)
    ]
    grouped_records: dict[Path, list[SampleRecord]] = defaultdict(list)
    for record in relevant_records:
        grouped_records[record.source_video_path].append(record)

    video_jobs = sorted(grouped_records.items(), key=lambda item: str(item[0]))
    prepared_samples: list[PreparedSample] = []
    index_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _build_video_cache_entries,
                video_path,
                video_records,
                protocol,
                samples_root,
            )
            for video_path, video_records in video_jobs
        ]
        for future in futures:
            video_prepared_samples, video_index_rows = future.result()
            prepared_samples.extend(video_prepared_samples)
            index_rows.extend(video_index_rows)

    prepared_samples.sort(key=lambda sample: sample.sample_id)
    index_rows.sort(key=lambda row: row["sample_id"])
    write_jsonl(protocol_root / "index.jsonl", index_rows)
    stats_payload = _protocol_stats(prepared_samples)
    stats_payload["ignored_unsupported_sample_count"] = data_status[
        "ignored_unsupported_sample_count"
    ]
    stats_payload["ignored_unsupported_task_counts"] = data_status[
        "ignored_unsupported_task_counts"
    ]
    write_json(protocol_root / "stats.json", stats_payload)
    write_json(
        protocol_root / "build_manifest.json",
        {
            "protocol": protocol.to_dict(),
            "data_status": data_status,
            "supported_dataset_fingerprint": dataset_fingerprint(records),
            "num_prepared_samples": len(prepared_samples),
        },
    )
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_root": str(protocol_root),
        "num_prepared_samples": len(prepared_samples),
        "supported_issue_count": data_status["supported_issue_count"],
        "ignored_unsupported_sample_count": data_status["ignored_unsupported_sample_count"],
    }


def build_prepared_data(
    data_root: Path,
    prepared_root: Path,
    protocol_ids: list[str],
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("prepare-data workers must be >= 1")
    scan_report = scan_dataset_report(data_root)
    if scan_report.issues:
        issue_text = "\n".join(scan_report.issues[:50])
        raise ValueError(
            f"dataset validation failed with {len(scan_report.issues)} issue(s):\n{issue_text}"
        )
    records = scan_report.supported_records
    data_status = summarize_scan_report(scan_report)
    results = []
    for protocol_id in protocol_ids:
        protocol = get_protocol(protocol_id)
        results.append(
            build_protocol_cache(
                records,
                protocol,
                prepared_root,
                data_status=data_status,
                workers=workers,
            )
        )
    return results


def load_prepared_samples(prepared_root: Path, protocol_id: str) -> list[PreparedSample]:
    protocol_root = prepared_root / protocol_id
    index_rows = read_jsonl(protocol_root / "index.jsonl")
    prepared: list[PreparedSample] = []
    for row in index_rows:
        manifest_path = protocol_root / row["manifest_path"]
        sample = PreparedSample.from_dict(read_json(manifest_path))
        sample.frame_files = [
            str((manifest_path.parent / relative_path).resolve()) for relative_path in sample.frame_files
        ]
        sample.metadata = dict(sample.metadata)
        sample.metadata["bundle_dir"] = str(manifest_path.parent.resolve())
        prepared.append(sample)
    return prepared
