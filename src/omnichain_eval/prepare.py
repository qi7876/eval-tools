"""Prepared-data cache generation."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional fast path
    import av  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment dependent
    av = None
from PIL import Image, ImageDraw

from .constants import (
    COORDINATE_SYSTEM_NORMALIZED_1000,
    SINGLE_FRAME_TASKS,
    STG_UPSTREAM_TASKS,
    VIDEO_FPS,
)
from .coordinates import (
    denormalize_mot_box_to_pixel_corners,
    normalize_corner_box_from_pixels,
    normalize_mot_box_from_pixels,
)
from .dataset import dataset_fingerprint, scan_dataset_report, summarize_scan_report
from .protocols import (
    BaseProtocol,
    original_interval_to_sampled_interval,
    protocol_supports_task,
    resolve_protocol,
    sample_frames_for_sample,
    sampled_to_original_mapping,
)
from .schema import PreparedSample, SampleRecord
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
    pyav_module = _require_pyav()
    return _decode_selected_frames_with_av(pyav_module, video_path, frame_indices)


def _decode_selected_frames_with_av(
    pyav_module: Any,
    video_path: Path,
    frame_indices: list[int],
) -> dict[int, Image.Image]:
    targets = sorted(set(frame_indices))
    results: dict[int, Image.Image] = {}
    target_cursor = 0
    with pyav_module.open(str(video_path)) as container:
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


def _prepare_tracking_rows(
    tracking_original: list[dict[str, Any]],
    sampled_frames_original: list[int],
    *,
    frame_width: int,
    frame_height: int,
    valid_interval: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    by_original = {
        int(row["frame_original"]): normalize_mot_box_from_pixels(
            row["bbox_mot"],
            frame_width=frame_width,
            frame_height=frame_height,
        )
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
    frame_width, frame_height = sample.video_metadata.resolution
    payload = deepcopy(sample.reference_payload)
    if sample.task_name == "Scoreboard_Single":
        payload["bbox"] = normalize_corner_box_from_pixels(
            payload["bbox"],
            frame_width=frame_width,
            frame_height=frame_height,
        )
    if sample.task_name == "Objects_Spatial_Relationships":
        payload["objects"] = [
            {
                **obj,
                "bbox": normalize_corner_box_from_pixels(
                    obj["bbox"],
                    frame_width=frame_width,
                    frame_height=frame_height,
                ),
            }
            for obj in payload["objects"]
        ]
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
            frame_width=frame_width,
            frame_height=frame_height,
        )
        payload.pop("tracking_original", None)
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
            frame_width=frame_width,
            frame_height=frame_height,
            valid_interval=(start_frame, end_frame),
        )
        payload.pop("tracking_original", None)
    return payload


def _sampled_video_relative_path() -> str:
    return str(Path("sampled_video.mp4"))


def _oracle_visual_frame_relative_path(sampled_index: int) -> str:
    return str(Path("oracle_visual") / "frames" / f"{sampled_index:04d}.jpg")


def _oracle_visual_video_relative_path() -> str:
    return str(Path("oracle_visual") / "sampled_video.mp4")


def _normalize_media_formats(media_formats: list[str] | None) -> set[str]:
    values = media_formats or ["frames"]
    allowed_values = {"frames", "sampled_video"}
    normalized_values: set[str] = set()
    for value in values:
        if value not in allowed_values:
            raise ValueError(f"unsupported media format: {value}")
        normalized_values.add(value)
    if not normalized_values:
        raise ValueError("at least one media format must be requested")
    return normalized_values


def _sampled_video_fps_for_frames(sampled_frames_original: list[int]) -> float | None:
    if len(sampled_frames_original) <= 1:
        return None
    first_frame = sampled_frames_original[0]
    last_frame = sampled_frames_original[-1]
    interval = max(last_frame - first_frame, 1)
    return ((len(sampled_frames_original) - 1) * VIDEO_FPS) / interval


def _require_pyav() -> Any:
    if av is None:
        raise RuntimeError(
            "prepared-data video decoding/encoding requires PyAV, but the `av` package "
            "is not installed"
        )
    return av


def _ensure_libx264_encoder(pyav_module: Any) -> None:
    try:
        pyav_module.CodecContext.create("libx264", "w")
    except Exception as exc:  # pragma: no cover - depends on local ffmpeg build
        raise RuntimeError(
            "sampled_video output requires the libx264 encoder, but it is not available "
            "in the current PyAV/FFmpeg build"
        ) from exc


def _write_sampled_video(
    output_path: Path,
    frame_images: list[Image.Image],
    *,
    sampled_video_fps: float,
) -> None:
    pyav_module = _require_pyav()
    _ensure_libx264_encoder(pyav_module)
    if not frame_images:
        raise ValueError("cannot write sampled video without frames")
    width, height = frame_images[0].size
    rate = Fraction(str(sampled_video_fps)).limit_denominator(1000)
    try:
        with pyav_module.open(str(output_path), mode="w") as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {
                "crf": "23",
                "preset": "medium",
            }
            for image in frame_images:
                if image.size != (width, height):
                    raise ValueError(
                        f"inconsistent sampled frame sizes for video encode: "
                        f"expected {(width, height)}, got {image.size}"
                    )
                video_frame = pyav_module.VideoFrame.from_image(image)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        raise RuntimeError(f"failed to encode sampled video at {output_path}") from exc


def _tracking_rows_by_sampled_frame(
    tracking_rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    rows_by_sampled_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracking_rows:
        rows_by_sampled_frame[int(row["frame_sampled"])].append(row)
    return rows_by_sampled_frame


def _draw_oracle_tracking_overlay(
    image: Image.Image,
    tracking_rows: list[dict[str, Any]],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for row in tracking_rows:
        x1, y1, x2, y2 = denormalize_mot_box_to_pixel_corners(
            row["bbox_mot"],
            frame_width=overlay.width,
            frame_height=overlay.height,
        )
        if x2 <= x1:
            x2 = min(overlay.width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(overlay.height - 1, y1 + 1)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label_top = max(0, y1 - 16)
        label_right = min(overlay.width - 1, x1 + 58)
        label_bottom = min(overlay.height - 1, label_top + 14)
        draw.rectangle(
            [x1, label_top, label_right, label_bottom],
            fill=(0, 255, 0),
        )
        draw.text((x1 + 3, label_top + 1), "TARGET", fill=(0, 0, 0))
    return overlay


def build_prepared_sample(
    sample: SampleRecord,
    protocol: BaseProtocol,
    sampled_frames_original: list[int],
    bundle_dir: Path,
    decoded_frames: dict[int, Image.Image],
    *,
    media_formats: set[str],
    generate_oracle_visual_media: bool,
) -> PreparedSample:
    reference_payload = _prepare_reference_payload(sample, sampled_frames_original)
    frame_files: list[str] = []
    if "frames" in media_formats:
        frames_dir = ensure_directory(bundle_dir / "frames")
        for sampled_index, frame_original in enumerate(sampled_frames_original):
            output_name = f"{sampled_index:04d}.jpg"
            output_path = frames_dir / output_name
            decoded_frames[frame_original].save(output_path, format="JPEG", quality=95)
            frame_files.append(str(Path("frames") / output_name))
    sampled_video_fps = _sampled_video_fps_for_frames(sampled_frames_original)
    sampled_video_file: str | None = None
    if "sampled_video" in media_formats and sampled_video_fps is not None:
        sampled_video_path = bundle_dir / _sampled_video_relative_path()
        sampled_images = [decoded_frames[index] for index in sampled_frames_original]
        _write_sampled_video(
            sampled_video_path,
            sampled_images,
            sampled_video_fps=sampled_video_fps,
        )
        sampled_video_file = _sampled_video_relative_path()

    oracle_visual_frame_files: list[str] = []
    oracle_visual_sampled_video_file: str | None = None
    if generate_oracle_visual_media and sample.task_name in STG_UPSTREAM_TASKS:
        tracking_gt = reference_payload.get("tracking_gt_sampled")
        if not isinstance(tracking_gt, list) or not tracking_gt:
            raise ValueError(f"{sample.sample_id}: missing tracking_gt_sampled for oracle visual media")
        tracking_by_sampled_frame = _tracking_rows_by_sampled_frame(tracking_gt)
        oracle_visual_frames_dir = ensure_directory(bundle_dir / "oracle_visual" / "frames")
        overlay_images: list[Image.Image] = []
        for sampled_index, frame_original in enumerate(sampled_frames_original):
            overlay_image = _draw_oracle_tracking_overlay(
                decoded_frames[frame_original],
                tracking_by_sampled_frame.get(sampled_index, []),
            )
            overlay_output_path = oracle_visual_frames_dir / f"{sampled_index:04d}.jpg"
            overlay_image.save(overlay_output_path, format="JPEG", quality=95)
            oracle_visual_frame_files.append(_oracle_visual_frame_relative_path(sampled_index))
            overlay_images.append(overlay_image)
        if sampled_video_fps is None:
            raise ValueError(
                f"{sample.sample_id}: oracle visual media requires multi-frame sampled video support"
            )
        oracle_visual_video_path = bundle_dir / _oracle_visual_video_relative_path()
        _write_sampled_video(
            oracle_visual_video_path,
            overlay_images,
            sampled_video_fps=sampled_video_fps,
        )
        oracle_visual_sampled_video_file = _oracle_visual_video_relative_path()
    if not frame_files and sampled_video_file is None:
        if sample.task_name in SINGLE_FRAME_TASKS:
            raise ValueError(
                f"{sample.sample_id}: no prepared media produced; single-frame tasks require "
                "`frames` in [prepare_data].media_formats"
            )
        raise ValueError(f"{sample.sample_id}: no prepared media produced")
    q_window_sampled = protocol.project_question_window(sample, sampled_frames_original)
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
        reference_payload=reference_payload,
        timestamp_frame=sample.timestamp_frame,
        q_window=sample.q_window,
        q_window_sampled=q_window_sampled,
        a_window=sample.a_window,
        source_tracking_path=(
            str(sample.source_tracking_path) if sample.source_tracking_path is not None else None
        ),
        upstream_annotation_id=sample.upstream_annotation_id,
        sampled_video_file=sampled_video_file,
        sampled_video_fps=sampled_video_fps,
        oracle_visual_frame_files=oracle_visual_frame_files,
        oracle_visual_sampled_video_file=oracle_visual_sampled_video_file,
        metadata={
            "num_sampled_frames": len(sampled_frames_original),
            "video_total_frames": sample.video_metadata.total_frames,
            "frame_width": sample.video_metadata.resolution[0],
            "frame_height": sample.video_metadata.resolution[1],
            "coordinate_system": COORDINATE_SYSTEM_NORMALIZED_1000,
            "media_formats": sorted(media_formats),
            "generate_oracle_visual_media": generate_oracle_visual_media,
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
    protocol: BaseProtocol,
    samples_root: Path,
    *,
    media_formats: set[str],
    generate_oracle_visual_media: bool,
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
            media_formats=media_formats,
            generate_oracle_visual_media=generate_oracle_visual_media,
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
    protocol: BaseProtocol,
    protocol_spec: str,
    prepared_root: Path,
    *,
    data_status: dict[str, Any],
    media_formats: list[str],
    generate_oracle_visual_media: bool,
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("prepare-data workers must be >= 1")
    protocol_root = ensure_directory(prepared_root / protocol.protocol_id)
    samples_root = ensure_directory(protocol_root / "samples")
    requested_media_formats = _normalize_media_formats(media_formats)
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
                media_formats=requested_media_formats,
                generate_oracle_visual_media=generate_oracle_visual_media,
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
            "protocol_id": protocol.protocol_id,
            "protocol_spec": protocol_spec,
            "protocol_manifest": protocol.to_manifest_dict(),
            "data_status": data_status,
            "supported_dataset_fingerprint": dataset_fingerprint(records),
            "num_prepared_samples": len(prepared_samples),
            "coordinate_system": COORDINATE_SYSTEM_NORMALIZED_1000,
            "media_formats": sorted(requested_media_formats),
            "generate_oracle_visual_media": generate_oracle_visual_media,
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
    protocol_specs: list[str],
    *,
    media_formats: list[str] | None = None,
    generate_oracle_visual_media: bool = False,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("prepare-data workers must be >= 1")
    requested_media_formats = sorted(_normalize_media_formats(media_formats))
    if generate_oracle_visual_media and (
        "frames" not in requested_media_formats or "sampled_video" not in requested_media_formats
    ):
        raise ValueError(
            "oracle visual media generation requires media_formats to include both frames "
            "and sampled_video"
        )
    scan_report = scan_dataset_report(data_root)
    if scan_report.issues:
        issue_text = "\n".join(scan_report.issues[:50])
        raise ValueError(
            f"dataset validation failed with {len(scan_report.issues)} issue(s):\n{issue_text}"
        )
    records = scan_report.supported_records
    data_status = summarize_scan_report(scan_report)
    results = []
    seen_protocol_ids: dict[str, str] = {}
    for protocol_spec in protocol_specs:
        protocol = resolve_protocol(protocol_spec)
        existing_spec = seen_protocol_ids.get(protocol.protocol_id)
        if existing_spec is not None and existing_spec != protocol_spec:
            raise ValueError(
                f"prepare-data protocol specs {existing_spec!r} and {protocol_spec!r} "
                f"both resolve to protocol_id {protocol.protocol_id!r}"
            )
        seen_protocol_ids[protocol.protocol_id] = protocol_spec
        results.append(
            build_protocol_cache(
                records,
                protocol,
                protocol_spec,
                prepared_root,
                data_status=data_status,
                media_formats=requested_media_formats,
                generate_oracle_visual_media=generate_oracle_visual_media,
                workers=workers,
            )
        )
    return results


def load_prepared_samples(prepared_root: Path, protocol_spec: str) -> list[PreparedSample]:
    protocol = resolve_protocol(protocol_spec)
    expected_manifest = protocol.to_manifest_dict()
    protocol_id = protocol.protocol_id
    protocol_root = prepared_root / protocol_id
    build_manifest = read_json(protocol_root / "build_manifest.json")
    if build_manifest.get("protocol_id") != protocol_id:
        raise ValueError(
            f"{protocol_root}: prepared protocol_id {build_manifest.get('protocol_id')!r} "
            f"does not match requested {protocol_id!r}; rebuild prepared data"
        )
    if build_manifest.get("protocol_spec") != protocol_spec:
        raise ValueError(
            f"{protocol_root}: prepared protocol_spec {build_manifest.get('protocol_spec')!r} "
            f"does not match requested {protocol_spec!r}; rebuild prepared data"
        )
    if build_manifest.get("protocol_manifest") != expected_manifest:
        raise ValueError(
            f"{protocol_root}: prepared protocol manifest does not match the current "
            f"definition for {protocol_spec!r}; rebuild prepared data"
        )
    if build_manifest.get("coordinate_system") != COORDINATE_SYSTEM_NORMALIZED_1000:
        raise ValueError(
            f"{protocol_root}: unsupported prepared coordinate_system "
            f"{build_manifest.get('coordinate_system')!r}; rebuild prepared data"
        )
    index_rows = read_jsonl(protocol_root / "index.jsonl")
    prepared: list[PreparedSample] = []
    for row in index_rows:
        manifest_path = protocol_root / row["manifest_path"]
        sample = PreparedSample.from_dict(read_json(manifest_path))
        sample.metadata = dict(sample.metadata)
        if sample.metadata.get("coordinate_system") != COORDINATE_SYSTEM_NORMALIZED_1000:
            raise ValueError(
                f"{manifest_path}: unsupported sample coordinate_system "
                f"{sample.metadata.get('coordinate_system')!r}; rebuild prepared data"
            )
        sample.frame_files = [
            str((manifest_path.parent / relative_path).resolve()) for relative_path in sample.frame_files
        ]
        if len(sample.sampled_frames_original) > 1 and sample.sampled_video_fps is None:
            raise ValueError(f"{manifest_path}: missing sampled_video_fps; rebuild prepared data")
        if sample.sampled_video_file is not None:
            sample.sampled_video_file = str((manifest_path.parent / sample.sampled_video_file).resolve())
        sample.oracle_visual_frame_files = [
            str((manifest_path.parent / relative_path).resolve())
            for relative_path in sample.oracle_visual_frame_files
        ]
        if sample.oracle_visual_sampled_video_file is not None:
            sample.oracle_visual_sampled_video_file = str(
                (manifest_path.parent / sample.oracle_visual_sampled_video_file).resolve()
            )
        sample.metadata["bundle_dir"] = str(manifest_path.parent.resolve())
        sample.metadata["protocol_spec"] = protocol_spec
        sample.metadata["protocol_manifest"] = expected_manifest
        prepared.append(sample)
    return prepared
