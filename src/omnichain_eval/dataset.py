"""Raw dataset loading and validation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    ALL_TASKS,
    SEGMENT_TASKS,
    SINGLE_FRAME_TASKS,
    STG_UPSTREAM_TASKS,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_SINGLE,
    TASK_SPATIAL_IMAGINATION,
    TASK_STG,
    TEXT_ONLY_TASKS,
    VIDEO_FPS,
)
from .schema import SampleRecord, VideoMetadata
from .utils import canonical_interval, parse_interval_string, read_json, stable_hash


@dataclass(slots=True)
class UnsupportedSample:
    sample_id: str
    annotation_id: str
    video_key: str
    task_name: str
    source_annotation_path: Path


@dataclass(slots=True)
class DatasetScanReport:
    supported_records: list[SampleRecord]
    unsupported_samples: list[UnsupportedSample]
    issues: list[str]


def iter_annotation_files(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.rglob("*.json") if path.with_suffix(".mp4").exists())


def build_video_key(annotation_path: Path, data_root: Path) -> str:
    relative = annotation_path.relative_to(data_root)
    return str(relative.parent / annotation_path.stem)


def build_sample_id(annotation_path: Path, annotation_id: str, data_root: Path) -> str:
    return f"{build_video_key(annotation_path, data_root)}#{annotation_id}"


def resolve_dataset_path(data_root: Path, raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    path = Path(raw_value)
    if path.is_absolute():
        return path
    text = raw_value.strip()
    candidates = [text]
    for prefix in ("./data/", "data/", "./dataset/", "dataset/"):
        if text.startswith(prefix):
            candidates.append(text[len(prefix) :])
    candidates.append(text.lstrip("./"))
    for candidate in candidates:
        candidate_path = data_root / candidate
        if candidate_path.exists():
            return candidate_path
    return data_root / candidates[-1]


def load_tracking_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) < 6:
                raise ValueError(f"tracking row has fewer than 6 columns: {line}")
            frame_index = int(float(pieces[0]))
            bbox = [float(pieces[2]), float(pieces[3]), float(pieces[4]), float(pieces[5])]
            rows.append({"frame_original": frame_index, "bbox_mot": bbox})
    return rows


def _normalize_objects_boxes(raw_boxes: list[dict[str, Any]]) -> dict[str, Any]:
    if len(raw_boxes) != 2:
        raise ValueError("Objects_Spatial_Relationships expects exactly two boxes")
    objects: list[dict[str, Any]] = []
    for index, raw_box in enumerate(raw_boxes):
        label = str(raw_box.get("label", "")).strip()
        if not label:
            raise ValueError(
                f"Objects_Spatial_Relationships box {index} is missing a non-empty label"
            )
        objects.append(
            {
                "label": label,
                "bbox": [float(value) for value in raw_box["box"]],
            }
        )
    return {
        "text": "",
        "objects": objects,
    }


def _build_segments_payload(
    intervals: list[tuple[int, int]], texts: list[str]
) -> dict[str, Any]:
    if len(intervals) != len(texts):
        raise ValueError("segment interval count does not match answer count")
    segments = []
    for interval, text in zip(intervals, texts, strict=True):
        start, end = canonical_interval(interval)
        segments.append({"start_frame": start, "end_frame": end, "text": text})
    return {
        "segments_original": segments,
        "linearized_text": " ".join(texts),
    }


def build_reference_payload(
    task_name: str,
    annotation: dict[str, Any],
    tracking_path: Path | None,
) -> dict[str, Any]:
    if task_name in TEXT_ONLY_TASKS:
        return {"text": annotation["answer"]}
    if task_name == TASK_SCOREBOARD_SINGLE:
        return {
            "text": annotation["answer"],
            "bbox": [float(value) for value in annotation["bounding_box"]],
        }
    if task_name == TASK_OBJECTS_SPATIAL:
        payload = _normalize_objects_boxes(annotation["bounding_box"])
        payload["text"] = annotation["answer"]
        return payload
    if task_name == TASK_CONTINUOUS_EVENTS:
        intervals = [parse_interval_string(value) for value in annotation["A_window_frame"]]
        return _build_segments_payload(intervals, annotation["answer"])
    if task_name == TASK_CONTINUOUS_ACTIONS:
        intervals = [parse_interval_string(value) for value in annotation["A_window_frame"]]
        payload = _build_segments_payload(intervals, annotation["answer"])
        payload["tracking_original"] = load_tracking_rows(tracking_path)
        return payload
    if task_name == TASK_STG:
        a_window = canonical_interval(annotation["A_window_frame"])
        return {
            "time_window_original": list(a_window),
            "tracking_original": load_tracking_rows(tracking_path),
        }
    raise ValueError(f"unsupported task for GT building: {task_name}")


def _question_text(annotation: dict[str, Any]) -> str:
    if "question" in annotation:
        return str(annotation["question"])
    if "query" in annotation:
        return str(annotation["query"])
    return ""


def _parse_video_metadata(payload: dict[str, Any]) -> VideoMetadata:
    metadata = payload["video_metadata"]
    resolution = metadata.get("resolution", [0, 0])
    return VideoMetadata(
        duration_sec=float(metadata["duration_sec"]),
        fps=int(metadata["fps"]),
        total_frames=int(metadata["total_frames"]),
        resolution=(int(resolution[0]), int(resolution[1])),
    )


def load_annotation_file(
    annotation_path: Path,
    data_root: Path,
) -> tuple[list[SampleRecord], list[UnsupportedSample], list[str]]:
    payload = read_json(annotation_path)
    video_metadata = _parse_video_metadata(payload)
    video_path = annotation_path.with_suffix(".mp4")
    records: list[SampleRecord] = []
    unsupported_samples: list[UnsupportedSample] = []
    issues: list[str] = []
    for annotation in payload.get("annotations", []):
        annotation_id = str(annotation.get("annotation_id", "<missing>"))
        try:
            task_name = annotation["task_L2"]
            sample_id = build_sample_id(annotation_path, annotation_id, data_root)
            video_key = build_video_key(annotation_path, data_root)
            if task_name not in ALL_TASKS:
                unsupported_samples.append(
                    UnsupportedSample(
                        sample_id=sample_id,
                        annotation_id=annotation_id,
                        video_key=video_key,
                        task_name=task_name,
                        source_annotation_path=annotation_path,
                    )
                )
                continue
            q_window = (
                canonical_interval(annotation["Q_window_frame"])
                if "Q_window_frame" in annotation
                else None
            )
            a_window = None
            if task_name == TASK_STG:
                a_window = canonical_interval(annotation["A_window_frame"])
            tracking_path = resolve_dataset_path(data_root, annotation.get("tracking_bboxes"))
            reference_payload = build_reference_payload(
                task_name,
                annotation,
                tracking_path,
            )
            record = SampleRecord(
                sample_id=sample_id,
                annotation_id=annotation_id,
                video_key=video_key,
                task_name=task_name,
                task_level=str(annotation["task_L1"]),
                question_text=_question_text(annotation),
                source_annotation_path=annotation_path,
                source_video_path=video_path,
                video_metadata=video_metadata,
                raw_annotation=annotation,
                reference_payload=reference_payload,
                timestamp_frame=int(annotation["timestamp_frame"])
                if "timestamp_frame" in annotation
                else None,
                q_window=q_window,
                a_window=a_window,
                source_tracking_path=tracking_path,
                upstream_annotation_id=(
                    str(annotation["upstream_annotation_id"])
                    if annotation.get("upstream_annotation_id") is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                f"{annotation_path}#{annotation_id}: failed to load annotation: {exc}"
            )
            continue
        records.append(record)
    return records, unsupported_samples, issues


def _validate_sample_structure(sample: SampleRecord) -> list[str]:
    issues: list[str] = []
    if sample.task_name not in ALL_TASKS:
        issues.append(f"{sample.sample_id}: unknown task {sample.task_name}")
    if sample.video_metadata.fps != VIDEO_FPS:
        issues.append(
            f"{sample.sample_id}: expected fps {VIDEO_FPS}, got {sample.video_metadata.fps}"
        )
    if not sample.source_video_path.exists():
        issues.append(f"{sample.sample_id}: missing video {sample.source_video_path}")
    if sample.task_name in SINGLE_FRAME_TASKS and sample.timestamp_frame is None:
        issues.append(f"{sample.sample_id}: missing timestamp_frame")
    if sample.task_name not in SINGLE_FRAME_TASKS and sample.task_name != TASK_STG:
        if sample.q_window is None:
            issues.append(f"{sample.sample_id}: missing Q_window_frame")
    if sample.task_name == TASK_STG and sample.a_window is None:
        issues.append(f"{sample.sample_id}: missing A_window_frame")
    if sample.task_name in SEGMENT_TASKS | {TASK_STG}:
        if sample.source_tracking_path and not sample.source_tracking_path.exists():
            issues.append(
                f"{sample.sample_id}: missing tracking file {sample.source_tracking_path}"
            )
    return issues


def scan_dataset_report(data_root: Path) -> DatasetScanReport:
    records: list[SampleRecord] = []
    unsupported_samples: list[UnsupportedSample] = []
    issues: list[str] = []
    for annotation_path in iter_annotation_files(data_root):
        try:
            file_records, file_unsupported, file_issues = load_annotation_file(annotation_path, data_root)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{annotation_path}: failed to load annotation file: {exc}")
            continue
        issues.extend(file_issues)
        unsupported_samples.extend(file_unsupported)
        for record in file_records:
            records.append(record)
            issues.extend(_validate_sample_structure(record))
    samples_by_file: dict[Path, dict[str, SampleRecord]] = defaultdict(dict)
    for record in records:
        samples_by_file[record.source_annotation_path][record.annotation_id] = record
    for record in records:
        if record.task_name != TASK_SPATIAL_IMAGINATION:
            continue
        if record.upstream_annotation_id is None:
            issues.append(f"{record.sample_id}: missing upstream_annotation_id")
            continue
        target = samples_by_file[record.source_annotation_path].get(record.upstream_annotation_id)
        if target is None:
            issues.append(
                f"{record.sample_id}: upstream annotation {record.upstream_annotation_id} not found"
            )
            continue
        if target.task_name not in STG_UPSTREAM_TASKS:
            issues.append(
                f"{record.sample_id}: upstream task {target.task_name} is not allowed"
            )
    return DatasetScanReport(
        supported_records=sorted(records, key=lambda item: item.sample_id),
        unsupported_samples=sorted(unsupported_samples, key=lambda item: item.sample_id),
        issues=issues,
    )


def scan_dataset(data_root: Path) -> tuple[list[SampleRecord], list[str]]:
    report = scan_dataset_report(data_root)
    return report.supported_records, report.issues


def load_dataset(data_root: Path, *, strict: bool = True) -> list[SampleRecord]:
    report = scan_dataset_report(data_root)
    if strict and report.issues:
        issue_text = "\n".join(report.issues[:50])
        raise ValueError(
            f"dataset validation failed with {len(report.issues)} issue(s):\n{issue_text}"
        )
    return report.supported_records


def dataset_fingerprint(records: list[SampleRecord]) -> str:
    material = [
        {
            "sample_id": record.sample_id,
            "annotation": str(record.source_annotation_path),
            "video": str(record.source_video_path),
            "tracking": str(record.source_tracking_path) if record.source_tracking_path else None,
            "task": record.task_name,
        }
        for record in records
    ]
    return stable_hash(material)


def summarize_records(records: list[SampleRecord]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.task_name] += 1
    return {
        "num_samples": len(records),
        "num_videos": len({record.video_key for record in records}),
        "task_counts": dict(sorted(counts.items())),
        "overall_tasks": sorted(ALL_TASKS),
    }


def summarize_scan_report(report: DatasetScanReport) -> dict[str, Any]:
    raw_task_counts: dict[str, int] = defaultdict(int)
    raw_video_keys = {record.video_key for record in report.supported_records}
    for record in report.supported_records:
        raw_task_counts[record.task_name] += 1
    unsupported_task_counts: dict[str, int] = defaultdict(int)
    for sample in report.unsupported_samples:
        raw_task_counts[sample.task_name] += 1
        unsupported_task_counts[sample.task_name] += 1
        raw_video_keys.add(sample.video_key)
    return {
        "raw_dataset_summary": {
            "num_samples": len(report.supported_records) + len(report.unsupported_samples),
            "num_videos": len(raw_video_keys),
            "task_counts": dict(sorted(raw_task_counts.items())),
            "overall_tasks": sorted(raw_task_counts),
        },
        "supported_dataset_summary": summarize_records(report.supported_records),
        "ignored_unsupported_sample_count": len(report.unsupported_samples),
        "ignored_unsupported_task_counts": dict(sorted(unsupported_task_counts.items())),
        "supported_issue_count": len(report.issues),
        "supported_issues": report.issues[:100],
    }
