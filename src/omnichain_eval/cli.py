"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters import resolve_adapter
from .dataset import load_dataset, scan_dataset, summarize_records
from .experiments import (
    build_chain_manifest,
    evaluate_oracle_chain,
    evaluate_prepared_predictions,
    load_chain_pairs,
    load_prepared_for_protocol,
    summarize_experiment_b,
)
from .judge import OpenAIJudgeClient, StaticJudgeClient
from .prepare import build_prepared_data
from .utils import ensure_directory, read_jsonl, write_json, write_jsonl


def _judge_client_from_args(args: argparse.Namespace):
    if args.judge_backend == "static-pass":
        return StaticJudgeClient(always_pass=True)
    if args.judge_backend == "static-fail":
        return StaticJudgeClient(always_pass=False)
    return OpenAIJudgeClient.from_env(cache_dir=Path(args.artifacts_root) / "judge_cache")


def _load_prediction_map(path: Path) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for row in read_jsonl(path):
        payload = row.get("normalized_prediction", row.get("raw_output"))
        mapping[row["sample_id"]] = payload
    return mapping


def cmd_validate_data(args: argparse.Namespace) -> int:
    records, issues = scan_dataset(Path(args.data_root))
    summary = summarize_records(records)
    print(f"Validated {summary['num_samples']} samples across {summary['num_videos']} videos.")
    for task_name, count in summary["task_counts"].items():
        print(f"  {task_name}: {count}")
    if issues:
        print(f"Found {len(issues)} issue(s).")
        for issue in issues[:50]:
            print(f"  - {issue}")
        return 1
    print("No validation issues found.")
    return 0


def cmd_build_chain_manifest(args: argparse.Namespace) -> int:
    pairs = build_chain_manifest(Path(args.data_root), Path(args.out))
    print(f"Wrote {len(pairs)} chain pairs to {args.out}")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    results = build_prepared_data(
        Path(args.data_root),
        Path(args.prepared_root),
        args.protocol,
    )
    for result in results:
        print(
            f"Prepared {result['num_prepared_samples']} samples for {result['protocol_id']} "
            f"at {result['protocol_root']}"
        )
        if result["dataset_issue_count"]:
            print(
                f"  warnings: skipped {result['dataset_issue_count']} invalid annotation issue(s); "
                "see build_manifest.json for details"
            )
    return 0


def _default_run_name(model_name: str, protocol_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{model_name}_{protocol_id}"


def cmd_run_eval(args: argparse.Namespace) -> int:
    prepared_root = Path(args.prepared_root)
    prepared_samples = load_prepared_for_protocol(prepared_root, args.protocol)
    prepared_by_sample_id = {sample.sample_id: sample for sample in prepared_samples}

    prediction_map: dict[str, Any]
    commentary_supported = not args.commentary_unsupported
    adapter = None
    if args.adapter:
        adapter = resolve_adapter(args.adapter)
        commentary_supported = adapter.supports_commentary()
        prediction_map = {}
        for sample in prepared_samples:
            if sample.task_name == "Commentary" and not commentary_supported:
                continue
            prediction_map[sample.sample_id] = adapter.predict(sample)
        model_name = args.model_name or adapter.name
    else:
        prediction_map = _load_prediction_map(Path(args.predictions))
        model_name = args.model_name or Path(args.predictions).stem

    artifacts_root = ensure_directory(Path(args.artifacts_root))
    run_dir = ensure_directory(artifacts_root / (args.run_name or _default_run_name(model_name, args.protocol)))
    judge_client = _judge_client_from_args(args)

    evaluation = evaluate_prepared_predictions(
        prepared_samples,
        prediction_map,
        model_name=model_name,
        judge_client=judge_client,
        commentary_supported=commentary_supported,
        enable_bertscore=args.enable_bertscore,
    )

    oracle_summary = None
    if args.chain_manifest:
        chain_pairs = load_chain_pairs(Path(args.chain_manifest))
        oracle_pair_results = None
        if adapter is not None and args.enable_oracle_track:
            if not adapter.supports_oracle_track():
                raise ValueError(f"adapter {adapter.name} does not support oracle_track reruns")
            oracle_pair_results = evaluate_oracle_chain(
                adapter,
                prepared_by_sample_id,
                chain_pairs,
                judge_client=judge_client,
            )
        elif args.oracle_predictions:
            oracle_prediction_map = _load_prediction_map(Path(args.oracle_predictions))
            oracle_records_by_sample_id = {}
            for pair in chain_pairs:
                upstream_sample = prepared_by_sample_id[pair.upstream_sample_id]
                downstream_sample = prepared_by_sample_id[pair.downstream_sample_id]
                if upstream_sample.sample_id not in oracle_records_by_sample_id:
                    from .metrics import evaluate_sample

                    oracle_records_by_sample_id[upstream_sample.sample_id] = evaluate_sample(
                        upstream_sample,
                        oracle_prediction_map.get(upstream_sample.sample_id),
                        judge_client=judge_client,
                        override_tracking_with_gt=upstream_sample.task_name
                        in {"Continuous_Actions_Caption", "Spatial_Temporal_Grounding"},
                    )
                if downstream_sample.sample_id not in oracle_records_by_sample_id:
                    from .metrics import evaluate_sample

                    oracle_records_by_sample_id[downstream_sample.sample_id] = evaluate_sample(
                        downstream_sample,
                        oracle_prediction_map.get(downstream_sample.sample_id),
                        judge_client=judge_client,
                    )
            oracle_pair_results = {
                pair.pair_id: {
                    "upstream": oracle_records_by_sample_id[pair.upstream_sample_id],
                    "downstream": oracle_records_by_sample_id[pair.downstream_sample_id],
                }
                for pair in chain_pairs
            }

        oracle_summary = summarize_experiment_b(
            chain_pairs,
            evaluation["records_by_sample_id"],
            oracle_pair_results=oracle_pair_results,
        )

    prediction_rows = [
        {"sample_id": sample_id, "raw_output": raw_output, "protocol_id": args.protocol}
        for sample_id, raw_output in sorted(prediction_map.items())
    ]
    write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(run_dir / "sample_results.jsonl", [record.to_dict() for record in evaluation["records"]])
    write_json(run_dir / "task_summaries.json", [summary.to_dict() for summary in evaluation["task_summaries"]])
    summary_payload = {
        "model_name": model_name,
        "protocol_id": args.protocol,
        "overall": evaluation["overall"],
        "task_summaries": [summary.to_dict() for summary in evaluation["task_summaries"]],
        "experiment_b": oracle_summary,
        "commentary_supported": commentary_supported,
    }
    write_json(run_dir / "summary.json", summary_payload)
    print(f"Wrote evaluation artifacts to {run_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    artifacts_root = Path(args.artifacts_root)
    summary_paths = sorted(artifacts_root.rglob("summary.json"))
    summaries = []
    for summary_path in summary_paths:
        summaries.append({"path": str(summary_path), **json.loads(summary_path.read_text())})
    output_path = Path(args.out) if args.out else artifacts_root / "report.json"
    write_json(output_path, {"runs": summaries})
    print(f"Wrote consolidated report with {len(summaries)} run(s) to {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnichain-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data")
    validate_parser.add_argument("--data-root", default="data")
    validate_parser.set_defaults(func=cmd_validate_data)

    chain_parser = subparsers.add_parser("build-chain-manifest")
    chain_parser.add_argument("--data-root", default="data")
    chain_parser.add_argument("--out", required=True)
    chain_parser.set_defaults(func=cmd_build_chain_manifest)

    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--data-root", default="data")
    prepare_parser.add_argument("--prepared-root", default="prepared_data")
    prepare_parser.add_argument("--protocol", action="append", required=True)
    prepare_parser.set_defaults(func=cmd_prepare_data)

    run_parser = subparsers.add_parser("run-eval")
    run_source = run_parser.add_mutually_exclusive_group(required=True)
    run_source.add_argument("--adapter")
    run_source.add_argument("--predictions")
    run_parser.add_argument("--oracle-predictions")
    run_parser.add_argument("--prepared-root", default="prepared_data")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--artifacts-root", default="artifacts/runs")
    run_parser.add_argument("--run-name")
    run_parser.add_argument("--model-name")
    run_parser.add_argument("--chain-manifest")
    run_parser.add_argument("--enable-oracle-track", action="store_true")
    run_parser.add_argument("--enable-bertscore", action="store_true")
    run_parser.add_argument(
        "--judge-backend",
        choices=("openai", "static-pass", "static-fail"),
        default="openai",
    )
    run_parser.add_argument("--commentary-unsupported", action="store_true")
    run_parser.set_defaults(func=cmd_run_eval)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--artifacts-root", default="artifacts/runs")
    report_parser.add_argument("--out")
    report_parser.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
