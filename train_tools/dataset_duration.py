#!/usr/bin/env python3
"""Report trajectory dataset duration from a FlowNav config.

The processed FlowNav trajectories usually store `position` and `yaw` in
`traj_data.pkl`. If a trajectory also contains timestamps, this script uses
them directly. Otherwise pass `--hz` or `--dataset-hz DATASET=HZ` so duration
can be computed from the number of trajectory steps.

Examples:
  python3 train_tools/dataset_duration.py --hz 4
  python3 train_tools/dataset_duration.py --dataset-hz recon=4 --dataset-hz scand=10
  python3 train_tools/dataset_duration.py --datasets recon sacson --splits train
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("flownav/config/flownav.yaml")
DEFAULT_OUTPUT = Path("test_tools_logs/datasets_statistics.txt")
TIME_KEYS = ("timestamp", "timestamps", "time", "times")


@dataclass
class SplitStats:
    dataset: str
    split: str
    listed_trajectories: int = 0
    readable_trajectories: int = 0
    missing_trajectories: int = 0
    total_steps: int = 0
    timestamp_seconds: float | None = None
    min_steps: int | None = None
    median_steps: float | None = None
    mean_steps: float | None = None
    max_steps: int | None = None

    def duration_seconds(self, hz: float | None) -> float | None:
        if self.timestamp_seconds is not None:
            return self.timestamp_seconds
        if hz:
            return self.total_steps / hz
        return None


def parse_dataset_hz(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"Expected DATASET=HZ for --dataset-hz, got {value!r}"
            )
        dataset, hz_text = value.split("=", 1)
        dataset = dataset.strip()
        if not dataset:
            raise argparse.ArgumentTypeError("--dataset-hz dataset name is empty")
        hz = float(hz_text)
        if hz <= 0:
            raise argparse.ArgumentTypeError("--dataset-hz value must be positive")
        result[dataset] = hz
    return result


def load_traj_names(split_folder: Path) -> list[str]:
    names_file = split_folder / "traj_names.txt"
    if not names_file.exists():
        raise FileNotFoundError(f"Missing split file: {names_file}")
    return [line.strip() for line in names_file.read_text().splitlines() if line.strip()]


def sequence_length(value: Any) -> int:
    try:
        return len(value)
    except TypeError as exc:
        raise ValueError("trajectory field is not sequence-like") from exc


def timestamp_duration_seconds(traj_data: dict[str, Any]) -> float | None:
    for key in TIME_KEYS:
        if key not in traj_data:
            continue
        times = traj_data[key]
        if sequence_length(times) < 2:
            return 0.0
        first = float(times[0])
        last = float(times[-1])
        duration = last - first
        if duration < 0:
            raise ValueError(f"timestamp field {key!r} is not monotonic")
        return duration
    return None


def trajectory_steps(traj_data: dict[str, Any]) -> int:
    if "position" in traj_data:
        return sequence_length(traj_data["position"])
    if "yaw" in traj_data:
        return sequence_length(traj_data["yaw"])
    raise ValueError("traj_data.pkl has neither `position` nor `yaw`")


def cached_index_path(
    split_folder: Path, data_cfg: dict[str, Any], config: dict[str, Any]
) -> Path:
    distance = config["distance"]
    context_type = config.get("context_type", "temporal")
    context_size = config["context_size"]
    end_slack = data_cfg.get("end_slack", 0)
    return split_folder / (
        f"dataset_dist_{distance['min_dist_cat']}_to_{distance['max_dist_cat']}"
        f"_context_{context_type}_n{context_size}_slack_{end_slack}.pkl"
    )


def summarize_from_cached_index(
    dataset: str,
    split: str,
    split_folder: Path,
    data_cfg: dict[str, Any],
    config: dict[str, Any],
) -> SplitStats | None:
    index_path = cached_index_path(split_folder, data_cfg, config)
    if not index_path.exists():
        return None

    with index_path.open("rb") as file:
        _, goals_index = pickle.load(file)

    traj_names = load_traj_names(split_folder)
    counts = Counter(traj_name for traj_name, _ in goals_index)
    lengths = [counts[name] for name in traj_names if counts[name] > 0]

    stats = SplitStats(dataset=dataset, split=split, listed_trajectories=len(traj_names))
    stats.readable_trajectories = len(lengths)
    stats.missing_trajectories = len(traj_names) - len(lengths)
    stats.total_steps = sum(lengths)
    if lengths:
        stats.min_steps = min(lengths)
        stats.median_steps = statistics.median(lengths)
        stats.mean_steps = statistics.mean(lengths)
        stats.max_steps = max(lengths)
    return stats


def summarize_split(
    dataset: str,
    split: str,
    data_cfg: dict[str, Any],
    config: dict[str, Any],
    prefer_cached_index: bool,
) -> SplitStats:
    split_folder = Path(data_cfg[split])
    data_folder = Path(data_cfg["data_folder"])
    if prefer_cached_index:
        cached_stats = summarize_from_cached_index(
            dataset, split, split_folder, data_cfg, config
        )
        if cached_stats is not None:
            return cached_stats

    traj_names = load_traj_names(split_folder)

    stats = SplitStats(dataset=dataset, split=split, listed_trajectories=len(traj_names))
    lengths: list[int] = []
    timestamp_seconds = 0.0
    all_have_timestamps = True

    for traj_name in traj_names:
        traj_path = data_folder / traj_name / "traj_data.pkl"
        if not traj_path.exists():
            stats.missing_trajectories += 1
            continue

        with traj_path.open("rb") as file:
            traj_data = pickle.load(file)

        steps = trajectory_steps(traj_data)
        lengths.append(steps)
        stats.total_steps += steps
        stats.readable_trajectories += 1

        duration = timestamp_duration_seconds(traj_data)
        if duration is None:
            all_have_timestamps = False
        else:
            timestamp_seconds += duration

    if lengths:
        stats.min_steps = min(lengths)
        stats.median_steps = statistics.median(lengths)
        stats.mean_steps = statistics.mean(lengths)
        stats.max_steps = max(lengths)
    if lengths and all_have_timestamps:
        stats.timestamp_seconds = timestamp_seconds

    return stats


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    hours = seconds / 3600.0
    return f"{hours:.3f} h"


def build_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "dataset",
        "split",
        "traj",
        "missing",
        "steps",
        "min",
        "median",
        "mean",
        "max",
        "hz",
        "duration",
    ]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count FlowNav trajectory steps and estimate dataset duration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to include. Defaults to all datasets in the config.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Splits to include.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        help="Default trajectory sampling rate. Used when no timestamps are stored.",
    )
    parser.add_argument(
        "--dataset-hz",
        action="append",
        default=[],
        metavar="DATASET=HZ",
        help="Override sampling rate for one dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the statistics text file.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print only; do not write the statistics file.",
    )
    parser.add_argument(
        "--no-cached-index",
        action="store_true",
        help="Ignore split index caches and read every traj_data.pkl directly.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.hz is not None and args.hz <= 0:
        raise SystemExit("--hz must be positive")

    dataset_hz = parse_dataset_hz(args.dataset_hz)
    with args.config.open("r") as file:
        config = yaml.safe_load(file)

    all_datasets = config["datasets"]
    selected_datasets = args.datasets or list(all_datasets.keys())
    rows: list[dict[str, str]] = []
    totals: dict[str, SplitStats] = {}
    timestamp_totals: dict[str, float] = {}
    timestamp_complete: dict[str, bool] = {}

    for dataset in selected_datasets:
        if dataset not in all_datasets:
            raise SystemExit(f"Dataset {dataset!r} not found in {args.config}")
        hz = dataset_hz.get(dataset, args.hz)
        timestamp_totals[dataset] = 0.0
        timestamp_complete[dataset] = True
        for split in args.splits:
            if split not in all_datasets[dataset]:
                continue
            stats = summarize_split(
                dataset,
                split,
                all_datasets[dataset],
                config,
                prefer_cached_index=not args.no_cached_index,
            )
            total = totals.setdefault(dataset, SplitStats(dataset=dataset, split="total"))
            total.listed_trajectories += stats.listed_trajectories
            total.readable_trajectories += stats.readable_trajectories
            total.missing_trajectories += stats.missing_trajectories
            total.total_steps += stats.total_steps
            if stats.timestamp_seconds is None:
                timestamp_complete[dataset] = False
            else:
                timestamp_totals[dataset] += stats.timestamp_seconds

            rows.append(stats_to_row(stats, hz))

    for dataset in selected_datasets:
        hz = dataset_hz.get(dataset, args.hz)
        if timestamp_complete[dataset] and totals[dataset].readable_trajectories:
            totals[dataset].timestamp_seconds = timestamp_totals[dataset]
        rows.append(stats_to_row(totals[dataset], hz))

    summary_lines = build_duration_summary(totals, selected_datasets, dataset_hz, args.hz)
    if selected_datasets:
        rows.append(grand_total_row(totals, selected_datasets, dataset_hz, args.hz))

    output_lines = [
        f"config: {args.config}",
        f"datasets: {', '.join(selected_datasets)}",
        f"splits: {', '.join(args.splits)}",
    ]
    if args.hz is not None:
        output_lines.append(f"default_hz: {args.hz}")
    if dataset_hz:
        overrides = ", ".join(f"{name}={hz}" for name, hz in sorted(dataset_hz.items()))
        output_lines.append(f"dataset_hz: {overrides}")
    output_lines.extend(["", *summary_lines, "", build_table(rows)])

    if args.hz is None and not dataset_hz:
        output_lines.extend(
            [
                "",
                "No Hz was provided. Duration is shown only for trajectories with timestamps.",
                "For current FlowNav-style `position/yaw` pkls, pass `--hz` or `--dataset-hz`.",
            ]
        )

    output_text = "\n".join(output_lines) + "\n"
    print(output_text, end="")
    if not args.no_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            args.output.unlink()
        temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temp_output.write_text(output_text)
        os.replace(temp_output, args.output)
        print(f"\nWrote statistics to {args.output}")


def stats_to_row(stats: SplitStats, hz: float | None) -> dict[str, str]:
    duration = stats.duration_seconds(hz)
    return {
        "dataset": stats.dataset,
        "split": stats.split,
        "traj": str(stats.readable_trajectories),
        "missing": str(stats.missing_trajectories),
        "steps": str(stats.total_steps),
        "min": "-" if stats.min_steps is None else str(stats.min_steps),
        "median": "-" if stats.median_steps is None else f"{stats.median_steps:.1f}",
        "mean": "-" if stats.mean_steps is None else f"{stats.mean_steps:.1f}",
        "max": "-" if stats.max_steps is None else str(stats.max_steps),
        "hz": "timestamp" if stats.timestamp_seconds is not None else (str(hz) if hz else "-"),
        "duration": format_duration(duration),
    }


def build_duration_summary(
    totals: dict[str, SplitStats],
    selected_datasets: list[str],
    dataset_hz: dict[str, float],
    default_hz: float | None,
) -> list[str]:
    lines = ["Duration summary:"]
    total_seconds = 0.0
    has_all_durations = True

    for dataset in selected_datasets:
        hz = dataset_hz.get(dataset, default_hz)
        duration = totals[dataset].duration_seconds(hz)
        if duration is None:
            has_all_durations = False
            duration_text = "-"
        else:
            total_seconds += duration
            duration_text = format_duration(duration)
        lines.append(f"  {dataset}: {duration_text}")

    lines.append(f"  ALL: {format_duration(total_seconds if has_all_durations else None)}")
    return lines


def grand_total_row(
    totals: dict[str, SplitStats],
    selected_datasets: list[str],
    dataset_hz: dict[str, float],
    default_hz: float | None,
) -> dict[str, str]:
    total_seconds = 0.0
    has_duration = True
    total_traj = 0
    total_missing = 0
    total_steps = 0

    for dataset in selected_datasets:
        stats = totals[dataset]
        total_traj += stats.readable_trajectories
        total_missing += stats.missing_trajectories
        total_steps += stats.total_steps
        hz = dataset_hz.get(dataset, default_hz)
        duration = stats.duration_seconds(hz)
        if duration is None:
            has_duration = False
        else:
            total_seconds += duration

    hz_label = "mixed" if dataset_hz else (str(default_hz) if default_hz else "-")
    return {
        "dataset": "ALL",
        "split": "total",
        "traj": str(total_traj),
        "missing": str(total_missing),
        "steps": str(total_steps),
        "min": "-",
        "median": "-",
        "mean": "-",
        "max": "-",
        "hz": hz_label,
        "duration": format_duration(total_seconds if has_duration else None),
    }


if __name__ == "__main__":
    main()
