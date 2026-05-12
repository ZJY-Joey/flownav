import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_CONFIG,
    build_dataset,
    build_matched_goal_set_for_index,
    build_model,
    get_device,
    imagenet_transform,
    load_config,
    log,
    resolve_checkpoint,
    run_dist_pred,
    timestamp_name,
    write_csv,
    write_json,
)


DIRECTIONS = ("left", "forward", "right")
PAIRS = (("left", "forward"), ("left", "right"), ("forward", "right"))
VARIANT_ORDER = ("flownav_baseline", "flownav_cluster")


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_corr(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2:
        return np.nan
    first_rank = rankdata(first)
    second_rank = rankdata(second)
    first_std = float(first_rank.std())
    second_std = float(second_rank.std())
    if first_std <= 1e-12 or second_std <= 1e-12:
        return np.nan
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def pairwise_rank_accuracy(pred_by_direction, true_by_direction):
    correct = 0
    total = 0
    available = set(pred_by_direction) & set(true_by_direction)
    for first, second in PAIRS:
        if first not in available or second not in available:
            continue
        true_delta = true_by_direction[first] - true_by_direction[second]
        if abs(true_delta) <= 1e-12:
            continue
        pred_delta = pred_by_direction[first] - pred_by_direction[second]
        total += 1
        if np.sign(pred_delta) == np.sign(true_delta):
            correct += 1
    return float(correct / total) if total else np.nan


def scalar_dist_pred(model, obs, goal, device):
    pred = run_dist_pred(model, obs, goal, device)
    return float(pred.detach().cpu().reshape(-1).mean().item())


def flow_metric_from_swap_row(swap_row, key):
    value = swap_row.get(key)
    return float(value) if isinstance(value, (int, float)) else np.nan


def flow_endpoint_pair_distance_from_swap_row(swap_row):
    value = flow_metric_from_swap_row(swap_row, "flow_endpoint_pair_distance")
    if np.isfinite(value):
        return value
    return flow_metric_from_swap_row(swap_row, "mean_endpoint_mean_distance")


def item_key(item):
    return (
        item.get("dataset_index"),
        item.get("trajectory"),
        item.get("curr_time"),
    )


def clear_existing_backfill_outputs(output_dir: Path):
    for pattern in (
        "dist_head_backfill_summary_*.json",
        "dist_head_backfill_items_*.csv",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()


def process_summary(summary_path: Path, args):
    with open(summary_path, "r") as f:
        swap_summary = json.load(f)
    if swap_summary.get("test") != "goal_swap_visualization":
        raise ValueError(f"Not a goal_swap_visualization summary: {summary_path}")
    if swap_summary.get("stage") != "all_samples":
        raise ValueError(f"Dist backfill expects all_samples summary: {summary_path}")

    config_path = args.config or swap_summary.get("config") or str(DEFAULT_CONFIG)
    config = load_config(config_path)
    checkpoint_path = resolve_checkpoint(
        config,
        args.checkpoint or swap_summary.get("checkpoint"),
    )
    device = get_device(args.device)
    dataset_name = args.dataset or swap_summary["dataset"]
    split = args.split or swap_summary.get("split", "test")

    log(f"Loading model for dist backfill: {checkpoint_path}")
    model = build_model(config, checkpoint_path, device)
    transform = imagenet_transform()
    dataset = build_dataset(config, dataset_name, split)

    swap_rows = swap_summary.get("metrics", [])
    if args.max_items is not None:
        swap_rows = swap_rows[: args.max_items]

    rows = []
    failures = []
    for row_idx, swap_row in enumerate(swap_rows):
        dataset_index = int(swap_row["dataset_index"])
        if row_idx % 25 == 0:
            log(
                f"Backfilling dist head {row_idx + 1}/{len(swap_rows)} "
                f"for {dataset_name}: dataset_index={dataset_index}"
            )
        try:
            directional_set = build_matched_goal_set_for_index(
                dataset=dataset,
                sample_index=dataset_index,
                transform=transform,
                device=device,
                angle_threshold_deg=float(swap_summary["angle_threshold_deg"]),
                direction_source=swap_summary.get("direction_source", "goal_pos"),
                min_goal_offset=swap_summary.get("min_goal_offset"),
                max_goal_offset=swap_summary.get("max_goal_offset"),
                min_goal_pos_dist=swap_summary.get("min_goal_pos_dist"),
                max_goal_pos_dist=swap_summary.get("max_goal_pos_dist"),
                max_direction_angle_deg=swap_summary.get("max_direction_angle_deg"),
                max_endpoint_goal_dist=swap_summary.get("max_endpoint_goal_dist"),
                filter_goal_heading=bool(swap_summary.get("filter_goal_heading", True)),
            )
            if directional_set is None:
                raise RuntimeError("could not reconstruct directional goals")
            source = directional_set["base_source"]
            expected = item_key(swap_row)
            actual = (
                source.get("dataset_index"),
                source.get("trajectory"),
                source.get("curr_time"),
            )
            if expected != actual:
                raise RuntimeError(f"reconstructed source mismatch: {actual} != {expected}")

            target_directions = [
                direction
                for direction in swap_row.get("available_directions", DIRECTIONS)
                if direction in DIRECTIONS
            ]
            if not target_directions:
                target_directions = [
                    direction
                    for direction in DIRECTIONS
                    if direction in directional_set["candidates"]
                ]
            missing_directions = [
                direction
                for direction in target_directions
                if direction not in directional_set["candidates"]
            ]
            if missing_directions:
                raise RuntimeError(
                    "could not reconstruct expected directions: "
                    + ",".join(missing_directions)
                )

            dist_pred = {}
            goal_offset = {}
            goal_pos_norm = {}
            goal_pos = {}
            goal_angle = {}
            for direction in target_directions:
                candidate = directional_set["candidates"][direction]
                dist_pred[direction] = scalar_dist_pred(
                    model,
                    directional_set["base_obs"],
                    candidate["goal"],
                    device,
                )
                goal_offset[direction] = float(candidate["goal_offset"])
                goal_pos_xy = np.asarray(candidate["goal_pos"][:2], dtype=np.float64)
                goal_pos[direction] = goal_pos_xy
                goal_pos_norm[direction] = float(np.linalg.norm(goal_pos_xy))
                goal_angle[direction] = float(candidate["goal_angle_deg"])

            available_pairs = [
                (first, second)
                for first, second in PAIRS
                if first in dist_pred and second in dist_pred
            ]
            pair_l2 = {
                f"{first}_{second}": abs(dist_pred[first] - dist_pred[second])
                for first, second in available_pairs
            }
            goal_pos_pair_distance = {
                f"{first}_{second}": float(
                    np.linalg.norm(goal_pos[first] - goal_pos[second])
                )
                for first, second in available_pairs
            }
            normalized_dist_pair_l2 = {
                pair: pair_l2[pair] / max(goal_pos_pair_distance[pair], 1e-12)
                for pair in pair_l2
            }
            pred_values = [dist_pred[direction] for direction in target_directions]
            offset_values = [goal_offset[direction] for direction in target_directions]
            norm_values = [goal_pos_norm[direction] for direction in target_directions]
            goal_pair_distance_mean = (
                float(np.mean(list(goal_pos_pair_distance.values())))
                if goal_pos_pair_distance
                else np.nan
            )
            item = {
                "matched_index": int(swap_row.get("matched_index", row_idx)),
                "dataset_index": dataset_index,
                "trajectory": source["trajectory"],
                "curr_time": int(source["curr_time"]),
                "available_directions": ",".join(target_directions),
                "available_pairs": ",".join(pair_l2.keys()),
                "num_available_directions": len(target_directions),
                "num_available_pairs": len(pair_l2),
                "dist_pred_pair_l2": (
                    float(np.mean(list(pair_l2.values()))) if pair_l2 else np.nan
                ),
                "goal_pos_pair_distance": goal_pair_distance_mean,
                "dist_pred_goal_normalized_sensitivity": (
                    float(np.mean(list(normalized_dist_pair_l2.values())))
                    if normalized_dist_pair_l2
                    else np.nan
                ),
                "dist_pred_rank_accuracy": pairwise_rank_accuracy(
                    dist_pred, goal_offset
                ),
                "dist_pred_goal_offset_spearman": spearman_corr(
                    pred_values, offset_values
                ),
                "dist_pred_goal_pos_norm_spearman": spearman_corr(
                    pred_values, norm_values
                ),
                "flow_endpoint_pair_distance": flow_endpoint_pair_distance_from_swap_row(
                    swap_row
                ),
                "flow_goal_direction_alignment": flow_metric_from_swap_row(
                    swap_row, "flow_goal_direction_alignment"
                ),
                "source_goal_swap_mean_endpoint_mean_distance": flow_metric_from_swap_row(
                    swap_row, "mean_endpoint_mean_distance"
                ),
            }
            if (
                np.isfinite(item["flow_endpoint_pair_distance"])
                and item["dist_pred_pair_l2"] > 1e-12
            ):
                item["flow_vs_dist_sensitivity_ratio"] = (
                    item["flow_endpoint_pair_distance"] / item["dist_pred_pair_l2"]
                )
            else:
                item["flow_vs_dist_sensitivity_ratio"] = np.nan
            if (
                np.isfinite(item["flow_endpoint_pair_distance"])
                and item["goal_pos_pair_distance"] > 1e-12
            ):
                item["flow_goal_normalized_sensitivity"] = (
                    item["flow_endpoint_pair_distance"]
                    / item["goal_pos_pair_distance"]
                )
            else:
                item["flow_goal_normalized_sensitivity"] = np.nan
            if item["dist_pred_goal_normalized_sensitivity"] > 1e-12:
                item["flow_vs_dist_goal_normalized_ratio"] = (
                    item["flow_goal_normalized_sensitivity"]
                    / item["dist_pred_goal_normalized_sensitivity"]
                    if np.isfinite(item["flow_goal_normalized_sensitivity"])
                    else np.nan
                )
            else:
                item["flow_vs_dist_goal_normalized_ratio"] = np.nan
            for direction in DIRECTIONS:
                item[f"{direction}_dist_pred"] = dist_pred.get(direction, np.nan)
                item[f"{direction}_goal_offset"] = goal_offset.get(direction, np.nan)
                item[f"{direction}_goal_pos_norm"] = goal_pos_norm.get(direction, np.nan)
                item[f"{direction}_goal_angle_deg"] = goal_angle.get(direction, np.nan)
                point = goal_pos.get(direction)
                item[f"{direction}_goal_pos_x"] = (
                    float(point[0]) if point is not None else np.nan
                )
                item[f"{direction}_goal_pos_y"] = (
                    float(point[1]) if point is not None else np.nan
                )
            for first, second in PAIRS:
                pair = f"{first}_{second}"
                item[f"dist_pred_pair_l2_{pair}"] = pair_l2.get(pair, np.nan)
                item[f"goal_pos_pair_distance_{pair}"] = goal_pos_pair_distance.get(
                    pair,
                    np.nan,
                )
                item[f"dist_pred_goal_normalized_sensitivity_{pair}"] = (
                    normalized_dist_pair_l2.get(pair, np.nan)
                )
            rows.append(item)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "row_index": row_idx,
                    "dataset_index": dataset_index,
                    "error": str(exc),
                }
            )
            if not args.keep_going:
                raise

    output_dir = summary_path.parent
    aggregate_keys = [
        "dist_pred_pair_l2",
        "goal_pos_pair_distance",
        "dist_pred_goal_normalized_sensitivity",
        "dist_pred_rank_accuracy",
        "dist_pred_goal_offset_spearman",
        "dist_pred_goal_pos_norm_spearman",
        "flow_endpoint_pair_distance",
        "flow_goal_normalized_sensitivity",
        "flow_goal_direction_alignment",
        "flow_vs_dist_sensitivity_ratio",
        "flow_vs_dist_goal_normalized_ratio",
    ]
    aggregate = {}
    for key in aggregate_keys:
        values = [
            row[key]
            for row in rows
            if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])
        ]
        aggregate[f"mean_{key}"] = float(np.mean(values)) if values else None

    summary = {
        "test": "dist_head_backfill",
        "source_goal_swap_summary_path": str(summary_path),
        "config": config_path,
        "checkpoint": checkpoint_path,
        "dataset": dataset_name,
        "split": split,
        "stage": swap_summary.get("stage"),
        "angle_threshold_deg": swap_summary.get("angle_threshold_deg"),
        "filter_goal_heading": swap_summary.get("filter_goal_heading"),
        "min_goal_offset": swap_summary.get("min_goal_offset"),
        "max_goal_offset": swap_summary.get("max_goal_offset"),
        "min_goal_pos_dist": swap_summary.get("min_goal_pos_dist"),
        "max_goal_pos_dist": swap_summary.get("max_goal_pos_dist"),
        "num_source_items": len(swap_rows),
        "num_backfilled_items": len(rows),
        "num_failures": len(failures),
        "failures": failures,
        "metrics": rows,
        **aggregate,
    }
    clear_existing_backfill_outputs(output_dir)
    summary_path_out = output_dir / timestamp_name("dist_head_backfill_summary", "json")
    csv_path_out = output_dir / timestamp_name("dist_head_backfill_items", "csv")
    write_json(summary_path_out, summary)
    write_csv(csv_path_out, rows)
    log(f"Saved dist backfill summary: {summary_path_out}")
    log(f"Saved dist backfill rows: {csv_path_out}")
    return summary_path_out


def summary_variant(summary_path: Path, summary: dict):
    if summary.get("output_variant"):
        return summary["output_variant"]
    for part in summary_path.parts:
        if part in VARIANT_ORDER:
            return part
    return "flownav_baseline"


def discover_summaries(args):
    if args.summary_path:
        return [Path(path) for path in args.summary_path]
    log_root = Path(args.log_root)
    selected = []
    for path in sorted(log_root.rglob("goal_swap_visualization_summary_*.json")):
        with open(path, "r") as f:
            summary = json.load(f)
        if summary.get("test") != "goal_swap_visualization":
            continue
        if summary.get("stage") != "all_samples":
            continue
        if summary_variant(path, summary) != args.variant:
            continue
        if args.dataset and summary.get("dataset") != args.dataset:
            continue
        if args.angle is not None and float(summary.get("angle_threshold_deg")) != float(
            args.angle
        ):
            continue
        if args.setting:
            setting = (
                "heading_filter"
                if summary.get("filter_goal_heading")
                else "no_heading_filter"
            )
            if setting != args.setting:
                continue
        selected.append(path)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Backfill dist_pred for existing goal-swap matched samples without "
            "rerunning the flow head."
        )
    )
    parser.add_argument("--summary-path", action="append", default=[])
    parser.add_argument("--log-root", default="test_logs")
    parser.add_argument("--variant", default="flownav_baseline")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--angle", type=float, default=None)
    parser.add_argument(
        "--setting",
        choices=("heading_filter", "no_heading_filter"),
        default=None,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = discover_summaries(args)
    if not summaries:
        raise RuntimeError("No matching goal_swap all_samples summaries found.")
    log(f"Found {len(summaries)} goal-swap summaries for dist backfill.")
    outputs = []
    for summary_path in summaries:
        outputs.append(process_summary(summary_path, args))
    print("Backfilled dist head summaries:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
