import argparse
import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ["go_stanford", "recon", "sacson"]
HORIZON_BUCKETS = {
    "short": (0.0, 2.0),
    "mid": (2.0, 6.0),
    "long": (6.0, None),
}
HORIZON_ORDER = ["short", "mid", "long"]
SWAP_METRICS = [
    "mean_endpoint_mean_distance",
    "mean_endpoint_rbf_mmd",
    "mean_endpoint_sliced_wasserstein",
    "flow_goal_direction_alignment",
]
DIRECTIONS = ["left", "forward", "right"]
DIRECTION_COLORS = {
    "left": "#4c78a8",
    "forward": "#59a14f",
    "right": "#e15759",
}


def latest_path(paths):
    paths = sorted(paths)
    return paths[-1] if paths else None


def summary_variant(summary_path: Path, summary: dict) -> str:
    if summary.get("output_variant"):
        return summary["output_variant"]
    for part in summary_path.parts:
        if part.startswith("flownav_"):
            return part
    return "flownav_baseline"


def path_has_variant(summary_path: Path, variant: str) -> bool:
    return variant in summary_path.parts


def finite_mean(values):
    values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(values)) if values else np.nan


def numeric_value(value):
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else np.nan


def metric_mean(rows, key):
    return finite_mean(
        row.get(key)
        for row in rows
        if isinstance(row.get(key), (int, float))
    )


def dataset_from_path(path: Path):
    for part in path.parts:
        if part in DATASET_ORDER:
            return part
    return None


def load_mask_bucket_rows(horizon_root: Path, variant: str):
    rows = []
    for bucket in HORIZON_ORDER:
        bucket_root = horizon_root / bucket
        for csv_path in sorted(
            bucket_root.rglob("goal_mask_sensitivity_items_*.csv")
        ):
            if variant not in csv_path.parts:
                continue
            dataset = dataset_from_path(csv_path)
            if dataset is None:
                continue
            with open(csv_path, "r") as f:
                items = list(csv.DictReader(f))
            numeric = []
            for item in items:
                try:
                    goal_offset = int(item["goal_time"]) - int(item["curr_time"])
                    numeric.append(
                        {
                            "goal_offset": goal_offset,
                            "endpoint_mean_l2": float(item["endpoint_mean_l2"]),
                            "matched_sample_ade": float(item["matched_sample_ade"]),
                            "matched_sample_fde": float(item["matched_sample_fde"]),
                            "endpoint_sliced_wasserstein": float(
                                item["endpoint_sliced_wasserstein"]
                            ),
                        }
                    )
                except (KeyError, ValueError):
                    continue
            row = {
                "source": str(csv_path),
                "bucket": bucket,
                "dataset": dataset,
                "count": len(numeric),
                "goal_offset_mean": metric_mean(numeric, "goal_offset"),
            }
            for metric in [
                "endpoint_mean_l2",
                "matched_sample_ade",
                "matched_sample_fde",
                "endpoint_sliced_wasserstein",
            ]:
                row[metric] = metric_mean(numeric, metric)
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]),
            HORIZON_ORDER.index(row["bucket"]),
        ),
    )


def load_swap_bucket_rows(horizon_root: Path, variant: str):
    rows = []
    for bucket in HORIZON_ORDER:
        bucket_root = horizon_root / bucket
        for summary_path in sorted(
            bucket_root.rglob("goal_swap_visualization_summary_*.json")
        ):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            if summary.get("test") != "goal_swap_visualization":
                continue
            if summary.get("stage") != "all_samples":
                continue
            if summary.get("filter_goal_heading") is not False:
                continue
            if summary_variant(summary_path, summary) != variant and not path_has_variant(
                summary_path, variant
            ):
                continue
            metrics = summary.get("metrics", [])
            row = {
                "source": str(summary_path),
                "bucket": bucket,
                "dataset": summary.get("dataset"),
                "count": int(summary.get("num_matched_sets", 0)),
                "min_goal_offset": summary.get("min_goal_offset"),
                "max_goal_offset": summary.get("max_goal_offset"),
                "min_goal_pos_dist": summary.get("min_goal_pos_dist"),
                "max_goal_pos_dist": summary.get("max_goal_pos_dist"),
            }
            for metric in SWAP_METRICS:
                row[metric] = metric_mean(metrics, metric)
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"])
            if row["dataset"] in DATASET_ORDER
            else 999,
            HORIZON_ORDER.index(row["bucket"]),
        ),
    )


def load_head_rows(head_log_root: Path, variant: str, angle: float):
    rows = []
    for summary_path in sorted(head_log_root.rglob("dist_head_backfill_summary_*.json")):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if summary.get("test") != "dist_head_backfill":
            continue
        if summary.get("filter_goal_heading") is not False:
            continue
        if float(summary.get("angle_threshold_deg", np.nan)) != float(angle):
            continue
        source_path = Path(summary.get("source_goal_swap_summary_path", ""))
        if variant not in source_path.parts and variant not in summary_path.parts:
            continue
        row = {
            "source": str(summary_path),
            "dataset": summary.get("dataset"),
            "count": int(summary.get("num_backfilled_items", 0)),
        }
        for metric in [
            "mean_dist_pred_pair_l2",
            "mean_dist_pred_rank_accuracy",
            "mean_dist_pred_goal_offset_spearman",
            "mean_flow_endpoint_pair_distance",
            "mean_flow_goal_direction_alignment",
            "mean_flow_vs_dist_sensitivity_ratio",
        ]:
            row[metric] = summary.get(metric, np.nan)
        if not np.isfinite(numeric_value(row["mean_flow_vs_dist_sensitivity_ratio"])):
            flow = numeric_value(row["mean_flow_endpoint_pair_distance"])
            dist = numeric_value(row["mean_dist_pred_pair_l2"])
            row["mean_flow_vs_dist_sensitivity_ratio"] = (
                flow / dist if np.isfinite(flow) and dist > 1e-12 else np.nan
            )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"])
            if row["dataset"] in DATASET_ORDER
            else 999
        ),
    )


def load_horizon_head_rows(horizon_root: Path, variant: str, angle: float):
    rows = []
    for bucket in HORIZON_ORDER:
        bucket_root = horizon_root / bucket
        for summary_path in sorted(bucket_root.rglob("dist_head_backfill_summary_*.json")):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            if summary.get("test") != "dist_head_backfill":
                continue
            if summary.get("filter_goal_heading") is not False:
                continue
            if float(summary.get("angle_threshold_deg", np.nan)) != float(angle):
                continue
            source_path = Path(summary.get("source_goal_swap_summary_path", ""))
            if variant not in source_path.parts and variant not in summary_path.parts:
                continue
            rows.append(
                {
                    "source": str(summary_path),
                    "bucket": bucket,
                    "dataset": summary.get("dataset"),
                    "count": int(summary.get("num_backfilled_items", 0)),
                    "num_source_items": int(summary.get("num_source_items", 0)),
                    "num_failures": int(summary.get("num_failures", 0)),
                    "min_goal_pos_dist": summary.get("min_goal_pos_dist"),
                    "max_goal_pos_dist": summary.get("max_goal_pos_dist"),
                    "mean_dist_pred_pair_l2": summary.get(
                        "mean_dist_pred_pair_l2",
                        np.nan,
                    ),
                    "mean_goal_pos_pair_distance": summary.get(
                        "mean_goal_pos_pair_distance",
                        np.nan,
                    ),
                    "mean_dist_pred_goal_normalized_sensitivity": summary.get(
                        "mean_dist_pred_goal_normalized_sensitivity",
                        np.nan,
                    ),
                    "mean_dist_pred_rank_accuracy": summary.get(
                        "mean_dist_pred_rank_accuracy",
                        np.nan,
                    ),
                    "mean_dist_pred_goal_pos_norm_spearman": summary.get(
                        "mean_dist_pred_goal_pos_norm_spearman",
                        np.nan,
                    ),
                    "mean_flow_endpoint_pair_distance": summary.get(
                        "mean_flow_endpoint_pair_distance",
                        np.nan,
                    ),
                    "mean_flow_goal_normalized_sensitivity": summary.get(
                        "mean_flow_goal_normalized_sensitivity",
                        np.nan,
                    ),
                    "mean_flow_vs_dist_sensitivity_ratio": summary.get(
                        "mean_flow_vs_dist_sensitivity_ratio",
                        np.nan,
                    ),
                    "mean_flow_vs_dist_goal_normalized_ratio": summary.get(
                        "mean_flow_vs_dist_goal_normalized_ratio",
                        np.nan,
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"])
            if row["dataset"] in DATASET_ORDER
            else 999,
            HORIZON_ORDER.index(row["bucket"]),
        ),
    )


def load_dist_pred_items(head_log_root: Path, variant: str, angle: float):
    records = []
    for summary_path in sorted(head_log_root.rglob("dist_head_backfill_summary_*.json")):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if summary.get("test") != "dist_head_backfill":
            continue
        if summary.get("filter_goal_heading") is not False:
            continue
        if float(summary.get("angle_threshold_deg", np.nan)) != float(angle):
            continue
        source_path = Path(summary.get("source_goal_swap_summary_path", ""))
        if variant not in source_path.parts and variant not in summary_path.parts:
            continue
        dataset = summary.get("dataset") or dataset_from_path(summary_path)
        item_path = latest_path(summary_path.parent.glob("dist_head_backfill_items_*.csv"))
        if item_path is None:
            continue
        with open(item_path, "r") as f:
            for row in csv.DictReader(f):
                for direction in DIRECTIONS:
                    try:
                        records.append(
                            {
                                "source": str(item_path),
                                "dataset": dataset,
                                "direction": direction,
                                "goal_offset": float(row[f"{direction}_goal_offset"]),
                                "goal_pos_dist": float(row[f"{direction}_goal_pos_norm"]),
                                "dist_pred": float(row[f"{direction}_dist_pred"]),
                            }
                        )
                    except (KeyError, ValueError):
                        continue
    return sorted(
        records,
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"])
            if row["dataset"] in DATASET_ORDER
            else 999,
            row["goal_pos_dist"],
        ),
    )


def find_endpoint_distribution_images(horizon_root: Path, variant: str):
    rows = []
    for bucket in HORIZON_ORDER:
        bucket_root = horizon_root / bucket
        for summary_path in sorted(
            bucket_root.rglob("goal_swap_visualization_summary_*.json")
        ):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            if summary.get("test") != "goal_swap_visualization":
                continue
            if summary.get("stage") != "all_samples":
                continue
            if summary.get("filter_goal_heading") is not False:
                continue
            if summary_variant(summary_path, summary) != variant and not path_has_variant(
                summary_path, variant
            ):
                continue
            image_path = latest_path(
                summary_path.parent.glob("goal_swap_global_endpoints_*.png")
            )
            rows.append(
                {
                    "bucket": bucket,
                    "dataset": summary.get("dataset") or dataset_from_path(summary_path),
                    "count": int(summary.get("num_matched_sets", 0)),
                    "min_goal_offset": summary.get("min_goal_offset"),
                    "max_goal_offset": summary.get("max_goal_offset"),
                    "min_goal_pos_dist": summary.get("min_goal_pos_dist"),
                    "max_goal_pos_dist": summary.get("max_goal_pos_dist"),
                    "summary_path": str(summary_path),
                    "image_path": str(image_path) if image_path else "",
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"])
            if row["dataset"] in DATASET_ORDER
            else 999,
            HORIZON_ORDER.index(row["bucket"]),
        ),
    )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def rows_by_dataset(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["dataset"], {})[row["bucket"]] = row
    return grouped


def plot_dist_pred_by_goal_pos_dist(records, output_path: Path):
    if not records:
        return
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(17, 5), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, DATASET_ORDER):
        dataset_records = [row for row in records if row["dataset"] == dataset]
        if not dataset_records:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dataset)
            continue
        for direction in DIRECTIONS:
            points = [row for row in dataset_records if row["direction"] == direction]
            distances = np.asarray([row["goal_pos_dist"] for row in points], dtype=float)
            preds = np.asarray([row["dist_pred"] for row in points], dtype=float)
            if len(distances) == 0:
                continue
            ax.scatter(
                distances,
                preds,
                s=12,
                alpha=0.22,
                color=DIRECTION_COLORS[direction],
                label=direction,
            )
        distance_bins = sorted(
            {
                int(np.floor(row["goal_pos_dist"]))
                for row in dataset_records
                if np.isfinite(row["goal_pos_dist"])
            }
        )
        means = []
        for distance_bin in distance_bins:
            values = [
                row["dist_pred"]
                for row in dataset_records
                if int(np.floor(row["goal_pos_dist"])) == distance_bin
            ]
            means.append(float(np.mean(values)))
        ax.plot(
            distance_bins,
            means,
            color="black",
            marker="o",
            linewidth=2.0,
            label="offset mean",
        )
        ax.set_title(dataset)
        ax.set_xlabel("local goal distance ||goal_pos|| (m)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("dist_pred")
    fig.suptitle(
        "Dist head: dist_pred values across local goal distances and directions",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_endpoint_goal_distribution_collage(rows, output_path: Path):
    if not rows:
        return
    grouped = rows_by_dataset(rows)
    fig, axes = plt.subplots(
        len(DATASET_ORDER),
        len(HORIZON_ORDER),
        figsize=(18, 13),
    )
    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, bucket in enumerate(HORIZON_ORDER):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            record = grouped.get(dataset, {}).get(bucket)
            if record is None or not record.get("image_path"):
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_title(f"{dataset} | {bucket}")
                continue
            image_path = Path(record["image_path"])
            if not image_path.exists():
                ax.text(0.5, 0.5, "missing image", ha="center", va="center")
                ax.set_title(f"{dataset} | {bucket}")
                continue
            ax.imshow(mpimg.imread(image_path))
            dist_min = record.get("min_goal_pos_dist")
            dist_max = record.get("max_goal_pos_dist")
            if dist_min is not None or dist_max is not None:
                lower = "0" if dist_min is None else f"{float(dist_min):g}"
                upper = "inf" if dist_max is None else f"{float(dist_max):g}"
                range_text = f"local dist {lower}-{upper}m"
            else:
                range_text = (
                    f"offset {record.get('min_goal_offset')}-"
                    f"{record.get('max_goal_offset')}"
                )
            ax.set_title(
                f"{dataset} | {bucket} ({range_text}), n={record.get('count')}",
                fontsize=10,
            )
    fig.suptitle(
        "Flow endpoint and goal-position distributions by local goal-distance bucket",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_endpoint_mmd_emd_by_horizon(rows, output_path: Path):
    if not rows:
        return
    grouped = rows_by_dataset(rows)
    x = np.arange(len(HORIZON_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    panels = [
        (
            "mean_endpoint_rbf_mmd",
            "Endpoint RBF-MMD",
            "mean endpoint RBF-MMD",
        ),
        (
            "mean_endpoint_sliced_wasserstein",
            "Endpoint sliced Wasserstein",
            "mean endpoint SWD",
        ),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for dataset in DATASET_ORDER:
            values = []
            counts = []
            for bucket in HORIZON_ORDER:
                row = grouped.get(dataset, {}).get(bucket)
                values.append(numeric_value(row.get(metric)) if row else np.nan)
                counts.append(int(row.get("count", 0)) if row else 0)
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2.0,
                label=dataset,
            )
            for xi, yi, count in zip(x, values, counts):
                if np.isfinite(yi):
                    ax.annotate(
                        f"n={count}",
                        xy=(xi, yi),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZON_ORDER)
        ax.set_xlabel("local goal-distance bucket")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Flow endpoint distribution sensitivity across goal-distance buckets",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_flow_dist_sensitivity_by_horizon(rows, output_path: Path):
    if not rows:
        return
    grouped = rows_by_dataset(rows)
    x = np.arange(len(HORIZON_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
    panels = [
        (
            "mean_flow_endpoint_pair_distance",
            "Flow head sensitivity",
            "mean endpoint pair distance",
        ),
        (
            "mean_dist_pred_pair_l2",
            "Dist head sensitivity",
            "mean dist_pred pair L1",
        ),
        (
            "mean_flow_vs_dist_sensitivity_ratio",
            "Flow / dist sensitivity ratio",
            "ratio",
        ),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for dataset in DATASET_ORDER:
            values = []
            counts = []
            for bucket in HORIZON_ORDER:
                row = grouped.get(dataset, {}).get(bucket)
                values.append(numeric_value(row.get(metric)) if row else np.nan)
                counts.append(int(row.get("count", 0)) if row else 0)
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2.0,
                label=dataset,
            )
            finite_values = [value for value in values if np.isfinite(value)]
            if len(finite_values) >= 2:
                gradient = finite_values[-1] - finite_values[0]
                ax.text(
                    0.02,
                    0.94 - 0.08 * DATASET_ORDER.index(dataset),
                    f"{dataset} long-short={gradient:.3g}",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            for xi, yi, count in zip(x, values, counts):
                if np.isfinite(yi):
                    ax.annotate(
                        f"n={count}",
                        xy=(xi, yi),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZON_ORDER)
        ax.set_xlabel("local goal-distance bucket")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Flow head vs dist head goal sensitivity across goal-distance buckets",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_goal_normalized_flow_dist_sensitivity(rows, output_path: Path):
    if not rows:
        return
    grouped = rows_by_dataset(rows)
    x = np.arange(len(HORIZON_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), sharex=True)
    panels = [
        (
            "mean_flow_goal_normalized_sensitivity",
            "Flow sensitivity / goal-pose distance",
            "endpoint pair distance / goal pair distance",
        ),
        (
            "mean_dist_pred_goal_normalized_sensitivity",
            "Dist sensitivity / goal-pose distance",
            "dist_pred pair difference / goal pair distance",
        ),
        (
            "mean_flow_vs_dist_goal_normalized_ratio",
            "Goal-normalized flow / dist ratio",
            "normalized ratio",
        ),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for dataset in DATASET_ORDER:
            values = []
            counts = []
            for bucket in HORIZON_ORDER:
                row = grouped.get(dataset, {}).get(bucket)
                values.append(numeric_value(row.get(metric)) if row else np.nan)
                counts.append(int(row.get("count", 0)) if row else 0)
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2.0,
                label=dataset,
            )
            finite_values = [value for value in values if np.isfinite(value)]
            if len(finite_values) >= 2:
                gradient = finite_values[-1] - finite_values[0]
                ax.text(
                    0.02,
                    0.94 - 0.08 * DATASET_ORDER.index(dataset),
                    f"{dataset} long-short={gradient:.3g}",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            for xi, yi, count in zip(x, values, counts):
                if np.isfinite(yi):
                    ax.annotate(
                        f"n={count}",
                        xy=(xi, yi),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZON_ORDER)
        ax.set_xlabel("local goal-distance bucket")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Goal-space calibrated flow head vs dist head sensitivity",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate focused figures for dist_pred vs goal_offset and "
            "endpoint/goal-position distributions by horizon bucket."
        )
    )
    parser.add_argument("--head-log-root", default="test_logs")
    parser.add_argument("--horizon-root", default="test_logs_horizon")
    parser.add_argument("--variant", default="flownav_baseline")
    parser.add_argument("--angle", type=float, default=10)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.horizon_root) / args.variant / "head_horizon_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_figure in output_dir.glob("fig_*.png"):
        old_figure.unlink()

    head_rows = load_head_rows(Path(args.head_log_root), args.variant, args.angle)
    dist_pred_items = load_dist_pred_items(
        Path(args.head_log_root),
        args.variant,
        args.angle,
    )
    mask_rows = load_mask_bucket_rows(Path(args.horizon_root), args.variant)
    swap_rows = load_swap_bucket_rows(Path(args.horizon_root), args.variant)
    horizon_head_rows = load_horizon_head_rows(
        Path(args.horizon_root),
        args.variant,
        args.angle,
    )
    endpoint_image_rows = find_endpoint_distribution_images(
        Path(args.horizon_root),
        args.variant,
    )

    write_csv(output_dir / "head_level_summary.csv", head_rows)
    write_csv(output_dir / "dist_pred_goal_pos_dist_items.csv", dist_pred_items)
    write_csv(output_dir / "horizon_mask_summary.csv", mask_rows)
    write_csv(output_dir / "horizon_swap_summary.csv", swap_rows)
    write_csv(output_dir / "horizon_head_level_summary.csv", horizon_head_rows)
    write_csv(output_dir / "endpoint_distribution_images.csv", endpoint_image_rows)

    figure_paths = []
    if dist_pred_items:
        path = output_dir / "fig_dist_pred_by_goal_pos_dist.png"
        plot_dist_pred_by_goal_pos_dist(
            dist_pred_items,
            path,
        )
        figure_paths.append(path)
    if endpoint_image_rows:
        path = output_dir / "fig_endpoint_goal_distribution_by_horizon.png"
        plot_endpoint_goal_distribution_collage(
            endpoint_image_rows,
            path,
        )
        figure_paths.append(path)
    if swap_rows:
        path = output_dir / "fig_endpoint_mmd_emd_by_horizon.png"
        plot_endpoint_mmd_emd_by_horizon(swap_rows, path)
        figure_paths.append(path)
    if horizon_head_rows:
        path = output_dir / "fig_flow_vs_dist_sensitivity_by_horizon.png"
        plot_flow_dist_sensitivity_by_horizon(horizon_head_rows, path)
        figure_paths.append(path)
        calibrated_path = (
            output_dir / "fig_goal_normalized_flow_vs_dist_sensitivity_by_horizon.png"
        )
        plot_goal_normalized_flow_dist_sensitivity(
            horizon_head_rows,
            calibrated_path,
        )
        figure_paths.append(calibrated_path)

    missing = {
        "head_backfill_found": bool(head_rows),
        "dist_pred_item_rows": len(dist_pred_items),
        "mask_bucket_rows": len(mask_rows),
        "swap_bucket_rows": len(swap_rows),
        "horizon_head_bucket_rows": len(horizon_head_rows),
        "endpoint_distribution_images": len(
            [row for row in endpoint_image_rows if row.get("image_path")]
        ),
        "expected_mask_bucket_rows": len(DATASET_ORDER) * len(HORIZON_ORDER),
        "expected_swap_bucket_rows": len(DATASET_ORDER) * len(HORIZON_ORDER),
        "expected_endpoint_distribution_images": len(DATASET_ORDER)
        * len(HORIZON_ORDER),
    }
    with open(output_dir / "missing_or_counts.json", "w") as f:
        json.dump(missing, f, indent=2)

    print(output_dir / "head_level_summary.csv")
    print(output_dir / "dist_pred_goal_pos_dist_items.csv")
    print(output_dir / "horizon_mask_summary.csv")
    print(output_dir / "horizon_swap_summary.csv")
    print(output_dir / "horizon_head_level_summary.csv")
    print(output_dir / "endpoint_distribution_images.csv")
    for path in figure_paths:
        print(path)
    print(output_dir / "missing_or_counts.json")


if __name__ == "__main__":
    main()
