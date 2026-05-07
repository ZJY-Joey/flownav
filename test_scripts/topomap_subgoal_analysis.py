import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch

from common import (
    DEFAULT_CONFIG,
    build_dataset,
    build_model,
    find_all_matched_directional_goal_sets,
    get_device,
    imagenet_transform,
    load_config,
    log,
    prepare_goal_image,
    resolve_checkpoint,
    run_dist_pred,
    safe_load_image,
    timestamp_name,
    write_csv,
    write_json,
)
from flownav.data.data_utils import to_local_coords


DATASETS = ("go_stanford", "recon", "sacson")
DIRECTIONS = ("left", "forward", "right")
HORIZON_BUCKETS = {
    "short": (0.0, 7.0),
    "mid": (8.0, 12.0),
    "long": (13.0, 19.0),
}
BUCKET_ORDER = ("short", "mid", "long")
COLORS = {"left": "#4c78a8", "forward": "#59a14f", "right": "#e15759"}
VARIANT = "flownav_baseline"
SUBGOAL_GOAL_PANEL = "panel_topomap_subgoal_vs_goal_pose.png"
FLOW_SUBGOAL_PANEL = "panel_flow_endpoint_vs_topomap_subgoal.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline topomap subgoal analysis under left/forward/right final "
            "goals and metric goal-distance buckets."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--angle-threshold-deg", type=float, default=10.0)
    parser.add_argument("--scan-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-visualizations", type=int, default=None)
    parser.add_argument("--max-direction-angle-deg", type=float, default=90.0)
    parser.add_argument("--topomap-radius", type=int, default=4)
    parser.add_argument("--close-threshold", type=float, default=3.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="test_logs_horizon")
    parser.add_argument(
        "--swap-log-root",
        default="test_logs_horizon",
        help="Root containing metric-distance goal_swap logs for flow endpoint panels.",
    )
    parser.add_argument("--global-max-points-per-direction", type=int, default=5000)
    parser.add_argument(
        "--montage-only",
        action="store_true",
        help=(
            "Only stitch existing per-dataset/per-bucket panel PNGs into the "
            "two summary figures. Does not load the model or rerun inference."
        ),
    )
    return parser.parse_args()


def local_metric_position(traj_data, curr_time: int, target_time: int) -> np.ndarray:
    curr_yaw = float(np.asarray(traj_data["yaw"][curr_time]).squeeze())
    curr_pos = np.asarray(traj_data["position"][curr_time][:2], dtype=np.float32)
    target_pos = np.asarray(traj_data["position"][target_time][:2], dtype=np.float32)
    return to_local_coords(target_pos[None], curr_pos, curr_yaw)[0].astype(np.float32)


def topomap_node_times(dataset, curr_time: int, goal_time: int, radius: int) -> list[int]:
    step = int(dataset.waypoint_spacing)
    start = max(curr_time - radius * step, 0)
    end = min(curr_time + (radius + 1) * step, goal_time)
    if end < start:
        end = curr_time
    times = list(range(start, end + 1, step))
    if curr_time not in times:
        times.append(curr_time)
    return sorted(set(int(t) for t in times))


def select_topomap_subgoal(
    model,
    dataset,
    transform,
    device,
    directional_set,
    direction: str,
    radius: int,
    close_threshold: float,
):
    source = directional_set["base_source"]
    candidate = directional_set["candidates"][direction]
    traj_name = source["trajectory"]
    curr_time = int(source["curr_time"])
    goal_time = int(candidate["goal_time"])
    node_times = topomap_node_times(dataset, curr_time, goal_time, radius)
    goal_images = torch.cat(
        [
            prepare_goal_image(
                safe_load_image(dataset, traj_name, node_time),
                transform,
                device,
            )
            for node_time in node_times
        ],
        dim=0,
    )
    obs = directional_set["base_obs"].repeat(len(node_times), 1, 1, 1)
    dists = run_dist_pred(model, obs, goal_images, device).detach().cpu().reshape(-1).numpy()
    final_goal_dist_pred = float(
        run_dist_pred(
            model,
            directional_set["base_obs"],
            candidate["goal"],
            device,
        )
        .detach()
        .cpu()
        .reshape(-1)
        .mean()
        .item()
    )
    closest_idx = int(np.argmin(dists))
    subgoal_idx = min(
        closest_idx + int(float(dists[closest_idx]) < close_threshold),
        len(node_times) - 1,
    )
    subgoal_time = int(node_times[subgoal_idx])
    traj_data = dataset._get_trajectory(traj_name)
    subgoal_pos = local_metric_position(traj_data, curr_time, subgoal_time)
    goal_pos = np.asarray(candidate.get("goal_pos_metric"), dtype=np.float32)
    if goal_pos.shape != (2,):
        goal_pos = local_metric_position(traj_data, curr_time, goal_time)
    return {
        "node_times": node_times,
        "dists": dists,
        "closest_idx": closest_idx,
        "closest_time": int(node_times[closest_idx]),
        "final_goal_dist_pred": final_goal_dist_pred,
        "subgoal_idx": subgoal_idx,
        "subgoal_time": subgoal_time,
        "subgoal_pos": subgoal_pos,
        "goal_pos": goal_pos,
    }


def run_bucket(args, model, config, device, dataset_name: str, bucket: str, dist_range):
    min_dist, max_dist = dist_range
    dataset = build_dataset(config, dataset_name, args.split)
    transform = imagenet_transform()
    try:
        directional_sets = find_all_matched_directional_goal_sets(
            dataset=dataset,
            transform=transform,
            device=device,
            angle_threshold_deg=args.angle_threshold_deg,
            scan_items=args.scan_batches * args.batch_size,
            min_goal_pos_dist=min_dist,
            max_goal_pos_dist=max_dist,
            max_direction_angle_deg=args.max_direction_angle_deg,
            filter_goal_heading=False,
            require_all_directions=False,
        )
    except RuntimeError as exc:
        log(
            f"No directional goals for {dataset_name}/{bucket} "
            f"with local goal distance {min_dist}-{max_dist}m: {exc}"
        )
        return []
    if args.max_visualizations is not None:
        directional_sets = directional_sets[: args.max_visualizations]

    rows = []
    for matched_index, directional_set in enumerate(directional_sets):
        source = directional_set["base_source"]
        if matched_index % 25 == 0:
            log(
                f"topomap subgoal {dataset_name}/{bucket} "
                f"{matched_index + 1}/{len(directional_sets)}"
            )
        for direction in DIRECTIONS:
            if direction not in directional_set["candidates"]:
                continue
            candidate = directional_set["candidates"][direction]
            selected = select_topomap_subgoal(
                model=model,
                dataset=dataset,
                transform=transform,
                device=device,
                directional_set=directional_set,
                direction=direction,
                radius=args.topomap_radius,
                close_threshold=args.close_threshold,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "bucket": bucket,
                    "min_goal_pos_dist": min_dist,
                    "max_goal_pos_dist": max_dist,
                    "matched_index": matched_index,
                    "direction": direction,
                    "dataset_index": source["dataset_index"],
                    "trajectory": source["trajectory"],
                    "curr_time": source["curr_time"],
                    "goal_time": candidate["goal_time"],
                    "goal_offset": candidate["goal_offset"],
                    "goal_pos_x": float(selected["goal_pos"][0]),
                    "goal_pos_y": float(selected["goal_pos"][1]),
                    "goal_pos_dist": float(np.linalg.norm(selected["goal_pos"])),
                    "closest_time": selected["closest_time"],
                    "closest_pred_dist": float(selected["dists"][selected["closest_idx"]]),
                    "final_goal_dist_pred": selected["final_goal_dist_pred"],
                    "final_goal_dist_error": float(
                        selected["final_goal_dist_pred"]
                        - np.linalg.norm(selected["goal_pos"])
                    ),
                    "final_goal_abs_dist_error": float(
                        abs(
                            selected["final_goal_dist_pred"]
                            - np.linalg.norm(selected["goal_pos"])
                        )
                    ),
                    "subgoal_time": selected["subgoal_time"],
                    "subgoal_pos_x": float(selected["subgoal_pos"][0]),
                    "subgoal_pos_y": float(selected["subgoal_pos"][1]),
                    "subgoal_pos_dist": float(np.linalg.norm(selected["subgoal_pos"])),
                    "subgoal_time_delta": int(selected["subgoal_time"] - source["curr_time"]),
                    "topomap_window_start": int(selected["node_times"][0]),
                    "topomap_window_end": int(selected["node_times"][-1]),
                    "topomap_window_count": len(selected["node_times"]),
                }
            )
    return rows


def latest_summary_for_bucket(swap_log_root: Path, dataset: str, bucket: str):
    candidates = []
    bucket_root = swap_log_root / bucket
    for path in bucket_root.rglob("goal_swap_visualization_summary_*.json"):
        with open(path, "r") as f:
            summary = json.load(f)
        if summary.get("test") != "goal_swap_visualization":
            continue
        if summary.get("stage") != "all_samples":
            continue
        if summary.get("dataset") != dataset:
            continue
        if summary.get("filter_goal_heading") is not False:
            continue
        if "flownav_baseline" not in path.parts and summary.get("output_variant") != VARIANT:
            continue
        candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def finite_mean(values):
    numeric = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            numeric.append(value)
    return float(np.mean(numeric)) if numeric else np.nan


def load_endpoint_mmd_emd_rows(swap_log_root: Path):
    rows = []
    for dataset in DATASETS:
        for bucket in BUCKET_ORDER:
            summary_path = latest_summary_for_bucket(swap_log_root, dataset, bucket)
            if summary_path is None:
                rows.append(
                    {
                        "dataset": dataset,
                        "bucket": bucket,
                        "count": 0,
                        "mean_endpoint_rbf_mmd": np.nan,
                        "mean_endpoint_sliced_wasserstein": np.nan,
                        "source_summary": "",
                    }
                )
                continue
            with open(summary_path, "r") as f:
                summary = json.load(f)
            metrics = summary.get("metrics", [])
            rows.append(
                {
                    "dataset": dataset,
                    "bucket": bucket,
                    "count": int(summary.get("num_matched_sets", len(metrics))),
                    "mean_endpoint_rbf_mmd": finite_mean(
                        row.get("mean_endpoint_rbf_mmd") for row in metrics
                    ),
                    "mean_endpoint_sliced_wasserstein": finite_mean(
                        row.get("mean_endpoint_sliced_wasserstein") for row in metrics
                    ),
                    "source_summary": str(summary_path),
                }
            )
    return rows


def load_flow_endpoint_means(swap_log_root: Path):
    rows = []
    for dataset in DATASETS:
        for bucket in BUCKET_ORDER:
            summary_path = latest_summary_for_bucket(swap_log_root, dataset, bucket)
            if summary_path is None:
                continue
            with open(summary_path, "r") as f:
                summary = json.load(f)
            for metric_row in summary.get("metrics", []):
                endpoint_means = metric_row.get("endpoint_means", {})
                for direction in DIRECTIONS:
                    point = endpoint_means.get(direction)
                    if not point:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "bucket": bucket,
                            "direction": direction,
                            "endpoint_x": float(point[0]),
                            "endpoint_y": float(point[1]),
                            "source_summary": str(summary_path),
                        }
                    )
    return rows


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def draw_covariance_circle(ax, points: np.ndarray, color: str):
    if len(points) < 2:
        return
    mean = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov + 1e-6 * np.eye(2))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    theta = np.linspace(0, 2 * np.pi, 160)
    scale = 2.0
    ellipse = np.stack(
        [np.cos(theta) * np.sqrt(eigvals[0]), np.sin(theta) * np.sqrt(eigvals[1])],
        axis=0,
    )
    ellipse = (eigvecs @ (scale * ellipse)).T + mean[None]
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linewidth=1.8)


def scatter_by_direction(ax, rows, x_key, y_key, title, max_points):
    plotted = False
    for direction in DIRECTIONS:
        points = np.asarray(
            [
                [float(row[x_key]), float(row[y_key])]
                for row in rows
                if row["direction"] == direction
            ],
            dtype=np.float32,
        )
        if len(points) == 0:
            continue
        display = downsample(points, max_points)
        ax.scatter(
            display[:, 0],
            display[:, 1],
            s=12,
            alpha=0.28,
            color=COLORS[direction],
            label=direction,
        )
        draw_covariance_circle(ax, points, COLORS[direction])
        plotted = True
    ax.scatter([0.0], [0.0], c="black", marker="x", s=35, label="current")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if not plotted:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)


def plot_subgoal_vs_goal(
    rows,
    output_path: Path,
    max_points: int,
    dataset_name: str | None = None,
    bucket_name: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    dataset = rows[0]["dataset"] if rows else dataset_name or "missing"
    bucket = rows[0]["bucket"] if rows else bucket_name or "missing"
    scatter_by_direction(
        axes[0],
        rows,
        "subgoal_pos_x",
        "subgoal_pos_y",
        "selected subgoals",
        max_points,
    )
    scatter_by_direction(
        axes[1],
        rows,
        "goal_pos_x",
        "goal_pos_y",
        "final goal poses",
        max_points,
    )
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{dataset} / {bucket}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_flow_vs_subgoal(
    subgoal_rows,
    flow_rows,
    output_path: Path,
    max_points: int,
    dataset_name: str | None = None,
    bucket_name: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    dataset = subgoal_rows[0]["dataset"] if subgoal_rows else (
        flow_rows[0]["dataset"] if flow_rows else dataset_name or "missing"
    )
    bucket = subgoal_rows[0]["bucket"] if subgoal_rows else (
        flow_rows[0]["bucket"] if flow_rows else bucket_name or "missing"
    )
    scatter_by_direction(
        axes[0],
        flow_rows,
        "endpoint_x",
        "endpoint_y",
        "flow endpoint means",
        max_points,
    )
    scatter_by_direction(
        axes[1],
        subgoal_rows,
        "subgoal_pos_x",
        "subgoal_pos_y",
        "selected subgoals",
        max_points,
    )
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{dataset} / {bucket}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def panel_path(root: Path, dataset: str, bucket: str, panel_name: str) -> Path:
    return root / bucket / VARIANT / dataset / panel_name


def existing_panel_path(root: Path, dataset: str, bucket: str, panel_name: str) -> Path | None:
    path = panel_path(root, dataset, bucket, panel_name)
    if path.exists():
        return path
    cell_dir = root / bucket / VARIANT / dataset
    fallback_patterns = {
        SUBGOAL_GOAL_PANEL: (
            "*topomap*subgoal*goal*.png",
            "*subgoal*goal*.png",
        ),
        FLOW_SUBGOAL_PANEL: (
            "*flow*subgoal*.png",
            "*endpoint*subgoal*.png",
        ),
    }
    for pattern in fallback_patterns.get(panel_name, ()):
        matches = sorted(cell_dir.glob(pattern))
        if matches:
            return matches[-1]
    return None


def latest_goal_swap_global_endpoint_png(swap_log_root: Path, dataset: str, bucket: str):
    bucket_root = swap_log_root / bucket
    candidates = []
    for path in bucket_root.rglob("goal_swap_global_endpoints_*.png"):
        parts = set(path.parts)
        if dataset not in parts:
            continue
        if "all_samples" not in parts or "no_heading_filter" not in parts:
            continue
        if VARIANT not in parts:
            continue
        candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def compose_panel_montage(root: Path, panel_name: str, output_path: Path, title: str):
    fig, axes = plt.subplots(len(DATASETS), len(BUCKET_ORDER), figsize=(18, 12))
    for r, dataset in enumerate(DATASETS):
        for c, bucket in enumerate(BUCKET_ORDER):
            ax = axes[r, c]
            path = existing_panel_path(root, dataset, bucket, panel_name)
            if path is not None:
                ax.imshow(mpimg.imread(path))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "missing panel",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(f"{dataset} / {bucket}", fontsize=10)
            ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compose_goal_swap_montage(swap_log_root: Path, output_path: Path):
    fig, axes = plt.subplots(len(DATASETS), len(BUCKET_ORDER), figsize=(18, 12))
    for r, dataset in enumerate(DATASETS):
        for c, bucket in enumerate(BUCKET_ORDER):
            ax = axes[r, c]
            path = latest_goal_swap_global_endpoint_png(swap_log_root, dataset, bucket)
            if path is not None:
                ax.imshow(mpimg.imread(path))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "missing goal-swap panel",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(f"{dataset} / {bucket}", fontsize=10)
            ax.axis("off")
    fig.suptitle("Flow endpoint distribution vs goal position distribution", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def rows_by_dataset_bucket(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["dataset"], {})[row["bucket"]] = row
    return grouped


def plot_endpoint_mmd_emd_by_bucket(rows, output_path: Path):
    grouped = rows_by_dataset_bucket(rows)
    x = np.arange(len(BUCKET_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    panels = [
        ("mean_endpoint_rbf_mmd", "Endpoint RBF-MMD", "mean endpoint RBF-MMD"),
        (
            "mean_endpoint_sliced_wasserstein",
            "Endpoint sliced Wasserstein",
            "mean endpoint SWD",
        ),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for dataset in DATASETS:
            values = []
            counts = []
            for bucket in BUCKET_ORDER:
                row = grouped.get(dataset, {}).get(bucket)
                value = row.get(metric) if row else np.nan
                values.append(float(value) if np.isfinite(float(value)) else np.nan)
                counts.append(int(row.get("count", 0)) if row else 0)
            ax.plot(x, values, marker="o", linewidth=2.0, label=dataset)
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
        ax.set_xticklabels(BUCKET_ORDER)
        ax.set_xlabel("local goal-distance bucket")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Flow endpoint distribution sensitivity by goal-distance bucket", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def summarize_final_goal_dist(rows):
    summary = []
    for dataset in DATASETS:
        for bucket in BUCKET_ORDER:
            cell_rows = [
                row for row in rows
                if row["dataset"] == dataset and row["bucket"] == bucket
            ]
            pred_mean = finite_mean(row.get("final_goal_dist_pred") for row in cell_rows)
            pose_mean = finite_mean(row.get("goal_pos_dist") for row in cell_rows)
            error_mean = finite_mean(row.get("final_goal_dist_error") for row in cell_rows)
            abs_error_mean = finite_mean(
                row.get("final_goal_abs_dist_error") for row in cell_rows
            )
            if not np.isfinite(error_mean) and np.isfinite(pred_mean) and np.isfinite(pose_mean):
                error_mean = pred_mean - pose_mean
                abs_error_mean = abs(error_mean)
            summary.append(
                {
                    "dataset": dataset,
                    "bucket": bucket,
                    "count": len(cell_rows),
                    "final_goal_dist_pred_mean": pred_mean,
                    "goal_pos_dist_mean": pose_mean,
                    "final_goal_dist_error_mean": error_mean,
                    "final_goal_abs_dist_error_mean": abs_error_mean,
                }
            )
    return summary


def plot_final_goal_dist_error_by_bucket(rows, output_path: Path):
    grouped = rows_by_dataset_bucket(rows)
    x = np.arange(len(BUCKET_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    for dataset in DATASETS:
        pred_values = []
        pose_values = []
        error_values = []
        counts = []
        for bucket in BUCKET_ORDER:
            row = grouped.get(dataset, {}).get(bucket)
            pred_values.append(row.get("final_goal_dist_pred_mean", np.nan) if row else np.nan)
            pose_values.append(row.get("goal_pos_dist_mean", np.nan) if row else np.nan)
            error_values.append(
                row.get("final_goal_dist_error_mean", np.nan) if row else np.nan
            )
            counts.append(int(row.get("count", 0)) if row else 0)
        axes[0].plot(
            x,
            pred_values,
            marker="o",
            linewidth=2.0,
            label=f"{dataset} dist_pred",
        )
        axes[0].plot(
            x,
            pose_values,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label=f"{dataset} goal pose",
        )
        axes[1].plot(
            x,
            error_values,
            marker="o",
            linewidth=2.0,
            label=dataset,
        )
        for xi, yi, count in zip(x, error_values, counts):
            if np.isfinite(float(yi)):
                axes[1].annotate(
                    f"n={count}",
                    xy=(xi, yi),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    axes[0].set_title("Mean final-goal dist_pred vs true goal distance")
    axes[0].set_ylabel("mean distance")
    axes[1].set_title("Mean dist_pred - true goal distance")
    axes[1].set_ylabel("mean error")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_ORDER)
        ax.set_xlabel("local goal-distance bucket")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Dist head final-goal distance calibration by goal-distance bucket", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary_montages(root: Path, summary_dir: Path, swap_log_root: Path):
    fig1 = summary_dir / "fig_topomap_subgoal_vs_goal_pose.png"
    fig2 = summary_dir / "fig_flow_endpoint_vs_topomap_subgoal.png"
    fig3 = summary_dir / "fig_flow_endpoint_vs_goal_pose.png"
    fig4 = summary_dir / "fig_endpoint_mmd_emd_by_goal_distance.png"
    compose_panel_montage(
        root,
        SUBGOAL_GOAL_PANEL,
        fig1,
        "Topomap subgoal selection vs final goal pose by metric goal distance",
    )
    compose_panel_montage(
        root,
        FLOW_SUBGOAL_PANEL,
        fig2,
        "Flow endpoint means vs selected topomap subgoals by metric goal distance",
    )
    compose_goal_swap_montage(swap_log_root, fig3)
    endpoint_rows = load_endpoint_mmd_emd_rows(swap_log_root)
    write_csv(summary_dir / "endpoint_mmd_emd_by_goal_distance.csv", endpoint_rows)
    plot_endpoint_mmd_emd_by_bucket(endpoint_rows, fig4)
    return fig1, fig2, fig3, fig4


def main():
    args = parse_args()
    root = Path(args.output_dir) / "topomap_subgoal_analysis"
    summary_dir = root / VARIANT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if args.montage_only:
        fig1, fig2, fig3, fig4 = write_summary_montages(
            root,
            summary_dir,
            Path(args.swap_log_root),
        )
        print(fig1)
        print(fig2)
        print(fig3)
        print(fig4)
        return

    torch.manual_seed(0)
    np.random.seed(0)
    config = load_config(args.config)
    device = get_device(args.device)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    log(f"Loading model checkpoint: {checkpoint_path}")
    model = build_model(config, checkpoint_path, device)

    root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    rows_by_cell = {}
    for dataset_name in args.datasets:
        for bucket, dist_range in HORIZON_BUCKETS.items():
            rows = run_bucket(args, model, config, device, dataset_name, bucket, dist_range)
            rows_by_cell[(dataset_name, bucket)] = rows
            out_dir = root / bucket / VARIANT / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / timestamp_name("topomap_subgoal_items", "csv")
            json_path = out_dir / timestamp_name("topomap_subgoal_summary", "json")
            write_csv(csv_path, rows)
            write_json(
                json_path,
                {
                    "test": "topomap_subgoal_analysis",
                    "dataset": dataset_name,
                    "bucket": bucket,
                    "min_goal_pos_dist": dist_range[0],
                    "max_goal_pos_dist": dist_range[1],
                    "num_rows": len(rows),
                    "num_observations": len(
                        {
                            (
                                row["dataset_index"],
                                row["trajectory"],
                                row["curr_time"],
                            )
                            for row in rows
                        }
                    ),
                    "direction_counts": {
                        direction: sum(row["direction"] == direction for row in rows)
                        for direction in DIRECTIONS
                    },
                    "missing": len(rows) == 0,
                    "csv_path": str(csv_path),
                },
            )
            all_rows.extend(rows)
            plot_subgoal_vs_goal(
                rows,
                out_dir / SUBGOAL_GOAL_PANEL,
                args.global_max_points_per_direction,
            )
            log(f"Saved {dataset_name}/{bucket}: {csv_path}")

    all_csv = summary_dir / "topomap_subgoal_all_items.csv"
    write_csv(all_csv, all_rows)
    final_goal_dist_summary = summarize_final_goal_dist(all_rows)
    final_goal_dist_csv = summary_dir / "final_goal_dist_pred_vs_goal_pose.csv"
    write_csv(final_goal_dist_csv, final_goal_dist_summary)
    final_goal_dist_fig = summary_dir / "fig_final_goal_dist_pred_error_by_goal_distance.png"
    plot_final_goal_dist_error_by_bucket(final_goal_dist_summary, final_goal_dist_fig)
    flow_rows = load_flow_endpoint_means(Path(args.swap_log_root))
    flow_csv = summary_dir / "flow_endpoint_mean_items.csv"
    write_csv(flow_csv, flow_rows)
    for dataset_name in args.datasets:
        for bucket in BUCKET_ORDER:
            out_dir = root / bucket / VARIANT / dataset_name
            cell_flow_rows = [
                row for row in flow_rows
                if row["dataset"] == dataset_name and row["bucket"] == bucket
            ]
            plot_flow_vs_subgoal(
                rows_by_cell.get((dataset_name, bucket), []),
                cell_flow_rows,
                out_dir / FLOW_SUBGOAL_PANEL,
                args.global_max_points_per_direction,
            )
    fig1, fig2, fig3, fig4 = write_summary_montages(
        root,
        summary_dir,
        Path(args.swap_log_root),
    )
    counts = {
        "subgoal_rows": len(all_rows),
        "flow_endpoint_rows": len(flow_rows),
        "expected_dataset_bucket_cells": len(DATASETS) * len(BUCKET_ORDER),
        "nonempty_subgoal_cells": len(
            {
                (row["dataset"], row["bucket"])
                for row in all_rows
            }
        ),
        "missing_subgoal_cells": [
            {"dataset": dataset, "bucket": bucket}
            for dataset in DATASETS
            for bucket in BUCKET_ORDER
            if not any(
                row["dataset"] == dataset and row["bucket"] == bucket
                for row in all_rows
            )
        ],
    }
    write_json(summary_dir / "counts.json", counts)
    print(all_csv)
    print(final_goal_dist_csv)
    print(flow_csv)
    print(fig1)
    print(fig2)
    print(fig3)
    print(fig4)
    print(final_goal_dist_fig)
    print(summary_dir / "counts.json")


if __name__ == "__main__":
    main()
