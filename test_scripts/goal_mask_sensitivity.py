import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    angle_tag,
    build_dataloader,
    build_model,
    ensure_output_dir,
    find_all_matched_directional_goal_sets,
    get_device,
    imagenet_transform,
    load_config,
    log,
    resolve_checkpoint,
    run_model,
    timestamp_name,
    write_csv,
    write_json,
)
from goal_swap_visualization import (
    COLORS,
    DIRECTIONS,
    SELECTION_VARIANTS,
    apply_trajectory_selection,
    endpoint_pairwise_distribution_metrics,
    format_pair_metric_text,
    output_root_for_selection,
    plot_covariance_circle,
    rbf_mmd,
    sliced_wasserstein_distance,
)


MASKED_COLOR = "#6b7280"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare no-heading-filter left/forward/right goal-conditioned "
            "trajectories with goal-masked trajectories."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scan-batches", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--angle-threshold-deg", type=float, default=25.0)
    parser.add_argument("--min-goal-offset", type=int, default=None)
    parser.add_argument("--max-goal-offset", type=int, default=None)
    parser.add_argument("--max-direction-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-endpoint-goal-dist", type=float, default=None)
    parser.add_argument("--max-visualizations", type=int, default=None)
    parser.add_argument(
        "--trajectory-selection",
        choices=sorted(SELECTION_VARIANTS.keys()),
        default="baseline",
        help=(
            "Trajectory selection mode for metric/artifact generation. "
            "`baseline` keeps sampled trajectories; `cluster` selects the "
            "largest-cluster medoid per direction and condition."
        ),
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.35,
        help="Weighted trajectory distance threshold used by --trajectory-selection cluster.",
    )
    parser.add_argument("--global-endpoint-max-points-per-class", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def mask_run_tag(angle_threshold_deg: float) -> str:
    return f"{angle_tag(angle_threshold_deg)}-no_heading_filter"


def sample_goal_and_masked_trajectories(model, config, device, directional_set, num_samples):
    trajectories = {}
    for name in DIRECTIONS:
        candidate = directional_set["candidates"][name]
        outputs = run_model(
            model,
            directional_set["base_obs"],
            candidate["goal"],
            config["len_traj_pred"],
            num_samples,
            device,
        )
        trajectories[name] = {
            "goal": (
                outputs["gc_actions"]
                .reshape(num_samples, config["len_traj_pred"], 2)
                .detach()
                .cpu()
                .numpy()
            ),
            "masked": (
                outputs["uc_actions"]
                .reshape(num_samples, config["len_traj_pred"], 2)
                .detach()
                .cpu()
                .numpy()
            ),
        }
    return trajectories


def endpoint_metrics(goal_samples: np.ndarray, masked_samples: np.ndarray) -> dict:
    goal_endpoints = goal_samples[:, -1]
    masked_endpoints = masked_samples[:, -1]
    paired_count = min(len(goal_endpoints), len(masked_endpoints))
    paired_dist = np.linalg.norm(
        goal_samples[:paired_count] - masked_samples[:paired_count],
        axis=-1,
    )
    return {
        "endpoint_mean_l2": float(
            np.linalg.norm(goal_endpoints.mean(axis=0) - masked_endpoints.mean(axis=0))
        ),
        "endpoint_rbf_mmd": rbf_mmd(goal_endpoints, masked_endpoints),
        "endpoint_sliced_wasserstein": sliced_wasserstein_distance(
            goal_endpoints,
            masked_endpoints,
        ),
        "matched_sample_ade": float(paired_dist.mean()),
        "matched_sample_fde": float(paired_dist[:, -1].mean()),
    }


def aggregate_rows(rows: list[dict]) -> dict:
    numeric_keys = [
        "endpoint_mean_l2",
        "endpoint_rbf_mmd",
        "endpoint_sliced_wasserstein",
        "matched_sample_ade",
        "matched_sample_fde",
    ]
    summary = {}
    for key in numeric_keys:
        values = [row[key] for row in rows]
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"median_{key}"] = float(np.median(values))
    return summary


def global_shift_metrics(all_endpoints: dict) -> dict:
    per_direction = {}
    for name in DIRECTIONS:
        goal_points = np.asarray(all_endpoints["goal"][name], dtype=np.float32)
        masked_points = np.asarray(all_endpoints["masked"][name], dtype=np.float32)
        per_direction[name] = {
            "endpoint_rbf_mmd": rbf_mmd(goal_points, masked_points),
            "endpoint_sliced_wasserstein": sliced_wasserstein_distance(
                goal_points,
                masked_points,
            ),
            "endpoint_mean_l2": float(
                np.linalg.norm(goal_points.mean(axis=0) - masked_points.mean(axis=0))
            ),
            "goal_count": int(len(goal_points)),
            "masked_count": int(len(masked_points)),
        }
    return per_direction


def select_condition_trajectories(trajectories: dict, condition: str, args):
    condition_trajectories = {
        name: trajectories[name][condition] for name in DIRECTIONS
    }
    return apply_trajectory_selection(condition_trajectories, args)


def downsample_points(points: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def scatter_goal_masked_panel(
    ax,
    goal_points: np.ndarray,
    masked_points: np.ndarray,
    direction: str,
    max_points_per_class: int | None,
) -> dict:
    goal_display = downsample_points(goal_points, max_points_per_class)
    masked_display = downsample_points(masked_points, max_points_per_class)
    color = COLORS[direction]
    ax.scatter([0.0], [0.0], c="black", s=45, label="robot")
    ax.quiver(
        [0.0],
        [0.0],
        [0.6],
        [0.0],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.005,
    )
    ax.scatter(
        goal_display[:, 0],
        goal_display[:, 1],
        c=color,
        alpha=0.22,
        s=14,
        label=f"with {direction} goal ({len(goal_display)}/{len(goal_points)})",
    )
    ax.scatter(
        masked_display[:, 0],
        masked_display[:, 1],
        c=MASKED_COLOR,
        alpha=0.18,
        s=14,
        marker="x",
        label=f"goal masked ({len(masked_display)}/{len(masked_points)})",
    )
    plot_covariance_circle(ax, goal_points, color=color, edge_color=color)
    plot_covariance_circle(
        ax,
        masked_points,
        color=MASKED_COLOR,
        edge_color="#374151",
        fill_alpha=0.035,
    )
    metrics = {
        "endpoint_rbf_mmd": rbf_mmd(goal_points, masked_points),
        "endpoint_sliced_wasserstein": sliced_wasserstein_distance(
            goal_points,
            masked_points,
        ),
        "endpoint_mean_l2": float(
            np.linalg.norm(goal_points.mean(axis=0) - masked_points.mean(axis=0))
        ),
    }
    ax.text(
        0.02,
        0.98,
        (
            f"MMD={metrics['endpoint_rbf_mmd']:.4f}\n"
            f"EMD~={metrics['endpoint_sliced_wasserstein']:.4f}\n"
            f"meanL2={metrics['endpoint_mean_l2']:.4f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    ax.set_title(f"{direction}: with goal vs masked")
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7)
    return metrics


def plot_endpoint_shift_by_direction(
    all_endpoints: dict,
    output_path: Path,
    max_points_per_class: int,
) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.2))
    metrics = {}
    for ax, name in zip(axes, DIRECTIONS):
        metrics[name] = scatter_goal_masked_panel(
            ax,
            np.asarray(all_endpoints["goal"][name], dtype=np.float32),
            np.asarray(all_endpoints["masked"][name], dtype=np.float32),
            name,
            max_points_per_class=max_points_per_class,
        )
    fig.suptitle("Goal mask sensitivity: endpoint shift by direction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return metrics


def plot_direction_distribution_comparison(
    all_endpoints: dict,
    all_goals: dict,
    output_path: Path,
    max_points_per_class: int,
) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(22.0, 6.4))
    for name in DIRECTIONS:
        goal_points = np.asarray(all_endpoints["goal"][name], dtype=np.float32)
        display = downsample_points(goal_points, max_points_per_class)
        axes[0].scatter(
            display[:, 0],
            display[:, 1],
            c=COLORS[name],
            alpha=0.18,
            s=12,
            label=f"{name} ({len(display)}/{len(goal_points)})",
        )
        plot_covariance_circle(axes[0], goal_points, COLORS[name], COLORS[name])

        masked_points = np.asarray(all_endpoints["masked"][name], dtype=np.float32)
        display_masked = downsample_points(masked_points, max_points_per_class)
        axes[1].scatter(
            display_masked[:, 0],
            display_masked[:, 1],
            c=COLORS[name],
            alpha=0.18,
            s=12,
            label=f"{name} request ({len(display_masked)}/{len(masked_points)})",
        )
        plot_covariance_circle(axes[1], masked_points, COLORS[name], COLORS[name])

        goal_pos = np.asarray(all_goals[name], dtype=np.float32)
        display_goals = downsample_points(goal_pos, max_points_per_class)
        axes[2].scatter(
            display_goals[:, 0],
            display_goals[:, 1],
            c=COLORS[name],
            alpha=0.35,
            s=20,
            label=f"{name} goals ({len(display_goals)}/{len(goal_pos)})",
        )
        plot_covariance_circle(axes[2], goal_pos, COLORS[name], COLORS[name])

    goal_pair_metrics = endpoint_pairwise_distribution_metrics(all_endpoints["goal"])
    masked_pair_metrics = endpoint_pairwise_distribution_metrics(all_endpoints["masked"])
    axes[0].text(
        0.02,
        0.98,
        format_pair_metric_text(goal_pair_metrics),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    axes[1].text(
        0.02,
        0.98,
        format_pair_metric_text(masked_pair_metrics),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    titles = [
        "With goal: endpoint distributions",
        "Masked goal: grouped by requested goal",
        "Matched goal positions",
    ]
    for ax, title in zip(axes, titles):
        ax.scatter([0.0], [0.0], c="black", s=45, label="robot")
        ax.quiver(
            [0.0],
            [0.0],
            [0.6],
            [0.0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.005,
        )
        ax.set_title(title)
        ax.set_xlabel("local x")
        ax.set_ylabel("local y")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {
        "goal_direction_pair_metrics": goal_pair_metrics,
        "masked_direction_pair_metrics": masked_pair_metrics,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(
        "Starting goal_mask_sensitivity "
        f"dataset={args.dataset} split={args.split} scan_batches={args.scan_batches} "
        f"batch_size={args.batch_size} num_samples={args.num_samples}"
    )
    config = load_config(args.config)
    device = get_device(args.device)
    log(f"Using device: {device}")
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    log(f"Loading model checkpoint: {checkpoint_path}")
    model = build_model(config, checkpoint_path, device)
    log("Model loaded.")
    dataloader = build_dataloader(
        config=config,
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=True,
    )
    transform = imagenet_transform()
    directional_sets = find_all_matched_directional_goal_sets(
        dataset=dataloader.dataset,
        transform=transform,
        device=device,
        angle_threshold_deg=args.angle_threshold_deg,
        scan_items=args.scan_batches * args.batch_size,
        min_goal_offset=args.min_goal_offset,
        max_goal_offset=args.max_goal_offset,
        max_direction_angle_deg=args.max_direction_angle_deg,
        max_endpoint_goal_dist=args.max_endpoint_goal_dist,
        filter_goal_heading=False,
    )
    if args.max_visualizations is not None:
        directional_sets = directional_sets[: args.max_visualizations]

    selection_output_root = output_root_for_selection(
        args.output_dir,
        args.trajectory_selection,
    )
    output_dir = ensure_output_dir(
        selection_output_root,
        args.dataset,
        "goal_mask_sensitivity",
        mask_run_tag(args.angle_threshold_deg),
    )
    log(f"Writing outputs to: {output_dir}")

    rows = []
    all_endpoints = {
        "goal": {name: [] for name in DIRECTIONS},
        "masked": {name: [] for name in DIRECTIONS},
        "matched_goals": {name: [] for name in DIRECTIONS},
    }
    for item_idx, directional_set in enumerate(directional_sets):
        source = directional_set["base_source"]
        log(
            f"Evaluating mask sensitivity {item_idx + 1}/{len(directional_sets)}: "
            f"dataset_index={source['dataset_index']}, "
            f"trajectory={source['trajectory']}, curr_time={source['curr_time']}"
        )
        trajectories = sample_goal_and_masked_trajectories(
            model=model,
            config=config,
            device=device,
            directional_set=directional_set,
            num_samples=args.num_samples,
        )
        goal_trajectories, goal_selection_info = select_condition_trajectories(
            trajectories,
            "goal",
            args,
        )
        masked_trajectories, masked_selection_info = select_condition_trajectories(
            trajectories,
            "masked",
            args,
        )
        for name in DIRECTIONS:
            metrics = endpoint_metrics(
                goal_trajectories[name],
                masked_trajectories[name],
            )
            rows.append(
                {
                    "matched_index": item_idx,
                    "direction": name,
                    "dataset_index": source["dataset_index"],
                    "trajectory": source["trajectory"],
                    "curr_time": source["curr_time"],
                    "goal_time": directional_set["candidates"][name]["goal_time"],
                    "goal_angle_deg": directional_set["candidates"][name][
                        "goal_angle_deg"
                    ],
                    "num_trajectories": args.num_samples,
                    "trajectory_selection": args.trajectory_selection,
                    "cluster_threshold": (
                        args.cluster_threshold
                        if args.trajectory_selection == "cluster"
                        else None
                    ),
                    "goal_selection_info": goal_selection_info[name],
                    "masked_selection_info": masked_selection_info[name],
                    **metrics,
                }
            )
            all_endpoints["goal"][name].extend(
                goal_trajectories[name][:, -1].astype(float).tolist()
            )
            all_endpoints["masked"][name].extend(
                masked_trajectories[name][:, -1].astype(float).tolist()
            )
            all_endpoints["matched_goals"][name].append(
                np.asarray(
                    directional_set["candidates"][name]["goal_pos"][:2],
                    dtype=float,
                ).tolist()
            )

    json_path = output_dir / timestamp_name("goal_mask_sensitivity_summary", "json")
    csv_path = output_dir / timestamp_name("goal_mask_sensitivity_items", "csv")
    endpoint_npz_path = output_dir / timestamp_name(
        "goal_mask_sensitivity_endpoints",
        "npz",
    )
    endpoint_arrays = {}
    for name in DIRECTIONS:
        endpoint_arrays[f"goal_{name}"] = np.asarray(
            all_endpoints["goal"][name],
            dtype=np.float32,
        )
        endpoint_arrays[f"masked_{name}"] = np.asarray(
            all_endpoints["masked"][name],
            dtype=np.float32,
        )
        endpoint_arrays[f"matched_goals_{name}"] = np.asarray(
            all_endpoints["matched_goals"][name],
            dtype=np.float32,
        )
    np.savez_compressed(endpoint_npz_path, **endpoint_arrays)
    endpoint_shift_path = output_dir / timestamp_name(
        "goal_mask_endpoint_shift_by_direction",
        "png",
    )
    endpoint_shift_metrics = plot_endpoint_shift_by_direction(
        all_endpoints,
        endpoint_shift_path,
        max_points_per_class=args.global_endpoint_max_points_per_class,
    )
    direction_distribution_path = output_dir / timestamp_name(
        "goal_mask_direction_distribution_comparison",
        "png",
    )
    direction_distribution_metrics = plot_direction_distribution_comparison(
        all_endpoints,
        all_endpoints["matched_goals"],
        direction_distribution_path,
        max_points_per_class=args.global_endpoint_max_points_per_class,
    )

    summary = {
        "test": "goal_mask_sensitivity",
        "trajectory_selection": args.trajectory_selection,
        "output_variant": SELECTION_VARIANTS[args.trajectory_selection],
        "cluster_threshold": (
            args.cluster_threshold if args.trajectory_selection == "cluster" else None
        ),
        "config": args.config,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "angle_threshold_deg": args.angle_threshold_deg,
        "scan_batches": args.scan_batches,
        "scan_items": args.scan_batches * args.batch_size,
        "num_matched_sets": len(directional_sets),
        "num_items": len(rows),
        "num_samples": args.num_samples,
        "goal_matching": "same_trajectory_same_curr_time",
        "direction_source": "goal_pos",
        "filter_goal_heading": False,
        "min_goal_offset": args.min_goal_offset,
        "max_goal_offset": args.max_goal_offset,
        "max_direction_angle_deg": args.max_direction_angle_deg,
        "max_endpoint_goal_dist": args.max_endpoint_goal_dist,
        "global_goal_vs_masked_metrics": global_shift_metrics(all_endpoints),
        "global_goal_direction_pair_metrics": endpoint_pairwise_distribution_metrics(
            all_endpoints["goal"]
        ),
        "global_masked_direction_pair_metrics": endpoint_pairwise_distribution_metrics(
            all_endpoints["masked"]
        ),
        "endpoint_npz_path": str(endpoint_npz_path),
        "endpoint_shift_path": str(endpoint_shift_path),
        "direction_distribution_path": str(direction_distribution_path),
        "endpoint_shift_metrics": endpoint_shift_metrics,
        "direction_distribution_metrics": direction_distribution_metrics,
        "metrics": rows,
        **aggregate_rows(rows),
    }
    write_json(json_path, summary)
    write_csv(csv_path, rows)

    print(f"Saved summary: {json_path}")
    print(f"Saved per-item metrics: {csv_path}")
    print(f"Saved endpoint artifact: {endpoint_npz_path}")
    print(f"Saved endpoint shift figure: {endpoint_shift_path}")
    print(f"Saved direction distribution figure: {direction_distribution_path}")


if __name__ == "__main__":
    main()
