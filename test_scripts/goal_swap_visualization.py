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
    plot_directional_goal_samples,
    resolve_checkpoint,
    run_model,
    timestamp_name,
    write_json,
)
try:
    from flownav.training.utils import cluster_trajectory_samples
except ImportError:
    cluster_trajectory_samples = None


DIRECTIONS = ("left", "forward", "right")
PAIRS = (("left", "forward"), ("left", "right"), ("forward", "right"))
COLORS = {"left": "tab:blue", "forward": "tab:green", "right": "tab:red"}
ELLIPSE_COLORS = {"left": "navy", "forward": "darkgreen", "right": "darkred"}
NO_HEADING_COLORS = {"left": "#8a6f00", "forward": "#7a4d8f", "right": "#9b4d2e"}
SELECTION_VARIANTS = {
    "baseline": "flownav_baseline",
    "cluster": "flownav_cluster",
}


def use_metric_goal_pos(args) -> bool:
    return args.min_goal_pos_dist is not None or args.max_goal_pos_dist is not None


def is_horizon_filtered_run(args) -> bool:
    return any(
        value is not None
        for value in (
            args.min_goal_offset,
            args.max_goal_offset,
            args.min_goal_pos_dist,
            args.max_goal_pos_dist,
        )
    )


def candidate_goal_pos_for_plot(candidate, metric_goal_pos: bool = False):
    if metric_goal_pos:
        return np.asarray(
            candidate.get("goal_pos_metric", candidate["goal_pos"])[:2],
            dtype=np.float32,
        )
    return np.asarray(
        candidate["goal_pos"][:2],
        dtype=np.float32,
    )


def value_tag(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def swap_run_tag(angle_threshold_deg: float, mmd_threshold: float, emd_threshold: float) -> str:
    return (
        f"{angle_tag(angle_threshold_deg)}-"
        f"mmd{value_tag(mmd_threshold)}-"
        f"emd{value_tag(emd_threshold)}"
    )


def heading_filter_tag(enabled: bool) -> str:
    return "heading_filter" if enabled else "no_heading_filter"


def selected_heading_filter_values(mode: str) -> list[bool]:
    if mode == "both":
        return [True, False]
    if mode == "heading_filter":
        return [True]
    if mode == "no_heading_filter":
        return [False]
    raise ValueError(f"Unsupported heading filter mode: {mode}")


def available_directions_from_candidates(directional_set) -> tuple[str, ...]:
    return tuple(name for name in DIRECTIONS if name in directional_set["candidates"])


def available_pairs(direction_names) -> list[tuple[str, str]]:
    available = set(direction_names)
    return [(first, second) for first, second in PAIRS if first in available and second in available]


def angle_diff_deg(first: float, second: float) -> float:
    diff = np.arctan2(np.sin(np.deg2rad(first - second)), np.cos(np.deg2rad(first - second)))
    return float(abs(np.rad2deg(diff)))


def classify_angle(angle_deg: float, threshold_deg: float) -> str:
    if angle_deg > threshold_deg:
        return "left"
    if angle_deg < -threshold_deg:
        return "right"
    return "forward"


def gaussian_symmetric_kl(points_a: np.ndarray, points_b: np.ndarray, eps: float) -> float:
    dim = points_a.shape[1]
    mean_a = points_a.mean(axis=0)
    mean_b = points_b.mean(axis=0)
    cov_a = np.cov(points_a.T) + eps * np.eye(dim)
    cov_b = np.cov(points_b.T) + eps * np.eye(dim)
    inv_a = np.linalg.inv(cov_a)
    inv_b = np.linalg.inv(cov_b)

    def kl(mean_p, cov_p, mean_q, cov_q, inv_q):
        delta = mean_q - mean_p
        return 0.5 * (
            np.trace(inv_q @ cov_p)
            + delta.T @ inv_q @ delta
            - dim
            + np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
        )

    return float(
        0.5
        * (
            kl(mean_a, cov_a, mean_b, cov_b, inv_b)
            + kl(mean_b, cov_b, mean_a, cov_a, inv_a)
        )
    )


def rbf_mmd(points_a: np.ndarray, points_b: np.ndarray, sigma: float | None = None) -> float:
    combined = np.concatenate([points_a, points_b], axis=0)
    if sigma is None:
        pairwise = np.linalg.norm(combined[:, None] - combined[None], axis=-1)
        nonzero = pairwise[pairwise > 0]
        sigma = float(np.median(nonzero)) if len(nonzero) else 1.0
    sigma = max(float(sigma), 1e-6)

    def kernel(x, y):
        dist_sq = np.sum((x[:, None] - y[None]) ** 2, axis=-1)
        return np.exp(-dist_sq / (2.0 * sigma**2))

    return float(
        kernel(points_a, points_a).mean()
        + kernel(points_b, points_b).mean()
        - 2.0 * kernel(points_a, points_b).mean()
    )


def sliced_wasserstein_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    num_projections: int = 64,
) -> float:
    rng = np.random.default_rng(0)
    projections = rng.normal(size=(num_projections, points_a.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    distances = []
    for projection in projections:
        projected_a = np.sort(points_a @ projection)
        projected_b = np.sort(points_b @ projection)
        count = min(len(projected_a), len(projected_b))
        distances.append(np.abs(projected_a[:count] - projected_b[:count]).mean())
    return float(np.mean(distances))


def dtw_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    len_a, len_b = len(traj_a), len(traj_b)
    dp = np.full((len_a + 1, len_b + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = np.linalg.norm(traj_a[i - 1] - traj_b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[len_a, len_b])


def discrete_frechet_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    len_a, len_b = len(traj_a), len(traj_b)
    cache = np.full((len_a, len_b), -1.0)

    def recurse(i: int, j: int) -> float:
        if cache[i, j] >= 0:
            return cache[i, j]
        dist = np.linalg.norm(traj_a[i] - traj_b[j])
        if i == 0 and j == 0:
            value = dist
        elif i > 0 and j == 0:
            value = max(recurse(i - 1, 0), dist)
        elif i == 0 and j > 0:
            value = max(recurse(0, j - 1), dist)
        else:
            value = max(
                min(recurse(i - 1, j), recurse(i - 1, j - 1), recurse(i, j - 1)),
                dist,
            )
        cache[i, j] = value
        return value

    return float(recurse(len_a - 1, len_b - 1))


def sample_directional_trajectories(model, config, device, directional_set, num_samples):
    trajectories = {}
    for name in available_directions_from_candidates(directional_set):
        candidate = directional_set["candidates"][name]
        outputs = run_model(
            model,
            directional_set["base_obs"],
            candidate["goal"],
            config["len_traj_pred"],
            num_samples,
            device,
        )
        trajectories[name] = (
            outputs["gc_actions"]
            .reshape(num_samples, config["len_traj_pred"], 2)
            .detach()
            .cpu()
            .numpy()
        )
    return trajectories


def output_root_for_selection(output_dir: str, trajectory_selection: str) -> str:
    root = Path(output_dir)
    variant = SELECTION_VARIANTS[trajectory_selection]
    if root.name in set(SELECTION_VARIANTS.values()):
        return str(root)
    return str(root / variant)


def select_clustered_trajectories(trajectories, cluster_threshold: float):
    if cluster_trajectory_samples is None:
        raise ImportError(
            "cluster trajectory selection requires "
            "flownav.training.utils.cluster_trajectory_samples, but it is not "
            "available in this checkout. Use --trajectory-selection baseline."
        )
    selected = {}
    selection_info = {}
    for name, samples in trajectories.items():
        cluster_info = cluster_trajectory_samples(
            samples,
            distance_threshold=cluster_threshold,
        )
        selected[name] = np.repeat(
            cluster_info["selected_trajectory"][None],
            repeats=len(samples),
            axis=0,
        )
        selection_info[name] = {
            "selected_index": int(cluster_info["selected_index"]),
            "selected_cluster_size": int(len(cluster_info["selected_cluster"])),
            "num_samples": int(len(samples)),
            "num_clusters": int(len(cluster_info["clusters"])),
            "cluster_sizes": [int(len(cluster)) for cluster in cluster_info["clusters"]],
        }
    return selected, selection_info


def apply_trajectory_selection(trajectories, args):
    if args.trajectory_selection == "baseline":
        return trajectories, {
            name: {
                "selected_index": 0,
                "selected_cluster_size": None,
                "num_samples": int(len(samples)),
                "num_clusters": None,
                "cluster_sizes": None,
            }
            for name, samples in trajectories.items()
        }
    if args.trajectory_selection == "cluster":
        return select_clustered_trajectories(
            trajectories,
            cluster_threshold=args.cluster_threshold,
        )
    raise ValueError(f"Unsupported trajectory selection: {args.trajectory_selection}")


def compute_sensitivity_metrics(
    trajectories,
    directional_set,
    angle_threshold_deg,
    eps,
    metric_goal_pos: bool = False,
):
    direction_names = tuple(name for name in DIRECTIONS if name in trajectories)
    pair_names = available_pairs(direction_names)
    endpoints = {name: trajectories[name][:, -1] for name in direction_names}
    endpoint_means = {name: endpoints[name].mean(axis=0) for name in direction_names}
    goal_pos_by_direction = {
        name: candidate_goal_pos_for_plot(
            directional_set["candidates"][name],
            metric_goal_pos=metric_goal_pos,
        )
        for name in direction_names
    }
    mean_trajs = {name: trajectories[name].mean(axis=0) for name in direction_names}
    final_headings = {
        name: np.rad2deg(
            np.arctan2(trajectories[name][:, -1, 1], trajectories[name][:, -1, 0])
        )
        for name in direction_names
    }
    mean_headings = {
        name: float(
            np.rad2deg(
                np.arctan2(
                    np.sin(np.deg2rad(final_headings[name])).mean(),
                    np.cos(np.deg2rad(final_headings[name])).mean(),
                )
            )
        )
        for name in direction_names
    }
    class_probs = {}
    for name in direction_names:
        labels = [classify_angle(angle, angle_threshold_deg) for angle in final_headings[name]]
        class_probs[name] = {label: labels.count(label) / len(labels) for label in DIRECTIONS}

    pair_metrics = {}
    for first, second in pair_names:
        key = f"{first}_{second}"
        goal_a = goal_pos_by_direction[first]
        goal_b = goal_pos_by_direction[second]
        goal_dist = float(np.linalg.norm(goal_a - goal_b))
        endpoint_mean_dist = float(
            np.linalg.norm(endpoint_means[first] - endpoint_means[second])
        )
        endpoint_delta = endpoint_means[second] - endpoint_means[first]
        goal_delta = goal_b - goal_a
        endpoint_delta_norm = float(np.linalg.norm(endpoint_delta))
        goal_delta_norm = float(np.linalg.norm(goal_delta))
        goal_direction_alignment = float(
            np.dot(endpoint_delta, goal_delta)
            / max(endpoint_delta_norm * goal_delta_norm, eps)
        )
        paired_count = min(len(endpoints[first]), len(endpoints[second]))
        endpoint_displacement = float(
            np.linalg.norm(
                endpoints[first][:paired_count] - endpoints[second][:paired_count],
                axis=1,
            ).mean()
        )
        tv_distance = 0.5 * sum(
            abs(class_probs[first][label] - class_probs[second][label])
            for label in DIRECTIONS
        )
        pair_metrics[key] = {
            "endpoint_mean_distance": endpoint_mean_dist,
            "goal_pos_distance": goal_dist,
            "s_goal": endpoint_mean_dist / max(goal_dist, eps),
            "flow_goal_direction_alignment": goal_direction_alignment,
            "endpoint_displacement_difference": endpoint_displacement,
            "heading_diff_deg": angle_diff_deg(mean_headings[first], mean_headings[second]),
            "class_tv_distance": float(tv_distance),
            "endpoint_symmetric_kl": gaussian_symmetric_kl(
                endpoints[first], endpoints[second], eps
            ),
            "endpoint_rbf_mmd": rbf_mmd(endpoints[first], endpoints[second]),
            "endpoint_sliced_wasserstein": sliced_wasserstein_distance(
                endpoints[first], endpoints[second]
            ),
            "mean_traj_dtw": dtw_distance(mean_trajs[first], mean_trajs[second]),
            "mean_traj_frechet": discrete_frechet_distance(
                mean_trajs[first], mean_trajs[second]
            ),
        }

    aggregate_keys = [
        "endpoint_mean_distance",
        "s_goal",
        "flow_goal_direction_alignment",
        "endpoint_displacement_difference",
        "heading_diff_deg",
        "class_tv_distance",
        "endpoint_symmetric_kl",
        "endpoint_rbf_mmd",
        "endpoint_sliced_wasserstein",
        "mean_traj_dtw",
        "mean_traj_frechet",
    ]
    if pair_metrics:
        aggregate = {
            f"mean_{key}": float(np.mean([pair_metrics[pair][key] for pair in pair_metrics]))
            for key in aggregate_keys
        }
        aggregate.update(
            {
                f"min_{key}": float(
                    np.min([pair_metrics[pair][key] for pair in pair_metrics])
                )
                for key in aggregate_keys
            }
        )
    else:
        aggregate = {f"mean_{key}": None for key in aggregate_keys}
        aggregate.update({f"min_{key}": None for key in aggregate_keys})

    return {
        "endpoint_means": {
            name: endpoint_means[name].astype(float).tolist() for name in direction_names
        },
        "goal_pos_by_direction": {
            name: goal_pos_by_direction[name].astype(float).tolist()
            for name in direction_names
        },
        "available_directions": list(direction_names),
        "available_pairs": list(pair_metrics.keys()),
        "flow_endpoint_pair_distance": aggregate["mean_endpoint_mean_distance"],
        "flow_goal_direction_alignment": aggregate["mean_flow_goal_direction_alignment"],
        "mean_headings_deg": mean_headings,
        "class_probs": class_probs,
        "pairs": pair_metrics,
        **aggregate,
    }


def is_anomaly(metrics, args) -> bool:
    if metrics.get("mean_endpoint_rbf_mmd") is None:
        return False
    return (
        metrics["mean_endpoint_rbf_mmd"] <= args.anomaly_mmd_threshold
        or metrics["mean_endpoint_sliced_wasserstein"] <= args.anomaly_emd_threshold
    )


def plot_covariance_circle(
    ax,
    points: np.ndarray,
    color: str,
    edge_color: str | None = None,
    quantile: float = 0.95,
    fill_alpha: float = 0.06,
) -> None:
    mean = points.mean(axis=0)
    if len(points) < 2:
        ax.scatter([mean[0]], [mean[1]], c=color, marker="x", s=80)
        return
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    centered = points - mean[None]
    inv_cov = np.linalg.pinv(cov + 1e-8 * np.eye(cov.shape[0]))
    mahalanobis = np.sqrt(np.sum((centered @ inv_cov) * centered, axis=1))
    radius = float(np.quantile(mahalanobis, quantile))
    theta = np.linspace(0, 2 * np.pi, 160)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    ellipse = eigvecs @ (
        radius * np.sqrt(np.maximum(eigvals, 1e-8))[:, None] * circle
    )
    ellipse = ellipse.T + mean[None]
    edge_color = edge_color or color
    ax.fill(ellipse[:, 0], ellipse[:, 1], color=color, alpha=fill_alpha)
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=edge_color, linewidth=2.4)
    ax.scatter([mean[0]], [mean[1]], c=color, marker="x", s=90, linewidths=2.0)


def endpoint_pairwise_distribution_metrics(all_endpoints, max_points_per_class: int | None = None):
    pair_metrics = {}
    for first, second in PAIRS:
        raw_points_a = np.asarray(all_endpoints[first], dtype=np.float32)
        raw_points_b = np.asarray(all_endpoints[second], dtype=np.float32)
        points_a = downsample_points(raw_points_a, max_points_per_class)
        points_b = downsample_points(raw_points_b, max_points_per_class)
        key = f"{first}_{second}"
        if len(points_a) and len(points_b):
            pair_metrics[key] = {
                "endpoint_rbf_mmd": rbf_mmd(points_a, points_b),
                "endpoint_sliced_wasserstein": sliced_wasserstein_distance(
                    points_a, points_b
                ),
                "first_count": int(len(raw_points_a)),
                "second_count": int(len(raw_points_b)),
                "first_metric_count": int(len(points_a)),
                "second_metric_count": int(len(points_b)),
            }
        else:
            pair_metrics[key] = {
                "endpoint_rbf_mmd": None,
                "endpoint_sliced_wasserstein": None,
                "first_count": int(len(raw_points_a)),
                "second_count": int(len(raw_points_b)),
                "first_metric_count": int(len(points_a)),
                "second_metric_count": int(len(points_b)),
            }
    return pair_metrics


def format_pair_metric_text(pair_metrics):
    lines = []
    for pair_name, metrics in pair_metrics.items():
        mmd = metrics["endpoint_rbf_mmd"]
        emd = metrics["endpoint_sliced_wasserstein"]
        if mmd is None or emd is None:
            lines.append(f"{pair_name}: MMD=n/a, EMD~=n/a")
        else:
            lines.append(f"{pair_name}: MMD={mmd:.4f}, EMD~={emd:.4f}")
    return "\n".join(lines)


def scatter_distribution_panel(
    ax,
    distributions,
    title: str,
    point_label: str,
    max_points_per_class: int | None = None,
):
    ax.scatter([0.0], [0.0], c="black", s=50, label="robot")
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
    for name in DIRECTIONS:
        points = np.asarray(distributions[name], dtype=np.float32)
        if len(points) == 0:
            continue
        display_points = downsample_points(points, max_points_per_class)
        color = COLORS[name]
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            c=color,
            alpha=0.18,
            s=12,
            label=f"{name} {point_label} ({len(display_points)}/{len(points)})",
        )
        plot_covariance_circle(
            ax,
            points,
            color=color,
            edge_color=ELLIPSE_COLORS[name],
            quantile=0.98,
            fill_alpha=0.045,
        )
    ax.set_title(title)
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)


def downsample_points(points: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def plot_endpoint_distribution(
    trajectories,
    directional_set,
    metrics,
    output_path,
    metric_goal_pos: bool = False,
):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter([0.0], [0.0], c="black", s=50, label="robot")
    ax.quiver(
        [0.0],
        [0.0],
        [0.6],
        [0.0],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.006,
    )

    for name in tuple(name for name in DIRECTIONS if name in trajectories):
        endpoints = trajectories[name][:, -1]
        goal_pos = candidate_goal_pos_for_plot(
            directional_set["candidates"][name],
            metric_goal_pos=metric_goal_pos,
        )
        color = COLORS[name]
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            c=color,
            alpha=0.65,
            s=38,
            label=f"{name} endpoints",
        )
        plot_covariance_circle(ax, endpoints, color)
        ax.scatter(
            [goal_pos[0]],
            [goal_pos[1]],
            c=color,
            marker="*",
            s=150,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            f"{name} goal",
            xy=(goal_pos[0], goal_pos[1]),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=8,
        )

    pair_text = []
    for pair_name, pair_metrics in metrics["pairs"].items():
        pair_text.append(
            f"{pair_name}: MMD={pair_metrics['endpoint_rbf_mmd']:.3f}, "
            f"SWD={pair_metrics['endpoint_sliced_wasserstein']:.3f}"
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(pair_text),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    ax.set_title("Sampled Endpoint Distributions by Goal")
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_global_endpoint_distribution(
    all_endpoints,
    all_goals,
    output_path,
    max_points_per_class: int | None = None,
    max_metric_points_per_class: int | None = None,
):
    endpoint_pair_metrics = endpoint_pairwise_distribution_metrics(
        all_endpoints,
        max_points_per_class=max_metric_points_per_class,
    )
    goal_pair_metrics = endpoint_pairwise_distribution_metrics(
        all_goals,
        max_points_per_class=max_metric_points_per_class,
    )
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.0))
    scatter_distribution_panel(
        axes[0],
        all_endpoints,
        "Global Sampled Endpoint Distributions by Goal",
        "endpoints",
        max_points_per_class=max_points_per_class,
    )
    scatter_distribution_panel(
        axes[1],
        all_goals,
        "Matched Goal Position Distributions",
        "goals",
        max_points_per_class=max_points_per_class,
    )
    axes[0].text(
        0.02,
        0.98,
        format_pair_metric_text(endpoint_pair_metrics),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )
    axes[1].text(
        0.02,
        0.98,
        format_pair_metric_text(goal_pair_metrics),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {
        "endpoint_pair_metrics": endpoint_pair_metrics,
        "goal_pair_metrics": goal_pair_metrics,
    }


def plot_heading_filter_comparison(
    results_by_tag,
    output_path,
    max_points_per_class: int | None = None,
    max_metric_points_per_class: int | None = None,
):
    filtered = results_by_tag["heading_filter"]["all_endpoints"]
    unfiltered = results_by_tag["no_heading_filter"]["all_endpoints"]
    comparison = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, name in zip(axes, DIRECTIONS):
        filtered_points = np.asarray(filtered[name], dtype=np.float32)
        unfiltered_points = np.asarray(unfiltered[name], dtype=np.float32)
        color = COLORS[name]
        edge_color = ELLIPSE_COLORS[name]
        unfiltered_color = NO_HEADING_COLORS[name]

        if len(filtered_points):
            filtered_display = downsample_points(filtered_points, max_points_per_class)
            ax.scatter(
                filtered_display[:, 0],
                filtered_display[:, 1],
                c=color,
                alpha=0.30,
                s=14,
                marker="o",
                label=f"heading_filter ({len(filtered_display)}/{len(filtered_points)})",
            )
            plot_covariance_circle(
                ax,
                filtered_points,
                color=color,
                edge_color=edge_color,
                quantile=0.98,
                fill_alpha=0.035,
            )
        if len(unfiltered_points):
            unfiltered_display = downsample_points(
                unfiltered_points, max_points_per_class
            )
            ax.scatter(
                unfiltered_display[:, 0],
                unfiltered_display[:, 1],
                c=unfiltered_color,
                alpha=0.28,
                s=18,
                marker="^",
                label=(
                    f"no_heading_filter "
                    f"({len(unfiltered_display)}/{len(unfiltered_points)})"
                ),
            )
            plot_covariance_circle(
                ax,
                unfiltered_points,
                color=unfiltered_color,
                edge_color=unfiltered_color,
                quantile=0.98,
                fill_alpha=0.015,
            )

        filtered_metric_points = downsample_points(
            filtered_points,
            max_metric_points_per_class,
        )
        unfiltered_metric_points = downsample_points(
            unfiltered_points,
            max_metric_points_per_class,
        )
        if len(filtered_metric_points) and len(unfiltered_metric_points):
            mmd = rbf_mmd(filtered_metric_points, unfiltered_metric_points)
            emd = sliced_wasserstein_distance(
                filtered_metric_points,
                unfiltered_metric_points,
            )
        else:
            mmd = None
            emd = None
        comparison[name] = {
            "endpoint_rbf_mmd": mmd,
            "endpoint_sliced_wasserstein": emd,
            "heading_filter_count": int(len(filtered_points)),
            "no_heading_filter_count": int(len(unfiltered_points)),
            "heading_filter_metric_count": int(len(filtered_metric_points)),
            "no_heading_filter_metric_count": int(len(unfiltered_metric_points)),
        }

        metric_text = (
            "heading vs no-heading\nMMD=n/a\nEMD~=n/a"
            if mmd is None
            else f"heading vs no-heading\nMMD={mmd:.4f}\nEMD~={emd:.4f}"
        )
        ax.text(
            0.02,
            0.98,
            metric_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        ax.set_title(f"{name}: heading filter vs no filter")
        ax.set_xlabel("local x")
        ax.set_ylabel("local y")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return comparison


def plot_sensitivity_overview(rows, output_path):
    if not rows:
        return
    indices = np.arange(len(rows))
    endpoint = np.array(
        [
            np.nan if row["mean_endpoint_mean_distance"] is None else row["mean_endpoint_mean_distance"]
            for row in rows
        ],
        dtype=float,
    )
    mmd = np.array(
        [
            np.nan if row["mean_endpoint_rbf_mmd"] is None else row["mean_endpoint_rbf_mmd"]
            for row in rows
        ],
        dtype=float,
    )
    emd = np.array(
        [
            np.nan
            if row["mean_endpoint_sliced_wasserstein"] is None
            else row["mean_endpoint_sliced_wasserstein"]
            for row in rows
        ],
        dtype=float,
    )
    anomaly = np.array([row["is_anomaly"] for row in rows], dtype=bool)
    valid = {
        "endpoint": np.isfinite(endpoint),
        "mmd": np.isfinite(mmd),
        "emd": np.isfinite(emd),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, values, mask, title, ylabel in [
        (
            axes[0],
            endpoint,
            valid["endpoint"],
            "Endpoint Mean Separation",
            "mean endpoint distance",
        ),
        (axes[1], mmd, valid["mmd"], "Endpoint Distribution MMD", "mean RBF-MMD"),
        (
            axes[2],
            emd,
            valid["emd"],
            "Endpoint Distribution EMD Approx.",
            "mean sliced Wasserstein",
        ),
    ]:
        normal = mask & ~anomaly
        anomalous = mask & anomaly
        ax.scatter(indices[normal], values[normal], c="steelblue", s=14, label="normal")
        ax.scatter(indices[anomalous], values[anomalous], c="crimson", s=20, label="anomaly")
        missing = ~mask
        if np.any(missing):
            ax.scatter(
                indices[missing],
                np.zeros_like(indices[missing], dtype=float),
                c="gray",
                s=12,
                marker="x",
                label="no pair",
            )
        ax.set_title(title)
        ax.set_xlabel("matched sample index")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goal Swap Visualization for left/right/forward goal conditions."
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
    parser.add_argument("--min-goal-pos-dist", type=float, default=None)
    parser.add_argument("--max-goal-pos-dist", type=float, default=None)
    parser.add_argument("--max-direction-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-endpoint-goal-dist", type=float, default=None)
    parser.add_argument("--max-visualizations", type=int, default=None)
    parser.add_argument(
        "--skip-anomaly-stage",
        action="store_true",
        help=(
            "Skip the anomaly_samples stage. Horizon-filtered runs also skip "
            "this stage automatically."
        ),
    )
    parser.add_argument("--anomaly-mmd-threshold", type=float, default=0.12)
    parser.add_argument("--anomaly-emd-threshold", type=float, default=0.45)
    parser.add_argument("--global-endpoint-max-points-per-class", type=int, default=10000)
    parser.add_argument(
        "--global-metric-max-points-per-class",
        type=int,
        default=5000,
        help=(
            "Maximum points per direction used for global pairwise MMD/EMD. "
            "This avoids O(N^2) memory blowups; set <=0 to use all points."
        ),
    )
    parser.add_argument(
        "--heading-filter-mode",
        choices=("both", "heading_filter", "no_heading_filter"),
        default="both",
        help=(
            "Which goal-heading filter setting to evaluate. `both` preserves the "
            "previous behavior and also writes the heading-filter comparison."
        ),
    )
    parser.add_argument(
        "--trajectory-selection",
        choices=sorted(SELECTION_VARIANTS.keys()),
        default="baseline",
        help=(
            "Trajectory selection mode for metrics and saved endpoint logs. "
            "`baseline` keeps the sampled FlowNav trajectories; `cluster` "
            "selects the medoid of the largest trajectory cluster per goal."
        ),
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.35,
        help="Weighted trajectory distance threshold used by --trajectory-selection cluster.",
    )
    parser.add_argument("--kl-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    skip_anomaly_stage = args.skip_anomaly_stage or is_horizon_filtered_run(args)

    log(
        "Starting goal_swap_visualization "
        f"dataset={args.dataset} split={args.split} scan_batches={args.scan_batches} "
        f"batch_size={args.batch_size} num_samples={args.num_samples} "
        f"heading_filter_mode={args.heading_filter_mode} "
        f"skip_anomaly_stage={skip_anomaly_stage}"
    )
    log(f"Loading config: {args.config}")
    config = load_config(args.config)
    device = get_device(args.device)
    log(f"Using device: {device}")
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    log(f"Loading model checkpoint: {checkpoint_path}")
    model = build_model(config, checkpoint_path, device)
    log("Model loaded.")
    log("Building dataloader...")
    dataloader = build_dataloader(
        config=config,
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=True,
    )
    log("Dataloader ready.")
    transform = imagenet_transform()

    selection_output_root = output_root_for_selection(
        args.output_dir,
        args.trajectory_selection,
    )
    parent_output_dir = ensure_output_dir(
        selection_output_root,
        args.dataset,
        "goal_swap_visualization",
        swap_run_tag(
            args.angle_threshold_deg,
            args.anomaly_mmd_threshold,
            args.anomaly_emd_threshold,
        ),
    )
    log(f"Writing outputs under: {parent_output_dir}")
    require_all_directions = not is_horizon_filtered_run(args)

    def run_stage(filter_goal_heading: bool, stage_name: str, save_anomalies: bool):
        tag = heading_filter_tag(filter_goal_heading)
        output_dir = ensure_output_dir(parent_output_dir, stage_name, tag)
        log(f"Running stage={stage_name}, setting={tag}")
        scan_error = None
        try:
            directional_sets = find_all_matched_directional_goal_sets(
                dataset=dataloader.dataset,
                transform=transform,
                device=device,
                angle_threshold_deg=args.angle_threshold_deg,
                scan_items=args.scan_batches * args.batch_size,
                min_goal_offset=args.min_goal_offset,
                max_goal_offset=args.max_goal_offset,
                min_goal_pos_dist=args.min_goal_pos_dist,
                max_goal_pos_dist=args.max_goal_pos_dist,
                max_direction_angle_deg=args.max_direction_angle_deg,
                max_endpoint_goal_dist=args.max_endpoint_goal_dist,
                filter_goal_heading=filter_goal_heading,
                require_all_directions=require_all_directions,
            )
        except RuntimeError as exc:
            if not is_horizon_filtered_run(args):
                raise
            scan_error = str(exc)
            log(f"[{stage_name}/{tag}] No matched directional goals: {scan_error}")
            directional_sets = []
        if args.max_visualizations is not None:
            directional_sets = directional_sets[: args.max_visualizations]

        summary = {
            "test": "goal_swap_visualization",
            "trajectory_selection": args.trajectory_selection,
            "output_variant": SELECTION_VARIANTS[args.trajectory_selection],
            "cluster_threshold": (
                args.cluster_threshold if args.trajectory_selection == "cluster" else None
            ),
            "stage": stage_name,
            "config": args.config,
            "checkpoint": checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "angle_threshold_deg": args.angle_threshold_deg,
            "scan_batches": args.scan_batches,
            "scan_items": args.scan_batches * args.batch_size,
            "num_matched_sets": len(directional_sets),
            "scan_error": scan_error,
            "goal_matching": "same_trajectory_same_curr_time",
            "direction_source": "goal_pos",
            "require_all_directions": require_all_directions,
            "min_goal_offset": args.min_goal_offset,
            "max_goal_offset": args.max_goal_offset,
            "min_goal_pos_dist": args.min_goal_pos_dist,
            "max_goal_pos_dist": args.max_goal_pos_dist,
            "max_direction_angle_deg": args.max_direction_angle_deg,
            "max_endpoint_goal_dist": args.max_endpoint_goal_dist,
            "filter_goal_heading": filter_goal_heading,
            "anomaly_mmd_threshold": args.anomaly_mmd_threshold,
            "anomaly_emd_threshold": args.anomaly_emd_threshold,
            "skip_anomaly_stage": skip_anomaly_stage,
            "horizon_filtered_run": is_horizon_filtered_run(args),
            "save_anomaly_visualizations": save_anomalies,
            "global_endpoint_max_points_per_class": (
                args.global_endpoint_max_points_per_class
            ),
            "global_metric_max_points_per_class": (
                args.global_metric_max_points_per_class
            ),
            "items": [],
        }

        metric_rows = []
        anomaly_sets = []
        all_endpoints = {name: [] for name in DIRECTIONS}
        all_goals = {name: [] for name in DIRECTIONS}
        anomaly_endpoints = {name: [] for name in DIRECTIONS}
        anomaly_goals = {name: [] for name in DIRECTIONS}
        for item_idx, directional_set in enumerate(directional_sets):
            source = directional_set["base_source"]
            direction_names = available_directions_from_candidates(directional_set)
            log(
                f"[{stage_name}/{tag}] Evaluating sensitivity "
                f"{item_idx + 1}/{len(directional_sets)}: "
                f"dataset_index={source['dataset_index']}, "
                f"trajectory={source['trajectory']}, curr_time={source['curr_time']}, "
                f"directions={','.join(direction_names)}"
            )
            trajectories = sample_directional_trajectories(
                model=model,
                config=config,
                device=device,
                directional_set=directional_set,
                num_samples=args.num_samples,
            )
            metric_trajectories, selection_info = apply_trajectory_selection(
                trajectories,
                args,
            )
            metrics = compute_sensitivity_metrics(
                trajectories=metric_trajectories,
                directional_set=directional_set,
                angle_threshold_deg=args.angle_threshold_deg,
                eps=args.kl_eps,
                metric_goal_pos=use_metric_goal_pos(args),
            )
            for name in direction_names:
                all_endpoints[name].extend(
                    metric_trajectories[name][:, -1].astype(float).tolist()
                )
                all_goals[name].append(
                    candidate_goal_pos_for_plot(
                        directional_set["candidates"][name],
                        metric_goal_pos=use_metric_goal_pos(args),
                    ).astype(float).tolist()
                )
            anomalous = is_anomaly(metrics, args)
            row = {
                "matched_index": item_idx,
                "dataset_index": source["dataset_index"],
                "trajectory": source["trajectory"],
                "curr_time": source["curr_time"],
                "available_directions": metrics["available_directions"],
                "available_pairs": metrics["available_pairs"],
                "num_available_directions": len(metrics["available_directions"]),
                "num_available_pairs": len(metrics["available_pairs"]),
                "is_anomaly": anomalous if save_anomalies else False,
                "would_be_anomaly": anomalous,
                **{
                    key: value
                    for key, value in metrics.items()
                    if value is None or isinstance(value, (float, int, bool))
                },
                "endpoint_means": metrics["endpoint_means"],
                "goal_pos_by_direction": metrics["goal_pos_by_direction"],
                "pair_metrics": metrics["pairs"],
                "trajectory_selection": args.trajectory_selection,
                "cluster_threshold": (
                    args.cluster_threshold
                    if args.trajectory_selection == "cluster"
                    else None
                ),
                "selection_info": selection_info,
            }
            metric_rows.append(row)
            if anomalous and save_anomalies:
                anomaly_sets.append(
                    (item_idx, directional_set, metrics, metric_trajectories, selection_info)
                )
                for name in direction_names:
                    anomaly_endpoints[name].extend(
                        metric_trajectories[name][:, -1].astype(float).tolist()
                    )
                    anomaly_goals[name].append(
                        candidate_goal_pos_for_plot(
                            directional_set["candidates"][name],
                            metric_goal_pos=use_metric_goal_pos(args),
                        ).astype(float).tolist()
                    )

        anomaly_txt_path = None
        if save_anomalies:
            anomaly_txt_path = output_dir / "anomaly_indices.txt"
            with open(anomaly_txt_path, "w") as f:
                for item_idx, directional_set, metrics, _, _ in anomaly_sets:
                    source = directional_set["base_source"]
                    f.write(
                        f"{item_idx}\tdataset_index={source['dataset_index']}\t"
                        f"trajectory={source['trajectory']}\t"
                        f"curr_time={source['curr_time']}\t"
                        f"mean_mmd={metrics['mean_endpoint_rbf_mmd']:.6f}\t"
                        f"mean_emd={metrics['mean_endpoint_sliced_wasserstein']:.6f}\n"
                    )

        overview_path = None
        if metric_rows:
            overview_path = output_dir / timestamp_name(
                "goal_swap_sensitivity_overview", "png"
            )
            plot_sensitivity_overview(metric_rows, overview_path)
        global_endpoint_path = output_dir / timestamp_name(
            "goal_swap_global_endpoints", "png"
        )
        global_pair_metrics = plot_global_endpoint_distribution(
            all_endpoints,
            all_goals,
            global_endpoint_path,
            max_points_per_class=args.global_endpoint_max_points_per_class,
            max_metric_points_per_class=args.global_metric_max_points_per_class,
        )

        if save_anomalies:
            anomaly_global_endpoint_path = output_dir / timestamp_name(
                "goal_swap_anomaly_global_endpoints", "png"
            )
            anomaly_pair_metrics = plot_global_endpoint_distribution(
                anomaly_endpoints,
                anomaly_goals,
                anomaly_global_endpoint_path,
                max_points_per_class=args.global_endpoint_max_points_per_class,
                max_metric_points_per_class=args.global_metric_max_points_per_class,
            )
        else:
            anomaly_global_endpoint_path = None
            anomaly_pair_metrics = None

        for anomaly_rank, (
            item_idx,
            directional_set,
            metrics,
            trajectories,
            selection_info,
        ) in enumerate(
            anomaly_sets
        ):
            source = directional_set["base_source"]
            prefix = (
                f"anomaly_{anomaly_rank:05d}_matched{item_idx:05d}_"
                f"datasetidx{source['dataset_index']}_currtime{source['curr_time']}"
            )
            png_path = output_dir / f"{prefix}.png"
            endpoint_png_path = output_dir / f"{prefix}_endpoints.png"
            json_path = output_dir / f"{prefix}.json"
            log(
                f"[{stage_name}/{tag}] Rendering anomaly "
                f"{anomaly_rank + 1}/{len(anomaly_sets)}: "
                f"dataset_index={source['dataset_index']}, "
                f"trajectory={source['trajectory']}, curr_time={source['curr_time']}"
            )
            metadata = plot_directional_goal_samples(
                model=model,
                config=config,
                device=device,
                directional_set=directional_set,
                num_samples=args.num_samples,
                output_path=png_path,
                title="Goal swap visualization: same observation, matched goals",
                precomputed_trajectories=trajectories,
            )
            plot_endpoint_distribution(
                trajectories=trajectories,
                directional_set=directional_set,
                metrics=metrics,
                output_path=endpoint_png_path,
                metric_goal_pos=use_metric_goal_pos(args),
            )
            metadata.update(
                {
                    "test": "goal_swap_visualization_item",
                    "config": args.config,
                    "checkpoint": checkpoint_path,
                    "dataset": args.dataset,
                    "split": args.split,
                    "angle_threshold_deg": args.angle_threshold_deg,
                    "goal_matching": "same_trajectory_same_curr_time",
                    "direction_source": "goal_pos",
                    "min_goal_offset": args.min_goal_offset,
                    "max_goal_offset": args.max_goal_offset,
                    "min_goal_pos_dist": args.min_goal_pos_dist,
                    "max_goal_pos_dist": args.max_goal_pos_dist,
                    "require_all_directions": require_all_directions,
                    "max_direction_angle_deg": args.max_direction_angle_deg,
                    "max_endpoint_goal_dist": args.max_endpoint_goal_dist,
                    "filter_goal_heading": filter_goal_heading,
                    "trajectory_selection": args.trajectory_selection,
                    "output_variant": SELECTION_VARIANTS[args.trajectory_selection],
                    "cluster_threshold": (
                        args.cluster_threshold
                        if args.trajectory_selection == "cluster"
                        else None
                    ),
                    "selection_info": selection_info,
                    "anomaly_metrics": metrics,
                    "sampled_endpoints": {
                        name: trajectories[name][:, -1].astype(float).tolist()
                        for name in trajectories
                    },
                    "png_path": str(png_path),
                    "endpoint_png_path": str(endpoint_png_path),
                }
            )
            write_json(json_path, metadata)
            summary["items"].append(
                {
                    "matched_index": item_idx,
                    "dataset_index": source["dataset_index"],
                    "trajectory": source["trajectory"],
                    "curr_time": source["curr_time"],
                    "available_directions": metrics["available_directions"],
                    "available_pairs": metrics["available_pairs"],
                    "is_anomaly": True,
                    "mean_endpoint_mean_distance": metrics["mean_endpoint_mean_distance"],
                    "mean_s_goal": metrics["mean_s_goal"],
                    "mean_endpoint_symmetric_kl": metrics["mean_endpoint_symmetric_kl"],
                    "mean_endpoint_rbf_mmd": metrics["mean_endpoint_rbf_mmd"],
                    "mean_endpoint_sliced_wasserstein": metrics[
                        "mean_endpoint_sliced_wasserstein"
                    ],
                    "png_path": str(png_path),
                    "endpoint_png_path": str(endpoint_png_path),
                    "json_path": str(json_path),
                }
            )

        summary["num_anomalies"] = len(anomaly_sets)
        summary["anomaly_txt_path"] = str(anomaly_txt_path) if anomaly_txt_path else None
        summary["overview_path"] = str(overview_path) if overview_path else None
        summary["global_endpoint_path"] = str(global_endpoint_path)
        summary["global_endpoint_pair_metrics"] = global_pair_metrics
        summary["anomaly_global_endpoint_path"] = (
            str(anomaly_global_endpoint_path) if anomaly_global_endpoint_path else None
        )
        summary["anomaly_global_endpoint_pair_metrics"] = anomaly_pair_metrics
        summary["metrics"] = metric_rows
        summary_path = output_dir / timestamp_name("goal_swap_visualization_summary", "json")
        write_json(summary_path, summary)
        comparison_endpoints = anomaly_endpoints if save_anomalies else all_endpoints
        result = {
            "summary_path": str(summary_path),
            "output_dir": str(output_dir),
            "all_endpoints": comparison_endpoints,
            "num_matched_sets": len(directional_sets),
            "num_anomalies": len(anomaly_sets),
        }

        print(
            f"[{stage_name}/{tag}] Evaluated {len(directional_sets)} matched goal sets."
        )
        if save_anomalies:
            print(
                f"[{stage_name}/{tag}] Saved {len(anomaly_sets)} anomaly "
                f"visualizations in: {output_dir}"
            )
        else:
            print(f"[{stage_name}/{tag}] Saved global-only outputs in: {output_dir}")
        print(f"[{stage_name}/{tag}] Saved summary: {summary_path}")
        return result

    stage_results = {}
    stages = [("all_samples", False)]
    if skip_anomaly_stage:
        log("Skipping anomaly_samples stage for this run.")
    else:
        stages.append(("anomaly_samples", True))

    for stage_name, save_anomalies in stages:
        results_by_tag = {}
        for filter_goal_heading in selected_heading_filter_values(
            args.heading_filter_mode
        ):
            result = run_stage(
                filter_goal_heading=filter_goal_heading,
                stage_name=stage_name,
                save_anomalies=save_anomalies,
            )
            results_by_tag[heading_filter_tag(filter_goal_heading)] = result
        stage_results[stage_name] = results_by_tag

    all_sample_results = stage_results["all_samples"]
    if set(all_sample_results) != {"heading_filter", "no_heading_filter"}:
        log(
            "Skipping heading-filter comparison because only "
            f"{', '.join(sorted(all_sample_results))} was run."
        )
        return

    comparison_path = parent_output_dir / timestamp_name(
        "goal_swap_all_samples_heading_filter_endpoint_comparison", "png"
    )
    comparison = plot_heading_filter_comparison(
        all_sample_results,
        comparison_path,
        max_points_per_class=args.global_endpoint_max_points_per_class,
        max_metric_points_per_class=args.global_metric_max_points_per_class,
    )
    comparison_summary_path = parent_output_dir / timestamp_name(
        "goal_swap_all_samples_heading_filter_comparison", "json"
    )
    write_json(
        comparison_summary_path,
        {
            "test": "goal_swap_heading_filter_comparison",
            "trajectory_selection": args.trajectory_selection,
            "output_variant": SELECTION_VARIANTS[args.trajectory_selection],
            "cluster_threshold": (
                args.cluster_threshold if args.trajectory_selection == "cluster" else None
            ),
            "stage": "all_samples",
            "config": args.config,
            "checkpoint": checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "angle_threshold_deg": args.angle_threshold_deg,
            "anomaly_mmd_threshold": args.anomaly_mmd_threshold,
            "anomaly_emd_threshold": args.anomaly_emd_threshold,
            "global_endpoint_max_points_per_class": (
                args.global_endpoint_max_points_per_class
            ),
            "global_metric_max_points_per_class": (
                args.global_metric_max_points_per_class
            ),
            "comparison": comparison,
            "settings": {
                tag: {
                    "summary_path": all_sample_results[tag]["summary_path"],
                    "output_dir": all_sample_results[tag]["output_dir"],
                    "num_matched_sets": all_sample_results[tag]["num_matched_sets"],
                    "num_anomalies": all_sample_results[tag]["num_anomalies"],
                }
                for tag in all_sample_results
            },
            "comparison_png_path": str(comparison_path),
        },
    )

    print(f"[all_samples] Saved heading filter comparison: {comparison_path}")
    print(
        "[all_samples] Saved heading filter comparison summary: "
        f"{comparison_summary_path}"
    )


if __name__ == "__main__":
    main()
