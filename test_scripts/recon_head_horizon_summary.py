import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdiffeq

from common import (
    DEFAULT_CONFIG,
    build_dataset,
    build_model,
    direction_name,
    get_device,
    imagenet_transform,
    load_config,
    log,
    prepare_goal_image,
    prepare_obs_image,
    resolve_checkpoint,
    run_dist_pred,
    safe_load_image,
    timestamp_name,
    write_csv,
    write_json,
)
from flownav.data.data_utils import calculate_sin_cos, to_local_coords
from flownav.training.utils import ACTION_STATS, get_action
from goal_swap_visualization import rbf_mmd, sliced_wasserstein_distance


DIRECTIONS = ("left", "forward", "right")
BUCKETS = {
    "short": (0.0, 2.0),
    "mid": (2.0, 4.0),
    "long": (4.0, None),
}
BUCKET_ORDER = ("short", "mid", "long")
BUCKET_COLORS = {
    "short": "#4c78a8",
    "mid": "#59a14f",
    "long": "#e15759",
}
VARIANT = "flownav_baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recon-only head horizon experiment. For each matched observation, "
            "select same-direction short/mid/long goal poses in local metric "
            "coordinates, compute topomap subgoals with the distance head, and "
            "summarize subgoal-vs-goal distributions."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--split", default="test")
    parser.add_argument("--angle-threshold-deg", type=float, default=10.0)
    parser.add_argument("--scan-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-sets", type=int, default=None)
    parser.add_argument("--max-direction-angle-deg", type=float, default=None)
    parser.add_argument("--max-goal-pos-dist", type=float, default=None)
    parser.add_argument("--topomap-radius", type=int, default=4)
    parser.add_argument("--close-threshold", type=float, default=3.0)
    parser.add_argument("--num-flow-samples", type=int, default=0)
    parser.add_argument("--flow-ode-steps", type=int, default=10)
    parser.add_argument("--max-points-per-cell", type=int, default=4000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default="test_logs_horizon/flownav_baseline/head_horizon_summary",
    )
    return parser.parse_args()


def bucket_for_dist(dist: float, max_goal_pos_dist: Optional[float]) -> Optional[str]:
    if dist < 0:
        return None
    if dist < 2.0:
        return "short"
    if dist < 4.0:
        return "mid"
    if max_goal_pos_dist is not None and dist > max_goal_pos_dist:
        return None
    return "long"


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


def bucket_target_dist(bucket: str, candidate_dist: float, max_goal_pos_dist: Optional[float]) -> float:
    if bucket == "short":
        return 1.0
    if bucket == "mid":
        return 3.0
    if max_goal_pos_dist is not None:
        return 0.5 * (4.0 + float(max_goal_pos_dist))
    return candidate_dist


def candidate_score(candidate: Dict[str, Any], bucket: str, max_goal_pos_dist: Optional[float]) -> float:
    target_dist = bucket_target_dist(
        bucket,
        float(candidate["goal_pos_dist"]),
        max_goal_pos_dist,
    )
    dist_score = abs(float(candidate["goal_pos_dist"]) - target_dist)
    if candidate["direction"] == "forward":
        angle_score = abs(float(candidate["goal_angle_deg"])) / 180.0
    elif candidate["direction"] == "left":
        angle_score = -float(candidate["goal_angle_deg"]) / 180.0
    else:
        angle_score = float(candidate["goal_angle_deg"]) / 180.0
    return dist_score + angle_score


def build_direction_horizon_set_for_index(
    dataset,
    sample_index: int,
    transform,
    device,
    angle_threshold_deg: float,
    max_direction_angle_deg: Optional[float],
    max_goal_pos_dist: Optional[float],
) -> Optional[Dict[str, Any]]:
    traj_name, curr_time, max_goal_dist = dataset.index_to_data[sample_index]
    traj_data = dataset._get_trajectory(traj_name)
    traj_len = len(traj_data["position"])
    context_times = list(
        range(
            curr_time + -dataset.context_size * dataset.waypoint_spacing,
            curr_time + 1,
            dataset.waypoint_spacing,
        )
    )
    try:
        obs_raw = torch.cat(
            [safe_load_image(dataset, traj_name, t) for t in context_times]
        )
    except FileNotFoundError:
        return None

    grouped = {
        direction: {bucket: [] for bucket in BUCKET_ORDER}
        for direction in DIRECTIONS
    }
    min_offset = max(1, int(dataset.min_action_distance) + 1)
    max_offset = min(
        int(max_goal_dist),
        int(dataset.max_dist_cat),
        int(dataset.max_action_distance) - 1,
    )
    for goal_offset in range(min_offset, max_offset + 1):
        goal_time = curr_time + goal_offset * dataset.waypoint_spacing
        if goal_time >= traj_len:
            continue
        goal_pos_metric = local_metric_position(traj_data, curr_time, goal_time)
        goal_pos_dist = float(np.linalg.norm(goal_pos_metric))
        bucket = bucket_for_dist(goal_pos_dist, max_goal_pos_dist)
        if bucket is None:
            continue
        goal_angle_deg = float(np.rad2deg(np.arctan2(goal_pos_metric[1], goal_pos_metric[0])))
        if (
            max_direction_angle_deg is not None
            and abs(goal_angle_deg) > float(max_direction_angle_deg)
        ):
            continue
        direction = direction_name(goal_angle_deg, angle_threshold_deg)
        try:
            actions, _ = dataset._compute_actions(traj_data, curr_time, goal_time)
            goal_raw = safe_load_image(dataset, traj_name, goal_time)
        except FileNotFoundError:
            continue
        actions_torch = torch.as_tensor(actions.astype(np.float32), dtype=torch.float32)
        if dataset.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        grouped[direction][bucket].append(
            {
                "direction": direction,
                "bucket": bucket,
                "goal_time": int(goal_time),
                "goal_offset": int(goal_offset),
                "goal_pos": goal_pos_metric.astype(np.float32),
                "goal_pos_dist": goal_pos_dist,
                "goal_angle_deg": goal_angle_deg,
                "goal_raw": goal_raw.detach().cpu(),
                "goal": prepare_goal_image(goal_raw, transform, device),
                "target_action": actions_torch.detach().cpu(),
            }
        )

    selected = {}
    for direction in DIRECTIONS:
        if not all(grouped[direction][bucket] for bucket in BUCKET_ORDER):
            continue
        selected[direction] = {
            bucket: min(
                grouped[direction][bucket],
                key=lambda item: candidate_score(item, bucket, max_goal_pos_dist),
            )
            for bucket in BUCKET_ORDER
        }
    if not selected:
        return None
    return {
        "base_obs_raw": obs_raw.detach().cpu(),
        "base_obs": prepare_obs_image(obs_raw, transform, device),
        "base_source": {
            "dataset_index": int(sample_index),
            "trajectory": traj_name,
            "curr_time": int(curr_time),
        },
        "candidates": selected,
    }


def find_direction_horizon_sets(
    dataset,
    transform,
    device,
    angle_threshold_deg: float,
    scan_items: int,
    max_direction_angle_deg: Optional[float],
    max_goal_pos_dist: Optional[float],
) -> list[Dict[str, Any]]:
    limit = min(scan_items, len(dataset))
    sets = []
    log(
        "Scanning recon samples for same-direction short/mid/long local goal poses, "
        f"up to {limit} items..."
    )
    for sample_index in range(limit):
        if sample_index % 100 == 0:
            log(f"Horizon scan item {sample_index + 1}/{limit}; matched_sets={len(sets)}")
        matched = build_direction_horizon_set_for_index(
            dataset=dataset,
            sample_index=sample_index,
            transform=transform,
            device=device,
            angle_threshold_deg=angle_threshold_deg,
            max_direction_angle_deg=max_direction_angle_deg,
            max_goal_pos_dist=max_goal_pos_dist,
        )
        if matched is not None:
            sets.append(matched)
    if not sets:
        raise RuntimeError(
            "Could not find any sample with a same-direction short/mid/long goal set."
        )
    return sets


def select_topomap_subgoal(
    model,
    dataset,
    transform,
    device,
    matched_set,
    direction: str,
    bucket: str,
    radius: int,
    close_threshold: float,
):
    source = matched_set["base_source"]
    candidate = matched_set["candidates"][direction][bucket]
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
    obs = matched_set["base_obs"].repeat(len(node_times), 1, 1, 1)
    dists = run_dist_pred(model, obs, goal_images, device).detach().cpu().reshape(-1).numpy()
    final_goal_dist_pred = float(
        run_dist_pred(model, matched_set["base_obs"], candidate["goal"], device)
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
    return {
        "node_times": node_times,
        "dists": dists,
        "closest_idx": closest_idx,
        "closest_time": int(node_times[closest_idx]),
        "final_goal_dist_pred": final_goal_dist_pred,
        "subgoal_idx": subgoal_idx,
        "subgoal_time": subgoal_time,
        "subgoal_pos": subgoal_pos,
        "goal_pos": np.asarray(candidate["goal_pos"], dtype=np.float32),
    }


def infer_flow_endpoints_with_subgoals(
    model,
    dataset,
    transform,
    device,
    matched_set,
    item_rows: list[Dict[str, Any]],
    num_samples: int,
    pred_horizon: int,
    ode_steps: int,
) -> list[Dict[str, Any]]:
    if num_samples <= 0 or not item_rows:
        return []
    subgoal_goals = torch.cat(
        [
            prepare_goal_image(
                safe_load_image(
                    dataset,
                    item_row["trajectory"],
                    int(item_row["subgoal_time"]),
                ),
                transform,
                device,
            )
            for item_row in item_rows
        ],
        dim=0,
    )
    batch_size = len(item_rows)
    no_mask = torch.zeros((batch_size,)).long().to(device)
    batch_obs = matched_set["base_obs"].repeat(batch_size, 1, 1, 1)
    with torch.no_grad():
        obsgoal_cond = model(
            "vision_encoder",
            obs_img=batch_obs,
            goal_img=subgoal_goals,
            input_goal_mask=no_mask,
        )
        obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)
        noisy_action = torch.randn(
            (len(obsgoal_cond), pred_horizon, 2),
            device=device,
        )
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net",
                sample=x,
                timestep=t,
                global_cond=obsgoal_cond,
            ),
            noisy_action,
            torch.linspace(0, 1, ode_steps, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        trajectories = (
            get_action(traj[-1], ACTION_STATS)
            .reshape(batch_size, num_samples, pred_horizon, 2)
            .detach()
            .cpu()
            .numpy()
        )
    rows = []
    for item_row, item_trajectories in zip(item_rows, trajectories):
        for sample_index, endpoint in enumerate(item_trajectories[:, -1]):
            rows.append(
                {
                    "dataset": item_row["dataset"],
                    "split": item_row["split"],
                    "matched_index": item_row["matched_index"],
                    "dataset_index": item_row["dataset_index"],
                    "trajectory": item_row["trajectory"],
                    "curr_time": item_row["curr_time"],
                    "direction": item_row["direction"],
                    "bucket": item_row["bucket"],
                    "goal_time": item_row["goal_time"],
                    "goal_offset": item_row["goal_offset"],
                    "subgoal_time": item_row["subgoal_time"],
                    "subgoal_pos_x": item_row["subgoal_pos_x"],
                    "subgoal_pos_y": item_row["subgoal_pos_y"],
                    "subgoal_pos_dist": item_row["subgoal_pos_dist"],
                    "flow_sample_index": sample_index,
                    "endpoint_x": float(endpoint[0]),
                    "endpoint_y": float(endpoint[1]),
                    "endpoint_dist": float(np.linalg.norm(endpoint)),
                }
            )
    return rows


def run_experiment(args, model, config, device):
    dataset = build_dataset(config, args.dataset, args.split)
    transform = imagenet_transform()
    matched_sets = find_direction_horizon_sets(
        dataset=dataset,
        transform=transform,
        device=device,
        angle_threshold_deg=args.angle_threshold_deg,
        scan_items=args.scan_batches * args.batch_size,
        max_direction_angle_deg=args.max_direction_angle_deg,
        max_goal_pos_dist=args.max_goal_pos_dist,
    )
    if args.max_sets is not None:
        matched_sets = matched_sets[: args.max_sets]

    rows = []
    flow_endpoint_rows = []
    for matched_index, matched_set in enumerate(matched_sets):
        source = matched_set["base_source"]
        if matched_index % 25 == 0:
            log(f"Computing subgoals {matched_index + 1}/{len(matched_sets)}")
        matched_item_rows = []
        for direction in DIRECTIONS:
            if direction not in matched_set["candidates"]:
                continue
            for bucket in BUCKET_ORDER:
                candidate = matched_set["candidates"][direction][bucket]
                selected = select_topomap_subgoal(
                    model=model,
                    dataset=dataset,
                    transform=transform,
                    device=device,
                    matched_set=matched_set,
                    direction=direction,
                    bucket=bucket,
                    radius=args.topomap_radius,
                    close_threshold=args.close_threshold,
                )
                item_row = {
                    "dataset": args.dataset,
                    "split": args.split,
                    "matched_index": matched_index,
                    "dataset_index": source["dataset_index"],
                    "trajectory": source["trajectory"],
                    "curr_time": source["curr_time"],
                    "direction": direction,
                    "bucket": bucket,
                    "bucket_min_goal_pos_dist": BUCKETS[bucket][0],
                    "bucket_max_goal_pos_dist": BUCKETS[bucket][1],
                    "goal_time": candidate["goal_time"],
                    "goal_offset": candidate["goal_offset"],
                    "goal_angle_deg": candidate["goal_angle_deg"],
                    "goal_pos_x": float(selected["goal_pos"][0]),
                    "goal_pos_y": float(selected["goal_pos"][1]),
                    "goal_pos_dist": float(np.linalg.norm(selected["goal_pos"])),
                    "subgoal_time": selected["subgoal_time"],
                    "subgoal_pos_x": float(selected["subgoal_pos"][0]),
                    "subgoal_pos_y": float(selected["subgoal_pos"][1]),
                    "subgoal_pos_dist": float(np.linalg.norm(selected["subgoal_pos"])),
                    "closest_time": selected["closest_time"],
                    "closest_pred_dist": float(selected["dists"][selected["closest_idx"]]),
                    "final_goal_dist_pred": selected["final_goal_dist_pred"],
                    "topomap_window_start": int(selected["node_times"][0]),
                    "topomap_window_end": int(selected["node_times"][-1]),
                    "topomap_window_count": len(selected["node_times"]),
                }
                rows.append(item_row)
                matched_item_rows.append(item_row)
        flow_endpoint_rows.extend(
            infer_flow_endpoints_with_subgoals(
                model=model,
                dataset=dataset,
                transform=transform,
                device=device,
                matched_set=matched_set,
                item_rows=matched_item_rows,
                num_samples=args.num_flow_samples,
                pred_horizon=config["len_traj_pred"],
                ode_steps=args.flow_ode_steps,
            )
        )
    return rows, flow_endpoint_rows


def finite_points(rows, direction: str, bucket: str, prefix: str) -> np.ndarray:
    points = [
        [float(row[f"{prefix}_pos_x"]), float(row[f"{prefix}_pos_y"])]
        for row in rows
        if row["direction"] == direction and row["bucket"] == bucket
    ]
    return np.asarray(points, dtype=np.float32)


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(0)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def draw_distribution_circle(ax, points: np.ndarray, color: str, linestyle: str):
    if len(points) == 0:
        return
    mean = points.mean(axis=0)
    if len(points) < 2:
        radius = 0.08
        circle = plt.Circle(mean, radius, fill=False, color=color, linestyle=linestyle, linewidth=1.7)
        ax.add_patch(circle)
        return
    cov = np.cov(points.T) + 1e-6 * np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    theta = np.linspace(0, 2 * np.pi, 180)
    ellipse = np.stack(
        [np.cos(theta) * np.sqrt(eigvals[0]), np.sin(theta) * np.sqrt(eigvals[1])],
        axis=0,
    )
    ellipse = (eigvecs @ (2.0 * ellipse)).T + mean[None]
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linestyle=linestyle, linewidth=1.8)


def compute_metric_rows(rows):
    metrics = []
    for direction in DIRECTIONS:
        for bucket in BUCKET_ORDER:
            subgoals = finite_points(rows, direction, bucket, "subgoal")
            goals = finite_points(rows, direction, bucket, "goal")
            if len(subgoals) and len(goals):
                mmd = rbf_mmd(subgoals, goals)
                emd = sliced_wasserstein_distance(subgoals, goals)
                subgoal_mean_dist = float(np.linalg.norm(subgoals, axis=1).mean())
                goal_mean_dist = float(np.linalg.norm(goals, axis=1).mean())
            else:
                mmd = np.nan
                emd = np.nan
                subgoal_mean_dist = np.nan
                goal_mean_dist = np.nan
            metrics.append(
                {
                    "direction": direction,
                    "bucket": bucket,
                    "count_subgoal": int(len(subgoals)),
                    "count_goal": int(len(goals)),
                    "subgoal_goal_rbf_mmd": float(mmd),
                    "subgoal_goal_sliced_wasserstein": float(emd),
                    "subgoal_mean_dist": float(subgoal_mean_dist),
                    "goal_mean_dist": float(goal_mean_dist),
                }
            )
    return metrics


def metric_lookup(metric_rows):
    return {(row["direction"], row["bucket"]): row for row in metric_rows}


def plot_distribution_summary(rows, metric_rows, output_path: Path, max_points: int):
    lookup = metric_lookup(metric_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        for bucket in BUCKET_ORDER:
            color = BUCKET_COLORS[bucket]
            subgoals = finite_points(rows, direction, bucket, "subgoal")
            goals = finite_points(rows, direction, bucket, "goal")
            if len(subgoals):
                shown = downsample(subgoals, max_points)
                ax.scatter(
                    shown[:, 0],
                    shown[:, 1],
                    s=13,
                    alpha=0.24,
                    color=color,
                    marker="o",
                    label=f"{bucket} subgoal",
                )
                draw_distribution_circle(ax, subgoals, color, "-")
            if len(goals):
                shown = downsample(goals, max_points)
                ax.scatter(
                    shown[:, 0],
                    shown[:, 1],
                    s=16,
                    alpha=0.24,
                    color=color,
                    marker="x",
                    label=f"{bucket} goal",
                )
                draw_distribution_circle(ax, goals, color, "--")
        lines = []
        for bucket in BUCKET_ORDER:
            row = lookup[(direction, bucket)]
            mmd = row["subgoal_goal_rbf_mmd"]
            emd = row["subgoal_goal_sliced_wasserstein"]
            count = min(row["count_subgoal"], row["count_goal"])
            if np.isfinite(mmd) and np.isfinite(emd):
                lines.append(f"{bucket}: MMD={mmd:.3f}, EMD~={emd:.3f}, n={count}")
            else:
                lines.append(f"{bucket}: missing")
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        ax.scatter([0.0], [0.0], c="black", marker="+", s=55, label="robot")
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.set_title(direction)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_xlabel("local x (m)")
    axes[0].set_ylabel("local y (m)")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Recon local goal pose and selected subgoal distributions by direction")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_local_subgoal_summary(rows, output_path: Path, max_points: int):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharex=True, sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        for bucket in BUCKET_ORDER:
            color = BUCKET_COLORS[bucket]
            subgoals = finite_points(rows, direction, bucket, "subgoal")
            if len(subgoals) == 0:
                continue
            shown = downsample(subgoals, max_points)
            ax.scatter(
                shown[:, 0],
                shown[:, 1],
                s=14,
                alpha=0.28,
                color=color,
                marker="o",
                label=f"{bucket} subgoal",
            )
            draw_distribution_circle(ax, subgoals, color, "-")
            mean_dist = float(np.linalg.norm(subgoals, axis=1).mean())
            ax.annotate(
                f"{bucket}: mean={mean_dist:.3f}m, n={len(subgoals)}",
                xy=tuple(subgoals.mean(axis=0)),
                xytext=(5, 5),
                textcoords="offset points",
                color=color,
                fontsize=8,
            )
        ax.scatter([0.0], [0.0], c="black", marker="+", s=55, label="robot")
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.set_title(direction)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_xlabel("local x (m)")
    axes[0].set_ylabel("local y (m)")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Recon local topomap subgoal distributions by final-goal bucket")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def endpoint_points(flow_endpoint_rows, direction: str, bucket: str) -> np.ndarray:
    points = [
        [float(row["endpoint_x"]), float(row["endpoint_y"])]
        for row in flow_endpoint_rows
        if row["direction"] == direction and row["bucket"] == bucket
    ]
    return np.asarray(points, dtype=np.float32)


def plot_subgoal_conditioned_flow_summary(
    rows,
    flow_endpoint_rows,
    output_path: Path,
    max_points: int,
):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 14.2), sharex=True, sharey=True)
    for row_idx, direction in enumerate(DIRECTIONS):
        endpoint_ax = axes[row_idx, 0]
        subgoal_ax = axes[row_idx, 1]
        for bucket in BUCKET_ORDER:
            color = BUCKET_COLORS[bucket]
            endpoints = endpoint_points(flow_endpoint_rows, direction, bucket)
            if len(endpoints):
                shown = downsample(endpoints, max_points)
                endpoint_ax.scatter(
                    shown[:, 0],
                    shown[:, 1],
                    s=10,
                    alpha=0.22,
                    color=color,
                    marker="o",
                    label=bucket,
                )
                draw_distribution_circle(endpoint_ax, endpoints, color, "-")
            subgoals = finite_points(rows, direction, bucket, "subgoal")
            if len(subgoals):
                shown = downsample(subgoals, max_points)
                subgoal_ax.scatter(
                    shown[:, 0],
                    shown[:, 1],
                    s=14,
                    alpha=0.28,
                    color=color,
                    marker="o",
                    label=bucket,
                )
                draw_distribution_circle(subgoal_ax, subgoals, color, "-")
        endpoint_ax.set_title(f"{direction}: flow endpoints conditioned on selected subgoal")
        subgoal_ax.set_title(f"{direction}: selected subgoal distribution")
        for ax in (endpoint_ax, subgoal_ax):
            ax.scatter([0.0], [0.0], c="black", marker="+", s=55, label="robot")
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
            ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.25)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.set_xlabel("local x (m)")
            ax.set_ylabel("local y (m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Flow head forced to use selected local subgoal images")
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def values_by_direction(metric_rows, key: str):
    grouped = metric_lookup(metric_rows)
    values = {}
    for direction in DIRECTIONS:
        values[direction] = [
            grouped[(direction, bucket)].get(key, np.nan)
            for bucket in BUCKET_ORDER
        ]
    return values


def plot_metric_summary(metric_rows, output_path: Path):
    x = np.arange(len(BUCKET_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True)
    panels = [
        (axes[0, 0], "subgoal_goal_rbf_mmd", "Raw MMD", "MMD"),
        (
            axes[0, 1],
            "subgoal_goal_sliced_wasserstein",
            "Raw EMD approximation",
            "sliced Wasserstein",
        ),
        (axes[1, 0], "subgoal_mean_dist", "Mean subgoal distance", "distance to robot (m)"),
        (axes[1, 1], "goal_mean_dist", "Mean goal distance", "distance to robot (m)"),
    ]
    direction_colors = {"left": "#4c78a8", "forward": "#59a14f", "right": "#e15759"}
    counts = values_by_direction(metric_rows, "count_subgoal")
    for ax, key, title, ylabel in panels:
        values = values_by_direction(metric_rows, key)
        for direction in DIRECTIONS:
            ax.plot(
                x,
                values[direction],
                marker="o",
                linewidth=2.0,
                color=direction_colors[direction],
                label=direction,
            )
            for xi, yi, count in zip(x, values[direction], counts[direction]):
                if np.isfinite(float(yi)):
                    ax.annotate(
                        f"n={int(count)}",
                        xy=(xi, yi),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_ORDER)
        ax.set_xlabel("goal pose local-distance bucket")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Recon subgoal-vs-goal horizon metrics")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading config: {args.config}")
    config = load_config(args.config)
    device = get_device(args.device)
    log(f"Using device: {device}")
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    log(f"Loading model checkpoint: {checkpoint_path}")
    model = build_model(config, checkpoint_path, device)
    log("Model loaded.")

    rows, flow_endpoint_rows = run_experiment(args, model, config, device)
    metric_rows = compute_metric_rows(rows)

    items_csv = output_dir / timestamp_name("recon_head_horizon_items", "csv")
    metrics_csv = output_dir / timestamp_name("recon_head_horizon_metrics", "csv")
    flow_endpoints_csv = output_dir / timestamp_name(
        "recon_head_horizon_subgoal_flow_endpoints", "csv"
    )
    summary_json = output_dir / timestamp_name("recon_head_horizon_summary", "json")
    dist_png = output_dir / "fig_recon_head_horizon_distribution.png"
    local_subgoal_png = output_dir / "fig_recon_head_horizon_local_subgoals.png"
    subgoal_flow_png = output_dir / "fig_recon_head_horizon_subgoal_conditioned_flow.png"
    metrics_png = output_dir / "fig_recon_head_horizon_metrics.png"

    write_csv(items_csv, rows)
    write_csv(metrics_csv, metric_rows)
    write_csv(flow_endpoints_csv, flow_endpoint_rows)
    plot_distribution_summary(rows, metric_rows, dist_png, args.max_points_per_cell)
    plot_local_subgoal_summary(rows, local_subgoal_png, args.max_points_per_cell)
    plot_subgoal_conditioned_flow_summary(
        rows,
        flow_endpoint_rows,
        subgoal_flow_png,
        args.max_points_per_cell,
    )
    plot_metric_summary(metric_rows, metrics_png)
    write_json(
        summary_json,
        {
            "test": "recon_head_horizon_summary",
            "output_variant": VARIANT,
            "config": args.config,
            "checkpoint": checkpoint_path,
            "dataset": args.dataset,
            "split": args.split,
            "angle_threshold_deg": args.angle_threshold_deg,
            "direction_convention": {
                "left": f"angle > {args.angle_threshold_deg} deg",
                "forward": f"|angle| <= {args.angle_threshold_deg} deg",
                "right": f"angle < -{args.angle_threshold_deg} deg",
            },
            "horizon_buckets_local_metric_m": {
                "short": "[0, 2)",
                "mid": "[2, 4)",
                "long": "[4, max_offset_range]",
            },
            "scan_batches": args.scan_batches,
            "batch_size": args.batch_size,
            "scan_items": args.scan_batches * args.batch_size,
            "max_sets": args.max_sets,
            "num_flow_samples": args.num_flow_samples,
            "flow_ode_steps": args.flow_ode_steps,
            "num_rows": len(rows),
            "num_subgoal_conditioned_flow_endpoints": len(flow_endpoint_rows),
            "num_matched_observations": len(
                {
                    (row["dataset_index"], row["trajectory"], row["curr_time"])
                    for row in rows
                }
            ),
            "metric_rows": metric_rows,
            "items_csv": str(items_csv),
            "metrics_csv": str(metrics_csv),
            "flow_endpoints_csv": str(flow_endpoints_csv),
            "distribution_png": str(dist_png),
            "local_subgoal_png": str(local_subgoal_png),
            "subgoal_conditioned_flow_png": str(subgoal_flow_png),
            "metrics_png": str(metrics_png),
        },
    )
    print(f"Saved item rows: {items_csv}")
    print(f"Saved metric rows: {metrics_csv}")
    print(f"Saved subgoal-conditioned flow endpoints: {flow_endpoints_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved distribution figure: {dist_png}")
    print(f"Saved local subgoal figure: {local_subgoal_png}")
    print(f"Saved subgoal-conditioned flow figure: {subgoal_flow_png}")
    print(f"Saved metric figure: {metrics_png}")


if __name__ == "__main__":
    main()
