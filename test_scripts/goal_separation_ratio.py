import argparse

import numpy as np
import torch

from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    build_matched_goal_set_for_index,
    build_dataloader,
    build_model,
    ensure_output_dir,
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


DIRECTIONS = ("left", "right", "forward")


def heading_filter_tag(enabled: bool) -> str:
    return "heading-filter" if enabled else "no-heading-filter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Goal Separation Ratio over hard left/right/forward goal triplets."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scan-batches", type=int, default=50)
    parser.add_argument("--max-triplets", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--angle-threshold-deg", type=float, default=25.0)
    parser.add_argument("--min-goal-offset", type=int, default=None)
    parser.add_argument("--max-goal-offset", type=int, default=None)
    parser.add_argument("--max-direction-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-endpoint-goal-dist", type=float, default=None)
    parser.add_argument("--disable-goal-heading-filter", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def collect_hard_triplets(
    dataset,
    transform,
    device: torch.device,
    angle_threshold_deg: float,
    scan_items: int,
    max_triplets: int,
    min_goal_offset: int | None,
    max_goal_offset: int | None,
    max_direction_angle_deg: float | None,
    max_endpoint_goal_dist: float | None,
    filter_goal_heading: bool,
) -> list[dict]:
    triplets = []
    log(
        "Scanning same-trajectory/same-time samples for "
        f"{max_triplets} left/right/forward hard triplets, "
        f"up to {scan_items} items..."
    )

    limit = min(scan_items, len(dataset))
    for sample_index in range(limit):
        if len(triplets) >= max_triplets:
            break
        if sample_index % 100 == 0:
            log(f"Triplet scan item {sample_index + 1}/{limit}")

        matched_set = build_matched_goal_set_for_index(
            dataset=dataset,
            sample_index=sample_index,
            transform=transform,
            device=device,
            angle_threshold_deg=angle_threshold_deg,
            min_goal_offset=min_goal_offset,
            max_goal_offset=max_goal_offset,
            max_direction_angle_deg=max_direction_angle_deg,
            max_endpoint_goal_dist=max_endpoint_goal_dist,
            filter_goal_heading=filter_goal_heading,
        )
        if matched_set is None or not {"left", "right", "forward"} <= set(
            matched_set["candidates"]
        ):
            continue

        triplets.append(
            {
                "base_obs": matched_set["base_obs"],
                "goals": {
                    name: matched_set["candidates"][name]["goal"]
                    for name in DIRECTIONS
                },
                "source": {
                    "base": matched_set["base_source"],
                    "goals": {
                        name: matched_set["candidates"][name]["source"]
                        for name in DIRECTIONS
                    },
                    "goal_angles_deg": {
                        name: matched_set["candidates"][name]["goal_angle_deg"]
                        for name in DIRECTIONS
                    },
                    "target_angles_deg": {
                        name: matched_set["candidates"][name]["target_angle_deg"]
                        for name in DIRECTIONS
                    },
                    "endpoint_goal_dists": {
                        name: matched_set["candidates"][name]["endpoint_goal_dist"]
                        for name in DIRECTIONS
                    },
                },
            }
        )
        source = matched_set["base_source"]
        log(
            f"Added matched triplet {len(triplets)}/{max_triplets}: "
            f"dataset_index={source['dataset_index']}, "
            f"trajectory={source['trajectory']}, curr_time={source['curr_time']}"
        )

    if not triplets:
        raise RuntimeError("Could not find any left/right/forward hard triplets.")
    return triplets


def endpoint_stats(
    model,
    config: dict,
    device: torch.device,
    triplet: dict,
    num_samples: int,
    epsilon: float,
) -> dict:
    endpoints_by_direction = {}
    means = {}
    spreads = {}

    for name in DIRECTIONS:
        outputs = run_model(
            model,
            triplet["base_obs"],
            triplet["goals"][name],
            config["len_traj_pred"],
            num_samples,
            device,
        )
        samples = outputs["gc_actions"].reshape(num_samples, config["len_traj_pred"], 2)
        endpoints = samples[:, -1, :]
        endpoint_mean = endpoints.mean(dim=0)
        spread = torch.linalg.norm(endpoints - endpoint_mean[None, :], dim=-1).mean()
        endpoints_by_direction[name] = endpoints
        means[name] = endpoint_mean
        spreads[name] = spread

    pair_distances = {}
    for i, first in enumerate(DIRECTIONS):
        for second in DIRECTIONS[i + 1 :]:
            pair_distances[f"{first}_{second}"] = torch.linalg.norm(
                means[first] - means[second]
            )

    inter_goal_distance = torch.stack(list(pair_distances.values())).mean()
    within_goal_dispersion = torch.stack([spreads[name] for name in DIRECTIONS]).mean()
    gsr = inter_goal_distance / torch.clamp(within_goal_dispersion, min=epsilon)

    row = {
        "inter_goal_distance": float(inter_goal_distance.item()),
        "within_goal_dispersion": float(within_goal_dispersion.item()),
        "goal_separation_ratio": float(gsr.item()),
    }
    for name in DIRECTIONS:
        row[f"{name}_spread"] = float(spreads[name].item())
        row[f"{name}_endpoint_mean_x"] = float(means[name][0].item())
        row[f"{name}_endpoint_mean_y"] = float(means[name][1].item())
    for name, value in pair_distances.items():
        row[f"{name}_endpoint_mean_distance"] = float(value.item())
    return row


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(
        "Starting goal_separation_ratio "
        f"dataset={args.dataset} split={args.split} scan_batches={args.scan_batches} "
        f"max_triplets={args.max_triplets} batch_size={args.batch_size} "
        f"num_samples={args.num_samples} angle_threshold_deg={args.angle_threshold_deg}"
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

    triplets = collect_hard_triplets(
        dataset=dataloader.dataset,
        transform=transform,
        device=device,
        angle_threshold_deg=args.angle_threshold_deg,
        scan_items=args.scan_batches * args.batch_size,
        max_triplets=args.max_triplets,
        min_goal_offset=args.min_goal_offset,
        max_goal_offset=args.max_goal_offset,
        max_direction_angle_deg=args.max_direction_angle_deg,
        max_endpoint_goal_dist=args.max_endpoint_goal_dist,
        filter_goal_heading=not args.disable_goal_heading_filter,
    )

    rows = []
    for triplet_idx, triplet in enumerate(triplets):
        log(f"Evaluating hard triplet {triplet_idx + 1}/{len(triplets)}...")
        metrics = endpoint_stats(
            model=model,
            config=config,
            device=device,
            triplet=triplet,
            num_samples=args.num_samples,
            epsilon=args.epsilon,
        )
        source = triplet["source"]
        rows.append(
            {
                "triplet": triplet_idx,
                "base_dataset_index": source["base"]["dataset_index"],
                "trajectory": source["base"]["trajectory"],
                "curr_time": source["base"]["curr_time"],
                "left_goal_time": source["goals"]["left"]["goal_time"],
                "right_goal_time": source["goals"]["right"]["goal_time"],
                "forward_goal_time": source["goals"]["forward"]["goal_time"],
                "left_goal_offset": source["goals"]["left"]["goal_offset"],
                "right_goal_offset": source["goals"]["right"]["goal_offset"],
                "forward_goal_offset": source["goals"]["forward"]["goal_offset"],
                "left_goal_angle_deg": source["goal_angles_deg"]["left"],
                "right_goal_angle_deg": source["goal_angles_deg"]["right"],
                "forward_goal_angle_deg": source["goal_angles_deg"]["forward"],
                "left_target_angle_deg": source["target_angles_deg"]["left"],
                "right_target_angle_deg": source["target_angles_deg"]["right"],
                "forward_target_angle_deg": source["target_angles_deg"]["forward"],
                "left_endpoint_goal_dist": source["endpoint_goal_dists"]["left"],
                "right_endpoint_goal_dist": source["endpoint_goal_dists"]["right"],
                "forward_endpoint_goal_dist": source["endpoint_goal_dists"][
                    "forward"
                ],
                "num_trajectories": args.num_samples,
                **metrics,
            }
        )
        log(
            f"Triplet {triplet_idx + 1} done: "
            f"gsr={metrics['goal_separation_ratio']:.4f}, "
            f"inter={metrics['inter_goal_distance']:.4f}, "
            f"within={metrics['within_goal_dispersion']:.4f}"
        )

    gsr_values = np.array([row["goal_separation_ratio"] for row in rows])
    inter_values = np.array([row["inter_goal_distance"] for row in rows])
    within_values = np.array([row["within_goal_dispersion"] for row in rows])
    summary = {
        "test": "goal_separation_ratio",
        "config": args.config,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "scan_batches": args.scan_batches,
        "num_triplets": len(rows),
        "num_samples": args.num_samples,
        "angle_threshold_deg": args.angle_threshold_deg,
        "min_goal_offset": args.min_goal_offset,
        "max_goal_offset": args.max_goal_offset,
        "max_direction_angle_deg": args.max_direction_angle_deg,
        "max_endpoint_goal_dist": args.max_endpoint_goal_dist,
        "filter_goal_heading": not args.disable_goal_heading_filter,
        "epsilon": args.epsilon,
        "goal_matching": "same_trajectory_same_curr_time",
        "mean_inter_goal_distance": float(inter_values.mean()),
        "median_inter_goal_distance": float(np.median(inter_values)),
        "mean_within_goal_dispersion": float(within_values.mean()),
        "median_within_goal_dispersion": float(np.median(within_values)),
        "mean_goal_separation_ratio": float(gsr_values.mean()),
        "median_goal_separation_ratio": float(np.median(gsr_values)),
        "p10_goal_separation_ratio": float(np.percentile(gsr_values, 10)),
        "p90_goal_separation_ratio": float(np.percentile(gsr_values, 90)),
    }

    output_dir = ensure_output_dir(
        args.output_dir,
        args.dataset,
        "goal_separation_ratio",
        heading_filter_tag(not args.disable_goal_heading_filter),
    )
    log(f"Writing outputs to: {output_dir}")
    json_path = output_dir / timestamp_name("goal_separation_ratio", "json")
    csv_path = output_dir / timestamp_name("goal_separation_ratio_triplets", "csv")
    write_json(json_path, summary)
    write_csv(csv_path, rows)

    print(f"Saved summary: {json_path}")
    print(f"Saved per-triplet metrics: {csv_path}")


if __name__ == "__main__":
    main()
