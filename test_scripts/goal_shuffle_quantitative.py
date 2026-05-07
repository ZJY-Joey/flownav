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
    find_matched_directional_goal_set,
    final_angle,
    get_device,
    imagenet_transform,
    load_config,
    log,
    plot_directional_goal_samples,
    prepare_batch,
    resolve_checkpoint,
    run_model,
    select_prediction,
    timestamp_name,
    trajectory_metrics,
    write_csv,
    write_json,
)


def angle_delta_deg(a: float, b: float) -> float:
    diff = np.arctan2(np.sin(np.deg2rad(a - b)), np.cos(np.deg2rad(a - b)))
    return float(abs(np.rad2deg(diff)))


def heading_filter_tag(enabled: bool) -> str:
    return "heading-filter" if enabled else "no-heading-filter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goal Shuffle Quantitative Test for FlowNav goal sensitivity."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--selection", choices=["first", "mean"], default="first")
    parser.add_argument("--success-fde-threshold", type=float, default=1.0)
    parser.add_argument("--visualization-samples", type=int, default=16)
    parser.add_argument("--scan-batches", type=int, default=30)
    parser.add_argument("--angle-threshold-deg", type=float, default=25.0)
    parser.add_argument("--min-goal-offset", type=int, default=None)
    parser.add_argument("--max-goal-offset", type=int, default=None)
    parser.add_argument("--min-alternative-angle-diff-deg", type=float, default=10.0)
    parser.add_argument("--max-alternative-angle-diff-deg", type=float, default=90.0)
    parser.add_argument("--max-direction-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-endpoint-goal-dist", type=float, default=None)
    parser.add_argument("--disable-goal-heading-filter", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(
        "Starting goal_shuffle_quantitative "
        f"dataset={args.dataset} split={args.split} max_batches={args.max_batches} "
        f"batch_size={args.batch_size} num_samples={args.num_samples}"
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

    correct_rows = []
    alternative_rows = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        log(f"Evaluating batch {batch_idx + 1}/{args.max_batches}...")
        obs, goals, actions, _, action_mask = prepare_batch(batch, transform, device)
        dataset_indices = batch[5].detach().cpu().tolist()
        batch_size = obs.shape[0]

        selected_indices = []
        alternative_goals = []
        for sample_idx, dataset_index in enumerate(dataset_indices):
            if action_mask[sample_idx] <= 0.5:
                continue
            matched_set = build_matched_goal_set_for_index(
                dataset=dataloader.dataset,
                sample_index=int(dataset_index),
                transform=transform,
                device=device,
                angle_threshold_deg=args.angle_threshold_deg,
                min_goal_offset=args.min_goal_offset,
                max_goal_offset=args.max_goal_offset,
                max_direction_angle_deg=args.max_direction_angle_deg,
                max_endpoint_goal_dist=args.max_endpoint_goal_dist,
                filter_goal_heading=not args.disable_goal_heading_filter,
            )
            if matched_set is None:
                continue

            current_angle = float(torch.rad2deg(final_angle(actions[sample_idx])).item())
            candidates = list(matched_set["candidates"].values())
            candidates = [
                candidate
                for candidate in candidates
                if args.min_alternative_angle_diff_deg
                <= angle_delta_deg(candidate["target_angle_deg"], current_angle)
                <= args.max_alternative_angle_diff_deg
            ]
            if not candidates:
                continue
            alternative = max(
                candidates,
                key=lambda candidate: angle_delta_deg(
                    candidate["target_angle_deg"], current_angle
                ),
            )
            selected_indices.append(sample_idx)
            alternative_goals.append(alternative["goal"])

        if not selected_indices:
            log(f"Batch {batch_idx + 1}: no matched alternative goals, skipping")
            continue

        selected = torch.as_tensor(selected_indices, device=device, dtype=torch.long)
        obs_eval = obs[selected]
        goals_eval = goals[selected]
        actions_eval = actions[selected]
        action_mask_eval = action_mask[selected]
        alternative_goals_eval = torch.cat(alternative_goals, dim=0)
        eval_batch_size = obs_eval.shape[0]

        correct_outputs = run_model(
            model, obs_eval, goals_eval, config["len_traj_pred"], args.num_samples, device
        )
        alternative_outputs = run_model(
            model,
            obs_eval,
            alternative_goals_eval,
            config["len_traj_pred"],
            args.num_samples,
            device,
        )

        correct_pred = select_prediction(
            correct_outputs["gc_actions"], eval_batch_size, args.selection
        )
        alternative_pred = select_prediction(
            alternative_outputs["gc_actions"], eval_batch_size, args.selection
        )

        correct_metrics = trajectory_metrics(
            correct_pred, actions_eval, action_mask_eval, args.success_fde_threshold
        )
        alternative_metrics = trajectory_metrics(
            alternative_pred, actions_eval, action_mask_eval, args.success_fde_threshold
        )
        correct_metrics["batch"] = batch_idx
        alternative_metrics["batch"] = batch_idx
        correct_metrics["num_matched_items"] = int(eval_batch_size)
        alternative_metrics["num_matched_items"] = int(eval_batch_size)
        correct_rows.append(correct_metrics)
        alternative_rows.append(alternative_metrics)
        log(
            f"Batch {batch_idx + 1} done: "
            f"correct_fde={correct_metrics['fde']:.4f}, "
            f"matched_alternative_fde={alternative_metrics['fde']:.4f}, "
            f"matched_items={eval_batch_size}/{batch_size}"
        )

    if not correct_rows:
        raise RuntimeError("No batches were evaluated.")

    metric_keys = [key for key in correct_rows[0] if key != "batch"]
    summary = {
        "test": "goal_shuffle_quantitative",
        "config": args.config,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "num_batches": len(correct_rows),
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "selection": args.selection,
        "success_fde_threshold": args.success_fde_threshold,
        "angle_threshold_deg": args.angle_threshold_deg,
        "min_goal_offset": args.min_goal_offset,
        "max_goal_offset": args.max_goal_offset,
        "min_alternative_angle_diff_deg": args.min_alternative_angle_diff_deg,
        "max_alternative_angle_diff_deg": args.max_alternative_angle_diff_deg,
        "max_direction_angle_deg": args.max_direction_angle_deg,
        "max_endpoint_goal_dist": args.max_endpoint_goal_dist,
        "filter_goal_heading": not args.disable_goal_heading_filter,
        "correct_goal": {
            key: float(np.mean([row[key] for row in correct_rows])) for key in metric_keys
        },
        "matched_alternative_goal": {
            key: float(np.mean([row[key] for row in alternative_rows]))
            for key in metric_keys
        },
        "goal_matching": "same_trajectory_same_curr_time",
    }
    summary["delta_alternative_minus_correct"] = {
        key: summary["matched_alternative_goal"][key] - summary["correct_goal"][key]
        for key in metric_keys
    }

    output_dir = ensure_output_dir(
        args.output_dir,
        args.dataset,
        "goal_shuffle_quantitative",
        heading_filter_tag(not args.disable_goal_heading_filter),
    )
    log(f"Writing outputs to: {output_dir}")
    json_path = output_dir / timestamp_name("goal_shuffle_quantitative", "json")
    csv_path = output_dir / timestamp_name("goal_shuffle_quantitative_batches", "csv")
    rows = []
    for correct, alternative in zip(correct_rows, alternative_rows):
        rows.append({"goal_condition": "correct", **correct})
        rows.append({"goal_condition": "matched_alternative", **alternative})
    write_json(json_path, summary)
    write_csv(csv_path, rows)

    if not args.skip_visualization:
        log("Starting directional visualization stage...")
        directional_set = find_matched_directional_goal_set(
            dataset=dataloader.dataset,
            transform=transform,
            device=device,
            angle_threshold_deg=args.angle_threshold_deg,
            scan_items=args.scan_batches * args.batch_size,
            min_goal_offset=args.min_goal_offset,
            max_goal_offset=args.max_goal_offset,
            max_direction_angle_deg=args.max_direction_angle_deg,
            max_endpoint_goal_dist=args.max_endpoint_goal_dist,
            filter_goal_heading=not args.disable_goal_heading_filter,
        )
        png_path = output_dir / timestamp_name("goal_shuffle_directional_swap", "png")
        vis_json_path = output_dir / timestamp_name(
            "goal_shuffle_directional_swap", "json"
        )
        vis_metadata = plot_directional_goal_samples(
            model=model,
            config=config,
            device=device,
            directional_set=directional_set,
            num_samples=args.visualization_samples,
            output_path=png_path,
            title=(
                "Goal shuffle qualitative check: same observation, "
                "left/right/forward goals"
            ),
        )
        vis_metadata.update(
            {
                "test": "goal_shuffle_directional_swap_visualization",
                "config": args.config,
                "checkpoint": checkpoint_path,
                "dataset": args.dataset,
                "split": args.split,
                "angle_threshold_deg": args.angle_threshold_deg,
                "scan_batches": args.scan_batches,
                "goal_matching": "same_trajectory_same_curr_time",
                "min_goal_offset": args.min_goal_offset,
                "max_goal_offset": args.max_goal_offset,
                "max_direction_angle_deg": args.max_direction_angle_deg,
                "max_endpoint_goal_dist": args.max_endpoint_goal_dist,
                "filter_goal_heading": not args.disable_goal_heading_filter,
            }
        )
        write_json(vis_json_path, vis_metadata)
        print(f"Saved directional visualization: {png_path}")
        print(f"Saved directional metadata: {vis_json_path}")

    print(f"Saved summary: {json_path}")
    print(f"Saved per-batch metrics: {csv_path}")


if __name__ == "__main__":
    main()
