import argparse

import numpy as np
import torch

from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    angle_diff_deg,
    build_dataloader,
    build_model,
    ensure_output_dir,
    final_angle,
    get_device,
    imagenet_transform,
    load_config,
    log,
    prepare_batch,
    resolve_checkpoint,
    run_model,
    timestamp_name,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goal-Inconsistent Rate test using final-waypoint direction."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--angle-threshold-deg", type=float, default=45.0)
    parser.add_argument("--include-invalid-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def angle_tag(angle_threshold_deg: float) -> str:
    value = f"{angle_threshold_deg:g}".replace("-", "neg").replace(".", "p")
    return f"angle{value}"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(
        "Starting goal_inconsistent_rate "
        f"dataset={args.dataset} split={args.split} max_batches={args.max_batches} "
        f"batch_size={args.batch_size} num_samples={args.num_samples} "
        f"angle_threshold_deg={args.angle_threshold_deg}"
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

    rows = []
    all_angle_diffs = []
    all_inconsistent = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        log(f"Evaluating batch {batch_idx + 1}/{args.max_batches}...")
        obs, goals, actions, _, action_mask = prepare_batch(batch, transform, device)
        valid = torch.ones_like(action_mask, dtype=torch.bool)
        if not args.include_invalid_actions:
            valid = action_mask > 0.5
        if not bool(valid.any()):
            log(f"Batch {batch_idx + 1}: no valid actions, skipping")
            continue

        batch_size = obs.shape[0]
        outputs = run_model(
            model, obs, goals, config["len_traj_pred"], args.num_samples, device
        )
        gc_samples = outputs["gc_actions"].reshape(
            batch_size, args.num_samples, config["len_traj_pred"], 2
        )

        target_angles = final_angle(actions[:, :, :2])
        pred_angles = final_angle(gc_samples)
        angle_diffs = angle_diff_deg(pred_angles, target_angles[:, None])
        inconsistent = angle_diffs > args.angle_threshold_deg

        valid_angle_diffs = angle_diffs[valid]
        valid_inconsistent = inconsistent[valid]
        all_angle_diffs.append(valid_angle_diffs.detach().cpu().reshape(-1).numpy())
        all_inconsistent.append(valid_inconsistent.detach().cpu().reshape(-1).numpy())

        valid_indices = torch.where(valid)[0].detach().cpu().tolist()
        per_item_mean_angle = valid_angle_diffs.mean(dim=1)
        per_item_median_angle = valid_angle_diffs.median(dim=1).values
        per_item_p90_angle = torch.quantile(valid_angle_diffs, 0.90, dim=1)
        per_item_p95_angle = torch.quantile(valid_angle_diffs, 0.95, dim=1)
        per_item_gir = valid_inconsistent.float().mean(dim=1)
        for local_idx, sample_idx in enumerate(valid_indices):
            row = {
                "batch": batch_idx,
                "sample_in_batch": sample_idx,
                "goal_inconsistent_rate": float(per_item_gir[local_idx].item()),
                "mean_angle_diff_deg": float(per_item_mean_angle[local_idx].item()),
                "median_angle_diff_deg": float(per_item_median_angle[local_idx].item()),
                "p90_angle_diff_deg": float(per_item_p90_angle[local_idx].item()),
                "p95_angle_diff_deg": float(per_item_p95_angle[local_idx].item()),
                "num_trajectories": args.num_samples,
            }
            rows.append(row)
        log(
            f"Batch {batch_idx + 1} done: valid_items={int(valid.sum().item())}, "
            f"gir={float(valid_inconsistent.float().mean().item()):.4f}, "
            f"mean_angle_diff={float(valid_angle_diffs.mean().item()):.2f} deg"
        )

    if not all_angle_diffs:
        raise RuntimeError("No valid samples were evaluated.")

    angle_diff_values = np.concatenate(all_angle_diffs)
    inconsistent_values = np.concatenate(all_inconsistent)
    summary = {
        "test": "goal_inconsistent_rate",
        "config": args.config,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "num_batches_requested": args.max_batches,
        "num_items": len(rows),
        "num_samples": args.num_samples,
        "angle_threshold_deg": args.angle_threshold_deg,
        "include_invalid_actions": args.include_invalid_actions,
        "goal_inconsistent_rate": float(inconsistent_values.mean()),
        "mean_angle_diff_deg": float(angle_diff_values.mean()),
        "median_angle_diff_deg": float(np.median(angle_diff_values)),
        "p90_angle_diff_deg": float(np.percentile(angle_diff_values, 90)),
        "p95_angle_diff_deg": float(np.percentile(angle_diff_values, 95)),
    }

    output_dir = ensure_output_dir(
        args.output_dir, args.dataset, "goal_inconsistent_rate"
    )
    log(f"Writing outputs to: {output_dir}")
    output_prefix = f"goal_inconsistent_rate_{angle_tag(args.angle_threshold_deg)}"
    json_path = output_dir / timestamp_name(output_prefix, "json")
    csv_path = output_dir / timestamp_name(f"{output_prefix}_items", "csv")
    write_json(json_path, summary)
    write_csv(csv_path, rows)

    print(f"Saved summary: {json_path}")
    print(f"Saved per-item metrics: {csv_path}")


if __name__ == "__main__":
    main()
