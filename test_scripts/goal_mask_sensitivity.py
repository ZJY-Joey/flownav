import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    build_dataloader,
    build_model,
    ensure_output_dir,
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
        description="Compare goal-conditioned and goal-masked trajectory distributions."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--include-invalid-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def endpoint_chamfer(gc_endpoints: torch.Tensor, uc_endpoints: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(gc_endpoints, uc_endpoints)
    return 0.5 * (
        distances.min(dim=2).values.mean(dim=1)
        + distances.min(dim=1).values.mean(dim=1)
    )


def plot_sensitivity(gc_samples: np.ndarray, uc_samples: np.ndarray, output_path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for traj in gc_samples:
        axes[0].plot(traj[:, 0], traj[:, 1], color="green", alpha=0.25, marker="o")
    axes[0].plot(0, 0, color="black", marker="o")
    axes[0].set_title("goal input: sampled trajectories")
    axes[0].set_aspect("equal", "box")

    for traj in uc_samples:
        axes[1].plot(traj[:, 0], traj[:, 1], color="red", alpha=0.25, marker="o")
    axes[1].plot(0, 0, color="black", marker="o")
    axes[1].set_title("goal masked: sampled trajectories")
    axes[1].set_aspect("equal", "box")

    gc_endpoints = gc_samples[:, -1]
    uc_endpoints = uc_samples[:, -1]
    axes[2].scatter(
        gc_endpoints[:, 0], gc_endpoints[:, 1], c="green", label="goal input"
    )
    axes[2].scatter(
        uc_endpoints[:, 0], uc_endpoints[:, 1], c="red", label="goal masked"
    )
    axes[2].scatter([0], [0], c="black", label="robot")
    axes[2].set_title("endpoint distribution")
    axes[2].set_aspect("equal", "box")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(
        "Starting goal_mask_sensitivity "
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

    rows = []
    all_endpoint_mean_l2 = []
    all_endpoint_chamfer = []
    all_matched_ade = []
    all_matched_fde = []
    vis_gc = None
    vis_uc = None

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        log(f"Evaluating batch {batch_idx + 1}/{args.max_batches}...")
        obs, goals, _, _, action_mask = prepare_batch(batch, transform, device)
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
        uc_samples = outputs["uc_actions"].reshape(
            batch_size, args.num_samples, config["len_traj_pred"], 2
        )

        gc_valid = gc_samples[valid]
        uc_valid = uc_samples[valid]
        matched_dist = torch.linalg.norm(gc_valid - uc_valid, dim=-1)
        matched_ade = matched_dist.mean(dim=(1, 2))
        matched_fde = matched_dist[:, :, -1].mean(dim=1)
        endpoint_mean_l2 = torch.linalg.norm(
            gc_valid[:, :, -1].mean(dim=1) - uc_valid[:, :, -1].mean(dim=1),
            dim=-1,
        )
        chamfer = endpoint_chamfer(gc_valid[:, :, -1], uc_valid[:, :, -1])

        all_endpoint_mean_l2.append(endpoint_mean_l2.detach().cpu().numpy())
        all_endpoint_chamfer.append(chamfer.detach().cpu().numpy())
        all_matched_ade.append(matched_ade.detach().cpu().numpy())
        all_matched_fde.append(matched_fde.detach().cpu().numpy())

        valid_indices = torch.where(valid)[0].detach().cpu().tolist()
        for local_idx, sample_idx in enumerate(valid_indices):
            rows.append(
                {
                    "batch": batch_idx,
                    "sample_in_batch": int(sample_idx),
                    "endpoint_mean_l2": float(endpoint_mean_l2[local_idx].item()),
                    "endpoint_chamfer": float(chamfer[local_idx].item()),
                    "matched_sample_ade": float(matched_ade[local_idx].item()),
                    "matched_sample_fde": float(matched_fde[local_idx].item()),
                    "num_trajectories": args.num_samples,
                }
            )

        if vis_gc is None:
            vis_gc = gc_valid[0].detach().cpu().numpy()
            vis_uc = uc_valid[0].detach().cpu().numpy()
        log(
            f"Batch {batch_idx + 1} done: valid_items={int(valid.sum().item())}, "
            f"mean_endpoint_chamfer={float(chamfer.mean().item()):.4f}"
        )

    if not rows:
        raise RuntimeError("No valid samples were evaluated.")

    endpoint_mean_l2 = np.concatenate(all_endpoint_mean_l2)
    endpoint_chamfer_values = np.concatenate(all_endpoint_chamfer)
    matched_ade_values = np.concatenate(all_matched_ade)
    matched_fde_values = np.concatenate(all_matched_fde)

    summary = {
        "test": "goal_mask_sensitivity",
        "config": args.config,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "num_batches_requested": args.max_batches,
        "num_items": len(rows),
        "num_samples": args.num_samples,
        "include_invalid_actions": args.include_invalid_actions,
        "mean_endpoint_mean_l2": float(endpoint_mean_l2.mean()),
        "median_endpoint_mean_l2": float(np.median(endpoint_mean_l2)),
        "mean_endpoint_chamfer": float(endpoint_chamfer_values.mean()),
        "median_endpoint_chamfer": float(np.median(endpoint_chamfer_values)),
        "mean_matched_sample_ade": float(matched_ade_values.mean()),
        "mean_matched_sample_fde": float(matched_fde_values.mean()),
    }

    output_dir = ensure_output_dir(args.output_dir, args.dataset, "goal_mask_sensitivity")
    log(f"Writing outputs to: {output_dir}")
    json_path = output_dir / timestamp_name("goal_mask_sensitivity", "json")
    csv_path = output_dir / timestamp_name("goal_mask_sensitivity_items", "csv")
    png_path = output_dir / timestamp_name("goal_mask_sensitivity", "png")
    write_json(json_path, summary)
    write_csv(csv_path, rows)
    log("Rendering sensitivity visualization...")
    plot_sensitivity(vis_gc, vis_uc, png_path)

    print(f"Saved summary: {json_path}")
    print(f"Saved per-item metrics: {csv_path}")
    print(f"Saved visualization: {png_path}")


if __name__ == "__main__":
    main()
