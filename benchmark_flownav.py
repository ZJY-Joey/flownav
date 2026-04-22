import argparse
import statistics
import time
from typing import Tuple

import torch
import yaml
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from flownav.data.data_utils import img_path_to_data
from flownav.models.nomad import DenseNetwork, NoMaD
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from flownav.training.utils import cluster_trajectory_samples, model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure performance benchmark runner for FlowNav inference."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="flownav/config/flownav.yaml",
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file or run directory. If omitted, use config load_run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device, e.g. cuda:0 or cpu. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Measured benchmark iterations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of trajectory samples per inference.",
    )
    parser.add_argument(
        "--include-cluster",
        action="store_true",
        help="Include trajectory clustering in the benchmark.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.35,
        help="Trajectory clustering threshold.",
    )
    parser.add_argument(
        "--obs-images",
        nargs="+",
        default=None,
        help="Optional observation images in temporal order. If omitted, use random tensors.",
    )
    parser.add_argument(
        "--goal-image",
        default=None,
        help="Optional goal image. Required when using --obs-images.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_checkpoint(config: dict, checkpoint_arg: str | None) -> str:
    checkpoint = checkpoint_arg
    if checkpoint is None:
        if "load_run" not in config:
            raise ValueError("No --checkpoint was provided and config has no load_run.")
        checkpoint = f"logs/{config['load_run']}"

    if torch.jit.is_scripting():
        pass

    import os

    if os.path.isdir(checkpoint):
        ema_path = os.path.join(checkpoint, "ema_latest.pth")
        latest_path = os.path.join(checkpoint, "latest.pth")
        if os.path.isfile(ema_path):
            return ema_path
        if os.path.isfile(latest_path):
            return latest_path
        raise FileNotFoundError(
            f"Could not find ema_latest.pth or latest.pth in {checkpoint}"
        )

    if os.path.isfile(checkpoint):
        return checkpoint

    raise FileNotFoundError(f"Could not find checkpoint: {checkpoint}")


def build_model(config: dict, device: torch.device) -> NoMaD:
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        depth_cfg=config["depth"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    checkpoint = torch.load(config["depth"]["weights_path"], map_location=device)
    saved_state_dict = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )
    updated_state_dict = {
        k.replace("pretrained.", ""): v
        for k, v in saved_state_dict.items()
        if "pretrained" in k
    }
    new_state_dict = {
        k: v
        for k, v in updated_state_dict.items()
        if k in model.vision_encoder.depth_encoder.state_dict()
    }
    model.vision_encoder.depth_encoder.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()


def normalize_batch(
    obs_stack: torch.Tensor, goal_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=obs_stack.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=obs_stack.dtype).view(1, 3, 1, 1)
    obs_frames = torch.split(obs_stack, 3, dim=1)
    obs_frames = [((frame - mean) / std) for frame in obs_frames]
    obs_stack = torch.cat(obs_frames, dim=1)
    goal_batch = (goal_batch - mean) / std
    return obs_stack, goal_batch


def make_inputs(args: argparse.Namespace, config: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = tuple(config["image_size"])
    expected_obs = config["context_size"] + 1

    if args.obs_images is not None:
        if len(args.obs_images) != expected_obs:
            raise ValueError(
                f"Expected {expected_obs} observation images, got {len(args.obs_images)}."
            )
        if args.goal_image is None:
            raise ValueError("--goal-image is required when using --obs-images.")
        obs_images = [img_path_to_data(path, image_size) for path in args.obs_images]
        goal_image = img_path_to_data(args.goal_image, image_size)
        obs_stack = torch.cat(obs_images, dim=0).unsqueeze(0)
        goal_batch = goal_image.unsqueeze(0)
    else:
        obs_stack = torch.rand(
            1, expected_obs * 3, image_size[1], image_size[0], dtype=torch.float32
        )
        goal_batch = torch.rand(
            1, 3, image_size[1], image_size[0], dtype=torch.float32
        )

    obs_stack, goal_batch = normalize_batch(obs_stack, goal_batch)
    return obs_stack.to(device), goal_batch.to(device)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: NoMaD,
    obs_stack: torch.Tensor,
    goal_batch: torch.Tensor,
    config: dict,
    num_samples: int,
    include_cluster: bool,
    cluster_threshold: float,
    device: torch.device,
) -> None:
    outputs = model_output(
        model=model,
        batch_obs_images=obs_stack,
        batch_goal_images=goal_batch,
        pred_horizon=config["len_traj_pred"],
        action_dim=2,
        num_samples=num_samples,
        device=device,
        use_wandb=False,
    )
    if include_cluster:
        cluster_trajectory_samples(
            outputs["gc_actions"].detach().cpu().numpy(),
            distance_threshold=cluster_threshold,
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    model = build_model(config, device)
    load_checkpoint(model, checkpoint_path, device)
    obs_stack, goal_batch = make_inputs(args, config, device)

    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(
        f"Benchmark settings: warmup={args.warmup}, iters={args.iters}, "
        f"num_samples={args.num_samples}, include_cluster={args.include_cluster}"
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            benchmark_once(
                model=model,
                obs_stack=obs_stack,
                goal_batch=goal_batch,
                config=config,
                num_samples=args.num_samples,
                include_cluster=args.include_cluster,
                cluster_threshold=args.cluster_threshold,
                device=device,
            )
        cuda_sync(device)

        durations_ms = []
        for _ in range(args.iters):
            start = time.perf_counter()
            benchmark_once(
                model=model,
                obs_stack=obs_stack,
                goal_batch=goal_batch,
                config=config,
                num_samples=args.num_samples,
                include_cluster=args.include_cluster,
                cluster_threshold=args.cluster_threshold,
                device=device,
            )
            cuda_sync(device)
            durations_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(durations_ms)
    median_ms = statistics.median(durations_ms)
    min_ms = min(durations_ms)
    max_ms = max(durations_ms)
    std_ms = statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0

    print("Results")
    print(f"mean:   {mean_ms:.2f} ms  ({1000.0 / mean_ms:.2f} FPS)")
    print(f"median: {median_ms:.2f} ms  ({1000.0 / median_ms:.2f} FPS)")
    print(f"min:    {min_ms:.2f} ms  ({1000.0 / min_ms:.2f} FPS)")
    print(f"max:    {max_ms:.2f} ms  ({1000.0 / max_ms:.2f} FPS)")
    print(f"std:    {std_ms:.2f} ms")


if __name__ == "__main__":
    main()
