import argparse
import os
import statistics
import time
from typing import Tuple

import torch
import torchdiffeq
import yaml

from flownav.data.data_utils import img_path_to_data
from flownav.models.factory import build_nomad_model, load_depth_encoder_weights
from flownav.models.nomad import NoMaD
from flownav.training.utils import ACTION_STATS, get_action, model_output

try:
    from flownav.training.utils import cluster_trajectory_samples
except ImportError:
    cluster_trajectory_samples = None


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
    parser.add_argument(
        "--stage-timing",
        action="store_true",
        help="Measure per-stage inference timings instead of only total time.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional txt file to write benchmark results.",
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
    model = build_nomad_model(config)
    load_depth_encoder_weights(model, config["depth"]["weights_path"], device)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device
) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def timed_stage(
    timings: dict[str, float],
    name: str,
    device: torch.device,
    func,
):
    cuda_sync(device)
    start = time.perf_counter()
    output = func()
    cuda_sync(device)
    timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start) * 1000.0
    return output


def benchmark_once_staged(
    model: NoMaD,
    obs_stack: torch.Tensor,
    goal_batch: torch.Tensor,
    config: dict,
    num_samples: int,
    include_cluster: bool,
    cluster_threshold: float,
    device: torch.device,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    goal_mask = torch.ones((goal_batch.shape[0],), dtype=torch.long, device=device)
    obs_cond = timed_stage(
        timings,
        "vision_encoder_masked_goal",
        device,
        lambda: model(
            "vision_encoder",
            obs_img=obs_stack,
            goal_img=goal_batch,
            input_goal_mask=goal_mask,
        ),
    )
    obs_cond = timed_stage(
        timings,
        "repeat_uc_condition",
        device,
        lambda: obs_cond.repeat_interleave(num_samples, dim=0),
    )

    no_mask = torch.zeros((goal_batch.shape[0],), dtype=torch.long, device=device)
    obsgoal_cond = timed_stage(
        timings,
        "vision_encoder_unmasked_goal",
        device,
        lambda: model(
            "vision_encoder",
            obs_img=obs_stack,
            goal_img=goal_batch,
            input_goal_mask=no_mask,
        ),
    )
    obsgoal_cond = timed_stage(
        timings,
        "repeat_gc_condition",
        device,
        lambda: obsgoal_cond.repeat_interleave(num_samples, dim=0),
    )

    pred_horizon = config["len_traj_pred"]
    action_dim = 2
    ode_steps = torch.linspace(0, 1, 10, device=device)

    uc_noise = timed_stage(
        timings,
        "randn_uc",
        device,
        lambda: torch.randn((len(obs_cond), pred_horizon, action_dim), device=device),
    )
    uc_traj = timed_stage(
        timings,
        "ode_uc_noise_pred_net",
        device,
        lambda: torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net", sample=x, timestep=t, global_cond=obs_cond
            ),
            uc_noise,
            ode_steps,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        ),
    )
    _ = timed_stage(
        timings,
        "postprocess_uc_get_action",
        device,
        lambda: get_action(uc_traj[-1], ACTION_STATS),
    )

    gc_noise = timed_stage(
        timings,
        "randn_gc",
        device,
        lambda: torch.randn((len(obs_cond), pred_horizon, action_dim), device=device),
    )
    gc_traj = timed_stage(
        timings,
        "ode_gc_noise_pred_net",
        device,
        lambda: torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net", sample=x, timestep=t, global_cond=obsgoal_cond
            ),
            gc_noise,
            ode_steps,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        ),
    )
    gc_actions = timed_stage(
        timings,
        "postprocess_gc_get_action",
        device,
        lambda: get_action(gc_traj[-1], ACTION_STATS),
    )

    _ = timed_stage(
        timings,
        "distance_head",
        device,
        lambda: model("dist_pred_net", obsgoal_cond=obsgoal_cond.flatten(start_dim=1)),
    )

    if include_cluster:
        if cluster_trajectory_samples is None:
            raise ImportError(
                "cluster_trajectory_samples is not available in flownav.training.utils"
            )
        _ = timed_stage(
            timings,
            "cluster_trajectory_samples",
            device,
            lambda: cluster_trajectory_samples(
                gc_actions.detach().cpu().numpy(),
                distance_threshold=cluster_threshold,
            ),
        )

    cuda_sync(device)
    timings["total"] = (time.perf_counter() - total_start) * 1000.0
    return timings


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
        if cluster_trajectory_samples is None:
            raise ImportError(
                "cluster_trajectory_samples is not available in flownav.training.utils"
            )
        cluster_trajectory_samples(
            outputs["gc_actions"].detach().cpu().numpy(),
            distance_threshold=cluster_threshold,
        )


def print_and_collect(lines: list[str], line: str) -> None:
    print(line)
    lines.append(line)


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

    output_lines: list[str] = []
    print_and_collect(output_lines, f"Using config: {args.config}")
    print_and_collect(output_lines, f"Using checkpoint: {checkpoint_path}")
    print_and_collect(output_lines, f"Using device: {device}")
    print_and_collect(
        output_lines,
        f"Benchmark settings: warmup={args.warmup}, iters={args.iters}, "
        f"num_samples={args.num_samples}, include_cluster={args.include_cluster}, "
        f"stage_timing={args.stage_timing}",
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            if args.stage_timing:
                benchmark_once_staged(
                    model=model,
                    obs_stack=obs_stack,
                    goal_batch=goal_batch,
                    config=config,
                    num_samples=args.num_samples,
                    include_cluster=args.include_cluster,
                    cluster_threshold=args.cluster_threshold,
                    device=device,
                )
            else:
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

        if args.stage_timing:
            stage_values: dict[str, list[float]] = {}
            for _ in range(args.iters):
                iter_timings = benchmark_once_staged(
                    model=model,
                    obs_stack=obs_stack,
                    goal_batch=goal_batch,
                    config=config,
                    num_samples=args.num_samples,
                    include_cluster=args.include_cluster,
                    cluster_threshold=args.cluster_threshold,
                    device=device,
                )
                for key, value in iter_timings.items():
                    stage_values.setdefault(key, []).append(value)
        else:
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

    print_and_collect(output_lines, "Results")
    if args.stage_timing:
        total_summary = summarize(stage_values["total"])
        print_and_collect(
            output_lines,
            f"total mean:   {total_summary['mean']:.2f} ms  "
            f"({1000.0 / total_summary['mean']:.2f} FPS)",
        )
        print_and_collect(
            output_lines,
            f"total median: {total_summary['median']:.2f} ms  "
            f"({1000.0 / total_summary['median']:.2f} FPS)",
        )
        print_and_collect(
            output_lines,
            f"total min:    {total_summary['min']:.2f} ms  "
            f"({1000.0 / total_summary['min']:.2f} FPS)",
        )
        print_and_collect(
            output_lines,
            f"total max:    {total_summary['max']:.2f} ms  "
            f"({1000.0 / total_summary['max']:.2f} FPS)",
        )
        print_and_collect(output_lines, f"total std:    {total_summary['std']:.2f} ms")
        print_and_collect(output_lines, "")
        print_and_collect(output_lines, "Stage breakdown")
        for stage_name in stage_values:
            stage_summary = summarize(stage_values[stage_name])
            print_and_collect(
                output_lines,
                f"{stage_name}: mean={stage_summary['mean']:.2f} ms, "
                f"median={stage_summary['median']:.2f} ms, "
                f"min={stage_summary['min']:.2f} ms, "
                f"max={stage_summary['max']:.2f} ms, "
                f"std={stage_summary['std']:.2f} ms",
            )
    else:
        summary = summarize(durations_ms)
        print_and_collect(
            output_lines, f"mean:   {summary['mean']:.2f} ms  ({1000.0 / summary['mean']:.2f} FPS)"
        )
        print_and_collect(
            output_lines,
            f"median: {summary['median']:.2f} ms  ({1000.0 / summary['median']:.2f} FPS)",
        )
        print_and_collect(
            output_lines, f"min:    {summary['min']:.2f} ms  ({1000.0 / summary['min']:.2f} FPS)"
        )
        print_and_collect(
            output_lines, f"max:    {summary['max']:.2f} ms  ({1000.0 / summary['max']:.2f} FPS)"
        )
        print_and_collect(output_lines, f"std:    {summary['std']:.2f} ms")

    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            f.write("\n".join(output_lines))
            f.write("\n")


if __name__ == "__main__":
    main()
