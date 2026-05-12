import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flownav.data.vint_dataset import ViNT_Dataset
from flownav.data.data_utils import (
    calculate_sin_cos,
    get_data_path,
    img_path_to_data,
    to_local_coords,
    yaw_rotmat,
)
from flownav.models.factory import build_nomad_model
from flownav.models.nomad import NoMaD
from flownav.training.utils import model_output
from flownav.visualizing.plot import plot_trajs_and_points


DEFAULT_CONFIG = REPO_ROOT / "flownav" / "config" / "flownav.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "test_logs" / "flownav_baseline"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def angle_tag(angle_threshold_deg: float) -> str:
    value = f"{angle_threshold_deg:g}".replace("-", "neg").replace(".", "p")
    return f"angle{value}"


def load_config(config_path: str) -> dict:
    with open(DEFAULT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    with open(config_path, "r") as f:
        config.update(yaml.safe_load(f))
    return config


def resolve_checkpoint(config: dict, checkpoint_arg: Optional[str]) -> str:
    checkpoint = checkpoint_arg
    if checkpoint is None:
        if "load_run" not in config:
            raise ValueError("No --checkpoint provided and config has no `load_run`.")
        checkpoint = os.path.join("logs", config["load_run"])

    if os.path.isdir(checkpoint):
        latest_path = os.path.join(checkpoint, "latest.pth")
        if os.path.isfile(latest_path):
            return latest_path
        raise FileNotFoundError(f"Could not find latest.pth in {checkpoint}")
    if os.path.isfile(checkpoint):
        return checkpoint
    raise FileNotFoundError(f"Could not find checkpoint: {checkpoint}")


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(config: dict, checkpoint_path: str, device: torch.device) -> NoMaD:
    model = build_nomad_model(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def build_dataset(config: dict, dataset_name: str, split: str) -> ViNT_Dataset:
    data_config = config["datasets"][dataset_name]
    if split not in data_config:
        raise ValueError(f"Dataset {dataset_name} does not define split `{split}`.")
    return ViNT_Dataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config[split],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        waypoint_spacing=data_config["waypoint_spacing"],
        min_dist_cat=config["distance"]["min_dist_cat"],
        max_dist_cat=config["distance"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"],
        max_action_distance=config["action"]["max_dist_cat"],
        negative_mining=True,
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        context_size=config["context_size"],
        context_type=config["context_type"],
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        goal_type=config["goal_type"],
    )


def build_dataloader(
    config: dict,
    dataset_name: str,
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = build_dataset(config, dataset_name, split)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def imagenet_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def prepare_batch(
    batch: Tuple[torch.Tensor, ...],
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_image, goal_image, actions, distance, _, _, action_mask = batch
    obs_images = torch.split(obs_image, 3, dim=1)
    batch_obs_images = torch.cat([transform(obs) for obs in obs_images], dim=1).to(
        device
    )
    batch_goal_images = transform(goal_image).to(device)
    return (
        batch_obs_images,
        batch_goal_images,
        actions.to(device),
        distance.to(device),
        action_mask.to(device),
    )


def run_model(
    model: NoMaD,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    num_samples: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        return model_output(
            model=model,
            batch_obs_images=batch_obs_images,
            batch_goal_images=batch_goal_images,
            pred_horizon=pred_horizon,
            action_dim=2,
            num_samples=num_samples,
            device=device,
            use_wandb=False,
        )


def run_dist_pred(
    model: NoMaD,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
        obsgoal_cond = model(
            "vision_encoder",
            obs_img=batch_obs_images,
            goal_img=batch_goal_images,
            input_goal_mask=no_mask,
        )
        obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
        return model("dist_pred_net", obsgoal_cond=obsgoal_cond)


def select_prediction(
    gc_actions: torch.Tensor, batch_size: int, selection: str
) -> torch.Tensor:
    pred_horizon = gc_actions.shape[1]
    action_dim = gc_actions.shape[2]
    samples = gc_actions.reshape(batch_size, -1, pred_horizon, action_dim)
    if selection == "first":
        return samples[:, 0]
    if selection == "mean":
        return samples.mean(dim=1)
    raise ValueError(f"Unsupported selection: {selection}")


def trajectory_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor,
    success_fde_threshold: float,
) -> Dict[str, float]:
    valid = action_mask > 0.5
    if not bool(valid.any()):
        valid = torch.ones_like(action_mask, dtype=torch.bool)

    pred = pred_actions[valid, :, :2]
    target = target_actions[valid, :, :2]
    point_dist = torch.linalg.norm(pred - target, dim=-1)
    ade = point_dist.mean(dim=-1)
    fde = point_dist[:, -1]
    mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    waypoint_cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean(
        dim=-1
    )
    flat_cos = torch.nn.functional.cosine_similarity(
        pred.flatten(start_dim=1),
        target.flatten(start_dim=1),
        dim=-1,
    )
    success = fde <= success_fde_threshold

    return {
        "num_valid": int(valid.sum().item()),
        "action_loss_mse": float(mse.mean().item()),
        "ade": float(ade.mean().item()),
        "fde": float(fde.mean().item()),
        "waypoint_cos_sim": float(waypoint_cos.mean().item()),
        "trajectory_cos_sim": float(flat_cos.mean().item()),
        "success_rate_fde": float(success.float().mean().item()),
    }


def final_angle(actions: torch.Tensor) -> torch.Tensor:
    final_vec = actions[..., -1, :2]
    return torch.atan2(final_vec[..., 1], final_vec[..., 0])


def angle_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.atan2(torch.sin(a - b), torch.cos(a - b))
    return torch.rad2deg(torch.abs(diff))


def direction_name(angle_deg: float, threshold: float) -> str:
    if angle_deg > threshold:
        return "left"
    if angle_deg < -threshold:
        return "right"
    return "forward"


def heading_matches_goal_side(
    goal_side: str, heading_angle_deg: float, threshold: float
) -> bool:
    if goal_side == "forward":
        return abs(heading_angle_deg) <= threshold
    if goal_side == "left":
        return heading_angle_deg >= -threshold
    if goal_side == "right":
        return heading_angle_deg <= threshold
    raise ValueError(f"Unsupported goal_side: {goal_side}")


def prepare_obs_image(
    obs_image: torch.Tensor,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    obs_images = torch.split(obs_image.unsqueeze(0), 3, dim=1)
    return torch.cat([transform(obs) for obs in obs_images], dim=1).to(device)


def prepare_goal_image(
    goal_image: torch.Tensor,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    return transform(goal_image.unsqueeze(0)).to(device)


def safe_load_image(dataset: ViNT_Dataset, trajectory_name: str, time_index: int) -> torch.Tensor:
    image = dataset._load_image(trajectory_name, time_index)
    if image is not None:
        return image

    image_path = get_data_path(dataset.data_folder, trajectory_name, time_index)
    try:
        return img_path_to_data(image_path, dataset.image_size)
    except Exception as exc:
        raise FileNotFoundError(
            "Could not load image from dataset cache or filesystem: "
            f"{image_path}"
        ) from exc


def build_matched_goal_set_for_index(
    dataset: ViNT_Dataset,
    sample_index: int,
    transform: transforms.Compose,
    device: torch.device,
    angle_threshold_deg: float,
    direction_source: str = "goal_pos",
    early_waypoints: int = 3,
    min_goal_offset: Optional[int] = None,
    max_goal_offset: Optional[int] = None,
    min_goal_pos_dist: Optional[float] = None,
    max_goal_pos_dist: Optional[float] = None,
    max_direction_angle_deg: Optional[float] = 90.0,
    max_endpoint_goal_dist: Optional[float] = None,
    filter_goal_heading: bool = True,
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
    obs = prepare_obs_image(obs_raw, transform, device)

    candidates_by_direction: Dict[str, list] = {
        "left": [],
        "right": [],
        "forward": [],
    }
    min_offset = (
        dataset.min_action_distance + 1
        if min_goal_offset is None
        else max(1, int(min_goal_offset))
    )
    max_offset = min(
        int(max_goal_dist),
        dataset.max_dist_cat,
        dataset.max_action_distance - 1,
    )
    if max_goal_offset is not None:
        max_offset = min(max_offset, int(max_goal_offset))

    for goal_offset in range(min_offset, max_offset + 1):
        if not (dataset.min_action_distance < goal_offset < dataset.max_action_distance):
            continue
        goal_time = curr_time + goal_offset * dataset.waypoint_spacing
        if goal_time >= traj_len:
            continue

        curr_yaw = traj_data["yaw"][curr_time]
        curr_yaw = float(np.asarray(curr_yaw).squeeze())
        goal_pos_metric = to_local_coords(
            np.asarray(traj_data["position"][goal_time][:2], dtype=np.float32)[None],
            np.asarray(traj_data["position"][curr_time][:2], dtype=np.float32),
            curr_yaw,
        )[0]
        goal_pos_metric_dist = float(np.linalg.norm(goal_pos_metric[:2]))
        actions, goal_pos = dataset._compute_actions(traj_data, curr_time, goal_time)
        if min_goal_pos_dist is not None and goal_pos_metric_dist < float(min_goal_pos_dist):
            continue
        if max_goal_pos_dist is not None and goal_pos_metric_dist > float(max_goal_pos_dist):
            continue
        actions_torch = torch.as_tensor(actions.astype(np.float32), dtype=torch.float32)
        if dataset.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        goal_yaw = traj_data["yaw"][goal_time]
        goal_yaw = float(np.asarray(goal_yaw).squeeze())
        goal_yaw_heading_rad = np.arctan2(
            np.sin(goal_yaw - curr_yaw), np.cos(goal_yaw - curr_yaw)
        )
        goal_yaw_heading_angle_deg = float(np.rad2deg(goal_yaw_heading_rad))

        positions = traj_data["position"]
        if goal_time + 1 < traj_len:
            heading_world = positions[goal_time + 1] - positions[goal_time]
        else:
            heading_world = positions[goal_time] - positions[goal_time - 1]
        heading_world = np.asarray(heading_world[:2], dtype=np.float32)
        heading_norm = float(np.linalg.norm(heading_world))
        if heading_norm > 1e-6:
            heading_local = heading_world.dot(yaw_rotmat(curr_yaw)[:2, :2])
            goal_image_heading_angle_deg = float(
                np.rad2deg(np.arctan2(heading_local[1], heading_local[0]))
            )
        else:
            goal_image_heading_angle_deg = goal_yaw_heading_angle_deg
        goal_angle_deg = float(np.rad2deg(np.arctan2(goal_pos[1], goal_pos[0])))
        final_waypoint = actions_torch[-1, :2].numpy()
        target_final_angle_deg = float(
            np.rad2deg(np.arctan2(final_waypoint[1], final_waypoint[0]))
        )
        early_count = max(1, min(int(early_waypoints), actions_torch.shape[0]))
        early_waypoint = actions_torch[:early_count, :2].mean(dim=0).numpy()
        target_early_angle_deg = float(
            np.rad2deg(np.arctan2(early_waypoint[1], early_waypoint[0]))
        )
        if direction_source == "goal_pos":
            direction_angle_deg = goal_angle_deg
        elif direction_source == "target_final":
            direction_angle_deg = target_final_angle_deg
        elif direction_source == "target_early":
            direction_angle_deg = target_early_angle_deg
        else:
            raise ValueError(f"Unsupported direction_source: {direction_source}")
        if (
            max_direction_angle_deg is not None
            and abs(direction_angle_deg) > max_direction_angle_deg
        ):
            continue
        endpoint_goal_dist = float(np.linalg.norm(final_waypoint - goal_pos[:2]))
        if (
            max_endpoint_goal_dist is not None
            and endpoint_goal_dist > max_endpoint_goal_dist
        ):
            continue
        name = direction_name(direction_angle_deg, angle_threshold_deg)
        if filter_goal_heading and not heading_matches_goal_side(
            name, goal_image_heading_angle_deg, angle_threshold_deg
        ):
            continue
        try:
            goal_raw = safe_load_image(dataset, traj_name, goal_time)
        except FileNotFoundError:
            continue
        candidates_by_direction[name].append(
            {
                "source": {
                    "dataset_index": int(sample_index),
                    "trajectory": traj_name,
                    "curr_time": int(curr_time),
                    "goal_time": int(goal_time),
                    "goal_offset": int(goal_offset),
                },
                "goal_time": int(goal_time),
                "goal_offset": int(goal_offset),
                "goal_raw": goal_raw.detach().cpu(),
                "goal": prepare_goal_image(goal_raw, transform, device),
                "target_action": actions_torch.detach().cpu(),
                "goal_pos": goal_pos.astype(np.float32),
                "goal_pos_dist": goal_pos_metric_dist,
                "goal_pos_metric": goal_pos_metric.astype(np.float32),
                "goal_pos_metric_dist": goal_pos_metric_dist,
                "goal_angle_deg": goal_angle_deg,
                "goal_image_heading_angle_deg": goal_image_heading_angle_deg,
                "goal_yaw_heading_angle_deg": goal_yaw_heading_angle_deg,
                "target_angle_deg": direction_angle_deg,
                "target_final_angle_deg": target_final_angle_deg,
                "target_early_angle_deg": target_early_angle_deg,
                "direction_source": direction_source,
                "endpoint_goal_dist": endpoint_goal_dist,
            }
        )

    selected = {}
    if candidates_by_direction["left"]:
        selected["left"] = max(
            candidates_by_direction["left"], key=lambda item: item["target_angle_deg"]
        )
    if candidates_by_direction["right"]:
        selected["right"] = min(
            candidates_by_direction["right"], key=lambda item: item["target_angle_deg"]
        )
    if candidates_by_direction["forward"]:
        selected["forward"] = min(
            candidates_by_direction["forward"],
            key=lambda item: abs(item["target_angle_deg"]),
        )

    if not selected:
        return None

    return {
        "base_obs_raw": obs_raw.detach().cpu(),
        "base_obs": obs,
        "base_source": {
            "dataset_index": int(sample_index),
            "trajectory": traj_name,
            "curr_time": int(curr_time),
        },
        "candidates": selected,
    }


def find_matched_directional_goal_set(
    dataset: ViNT_Dataset,
    transform: transforms.Compose,
    device: torch.device,
    angle_threshold_deg: float,
    scan_items: int,
    direction_source: str = "goal_pos",
    early_waypoints: int = 3,
    min_goal_offset: Optional[int] = None,
    max_goal_offset: Optional[int] = None,
    min_goal_pos_dist: Optional[float] = None,
    max_goal_pos_dist: Optional[float] = None,
    max_direction_angle_deg: Optional[float] = 90.0,
    max_endpoint_goal_dist: Optional[float] = None,
    filter_goal_heading: bool = True,
) -> Dict[str, Any]:
    log(
        "Scanning same-trajectory/same-time samples for matched "
        f"left/right/forward goals, up to {scan_items} items..."
    )
    limit = min(scan_items, len(dataset))
    for sample_index in range(limit):
        if sample_index % 100 == 0:
            log(f"Matched goal scan item {sample_index + 1}/{limit}")
        goal_set = build_matched_goal_set_for_index(
            dataset=dataset,
            sample_index=sample_index,
            transform=transform,
            device=device,
            angle_threshold_deg=angle_threshold_deg,
            direction_source=direction_source,
            early_waypoints=early_waypoints,
            min_goal_offset=min_goal_offset,
            max_goal_offset=max_goal_offset,
            min_goal_pos_dist=min_goal_pos_dist,
            max_goal_pos_dist=max_goal_pos_dist,
            max_direction_angle_deg=max_direction_angle_deg,
            max_endpoint_goal_dist=max_endpoint_goal_dist,
            filter_goal_heading=filter_goal_heading,
        )
        if goal_set is None:
            continue
        if {"left", "right", "forward"} <= set(goal_set["candidates"]):
            source = goal_set["base_source"]
            log(
                "Found matched left/right/forward goals at "
                f"dataset_index={source['dataset_index']}, "
                f"trajectory={source['trajectory']}, curr_time={source['curr_time']}"
            )
            return goal_set

    raise RuntimeError(
        "Could not find a same-trajectory/same-time sample with left/right/forward "
        "matched goals. Try increasing --scan-batches or lowering "
        "--angle-threshold-deg."
    )


def find_all_matched_directional_goal_sets(
    dataset: ViNT_Dataset,
    transform: transforms.Compose,
    device: torch.device,
    angle_threshold_deg: float,
    scan_items: int,
    direction_source: str = "goal_pos",
    early_waypoints: int = 3,
    min_goal_offset: Optional[int] = None,
    max_goal_offset: Optional[int] = None,
    min_goal_pos_dist: Optional[float] = None,
    max_goal_pos_dist: Optional[float] = None,
    max_direction_angle_deg: Optional[float] = 90.0,
    max_endpoint_goal_dist: Optional[float] = None,
    filter_goal_heading: bool = True,
    require_all_directions: bool = True,
) -> list[Dict[str, Any]]:
    required_directions = {"left", "right", "forward"}
    log(
        "Scanning same-trajectory/same-time samples for all matched "
        f"{'left/right/forward' if require_all_directions else 'directional'} "
        f"goal sets, up to {scan_items} items..."
    )
    goal_sets = []
    limit = min(scan_items, len(dataset))
    for sample_index in range(limit):
        if sample_index % 100 == 0:
            log(
                f"Matched goal scan item {sample_index + 1}/{limit}; "
                f"matched_sets={len(goal_sets)}"
            )
        goal_set = build_matched_goal_set_for_index(
            dataset=dataset,
            sample_index=sample_index,
            transform=transform,
            device=device,
            angle_threshold_deg=angle_threshold_deg,
            direction_source=direction_source,
            early_waypoints=early_waypoints,
            min_goal_offset=min_goal_offset,
            max_goal_offset=max_goal_offset,
            min_goal_pos_dist=min_goal_pos_dist,
            max_goal_pos_dist=max_goal_pos_dist,
            max_direction_angle_deg=max_direction_angle_deg,
            max_endpoint_goal_dist=max_endpoint_goal_dist,
            filter_goal_heading=filter_goal_heading,
        )
        if goal_set is None:
            continue
        if (
            required_directions <= set(goal_set["candidates"])
            if require_all_directions
            else bool(goal_set["candidates"])
        ):
            goal_sets.append(goal_set)

    if not goal_sets:
        target = "left/right/forward matched" if require_all_directions else "directional"
        raise RuntimeError(
            "Could not find any same-trajectory/same-time samples with "
            f"{target} goals. Try increasing --scan-batches or lowering "
            "--angle-threshold-deg."
        )
    return goal_sets


def find_directional_goal_set(
    dataloader: DataLoader,
    transform: transforms.Compose,
    device: torch.device,
    angle_threshold_deg: float,
    scan_batches: int,
) -> Dict[str, Any]:
    base_obs_raw = None
    base_obs = None
    base_source = None
    candidates = {}

    log(f"Scanning up to {scan_batches} batches for left/right/forward goal examples...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= scan_batches:
            break
        log(f"Directional scan batch {batch_idx + 1}/{scan_batches}")
        obs_raw, goal_raw, actions_raw, _, _, _, action_mask_raw = batch
        obs, goals, actions, _, action_mask = prepare_batch(batch, transform, device)
        valid_indices = torch.where(action_mask > 0.5)[0].detach().cpu().tolist()
        if not valid_indices:
            log(f"Directional scan batch {batch_idx + 1}: no valid actions, skipping")
            continue

        if base_obs is None:
            base_idx = valid_indices[0]
            base_obs_raw = obs_raw[base_idx].detach().cpu()
            base_obs = obs[base_idx : base_idx + 1]
            base_source = {"batch": batch_idx, "sample_in_batch": int(base_idx)}
            log(f"Selected base observation from batch {batch_idx}, sample {base_idx}")

        angles_deg = torch.rad2deg(final_angle(actions)).detach().cpu().tolist()
        for idx in valid_indices:
            name = direction_name(angles_deg[idx], angle_threshold_deg)
            if name not in candidates:
                candidates[name] = {
                    "goal_raw": goal_raw[idx].detach().cpu(),
                    "goal": goals[idx : idx + 1],
                    "target_action": actions[idx].detach().cpu(),
                    "target_angle_deg": float(angles_deg[idx]),
                    "source": {"batch": batch_idx, "sample_in_batch": int(idx)},
                }
                log(
                    f"Found {name} goal candidate at batch {batch_idx}, "
                    f"sample {idx}, angle={angles_deg[idx]:.1f} deg"
                )

        if base_obs is not None and {"left", "right", "forward"} <= set(candidates):
            log("Found all left/right/forward goal candidates.")
            break

    if base_obs is None:
        raise RuntimeError("Could not find a valid base observation.")
    if not candidates:
        raise RuntimeError("Could not find valid goal candidates.")

    return {
        "base_obs_raw": base_obs_raw,
        "base_obs": base_obs,
        "base_source": base_source,
        "candidates": candidates,
    }


def plot_directional_goal_samples(
    model: NoMaD,
    config: dict,
    device: torch.device,
    directional_set: Dict[str, Any],
    num_samples: int,
    output_path: Path,
    title: str,
    precomputed_trajectories: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    ordered_names = [
        name
        for name in ["left", "forward", "right"]
        if name in directional_set["candidates"]
    ]
    if not ordered_names:
        raise RuntimeError("No directional goals are available for visualization.")

    fig, axes = plt.subplots(
        2, len(ordered_names), figsize=(5.2 * len(ordered_names), 8.0)
    )
    if len(ordered_names) == 1:
        axes = np.expand_dims(axes, axis=1)
    fig.suptitle(title)

    metadata = {
        "base_source": directional_set["base_source"],
        "num_samples": num_samples,
        "directions": {},
    }

    for col_idx, name in enumerate(ordered_names):
        log(f"Rendering directional visualization for {name} goal...")
        candidate = directional_set["candidates"][name]
        if precomputed_trajectories is None:
            outputs = run_model(
                model,
                directional_set["base_obs"],
                candidate["goal"],
                config["len_traj_pred"],
                num_samples,
                device,
            )
            gc_samples = outputs["gc_actions"].reshape(
                num_samples, config["len_traj_pred"], 2
            )
            gc_np = gc_samples.detach().cpu().numpy()
        else:
            gc_np = np.asarray(precomputed_trajectories[name], dtype=np.float32)
        target = candidate["target_action"].numpy()

        image_pair = np.concatenate(
            [
                resize_for_display(directional_set["base_obs_raw"][-3:]),
                resize_for_display(candidate["goal_raw"]),
            ],
            axis=1,
        )
        direction_source = candidate.get("direction_source", "unknown")
        goal_angle = candidate.get("goal_angle_deg")
        goal_image_heading_angle = candidate.get("goal_image_heading_angle_deg")
        target_final_angle = candidate.get("target_final_angle_deg")
        target_early_angle = candidate.get("target_early_angle_deg")
        angle_suffix = ""
        if goal_angle is not None:
            angle_suffix = (
                f"\nlabel_by={direction_source}, goal_pos={goal_angle:.1f} deg"
            )
            if goal_image_heading_angle is not None:
                angle_suffix += f", goal_image_heading={goal_image_heading_angle:.1f} deg"
            if target_early_angle is not None:
                angle_suffix += f", target_early={target_early_angle:.1f} deg"
            if target_final_angle is not None:
                angle_suffix += f", target_final={target_final_angle:.1f} deg"

        axes[0, col_idx].imshow(image_pair)
        axes[0, col_idx].set_title(
            f"column {col_idx + 1}: {name} goal\nobs | goal{angle_suffix}",
            fontsize=9,
        )
        axes[0, col_idx].axis("off")

        plot_trajs_and_points(
            ax=axes[1, col_idx],
            list_trajs=np.concatenate([gc_np, target[None]], axis=0),
            list_points=[np.array([0.0, 0.0]), candidate["goal_pos"][:2]],
            traj_colors=(["green"] * len(gc_np)) + ["magenta"],
            point_colors=["black", "yellow"],
            traj_labels=None,
            point_labels=None,
            traj_alphas=([0.22] * len(gc_np)) + [1.0],
            point_alphas=[1.0, 1.0],
            quiver_freq=0,
        )
        goal_xy = candidate["goal_pos"][:2]
        arrow_len = max(0.35, 0.08 * float(np.linalg.norm(goal_xy)))
        axes[1, col_idx].quiver(
            [0.0],
            [0.0],
            [arrow_len],
            [0.0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.008,
        )
        goal_heading_deg = candidate.get(
            "goal_image_heading_angle_deg", candidate.get("goal_angle_deg", 0.0)
        )
        goal_heading_rad = np.deg2rad(goal_heading_deg)
        axes[1, col_idx].quiver(
            [goal_xy[0]],
            [goal_xy[1]],
            [arrow_len * np.cos(goal_heading_rad)],
            [arrow_len * np.sin(goal_heading_rad)],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="gold",
            width=0.008,
        )
        axes[1, col_idx].annotate(
            "robot",
            xy=(0.0, 0.0),
            xytext=(5, 5),
            textcoords="offset points",
            color="black",
            fontsize=8,
        )
        axes[1, col_idx].annotate(
            "goal",
            xy=(goal_xy[0], goal_xy[1]),
            xytext=(5, 5),
            textcoords="offset points",
            color="darkgoldenrod",
            fontsize=8,
        )
        selected_angle = candidate.get("target_angle_deg")
        axes[1, col_idx].set_title(
            f"column {col_idx + 1}: {name}\n"
            f"{len(gc_np)} sampled trajectories"
        )

        point_dist = np.linalg.norm(gc_np - target[None], axis=-1)
        metadata["directions"][name] = {
            "source": candidate.get("source"),
            "direction_source": direction_source,
            "selected_direction_angle_deg": selected_angle,
            "goal_angle_deg": candidate.get("goal_angle_deg"),
            "goal_image_heading_angle_deg": candidate.get("goal_image_heading_angle_deg"),
            "goal_yaw_heading_angle_deg": candidate.get("goal_yaw_heading_angle_deg"),
            "target_early_angle_deg": candidate.get("target_early_angle_deg"),
            "target_final_angle_deg": candidate.get("target_final_angle_deg"),
            "endpoint_goal_dist": candidate.get("endpoint_goal_dist"),
            "mean_ade_to_candidate_label": float(point_dist.mean(axis=1).mean()),
            "mean_fde_to_candidate_label": float(point_dist[:, -1].mean()),
        }

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return metadata


def timestamp_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{suffix}"


def ensure_output_dir(output_dir: str, *subdirs: Optional[str]) -> Path:
    path = Path(output_dir)
    for subdir in subdirs:
        if subdir is not None:
            path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tensor_image_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    return np.moveaxis(image.numpy(), 0, -1)


def last_obs_image(obs_tensor: torch.Tensor) -> np.ndarray:
    return tensor_image_to_numpy(obs_tensor[-3:])


def resize_for_display(
    image_tensor: torch.Tensor, size: Tuple[int, int] = (160, 120)
) -> np.ndarray:
    image = TF.resize(image_tensor.detach().cpu(), size[::-1])
    return tensor_image_to_numpy(image)
