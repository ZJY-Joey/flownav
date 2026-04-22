import argparse
import io
import json
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from PIL import Image
from torchvision import transforms

from flownav.data.data_utils import img_path_to_data
from flownav.models.nomad import DenseNetwork, NoMaD
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from flownav.training.utils import cluster_trajectory_samples, model_output, to_numpy
from flownav.visualizing.plot import plot_trajs_and_points

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    ROS2_BAG_AVAILABLE = True
except ImportError:
    ROS2_BAG_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    ROS2_LIVE_AVAILABLE = True
except ImportError:
    ROS2_LIVE_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlowNav RGB waypoint inference from files, a ROS2 bag, or live ROS2 topics."
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
        help=(
            "Checkpoint file or run directory. "
            "If omitted, uses `load_run` from the config."
        ),
    )
    parser.add_argument(
        "--obs-images",
        nargs="+",
        default=None,
        help=(
            "Observation RGB frames in temporal order. "
            "Expected count is context_size + 1."
        ),
    )
    parser.add_argument(
        "--bag-path",
        default=None,
        help="Path to a ROS2 bag directory. Used instead of --obs-images.",
    )
    parser.add_argument(
        "--ros-live",
        action="store_true",
        help="Subscribe to a live ROS2 image topic and display overlay in real time.",
    )
    parser.add_argument(
        "--image-topic",
        default="/camera/camera/color/image_raw/compressed",
        help="Image topic inside the ROS2 bag or live ROS2 graph.",
    )
    parser.add_argument(
        "--storage-id",
        default="sqlite3",
        help="rosbag2 storage id, usually sqlite3 or mcap.",
    )
    parser.add_argument(
        "--goal-image",
        default=None,
        help="Goal RGB frame path. Required for bag/live modes.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of sampled trajectories for UC and GC outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="inference_outputs",
        help="Directory to save visualization and numeric outputs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force a device such as cuda:0 or cpu. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--inference-hz",
        type=float,
        default=6.0,
        help="Live ROS2 inference frequency.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.75,
        help="Scale factor for the live OpenCV window.",
    )
    parser.add_argument(
        "--window-name",
        default="FlowNav Overlay",
        help="OpenCV window title for live visualization.",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="Optional output mp4 path for live overlay recording.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=0.41,
        help="Camera height above ground for waypoint projection.",
    )
    parser.add_argument(
        "--fisheye-f",
        type=float,
        default=790.0,
        help="Projection focal length.",
    )
    parser.add_argument(
        "--fisheye-cx",
        type=float,
        default=None,
        help="Projection principal point cx. Defaults to image_width / 2.",
    )
    parser.add_argument(
        "--fisheye-cy",
        type=float,
        default=None,
        help="Projection principal point cy. Defaults to image_height / 2.",
    )
    parser.add_argument(
        "--fisheye-k",
        type=float,
        default=0.0,
        help="Projection distortion coefficient.",
    )
    parser.add_argument(
        "--live-mode",
        choices=["gc_cluster", "uc_cluster", "gc_mean", "gc_first", "uc_mean", "uc_first"],
        default="gc_cluster",
        help="Which predicted trajectory to project in live mode.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.35,
        help="Distance threshold for trajectory clustering in action space.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_checkpoint(config: dict, checkpoint_arg: Optional[str]) -> str:
    checkpoint = checkpoint_arg
    if checkpoint is None:
        if "load_run" not in config:
            raise ValueError(
                "No --checkpoint was provided and config does not define `load_run`."
            )
        checkpoint = os.path.join("logs", config["load_run"])

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
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    obs_frames = torch.split(obs_stack, 3, dim=1)
    obs_frames = [normalize(frame) for frame in obs_frames]
    obs_stack = torch.cat(obs_frames, dim=1)
    goal_batch = normalize(goal_batch)
    return obs_stack, goal_batch


def load_rgb_tensor(image_path: str, image_size: Tuple[int, int]) -> torch.Tensor:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img_path_to_data(image_path, image_size)


def pil_to_model_tensor(image: Image.Image, image_size: Tuple[int, int]) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image)
    bgr_array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr_array)
    if not success:
        raise RuntimeError("Failed to encode PIL image for preprocessing.")
    return img_path_to_data(io.BytesIO(encoded.tobytes()), image_size)


def decode_ros_image(message: Any, message_type: str) -> Image.Image:
    if message_type == "sensor_msgs/msg/CompressedImage":
        image_array = np.frombuffer(message.data, dtype=np.uint8)
        decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode compressed image from ROS2 data.")
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    if message_type == "sensor_msgs/msg/Image":
        image_array = np.frombuffer(message.data, dtype=np.uint8)
        channels = max(message.step // message.width, 1)
        image_array = image_array.reshape(message.height, message.width, channels)

        if message.encoding in {"rgb8", "8UC3"}:
            rgb = image_array[:, :, :3]
        elif message.encoding == "bgr8":
            rgb = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_BGR2RGB)
        elif message.encoding == "rgba8":
            rgb = cv2.cvtColor(image_array[:, :, :4], cv2.COLOR_RGBA2RGB)
        elif message.encoding == "bgra8":
            rgb = cv2.cvtColor(image_array[:, :, :4], cv2.COLOR_BGRA2RGB)
        elif message.encoding in {"mono8", "8UC1"}:
            rgb = cv2.cvtColor(image_array[:, :, 0], cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported ROS image encoding: {message.encoding}")
        return Image.fromarray(rgb)

    raise ValueError(f"Unsupported ROS message type: {message_type}")


def project_waypoints_to_fisheye_image_with_polygon_new(
    waypoints: np.ndarray,
    intrinsic_params: List[float],
    image_path: Any,
    save_path: str = "projected_waypoints_fisheye.jpg",
    camera_height: float = 0.41,
    save_fig: bool = False,
    color: Optional[Tuple[int, int, int]] = None,
    base_alpha: float = 0.15,
    text_to_visualize: Optional[str] = None,
    plotted_number: Optional[int] = None,
) -> np.ndarray:
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints should be of shape (N, 2)")
    if len(intrinsic_params) != 4:
        raise ValueError("intrinsic_params should be [f, cx, cy, k]")

    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported image_path type: {type(image_path)}")

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width = img.shape[:2]
    f, cx, cy, k = intrinsic_params
    path_width = 0.22
    offsets_left, offsets_right = [], []

    for i in range(len(waypoints) - 1):
        p1, p2 = waypoints[i], waypoints[i + 1]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        dir_unit = direction / norm
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(p1 + path_width * normal)
        offsets_right.append(p1 - path_width * normal)

    if len(waypoints) > 1:
        direction = waypoints[-1] - waypoints[-2]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            dir_unit = direction / norm
            normal = np.array([-dir_unit[1], dir_unit[0]])
            offsets_left.append(waypoints[-1] + path_width * normal)
            offsets_right.append(waypoints[-1] - path_width * normal)

    offsets_left = np.array(offsets_left)
    offsets_right = np.array(offsets_right)

    def project_points_fisheye(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0, 2))
        num_points = points.shape[0]
        ego_points = np.zeros((num_points, 3))
        ego_points[:, :2] = points
        cam_translation = np.array([-0.1, 0, camera_height])
        rotation = np.array(
            [
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ]
        )
        cam_points = (rotation @ (ego_points - cam_translation).T).T
        valid_mask = cam_points[:, 2] > 0
        cam_points = cam_points[valid_mask]
        if cam_points.shape[0] == 0:
            return np.zeros((0, 2))

        x = cam_points[:, 0] / cam_points[:, 2]
        y = cam_points[:, 1] / cam_points[:, 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(r)
        theta_d = theta * (1 + k * theta**2)

        scaling = np.ones_like(r)
        mask = r > 1e-8
        scaling[mask] = theta_d[mask] / r[mask]
        x_distorted = x * scaling
        y_distorted = y * scaling
        u = f * x_distorted + cx
        v = f * y_distorted + cy
        return np.column_stack((u, v))

    points_center = project_points_fisheye(waypoints)
    points_left = project_points_fisheye(offsets_left)
    points_right = project_points_fisheye(offsets_right)

    if len(points_left) > 1 and len(points_right) > 1 and waypoints[-1, 0] > 1.0:
        polygon = np.vstack([points_left, points_right[::-1]]).astype(np.int32)
        nav_green = (65, 121, 76) if color is None else color
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        gradient_mask = np.zeros((height, width), dtype=np.float32)
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            normalized_y = y_coords / height
            alpha_values = base_alpha + 0.45 * (normalized_y ** 1.5)
            gradient_mask[y_coords, x_coords] = alpha_values

        overlay = img.copy()
        overlay[mask > 0] = nav_green
        for c in range(3):
            img[:, :, c] = (
                img[:, :, c] * (1 - gradient_mask) + overlay[:, :, c] * gradient_mask
            )

        if len(points_left) > 0 and len(points_right) > 0 and waypoints[-1, 0] > 2.0:
            bar_width = 5
            bar_color = (
                int(nav_green[0] * 0.7),
                int(nav_green[1] * 0.7),
                int(nav_green[2] * 0.7),
            )
            left_points = points_left.astype(np.int32)
            valid_left = left_points[
                (left_points[:, 0] >= 0)
                & (left_points[:, 0] < width)
                & (left_points[:, 1] >= 0)
                & (left_points[:, 1] < height)
            ]
            if len(valid_left) > 1:
                cv2.polylines(img, [valid_left], False, bar_color, bar_width, cv2.LINE_AA)

            right_points = points_right.astype(np.int32)
            valid_right = right_points[
                (right_points[:, 0] >= 0)
                & (right_points[:, 0] < width)
                & (right_points[:, 1] >= 0)
                & (right_points[:, 1] < height)
            ]
            if len(valid_right) > 1:
                cv2.polylines(
                    img, [valid_right], False, bar_color, bar_width, cv2.LINE_AA
                )

    start_point = np.array([[width // 2, height - 1]])
    points_center = np.vstack((start_point, points_center))
    points_center = points_center[
        (points_center[:, 0] >= 0)
        & (points_center[:, 0] < width)
        & (points_center[:, 1] >= 0)
        & (points_center[:, 1] < height)
    ]

    if points_center.shape[0] > 1 and waypoints[-1, 0] > 2.0:
        nav_green = (65, 121, 76) if color is None else color
        for i in range(len(points_center) - 1):
            pt1 = tuple(points_center[i].astype(int))
            pt2 = tuple(points_center[i + 1].astype(int))
            y_avg = (pt1[1] + pt2[1]) / 2
            opacity = 0.1 + 0.2 * (y_avg / height)
            overlay = img.copy()
            cv2.line(
                overlay,
                pt1,
                pt2,
                (int(nav_green[0] * 0.8), int(nav_green[1] * 0.8), int(nav_green[2] * 0.8)),
                2,
                cv2.LINE_AA,
            )
            img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

    if text_to_visualize is not None and plotted_number is not None and color is not None:
        bar_width = 20
        bar_height = 45
        bar_x = 10
        bar_y = height - 10 - bar_height - (plotted_number * (bar_height + 2))
        color_bgr = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), color_bgr, -1)
        cv2.rectangle(
            img,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (255, 255, 255),
            2,
        )
        text_x = bar_x + bar_width + 10
        text_y = bar_y + bar_height - 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            text_to_visualize, font, font_scale, font_thickness
        )
        cv2.rectangle(
            img,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            text_to_visualize,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    if save_fig:
        cv2.imwrite(save_path, img)

    return img


def prepare_inputs_from_images(
    obs_image_paths: List[str],
    goal_image_path: str,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    obs_images = [load_rgb_tensor(path, image_size) for path in obs_image_paths]
    goal_image = load_rgb_tensor(goal_image_path, image_size)
    obs_stack = torch.cat(obs_images, dim=0).unsqueeze(0)
    goal_batch = goal_image.unsqueeze(0)

    viz_obs = np.moveaxis(obs_images[-1].numpy(), 0, -1)
    viz_goal = np.moveaxis(goal_image.numpy(), 0, -1)
    return obs_stack, goal_batch, viz_obs, viz_goal


def load_observations_from_bag(
    bag_path: str,
    image_topic: str,
    storage_id: str,
    expected_obs: int,
    image_size: Tuple[int, int],
) -> Tuple[List[torch.Tensor], np.ndarray, List[int]]:
    if not ROS2_BAG_AVAILABLE:
        raise ImportError(
            "ROS2 bag support requires rosbag2_py, rclpy, and rosidl_runtime_py in the active Python environment."
        )

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    if image_topic not in topic_types:
        available_topics = ", ".join(sorted(topic_types.keys()))
        raise ValueError(
            f"Image topic {image_topic} not found in bag. Available topics: {available_topics}"
        )

    message_type = topic_types[image_topic]
    if message_type not in {
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/msg/Image",
    }:
        raise ValueError(
            f"Unsupported image topic type {message_type} on {image_topic}."
        )

    msg_cls = get_message(message_type)
    selected_images: List[Image.Image] = []
    selected_timestamps_ns: List[int] = []

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic != image_topic:
            continue
        message = deserialize_message(data, msg_cls)
        selected_images.append(decode_ros_image(message, message_type))
        selected_timestamps_ns.append(timestamp_ns)

    if len(selected_images) < expected_obs:
        raise ValueError(
            f"Bag only provided {len(selected_images)} image frames on {image_topic}, but {expected_obs} are required."
        )

    selected_images = selected_images[-expected_obs:]
    selected_timestamps_ns = selected_timestamps_ns[-expected_obs:]
    obs_tensors = [pil_to_model_tensor(image, image_size) for image in selected_images]
    viz_obs = np.moveaxis(obs_tensors[-1].numpy(), 0, -1)
    return obs_tensors, viz_obs, selected_timestamps_ns


def prepare_inputs_from_bag(
    bag_path: str,
    image_topic: str,
    storage_id: str,
    expected_obs: int,
    goal_image_path: str,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, Dict[str, Any]]:
    obs_images, viz_obs, selected_timestamps_ns = load_observations_from_bag(
        bag_path=bag_path,
        image_topic=image_topic,
        storage_id=storage_id,
        expected_obs=expected_obs,
        image_size=image_size,
    )
    goal_image = load_rgb_tensor(goal_image_path, image_size)
    obs_stack = torch.cat(obs_images, dim=0).unsqueeze(0)
    goal_batch = goal_image.unsqueeze(0)
    viz_goal = np.moveaxis(goal_image.numpy(), 0, -1)

    source_info = {
        "bag_path": bag_path,
        "image_topic": image_topic,
        "storage_id": storage_id,
        "selected_obs_timestamps_ns": selected_timestamps_ns,
    }
    return obs_stack, goal_batch, viz_obs, viz_goal, source_info


def save_outputs(
    output_dir: str,
    source_info: Dict[str, Any],
    goal_image_path: str,
    uc_actions: np.ndarray,
    gc_actions: np.ndarray,
    gc_distance: np.ndarray,
    selected_uc: np.ndarray,
    selected_gc: np.ndarray,
    selected_uc_index: int,
    selected_gc_index: int,
    selected_uc_cluster_size: int,
    selected_gc_cluster_size: int,
    viz_obs: np.ndarray,
    viz_goal: np.ndarray,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "uc_actions.npy"), uc_actions)
    np.save(os.path.join(output_dir, "gc_actions.npy"), gc_actions)
    np.save(os.path.join(output_dir, "gc_distance.npy"), gc_distance)
    np.save(os.path.join(output_dir, "selected_uc.npy"), selected_uc)
    np.save(os.path.join(output_dir, "selected_gc.npy"), selected_gc)

    metadata = {
        **source_info,
        "goal_image": goal_image_path,
        "uc_actions_shape": list(uc_actions.shape),
        "gc_actions_shape": list(gc_actions.shape),
        "gc_distance_shape": list(gc_distance.shape),
        "gc_distance_values": gc_distance.reshape(-1).tolist(),
        "selected_uc_index": selected_uc_index,
        "selected_gc_index": selected_gc_index,
        "selected_uc_cluster_size": selected_uc_cluster_size,
        "selected_gc_cluster_size": selected_gc_cluster_size,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(18.5, 10.5))
    plot_trajs_and_points(
        ax=ax[0],
        list_trajs=np.concatenate(
            [uc_actions, gc_actions, selected_uc[None], selected_gc[None]], axis=0
        ),
        list_points=[np.array([0.0, 0.0])],
        traj_colors=(
            (["red"] * len(uc_actions))
            + (["green"] * len(gc_actions))
            + ["orange", "blue"]
        ),
        point_colors=["blue"],
        traj_labels=(
            (["UC samples"] * len(uc_actions))
            + (["GC samples"] * len(gc_actions))
            + ["selected UC"]
            + ["selected GC"]
        ),
        point_labels=["robot"],
        traj_alphas=(
            ([0.2] * len(uc_actions))
            + ([0.15] * len(gc_actions))
            + [1.0, 1.0]
        ),
        point_alphas=[1.0],
        quiver_freq=0,
    )
    ax[0].set_title(
        "UC(red) / GC(green) samples\n"
        f"selected UC(orange)={selected_uc_index} [{selected_uc_cluster_size}/{len(uc_actions)}], "
        f"selected GC(blue)={selected_gc_index} [{selected_gc_cluster_size}/{len(gc_actions)}]\n"
        f"GC distance mean={gc_distance.mean():.2f}"
    )
    ax[1].imshow(viz_obs)
    ax[1].set_title("Last observation frame")
    ax[2].imshow(viz_goal)
    ax[2].set_title("Goal frame")
    for axis in ax[1:]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "waypoint_visualization.png"))
    plt.close(fig)


def choose_live_waypoints(
    outputs: Dict[str, torch.Tensor], mode: str, cluster_threshold: float
) -> np.ndarray:
    if mode == "gc_cluster":
        waypoints = cluster_trajectory_samples(
            to_numpy(outputs["gc_actions"]),
            distance_threshold=cluster_threshold,
        )["selected_trajectory"]
    elif mode == "uc_cluster":
        waypoints = cluster_trajectory_samples(
            to_numpy(outputs["uc_actions"]),
            distance_threshold=cluster_threshold,
        )["selected_trajectory"]
    elif mode == "gc_first":
        waypoints = to_numpy(outputs["gc_actions"][0])
    elif mode == "gc_mean":
        waypoints = to_numpy(outputs["gc_actions"].mean(dim=0))
    elif mode == "uc_first":
        waypoints = to_numpy(outputs["uc_actions"][0])
    elif mode == "uc_mean":
        waypoints = to_numpy(outputs["uc_actions"].mean(dim=0))
    else:
        raise ValueError(f"Unsupported live mode: {mode}")
    return waypoints


class LiveOverlayWriter:
    def __init__(self, output_path: str, fps: float):
        self.output_path = output_path
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.writer is None:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            height, width = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        if self.writer is not None:
            self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class FlowNavLiveNode(Node):
    def __init__(
        self,
        model: NoMaD,
        goal_batch: torch.Tensor,
        config: dict,
        args: argparse.Namespace,
        device: torch.device,
    ):
        super().__init__("flownav_live_overlay")
        self.model = model
        self.goal_batch = goal_batch.to(device)
        self.config = config
        self.args = args
        self.device = device
        self.expected_obs = config["context_size"] + 1
        self.image_size = tuple(config["image_size"])
        self.obs_queue: deque[torch.Tensor] = deque(maxlen=self.expected_obs)
        self.frame_lock = threading.Lock()
        self.latest_rgb_frame: Optional[np.ndarray] = None
        self.latest_overlay_waypoints: Optional[np.ndarray] = None
        self.latest_gc_distance: Optional[float] = None
        self.latest_frame_time = 0.0
        self.latest_inference_time = 0.0
        self.inference_durations_ms: deque[float] = deque(maxlen=30)
        self.inference_timestamps: deque[float] = deque(maxlen=30)
        self.latest_inference_ms: Optional[float] = None
        self.latest_inference_hz: Optional[float] = None
        self.shutdown_requested = False
        self.window_name = args.window_name
        self.display_scale = args.display_scale
        self.video_writer = (
            LiveOverlayWriter(args.save_video, max(args.inference_hz, 1.0))
            if args.save_video
            else None
        )
        self.sub = self.create_subscription(
            CompressedImage,
            args.image_topic,
            self.image_callback,
            10,
        )
        self.worker = threading.Thread(target=self.inference_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"Subscribed to {args.image_topic}, waiting for {self.expected_obs} frames."
        )

    def image_callback(self, msg: CompressedImage) -> None:
        try:
            image_array = np.frombuffer(msg.data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                self.get_logger().warning("Failed to decode compressed image")
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_tensor = pil_to_model_tensor(Image.fromarray(frame_rgb), self.image_size)

            with self.frame_lock:
                self.obs_queue.append(frame_tensor)
                self.latest_rgb_frame = frame_rgb
                self.latest_frame_time = time.time()

            self.render_latest_frame()
        except Exception as e:
            self.get_logger().error(f"Image callback failed: {e}")

    def get_intrinsics(self, frame_rgb: np.ndarray) -> List[float]:
        height, width = frame_rgb.shape[:2]
        cx = self.args.fisheye_cx if self.args.fisheye_cx is not None else width / 2.0
        cy = self.args.fisheye_cy if self.args.fisheye_cy is not None else height / 2.0
        return [self.args.fisheye_f, cx, cy, self.args.fisheye_k]

    def render_latest_frame(self) -> None:
        with self.frame_lock:
            if self.latest_rgb_frame is None:
                return
            frame_rgb = self.latest_rgb_frame.copy()
            waypoints = (
                None
                if self.latest_overlay_waypoints is None
                else np.array(self.latest_overlay_waypoints, copy=True)
            )
            gc_distance = self.latest_gc_distance
            latest_inference_ms = self.latest_inference_ms
            latest_inference_hz = self.latest_inference_hz
            avg_inference_ms = (
                None
                if not self.inference_durations_ms
                else float(np.mean(self.inference_durations_ms))
            )

        if waypoints is not None:
            overlay_bgr = project_waypoints_to_fisheye_image_with_polygon_new(
                waypoints=waypoints,
                intrinsic_params=self.get_intrinsics(frame_rgb),
                image_path=frame_rgb,
                camera_height=self.args.camera_height,
                save_fig=False,
                color=(100, 230, 100),
                base_alpha=0.25,
                text_to_visualize=(
                    None if gc_distance is None else f"gc_dist={gc_distance:.2f}"
                ),
                plotted_number=0 if gc_distance is not None else None,
            )
        else:
            overlay_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        cv2.putText(
            overlay_bgr,
            f"mode={self.args.live_mode} frames={len(self.obs_queue)}/{self.expected_obs}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if gc_distance is not None:
            cv2.putText(
                overlay_bgr,
                f"gc_dist={gc_distance:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if latest_inference_ms is not None:
            cv2.putText(
                overlay_bgr,
                f"infer={latest_inference_ms:.1f}ms avg={avg_inference_ms:.1f}ms",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if latest_inference_hz is not None:
            cv2.putText(
                overlay_bgr,
                f"actual_hz={latest_inference_hz:.2f}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self.display_scale != 1.0:
            overlay_bgr = cv2.resize(
                overlay_bgr,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_LINEAR,
            )

        cv2.imshow(self.window_name, overlay_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.shutdown_requested = True
            rclpy.shutdown()

        if self.video_writer is not None:
            self.video_writer.write(overlay_bgr)

    def inference_loop(self) -> None:
        while rclpy.ok() and not self.shutdown_requested:
            try:
                now = time.time()
                interval = 1.0 / max(self.args.inference_hz, 1e-3)
                if now - self.latest_inference_time < interval:
                    time.sleep(0.01)
                    continue

                with self.frame_lock:
                    if len(self.obs_queue) < self.expected_obs:
                        obs_stack = None
                    else:
                        obs_stack = torch.cat(list(self.obs_queue), dim=0).unsqueeze(0)
                        self.latest_inference_time = now

                if obs_stack is None:
                    time.sleep(0.01)
                    continue

                goal_batch = self.goal_batch.clone()
                obs_stack, goal_batch = normalize_batch(obs_stack, goal_batch)
                obs_stack = obs_stack.to(self.device)
                goal_batch = goal_batch.to(self.device)

                infer_start = time.perf_counter()
                outputs = model_output(
                    model=self.model,
                    batch_obs_images=obs_stack,
                    batch_goal_images=goal_batch,
                    pred_horizon=self.config["len_traj_pred"],
                    action_dim=2,
                    num_samples=self.args.num_samples,
                    device=self.device,
                    use_wandb=False,
                )
                selected_waypoints = choose_live_waypoints(
                    outputs, self.args.live_mode, self.args.cluster_threshold
                )
                gc_distance = float(to_numpy(outputs["gc_distance"]).reshape(-1)[0])
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                now_ts = time.time()

                with self.frame_lock:
                    self.latest_overlay_waypoints = selected_waypoints
                    self.latest_gc_distance = gc_distance
                    self.inference_durations_ms.append(infer_ms)
                    self.inference_timestamps.append(now_ts)
                    self.latest_inference_ms = infer_ms
                    if len(self.inference_timestamps) >= 2:
                        total_dt = (
                            self.inference_timestamps[-1]
                            - self.inference_timestamps[0]
                        )
                        if total_dt > 1e-6:
                            self.latest_inference_hz = (
                                (len(self.inference_timestamps) - 1) / total_dt
                            )
                    avg_infer_ms = float(np.mean(self.inference_durations_ms))
                    actual_hz = self.latest_inference_hz
                self.get_logger().info(
                    f"[perf] infer={infer_ms:.1f}ms avg={avg_infer_ms:.1f}ms"
                    + (
                        ""
                        if actual_hz is None
                        else f" actual_hz={actual_hz:.2f}"
                    )
                )
            except Exception as e:
                self.get_logger().error(f"Live inference failed: {e}")
                time.sleep(0.1)

    def close(self) -> None:
        self.shutdown_requested = True
        if self.video_writer is not None:
            self.video_writer.close()
        cv2.destroyAllWindows()


def validate_input_mode(args: argparse.Namespace) -> str:
    mode_count = int(args.obs_images is not None) + int(args.bag_path is not None) + int(args.ros_live)
    if mode_count != 1:
        raise ValueError(
            "Choose exactly one mode: --obs-images, --bag-path, or --ros-live."
        )
    if args.obs_images is not None:
        return "images"
    if args.bag_path is not None:
        return "bag"
    return "live"


def run_snapshot_inference(
    args: argparse.Namespace,
    mode: str,
    config: dict,
    device: torch.device,
    model: NoMaD,
) -> None:
    expected_obs = config["context_size"] + 1
    image_size = tuple(config["image_size"])

    goal_image_path = args.goal_image
    if goal_image_path is None and mode != "images":
        raise ValueError("--goal-image is required for bag mode.")
    if goal_image_path is None:
        goal_image_path = args.obs_images[-1]
        print("No --goal-image provided; reusing the last observation frame as goal.")

    if mode == "images":
        if len(args.obs_images) != expected_obs:
            raise ValueError(
                f"Expected {expected_obs} observation images, got {len(args.obs_images)}."
            )
        obs_stack, goal_batch, viz_obs, viz_goal = prepare_inputs_from_images(
            args.obs_images,
            goal_image_path,
            image_size,
        )
        source_info = {"obs_images": args.obs_images}
    else:
        obs_stack, goal_batch, viz_obs, viz_goal, source_info = prepare_inputs_from_bag(
            bag_path=args.bag_path,
            image_topic=args.image_topic,
            storage_id=args.storage_id,
            expected_obs=expected_obs,
            goal_image_path=goal_image_path,
            image_size=image_size,
        )
        print(
            "Selected observation timestamps (ns): "
            + ", ".join(str(x) for x in source_info["selected_obs_timestamps_ns"])
        )

    obs_stack, goal_batch = normalize_batch(obs_stack, goal_batch)
    obs_stack = obs_stack.to(device)
    goal_batch = goal_batch.to(device)

    outputs = model_output(
        model=model,
        batch_obs_images=obs_stack,
        batch_goal_images=goal_batch,
        pred_horizon=config["len_traj_pred"],
        action_dim=2,
        num_samples=args.num_samples,
        device=device,
        use_wandb=False,
    )

    uc_actions = to_numpy(outputs["uc_actions"])
    gc_actions = to_numpy(outputs["gc_actions"])
    gc_distance = to_numpy(outputs["gc_distance"])
    uc_cluster = cluster_trajectory_samples(
        uc_actions, distance_threshold=args.cluster_threshold
    )
    gc_cluster = cluster_trajectory_samples(
        gc_actions, distance_threshold=args.cluster_threshold
    )

    save_outputs(
        output_dir=args.output_dir,
        source_info=source_info,
        goal_image_path=goal_image_path,
        uc_actions=uc_actions,
        gc_actions=gc_actions,
        gc_distance=gc_distance,
        selected_uc=uc_cluster["selected_trajectory"],
        selected_gc=gc_cluster["selected_trajectory"],
        selected_uc_index=uc_cluster["selected_index"],
        selected_gc_index=gc_cluster["selected_index"],
        selected_uc_cluster_size=len(uc_cluster["selected_cluster"]),
        selected_gc_cluster_size=len(gc_cluster["selected_cluster"]),
        viz_obs=viz_obs,
        viz_goal=viz_goal,
    )

    print(f"Saved outputs to: {args.output_dir}")
    print(f"UC trajectories shape: {uc_actions.shape}")
    print(f"GC trajectories shape: {gc_actions.shape}")
    print(f"GC distance predictions: {gc_distance.reshape(-1).tolist()}")


def run_live_inference(
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
    model: NoMaD,
) -> None:
    if not ROS2_LIVE_AVAILABLE:
        raise ImportError(
            "Live ROS2 mode requires rclpy and sensor_msgs in the active Python environment."
        )
    if args.goal_image is None:
        raise ValueError("--goal-image is required for --ros-live mode.")

    goal_tensor = load_rgb_tensor(args.goal_image, tuple(config["image_size"])).unsqueeze(0)
    rclpy.init(args=None)
    node = FlowNavLiveNode(
        model=model,
        goal_batch=goal_tensor,
        config=config,
        args=args,
        device=device,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    args = parse_args()
    mode = validate_input_mode(args)
    config = load_config(args.config)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)

    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")

    model = build_model(config, device)
    load_checkpoint(model, checkpoint_path, device)

    if mode == "live":
        run_live_inference(args, config, device, model)
    else:
        run_snapshot_inference(args, mode, config, device, model)


if __name__ == "__main__":
    main()
