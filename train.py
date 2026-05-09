import argparse
import os
import time

import click
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from flownav.data.vint_dataset import ViNT_Dataset
from flownav.models.nomad import DenseNetwork, NoMaD
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from flownav.training.loop import main_loop
from warmup_scheduler import GradualWarmupScheduler


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(config: dict) -> bool:
    return int(config.get("rank", 0)) == 0


def rank0_echo(config: dict, message: str, **style_kwargs) -> None:
    if is_main_process(config):
        click.echo(click.style(message, **style_kwargs))


def setup_distributed(config: dict) -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = bool(config.get("use_ddp", False)) or world_size > 1
    if use_ddp and world_size == 1:
        raise RuntimeError("DDP is enabled; launch with torchrun so WORLD_SIZE > 1.")
    config["distributed"] = use_ddp
    config["rank"] = int(os.environ.get("RANK", "0"))
    config["local_rank"] = int(os.environ.get("LOCAL_RANK", "0"))
    config["world_size"] = world_size

    if not use_ddp:
        return config

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs.")
    if not is_dist_initialized():
        dist.init_process_group(backend=config.get("dist_backend", "nccl"))
    torch.cuda.set_device(config["local_rank"])
    return config


def cleanup_distributed() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()


def broadcast_from_rank0(value):
    if not is_dist_initialized():
        return value
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def dataloader_kwargs(config: dict, split: str) -> dict:
    num_workers_key = "num_workers" if split == "train" else "eval_num_workers"
    num_workers = int(config.get(num_workers_key, config.get("num_workers", 0)))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", torch.cuda.is_available())),
        "drop_last": bool(config.get(f"{split}_drop_last", split == "train")),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 4))
    return kwargs


def main(config: dict) -> None:
    train_model = bool(config["train"])
    distributed = bool(config.get("distributed", False))
    rank = int(config.get("rank", 0))
    local_rank = int(config.get("local_rank", 0))
    main_process = is_main_process(config)

    # Set up the device
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]
        if distributed and len(config["gpu_ids"]) == 1 and config["world_size"] > 1:
            config["gpu_ids"] = list(range(config["world_size"]))
        if not distributed:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(x) for x in config["gpu_ids"]]
            )
        if main_process:
            if distributed:
                message = (
                    f">> Using DDP with {config['world_size']} processes; "
                    f"visible GPUs: {config['gpu_ids']}"
                )
            else:
                message = f">> Using GPUs: {config['gpu_ids']}"
            click.echo(click.style(message, fg="green", bold=True))
    else:
        rank0_echo(config, ">> No GPUs available, using CPU", fg="red", bold=True)
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    if "seed" in config:
        seed = int(config["seed"]) + rank
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # Set up the transformation for the dataset (from ImageNet)
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the data
    train_dataset = []
    test_dataloaders = {}
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        for data_split_type in ["train", "test"]:
            if data_split_type == "train" and not train_model:
                continue
            if data_split_type == "test" and distributed and not main_process:
                continue
            if data_split_type in data_config:
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
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
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset
    train_loader = None
    train_sampler = None
    if train_model:
        train_dataset = ConcatDataset(train_dataset)
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config["world_size"],
                rank=rank,
                shuffle=True,
                drop_last=bool(config.get("train_drop_last", True)),
            )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **dataloader_kwargs(config, "train"),
        )
        rank0_echo(
            config,
            (
                f">> Loaded {len(train_dataset)} training samples "
                f"({config['batch_size']} per GPU/process)"
            ),
            fg="cyan",
            bold=True,
        )
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]
    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset=dataset,
            batch_size=config["eval_batch_size"],
            shuffle=bool(config.get("eval_shuffle", False)),
            **dataloader_kwargs(config, "eval"),
        )
        rank0_echo(
            config,
            f">> Loaded {len(dataset)} test samples for {dataset_type}",
            fg="cyan",
            bold=True,
        )

    # Create the model
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
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    optimizer = None
    scheduler = None
    if train_model:
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config["epochs"]
        )
        scheduler = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=1,
            total_epoch=config["warmup_epochs"],
            after_scheduler=scheduler,
        )

    # Load Depth-Anything pre-trained weights before an optional FlowNav checkpoint,
    # so resumed/evaluated checkpoints keep their trained depth encoder weights.
    checkpoint = torch.load(
        config["depth"]["weights_path"],
        map_location=device,
    )
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

    # Load pre-trained model if specified
    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        click.echo(
            click.style(
                f">> Loading pre-trained model from {load_project_folder}",
                fg="yellow",
            )
        )
        if os.path.isdir(load_project_folder):
            latest_path = os.path.join(load_project_folder, "latest.pth")
        elif os.path.isfile(load_project_folder):
            latest_path = load_project_folder
        else:
            click.echo(
                click.style(
                    f">> Could not find pre-trained model at {load_project_folder}",
                    fg="red",
                )
            )
        latest_checkpoint = torch.load(latest_path, map_location=device)
        if "model" in latest_checkpoint:
            model.load_state_dict(
                strip_module_prefix(latest_checkpoint["model"]), strict=True
            )
        else:
            model.load_state_dict(strip_module_prefix(latest_checkpoint), strict=True)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
        if train_model and optimizer is not None and "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"])
        if train_model and scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"])

    # Multi-GPU setup
    model = model.to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(config.get("find_unused_parameters", False)),
        )
    elif len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model)

    # Run the training loop
    main_loop(
        train_model=train_model,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        goal_mask_prob=config["goal_mask_prob"],
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
        use_amp=config.get("use_amp", False),
        train_sampler=train_sampler,
        is_main_process=main_process,
        distributed=distributed,
    )
    if train_model:
        message = f">> Training completed. Model saved to {config['project_folder']}"
    else:
        message = f">> Evaluation completed. Logs saved to {config['project_folder']}"
    rank0_echo(config, message, fg="green", bold=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="flownav/config/flownav.yaml",
        type=str,
        help="Path to the config file",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()

    # Load the configuration
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{this_file_dir}/flownav/config/flownav.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f) or {}

    # Create the project folder and update the configuration
    config.update(user_config)
    config = setup_distributed(config)
    if is_main_process(config):
        click.echo(click.style(f">> Using config file: {args.config}", fg="yellow"))

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S") if is_main_process(config) else None
    timestamp = broadcast_from_rank0(timestamp)
    config["run_name"] += "_" + timestamp
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    if is_main_process(config):
        os.makedirs(config["project_folder"], exist_ok=True)
        click.echo(
            click.style(
                f">> Project folder created: {config['project_folder']}", fg="yellow"
            )
        )
    if is_dist_initialized():
        dist.barrier()

    # Set wandb configuration
    config["use_wandb"] = bool(config["use_wandb"] and is_main_process(config))
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity=config["entity"],
        )
        wandb.save(args.config, policy="now")
        wandb.run.name = config["run_name"]
        if wandb.run:
            wandb.config.update(config)

    try:
        main(config)
    finally:
        cleanup_distributed()
