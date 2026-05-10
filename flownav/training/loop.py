import os
from typing import Dict, Optional

import click
import torch
import torch.nn as nn
import wandb
from diffusers.training_utils import EMAModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from flownav.training.evaluate import evaluate
from flownav.training.train import train


def make_grad_scaler(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device="cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def state_dict_for_save(model: nn.Module) -> dict:
    return unwrap_model(model).state_dict()


def ema_state_dict(ema_model: EMAModel) -> dict:
    if hasattr(ema_model, "state_dict"):
        return ema_model.state_dict()
    return {"averaged_model": ema_model.averaged_model.state_dict()}


def load_ema_state_dict(ema_model: EMAModel, state_dict: dict) -> None:
    if hasattr(ema_model, "load_state_dict"):
        ema_model.load_state_dict(state_dict)
        return
    averaged_model_state = state_dict.get("averaged_model", state_dict)
    ema_model.averaged_model.load_state_dict(averaged_model_state)


def main_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Optional[Adam],
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_loader: Optional[DataLoader],
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    use_amp: bool = False,
    checkpoint_state: Optional[dict] = None,
    train_sampler: Optional[DistributedSampler] = None,
    is_main_process: bool = True,
    distributed: bool = False,
) -> None:
    # Set saving paths
    latest_path = os.path.join(project_folder, "latest.pth")

    # Create EMA model
    ema_model = EMAModel(model=unwrap_model(model), power=0.75)
    if checkpoint_state is not None and "ema" in checkpoint_state:
        load_ema_state_dict(ema_model, checkpoint_state["ema"])

    if not train_model:
        if not is_main_process:
            return
        for dataset_type in test_dataloaders:
            click.echo(
                click.style(
                    f"> Evaluating {dataset_type} dataset",
                    fg="blue",
                )
            )
            loader = test_dataloaders[dataset_type]
            evaluate(
                eval_type=dataset_type,
                ema_model=ema_model,
                dataloader=loader,
                transform=transform,
                device=device,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=current_epoch,
                print_log_freq=print_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                wandb_log_freq=wandb_log_freq,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        if use_wandb:
            wandb.log({})
        return

    assert optimizer is not None
    assert lr_scheduler is not None
    assert train_loader is not None
    scaler = make_grad_scaler(device, use_amp)
    if (
        checkpoint_state is not None
        and use_amp
        and scaler is not None
        and "scaler" in checkpoint_state
    ):
        scaler.load_state_dict(checkpoint_state["scaler"])

    # Run the epochs
    for epoch in range(current_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process:
            click.echo(
                click.style(
                    f"> Start epoch {epoch}/{epochs - 1}",
                    fg="magenta",
                )
            )
        train(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            dataloader=train_loader,
            transform=transform,
            device=device,
            goal_mask_prob=goal_mask_prob,
            project_folder=project_folder,
            epoch=epoch,
            print_log_freq=print_log_freq,
            wandb_log_freq=wandb_log_freq,
            image_log_freq=image_log_freq,
            num_images_log=num_images_log,
            use_wandb=use_wandb,
            alpha=alpha,
            use_amp=use_amp,
            scaler=scaler,
            is_main_process=is_main_process,
            distributed=distributed,
        )
        # Save the model, EMA model, optimizer, and scheduler
        if is_main_process:
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(ema_model.averaged_model.state_dict(), numbered_path)
            ema_latest_path = os.path.join(project_folder, "ema_latest.pth")
            torch.save(ema_model.averaged_model.state_dict(), ema_latest_path)

            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(state_dict_for_save(model), numbered_path)

            checkpoint = {
                "epoch": epoch,
                "model": state_dict_for_save(model),
                "ema": ema_state_dict(ema_model),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "config": {
                    "use_amp": use_amp,
                    "distributed": distributed,
                },
            }

            latest_optimizer_path = os.path.join(project_folder, "optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            latest_scheduler_path = os.path.join(project_folder, "scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

            latest_scaler_path = os.path.join(project_folder, "scaler_latest.pth")
            if use_amp and scaler is not None:
                scaler_state = scaler.state_dict()
                checkpoint["scaler"] = scaler_state
                torch.save(scaler_state, latest_scaler_path)

            torch.save(checkpoint, latest_path)

        # In case of evaluation
        if is_main_process and (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                click.echo(
                    click.style(
                        f"> Evaluating {dataset_type} dataset at epoch {epoch}",
                        fg="blue",
                    )
                )
                loader = test_dataloaders[dataset_type]
                evaluate(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    image_log_freq=image_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )

        # Log the current learning rate
        if use_wandb and is_main_process:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                },
                commit=False,
            )

        if lr_scheduler is not None:
            lr_scheduler.step()
        if distributed:
            torch.distributed.barrier()

    # Flush the last set of eval logs
    if use_wandb and is_main_process:
        wandb.log({})
