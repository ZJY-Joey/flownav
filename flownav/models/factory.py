import torch
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from flownav.models.nomad import DenseNetwork, NoMaD
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from flownav.models.nomad_vint_bicross import NoMaD_ViNT_BiCross
from flownav.models.nomad_vint_bicross_hybrid import NoMaD_ViNT_BiCrossHybrid


def build_vision_encoder(config: dict) -> torch.nn.Module:
    vision_encoder_name = config.get("vision_encoder", "nomad_vint")
    common_kwargs = {
        "obs_encoder": config.get("obs_encoder", "efficientnet-b0"),
        "obs_encoding_size": config["encoding_size"],
        "context_size": config["context_size"],
        "mha_num_attention_heads": config["mha_num_attention_heads"],
        "mha_num_attention_layers": config["mha_num_attention_layers"],
        "mha_ff_dim_factor": config["mha_ff_dim_factor"],
        "depth_cfg": config["depth"],
    }

    if vision_encoder_name == "nomad_vint":
        vision_encoder = NoMaD_ViNT(**common_kwargs)
    elif vision_encoder_name in {"nomad_vint_bicross", "bicross"}:
        vision_encoder = NoMaD_ViNT_BiCross(**common_kwargs)
    elif vision_encoder_name in {"nomad_vint_bicross_hybrid", "bicross_hybrid"}:
        vision_encoder = NoMaD_ViNT_BiCrossHybrid(**common_kwargs)
    else:
        raise ValueError(f"Unsupported vision_encoder: {vision_encoder_name}")

    return replace_bn_with_gn(vision_encoder)


def build_nomad_model(config: dict) -> NoMaD:
    vision_encoder = build_vision_encoder(config)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    return NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )


def load_depth_encoder_weights(
    model: NoMaD, weights_path: str, device: torch.device
) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
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
