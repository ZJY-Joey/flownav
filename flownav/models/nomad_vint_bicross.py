from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_anything_v2.dinov2 import DINOv2
from efficientnet_pytorch import EfficientNet

from flownav.models.attention import PositionalEncoding
from flownav.models.nomad_vint import replace_bn_with_gn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = self.norm1(query + attn_out)
        return self.norm2(query + self.ff(query))


class NoMaD_ViNT_BiCross(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        depth_cfg: Optional[dict] = {},
    ) -> None:
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.depth_cfg = depth_cfg

        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError

        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size
            )
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(
                self.num_goal_features, self.goal_encoding_size
            )
        else:
            self.compress_goal_enc = nn.Identity()

        self.depth_enc_str = depth_cfg["depth_encoder"]
        self.depth_encoder = DINOv2(model_name=self.depth_enc_str)
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        self.depth_layer_idx = depth_cfg["dino_layer_idx"][self.depth_enc_str]
        self.depth_pool_dim = depth_cfg["pool_dim"]
        self.depth_enc_dim = depth_cfg["out_dim"][self.depth_enc_str]
        self.num_depth_features = self.depth_enc_dim * self.depth_pool_dim
        if self.num_depth_features != self.goal_encoding_size:
            self.compress_depth_enc = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.depth_pool_dim),
                nn.Flatten(),
                nn.Linear(self.num_depth_features, self.goal_encoding_size),
            )
        else:
            self.compress_depth_enc = nn.Identity()

        self.obs_depth_positional_encoding = PositionalEncoding(
            self.obs_encoding_size, max_seq_len=self.context_size + 2
        )
        self.goal_positional_encoding = PositionalEncoding(
            self.obs_encoding_size, max_seq_len=1
        )

        ff_dim = mha_ff_dim_factor * self.obs_encoding_size
        self.goal_to_obs_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.obs_encoding_size,
                    nhead=mha_num_attention_heads,
                    dim_feedforward=ff_dim,
                )
                for _ in range(mha_num_attention_layers)
            ]
        )
        self.obs_to_goal_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.obs_encoding_size,
                    nhead=mha_num_attention_heads,
                    dim_feedforward=ff_dim,
                )
                for _ in range(mha_num_attention_layers)
            ]
        )

    def _encode_goal(self, goal_img: torch.Tensor) -> torch.Tensor:
        goal_encoding = self.goal_encoder.extract_features(goal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        return goal_encoding.unsqueeze(1)

    def _encode_depth(self, obs_img: torch.Tensor) -> torch.Tensor:
        depth_inp = obs_img[:, 3 * self.context_size :, :, :]
        depth_inp = F.pad(depth_inp, (1, 1, 1, 1), mode="constant", value=0)
        dpt_enc_all = self.depth_encoder.get_intermediate_layers(
            depth_inp, self.depth_layer_idx, return_class_token=False
        )
        dpt_enc_last = dpt_enc_all[-1].permute(0, 2, 1)
        depth_encoding = self.compress_depth_enc(dpt_enc_last.float())
        return depth_encoding.unsqueeze(1)

    def _encode_obs(self, obs_img: torch.Tensor) -> torch.Tensor:
        batch_size = obs_img.shape[0]
        obs_frames = torch.split(obs_img, 3, dim=1)
        obs_frames = torch.concat(obs_frames, dim=0)
        obs_encoding = self.obs_encoder.extract_features(obs_frames)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape(
            (self.context_size + 1, batch_size, self.obs_encoding_size)
        )
        return torch.transpose(obs_encoding, 0, 1)

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        obs_tokens = self._encode_obs(obs_img)
        depth_token = self._encode_depth(obs_img)
        goal_token = self._encode_goal(goal_img)

        obs_depth_tokens = torch.cat((obs_tokens, depth_token), dim=1)
        obs_depth_tokens = self.obs_depth_positional_encoding(obs_depth_tokens)
        goal_token = self.goal_positional_encoding(goal_token)

        goal_to_obs = goal_token
        for layer in self.goal_to_obs_layers:
            goal_to_obs = layer(goal_to_obs, obs_depth_tokens)

        obs_to_goal = obs_depth_tokens
        for layer in self.obs_to_goal_layers:
            obs_to_goal = layer(obs_to_goal, goal_token)

        fused_tokens = torch.cat((goal_to_obs, obs_to_goal), dim=1)
        fused_cond = torch.mean(fused_tokens, dim=1)

        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(obs_img.device).bool().unsqueeze(-1)
            obs_only_cond = torch.mean(obs_depth_tokens, dim=1)
            fused_cond = torch.where(goal_mask, obs_only_cond, fused_cond)

        return fused_cond
