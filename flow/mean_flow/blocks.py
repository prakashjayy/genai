# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import (
    Downsample2D,
    Upsample2D,
)
from mean_flow.resnet import ResnetBlock2D


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnDownBlock2D":
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or "conv"  # default to 'conv'
        return AttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            downsample_type=downsample_type,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    mid_block_only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = 1,
    dropout: float = 0.0,
):
    
    if mid_block_type == "UNetMidBlock2D":
        return UNetMidBlock2D(
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=0,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=False,
        )
    elif mid_block_type is None:
        return None
    else:
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "AttnUpBlock2D":
        if add_upsample is False:
            upsample_type = None
        else:
            upsample_type = upsample_type or "conv"  # default to 'conv'
        return AttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            upsample_type=upsample_type,
        )
    elif up_block_type == "ResnetUpsampleBlock2D":
        return ResnetUpsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock2D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size, in_channels,
        height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        if resnet_time_scale_shift == "spatial":
            raise ValueError("spatial is not supported")
            # resnets = [
            #     ResnetBlockCondNorm2D(
            #         in_channels=in_channels,
            #         out_channels=in_channels,
            #         temb_channels=temb_channels,
            #         eps=resnet_eps,
            #         groups=resnet_groups,
            #         dropout=dropout,
            #         time_embedding_norm="spatial",
            #         non_linearity=resnet_act_fn,
            #         output_scale_factor=output_scale_factor,
            #     )
            # ]
        else:
            resnets = [
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            ]
        attentions = []

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            if resnet_time_scale_shift == "spatial":
                # resnets.append(
                #     ResnetBlockCondNorm2D(
                #         in_channels=in_channels,
                #         out_channels=in_channels,
                #         temb_channels=temb_channels,
                #         eps=resnet_eps,
                #         groups=resnet_groups,
                #         dropout=dropout,
                #         time_embedding_norm="spatial",
                #         non_linearity=resnet_act_fn,
                #         output_scale_factor=output_scale_factor,
                #     )
                # )
                raise ValueError("spatial is not supported")
            else:
                resnets.append(
                    ResnetBlock2D(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, delta_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb, delta_emb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if attn is not None:
                    hidden_states = attn(hidden_states)
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                if attn is not None:
                    hidden_states = attn(hidden_states)
                hidden_states = resnet(hidden_states, temb, delta_emb)

        return hidden_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, delta_emb: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        output_states = ()

        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, delta_emb)
            else:
                hidden_states = resnet(hidden_states, temb, delta_emb)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class AttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.downsample_type = downsample_type

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if downsample_type == "conv":
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        elif downsample_type == "resnet":
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        down=True,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        delta_emb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, delta_emb)
                hidden_states = attn(hidden_states, **cross_attention_kwargs)
                output_states = output_states + (hidden_states,)
            else:
                hidden_states = resnet(hidden_states, temb, delta_emb)
                hidden_states = attn(hidden_states, **cross_attention_kwargs)
                output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb=temb, delta_emb=delta_emb)
                else:
                    hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        delta_emb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, delta_emb)
            else:
                hidden_states = resnet(hidden_states, temb, delta_emb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

class ResnetUpsampleBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        skip_time_act: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=skip_time_act,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        skip_time_act=skip_time_act,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        delta_emb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, delta_emb)
            else:
                hidden_states = resnet(hidden_states, temb, delta_emb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb, delta_emb)

        return hidden_states


class AttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        upsample_type: str = "conv",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.upsample_type = upsample_type

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        elif upsample_type == "resnet":
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        delta_emb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, delta_emb)
                hidden_states = attn(hidden_states)
            else:
                hidden_states = resnet(hidden_states, temb, delta_emb)
                hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.upsample_type == "resnet":
                    hidden_states = upsampler(hidden_states, temb=temb, delta_emb=delta_emb)
                else:
                    hidden_states = upsampler(hidden_states)

        return hidden_states
