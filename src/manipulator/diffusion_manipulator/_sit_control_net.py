import math
from copy import deepcopy
from typing import Optional, Union

import torch
from torch import Tensor, nn

from ._internal.models.sit import SiT


def mean_flat(x: Tensor) -> Tensor:
    """
    Take the mean over all non-batch dimensions.

    :param x: A tensor to average over.
    :returns: The average of the tensor over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class ZeroLinear(nn.Linear):
    """A Zero-initialized Linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initializer the ZeroLinear layer.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param bias: Whether to use a bias.
        """
        super().__init__(in_features, out_features, bias)
        nn.init.zeros_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)


class SiTControlNet(nn.Module):
    """A ControlNet implementation for SiT."""

    def __init__(
        self, sit_model: SiT, control_shape: tuple[int, ...], train_frac: Optional[float] = None
    ) -> None:
        """
        Initialize the SiTControlNet.

        :param sit_model: The underlying SiT model to control.
        :param control_shape: The shape of the control tokens to use. Should not include the batch dimension.
        :param train_frac: The fraction of blocks to be trainable (Default: Encoder Depth of SiT.).
        """
        super().__init__()
        if train_frac is None:
            self.cutoff = sit_model.encoder_depth
        else:
            assert 0 < train_frac <= 1, "Train fraction must be between 0 and 1."
            self.cutoff = int(len(sit_model.blocks) * train_frac)
            assert self.cutoff > 0, "Train fraction must be greater than 0."

        """Define Control Blocks."""
        self.control_layers = nn.ModuleList(deepcopy(sit_model.blocks[: self.cutoff]))
        module_list = []
        for block in self.control_layers:
            in_features = block.mlp.fc1.in_features
            out_features = block.mlp.fc2.out_features
            module_list.append(ZeroLinear(in_features, out_features))
        self.zero_layers = nn.ModuleList(module_list)

        """Define projection of control input."""
        in_size = sit_model.blocks[0].attn.qkv.in_features
        self.control_embed = ZeroLinear(math.prod(control_shape), in_size)
        self.control_projection = nn.MultiheadAttention(in_size, num_heads=8, batch_first=True)

        """Copy SiT functionality."""
        self.base_model = sit_model
        for param in self.base_model.parameters():
            param.requires_grad_(False)  # Freeze parameters

    def trainable_parameters(self) -> list[nn.Parameter]:
        """
        Parse all trainable parameters in the model.

        :returns: A list of trainable parameters in the model (Control-Layers, Control-Projection, Zero-Layers, Control-Embed).
        """
        return [
            *self.control_layers.parameters(),
            *self.control_projection.parameters(),
            *self.zero_layers.parameters(),
            *self.control_embed.parameters(),
        ]

    def _time_input(
        self, time_input: Optional[Tensor], normalized_x: Tensor, weighting: str, path_type: str
    ) -> Tensor:
        """
        Generates time input for the forward pass.

        :param time_input: The time input tensor, if provided.
        :param normalized_x: The normalized input tensor.
        :param weighting: Weighting scheme for time input (uniform or lognormal).
        :param path_type: Interpolant parth (linear or cosine).
        :returns: The time input tensor.
        :raises NotImplementedError: If the weighting scheme is not supported.
        """
        if time_input is None:
            if weighting == "uniform":
                time_input = torch.rand((normalized_x.shape[0], 1, 1, 1))
            elif weighting == "lognormal":
                rnd_normal = torch.randn((normalized_x.shape[0], 1, 1, 1))
                sigma = rnd_normal.exp()
                if path_type == "linear":
                    time_input = sigma / (1 + sigma)
                elif path_type == "cosine":
                    time_input = 2 / torch.pi * torch.atan(sigma)
            else:
                raise NotImplementedError(f"Weighting scheme {weighting} not implemented.")
        elif time_input.ndim == 1:
            time_input = time_input[:, None, None, None]
        return time_input

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        weighting: str,
        path_type: str,
        control_map: Tensor,
        time_input: Optional[Tensor] = None,
        noises: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of SiTControlNet, integrating the loss function computation

        :param x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images UNNORMALIZED).
        :param y: (N,) tensor of class labels.
        :param weighting: Weighting scheme for time input (uniform or lognormal).
        :param path_type: Interpolant parth (linear or cosine).
        :param control_map: (N, *S) tensor of control tokens to use for the forward pass.
        :param time_input: Provide an optional tensor of timesteps to use for the forward pass, otherwise sample from a distribution.
        :param noises: Provide an optional tensor of noises to use for the forward pass, otherwise sample from a distribution.
        :returns: The results of the forward pass, including the denoising loss, the projection loss, and the timesteps used for the forward pass.
        """
        x = self.base_model.bn(x)
        """Get standard input noise."""
        time_input = self._time_input(time_input, x, weighting, path_type)
        time_input = time_input.to(device=x.device, dtype=x.dtype)

        noises = (
            torch.randn_like(x) if noises is None else noises.to(device=x.device, dtype=x.dtype)
        )
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.base_model.interpolant(
            time_input, path_type=path_type
        )

        model_input = alpha_t * x + sigma_t * noises
        x = self.base_model.x_embedder(model_input) + self.base_model.pos_embed
        # model_target = d_alpha_t * x + d_sigma_t * noises

        """Get control input."""
        control_map = control_map.to(device=x.device, dtype=x.dtype)
        control_embed = self.control_embed(control_map.flatten(1)).unsqueeze(1)  # (N, D)
        control_tokens, _ = self.control_projection(x, control_embed, control_embed)  # (N, T, D)

        x_control = x + control_tokens  # The input for ControlNet

        """Get standard conditioning of the LDM -> time + yembedding."""
        t_embed = self.base_model.t_embedder(time_input.flatten())  # (N, D)
        y_embed = y  # self.base_model.y_embedder(y, self.training)  # (N, D)
        c = t_embed + y_embed  # (N, D)

        """Pass control input through copied encoder blocks of the ControlNet."""
        controlnet_outputs = []
        for block, zero_layer in zip(self.control_layers, self.zero_layers):
            x_control = block(x_control, c)
            control_residual = zero_layer(x_control)
            controlnet_outputs.append(control_residual)

        """Pass input through frozen LDM."""
        for i, block in enumerate(self.base_model.blocks):
            controlnet_out: Union[Tensor, float] = (
                controlnet_outputs.pop(0) if controlnet_outputs else 0.0
            )
            x = block(x + controlnet_out, c)

        x = self.base_model.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.base_model.unpatchify(x)  # (N, out_channels, H, W)
        # denoising_loss = mean_flat((x - model_target) ** 2)
        return x

    @torch.no_grad()
    def inference(self, x: Tensor, t: Tensor, y: Tensor, control: Tensor) -> Tensor:
        """
        Do a diffusion step in the ControlNet.

        :param x: The noisy latent representations of images.
        :param t: The timesteps to use for the diffusion step.
        :param y: The class labels to use for the diffusion step.
        :param control: The control map to use for the diffusion step.
        :returns: The one step denoised latent representations of images.
        """
        x = self.base_model.x_embedder(x) + self.base_model.pos_embed

        t_embed = self.base_model.t_embedder(t)  # (N, D)
        # y = self.base_model.y_embedder(y, self.training)  # (N, D)
        c = t_embed + y  # (N, D)

        control = control.to(device=x.device, dtype=x.dtype)
        control_embed = self.control_embed(control.flatten(1)).unsqueeze(1)  # (N, 1, D)
        control_tokens, _ = self.control_projection(x, control_embed, control_embed)  # (N, T, D)

        x_control = x + control_tokens

        controlnet_outputs = []
        for block, zero_layer in zip(self.control_layers, self.zero_layers):
            x_control = block(x_control, c)
            control_residual = zero_layer(x_control)
            controlnet_outputs.append(control_residual)

        for block in self.base_model.blocks:
            controlnet_out: Union[Tensor, float] = (
                controlnet_outputs.pop(0) if controlnet_outputs else 0.0
            )
            x = block(x + controlnet_out, c)  # (N, T, D)

        x = self.base_model.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.base_model.unpatchify(x)  # (N, out_channels, H, W)
        return x
