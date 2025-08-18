import math
from copy import deepcopy
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ._internal.models.sit import SiT


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
            *self.control_embed.parameters(),
            *self.zero_layers.parameters(),
        ]

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        control: Tensor,
        cfg: float,
        guidance_bounds: tuple[float, float],
        y_null: Optional[Tensor] = None,
        timesteps: int = 50,
    ) -> Tensor:
        """
        Forward pass of SiTControlNet.

        :param x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images UNNORMALIZED).
        :param y: (N) tensor of class labels.
        :param control: (N, *S) tensor of control tokens to use for the forward pass.
        :param cfg: Classifier free guidance scale.
        :param guidance_bounds: For which timesteps guidance should be permitted.
        :param y_null: The null-class tensor if guidance is used.
        :param timesteps: Provide the amount of denoising timesteps.
        :returns: The results of the forward pass.
        """
        device, dtype = x.device, x.dtype
        t_start, t_end = 1, 0
        time_inputs = torch.linspace(t_start, t_end, timesteps + 1, device=device, dtype=dtype)[1:]

        y_embed = y  # or embed if needed
        control = control.to(device=device, dtype=dtype)

        cond = cfg > 1.0 and guidance_bounds[1] >= t_start > t_end >= guidance_bounds[0]
        if cond and y_null is not None:
            x = x.repeat(2, *([1] * (x.ndim - 1)))
            y_embed = torch.cat((y_embed, y_null), dim=0)
            control = control.repeat(2, *([1] * (control.ndim - 1)))

        control_embed = self.control_embed(control.flatten(1)).unsqueeze(1)
        for t_cur, t_next in zip(time_inputs[:-1], time_inputs[1:]):
            x_initial = x
            t = t_cur.expand(x.size(0))

            x = self.base_model.x_embedder(x) + self.base_model.pos_embed
            control_tokens, _ = self.control_projection(x, control_embed, control_embed)

            t_embed = self.base_model.t_embedder(t)
            c = t_embed + y_embed

            x_control = x + control_tokens
            controlnet_outputs = checkpoint(
                self._control_forward, x_control, c, use_reentrant=False
            )

            x = self._backbone_forward(controlnet_outputs, x, c)
            x = self.base_model.final_layer(x, c)
            x = self.base_model.unpatchify(x)  # updated sample

            if cond:
                x_cond, x_uncond = x.chunk(2)
                x = x_uncond + cfg * (x_cond - x_uncond)  # Here x has shape (1, ...).

            x = x_initial + (t_next - t_cur) * x  # Here x is broadcasted to (2, ...) if cond.

        # If we are operating with conditioning the batch was doubled for null conditioning.
        # We only want to return one batch, therefore we take the first.
        return x.chunk(2)[0] if cond else x

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
        c = t_embed + y  # (N, D)

        control = control.to(device=x.device, dtype=x.dtype)
        control_embed = self.control_embed(control.flatten(1)).unsqueeze(1)  # (N, 1, D)
        control_tokens, _ = self.control_projection(x, control_embed, control_embed)  # (N, T, D)

        x_control = x + control_tokens

        controlnet_outputs = self._control_forward(x_control, c)
        x = self._backbone_forward(controlnet_outputs, x, c)

        x = self.base_model.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.base_model.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def _control_forward(self, x_control: Tensor, c: Tensor) -> list[Tensor]:
        """
        The forward pass of the ControlNet only.

        :param x_control: (N, C, H, W) tensor of control conditioned x.
        :param c: The conditioning tensor.
        :returns: A list of outputs; one Tensor for each block in the ControlNet.
        """
        controlnet_outputs = []
        for block, zero_layer in zip(self.control_layers, self.zero_layers):
            x_control = block(x_control, c)
            control_residual = zero_layer(x_control)
            controlnet_outputs.append(control_residual)
        return controlnet_outputs

    def _backbone_forward(self, controlnet_outputs: list[Tensor], x: Tensor, c: Tensor) -> Tensor:
        """
        A forward pass for the frozen backbone, with ControlNet activations.

        :param controlnet_outputs: The activations of the ControlNet.
        :param x: The current latent vector (unconditioned).
        :param c: The conditioning tensor.
        :returns: The output of the frozen backbone.
        """
        for block in self.base_model.blocks:
            controlnet_out: Union[Tensor, float] = (
                controlnet_outputs.pop(0) if controlnet_outputs else 0.0
            )
            x = block(x + controlnet_out, c)  # (N, T, D)
        return x
