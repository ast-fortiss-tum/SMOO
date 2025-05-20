import gc
from typing import Callable

import torch
from torch import Tensor, nn

from .repae.models.autoencoder import vae_models
from .repae.models.sit import SiT_models
from .repae.utils import load_encoders


class REPAEManipulator:

    _device: torch.device

    """Models used."""
    _vae: nn.Module
    _model: nn.Module

    """Model specific parameters."""
    _batch_size: int
    _latent_size: int
    _in_channels: int
    _latents_scale: Tensor
    _latents_bias: Tensor

    """Auxiliary lambdas for easy callings."""
    _embed_x: Callable[[Tensor], Tensor]
    _embed_y: Callable[[list[Tensor]], Tensor]
    _embed_t: Callable[[list[Tensor]], Tensor]

    def __init__(
        self,
        image_resolution: int,
        model_file: str,
        vae: str = "f16d32",
        model: str = "SiT-XL/1",
        encoder: str = "dinov2-vit-b",
        num_classes: int = 1000,
        batch_size: int = 8,
    ) -> None:
        """
        Initialize the manipulator based on REPA-E diffusion models.

        :param image_resolution: Image resolution for generation.
        :param model_file: Model file to load weights from.
        :param vae: The type of VAE model to use.
        :param model: The type of model to use.
        :param encoder: The type of encoder to use.
        :param num_classes: Number of classes in the dataset.
        :param batch_size: Batch size to use for generation of samples..
        raises: NotImplementedError If elements are not supported.
        """
        self._prepare_cuda()
        self._batch_size = batch_size
        state_dict = torch.load(model_file)

        """Prepare VAE model."""
        if vae == "f8d4":
            self._latent_size = image_resolution // 8
            self._in_channels = 4
        elif vae == "f16d32":
            self._latent_size = image_resolution // 16
            self._in_channels = 32
        else:
            raise NotImplementedError(f"VAE of type {vae} is not supported")
        self._vae = vae_models[vae]().to(self._device)
        self._vae.load_state_dict(state_dict["vae"])
        self._vae.eval()

        """Prepare SiT model."""
        encoders, _, _ = load_encoders(encoder, "cpu", image_resolution)
        z_dims = [encoder.embed_dim for encoder in encoders] if encoder != "None" else [0]
        del encoders
        gc.collect()

        self._model = SiT_models[model](
            input_size=self._latent_size,
            in_channels=self._in_channels,
            num_classes=num_classes,
            class_dropout_prob=0.1,
            z_dims=z_dims,
            encoder_depth=8,
            bn_momentum=0.1,
            fused_attn=True,
            qk_norm=False,
        ).to(self._device)
        self._model.load_state_dict(state_dict["ema"])
        self._model.eval()

        self._latents_scale = (
            state_dict["ema"]["bn.running_var"]
            .rsqrt()
            .view(1, self._in_channels, 1, 1)
            .to(self._device)
        )
        self._latents_bias = (
            state_dict["ema"]["bn.running_mean"].view(1, self._in_channels, 1, 1).to(self._device)
        )

        """Clean up."""
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        """Define Embedding lambdas"""
        self._embed_y = lambda y: self._model.y_embedder(
            torch.tensor(y, device=self._device), self._model.training
        )

    def manipulate(
        self,
        candidates: Tensor,  # TODO: Make those candidate lists
        candidate_ys: Tensor,
        weights: Tensor,
        weights_y: Tensor,
        t_cur: Tensor,
        step: float,
    ) -> Tensor:
        assert (
            candidates.shape[:1] == weights.shape
        ), f"ERROR, candidates have shape {candidates.shape}, weights have shape {weights.shape}"
        assert (
            candidate_ys.shape[:1] == weights_y.shape
        ), f"ERROR, candidates have shape {candidate_ys.shape}, weights have shape {weights_y.shape}"

        x_manip = torch.sum(
            candidates * weights[:, None, None, None], dim=0, keepdim=True
        )  # N x X -> 1 x X
        y_manip = torch.sum(
            candidate_ys * weights_y[:, None], dim=0, keepdim=True
        )  # N x Y -> 1 x Y

        x_cur = self._sample(t_cur, x_manip, y_manip, step)
        return x_cur

    def _sample(
        self,
        t: Tensor,
        x: Tensor,
        y: Tensor,
        step: float,
        cfg_scale: float = 1.5,
        guidance_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> Tensor:
        """
        Sampling new outputs based on euler_sampler in REPA-E repo.

        :param cfg_scale: CFG scale for conditions in the sampling.
        :param guidance_bounds: Guidance bounds for conditions in the sampling.
        :returns: The sampled outputs for the current timestep.
        """
        with torch.no_grad():
            t_curr = torch.ones(y.shape[0], device=self._device) * t
            if cfg_scale > 1.0 and guidance_bounds[1] >= t >= guidance_bounds[0]:
                model_input = torch.cat([x] * 2, dim=0)
                y_null = self._embed_y([1000] * y.shape[0])
                y_curr = torch.cat((y, y_null), dim=0)
                t_curr = torch.cat([t_curr] * 2, dim=0)
            else:
                model_input, y_curr = x, y

            d_cur = self._model.partial_inference(x=model_input, t=t_curr, y=y_curr)

            if cfg_scale > 1.0 and guidance_bounds[1] >= t >= guidance_bounds[0]:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next = x + step * d_cur
            return x_next

    def get_diff_steps(
        self, class_labels: list[int], n_steps: int = 50
    ) -> tuple[list[Tensor], Tensor]:
        """
        Get latent information for all diffusion steps.

        :param class_labels: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising.
        :returns: A list of latent vectors through denoising and the class embedding.
        """
        x_cur = torch.randn(
            len(class_labels),
            self._in_channels,
            self._latent_size,
            self._latent_size,
            device=self._device,
        )
        xs = [x_cur]

        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)
        y_cur = self._embed_y(class_labels)
        xs.extend(self.finish_diffusion(x_cur, y_cur, 0, t_steps, return_all=True))
        return xs, y_cur

    def finish_diffusion(
        self,
        x_cur: Tensor,
        y_cur: Tensor,
        t_cur_index: int,
        t_steps: Tensor,
        return_all: bool = False,
    ) -> Tensor:
        """
        Finish diffusion process from specific time step.

        :param x_cur: Current latent vector.
        :param y_cur: Current class latent vector.
        :param t_cur_index: Current time step index in range.
        :param t_steps: The time step range.
        :param return_all: Whether to return all diffusion steps or just the last one.
        :returns: The finished diffusion process or the last latent vector.
        """
        xs = []
        for t_cur, t_next in zip(t_steps[t_cur_index:-1], t_steps[t_cur_index + 1 :]):
            x_cur = self._sample(t=t_cur, x=x_cur, y=y_cur, step=t_next - t_cur)
            xs.append(x_cur)
        return xs if return_all else xs[-1]

    def get_image(self, z: Tensor) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :return: The decoded image.
        """
        element = self._vae.decode((z / self._latents_scale) + self._latents_bias).sample
        element = (element + 1) / 2.0
        element = torch.clamp(255.0 * element, 0, 255).permute(0, 2, 3, 1).type(torch.uint8)
        return element

    def _prepare_cuda(self) -> None:
        """Prepare cuda environment, as done in the REPA-E repository."""
        torch.backends.cuda.matmul.allow_tf32 = True
        assert (
            torch.cuda.is_available()
        ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
        torch.set_grad_enabled(False)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
