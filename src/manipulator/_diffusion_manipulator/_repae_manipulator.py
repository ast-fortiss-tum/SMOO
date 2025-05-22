import gc
from typing import Callable, Union

import torch
from torch import Tensor, nn

from ._diffusion_candidate import DiffusionCandidate, DiffusionCandidateList
from .repae.models.autoencoder import vae_models
from .repae.models.sit import SiT_models
from .repae.utils import load_encoders
from .. import Manipulator


class REPAEManipulator(Manipulator):

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
        :param batch_size: Batch size to use for generation of samples.
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
        candidates: DiffusionCandidateList,
        weights_x: Tensor,
        weights_y: Tensor,
        return_manipulation_history: bool = False,
    ) -> Union[Tensor, tuple[Tensor, DiffusionCandidate]]:
        """
        Manipulate the diffusion processes of candidates.

        :param candidates: Candidates to manipulate.
        :param weights_x: Weights to manipulate diffusion process.
        :param weights_y: Weights to manipulate class embeddings.
        :param return_manipulation_history: Whether to return a candidate representing the manipulation history.
        :returns: The resulting diffusion result.
        """
        n_steps = len(candidates[0].xt) - 1
        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)  # Define step range.

        """Now we step through the diffusion process and manipulate accordingly."""
        manip_xt, manip_y = candidates[0].xt, candidates[0].class_embedding
        """The manipulation process."""
        # Here we manipulate class embeddings through diffusion.
        weighted_class_embeddings = candidates.class_embeddings * weights_y[...,None]  # N x D x Y
        y_manip = torch.sum(weighted_class_embeddings, dim=0, keepdim=True).float()  # N x D x Y -> 1 x D x Y

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Here we manipulate the diffusion steps.
            weighted_candidates = candidates.xts[:,i,...] * weights_x[:,i][:, None, None, None]  # N x X
            x_manip = torch.sum(weighted_candidates, dim=0, keepdim=True).float()  # N x X -> 1 x X
            x_manip = (manip_xt[i] + x_manip) / 2

            x_cur = self._sample(t=t_cur, x=x_manip, y=y_manip[:,i,...], step=t_next - t_cur)

            # Log progress.
            manip_y[i] = y_manip[:,i,...]
            manip_xt[i + 1] = x_cur

        """Return a candidate with the manipulation history if wanted."""
        if return_manipulation_history:
            manip_candidate = DiffusionCandidate(class_embedding=manip_y, xt=manip_xt)
            return x_cur, manip_candidate
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

        :param t: The current time step.
        :param x: The current state of the diffusion process.
        :param y: The current class embedding of the diffusion process.
        :param step: The step size.
        :param cfg_scale: CFG scale for conditions in the sampling.
        :param guidance_bounds: Guidance bounds for conditions in the sampling.
        :returns: The sampled outputs for the current timestep.
        """
        with torch.no_grad():
            t_curr = torch.ones(y.shape[0], device=self._device) * t
            cond = cfg_scale > 1.0 and guidance_bounds[1] >= t >= guidance_bounds[0]
            if cond:
                model_input = torch.cat([x] * 2, dim=0)
                y_null = self._embed_y([1000] * y.shape[0])
                y_curr = torch.cat((y, y_null), dim=0)
                t_curr = torch.cat([t_curr] * 2, dim=0)
            else:
                model_input, y_curr = x, y

            d_cur = self._model.partial_inference(x=model_input, t=t_curr, y=y_curr)

            if cond:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next = x + step * d_cur
            return x_next

    def get_diff_steps(
        self, class_labels: list[int], n_steps: int = 50
    ) -> tuple[Tensor, Tensor]:
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
        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)
        y_cur = self._embed_y(class_labels)

        xs = []
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1 :]):
            x_cur = self._sample(t=t_cur, x=x_cur, y=y_cur, step=t_next - t_cur)
            xs.append(x_cur)
        return torch.stack(xs).squeeze(), y_cur

    def get_image(self, z: Tensor) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :return: The decoded image.
        """
        element = self._vae.decode((z / self._latents_scale) + self._latents_bias).sample
        element = (element + 1) / 2.0
        element = torch.clamp(element, 0, 1)
        return element

    def _prepare_cuda(self) -> None:
        """Prepare cuda environment, as done in the REPA-E repository."""
        torch.backends.cuda.matmul.allow_tf32 = True
        assert (
            torch.cuda.is_available()
        ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
        torch.set_grad_enabled(False)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
