import gc
import logging
from typing import Callable, Union

import torch
from torch import Tensor, nn

from ._diffusion_candidate import DiffusionCandidateList
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
    _embed_y: Callable[[list[Tensor]], Tensor]
    _embed_t: Callable[[list[Tensor]], Tensor]

    _manip_function: Callable[..., DiffusionCandidateList]

    def __init__(
        self,
        model_file: str,
        vae: str = "f16d32",
        model: str = "SiT-XL/1",
        encoder: str = "dinov2-vit-b",
        image_resolution: int = 256,
        num_classes: int = 1000,
        batch_size: int = 0,
        manipulation_strategy: str = "base",
        device: Union[torch.device, None] = None,
    ) -> None:
        """
        Initialize the manipulator based on REPA-E diffusion models.

        :param model_file: Model file to load weights from.
        :param vae: The type of VAE model to use.
        :param model: The type of model to use.
        :param encoder: The type of encoder to use.
        :param image_resolution: Image resolution for generation.
        :param num_classes: Number of classes in the dataset.
        :param batch_size: Batch size to use for generation of samples (0 means all elements get taken).
        :param manipulation_strategy: Manipulation strategy to use.
        :param device: CUDA device to use if available.
        :raises ValueError: If manipulation_strategy is not supported.
        """
        self._prepare_cuda(device)
        self._batch_size = batch_size
        state_dict = torch.load(model_file, weights_only=False)

        """Prepare VAE model."""
        if vae == "f8d4":
            self._latent_size = image_resolution // 8
            self._in_channels = 4
        elif vae == "f16d32":
            self._latent_size = image_resolution // 16
            self._in_channels = 32
        else:
            raise NotImplementedError(f"VAE of type {vae} is not supported")
        self._vae = vae_models[vae]()
        self._vae.load_state_dict(state_dict["vae"])
        self._vae.eval()
        self._vae.to(self._device)

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
        )
        self._model.load_state_dict(state_dict["ema"])
        self._model.eval()
        self._model.to(self._device)

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
        ).detach()

        if manipulation_strategy == "base":
            self._manip_function = self._manip_aggregate_base
        elif manipulation_strategy == "new":
            self._manip_function = self._manip_aggregate_new
        else:
            raise ValueError(f"Strategy '{manipulation_strategy}' is not supported.'")

    def manipulate(
        self,
        candidates: DiffusionCandidateList,
        weights_x: Tensor,
        weights_y: Tensor,
        return_manipulation_history: bool = False,
    ) -> Union[Tensor, tuple[Tensor, DiffusionCandidateList]]:
        """
        Manipulate the diffusion processes of candidates.

        :param candidates: Candidates to manipulate.
        :param weights_x: Weights to manipulate diffusion process.
        :param weights_y: Weights to manipulate class embeddings.
        :param return_manipulation_history: Whether to return a candidate representing the manipulation history.
        :returns: The resulting diffusion result.
        """
        logging.info(f"Manipulating {len(candidates)} candidates.")
        n_steps = len(candidates[0].xt) - 1
        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)  # Define step range.

        """The manipulation process."""
        manip_cand = self._manip_function(
            t_steps=t_steps,
            candidates=candidates,
            xt_weights=weights_x,
            y_weights=weights_y,
        )

        """Return a candidate with the manipulation history if wanted."""
        return (
            (manip_cand.xts[:, -1, ...], manip_cand)
            if return_manipulation_history
            else manip_cand.xts[:, -1, ...]
        )

    def _manip_aggregate_new(
        self,
        *,
        t_steps: Tensor,
        candidates: DiffusionCandidateList,
        xt_weights: Tensor,
        y_weights: Tensor,
    ) -> DiffusionCandidateList:
        """
        Manipulate latent vector of diffusion processes and aggregate them on the new candidate.

        :param t_steps: Time steps to manipulate.
        :param candidates: Candidates to manipulate with.
        :param xt_weights: Weights to manipulate diffusion process.
        :param y_weights: Weights to manipulate class embeddings with.
        :returns: The resulting diffusion candidate.
        """
        assert (
            xt_weights.ndim == y_weights.ndim == 2
        ), "ERROR: For now this strategy does not support batch wise manipulation"
        # TODO: make this batch compatible

        xt_template, y_template = candidates[0].xt, candidates[0].class_embedding
        weighted_class_embeddings = candidates.class_embeddings * y_weights[..., None]  # N x D x Y
        y_manip = torch.sum(
            weighted_class_embeddings, dim=0, keepdim=True
        ).float()  # N x D x Y -> 1 x D x Y
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Here we manipulate the diffusion steps.
            weighted_candidates = (
                candidates.xts[:, i, ...] * xt_weights[:, i][:, None, None, None]
            )  # N x X
            x_manip = torch.sum(weighted_candidates, dim=0, keepdim=True).float()  # N x X -> 1 x X
            x_manip = (xt_template[i] + x_manip) / 2

            x_cur = self._sample(t=t_cur, x=x_manip, y=y_manip[:, i, ...], step=t_next - t_cur)

            # Log progress.
            y_template = y_template.clone()
            y_template[i] = y_manip[:, i, ...]
            xt_template = xt_template.clone()
            xt_template[i + 1] = x_cur
        return DiffusionCandidateList.from_diffusion_output(xs=xt_template, emb=y_template)

    def _manip_aggregate_base(
        self,
        *,
        t_steps: Tensor,
        candidates: DiffusionCandidateList,
        xt_weights: Tensor,
        y_weights: Tensor,
    ) -> DiffusionCandidateList:
        """
        Manipulate latent vector of diffusion processes and aggregate them on the base candidate.

        :param t_steps: Time steps to manipulate.
        :param candidates: Candidates to manipulate with.
        :param xt_weights: Weights to manipulate diffusion process.
        :param y_weights: Weights to manipulate class embeddings with.
        :returns: The resulting diffusion candidate.
        """
        if xt_weights.ndim > 1:  # Check if we have a batch dimension in the weights.
            batch_size = xt_weights.shape[0]
        else:  # If not make a singleton batch dimension for compatability.
            batch_size = 1
            xt_weights = xt_weights.unsqueeze(0)
            y_weights = y_weights.unsqueeze(0)

        """Convert into batches."""
        xt_target = self._batch_expand(candidates.target[0].xt, batch_size)
        xt_origin = self._batch_expand(candidates.origin[0].xt, batch_size)

        y_target = self._batch_expand(candidates.target[0].class_embedding, batch_size)
        y_origin = self._batch_expand(candidates.origin[0].class_embedding, batch_size)

        """Do the manipulations on batches."""
        y_manip = self.slerp(y_origin, y_target, y_weights, dim=(1,)).float()
        del y_origin, y_target
        torch.cuda.empty_cache()
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_manip = self.slerp(
                xt_origin[:, i, ...], xt_target[:, i, ...], xt_weights[:, i], dim=(1,)
            ).float()
            x_cur = self._sample(t=t_cur, x=x_manip, y=y_manip[:, i, ...], step=t_next - t_cur)
            xt_target[:, i + 1, ...] = x_cur

        # Transposing from Batch x DiffSteps x Z -> DiffSteps x Batch x Z
        diff_candidates = DiffusionCandidateList.from_diffusion_output(
            xs=xt_target.transpose(0, 1).detach(), emb=y_manip.detach(), separate_candidates=False
        )
        del xt_target, xt_origin, x_cur, x_manip
        torch.cuda.empty_cache()
        return diff_candidates

    @staticmethod
    def _batch_expand(tensor: Tensor, batch_size: int) -> Tensor:
        """
        Batch expand a tensor.

        :param tensor: Tensor to expand.
        :param batch_size: Batch size.
        :returns: The expanded tensor.
        """
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape).clone()

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
        cond = cfg_scale > 1.0 and guidance_bounds[1] >= t >= guidance_bounds[0]

        with torch.no_grad():
            t_curr = torch.full(size=(y.size(0),), fill_value=t, device=self._device)  # noqa
            if cond:
                model_input = x.repeat(2, *([1] * (x.ndim - 1)))
                y_null = self._embed_y([1000] * y.shape[0])  # Get class unconditional embeddings.
                y_curr = torch.cat((y, y_null), dim=0)
                t_curr = t_curr.repeat(2, *([1] * (t_curr.ndim - 1)))
            else:
                model_input, y_curr = x, y
            d_cur = self._model.partial_inference(x=model_input, t=t_curr, y=y_curr)

        if cond:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

        return (x + step * d_cur).detach()

    def get_diff_steps(self, class_labels: list[int], n_steps: int = 50) -> tuple[Tensor, Tensor]:
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
        )  # TODO: is not saved, check if required
        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)
        y_cur = self._embed_y(class_labels)

        xs = []
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x_cur = self._sample(t=t_cur, x=x_cur, y=y_cur, step=t_next - t_cur)
            xs.append(x_cur)
        return torch.stack(xs), y_cur

    def get_image(self, z: Tensor) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :return: The decoded image.
        """
        logging.info("Sampling Images from denoised Latents.")
        decoded = []
        batch_size = max(self._batch_size or z.size(0), 1)
        chunks = (z.size(0) + batch_size - 1) // batch_size

        for z_chunk in torch.chunk(z, chunks, dim=0):
            with torch.no_grad():
                element = self._vae.decode(
                    (z_chunk / self._latents_scale) + self._latents_bias
                ).sample
            element = (element.detach() + 1.0) / 2.0
            element = torch.clamp(element, 0.0, 1.0)
            decoded.append(element)

        element = torch.cat(decoded, dim=0)
        return element

    def _prepare_cuda(self, device: Union[torch.device, None]) -> None:
        """Prepare cuda environment, as done in the REPA-E repository."""
        torch.backends.cuda.matmul.allow_tf32 = True
        assert (
            torch.cuda.is_available()
        ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
        torch.set_grad_enabled(False)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def slerp(
        p: Tensor, q: Tensor, weight: Tensor, epsilon=1e-6, dim: Union[tuple[int, ...], None] = None
    ) -> Tensor:
        """
        Implement spherical linear interpolation for Tensors.

        :param p: The first tensor.
        :param q: The second tensor.
        :param weight: The weight for the interpolation.
        :param epsilon: Cutoff value for precision.
        :param dim: The dimensions to do operations across.
        :return: The interpolated tensor.
        """
        p_norm = p / torch.linalg.norm(p, dim=dim, keepdim=True).clamp_min(epsilon)
        q_norm = q / torch.linalg.norm(q, dim=dim, keepdim=True).clamp_min(epsilon)
        dot = (p_norm * q_norm).sum(dim=dim, keepdim=True).clamp(-1.0 + epsilon, 1.0 - epsilon)

        if weight.ndim == 0:
            weight = weight.view(1)
        for _ in range(p.ndim - weight.ndim):
            weight = weight.unsqueeze(-1)

        if torch.all(torch.abs(dot) > 1 - (5 * epsilon)):
            wp, wq = 1 - weight, weight
        else:
            omega = torch.arccos(dot)
            sin_omega = torch.sin(omega).clamp_min(epsilon)
            omega_w = omega * weight
            wp = torch.sin(omega - omega_w) / sin_omega
            wq = torch.sin(omega_w) / sin_omega
        return wp * p + wq * q
