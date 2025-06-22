import gc
import logging
from typing import Callable, Union, Optional

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
    _interpolation_strategy: str
    _cfg: float
    _epsilon: float = 1e-6  # Precision constant

    """Auxiliary lambdas for easy callings."""
    _embed_y: Callable[[list[Tensor]], Tensor]
    _embed_t: Callable[[list[Tensor]], Tensor]

    _manip_function: Callable[..., DiffusionCandidateList]

    """Optimization caches."""
    _null_embedding_cache: Optional[Tensor] = None
    _pre_allocated_buffers: dict[str, Tensor]

    def __init__(
        self,
        model_file: str,
        cfg_scale: float = 1.5,
        vae: str = "f16d32",
        model: str = "SiT-XL/1",
        encoder: str = "dinov2-vit-b",
        image_resolution: int = 256,
        num_classes: int = 1000,
        batch_size: int = 0,
        manipulation_strategy: str = "new",
        interpolation_strategy: str = "linear",
        device: Union[torch.device, None] = None,
    ) -> None:
        """
        Initialize the manipulator based on REPA-E diffusion models.

        :param model_file: Model file to load weights from.
        :param cfg_scale: Classifier free guidance scale for conditions in the sampling.
        :param vae: The type of VAE model to use.
        :param model: The type of model to use.
        :param encoder: The type of encoder to use.
        :param image_resolution: Image resolution for generation.
        :param num_classes: Number of classes in the dataset.
        :param batch_size: Batch size to use for generation of samples (0 means all elements get taken).
        :param manipulation_strategy: Manipulation strategy to use.
        :param interpolation_strategy: Interpolation strategy to use.
        :param device: CUDA device to use if available.
        :raises ValueError: If manipulation_strategy is not supported.
        """
        self._prepare_cuda(device)
        self._batch_size = batch_size
        state_dict = torch.load(model_file, weights_only=False)
        self._cfg = cfg_scale

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

        """Prepare SiT model with optimized loading."""
        encoders, _, _ = load_encoders(encoder, "cpu", image_resolution)
        z_dims = [encoder.embed_dim for encoder in encoders] if encoder != "None" else [0]
        # Immediate cleanup
        del encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        """Clean up with optimized memory management."""
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        """Define Embedding lambdas"""
        self._embed_y = lambda y: self._model.y_embedder(
            torch.tensor(y, device=self._device), self._model.training
        ).detach()

        """Pre-cache null embedding for CFG."""
        self._null_embedding_cache = None

        self._interpolation_strategy = interpolation_strategy
        self._pre_allocated_buffers = {}
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
        weights_x = weights_x.to(self._device)
        weights_y = weights_y.to(self._device)

        logging.info(f"Manipulating {len(candidates)} candidates.")
        t_steps = torch.linspace(
            1, 0, len(candidates[0].xt), device=self._device
        )  # Define step range.

        if weights_x.ndim > 1:  # Check if we have a batch dimension in the weights.
            batch_size = weights_x.shape[0]
        else:  # If not make a singleton batch dimension for compatability.
            batch_size = 1
            weights_x = weights_x.unsqueeze(0)
            weights_y = weights_y.unsqueeze(0)

        """Convert into batches with optimized expansion."""
        target_xt = candidates.target[0].xt
        origin_xt = candidates.origin[0].xt
        target_emb = candidates.target[0].class_embedding
        origin_emb = candidates.origin[0].class_embedding

        xt_target = target_xt.unsqueeze(0).expand(batch_size, *target_xt.shape).contiguous()
        xt_origin = origin_xt.unsqueeze(0).expand(batch_size, *origin_xt.shape).contiguous()
        y_target = target_emb.unsqueeze(0).expand(batch_size, *target_emb.shape).contiguous()
        y_origin = origin_emb.unsqueeze(0).expand(batch_size, *origin_emb.shape).contiguous()

        """The manipulation process."""
        manip_cand = self._manip_function(
            t_steps=t_steps,
            origin=(xt_origin, y_origin),
            target=(xt_target, y_target),
            weights_x=weights_x,
            weights_y=weights_y,
        )

        # Explicit cleanup with immediate cache clearing
        del xt_target, xt_origin, y_origin, y_target, target_xt, origin_xt, target_emb, origin_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        origin: tuple[Tensor, Tensor],
        target: tuple[Tensor, Tensor],
        weights_x: Tensor,
        weights_y: Tensor,
    ) -> DiffusionCandidateList:
        """
        Manipulate latent vector of diffusion processes and aggregate them on the new candidate.

        :param t_steps: Time steps to manipulate.
        :param origin: Tuple of origin diffusion process and embedding.
        :param target: Tuple of target diffusion process and embedding.
        :param weights_x: Weights to manipulate diffusion process.
        :param weights_y: Weights to manipulate class embeddings.
        :returns: The resulting diffusion candidate.
        """
        xt_origin, y_origin = origin
        xt_target, y_target = target

        # Pre-compute interpolated embeddings
        y_manip = self.interpolate(y_origin, y_target, weights_y, dim=(1,)).float()

        # Pre-allocate template with better memory layout
        buffer_key = xt_origin.shape
        if buffer_key not in self._pre_allocated_buffers:
            self._pre_allocated_buffers[buffer_key] = torch.empty_like(xt_origin)
        xt_template = self._pre_allocated_buffers[buffer_key]
        xt_template.zero_()

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_manip = self.interpolate(
                xt_origin[:, i, ...], xt_target[:, i, ...], weights_x[:, i], dim=(1,)
            ).float()
            x_manip.add_(xt_template[:, i, ...]).mul_(0.5)
            x_cur = self._sample(t=t_cur, x=x_manip, y=y_manip[:, i, ...], step=t_next - t_cur)
            xt_template[:, i + 1, ...] = x_cur
        # Transposing from Batch x DiffSteps x Z -> DiffSteps x Batch x Z
        diff_candidates = DiffusionCandidateList.from_diffusion_output(
            xs=xt_template.transpose(0, 1).detach(), emb=y_manip.detach(), separate_candidates=False
        )
        return diff_candidates

    def _manip_aggregate_base(
        self,
        *,
        t_steps: Tensor,
        origin: tuple[Tensor, Tensor],
        target: tuple[Tensor, Tensor],
        weights_x: Tensor,
        weights_y: Tensor,
    ) -> DiffusionCandidateList:
        """
        Manipulate latent vector of diffusion processes and aggregate them on the base candidate.

        :param t_steps: Time steps to manipulate.
        :param origin: Tuple of origin diffusion process and embedding.
        :param target: Tuple of target diffusion process and embedding.
        :param weights_x: Weights to manipulate diffusion process.
        :param weights_y: Weights to manipulate class embeddings.
        :returns: The resulting diffusion candidate.
        """
        logging.warning(
            "Attention: This manipulation is very minimal as it depends on the origin image!"
        )
        xt_origin, y_origin = origin
        xt_target, y_target = target

        """Do the manipulations on batches with optimized memory usage."""
        y_manip = self.interpolate(y_origin, y_target, weights_y, dim=(1,)).float()

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_manip = self.interpolate(
                xt_origin[:, i, ...], xt_target[:, i, ...], weights_x[:, i], dim=(1,)
            ).float()
            x_cur = self._sample(t=t_cur, x=x_manip, y=y_manip[:, i, ...], step=t_next - t_cur)
            xt_target[:, i + 1, ...] = x_cur
            del x_manip, x_cur

        # Transposing from Batch x DiffSteps x Z -> DiffSteps x Batch x Z
        diff_candidates = DiffusionCandidateList.from_diffusion_output(
            xs=xt_target.transpose(0, 1).detach(), emb=y_manip.detach(), separate_candidates=False
        )
        return diff_candidates

    def _sample(
        self,
        t: Tensor,
        x: Tensor,
        y: Tensor,
        step: float,
        guidance_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> Tensor:
        """
        Sampling new outputs based on euler_sampler in REPA-E repo.

        :param t: The current time step.
        :param x: The current state of the diffusion process.
        :param y: The current class embedding of the diffusion process.
        :param step: The step size.
        :param guidance_bounds: Guidance bounds for conditions in the sampling.
        :returns: The sampled outputs for the current timestep.
        """
        cond = self._cfg > 1.0 and guidance_bounds[1] >= t >= guidance_bounds[0]

        with torch.no_grad():
            t_curr = torch.full(size=(y.size(0),), fill_value=t, device=self._device)  # noqa
            if cond:
                model_input = x.repeat(2, *([1] * (x.ndim - 1)))
                # Use cached null embedding if available
                if (
                    self._null_embedding_cache is None
                    or self._null_embedding_cache.shape[0] != y.shape[0]
                ):
                    self._null_embedding_cache = self._embed_y([1000] * y.shape[0])
                y_curr = torch.cat((y, self._null_embedding_cache), dim=0)
                t_curr = t_curr.repeat(2, *([1] * (t_curr.ndim - 1)))
            else:
                model_input, y_curr = x, y
            d_cur = self._model.partial_inference(x=model_input, t=t_curr, y=y_curr)

        if cond:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + self._cfg * (d_cur_cond - d_cur_uncond)

        return x + step * d_cur

    def get_diff_steps(self, class_labels: list[int], n_steps: int = 50) -> tuple[Tensor, Tensor]:
        """
        Get latent information for all diffusion steps with optimized memory usage.

        :param class_labels: Class label to generate diffusion steps for.
        :param n_steps: Number of steps in the denoising.
        :returns: A list of latent vectors through denoising and the class embedding.
        """
        batch_size = len(class_labels)

        x_cur = torch.randn(
            batch_size,
            self._in_channels,
            self._latent_size,
            self._latent_size,
            device=self._device,
        )

        t_steps = torch.linspace(1, 0, n_steps + 1, device=self._device)
        y_cur = self._embed_y(class_labels)

        xs = torch.empty(
            n_steps,
            batch_size,
            self._in_channels,
            self._latent_size,
            self._latent_size,
            device=self._device,
        )

        # Optimized diffusion loop with in-place updates
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = self._sample(t=t_cur, x=x_cur, y=y_cur, step=t_next - t_cur)
            xs[i] = x_cur

        return xs.detach(), y_cur

    def get_image(self, z: Tensor) -> Tensor:
        """
        Decode image from latent vector.

        :param z: The latent vector.
        :return: The decoded image.
        """
        logging.info("Sampling Images from denoised Latents.")
        if z.ndim == 3:  # Ensure batch dimension is present.
            z = z.unsqueeze(0)

        chunks = (
            (z.size(0) + self._batch_size - 1) // self._batch_size
            if self._batch_size > 0
            else z.size(0)
        )
        decoded = []
        for z_chunk in torch.chunk(z, chunks, dim=0):
            with torch.no_grad():
                decoded_latents = (z_chunk / self._latents_scale) + self._latents_bias
                element = self._vae.decode(decoded_latents).sample
                element = torch.clamp(element.mul_(0.5).add_(0.5), 0.0, 1.0)
                decoded.append(element)
            del decoded_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(decoded, dim=0)

    def _prepare_cuda(self, device: Union[torch.device, None]) -> None:
        """Prepare optimized CUDA environment."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        assert (
            torch.cuda.is_available()
        ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"

        torch.set_grad_enabled(False)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Additional CUDA optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)

    @staticmethod
    def linear(p: Tensor, q: Tensor, weight: Tensor) -> Tensor:
        """
        Weights for linear interpolation.

        :param p: The first tensor.
        :param q: The second tensor.
        :param weight: The weight for the interpolation.
        :return: The interpolated tensor.
        """
        wp, wq = 1 - weight, weight
        return wp * p + wq * q

    def interpolate(
        self, p: Tensor, q: Tensor, weight: Tensor, dim: Optional[tuple[int, ...]] = None
    ) -> Tensor:
        """
        Implement optimized interpolation for Tensors.

        :param p: The first tensor.
        :param q: The second tensor.
        :param weight: The weight for the interpolation.
        :param dim: The dimensions to do operations across.
        :return: The interpolated tensor.
        :raises: ValueError if interpolation strategy is not known.
        """
        if weight.ndim == 0:
            weight = weight.view(1)

        target_ndim = p.ndim
        current_ndim = weight.ndim
        if current_ndim < target_ndim:
            shape_ext = [1] * (target_ndim - current_ndim)
            weight = weight.view(*weight.shape, *shape_ext)

        # Branch prediction optimization - most common case first
        if self._interpolation_strategy == "linear":
            return self.linear(p, q, weight)
        elif self._interpolation_strategy == "slerp":
            return self.slerp(p, q, weight, dim)
        else:
            raise ValueError(f"Unknown interpolation strategy {self._interpolation_strategy}")

    def slerp(self, p: Tensor, q: Tensor, weight: Tensor, dim: Optional[tuple[int, ...]]) -> Tensor:
        """
        Optimized spherical linear interpolation.

        :param p: The first tensor.
        :param q: The second tensor.
        :param weight: The weight for the interpolation.
        :param dim: The dimensions to do operations across.
        :return: The interpolated tensor.
        """
        p_norm_val = torch.linalg.norm(p, dim=dim, keepdim=True).clamp_min(self._epsilon)
        q_norm_val = torch.linalg.norm(q, dim=dim, keepdim=True).clamp_min(self._epsilon)

        p_norm = p / p_norm_val
        q_norm = q / q_norm_val

        dot = torch.sum(p_norm * q_norm, dim=dim, keepdim=True).clamp(
            -1.0 + self._epsilon, 1.0 - self._epsilon
        )

        abs_dot = torch.abs(dot)
        fallback_threshold = 1.0 - (5 * self._epsilon)
        if torch.all(abs_dot > fallback_threshold):
            return self.linear(p, q, weight)

        omega = torch.arccos(dot)
        sin_omega = torch.sin(omega).clamp_min(self._epsilon)
        omega_w = omega * weight

        wp = torch.sin(omega - omega_w) / sin_omega
        wq = torch.sin(omega_w) / sin_omega
        return wp * p + wq * q
