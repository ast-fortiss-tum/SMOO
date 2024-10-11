import torch
import numpy as np
from torch import Tensor
from torch_utils.ops import upfirdn2d
import dnnlib

class StyleMixer:
    """A class geared towards style-mixing."""
    _generator: torch.nn.Module
    _device: torch.device
    _has_input_transform: bool

    _pinned_bufs: dict

    def __init__(
            self,
            generator: torch.nn.Module,
            device: torch.device,
    ):
        self._generator=generator
        self._device=device
        self._has_input_transform = (hasattr(generator.synthesis, 'input') and hasattr(generator.synthesis.input, 'transform'))

    def mix(
            self,
            seeds: list[int],
            weights: list[float],
            classes: list[int],
            stylemix_indices: list[int],
            stylemix_weights: list[float],
            rseed: int = 0,
            sel_channels: int = 3,
    ) -> Tensor:
        """
        Generate data using style mixing.

        This function is heavily inspired by the Renderer class of the original StyleGANv3 codebase.

        :param seeds: The seeds to use for generation.
        :param weights: The weights of seeds.
        :param classes: The classes of seeds.
        :param stylemix_indices: The style mix strategy (For styleGAN3 L0-L13).
        :param stylemix_weights: The weights for mixing.
        :param rseed: The seed for randomization.
        :param sel_channels: Channels to be selected for output.
        :returns: The generated data.
        """
        assert len(seeds) == len(weights) == len(classes) == len(stylemix_indices) == len(stylemix_weights), "Error: The parameters have to be of same length."

        if self._has_input_transform:
            m = np.eye(3)
            # TODO: maybe add custom transformations
            self._generator.synthesis.input.transform.copy_(torch.from_numpy(m))

        """Generate latents."""
        all_zs = np.zeros([len(seeds), self._generator.z_dim], dtype=np.float32)  # Latent inputs
        all_cs = np.zeros([len(seeds), self._generator.c_dim], dtype=np.float32)  # Input classes
        all_cs[:, classes] = 1  # Set classes in class vector

        all_zs = self._to_device(torch.from_numpy(all_zs))
        all_cs = self._to_device(torch.from_numpy(all_cs))

        ws_average = self._generator.mapping.w_avg
        all_ws = self._generator.mapping(z=all_zs, c=all_cs, truncation_psi=1, truncation_cutoff=0) - ws_average

        weight_tensor = torch.Tensor(weights)[:, None, None]
        w = all_ws[0]*weight_tensor[0]  # 16 x m; Only 1 w0 seed for now

        """
        Here we do style mixing.
        
        Since we want to mix our baseclass with a second class we take the layers to mix, and apply the second class with its weights.
        Note if the index to mix is baseclass, this has no effect.
        """
        w_dim = torch.arange(1,15, device=self._device)  # Dimensions of w -> we only want to mix style layers.
        smw_tensor = torch.Tensor(stylemix_weights)[:, None]  # 14 x 1
        w[w_dim] += all_ws[stylemix_indices, w_dim] * smw_tensor + all_ws[0, w_dim] * ((smw_tensor-1)*-1)
        w = w / 2 + ws_average

        torch.manual_seed(rseed)
        out, _ = self._run_synthesis_net(self._generator.synthesis, w[None, :, :], noise_mode="const", force_fp32=False)

        out = out[0].to(torch.float32)
        if sel_channels > out.shape[0]:
            sel_channels = 1
        base_channel = max(out.shape[0] - sel_channels, 0)
        sel = out[base_channel: base_channel + sel_channels]

        img = sel / sel.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)

        return img

    # ------------------ Borrowed from https://github.com/NVlabs/stylegan3/blob/main/viz/renderer.py -----------------------
    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def _to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    @staticmethod
    def _run_synthesis_net(net, *args, capture_layer=None, **kwargs): # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]]
            for idx, out in enumerate(outputs):
                if out.ndim == 5: # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == '':
                    name = 'output'
                if len(outputs) > 1:
                    name += f':{idx}'
                if name in unique_names:
                    suffix = 2
                    while f'{name}_{suffix}' in unique_names:
                        suffix += 1
                    name += f'_{suffix}'
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split('.')[-1]
                layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
        try:
            out = net(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for hook in hooks:
            hook.remove()
        return out, layers

    @staticmethod
    def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
        _N, _C, H, W = x.shape
        mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

        # Construct filter.
        f = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
        assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
        p = f.shape[0] // 2

        # Construct sampling grid.
        theta = mat.inverse()
        theta[:2, 2] *= 2
        theta[0, 2] += 1 / up / W
        theta[1, 2] += 1 / up / H
        theta[0, :] *= W / (W + p / up * 2)
        theta[1, :] *= H / (H + p / up * 2)
        theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
        g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

        # Resample image.
        y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
        z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Form mask.
        m = torch.zeros_like(y)
        c = p * 2 + 1
        m[:, :, c:-c, c:-c] = 1
        m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
        return z, m


def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f


def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)


def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out
