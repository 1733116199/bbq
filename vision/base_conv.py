import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform

class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)
        self.div_step = torch.tensor(0.0)
        self.mul_step = torch.tensor(0.0)
        self.ent = torch.tensor(0.0)

    def forward(self, x):
        return x
    
class HalfHadamardNoQuantizer(NoQuantizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aux_matrix = torch.nn.Parameter(
            hadamard_transform(
                torch.eye(64, dtype=torch.float32, device="cuda"), scale=2 ** (-6 / 2)
            ), 
            requires_grad=False
        )
    
    def forward(self, x: torch.Tensor):
        x = x.moveaxis(1, -1)
        xshape = x.shape
        x_had = torch.reshape(x.reshape((*xshape[:-1], xshape[-1] // 64, 64)) @ self.aux_matrix, xshape)
        x_had = x_had.moveaxis(-1, 1)
        return x_had

class UniformQuantizer(BaseQuantizer):
    def forward(self, x):
        if not self.training:
            return x
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True) + 1e-8
        step = scale * 2 / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        return x + (xq - x).detach()


OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}


class STEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True):
        super().__init__(bits)
        self.centered = centered

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, -scale * (self.n_levels - 2) / self.n_levels, scale)
            xq = torch.round(x_clip / step) * step

        return x + (xq - x).detach()


class ClipQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.clip_scale = clip_scale

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale * self.clip_scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = (
                (neg_scale * self.clip_scale <= x) & (x <= scale * self.clip_scale)
            ).float()
        return x * mask + (xq - x * mask).detach()

class HalfHadamardTrustQuantizer(STEQuantizer):

    def __init__(self, bits=4, trust=None, lhd=6, **kwargs):
        super().__init__(bits, True)
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust
        self.lhd = lhd
        self.hd = 1 << lhd
        self.aux_matrix = torch.nn.Parameter(
            hadamard_transform(
                torch.eye(self.hd, dtype=torch.float32, device="cuda"), scale=2 ** (-self.lhd / 2)
            ), 
            requires_grad=False
        )
        self.div_step = None
        self.mul_step = None
        self.ent = torch.tensor(0.0)

    def log_ent(self, w: torch.Tensor):
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                prob: torch.Tensor = torch.unique(w.int(), return_counts=True)[1]
                prob = prob / prob.sum()
                assert prob.numel() <= self.n_levels
                self.ent = torch.sum(-prob * torch.log2(prob))
    
    def forward(self, x: torch.Tensor):
        x = x.moveaxis(1, -1)
        xshape = x.shape
        x_had = torch.reshape(x.reshape((*xshape[:-1], xshape[-1] // self.hd, self.hd)) @ self.aux_matrix, xshape)
        x_had = x_had.moveaxis(-1, 1)
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=(-1, -2, -3), keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            self.div_step = step.detach()
            self.mul_step = step.detach()
            x_clip = torch.clamp(x_had, -scale, scale)
            temp = torch.round(x_clip / step + 1 / 2)
            self.log_ent(temp)
            xq = temp * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()
    
    def extra_repr(self):
        return f"bits={self.bits}, centered={self.centered}, trust={self.trust}, hd={self.hd}, lhd={self.lhd}"

class LSQQuantizer(nn.Module):
    """
    Implementation of LSQ quantizer from https://arxiv.org/abs/1902.08153
    LSQ uses a learnable step size for quantization. This learnable step size(alpha) is initialized using the optimal gaussian scale
    ans must be normalized with a weight decay.
    """

    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__()
        # NOTE: raise_zero should never be used with FP quantization

        self.bits = bits
        self.n_levels = 2**bits
        self.all_positive = all_positive
        self.raise_zero = raise_zero

        self.q_min, self.q_max = self.get_dtype_bounds()

        self.is_alpha_init = False
        self.alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.div_step = None
        self.mul_step = None
        self.ent = torch.tensor(0.0)

    def log_ent(self, w: torch.Tensor):
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                prob: torch.Tensor = torch.unique(w.int(), return_counts=True)[1]
                prob = prob / prob.sum()
                assert prob.numel() <= self.n_levels
                self.ent = torch.sum(-prob * torch.log2(prob))
    
    def get_dtype_bounds(self):
        if not self.all_positive:
            q_min = -self.n_levels / 2
            q_max = self.n_levels / 2 - 1
        else:
            q_min = 0
            q_max = self.n_levels - 1
        return q_min, q_max

    def cast(self, x):
        # This method can be inherited to use any casting, e.g. int, fp(e2m1, e1m2,...), optimal gaussian, etc.
        # NOTE: raise_zero should never be used with FP quantization
        return x.round()

    def ste_cast(self, x):
        return (self.cast(x) - x).detach() + x

    def grad_scale(self, x, scale):
        return (x - x * scale).detach() + x * scale

    @torch.no_grad()
    def get_initial_step_value(self, x):
        return (
            torch.mean(torch.abs(x.detach())) * 2 / (np.sqrt(self.q_max))
        )  # LSQ initialization

    def get_learnable_step(self, x):
        if (not torch.compiler.is_compiling()) and (not self.is_alpha_init):
            with torch.no_grad():
                step = self.get_initial_step_value(x)
                self.alpha_weight.data.multiply_(
                    torch.tensor(
                        step,
                        dtype=self.alpha_weight.dtype,
                        device=self.alpha_weight.device,
                    )
                )
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(self.alpha_weight, src=0)
            self.is_alpha_init = True
        return self.alpha_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        with torch.no_grad():
            self.div_step = step.detach()
            self.mul_step = step.detach()
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
            self.log_ent(xscr)
            xscr = xscr + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
            self.log_ent(xscr)
        xq = xscr * step

        return xq + step * 1e-9  # extra term to ensure gradient flow
    
    def extra_repr(self):
        return f"bits={self.bits}, raise_zero={self.raise_zero}, all_positive={self.all_positive}, is_alpha_init={self.is_alpha_init}"

class BBQVision(torch.nn.Module):
    def __init__(
        self,
        precision: int, 
        zero_point: float,
        learn_sx: bool=False,
        detach_rrms: bool=False,
        per_channel_rms: bool=False,
        ema_rrms: bool = False,
        factor: float = 1.0,
        init: float=2.45
    ) -> None:
        super().__init__()
        # enforce integer precision
        assert precision > 0
        assert round(precision) == precision
        precision = round(precision)
        self.precision = precision
        self.ema_rrms = ema_rrms
        self.factor = factor

        self.per_channel_rms = per_channel_rms
        self.n_levels = 1 << precision
        self.half_n_levels = 1 << (precision - 1)
        self.zero_point = zero_point
        self.learn_sx = learn_sx
        self.detach_rrms = detach_rrms

        self.sx = torch.nn.Parameter(torch.tensor(init), requires_grad=self.learn_sx)
        self.init_sx = False
        self.mul_step = None
        self.div_step = None
        self.ent = torch.tensor(0.0)
        if self.ema_rrms:
            self.running_rrms: torch.Tensor
            self.register_buffer("running_rrms", torch.tensor(1.0))

    def get_sx(self, x: torch.Tensor):
        sx = self.sx
        sx_backward = sx * (1 / (x.numel() ** 0.5))
        return sx.detach() + sx_backward - sx_backward.detach()

    def transform(self, x: torch.Tensor):
        raise NotImplementedError()

    def log_ent(self, w: torch.Tensor):
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                prob: torch.Tensor = torch.unique(w.int(), return_counts=True)[1]
                prob = prob / prob.sum()
                assert prob.numel() <= self.n_levels
                self.ent = torch.sum(-prob * torch.log2(prob))

    def forward(self, x: torch.Tensor):
        x = self.transform(x)
        sx = self.get_sx(x) # must be here to initialize sx with information from x
        
        rrms = None
        if self.per_channel_rms:
            rrms = 1 / x.square().mean(dim=(-1, -2, -3), keepdim=True).sqrt().clip(min=1e-8)
        else:
            rrms = x.square().mean().rsqrt()
        
        with torch.no_grad():
            self.div_step = (1 / rrms.detach())
            self.mul_step = (sx.detach() / rrms.detach() / self.half_n_levels)

        if self.ema_rrms:
            if not self.training:
                rrms = self.running_rrms
            else:
                with torch.no_grad():
                    self.running_rrms.data = 0.99 * self.running_rrms.data + 0.01 * rrms.mean()

        if self.detach_rrms:
            rrms = rrms.detach()
        
        x = 0.5 * (1 + torch.erf((2 ** (-0.5) * self.factor * rrms) * x)) # within range (0, 1)
        x = x * self.n_levels - 0.5 # within range (-0.5, 2^p-0.5)

        # the round is required, the clip should be noop if torch.round is 
        # the real round (break ties with the ceiling integer)
        # however, torch.round break ties with the even integer, 
        # hence the need to do an extra clip
        x_forward = torch.round(x).clip(0, self.n_levels - 1) # within range [0, 2^p-1]
        self.log_ent(x_forward)
        x = x_forward.detach() + x - x.detach()

        x = x - (self.half_n_levels + self.zero_point) # within range [-2^(p-1)-z, 2^(p-1)-1-z]
        
        return (sx / rrms / self.half_n_levels) * x

class BBQVisionHD(BBQVision):
    def __init__(
        self,
        precision: int, 
        zero_point: float,
        learn_sx: bool=False,
        detach_rrms: bool=False,
        per_channel_rms: bool=True,
        lhd=6, 
        init=2.45,
        factor=1.0,
        ema_rrms=False,
        **kwargs
    ) -> None:
        super().__init__(precision=precision, zero_point=zero_point, learn_sx=learn_sx, detach_rrms=detach_rrms, per_channel_rms=per_channel_rms, init=init, ema_rrms=ema_rrms, factor=factor)
        self.lhd = lhd
        self.hd = 1 << lhd
        self.aux_matrix = torch.nn.Parameter(
            hadamard_transform(
                torch.eye(self.hd, dtype=torch.float32, device="cuda"), scale=2 ** (-self.lhd / 2)
            ), 
            requires_grad=False
        )

    def transform(self, x: torch.Tensor):
        x = x.moveaxis(1, -1)
        xshape = x.shape
        x = torch.reshape(x.reshape((*xshape[:-1], xshape[-1] // self.hd, self.hd)) @ self.aux_matrix, xshape)
        x = x.moveaxis(-1, 1)
        return x
    
    def extra_repr(self):
        return f"precision={self.precision}, n={self.n_levels}, hn={self.half_n_levels}, "\
        f"per_channel_rms={self.per_channel_rms}, init_sx={self.init_sx}, zero_point={self.zero_point}, "\
        f"learn_sx={self.learn_sx}, detach_rrms={self.detach_rrms}, hd={self.hd}, lhd={self.lhd}, "\
        f"asx={self.sx.detach().mean()}, ema_rrms={self.ema_rrms}, factor={self.factor}"

QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "UniformQuantizer": UniformQuantizer,
    "STEQuantizer": STEQuantizer,
    "ClipQuantizer": ClipQuantizer,
    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    "LSQQuantizer": LSQQuantizer,
    "HalfHadamardNoQuantizer": HalfHadamardNoQuantizer,
    "BBQVisionHD": BBQVisionHD,
}


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_quantizer=None,
        activation_quantizer=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, x):
        x = self.activation_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)
    
class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        weight_quantizer=None,
        activation_quantizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        self.histx = None

    def forward(self, x):
        if self.training:
            self.histx = x.detach()
        x = self.activation_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return self._conv_forward(x, w, self.bias)
