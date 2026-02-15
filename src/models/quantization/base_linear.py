import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform
import math
import bitsandbytes.functional
from .nf import NFQuantizer
from .benchmark.quant import bbq

try:
    import i4mm
except:
    pass

class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x
    
    def entropy(self, x):
        return torch.tensor(0.0, device=x.device)

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


class HalfHadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardTrustQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True)
        self.matrix = None
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust
        self.ent = torch.tensor(0.0)
        self.sx = torch.tensor(0.0)

    @torch.no_grad()
    def log_ent(self, x: torch.Tensor):
        prob: torch.Tensor = torch.unique(x.int(), return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        self.ent = torch.sum(-prob * torch.log2(prob))
    
    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            temp = torch.round(x_clip / step + 1 / 2)
            # self.log_ent(temp)
            xq = temp * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


    def entropy(self, x: torch.Tensor):
        with torch.no_grad():
            if self.matrix is None:
                self.matrix = torch.block_diag(
                    *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
                )

            x_had = x @ self.matrix
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            x_int = torch.round(x_clip / step + 1 / 2).int()
            prob: torch.Tensor = torch.unique(x_int, return_counts=True)[1]
            prob = prob / prob.sum()
            assert prob.numel() <= self.n_levels
            return torch.sum(-prob * torch.log2(prob))

class TrustQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, trust=None):
        super().__init__(bits, centered)

        # in terms of std
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HadamardTrustQuantizer(TrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class GaussianSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4):
        super().__init__(bits)
        self.register_buffer("levels", self._compute_gaussian_levels())

    def _compute_gaussian_levels(self):
        levels = np.linspace(-3, 3, self.n_levels)
        boundaries = np.zeros(self.n_levels + 1)

        for _ in range(20):
            boundaries[1:-1] = (levels[1:] + levels[:-1]) / 2
            boundaries[0] = -float("inf")
            boundaries[-1] = float("inf")

            new_levels = []
            for i in range(self.n_levels):
                b_left, b_right = boundaries[i], boundaries[i + 1]

                def f(x):
                    return x * norm.pdf(x)

                integral_num = integrate.quad(f, b_left, b_right)[0]
                integral_den = integrate.quad(norm.pdf, b_left, b_right)[0]
                if integral_den > 1e-10:
                    new_levels.append(integral_num / integral_den)
                else:
                    new_levels.append(levels[i])
            levels = np.array(new_levels)
        return torch.tensor(levels, dtype=torch.float32)

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        return x + (xq - x).detach()


class GaussianClipQuantizer(GaussianSTEQuantizer):
    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (x_norm.abs() <= self.levels[-1]).float()
        return x * mask + (xq - x * mask).detach()


class GaussianTrustQuantizer(GaussianSTEQuantizer):
    def __init__(self, bits=4, trust=None):
        super().__init__(bits)
        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            xq = xq @ self.matrix.T
            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


FP4_LEVELS = [
    -2.92247856,
    -1.94831904,
    -1.46123928,
    -0.97415952,
    -0.73061964,
    -0.48707976,
    -0.24353988,
    0.0,
    0.0,
    0.24353988,
    0.48707976,
    0.73061964,
    0.97415952,
    1.46123928,
    1.94831904,
    2.92247856,
]


class FP4STEQuantizer(GaussianSTEQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class FP4ClipQuantizer(GaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class FP4TrustQuantizer(GaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HalfHadamardFP4ClipQuantizer(HalfHadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HadamardFP4ClipQuantizer(HadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HalfHadamardFP4TrustQuantizer(HalfHadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class HadamardFP4TrustQuantizer(HadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class FourEightMaskedQuantizer(BaseQuantizer):
    def __init__(self, p=2.0):
        super().__init__(16)
        self.p = p

    def forward(self, x):
        x_reshaped = x.reshape(-1, 4, 2)
        _, idx = x_reshaped.norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        mask = torch.ones_like(x_reshaped, dtype=torch.bool)
        mask[torch.arange(x_reshaped.size(0)).repeat(2, 1).T, idx, :] = False
        mask = mask.reshape(x.shape).float()

        return x * mask


class FourEightSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, p: float = 2.0):
        super().__init__(bits)
        self.p = p

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        return x + (xq - x).detach()


class FourEightClipQuantizer(FourEightSTEQuantizer):
    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(x) <= scale).float()
        return x * mask + (xq - x * mask).detach()


class FourEightTrustQuantizer(FourEightSTEQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, p)
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, trust)
        self.p = p

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, trust)
        self.p = p

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


# torch._dynamo.config.optimize_ddp=False # uncommend if actually using ErfClipQuantizer
class ErfFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xq, buffer, mask):
        ctx.save_for_backward(buffer, mask)
        return xq

    @staticmethod
    def backward(ctx, grad_output):
        buffer, mask = ctx.saved_tensors
        mask = mask.float()

        return (
            (grad_output + buffer) * mask,
            None,
            grad_output * (1 - mask) - buffer * mask,
            None,
        )


class ErfClipQuantizer(ClipQuantizer):
    def __init__(self, bits=4, acc_dtype=torch.float32):
        super().__init__(bits, True)
        self.acc_dtype = acc_dtype
        self.register_parameter("acc", None)

    def forward(self, x):
        with torch.no_grad():
            if self.acc is None:
                self.acc = nn.Parameter(
                    torch.zeros_like(x, dtype=self.acc_dtype), requires_grad=True
                )
            elif self.acc.grad is not None:
                self.acc.data += self.acc.grad
                self.acc.grad = None

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        mask = (torch.abs(x) <= scale).float()

        return ErfFn().apply(x, xq, self.acc, mask)


class FlushAccFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, acc):
        ctx.save_for_backward(acc)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (acc,) = ctx.saved_tensors
        return grad_output + acc, None


class ClipAccQuantizer(STEQuantizer):
    def __init__(
        self,
        bits=4,
        centered=True,
        flush_every: int = 64,
        acc_dtype=torch.float32,
        scale: float = None,
    ):
        super().__init__(bits, centered)

        if scale is None:
            scale = 1 / flush_every

        self.acc_dtype = acc_dtype
        self.flush_every = flush_every
        self.counter = 0
        self.scale = scale
        self.register_buffer("acc", None)

    def forward(self, x):
        with torch.no_grad():
            if self.counter == 0:
                if self.acc is None:
                    self.acc = torch.zeros_like(x, dtype=self.acc_dtype)
                else:
                    self.acc.data = torch.zeros_like(x, dtype=self.acc_dtype)

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = ((neg_scale <= x) & (x <= scale)).float()

        self.counter += 1
        if self.counter == self.flush_every:
            self.counter = 0
            grad_flow_output = FlushAccFn().apply(
                x * mask + x * (1 - mask) * self.scale,
                (self.acc * self.scale).to(x.dtype),
            )
        else:
            grad_flow_output = x * mask + self.acc * (1 - mask)

        return grad_flow_output + (xq - grad_flow_output).detach()


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
        if not self.is_alpha_init:
            with torch.no_grad():
                step = self.get_initial_step_value(x)
                self.alpha_weight.data.multiply_(
                    torch.tensor(
                        step,
                        dtype=self.alpha_weight.dtype,
                        device=self.alpha_weight.device,
                    )
                )
            self.is_alpha_init = True
        return self.alpha_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step

        return xq + step * 1e-9  # extra term to ensure gradient flow
    
    def entropy(self, x):
        with torch.no_grad():
            step = self.get_learnable_step(x)
            step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
            xs = x / step
            if self.raise_zero:
                xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
                x_int = self.ste_cast(xsc).int()
                prob: torch.Tensor = torch.unique(x_int, return_counts=True)[1]
                prob = prob / prob.sum()
                assert prob.numel() <= self.n_levels
                return torch.sum(-prob * torch.log2(prob))
            else:
                xsc = torch.clamp(xs, self.q_min, self.q_max)
                x_int = self.ste_cast(xsc).int()
                prob: torch.Tensor = torch.unique(x_int, return_counts=True)[1]
                prob = prob / prob.sum()
                assert prob.numel() <= self.n_levels
                return torch.sum(-prob * torch.log2(prob))


class LSQPlusWeightQuantizer(LSQQuantizer):
    @torch.no_grad()
    def get_initial_step_value(self, x):
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(x**2)) + 1e-8
        step = 2 * scale / (self.n_levels - 1)
        return step


class LSQPlusActivationQuantizer(LSQPlusWeightQuantizer):
    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__(bits, raise_zero, all_positive, **kwargs)
        self.beta_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.is_beta_init = False

    @torch.no_grad()
    def get_initial_bias_value(self, x):
        return x.min() - self.alpha_weight * self.q_min

    def get_learnable_bias(self, x):
        if not self.is_beta_init:
            with torch.no_grad():
                bias = self.get_initial_bias_value(x)
                self.beta_weight.data.add_(
                    torch.tensor(
                        bias,
                        dtype=self.beta_weight.dtype,
                        device=self.beta_weight.device,
                    )
                )
            self.is_beta_init = True
        return self.beta_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        bias = self.get_learnable_bias(x)
        bias = self.grad_scale(bias, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = (x - bias) / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step + bias
        return xq + step * 1e-9  # extra term to ensure gradient flow


class PACTQuantizer(LSQQuantizer):
    """
    Implementation of PACT quantizer from https://arxiv.org/abs/1805.06085
    PACT and LSQ are quite similar and do the same thing for forward pass.
    The difference is in the backward pass where PACT does not perform a full gradient flow.
    """

    def forward(self, x):
        step = self.get_learnable_step(x)
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs - 1 / 2)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs)
            xscr = self.ste_cast(xsc)
        xq = xscr * step
        xq = xq * clamp_mask + (xq - xq * clamp_mask).detach()
        return xq + step * 1e-9  # extra term to ensure gradient flow

class StretchedElasticQuant(torch.autograd.Function):
    """
    Copied from https://github.com/facebookresearch/ParetoQ/blob/main/models/utils_quant.py
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, log_entropy):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
            if log_entropy is not None:
                log_entropy(q_w)
        else:
            mytemp = torch.round(
                torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
            )
            if log_entropy is not None:
                log_entropy(mytemp)
            q_w = (
                mytemp
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum()
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                        - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                    - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None, None

class StretchedElasticQuantizer(torch.nn.Module):
    """
    Modified from https://github.com/facebookresearch/ParetoQ/blob/main/models/utils_quant.py
    """
    def __init__(self, bits, channels=1, weight_layerwise=False):
        super().__init__()
        self.w_bits = bits
        self.n_levels = (1 << bits)
        self.channels = channels
        self.weight_layerwise = weight_layerwise
        if self.weight_layerwise:
            self.weight_clip_val = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.weight_clip_val = nn.Parameter(torch.Tensor(channels, 1))
        self.init_clip_val = False
        self.ent = None

    @torch.no_grad()
    def log_ent(self, x: torch.Tensor):
        prob: torch.Tensor = torch.unique(x.int(), return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        self.ent = torch.sum(-prob * torch.log2(prob))

    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        self.ent = None
        StretchedElasticQuant.apply(
            x,
            self.weight_clip_val,
            self.w_bits,
            self.weight_layerwise,
            self.log_ent,
        )
        assert self.ent is not None
        return self.ent
    
    def forward(self, x: torch.Tensor):

        if (not torch.compiler.is_compiling()) and (not self.init_clip_val):
            self.init_clip_val = True
            with torch.no_grad():
                if self.weight_layerwise:
                    scale = x.abs().max()
                else:
                    scale, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
                self.weight_clip_val.copy_(scale)
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(self.weight_clip_val, src=0)
                print("init weight clip val", self.weight_clip_val.mean())

        x = StretchedElasticQuant.apply(
            x,
            self.weight_clip_val,
            self.w_bits,
            self.weight_layerwise,
            None,
        ).to(x.dtype)

        return x
    
    def extra_repr(self):
        return f"w_bits={self.w_bits}, channels={self.channels}, weight_layerwise={self.weight_layerwise}, init_clip_val={self.init_clip_val}"

class NormalFloatQuantizer(torch.nn.Module):
    def __init__(self, blocksize=512):
        super().__init__()
        self.ent = None
        self.n_levels = 16
        self.blocksize = blocksize
    
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            q, s = bitsandbytes.functional.quantize_nf4(x, blocksize=self.blocksize)
            xhat = bitsandbytes.functional.dequantize_nf4(q, s, blocksize=self.blocksize)
        return xhat.detach() + x - x.detach()

    @torch.no_grad()
    def log_ent(self, x: torch.Tensor):
        prob: torch.Tensor = torch.unique(x.int(), return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        self.ent = torch.sum(-prob * torch.log2(prob))
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        self.ent = None
        q, s = bitsandbytes.functional.quantize_nf4(x, blocksize=self.blocksize)
        shift = torch.Tensor([0, 4]).to(torch.uint8).to(x.device)
        data = (q >> shift) & 0b1111
        self.log_ent(data)
        assert self.ent is not None
        return self.ent

class BBQV5Base(torch.nn.Module):
    def __init__(
        self,
        precision: int, 
        zero_point: float,
        per_channel_rms: bool=False,
        ema_rrms: bool=False,
    ) -> None:
        super().__init__()
        # enforce integer precision
        assert precision > 0
        assert round(precision) == precision
        precision = round(precision)
        self.precision = precision
        self.ema_rrms = ema_rrms

        self.per_channel_rms = per_channel_rms
        self.n_levels = 1 << precision
        self.half_n_levels = 1 << (precision - 1)
        self.zero_point = zero_point

        self.sx = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.init_sx = False
        self.ent = torch.tensor(0.0)
        if self.ema_rrms:
            self.running_rrms: torch.Tensor
            self.register_buffer("running_rrms", torch.tensor(1.0))
            self.init_rrms = False

    def get_sx(self, x: torch.Tensor):
        if (not torch.compiler.is_compiling()) and (not self.init_sx):
            self.init_sx = True
            with torch.no_grad():
                sx = 1.694 * x.square().mean().sqrt()
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(sx, src=0)
                self.sx.data.copy_(torch.broadcast_to(sx, self.sx.shape))
                print("init_sx", self.sx.data.mean())
        sx = self.sx
        sx_backward = sx * (1 / (x.numel() ** 0.5))
        return sx.detach() + sx_backward - sx_backward.detach()

    def transform(self, x: torch.Tensor):
        raise NotImplementedError()
    

    def quant(self, x: torch.Tensor):
        x = self.transform(x)
        sx = self.get_sx(x) # must be here to initialize sx with information from x
        
        if self.per_channel_rms:
            rrms = 1 / x.square().mean(dim=-1, keepdim=True).sqrt().clip(min=1e-8)
        else:
            rrms = x.square().mean().rsqrt()

        if self.ema_rrms:
            if (not torch.compiler.is_compiling()) and (not self.init_rrms):
                self.init_rrms = True
                with torch.no_grad():
                    self.running_rrms.data.copy_(rrms)
                    if torch.distributed.is_initialized():
                        torch.distributed.broadcast(self.running_rrms.data, src=0)
                    print(f"init running rrms {self.running_rrms}")
            elif not self.training:
                rrms = self.running_rrms
            else:
                with torch.no_grad():
                    self.running_rrms.data = 0.99 * self.running_rrms.data + 0.01 * rrms
        
        x = 0.5 * (1 + torch.erf((2 ** (-0.5) * rrms) * x)) # within range (0, 1)
        x = x * self.n_levels - 0.5 # within range (-0.5, 2^p-0.5)

        # the round is required, the clip should be noop if torch.round is 
        # the real round (break ties with the ceiling integer)
        # however, torch.round break ties with the even integer, 
        # hence the need to do an extra clip
        x_forward = torch.round(x).clip(0, self.n_levels - 1) # within range [0, 2^p-1]
        x = x_forward.detach() + x - x.detach()

        x = x - (self.half_n_levels + self.zero_point) # within range [-2^(p-1)-z, 2^(p-1)-1-z]
        
        return sx / self.half_n_levels, x

    def forward(self, x: torch.Tensor):
        s, q = self.quant(x)
        return s * q
    
    def entropy(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.transform(x)
            
            if self.per_channel_rms:
                rrms = 1 / x.square().mean(dim=-1, keepdim=True).sqrt().clip(min=1e-8)
            else:
                rrms = x.square().mean().rsqrt()

            if self.ema_rrms:
                if not self.training:
                    rrms = self.running_rrms
            
            x = 0.5 * (1 + torch.erf((2 ** (-0.5) * rrms) * x)) # within range (0, 1)
            x = x * self.n_levels - 0.5 # within range (-0.5, 2^p-0.5)

            x_int = torch.round(x).clip(0, self.n_levels - 1).int() # within range [0, 2^p-1]
            prob: torch.Tensor = torch.unique(x_int, return_counts=True)[1]
            prob = prob / prob.sum()
            assert prob.numel() <= self.n_levels
            return torch.sum(-prob * torch.log2(prob))

class BBQV5HD(BBQV5Base):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )
    def __init__(
        self,
        precision: int, 
        zero_point: float,
        per_channel_rms: bool=False,
        ema_rrms=False,
    ) -> None:
        super().__init__(precision=precision, zero_point=zero_point, per_channel_rms=per_channel_rms, ema_rrms=ema_rrms)
        self.matrix = None

    def transform(self, x: torch.Tensor):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )
        
        return x @ self.matrix
    
    def extra_repr(self):
        return f"precision={self.precision}, n={self.n_levels}, hn={self.half_n_levels}, per_channel_rms={self.per_channel_rms}, init_sx={self.init_sx}, zero_point={self.zero_point}, ema_rrms={self.ema_rrms}"
    
class BBQV5HDChan(BBQV5HD):
    def __init__(
        self,
        precision: int, 
        zero_point: float,
        channels: int,
        ema_rrms=False,
    ) -> None:
        super().__init__(precision=precision, zero_point=zero_point, per_channel_rms=True, ema_rrms=ema_rrms)
        self.channels = channels
        self.sx = torch.nn.Parameter(torch.full((channels, 1), 1.0), requires_grad=True)

    def extra_repr(self):
        return f"{super().extra_repr()}, channels={self.channels}"

class LTQ(nn.Module):
    '''
    https://github.com/liuzechun/Nonuniform-to-Uniform-Quantization/blob/0140532b79168fe982b3bbbd936c6131de14b344/resnet/resnet.py#L36
    '''
    def __init__(self, num_bits):
        super(LTQ, self).__init__()
        init_range = 2.0
        self.n_val = 2 ** num_bits - 1
        self.interval = nn.Parameter(torch.tensor(init_range / self.n_val), requires_grad=False)
        self.start = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.two =nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one =nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)
        self.init = False

    def forward(self, x: torch.Tensor):
        # modified to match zero-mean dist
        if (not torch.compiler.is_compiling()) and (not self.init):
            self.init = True
            with torch.no_grad():
                a_init = (x.square().mean().sqrt() * 4) / self.n_val
                self.start.copy_(-a_init * self.n_val / 2)
                self.a.copy_(a_init)
                self.zero.copy_(-a_init * self.n_val / 2)
                self.interval.copy_(a_init)
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(self.start, src=0)
                    torch.distributed.broadcast(self.a, src=0)
                    torch.distributed.broadcast(self.zero, src=0)
                    torch.distributed.broadcast(self.interval, src=0)
                print("init", self.start.detach(), self.a.detach(), self.zero.detach(), self.interval.detach())

        x = x * self.scale1

        x_forward = x
        x_backward = x
        step_right = self.zero + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                thre_backward = self.start + 0.0
                x_forward = torch.where(x > thre_forward, step_right, self.zero)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, self.zero)
            else:
                thre_forward += a_pos[i-1] / 2 +  a_pos[i] / 2
                thre_backward += a_pos[i-1]
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, x_backward)

        thre_backward += a_pos[i]
        x_backward = torch.where(x > thre_backward, self.two, x_backward)

        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out * self.scale2

        return out
    

class LTQW(nn.Module):
    '''
    https://github.com/liuzechun/Nonuniform-to-Uniform-Quantization/blob/0140532b79168fe982b3bbbd936c6131de14b344/resnet/resnet.py#L84
    '''
    def __init__(self, num_bits):
        super(LTQW, self).__init__()
        self.n_levels = 1 << num_bits
        self.num_bits = num_bits
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=True)
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)

    @torch.no_grad()
    def log_ent(self, x: torch.Tensor):
        prob: torch.Tensor = torch.unique(x.int(), return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        self.ent = torch.sum(-prob * torch.log2(prob))

    @torch.no_grad()
    def entropy(self, real_weights: torch.Tensor):

        gamma = (2**self.num_bits - 1)/(2**(self.num_bits - 1))
        scaling_factor = gamma * torch.mean(torch.abs(real_weights))
        scaling_factor = scaling_factor.detach()

        scaled_weights = real_weights/scaling_factor

        cliped_weights = torch.where(scaled_weights < self.clip_val/2, scaled_weights, self.clip_val/2)
        cliped_weights = torch.where(cliped_weights > -self.clip_val/2, cliped_weights, -self.clip_val/2)

        n = float(2 ** self.num_bits - 1) / self.clip_val

        temp = torch.round((cliped_weights + self.clip_val/2) * n)
        self.log_ent(temp)
        assert self.ent is not None
        return self.ent
    
    def forward(self, real_weights: torch.Tensor):

        gamma = (2**self.num_bits - 1)/(2**(self.num_bits - 1))
        scaling_factor = gamma * torch.mean(torch.abs(real_weights), dim=-1, keepdim=True)
        scaling_factor = scaling_factor.detach()

        scaled_weights = real_weights/scaling_factor

        cliped_weights = torch.where(scaled_weights < self.clip_val/2, scaled_weights, self.clip_val/2)
        cliped_weights = torch.where(cliped_weights > -self.clip_val/2, cliped_weights, -self.clip_val/2)

        n = float(2 ** self.num_bits - 1) / self.clip_val

        temp = torch.round((cliped_weights + self.clip_val/2) * n)
        quan_weights_no_grad = scaling_factor * (temp / n - self.clip_val/2)

        quan_weights = quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights

        return quan_weights


from .ablation_nohd import BBQV5NoHD, BBQV5NoHDChan
from .ablation_nols import BBQV5NoLSHD, BBQV5NoLSHDChan
from .ablation_nolsnoinit import BBQV5NoLSNoInitHD, BBQV5NoLSNoInitHDChan
from .ablation_norms import BBQV5NoRMSHD, BBQV5NoRMSHDChan

QUANTIZER_CLASSES = {
    "BBQV5NoHD": BBQV5NoHD,
    "BBQV5NoHDChan": BBQV5NoHDChan,
    "BBQV5NoLSHD": BBQV5NoLSHD,
    "BBQV5NoLSHDChan": BBQV5NoLSHDChan,
    "BBQV5NoLSNoInitHD": BBQV5NoLSNoInitHD,
    "BBQV5NoLSNoInitHDChan": BBQV5NoLSNoInitHDChan,
    "BBQV5NoRMSHD": BBQV5NoRMSHD,
    "BBQV5NoRMSHDChan": BBQV5NoRMSHDChan,
    "NoQuantizer": NoQuantizer,
    "UniformQuantizer": UniformQuantizer,
    "STEQuantizer": STEQuantizer,
    "ClipQuantizer": ClipQuantizer,
    "HalfHadamardClipQuantizer": HalfHadamardClipQuantizer,
    "HadamardClipQuantizer": HadamardClipQuantizer,
    "TrustQuantizer": TrustQuantizer,
    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    "HadamardTrustQuantizer": HadamardTrustQuantizer,
    "GaussianSTEQuantizer": GaussianSTEQuantizer,
    "GaussianClipQuantizer": GaussianClipQuantizer,
    "GaussianTrustQuantizer": GaussianTrustQuantizer,
    "HadamardGaussianClipQuantizer": HadamardGaussianClipQuantizer,
    "HalfHadamardGaussianTrustQuantizer": HalfHadamardGaussianTrustQuantizer,
    "HadamaardGaussianTrustQuantizer": HadamardGaussianTrustQuantizer,
    "FP4STEQuantizer": FP4STEQuantizer,
    "FP4ClipQuantizer": FP4ClipQuantizer,
    "FP4TrustQuantizer": FP4TrustQuantizer,
    "HalfHadamardFP4ClipQuantizer": HalfHadamardFP4ClipQuantizer,
    "HadamardFP4ClipQuantizer": HadamardFP4ClipQuantizer,
    "HalfHadamardFP4TrustQuantizer": HalfHadamardFP4TrustQuantizer,
    "HadamardFP4TrustQuantizer": HadamardFP4TrustQuantizer,
    "FourEightMaskedQuantizer": FourEightMaskedQuantizer,
    "FourEightSTEQuantizer": FourEightSTEQuantizer,
    "FourEightClipQuantizer": FourEightClipQuantizer,
    "FourEightTrustQuantizer": FourEightTrustQuantizer,
    "HalfHadamardFourEightTrustQuantizer": HalfHadamardFourEightTrustQuantizer,
    "HadamardFourEightTrustQuantizer": HadamardFourEightTrustQuantizer,
    "ErfClipQuantizer": ErfClipQuantizer,
    "ClipAccQuantizer": ClipAccQuantizer,
    "PACTQuantizer": PACTQuantizer,
    "LSQQuantizer": LSQQuantizer,
    "LSQPlusActivationQuantizer": LSQPlusActivationQuantizer,
    "LSQPlusWeightQuantizer": LSQPlusWeightQuantizer,
    "BBQV5HD": BBQV5HD,
    "BBQV5HDChan": BBQV5HDChan,
    "StretchedElasticQuantizer": StretchedElasticQuantizer,
    "NormalFloatQuantizer": NormalFloatQuantizer,
    "NFQuantizer": NFQuantizer,
    "LTQW": LTQW,
    "LTQ": LTQ,
}

@torch.jit.script
def quant_matmul(qx: torch.Tensor, qw: torch.Tensor, xscale: torch.Tensor, wscale: torch.Tensor, scale: torch.Tensor, b: int, n: int, o: int):
    out = torch._scaled_mm(qx, qw.T, xscale, wscale, out_dtype=torch.bfloat16)
    out = out.view((b, n, o))
    return out * scale

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
        if self.weight is not None:
            x = self.activation_quantizer(x)
            w = self.weight_quantizer(self.weight)
            return F.linear(x, w, self.bias)
        else:
            with torch.no_grad():
                if isinstance(self.weight_quantizer, BBQV5HD):
                    qx = self.quant_act_triton(x)
                    b, n, c = qx.shape

                    qx = qx.view((b * n, c))
                    if self.datatype == "fp4":
                        if self.xscale is None:
                            self.xscale = torch.ones((qx.numel() // 8, ), device="cuda", dtype=torch.float8_e4m3fn)
                        return quant_matmul(qx, self.qw, self.xscale, self.wscale, self.scale, b, n, self.out_features)
                    else:
                        return i4mm.run(qx, self.qw).view((b, n, self.out_features)) * self.scale
                else:
                    assert isinstance(self.weight_quantizer, NoQuantizer)
                    assert isinstance(self.activation_quantizer, NoQuantizer)
                    data, state = bitsandbytes.functional.quantize_nf4(x, blocksize=512)
                    x = bitsandbytes.functional.dequantize_nf4(data, state, blocksize=512)
                    return bitsandbytes.matmul_4bit(x, self.data, self.state)
    
    def quant_act_triton(self, input: torch.Tensor):
        assert isinstance(self.weight_quantizer, BBQV5HDChan)
        assert isinstance(self.activation_quantizer, BBQV5HD)
        assert not isinstance(self.activation_quantizer, BBQV5HDChan)
        assert not self.activation_quantizer.per_channel_rms
        assert self.activation_quantizer.ema_rrms
        shape = (np.prod(input.shape[:-1]).item(), input.shape[-1])
        BLOCKSIZE = 128
        qfp4 = torch.empty((shape[0], shape[1] // 2), device="cuda", dtype=torch.uint8)
        bbq[(shape[0] // BLOCKSIZE, shape[1] // BLOCKSIZE)](
            input, 
            self.activation_quantizer.aux_matrix, 
            self.activation_quantizer.running_rrms, 
            qfp4, 
            shape[-1], 
            BLOCKSIZE=BLOCKSIZE, 
            DTYPE=self.datatype
        )

        oshape = input.shape
        if self.datatype == "fp4":
            return qfp4.view((*oshape[:-1], oshape[-1] // 2)).view(torch.float4_e2m1fn_x2)
        else:
            return qfp4.view((*oshape[:-1], oshape[-1] // 2))
    
    def float2fp4(self, input: torch.Tensor):
        assert input.is_contiguous()
        # record shape and flatten
        device = input.device
        shape = input.shape
        input = input.flatten().to(torch.float32)

        # measure distance to closest fp4 value, must be 
        values = torch.Tensor([
            0.0,  0.5,  1.0,  1.5,
            2.0,  3.0,  4.0,  6.0,
            0.0, -0.5, -1.0, -1.5,
            -2.0, -3.0, -4.0, -6.0,
        ]).to(torch.float32).to(device)
        dist = torch.abs(input[:,None] - values[None, :])
        
        # must be "equal to" at least one fp4 value
        assert torch.all(torch.amin(dist, dim=-1) < 5e-7)

        # convert to binary representation
        idx = torch.argmin(dist, dim=-1)
        binary = torch.arange(0, 16).to(torch.uint8).to(input.device)
        out = binary[idx]

        # merge every two value into one
        out = torch.reshape(out, (np.prod(shape[:-1]).item(), shape[-1] // 2, 2))
        shift = torch.Tensor([0, 4]).to(torch.uint8).to(input.device)[None, None, :]
        out = torch.sum(out << shift, dim=-1).to(torch.uint8)
        out = out.reshape((*shape[:-1], shape[-1] // 2))
        return out.view(torch.float4_e2m1fn_x2)
        
    def float2int4(self, input: torch.Tensor):
        assert input.is_contiguous()
        # record shape and flatten
        device = input.device
        shape = input.shape
        input = input.flatten().to(torch.float32)

        # measure distance to closest int4 value, must be 
        values = torch.Tensor([
             0.0,  1.0,  2.0,  3.0,
             4.0,  5.0,  6.0,  7.0,
            -8.0, -7.0, -6.0, -5.0, 
            -4.0, -3.0, -2.0, -1.0,
        ]).to(torch.float32).to(device)
        dist = torch.abs(input[:,None] - values[None, :])
        
        # must be "equal to" at least one int4 value
        assert torch.all(torch.amin(dist, dim=-1) < 5e-7)

        # convert to binary representation
        idx = torch.argmin(dist, dim=-1)
        binary = torch.arange(0, 16).to(torch.uint8).to(input.device)
        out = binary[idx]

        # merge every two value into one
        out = torch.reshape(out, (np.prod(shape[:-1]).item(), shape[-1] // 2, 2))
        shift = torch.Tensor([0, 4]).to(torch.uint8).to(input.device)[None, None, :]
        out = torch.sum(out << shift, dim=-1).to(torch.uint8)
        out = out.reshape((*shape[:-1], shape[-1] // 2))
        return out
    
    @torch.no_grad()
    def realquant(self, datatype="fp4"):
        if isinstance(self.weight_quantizer, BBQV5HD):
            assert isinstance(self.weight_quantizer, BBQV5HDChan)
            assert isinstance(self.activation_quantizer, BBQV5HD)
            assert not isinstance(self.activation_quantizer, BBQV5HDChan)
            assert not self.activation_quantizer.per_channel_rms
            assert self.activation_quantizer.ema_rrms
            self.datatype = datatype

            sw, qw = self.weight_quantizer.quant(self.weight)
            sx = self.activation_quantizer.sx / self.activation_quantizer.half_n_levels
            if datatype == "fp4":
                self.qw = self.float2fp4(qw)
                self.scale = (torch.flatten(sw)[None, None, :] * sx.item()).to(torch.bfloat16)
                self.weight = None
                self.wscale = torch.ones((self.qw.numel() // 8, ), device="cuda", dtype=torch.float8_e4m3fn)
                self.xscale = None
            else:
                self.qw = self.float2int4(qw)
                self.scale = (torch.flatten(sw)[None, None, :] * sx.item()).to(torch.bfloat16)
                self.weight = None
        else:
            assert isinstance(self.weight_quantizer, NoQuantizer)
            assert isinstance(self.activation_quantizer, NoQuantizer)
            data, state = bitsandbytes.functional.quantize_nf4(self.weight.T, blocksize=512)
            self.data = data
            self.state = state
            self.weight = None
