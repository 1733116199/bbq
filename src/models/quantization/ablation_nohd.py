import torch
from fast_hadamard_transform import hadamard_transform


class BBQV5Copy(torch.nn.Module):
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

    def forward(self, x: torch.Tensor):
        x = self.transform(x)
        sx = self.get_sx(x) # must be here to initialize sx with information from x
        
        if self.per_channel_rms:
            rrms = 1 / x.square().mean(dim=-1, keepdim=True).sqrt().clip(min=1e-8)
        else:
            rrms = x.square().mean().rsqrt()

        if self.ema_rrms:
            if not self.training:
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
        
        return (sx / self.half_n_levels) * x
    
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

class BBQV5NoHD(BBQV5Copy):
    def __init__(
        self,
        precision: int, 
        zero_point: float,
        per_channel_rms: bool=False,
        ema_rrms=False,
    ) -> None:
        super().__init__(precision=precision, zero_point=zero_point, per_channel_rms=per_channel_rms, ema_rrms=ema_rrms)

    def transform(self, x: torch.Tensor):
        return x
    
    def extra_repr(self):
        return f"precision={self.precision}, n={self.n_levels}, hn={self.half_n_levels}, per_channel_rms={self.per_channel_rms}, init_sx={self.init_sx}, zero_point={self.zero_point}, ema_rrms={self.ema_rrms}"
    
class BBQV5NoHDChan(BBQV5NoHD):
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