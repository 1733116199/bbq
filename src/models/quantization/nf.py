import torch
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
import triton
import math
import bitsandbytes.functional
import matplotlib.pyplot as plt

@triton.jit
def nf_quant(
    x_ptr,  
    m_ptr,
    q_ptr,  
    y_ptr,
    nx, 
    BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr=1024, 
):
    pid = tl.program_id(axis=0)  
    ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = ids < nx
    
    x = tl.load(x_ptr + ids, mask=mask, other=0.0)
    if BITS == 4:
        y = tl.where(
            x > tl.load(m_ptr + (8 - 1)),
            tl.where(
                x > tl.load(m_ptr + (12 - 1)),
                tl.where(
                    x > tl.load(m_ptr + (14 - 1)),
                    tl.where(
                        x > tl.load(m_ptr + (15 - 1)),
                        tl.load(q_ptr + (16 - 1)),
                        tl.load(q_ptr + (15 - 1)),
                    ),
                    tl.where(
                        x > tl.load(m_ptr + (13 - 1)),
                        tl.load(q_ptr + (14 - 1)),
                        tl.load(q_ptr + (13 - 1)),
                    )
                ),
                tl.where(
                    x > tl.load(m_ptr + (10 - 1)),
                    tl.where(
                        x > tl.load(m_ptr + (11 - 1)),
                        tl.load(q_ptr + (12 - 1)),
                        tl.load(q_ptr + (11 - 1)),
                    ),
                    tl.where(
                        x > tl.load(m_ptr + (9 - 1)),
                        tl.load(q_ptr + (10 - 1)),
                        tl.load(q_ptr + (9 - 1)),
                    )
                )
            ),
            tl.where(
                x > tl.load(m_ptr + (4 - 1)),
                tl.where(
                    x > tl.load(m_ptr + (6 - 1)),
                    tl.where(
                        x > tl.load(m_ptr + (7 - 1)),
                        tl.load(q_ptr + (8 - 1)),
                        tl.load(q_ptr + (7 - 1)),
                    ),
                    tl.where(
                        x > tl.load(m_ptr + (5 - 1)),
                        tl.load(q_ptr + (6 - 1)),
                        tl.load(q_ptr + (5 - 1)),
                    )
                ),
                tl.where(
                    x > tl.load(m_ptr + (2 - 1)),
                    tl.where(
                        x > tl.load(m_ptr + (3 - 1)),
                        tl.load(q_ptr + (4 - 1)),
                        tl.load(q_ptr + (3 - 1)),
                    ),
                    tl.where(
                        x > tl.load(m_ptr + (1 - 1)),
                        tl.load(q_ptr + (2 - 1)),
                        tl.load(q_ptr + (1 - 1)),
                    )
                )
            )
        )
    elif BITS == 3:
        y = tl.where(
            x > tl.load(m_ptr + (4 - 1)),
            tl.where(
                x > tl.load(m_ptr + (6 - 1)),
                tl.where(
                    x > tl.load(m_ptr + (7 - 1)),
                    tl.load(q_ptr + (8 - 1)),
                    tl.load(q_ptr + (7 - 1)),
                ),
                tl.where(
                    x > tl.load(m_ptr + (5 - 1)),
                    tl.load(q_ptr + (6 - 1)),
                    tl.load(q_ptr + (5 - 1)),
                )
            ),
            tl.where(
                x > tl.load(m_ptr + (2 - 1)),
                tl.where(
                    x > tl.load(m_ptr + (3 - 1)),
                    tl.load(q_ptr + (4 - 1)),
                    tl.load(q_ptr + (3 - 1)),
                ),
                tl.where(
                    x > tl.load(m_ptr + (1 - 1)),
                    tl.load(q_ptr + (2 - 1)),
                    tl.load(q_ptr + (1 - 1)),
                )
            )
        )
    else:
        y = tl.where(
            x > tl.load(m_ptr + (2 - 1)),
            tl.where(
                x > tl.load(m_ptr + (3 - 1)),
                tl.load(q_ptr + (4 - 1)),
                tl.load(q_ptr + (3 - 1)),
            ),
            tl.where(
                x > tl.load(m_ptr + (1 - 1)),
                tl.load(q_ptr + (2 - 1)),
                tl.load(q_ptr + (1 - 1)),
            )
        )

    tl.store(y_ptr + ids, y, mask=mask)

class NFQuantizer(torch.nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.ent = None
        self.bits = bits
        self.n_levels = (1 << bits)
        self.init_quantiles(bits)
        
    def init_quantiles(self, bits):
        assert bits >= 2
        offset_neg = 1 << (bits + 1)
        offset_pos = offset_neg - 2
        delta = 0.5 * (1 / offset_neg + 1 / offset_pos)
        step_neg = 1 << (bits - 1)
        step_pos = step_neg + 1
        p_neg = torch.linspace(delta, 0.5, step_neg)
        p_pos = torch.linspace(0.5, 1 - delta, step_pos)
        p = torch.cat((p_neg, p_pos[1:])).to(torch.float64)
        qs: torch.Tensor = torch.distributions.Normal(0, 1).icdf(p)
        dt = qs / qs.abs().max()
        mid = (dt[:-1] + dt[1:]) / 2
        self.boundaries = torch.nn.Parameter(mid.to(torch.float32), requires_grad=False)
        self.data_type = torch.nn.Parameter(dt.to(torch.float32), requires_grad=False)

    def bin_search(self, x: torch.Tensor, data: torch.Tensor):
        y = torch.zeros_like(x, dtype=data.dtype, device=x.device)
        blocksize = 1024
        nf_quant[(triton.cdiv(x.numel(), blocksize), )](x, self.boundaries, data, y, x.numel(), BITS=self.bits, BLOCK_SIZE=blocksize)
        return y
    
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            s = x.abs().amax(dim=-1, keepdim=True).clip(min=1e-6)
            xhat = self.bin_search(x / s, self.data_type) * s
        return xhat.detach() + x - x.detach()

    @torch.no_grad()
    def log_ent(self, x: torch.Tensor):
        prob: torch.Tensor = torch.unique(x.int(), return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        self.ent = torch.sum(-prob * torch.log2(prob))
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        s = x.abs().amax(dim=-1, keepdim=True).clip(min=1e-6)
        q = x / s
        qhat = self.bin_search(q, torch.arange(0, self.n_levels).to(x.device))
        self.log_ent(qhat)
        assert self.ent is not None
        return self.ent
    
    def extra_repr(self):
        return f"bits={self.bits}, n_levels={self.n_levels}, boundaries={self.boundaries}, data_type={self.data_type}"

def test():
    chansize = 512
    rand = torch.rand((1024, chansize)).cuda()


    quantizer = NFQuantizer(4).cuda()

    data, state = bitsandbytes.functional.quantize_nf4(rand, blocksize=chansize)
    exp = torch.reshape((data >> torch.Tensor([4, 0]).int().cuda()) & 0b1111, rand.shape)

    absmax = rand.abs().amax(dim=-1, keepdim=True)
    temp = rand / absmax
    act = quantizer.bin_search(temp, torch.arange(0, quantizer.n_levels).to(torch.uint8).cuda())

    print(quantizer.entropy(rand))

    assert torch.allclose(absmax.flatten(), state.absmax.flatten())

    assert torch.all(act == exp), f"{torch.abs(act - exp).max()}"
    print("PASSED")

    exp = bitsandbytes.functional.dequantize_nf4(data, state, blocksize=chansize)
    act = quantizer.forward(rand)
    assert torch.allclose(act, exp), f"{torch.abs(act - exp).max()}"
    print("PASSED")

if __name__ == "__main__":
    test()