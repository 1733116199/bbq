import torch
import numpy as np
from src.models.quantization.base_linear import BBQV5HD, BBQV5HDChan
from benchmark.quant import bbq


def float2fp4(input: torch.Tensor):
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

def fp4tofloat(input: torch.Tensor):
    assert input.is_contiguous()
    shape = input.shape
    device = input.device
    assert input.dtype == torch.float4_e2m1fn_x2
    input = input.view(torch.uint8)
    input = input.reshape((np.prod(shape[:-1]).item(), shape[-1], 1))
    shift = torch.Tensor([0, 4]).to(torch.uint8)[None, None, :].to(input.device)
    
    idx = torch.reshape((input >> shift) & 0b1111, (*shape[:-1], shape[-1] * 2))
    values = torch.Tensor([
        0.0,  0.5,  1.0,  1.5,
        2.0,  3.0,  4.0,  6.0,
        0.0, -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    ]).to(torch.float32).to(device)
    return values[idx.int()]

def quant_triton(input: torch.Tensor, quantizer: BBQV5HD):
    shape = (np.prod(input.shape[:-1]).item(), input.shape[-1])
    BLOCKSIZE = 128
    qfp4 = torch.zeros((shape[0], shape[1] // 2), device="cuda", dtype=torch.uint8)
    bbq[(shape[0] // BLOCKSIZE, shape[1] // BLOCKSIZE)](input, quantizer.aux_matrix, quantizer.running_rrms, qfp4, shape[-1], BLOCKSIZE=BLOCKSIZE, DTYPE="fp4")

    oshape = input.shape
    return qfp4.reshape((*oshape[:-1], oshape[-1] // 2)).view(torch.float4_e2m1fn_x2)


# activation quantization
x = torch.randn((64, 512, 640)).cuda()
act_quant = BBQV5HD(3, 0, ema_rrms=True).cuda()
for i in range(1000):
    act_quant(x)
sx, qx = act_quant.quant(x)

# weight quantization
w = torch.randn((640, 640)).cuda()
wei_quant = BBQV5HDChan(3, 0, 640).cuda()
for i in range(1000):
    wei_quant(w)
sw, qw = wei_quant.quant(w)

qxfp4 = quant_triton(x.to(torch.bfloat16), act_quant)
qxconverted = fp4tofloat(qxfp4)

err = torch.abs(qxconverted - qx)
bad = torch.sum(err > (1 + 1e-6))
ok = torch.sum(err > (1 - 1e-6)) / err.numel()
good = torch.sum(err < 1e-6) / err.numel()
assert bad == 0
assert ok < 0.01
assert good >= 0.99

qwfp4 = float2fp4(qw)
qwconverted = fp4tofloat(qwfp4)
assert torch.allclose(qw, qwconverted, atol=5e-7)

qxfp4 = qxfp4.reshape((np.prod(qxfp4.shape[:-1]).item(), qxfp4.shape[-1]))
xscale = torch.ones((qxfp4.numel() // 8, ), device="cuda", dtype=torch.float8_e4m3fn)
wscale = torch.ones((qwfp4.numel() // 8, ), device="cuda", dtype=torch.float8_e4m3fn)

act = torch._scaled_mm(qxfp4, qwfp4.T, xscale, wscale, out_dtype=torch.float32)
exp = torch.matmul(qxconverted.reshape((np.prod(qxconverted.shape[:-1]).item(), qxconverted.shape[-1])), qwconverted.T)

print(act.shape, exp.shape)
assert torch.allclose(act, exp)
print("PASSED")