import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6, 3]
plt.rcParams.update({'font.size': 16})
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
import triton
import math
from fast_hadamard_transform import hadamard_transform

@triton.jit
def normal_cdf_and_round_3bit(v: tl.tensor, DTYPE: tl.constexpr):
    if DTYPE == "int4":
        qmz = tl.where(
            v >= 0.0,
            tl.where(
                v >= 0.6744897501960818,
                tl.where(v >= 1.1503493803760083, 0b0011, 0b0010),
                tl.where(v >= 0.3186393639643752, 0b0001, 0b0000),
            ),
            tl.where(
                v >= -0.6744897501960818,
                tl.where(v >= -0.3186393639643752, 0b1111, 0b1110),
                tl.where(v >= -1.1503493803760083, 0b1101, 0b1100),
            )
        ).to(tl.int8)
        return qmz
    elif DTYPE == "fp4":
        qmz = tl.where(
            v >= 0.0,
            tl.where(
                v >= 0.6744897501960818,
                tl.where(v >= 1.1503493803760083, 0b0101, 0b0100),
                tl.where(v >= 0.3186393639643752, 0b0010, 0b0000),
            ),
            tl.where(
                v >= -0.6744897501960818,
                tl.where(v >= -0.3186393639643752, 0b1010, 0b1100),
                tl.where(v >= -1.1503493803760083, 0b1101, 0b1110),
            )
        ).to(tl.int8)
        return qmz

@triton.jit
def bbq(
    x_ptr,  
    h_ptr,
    rsigma_ptr,
    qmz_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCKSIZE: tl.constexpr=128, 
    DTYPE: tl.constexpr = "int4"
):
    rid = tl.program_id(axis=0) 
    cid = tl.program_id(axis=1)  
    xrids = rid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)[:, None]
    xcids = cid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)[None, :]
    xids = xrids * ROW_SIZE + xcids

    hrids = tl.arange(0, BLOCKSIZE)[:, None]
    hcids = tl.arange(0, BLOCKSIZE)[None, :]
    hids = hrids * BLOCKSIZE + hcids
    
    # load a block of x
    x = tl.load(x_ptr + xids).to(tl.bfloat16)
    # load pre-computed hadamard matrix shared across the entire kernel
    h = tl.load(h_ptr + hids)
    # load reciprocal of sigma
    rsigma = tl.load(rsigma_ptr)

    # hadamard transform
    v = tl.dot(x, h) * rsigma

    # accelerated normal CDF and rounding
    qmz = normal_cdf_and_round_3bit(v, DTYPE)

    # pack two 4-bit data type into a single byte
    shift = tl.arange(0, 2)[None, None, :] * 4
    qmz = tl.reshape(qmz, (BLOCKSIZE, BLOCKSIZE // 2, 2)) << shift
    qmz = tl.xor_sum(qmz, axis=-1, keep_dims=False)

    # write quantized data to memory
    qmzrids = rid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)[:, None]
    qmzcids = cid * (BLOCKSIZE // 2) + tl.arange(0, BLOCKSIZE // 2)[None, :]
    qmzids = qmzrids * (ROW_SIZE // 2) + qmzcids
    tl.store(qmz_ptr + qmzids, qmz)

def test():
    normal = torch.distributions.Normal(0, 1)
    LOG_BLOCKSIZE = 7
    BLOCKSIZE = 1 << LOG_BLOCKSIZE
    x = torch.randn((BLOCKSIZE, BLOCKSIZE), device="cuda", dtype=torch.bfloat16)
    h = hadamard_transform(
        torch.eye(BLOCKSIZE, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-LOG_BLOCKSIZE / 2)
    )
    rsigma = torch.rand((), device="cuda", dtype=torch.float32)
    shift = torch.arange(0, 2, device="cuda", dtype=torch.int32)[None, None, :] * 4
    
    xshape = x.shape
    yexp = torch.reshape(x.reshape((*xshape[:-1], xshape[-1] // BLOCKSIZE, BLOCKSIZE)) @ h, xshape) * rsigma
    yexp = (torch.floor(normal.cdf(yexp.to(torch.float64)) * 8) - 4).to(torch.int8)

    yact = torch.zeros((x.shape[0], x.shape[1] // 2), device="cuda", dtype=torch.int8)
    bbq[(xshape[0] // BLOCKSIZE, xshape[1] // BLOCKSIZE)](x, h, rsigma, yact, xshape[-1], BLOCKSIZE=BLOCKSIZE)
    yact = torch.reshape((yact[:, :, None] >> shift) & 0b1111, xshape)
    yact = torch.where(yact >= 8, -(16 - yact), yact)

    err = torch.abs(yexp - yact)
    assert torch.all(err < 2)
    ratio = (err < 1).sum() / err.numel()
    print(ratio)
    assert ratio > 0.995

@triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N"],  # Argument names to use as an x-axis for the plot
            x_vals=[256 * i for i in range(2, 32)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["bbq"],  # Label name for the lines
            line_names=["bbq"],  # Line styles
            styles=[("blue", "-")],
            ylabel="ms",  # Label name for the y-axis
            plot_name="quant-performance",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    ]
)
def benchmark(N, provider):

    quantiles = [0.5, 0.2, 0.8]
    
    LOG_BLOCKSIZE = 7
    BLOCKSIZE = 1 << LOG_BLOCKSIZE
    x = torch.randn((N, N), device="cuda", dtype=torch.bfloat16)
    h = hadamard_transform(
        torch.eye(BLOCKSIZE, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-LOG_BLOCKSIZE / 2)
    )
    rsigma = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    xshape = x.shape
    yact = torch.zeros_like(x, device="cuda", dtype=torch.int8)

    quant = lambda : bbq[(xshape[0] // BLOCKSIZE, xshape[1] // BLOCKSIZE)](x, h, rsigma, yact, xshape[-1], BLOCKSIZE=BLOCKSIZE)

    ms, min_ms, max_ms = triton.testing.do_bench(quant, quantiles=quantiles, warmup=100, rep=1000)

    print(provider, N, ms)
    return ms, max_ms, min_ms

if __name__ == "__main__":
    test()
    benchmark.run(save_path="./quant")
    
