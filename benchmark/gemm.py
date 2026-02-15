import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6, 3]
plt.rcParams.update({'font.size': 16})
import triton
from torch.profiler import profile, ProfilerActivity, record_function

raw = torch.randint(0, 256, (256, 256), device="cuda", dtype=torch.uint8)
a = raw.view(torch.float4_e2m1fn_x2)
raw = torch.randint(0, 256, (256, 256), device="cuda", dtype=torch.uint8)
b = raw.view(torch.float4_e2m1fn_x2).T

ascale = torch.ones((a.numel() // 8, ), device="cuda", dtype=torch.float8_e4m3fn)
bscale = torch.ones((b.numel() // 8, ), device="cuda", dtype=torch.float8_e4m3fn)

def trace_handler(prof):
    # please verify the following print statement prints some thing like
    # "cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu"
    # to ensure that the GPU is indeed using its tensor core's FP4 capability
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=200,                 # show more rows
        max_name_column_width=200      # << widen name column
    ))

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=trace_handler) as prof:
    c = torch._scaled_mm(a, b, ascale, bscale, out_dtype=torch.bfloat16)

@triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N"],  # Argument names to use as an x-axis for the plot
            x_vals=[256 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["fp4", "int8", "fp16"],  # Label name for the lines
            line_names=["fp4", "int8", "fp16"],  # Line styles
            styles=[("green", "-"), ("red", "-"), ("blue", "-")],
            ylabel="ms",  # Label name for the y-axis
            plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    ]
)
def benchmark(N, provider):

    quantiles = [0.5, 0.2, 0.8]
    if provider == "fp4":
        raw = torch.randint(0, 256, (N, N // 2), device="cuda", dtype=torch.uint8)
        a = raw.view(torch.float4_e2m1fn_x2)
        raw = torch.randint(0, 256, (N, N // 2), device="cuda", dtype=torch.uint8)
        b = raw.view(torch.float4_e2m1fn_x2).T

        ascale = torch.ones((a.numel() // 8,), device="cuda", dtype=torch.float8_e4m3fn)
        bscale = torch.ones((b.numel() // 8, ), device="cuda", dtype=torch.float8_e4m3fn)
        fp4_matmul = lambda: torch._scaled_mm(a, b, ascale, bscale, out_dtype=torch.float32)
        ms, min_ms, max_ms = triton.testing.do_bench(fp4_matmul, quantiles=quantiles, warmup=100, rep=1000)
    elif provider == "int8":
        a = torch.randint(-128, 128, (N, N), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (N, N), device="cuda", dtype=torch.int8).T
        int8_matmul = lambda: torch._int_mm(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(int8_matmul, quantiles=quantiles, warmup=100, rep=1000)
    elif provider == "fp16":
        a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16).T
        fp16_matmul = lambda: torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(fp16_matmul, quantiles=quantiles, warmup=100, rep=1000)
    else:
        assert False

    print(provider, N, ms)
    return ms, max_ms, min_ms


benchmark.run(save_path="./gemm")