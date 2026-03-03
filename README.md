# Boosting Entropy with Bell Box Quantization

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2603.01599)

This repository contains the implementation of our paper "Boosting Entropy with Bell Box Quantization", previously called "BBQ: Boosting Quantization Entropy with Bell Box Quantization". The paper is accepted at ICLR 2026. We provide scripts to recreate the tables and figures in our paper.

# Setup 

Create a conda environment and install dependencies (we recommend Python 3.10):

```bash
conda create -n env python=3.10
conda activate env
pip install -r requirements.txt
```

If your Nvidia GPU belongs to the 50 series, then you might need to manually install the `fast-hadamard-transform` uploaded in this package, instead of the one in `requirements.txt`.
```bash
cd fast-hadamard-transform;
python3 setup.py install;
```
Note that the only difference between the included version and the public version is the following two lines in `fast-hadamard-transform/setup.py`, which asks the `nvcc` compiler to generate byte code for Blackwell architectures.
```python
      cc_flag.append("-gencode")
      cc_flag.append("arch=compute_120,code=sm_120")
```

To run the source code in this package, you need the following hardware:
1. an Nvidia GPU that supports naive BF16 tensor ops (older Turing GPUs only support FP16 which will not work)
2. potentially 1 TB of disk space to store the raw C4 dataset (can be deleted later) and the pre-processed C4 dataset (used by the script).
3. Optional: one NVIDIA RTX 5090, and one NVIDIA A100 80GB to recreate latency profiling results in the paper.
4. Optional: four RTX 2080 Ti to recreate the vision model results.

# Code Structure and Sources
This repository is adapted from [QuEST](https://github.com/IST-DASLab/QuEST/tree/main) with a few modifications. The vision code in `vision/` is adapted from [Facebook/DeiT](https://github.com/facebookresearch/deit). 
We made the following changes to the QuEST source code.
1. We changed `src/models/quantization/base_linear.py` to include variants of the BBQ quantizer and add other baselines such as SEQ (ParetoQ) and LTQ (N2UQ). The final version of BBQ shown in the paper (Equation 5) is called `BBQV5HD` for activation quantizer and `BBQV5HDChan` for weight quantizer. We also modified `QuantLinear` with additional methods like `quant_act_triton()`, `float2fp4()`, `float2int4()`, and `realquant()` to support profiling inference latency with real (not fake) quantization.
2. We added `src/models/quantization/ablation*.py` to enable ablation studies.
3. We added `src/models/quantization/nf.py` to add QLoRA-style NormalFloats (NF4, NF3, and NF2).
4. We modified `src/models/base.py` to avoid applying weight decay to the scaling factor of BBQ quantizers.
5. We modified `src/optim/base.py` to log additional metrics during training, such as the scaling factors of quantizers, entropy of weights, gradient norms. While logging these metrics could potentially slow down training, when a bug occurs, these addition metrics helps identifying the root cause of the problem.
6. We modified `src/optim/utils.py` to profile inference latency  when flags `--profile_start` and `--profile_end` are configured. The code there relies on PyTorch profiler to produce events with names that match certain patterns, e.g. contains the substring `gemm`, `elementwise`, `bbq`, etc.
7. We added `i4mm/` for the CUTLASS int4 matrix multiplication kernel
8. We added `benchmark/` for the Triton FP4 quantization kernel and the Torch FP4 matrix multiplication kernel. 
9. We added `vision/` to include implementations of BBQ-Vision, DeiT, and Resnet.

# Reproducing Tables and Figures

## Perplexity and Entropy Evaluation: Reproducing Table 3, Table 4, and Figure 3
To reproduce the first four rows of Table 3, use the following 16 commands:
```bash
bash train_none.sh;
bash train_bbq.sh 4 0; 
bash train_bbq.sh 3 0; 
bash train_bbq.sh 2 -0.5; 
bash train_bbq.sh 1 -0.5;
bash train_quest.sh 4; 
bash train_quest.sh 3; 
bash train_quest.sh 2; 
bash train_quest.sh 1;
bash train_lsq.sh 4; 
bash train_lsq.sh 3; 
bash train_lsq.sh 2; 
```
To reproduce the second four rows of Table 3, please edit files `train_*.sh` by commenting the following lines

```bash
# 30M
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export LR=0.0012
export TOKENS=3000000000 # 3B
export MODEL_SIZE_PREFIX="30M"
```
and un-commenting the following lines
```bash
# # 50M
# export N_LAYER=7
# export N_EMBD=768
# export N_HEAD=6
# export LR=0.0012
# export TOKENS=5000000000 # 5B
# export MODEL_SIZE_PREFIX="50M"
```
And rerun the 16 commands above. Similarly, the other rows of Table 3 can be reproduced by changing the configuration of the model. We note that the comments above the script such as `# 30M` corresponds to non-embedding parameters, whereas in the paper we report total parameter counts. For example, 30M in the script corresponds to 95M reported in the paper, 50M in the script corresponds to 125M in the paper, 100M in the script corresponds to 200M in the paper, and 200M in the script corresponds to 300M in the paper. 

All metrics, including final weight entropy (`final-val/entw`) in Table 3, 
final perplexity in Table 3 and 4 (`final-val/perplexity`), 
weight entropy (`val/entw`) vs. training iterations (Figure 3), 
can be found on Weights and Biases after training completes.

To recreate Table 4, simply change the following line
```bash
    --model "llama" \
```
to the following line
```
    --model "base" \
```
for `train_none.sh`, `train_lsq.sh`, `train_quest.sh`, and `train_bbq.sh`. 
Next, rerun the 16 commands above.

To fully recreate Table 4, you also need to run `train_esq.sh`, `train_ltq.sh`, and `train_nf.sh` with the first command line argument as the quantization precision.
We note `train_ltq.sh` corresponds to N2UQ, `train_esq.sh` corresponds to SEQ, and `train_nf.sh` corresponds to NF.

## Kernel Latency Evaluation: Reproducing Figure 4
Please use the following commands:
```bash
cd benchmark;
python3 gemm.py;
python3 quant.py;
```
You can then find the figures in `benchmark/gemm/matmul-performance.png` and `benchmark/quant/quant-performance.png`. Please note that the original versions of these images are generated using an Nvidia RTX 5090 GPU, and therefore your profiling results may be different if your GPU is different.
In addition, PyTorch 2.8 is required to profile FP4 matmul performance in `gemm.py`.

## End-to-end Latency Evaluation: Reproducing Figure 5
To reproduce end-to-end latency measurements on real hardware, 
first make sure you have either A100 80GB or RTX 5090 as your GPU.
Then, edit both `bench_none.sh` and `bench_bqq.sh` to select the right model type based 
on whether you are recreating the left (700M on 5090) or the right sub-figure (2.4B on A100).
```bash
# # Total: 700M Nonembed: 475M
# export N_LAYER=6
# export N_EMBD=2560
# export N_HEAD=5
# export LR=0.0012
# export TOKENS=3000000000 # 3B
# export MODEL_SIZE_PREFIX="700M"

# Total: 2400M Nonembed: 1900M
export N_LAYER=6
export N_EMBD=5120
export N_HEAD=5
export LR=0.0012
export TOKENS=3000000000 # 3B
export MODEL_SIZE_PREFIX="2400M"
```
Then, use the following profiling script for fp16
```bash
bash bench_none.sh 0
```
Use the following profiling script for NF4
```bash
bash bench_none.sh 1
```
Use the following profiling script for BBQ FP4 (your GPU must be RTX 5090).
We note our profiling script only supports 3-bit quantization.
```bash
bash bench_bbq.sh 3 0 fp4
```
For BBQ INT4, your GPU must be A100 80GB. First install the i4mm plugin
```bash
cd i4mm/a100_80gb;
pip3 install -r requirements.txt;
python3 setup.py install;
python3 test.py;
```
Then use the following profiling script.
We note our profiling script only supports 3-bit quantization.
```bash
bash bench_bbq.sh 3 0 int4
```
Then check the `profile` stats on wandb for `profile/median/elementwise`, `profile/median/gemm`, `profile/median/bbq`, `profile/median/nf4`, and `profile/median/total`, and edit the corresponding values in `plot.ipynb` to recreate the figures.

## Ablation: Reproducing Table 5
Use the following script to recreate the second, third, fifth, and sixth row. 
```bash
for q in BBQV5NoHD BBQV5NoRMSHD BBQV5NoLSNoInitHD BBQV5NoLSHD; do bash ablate_bbq.sh 2 -0.5 $q; done
```
The first row and the fourth row is vanilla BBQ and QuEST and can be run using
```bash
bash train_bbq.sh 2 -0.5;
bash train_quest.sh 2 -0.5;
``` 

## Ablation of BBQ vs. BBQ-Fast: Reproducing Table 8
The first row can be recreated with the following.
```bash
bash train_bbq.sh 4 0; 
bash train_bbq.sh 3 0; 
bash train_bbq.sh 2 -0.5; 
bash train_bbq.sh 1 -0.5;
```
To recreate the second row, first edit the follow line in `train_bbq.sh`:
```bash
export A_QUANT_KWARGS="{\"precision\": $PRECISION, \"zero_point\": $2}"
```
to the following line:
```bash
export A_QUANT_KWARGS="{\"precision\": $PRECISION, \"zero_point\": $2, \"ema_rrms\": true}"
```
Then re-run the commands above. 

## Zero-shot Results: Reproducing Table 9
You must first run the pre-training experiments using `train_bbq.sh`, `train_quest.sh`, `train_lsq.sh`, and `train_none.sh`. Take note of the checkpoints under `exps/` that are generated for each run. Then edit `eval_bbq.sh`, `eval_quest.sh`, `eval_lsq.sh`, and `eval_none.sh` to change the `--resume-from` flag to point to the right checkpoint. Then run the `eval_*.sh` scripts with the same command line arguments to run evaluation. Then, look for `final-val/perplexity` on Weights and Biases.

## Vision results: Reproducing Table 10 and Figure 7
We note that we use a compute node with 4 NVIDIA RTX 2080 Ti to run all commands below. It is recommended that you use the same hardware configuration to recreate results.

To recreate Table 10, use the following commands.
```bash
cd vision;

# bbq deit_tiny
bash run_bbq.sh 4 0;
bash run_bbq.sh 3 0;
bash run_bbq.sh 2 -0.5;
bash run_bbq.sh 1 -0.5;

# quest deit_tiny
bash run_quest.sh 4;
bash run_quest.sh 3;
bash run_quest.sh 2;
bash run_quest.sh 1;

# lsq deit_tiny
bash run_lsq.sh 4;
bash run_lsq.sh 3;
bash run_lsq.sh 2;
bash run_lsq.sh 1;

# bbq deit_small
bash run_bbq_deits.sh 4 0;
bash run_bbq_deits.sh 3 0;
bash run_bbq_deits.sh 2 -0.5;
bash run_bbq_deits.sh 1 -0.5;

# quest deit_small
bash run_quest_deits.sh 4;
bash run_quest_deits.sh 3;
bash run_quest_deits.sh 2;
bash run_quest_deits.sh 1;

# lsq deit_small
bash run_lsq_deits.sh 4;
bash run_lsq_deits.sh 3;
bash run_lsq_deits.sh 2;
bash run_lsq_deits.sh 1;

# resnet-10
bash run_bbq_resnet10.sh 1 -0.5;
bash run_quest_resnet10.sh 1;

# resnet-18
bash run_bbq_resnet18.sh 1 -0.5;
bash run_quest_resnet18.sh 1;
```
Then look on Weights and Biases for `train/entw`, `train/loss`, and `val/acc1`.

To reproduce Figure 7, use the following commands:
```bash
bash train_bbq.sh 2 -0.5;
cd vision;
bash run_bbq_naive.sh 2 -0.5;
```
Take note of the path to checkpoints produced by both runs, and update `plot.ipynb` with these paths and reexcute the cells that produced Figure 7.

# Cite This Work

```bibtex
@misc{yang2026boostingentropybellbox,
      title={Boosting Entropy with Bell Box Quantization}, 
      author={Ningfeng Yang and Tor M. Aamodt},
      year={2026},
      eprint={2603.01599},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.01599}, 
}
```