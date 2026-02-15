#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set common environment variables
export VOCAB_SIZE=32000 # 50304
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512
export DATASET="dummy" # "slimpajama"

# # 30M
# export N_LAYER=6
# export N_EMBD=2560
# export N_HEAD=5
# export LR=0.0012
# export TOKENS=3000000000 # 3B
# export MODEL_SIZE_PREFIX="30M"

# # 50M
# export N_LAYER=7
# export N_EMBD=768
# export N_HEAD=6
# export LR=0.0012
# export TOKENS=5000000000 # 5B
# export MODEL_SIZE_PREFIX="50M"

# # 100M
# export N_LAYER=8
# export N_EMBD=1024
# export N_HEAD=8
# export LR=0.0006
# export TOKENS=10000000000 # 10B
# export MODEL_SIZE_PREFIX="100M"

# # 200M
# export N_LAYER=10
# export N_EMBD=1280
# export N_HEAD=10
# export LR=0.0003
# export TOKENS=20000000000 # 20B
# export MODEL_SIZE_PREFIX="200M"

# # 430M
# export N_LAYER=13
# export N_EMBD=1664
# export N_HEAD=13
# export LR=0.00015
# export TOKENS=43000000000 # 43B
# export MODEL_SIZE_PREFIX="430M"

# # 800M
# export N_LAYER=16
# export N_EMBD=2048
# export N_HEAD=16
# export LR=0.000075
# export TOKENS=80000000000 # 80B
# export MODEL_SIZE_PREFIX="800M"

# # 1700M
# export N_LAYER=20
# export N_EMBD=2688
# export N_HEAD=21
# export LR=0.0000375
# export TOKENS=10750000000 # 10.75B
# export MODEL_SIZE_PREFIX="3200M"

# # 3200M
# export N_LAYER=28
# export N_EMBD=3072
# export N_HEAD=24
# export LR=0.000075
# export TOKENS=20000000000 # 20B
# export MODEL_SIZE_PREFIX="3200M"

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

# Quantization configuration
export PRECISION=$1
# export W_QUANT="BBQV5HDChan"
# export A_QUANT="BBQV5HD"
# export W_QUANT_KWARGS="{\"precision\": $PRECISION, \"zero_point\": $2, \"channels\": 0}"
# export A_QUANT_KWARGS="{\"precision\": $PRECISION, \"zero_point\": $2, \"ema_rrms\": true}"
export W_QUANT="NoQuantizer"
export A_QUANT="NoQuantizer"
export W_QUANT_KWARGS="{}"
export A_QUANT_KWARGS="{}"

# Calculate the number of iterations based on tokens and batch settings
export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

WANDB_PREFIX="BENCH${1}-LLAMA-${MODEL_SIZE_PREFIX}-${W_QUANT}@${PRECISION}@${2}:${A_QUANT}@${PRECISION}@${2}-${DATASET}"

NUM_GPUS=1

torchrun --nproc_per_node=${NUM_GPUS} ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --model "llama" \
    --compile \
    --latest-ckpt-interval 0 \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "quest" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --eval_uncompiled 1 \
    --realquant ${1} \
    --profile_start 100 \
    --profile_end 500 \
    --full-eval-at 0 \
    --opt none \
    --scheduler none \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --a-quant-kwargs "${A_QUANT_KWARGS}"
