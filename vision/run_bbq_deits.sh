#!/bin/bash
#SBATCH --time 10-0
#SBATCH -p vip
#SBATCH --gres=gpu:4
#SBATCH -c 24
#SBATCH --mem=100G

port=$(($RANDOM % 10000 + 10000))
echo Using port ${port}

export W_QUANT="BBQVisionHD"
export A_QUANT="BBQVisionHD"
export QKSV_QUANT="NoQuantizer"
export W_QUANT_ARGS="{\"precision\": $1, \"zero_point\": $2}"
export A_QUANT_ARGS="{\"precision\": $1, \"zero_point\": $2}"
export QKSV_QUANT_ARGS="{}"

python3 -m torch.distributed.launch \
--master_port $port \
--nproc_per_node=4 \
--use_env main.py \
--w_quant $W_QUANT \
--a_quant $A_QUANT \
--qksv_quant $QKSV_QUANT \
--w_quant_args "$W_QUANT_ARGS" \
--a_quant_args "$A_QUANT_ARGS" \
--qksv_quant_args "$QKSV_QUANT_ARGS" \
--num_workers 3 \
--batch-size 32 \
--dtype float32 \
--model deit_small_patch16_224
