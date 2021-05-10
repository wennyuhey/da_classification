#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-20500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
variable=${@:3}
#CUDA_VISIBLE_DEVICES=4,5,6,7  
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/classwise_train.py $CONFIG --seed 2 --deterministic --work-dir /lustre/S/wangyu/checkpoint/classification/da/visda/dist/nonorm_scost_epsilon005/nobn --launcher pytorch ${@:3}
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/classwise_train.py $CONFIG --seed 2 --deterministic --work-dir /lustre/S/wangyu/checkpoint/classification/da/visda/balance --launcher pytorch ${@:3}
