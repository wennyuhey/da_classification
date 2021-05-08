#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
variable=${@:3}
#CUDA_VISIBLE_DEVICES=4,5,6,7 
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/classwise_train.py $CONFIG --work-dir /lustre/S/wangyu/checkpoint/classification/visda/oracle --launcher pytorch ${@:3}
