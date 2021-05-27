#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-20500}
DIR=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
variable=${@:4}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/classwise_train.py $CONFIG --resume-from /lustre/S/wangyu/da_log/visda/dist/nobn_nonorm_label_eps005_barlow/epoch_2.pth --seed 2 --work-dir $DIR --launcher pytorch ${@:4}
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/classwise_train.py $CONFIG --seed 2 --deterministic --work-dir /lustre/S/wangyu/checkpoint/classification/da/visda/balance --launcher pytorch ${@:3}
