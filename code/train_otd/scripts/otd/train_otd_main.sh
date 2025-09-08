#!/bin/bash

# custom config
DATA_ROOT=$1
TRAINER=CoCoOp

DATASET=$2
CFG=$3  # config file
DIR=$4 # output_path

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA_ROOT} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${CFG}.yaml \
    --output-dir ${DIR} \

fi
