#!/bin/bash
# run lm training process

EXP_TYPE="LM"
CONFIG="librispeech_lm"
DIR="/data/storage/harry/"

echo "Start running training process of LM"
CUDA_VISIBLE_DEVICES=$2 python3 main.py --config config/${CONFIG}.yaml \
    --name $1 \
    --njobs 8 \
    --seed 0 \
    --lm \
    --logdir ${DIR}${EXP_TYPE}/log/ \
    --ckpdir ${DIR}${EXP_TYPE}/ckpt/ \
    --outdir ${DIR}${EXP_TYPE}/result/ \
    --reserve_gpu 0 \
