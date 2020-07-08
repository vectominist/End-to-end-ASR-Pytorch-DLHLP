#!/bin/bash
# run training process

# $1 : experiment name
# $2 : cuda id

EXP_TYPE="E2E_ASR"
CONFIG="librispeech_asr"

DIR="/data/storage/harry/"

echo "Start running training process of E2E ASR"
CUDA_VISIBLE_DEVICES=$2 python3 main.py --config config/${CONFIG}.yaml \
    --name $1 \
    --njobs 16 \
    --seed 0 \
    --logdir ${DIR}${EXP_TYPE}/log/ \
    --ckpdir ${DIR}${EXP_TYPE}/ckpt/ \
    --outdir ${DIR}${EXP_TYPE}/result/ \
    --reserve_gpu 0 \
    # --load ${DIR}${EXP_TYPE}_ASR/ckpt/$1/last_ctc_LibriSpeech.pth \
