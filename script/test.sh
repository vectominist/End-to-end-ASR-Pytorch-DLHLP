#!/bin/bash
# run testing process

# $1 : Experiment name
# $2 : Cuda id
# $3 : Test type (w-greedy, n-greedy, etc)
# $4 : Config name
# $5 : Experiment type

EXP_TYPE=$5
CONFIG=$4
DIR="/data/storage/harry/"

echo "Start running testing process of E2E ASR"
CUDA_VISIBLE_DEVICES=$2 python3 main.py --config config/${CONFIG}.yaml \
    --name $1 \
	--test \
    --njobs 16 \
    --seed 0 \
    --ckpdir ${DIR}${EXP_TYPE}_ASR/ckpt/$1 \
	--outdir ${DIR}${EXP_TYPE}_ASR/test_result/$1

dev_csv=${DIR}${EXP_TYPE}_ASR/test_result/$1/dev_out/$1_dev_output-$3.csv
test_csv=${DIR}${EXP_TYPE}_ASR/test_result/$1/test_out/$1_test_output-$3.csv

mv ${DIR}${EXP_TYPE}_ASR/test_result/$1/$1_dev_output.csv $dev_csv
mv ${DIR}${EXP_TYPE}_ASR/test_result/$1/$1_test_output.csv $test_csv

# Eval
python3 eval.py --file $dev_csv
python3 eval.py --file $test_csv
