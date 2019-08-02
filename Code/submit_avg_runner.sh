#!/bin/bash

# activate conda environment
eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
conda activate py27tens12


# set log files
AVG_RUNNER_LOG="avg_runner.log"

TEST_DIR="/p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel_3/Test/"
TRAIN_DIR_CLIPS="/p/lscratchh/kochansk/summer2019/jenna/rescal_gan_model/Adversarial_Video_Generation/Data/.rescal_gaussian/"

chmod u+rwx *

MODEL_NAME="test/"

# run script
python avg_runner.py -S 2 -p 100 -t $TEST_DIR -c $TRAIN_DIR_CLIPS -r 1 -n $MODEL_NAME --stats_freq 100 --summary_freq 150 --img_save_freq 500 --test_freq 1000 --model_save_freq 2000 > $AVG_RUNNER_LOG
