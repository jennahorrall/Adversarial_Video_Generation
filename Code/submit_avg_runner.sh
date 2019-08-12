#!/bin/bash

# activate conda environment
eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
conda activate py27tens12


# set log files
AVG_RUNNER_LOG="avg_runner.log"

TEST_DIR="/p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel_3/Test/"
TRAIN_DIR_CLIPS="/p/lscratchh/kochansk/summer2019/jenna/rescal_gan_model/Adversarial_Video_Generation/Data/.summaryDebug/"

chmod u+rwx *

MODEL_NAME="summary_debug/"

# run script
python avg_runner.py -S 6 -p 100 -t $TEST_DIR -c $TRAIN_DIR_CLIPS -r 1 -n $MODEL_NAME -O --stats_freq 10 --summary_freq 10 --img_save_freq 50 --test_freq 50 --model_save_freq 50 > $AVG_RUNNER_LOG
