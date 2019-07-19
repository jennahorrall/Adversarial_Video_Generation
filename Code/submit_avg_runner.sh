#!/bin/bash


# activate conda environment
eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
conda activate py27tens12


# set log files
AVG_RUNNER_LOG="avg_runner.log"



TRAIN_DIR="/p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel/Train/"
TEST_DIR="/p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel/Test/"

chmod u+rwx *


# run script
python avg_runner.py -t $TEST_DIR -d $TRAIN_DIR -r 1 -n test_only -O --stats_freq 100 --summary_freq 100 --img_save_freq 100 --test_freq 200 --model_save_freq 1000 > $AVG_RUNNER_LOG




