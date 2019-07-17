#!/bin/bash


# activate conda environment
eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
conda activate py27tens12


# set log files
AVG_RUNNER_LOG="avg_runner.log"


chmod u+rwx *


# run script
python avg_runner.py -t /p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel/Test/ -r 2 -n rescal_24h_r2_7_16 -O --stats_freq 100 --summary_freq 100 --img_save_freq 500 --test_freq 2500 --model_save_freq 5000 > $AVG_RUNNER_LOG




