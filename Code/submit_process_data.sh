#!/bin/bash

# activate conda environment
#eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
#conda activate py27tens12

PROCESS_DATA_LOG="data.log"

TRAIN_DIR="/p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel_3/Train/"

TRAIN_DIR_CLIPS="/p/lscratchh/kochansk/summer2019/jenna/rescal_gan_model/Adversarial_Video_Generation/Data/.summaryDebug/"

chmod u+rwx *

python process_data.py -n 500 -s 2 -p 100 -c $TRAIN_DIR_CLIPS -t $TRAIN_DIR > $PROCESS_DATA_LOG
