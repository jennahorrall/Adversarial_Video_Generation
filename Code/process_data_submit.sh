#!/bin/bash

# Set up this run in the correct directory
#Configured for LL CPU not RC

# activate conda environment
eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
conda activate py27tens12

PROCESS_DATA_LOG="data.log"

chmod u+rwx *
python process_data.py -n 800000 -t /p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel/Train/ -T /p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel/Test/ -o >> $PROCESS_DATA_LOG


