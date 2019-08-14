#!/bin/bash


#      **  PARALLEL RUN SCRIPT FOR TRAINING MULTIPLE GAN MODELS  **
#
#      run submit_process_data.sh to do data pre-processing in each directory
#      run submit_avg_runner.sh to train model in each directory
#
#
#
#



output_root="../train_model_full_clips"

echo "Looking for jobs to run under ${output_root}"

output_dirs=(${output_root}/*)

output_dir="${output_dirs[${PMI_RANK}]}"
cd ${output_dir}
echo "Process ${PMI_RANK} running in  ${output_dir}"

# Need to be able to read and execute this directory
chmod u+rwx *

# train model in each directory
./submit_avg_runner.sh
