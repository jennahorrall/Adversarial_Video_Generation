#!/usr/bin/python

import shutil
import errno
import os, sys

#usage message
message = """
script for setting up parallel runs

	note: if you would like to process cropped clips, you will have to edit "get_full_clips" in Code/utils.py before running this script

to run: python setup_parallel_runs <NUM_DIRS> <SKIP_NUM> <INCREMENT> <DIR_NAME>

	NUM_DIRS = number of directories to create in parallel_run directory

	SKIP_NUM = amount of frames to "skip" over when processing data/training model

      		skip_num = 1: each directory will not skip over ANY frames, they will be processed in order.

      		skip_num = 2: each directory will skip process every second frame. i.e:

            		frame #1 = ALTI00000_t0.data
            		frame #2 = ALTI00002_t0.data
            		frame #3 = ALTI00004_t0.data
            		frame #4 = ALTI00006_t0.data
            		frame #5 = ALTI00008_t0.data

        think of this number as "i want to process every <SKIP_NUM> frames"

	INCREMENT = True or False

       		if increment = True, each directory will increment the by <SKIP_NUM> value.

       		i.e., if increment = True and SKIP_NUM = 2:

            	directory 1: skip_num = 2
            	directory 2: skip_num = 4
            	directory 3: skip_num = 6, etc.

       		if increment = False, each directory will have the constant <SKIP_NUM> value.

	DIR_NAME = name of root directory with parallel run directories inside.
"""


#error checking args
if len(sys.argv) != 5:
    if len(sys.argv) == 1 or len(sys.argv) == 2:
        print message
        sys.exit(0)
    else:
        print "ERROR: Please enter the correct number of arguments:\n"
        print message
        sys.exit(0)

#input values
num_dirs = int(sys.argv[1])
skip_num = int(sys.argv[2])
increment = sys.argv[3]
dir_name = sys.argv[4]

#options list
increment_options = ['True', 't', 'true', 'yes', 'y', '1']
increment_options_false = ['False', 'f', 'false', 'no', 'n', '0']

#more error checking for input values
if num_dirs < 0 or num_dirs > 50:
    print "Please enter a value for num_dirs that is greater than 0 and less than 50"
    sys.exit(0)

if skip_num < 0 or skip_num > 50:
    print "Please enter a value for skip_num that is greater than 0 and less than 50"
    sys.exit(0)

if increment not in increment_options and increment not in increment_options_false:
    print "Please enter a value for increment that is either True or False"
    sys.exit(0)


#make new directory for parallel runs
root_dir = '/p/lscratchh/kochansk/summer2019/jenna/rescal_gan_model/Adversarial_Video_Generation/'
new_dir = os.path.join(root_dir, dir_name)

if not os.path.exists(new_dir) and not os.path.isdir(new_dir):
	os.mkdir(new_dir)
print ("Directory created for parallel runs: %s\n" % new_dir)

# directory to copy code from
copy_path = '/p/lscratchh/kochansk/summer2019/jenna/rescal_gan_model/Adversarial_Video_Generation/Code'
increment_num = 0

for dir in range(num_dirs):

    dir_name = "train_rescal_" + str(dir)
    dir_path = os.path.join(new_dir, dir_name)    

    if not os.path.exists(dir_path) and not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    for dir_in_code in os.listdir(copy_path):

        if not dir_in_code.endswith('.sh'):
            copy = os.path.join(copy_path, dir_in_code)

        try:
            shutil.copytree(copy, dir_path)
        except OSError as e:
            # If the error was caused because the source wasn't a directory
            if e.errno == errno.ENOTDIR:
                shutil.copy(copy, dir_path)
    
    if increment in increment_options and dir > 0:
        increment_num += skip_num
   
    # process_data script
    process_data = dir_path + '/' + 'submit_process_data.sh'
    process_data_file = open(process_data, "wra+")
    process_data_file.write("""#!/bin/bash

# activate conda environment
eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
conda activate py27tens12

PROCESS_DATA_LOG="data.log"

TRAIN_DIR="/p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel_3/Train/"

TRAIN_DIR_CLIPS="/p/lscratchh/kochansk/summer2019/jenna/rescal_gan_model/Adversarial_Video_Generation/Data/""")

    
    process_data_file.write('.model_' + str(dir) + '_skip_' + str(skip_num + increment_num) + '/"')
    process_data_file.write('\n\nchmod u+rwx *\n') 
    process_data_file.write('python process_data.py -n 150000')
    process_data_file.write(' -s ' + str(skip_num + increment_num)) 
    process_data_file.write(' -p 100 -c $TRAIN_DIR_CLIPS -t $TRAIN_DIR > $PROCESS_DATA_LOG\n')
    process_data_file.close()
     


    # avg_runner script
    avg_runner = dir_path + '/' + 'submit_avg_runner.sh'
    avg_runner_file = open(avg_runner, "wra+")
    avg_runner_file.write("""#!/bin/bash

# activate conda environment
eval "$(/p/lscratchh/kochansk/summer2019/jenna/anaconda3/bin/conda shell.bash hook)"
conda activate py27tens12

# set log files
AVG_RUNNER_LOG="avg_runner.log"

TEST_DIR="/p/lscratchh/kochansk/summer2019/rescal-snow/test_gaussian_parallel_3/Test/"

""")

    avg_runner_file.write('TRAIN_DIR_CLIPS="/p/lscratchh/kochansk/summer2019/jenna/rescal_gan_model/Adversarial_Video_Generation/Data/')
    avg_runner_file.write('.model_' + str(dir) + '_skip_' + str(skip_num + increment_num) + '/"')
    avg_runner_file.write('\n\nchmod u+rwx *\n')

    avg_runner_file.write('MODEL_NAME="model_' + str(dir) + '_skip_' + str(skip_num + increment_num) + '/"\n\n')

    avg_runner_file.write('python avg_runner.py -S ' + str(skip_num + increment_num) + ' ') 
    avg_runner_file.write('-p 100 -t $TEST_DIR -c $TRAIN_DIR_CLIPS -r 1 -n $MODEL_NAME --stats_freq 100 --summary_freq 150 --img_save_freq 500 --test_freq 1000 --model_save_freq 2000 >> $AVG_RUNNER_LOG')




    avg_runner_file.close()

   
