import numpy as np
import getopt
import sys
from glob import glob
import os

import constants as c
from utils import process_clip


def process_training_data(num_clips):
    """
    Processes random training clips from the full training data. Saves to TRAIN_DIR_CLIPS by
    default.

    @param num_clips: The number of clips to process. Default = 5000000 (set in __main__).

    @warning: This can take a couple of hours to complete with large numbers of clips.
    """
    num_prev_clips = len(glob(c.TRAIN_DIR_CLIPS + '*'))

    # temp array to store 100 clips in
    clips = np.empty([100, 32, 32, 15])

    for clip_num in range(num_prev_clips, num_clips + num_prev_clips):

        clip = process_clip()

        # add axis to clip in order to stack in file
        reshape = np.expand_dims(clip,axis=0)

        # add single clip to array
        clips[clip_num % 100] = reshape
       
        # save every 100 clips to a file (0-100.npz, 100-200.npz, etc.)
        if (clip_num + 1) % 100 == 0 and clip_num != 0:            
            np.savez_compressed(c.TRAIN_DIR_CLIPS + 'clips' + str((clip_num + 1) - 100) + '_to_' + str(clip_num + 1), clips)
            clips = np.empty([100, 32, 32, 15])
            print('Processed %d clips' % (clip_num + 1))
        #if (clip_num + 1) % 100 == 0: 
            #print('Processed %d clips' % (clip_num + 1))
    
def usage():
    print('Options:')
    print('-n/--num_clips= <number of clips to process for training> (Default = 10000)')
    print('-s/--skip_num=  <number of files to skip over when processing data.' (Default = 10))
    print('-t/--train_dir= <Directory of full training frames>')
    print('-c/--clips_dir= <Save directory for processed clips>')
    print("                (I suggest making this a hidden dir so the filesystem doesn't freeze")
    print("                 with so many files. DON'T `ls` THIS DIRECTORY!)")
    print('-o/--overwrite  (Overwrites the previous data in clips_dir)')
    print('-H/--help       (Prints usage)')


def main():
    ##
    # Handle command line input
    ##

    num_clips = 10000

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'n:s:p:t:c:oH',
                                ['num_clips=', '--skip_num', 'max_pile_height=', 'train_dir=', 'clips_dir=', 'overwrite', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-n', '--num_clips'):
            num_clips = int(arg)
        if opt in ('-s', '--skip_num'):
            c.SKIP_NUM = int(arg)
        if opt in ('-p', '--max_pile_height'):
            c.PILE_HEIGHT = int(arg)
        if opt in ('-t', '--train_dir'):
            c.TRAIN_DIR = c.get_dir(arg)
        if opt in ('-c', '--clips_dir'):
            c.TRAIN_DIR_CLIPS = c.get_dir(arg)
        if opt in ('-o', '--overwrite'):
            c.clear_dir(c.TRAIN_DIR_CLIPS)
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)

    assert os.path.exists(c.TRAIN_DIR)


    ##
    # Process data for training
    ##

    process_training_data(num_clips)


if __name__ == '__main__':
    main()
