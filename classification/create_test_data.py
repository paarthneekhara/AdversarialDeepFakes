"""
Author: Shehzeen Hussain
"""

import os
import argparse
from os.path import join
import numpy
import json
import shutil

def main():
    """
    Creating Test Split for evaluating the attack
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--data_dir', '-data', type=str, 
        default="/data2/paarth/DeepFakeDataset/manipulated_sequences/") # dir containing face2face etc
    p.add_argument('--dest_dir', '-data', type=str, 
        default="/data2/paarth/DeepFakeDataset/manipulated_test_sequences/") # dir containing face2face etc
    p.add_argument('--cuda', action='store_true')
    

    args = p.parse_args()
    
    data_dir_path = args.data_dir
    dest_dir_path = args.dest_dir
    compress = args.compress
    cuda_run = args.cuda

    test_pair_list = json.loads("test_split.json")
    test_file_name_list = [ "{}_{}.mp4".format(pair[0], pair[1]) for pair in test_pair_list   ]

    fake_methods = ["Face2Face", "FaceSwap", "NeuralTextures"]
    compression_levels = ["c23", "c40", "raw"]
    for fake_method in fake_methods:
        for compression_level in compression_levels:
            
            input_folder_path = join(data_dir_path, fake_method, compression_level, "videos")
            if not os.path.isdir(input_folder_path)
                print ("Did not find input directory:", input_folder_path)
                continue
            
            # create destination directory if it does not exist
            dest_folder_path = join(dest_dir_path, fake_methods, compression_levels, "videos")
            if not os.path.isdir(dest_folder_path):
                os.makedirs(dest_folder_path)

            for input_fn in test_file_name_list:
                source_file = join(input_folder_path, input_fn)
                if os.path.exists(source_file):
                    dest_file = join(dest_folder_path, input_fn)
                    shutil.copyfile(source_file, dest_file)

if __name__ == '__main__':
    main()