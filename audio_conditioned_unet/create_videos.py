
import glob

import os
from subprocess import call

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Test Script for WoRMS Workshop')

    parser.add_argument('--param_path', help='path to the stored network', type=str)
    parser.add_argument('--frame_size', help='spectrogram frame size.', type=int, default=40)
    parser.add_argument('--test_path', help='path to test set.', type=str)

    args = parser.parse_args()

    # sort according to file size
    pairs = []
    for file in glob.glob(os.path.join(args.test_path, '*.mid')):
        # Use join to get full file path.

        # Get size and add to list of tuples.
        size = os.path.getsize(file)
        pairs.append((size, file))

    # Sort list of tuples by the first element, size.
    pairs.sort(key=lambda s: s[0])


    for pair in pairs[:]:

        # get piece path without file extension
        f = os.path.splitext(pair[1])[0]
        exit_code = call(["python", "compare_to_optimal.py",
                          "--param_path", args.param_path,
                          "--test_piece", f,
                          "--frame_size", str(args.frame_size)])
