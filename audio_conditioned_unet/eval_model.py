 
import argparse
import json
import os
import torch

import numpy as np
import multiprocessing as mp

from audio_conditioned_unet.network import ConditionalUNet
from audio_conditioned_unet.dataset import load_dataset, iterate_dataset
from audio_conditioned_unet.utils import load_game_config

# assuming 72 dpi
PXL2CM = 0.035277778

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation Script for ISMIR 2020')
    parser.add_argument('--param_path', help='path to the stored network', type=str)
    parser.add_argument('--test_dir', help='path to test dataset.', type=str)
    parser.add_argument('--config', help='path to config.', type=str)
    parser.add_argument('--scale_factor', help='sheet image scale factor.', type=int, default=3)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    parser.add_argument('--seq_len', help='batch size', type=int, default=128)
    parser.add_argument('--eval_onsets', help='evaluate only onset frames', default=False, action='store_true')
    parser.add_argument('--piecewise_stats', help='print stats for each piece', default=False, action='store_true')
    parser.add_argument('--split_file', help='split file to only evaluate a subset from the test dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # apparently the code gets stuck without this line when computing the spectrograms
    mp.set_start_method('spawn', force=True)

    # load network
    param_dir = os.path.dirname(args.param_path)
    with open(os.path.join(param_dir, 'net_config.json'), 'r') as f:
        net_config = json.load(f)

    config = load_game_config(args.config)
    network = ConditionalUNet(net_config)
    network.load_state_dict(torch.load(args.param_path, map_location=lambda storage, location: storage))
    print("Putting model to %s ..." % device)
    network.to(device)
    network.eval()

    # load test dataset
    n_frames = network.perf_encoder.n_input_frames
    dataset = load_dataset(args.test_dir, config, n_frames=n_frames, split_file=args.split_file,
                           scale_factor=args.scale_factor)

    # compute statistics for the test dataset
    stats = iterate_dataset(network, None, dataset, batch_size=args.batch_size, seq_len=args.seq_len, device=device,
                            train=False, average_stats=False, eval_center_of_mass=True,
                            eval_only_onsets=args.eval_onsets)

    if args.eval_onsets:
        frame_diffs = stats['frame_differences']
        thresholds = [0.05, 0.1, 0.5, 1.0, 5.0]

        time_stats = {}
        time_stats_mean = {}
        for key in frame_diffs:
            if key != 'onset_diffs':
                diffs = np.array(frame_diffs[key]['diffs'])
                diffs = diffs / config['spectrogram_params']['fps']
                total = len(diffs)

                cummulative_percentage = []
                for th in thresholds:
                    cummulative_percentage.append(np.round(100 * np.sum(diffs <= th) / total, 1))

                time_stats[key] = cummulative_percentage

                time_stats_mean[key] = np.mean(diffs)

        if args.piecewise_stats:
            sorted_time_stats = sorted(time_stats_mean.items(), key=lambda kv: kv[1], reverse=False)
            print(f'Thresholds <= {thresholds}')
            for key, _ in sorted_time_stats:
                print(f'{key}: {time_stats[key]}')

        # frames to seconds
        onset_diffs = np.asarray(frame_diffs['onset_diffs']) / config['spectrogram_params']['fps']
        total_onsets = len(onset_diffs)

        for th in thresholds:
            print(f'<= {th}: {np.round(100 * np.sum(onset_diffs <= th) / total_onsets, 1)}')

    else:
        # evaluate pixelwise measures and score distances
        _, precision, recall = stats['loss'], stats['precision'], stats['recall']

        f1 = {}
        for key in precision.keys():
            prec = precision[key]
            rec = recall[key]
            f1[key] = 0 if prec + rec == 0 else 2 * (prec * rec) / (prec + rec)

        if args.piecewise_stats:
            f1_sorted = sorted(f1.items(), key=lambda kv: kv[1], reverse=True)
            for name, fscore in f1_sorted:
                print('{} Precision: {}, Recalls: {}, F1: {}'.format(name, np.round(precision[name], 4),
                                                                     np.round(recall[name], 4), np.round(fscore, 4)))

        precision = np.mean(list(precision.values()))
        recall = np.mean(list(recall.values()))
        f1 = np.mean(list(f1.values()))

        print('Precision:', np.round(precision, 3))
        print('Recall:', np.round(recall, 3))
        print('F1:', np.round(f1, 3))
        print()

        com_diffs = stats['center_of_mass_differences']
        com_diff_pieces = sorted(com_diffs.items(), key=lambda kv: kv[1], reverse=False)

        if args.piecewise_stats:
            for name, com_diff in com_diff_pieces:
                com = com_diff * args.scale_factor * PXL2CM
                print(f'{name} Alignment Error [cm]: {np.round(com, 2)}')

        com_diff_mean = np.mean([entry[1] for entry in com_diff_pieces])
        com_diff_median = np.median([entry[1] for entry in com_diff_pieces])

        print('Mean Alignment Error [cm]:', np.round(com_diff_mean * args.scale_factor * PXL2CM, 2))
        print('Median Alignment Error [cm]:', np.round(com_diff_median * args.scale_factor * PXL2CM, 2))
        print()









