import json
import os
import torch
import tqdm

import numpy as np

from audio_conditioned_unet.dataset import load_dataset
from audio_conditioned_unet.network import UNetModular


SF_PATH = 'sound_fonts/grand-piano-YDP-20160804.sf2'


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Evaluation Script for WoRMS Workshop')

    parser.add_argument('--param_path', help='path to the stored network', type=str)

    parser.add_argument('--test_set', help='path to test dataset.', type=str)
    parser.add_argument('--threshold', help='threshold for highlighting', type=float, default=0.5)
    parser.add_argument('--frame_size', help='spectrogram frame size.', type=int, default=40)
    parser.add_argument('--only_onsets', help='only learn on onset frames', default=False, action='store_true')

    args = parser.parse_args()

    param_dir = os.path.dirname(args.param_path)
    with open(os.path.join(param_dir, 'net_config.json'), 'r') as f:
        net_config = json.load(f)

    scale_factor = 3
    test_set = args.test_set
    threshold = args.threshold

    network = UNetModular(net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network.load_state_dict(torch.load(args.param_path, map_location=lambda storage, location: storage))
    print("Putting model to %s ..." % device)
    network.to(device)
    network.eval()

    recall = []
    specifity = []

    test_dataset = load_dataset(test_set, sf_path=SF_PATH, augment=False, tempo_augment=False, only_onsets=args.only_onsets,
                                frame_size=args.frame_size, scale_factor=scale_factor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16)

    tp_sum, fp_sum, tn_sum, fn_sum = 0, 0, 0, 0

    precisions = []
    for i, data in enumerate(tqdm.tqdm(test_loader)):

        inputs, targets = data['inputs'], data['targets']

        # put data to appropriate device
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            pred = network(**inputs)

        true_position = targets['true_pos'].cpu().numpy()

        gt = targets['y'].numpy()[:, 0]

        pred = torch.sigmoid(pred).cpu().numpy()[:, 0]
        pred[pred < threshold] = 0
        pred[pred >= threshold] = 1

        pred = pred.flatten()
        gt = gt.flatten()

        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        tn = np.sum((pred == 0) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))

        tp_sum = np.sum((tp, tp_sum))
        fp_sum = np.sum((fp, fp_sum))
        tn_sum = np.sum((tn, tn_sum))
        fn_sum = np.sum((fn, fn_sum))

    precision = tp_sum/(tp_sum + fp_sum)
    recall = tp_sum/(tp_sum + fn_sum)

    print('Precision:', np.round(precision, 4))
    print('Recall:', np.round(recall, 4))
    print('F1:', np.round(2*(precision*recall)/(precision+recall), 4))

