
import cv2
import json
import os
import torch

import numpy as np

from audio_conditioned_unet.network import UNetModular
from audio_conditioned_unet.utils import load_song
from audio_conditioned_unet.video_utils import create_video, prepare_score_for_render, prepare_spec_for_render


SF_PATH = 'sound_fonts/grand-piano-YDP-20160804.sf2'

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Test Script for WoRMS Workshop')

    parser.add_argument('--param_path', help='path to the stored network', type=str)
    parser.add_argument('--test_piece', help='path to test piece (do not specify extension).', type=str)
    parser.add_argument('--plot', help='intermediate plotting', default=False, action='store_true')
    parser.add_argument('--frame_size', help='spectrogram frame size.', type=int, default=40)
    parser.add_argument('--only_onsets', help='only learn on onset frames', default=False, action='store_true')

    args = parser.parse_args()

    param_dir = os.path.dirname(args.param_path)
    with open(os.path.join(param_dir, 'net_config.json'), 'r') as f:
        net_config = json.load(f)

    scale_factor = 3
    piece_name = os.path.basename(args.test_piece)
    spectrogram_params = dict(sample_rate=22050, frame_size=2048, fps=20, pad=args.frame_size)
    org_score_res, score, midi, spec, interpol_fnc, onsets = load_song(args.test_piece, spectrogram_params, SF_PATH, scale_factor=scale_factor)

    network = UNetModular(net_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.load_state_dict(torch.load(args.param_path, map_location=lambda storage, location: storage))
    print("Putting model to %s ..." % device)
    network.to(device)

    network.eval()

    score_tensor = torch.from_numpy(score).unsqueeze(0).unsqueeze(0).to(device)

    observation_images = []

    threshold = 0.5

    if args.only_onsets:
        frames = onsets+args.frame_size
    else:
        frames = range(args.frame_size, spec.shape[-1])

    for frame in frames:

        spec_excerpt = spec[:, frame-args.frame_size:frame]
        perf_tensor = torch.from_numpy(spec_excerpt).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(network(score=score_tensor, perf=perf_tensor))

        true_position = np.array(interpol_fnc(frame - args.frame_size), dtype=np.int32)*scale_factor

        y_pred = pred.cpu().numpy()[0, 0]

        y_pred[y_pred < threshold] = 0
        y_pred[y_pred >= threshold] = 1

        img = prepare_score_for_render(org_score_res, y_pred)
        perf_img = prepare_spec_for_render(spec_excerpt, org_score_res)

        img = np.concatenate((img, perf_img), axis=1)

        img = np.array((img*255), dtype=np.uint8)
        cv2.line(img, (true_position[1], true_position[0] + 15),
                 (true_position[1], true_position[0] - 15), (255, 0, 0), 3)

        if args.plot:
            cv2.imshow('Prediction', img)
            cv2.waitKey(20)

        observation_images.append(img)

    create_video(observation_images, midi, piece_name, spectrogram_params, SF_PATH)
