
import cv2
import json
import os
import torch
import tqdm

import numpy as np

from audio_conditioned_unet.network import ConditionalUNet
from audio_conditioned_unet.utils import load_song, load_game_config
from audio_conditioned_unet.video_utils import create_video, prepare_score_for_render, prepare_spec_for_render


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Test Script for ISMIR 2020')
    parser.add_argument('--param_path', help='path to the stored network', type=str)
    parser.add_argument('--config', help='path to config.', type=str)
    parser.add_argument('--test_dir', help='path to test directory).', type=str)
    parser.add_argument('--test_piece', help='name of test piece (do not specify extension).', type=str)
    parser.add_argument('--plot', help='intermediate plotting', default=False, action='store_true')
    parser.add_argument('--scale_factor', help='change score scale factor', type=int, default=3)
    parser.add_argument('--tempo_factor', help='change tempo factor', type=int, default=1000)
    args = parser.parse_args()

    param_dir = os.path.dirname(args.param_path)
    with open(os.path.join(param_dir, 'net_config.json'), 'r') as f:
        net_config = json.load(f)

    config = load_game_config(args.config)

    tempo_factor = args.tempo_factor
    scale_factor = args.scale_factor
    piece_name = args.test_piece
    spectrogram_params = config['spectrogram_params']

    org_score_res, score, spec, _, _ = load_song(args.test_dir, piece_name, spectrogram_params, config['sf_path'],
                                                 scale_factor=scale_factor, tempo_factor=tempo_factor,
                                                 real_perf=config['real_perf'])
    network = ConditionalUNet(net_config)
    n_frames = network.perf_encoder.n_input_frames

    if n_frames == 1:
        vis_spec = np.zeros((spec.shape[0], spectrogram_params['pad']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.load_state_dict(torch.load(args.param_path, map_location=lambda storage, location: storage))
    print("Putting model to %s ..." % device)
    network.to(device)
    network.eval()

    score_tensor = torch.from_numpy(score).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

    if hasattr(network, 'rnn'):
        hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size).to(device),
                  torch.zeros(network.rnn_layers, 1, network.rnn_size).to(device))
    else:
        hidden = None

    frames = range(spectrogram_params['pad'], spec.shape[-1])
    observation_images = []
    for frame in tqdm.tqdm(frames):
        spec_excerpt = spec[:, frame - n_frames + 1:frame + 1]

        if n_frames == 1:
            vis_spec = np.roll(vis_spec, -1, axis=1)
            vis_spec[:, -1] = spec_excerpt[:, 0]
        else:
            vis_spec = spec_excerpt

        perf_tensor = torch.from_numpy(spec_excerpt).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            model_return = network(score=score_tensor, hidden=hidden, perf=perf_tensor)
            pred = model_return['segmentation']
            hidden = model_return['hidden']

        y_pred = pred.cpu().numpy()[0, 0]
        img_pred, mask_pred = prepare_score_for_render(org_score_res, y_pred)
        perf_img = prepare_spec_for_render(vis_spec, org_score_res)

        img = np.concatenate((img_pred, perf_img), axis=1)
        img = np.array((img*255), dtype=np.uint8)

        if args.plot:
            cv2.imshow('Prediction', img)
            cv2.waitKey(20)

        observation_images.append(img)

    if config['real_perf'] and str(tempo_factor) != "-1":

        wav_path = os.path.join(args.test_dir, 'performance', piece_name + f'_{tempo_factor}.wav')
        create_video(observation_images, wav_path, piece_name, spectrogram_params, config['sf_path'],
                     path="../videos", real_perf=True)
    else:

        if str(tempo_factor) == "-1":
            midi_path = os.path.join(args.test_dir, 'performance', piece_name + '.mid')
            create_video(observation_images, midi_path, piece_name, spectrogram_params,
                         config['sf_path'], path="../videos")
        else:
            midi_path = os.path.join(args.test_dir, 'performance', piece_name + f'_tempo_{tempo_factor}.mid')
            create_video(observation_images, midi_path, piece_name + f'_tempo_{tempo_factor}', spectrogram_params,
                         config['sf_path'], path="../videos")
