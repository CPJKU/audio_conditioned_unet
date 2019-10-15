
import cv2
import json
import os
import torch

import matplotlib.pyplot as plt
import numpy as np

from audio_conditioned_unet.network import UNetModular
from audio_conditioned_unet.utils import create_ground_truth, find_matching_patterns, load_song
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

    header_height = 50

    if args.only_onsets:
        frames = onsets+args.frame_size
    else:
        frames = range(args.frame_size, spec.shape[-1])

    n_frames = args.frame_size

    matches = find_matching_patterns(midi, max_frames=spec.shape[-1], pad=n_frames, fps=spectrogram_params['fps'])

    for frame in frames:

        spec_excerpt = spec[:, frame-args.frame_size:frame]
        perf_tensor = torch.from_numpy(spec_excerpt).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(network(score=score_tensor, perf=perf_tensor))

        true_position = np.array(interpol_fnc(frame - args.frame_size), dtype=np.int32)*scale_factor

        true_pos = (true_position/scale_factor)/[score.shape[0], score.shape[1]]

        other_matches = matches[matches[:, 0] == frame, 1] if len(matches) > 0 else []

        if len(other_matches) > 0:
            targets = np.concatenate((np.expand_dims(frame, 0), other_matches), axis=0)
        else:
            targets = np.expand_dims(frame, 0)

        targets = np.unique(targets, axis=0)
        pairs = []

        ranges = []
        for t in targets:
            start = np.array(interpol_fnc(t - n_frames*2), dtype=np.int32)
            end = np.array(interpol_fnc(t - n_frames*1), dtype=np.int32)
            pairs.append([start, end])
            ranges.append(np.array(interpol_fnc(np.arange(t - n_frames*2+1, t - n_frames*1+1)).T, dtype=np.int32))

        y_target = create_ground_truth(score, ranges)

        img_target = prepare_score_for_render(org_score_res, y_target)

        y_pred = pred.cpu().numpy()[0, 0]

        # ll[ll < threshold] = 0
        # ll[ll >= threshold] = 1

        img_pred = prepare_score_for_render(org_score_res, y_pred)
        perf_img = prepare_spec_for_render(spec_excerpt, org_score_res)


        img = np.concatenate((img_target, img_pred, perf_img), axis=1)

        # add frame on top for a header
        header = torch.ones((header_height, img.shape[1], 3))

        header[:, img_pred.shape[1] + img_target.shape[1]:, :] = 0
        img = np.concatenate((header, img), axis=0)

        img = np.array((img*255), dtype=np.uint8)

        # plot line between scores
        cv2.line(img, (img_pred.shape[1], 0),
                 (img_pred.shape[1], img_pred.shape[0] + header_height), (0, 0, 0), 3)

        # plot current note position
        cv2.line(img, (true_position[1], true_position[0] + 15 + header_height),
                 (true_position[1], true_position[0] - 15 + header_height), (255, 0, 0), 3)

        cv2.line(img, (img_target.shape[1] + true_position[1], true_position[0] + 15 + header_height),
                 (img_target.shape[1] + true_position[1], true_position[0] - 15 + header_height), (255, 0, 0), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        line_type = 2

        # write header text
        cv2.putText(img, 'Ground Truth',
                    (img_target.shape[1]//2-75, 40),
                    font,
                    font_scale,
                    (0, 0, 0),
                    line_type)

        cv2.putText(img, 'Prediction',
                    (img_target.shape[1] + img_pred.shape[1]//2-75, 40),
                    font,
                    font_scale,
                    (0, 0, 0),
                    line_type)

        if args.plot:

            cv2.putText(img, 'Frame: {}'.format(frame),
                        (1700, 40),
                        font,
                        font_scale,
                        (255, 255, 255),
                        line_type)

            cv2.imshow('Ground Truth vs Prediction', img)
            cv2.waitKey(20)

            if frame == 147:
                # jigs83

                # for Fig. 2 in the paper
                fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(15, 15))

                ax = axes[0]
                ax.set_title('Prediction')
                ax.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))

                ax = axes[1]
                ax.set_title('Ground Truth')
                ax.imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))

                # plt.subplots_adjust(hspace=0.5)
                for ax in axes.flatten():
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.figure()
                plt.imshow(np.flipud(spec_excerpt))

                plt.show()

        observation_images.append(img)

    create_video(observation_images, midi, piece_name, spectrogram_params, SF_PATH)
