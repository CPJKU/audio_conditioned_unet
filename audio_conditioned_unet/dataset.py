import glob
import cv2
import os
import torch
import tqdm


import numpy as np
import pretty_midi as pm

from multiprocessing import Pool
from scipy import interpolate
from torch.utils.data import Dataset
from audio_conditioned_unet.utils import create_ground_truth, dice_loss, find_matching_patterns,\
    merge_onsets, midi_to_spec_otf


class ScoreAudioDataset(Dataset):
    def __init__(self, scores, specs, data, augment=False, frame_size=40):

        self.scores = scores
        self.specs = specs
        self.data = data
        self.augment = augment
        self.length = len(data)
        self.frame_size = frame_size

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        data = self.data[item]

        spec = self.specs[data['perf']][data['tempo']]

        score = self.scores[data['score']]
        perf = spec[:, data['frame'] - self.frame_size:data['frame']]

        targets = np.unique(data['targets'], axis=0)

        y = create_ground_truth(score, targets)

        if self.augment:
            # perform y shift
            yshift = np.random.randint(0, score.shape[0])
            score = np.roll(score, yshift, 0)
            y = np.roll(y, yshift, 0)

            # perform x shift
            xshift = np.random.randint(-9, 13)
            score = np.roll(score, xshift, 1)
            y = np.roll(y, xshift, 1)

        return {'inputs': {'perf': np.expand_dims(perf, 0), 'score': np.expand_dims(score, 0)},
                'targets': {'y': np.expand_dims(y, 0), 'true_pos': np.array(data['true_pos'])}}


def iterate_dataset(network, optimizer, data_loader, epoch, num_epochs, train=True, device="cpu",
                    calc_stats=True, threshold=0.5):

    if train:
        prefix = "Tr"
        network.train()
    else:
        prefix = "Val"
        network.eval()

    tp_sum, fp_sum, tn_sum, fn_sum = 0, 0, 0, 0

    losses = []

    for i, data in enumerate(tqdm.tqdm(data_loader, desc='{}-Epoch %d/%d'.format(prefix) % (epoch, num_epochs))):

        inputs, targets = data['inputs'], data['targets']

        # put data to appropriate device
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        for key in targets.keys():
            targets[key] = targets[key].to(device)

        if train:
            pred = network(**inputs)
        else:
            with torch.no_grad():
                pred = network(**inputs)

        loss = dice_loss(torch.sigmoid(pred), targets['y'], smoothing=0.)

        if calc_stats:
            gt = targets['y'].cpu().numpy()[:, 0]

            pred = torch.sigmoid(pred).detach().cpu().numpy()[:, 0]
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

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    if calc_stats:
        precision = tp_sum/(tp_sum + fp_sum)
        recall = tp_sum/(tp_sum + fn_sum)
    else:
        precision = 0
        recall = 0

    return np.mean(losses), (precision, recall)


def load_dataset(path, sf_path, augment=False, tempo_augment=False, frame_size=40, only_onsets=False, scale_factor=3):

    scores = {}
    specs = {}
    data = []

    params = []

    if tempo_augment:
        tempi = [0.8, 0.9, 1.0, 1.1, 1.2]
    else:
        tempi = [1.0]

    for i, path in enumerate(tqdm.tqdm(glob.glob(os.path.join(path, '*.npz')))):

        params.append(dict(
            i=i,
            path=path,
            sf_path=sf_path,
            tempi=tempi,
            pad=frame_size,
            only_onsets=only_onsets,
            scale_factor=scale_factor
        ))

    pool = Pool(8)

    results = list(tqdm.tqdm(pool.imap_unordered(load_piece, params), total=len(params)))

    for result in results:
        i, score, d, spec = result
        specs[i] = spec
        scores[i] = score
        data.extend(d)

    pool.close()

    return ScoreAudioDataset(scores, specs, data, augment=augment, frame_size=frame_size)


def load_piece(params):

    i = params['i']
    path = params['path']
    tempi = params.get('tempi', [1.])
    scale_factor = params.get('scale_factor', 3)

    only_onsets = params.get('only_onsets', False)

    midi_path = path.replace('.npz', '.mid')

    npzfile = np.load(path, allow_pickle=True)

    score, coords, coord2onset = npzfile["sheet"], npzfile["coords"], npzfile['coord2onset']

    score = 1 - np.array(score, dtype=np.float32) / 255.
    score = cv2.resize(score, (int(score.shape[1] // scale_factor), int(score.shape[0] // scale_factor)),
                       interpolation=cv2.INTER_AREA)

    coords /= scale_factor

    spectrogram_params = dict(sample_rate=22050, frame_size=2048, fps=20)

    data = []
    specs = {}
    for tempo in tempi:

        midi = pm.PrettyMIDI(midi_path)

        if tempo != 1:
            for instrument in midi.instruments:
                for note in instrument.notes:
                    note.start = note.start * tempo
                    note.end = note.end * tempo

        spec = midi_to_spec_otf(midi, spectrogram_params, sound_font_path=params['sf_path'])

        spec = np.pad(spec, ((0, 0), (params['pad'], 0)), mode='constant')

        specs[tempo] = spec

        onsets = (midi.get_onsets() * spectrogram_params['fps']).astype(int)
        onsets, coords = merge_onsets(onsets, coords, coord2onset[0])

        interpol_fnc = interpolate.interp1d(onsets, coords.T, kind='previous', bounds_error=False,
                                            fill_value=(coords[0, :], coords[-1, :]))

        matches = find_matching_patterns(midi, max_frames=spec.shape[-1], pad=params['pad'], fps=spectrogram_params['fps'])
        padded_onsets = onsets + params['pad']

        frames = range(params['pad'], spec.shape[-1])
        if only_onsets:
            frames = padded_onsets

        for frame in frames:
            true_position = np.array(interpol_fnc(frame - params['pad']), dtype=np.int32)

            other_matches = matches[matches[:, 0] == frame, 1] if len(matches) > 0 else np.array([])

            if len(other_matches) > 0:
                targets = np.concatenate((np.expand_dims(frame, 0), other_matches), axis=0)
            else:
                targets = np.expand_dims(frame, 0)

            targets = np.unique(targets, axis=0)
            pair_targets = []

            for t in targets:

                pair_targets.append(
                    np.array(interpol_fnc(np.arange(t - params['pad'] * 2 + 1, t - params['pad'] * 1 + 1)).T, dtype=np.int32))

            data.append(dict(perf=i, y=true_position/[score.shape[0], score.shape[1]], score=i, frame=frame,
                             targets=pair_targets, true_pos=true_position, tempo=tempo))

    return i, score, data, specs





