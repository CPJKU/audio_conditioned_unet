
import glob
import os
import torch
import tqdm
import yaml

import numpy as np

from audio_conditioned_unet.utils import dice_loss, load_score, load_performance, center_of_mass, dummy_context
from multiprocessing import Pool
from random import shuffle
from scipy import interpolate
from torch.utils.data import Dataset

MSMD_Y_OFFSET = 20


class ScoreAudioDataset(Dataset):
    def __init__(self, scores, performances, piece_names, config, n_frames=40, augment=False, all_tempi=False):

        self.scores = scores

        self.augment = augment
        self.piece_names = piece_names
        self.performances = performances

        self.config = config
        self.pad = config['spectrogram_params']['pad']

        self.n_frames = n_frames

        self.gt_width = config['gt_width']
        self.all_tempi = all_tempi

        if self.all_tempi:
            self.all_perfs = [dict(perf=self.performances[perfs][p], score_n=perfs)
                     for perfs in self.performances for p in self.performances[perfs]]

            self.length = len(self.all_perfs)
        else:
            self.length = len(scores)

        self.score_shape = self.scores[0].shape

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        if self.all_tempi:
            perf = self.all_perfs[item]['perf']
            score_id = self.all_perfs[item]['score_n']
            score = self.scores[score_id]

        else:
            score_id = item
            score = self.scores[item]
            perfs = self.performances[item]
            perf = perfs[np.random.choice(list(perfs.keys()))]

        spec = perf['spec']
        inp = perf['interpol_fnc']
        onsets = perf['onsets']

        scores = []
        perfs = []
        ys = []
        true_positions = []

        max_y_shift = score.shape[0] - int(inp(spec.shape[-1])[0]) - MSMD_Y_OFFSET

        is_onset = []
        for i in range(self.pad, spec.shape[-1]):

            perfs.append(np.expand_dims(spec[:, i-self.n_frames+1:i+1], 0))

            true_position = np.array(inp(i - self.pad), dtype=np.int32)

            true_position, height = true_position[:-1], true_position[-1]

            y = np.zeros_like(score)

            # use adaptive height depending on staff
            y[true_position[0]-height//2:true_position[0]+height//2,
              true_position[1]-self.gt_width//2:true_position[1]+self.gt_width//2] = 1

            s = score
            if self.augment:

                yshift = np.random.randint(-9, max_y_shift)
                s = np.roll(score, yshift, 0)
                y = np.roll(y, yshift, 0)

                xshift = np.random.randint(-9, 13)
                s = np.roll(s, xshift, 1)
                y = np.roll(y, xshift, 1)

            ys.append(np.expand_dims(y, 0))
            scores.append(np.expand_dims(s, 0))
            true_positions.append(np.expand_dims(true_position, 0))

            is_onset.append((i - self.pad) in onsets)

        perfs = np.concatenate(perfs)[:, np.newaxis]
        ys = np.concatenate(ys)[:, np.newaxis]
        scores = np.concatenate(scores)[:, np.newaxis]
        true_positions = np.concatenate(true_positions)

        return {'inputs': {'perf': perfs, 'score': scores, 'length': scores.shape[0]},
                'targets': {'y': ys, 'true_positions': true_positions}, 'file_name': self.piece_names[score_id],
                'interpol_c2o': perf['interpol_c2o'], 'add_per_staff': perf['add_per_staff'], 'is_onset': is_onset}

    def get_score_shape(self):
        return self.score_shape

    def set_random_perfs(self):
        pass


class NonSequentialDatasetWrapper(Dataset):
    def __init__(self, dataset):

        self.org_dataset = dataset

        self.piece_excerpts = []
        self.tempo_aug = False
        self.all_piece_excerpts = {}

        self.unwrap_dataset()

        # will be set from outside
        self.length = 0
        self.rand_perf_indices = {}

    def unwrap_dataset(self):

        print('Unwrapping Sequential Dataset')
        for score_id in tqdm.tqdm(list(self.org_dataset.scores.keys())):

            score = self.org_dataset.scores[score_id]
            perfs = self.org_dataset.performances[score_id]

            self.all_piece_excerpts[score_id] = {}
            for perf_key in list(perfs.keys()):

                self.all_piece_excerpts[score_id][perf_key] = []

                perf = perfs[perf_key]

                spec = perf['spec']
                inp = perf['interpol_fnc']
                onsets = perf['onsets']

                max_y_shift = score.shape[0] - int(inp(spec.shape[-1])[0]) - MSMD_Y_OFFSET

                for i in range(self.org_dataset.pad, spec.shape[-1]):
                    true_position = np.array(inp(i - self.org_dataset.pad), dtype=np.int32)
                    true_position, height = true_position[:-1], true_position[-1]

                    spec_excerpt = np.expand_dims(spec[:, i-self.org_dataset.n_frames+1:i+1], 0)

                    piece_excerpt = {'score_id': score_id, 'true_position': true_position, 'height': height,
                                     'max_y_shift': max_y_shift, 'spec': spec_excerpt[:, np.newaxis],
                                     'is_onset': (i - self.org_dataset.pad) in onsets}

                    self.all_piece_excerpts[score_id][perf_key].append(piece_excerpt)

        # remove performances from original dataset as they are not used any more to free up memory
        del self.org_dataset.performances

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        piece_excerpt = self.piece_excerpts[item]
        score_id = piece_excerpt['score_id']

        true_position, height = piece_excerpt['true_position'], piece_excerpt['height']

        score = self.org_dataset.scores[score_id]

        spec = piece_excerpt['spec']

        max_y_shift = piece_excerpt['max_y_shift']

        y = np.zeros_like(score)

        # use adaptive height depending on staff
        y[true_position[0]-height//2:true_position[0]+height//2,
          true_position[1]-self.org_dataset.gt_width//2:true_position[1]+self.org_dataset.gt_width//2] = 1

        s = score
        if self.org_dataset.augment:
            yshift = np.random.randint(-9, max_y_shift)
            s = np.roll(score, yshift, 0)
            y = np.roll(y, yshift, 0)

            xshift = np.random.randint(-9, 13)
            s = np.roll(s, xshift, 1)
            y = np.roll(y, xshift, 1)


        return {'inputs': {'perf': spec, 'score': s[np.newaxis, np.newaxis, :], 'length': 1},
                'targets': {'y': y[np.newaxis, np.newaxis, :], 'true_positions': true_position},
                'file_name': self.org_dataset.piece_names[score_id], 'is_onset': [piece_excerpt['is_onset']]}

    def get_score_shape(self):
        return self.org_dataset.score_shape

    def set_random_perfs(self):
        self.rand_perf_indices = {}
        self.length = 0
        self.piece_excerpts = []

        for score_id in self.all_piece_excerpts:

            all_perf_keys = list(self.all_piece_excerpts[score_id].keys())
            self.rand_perf_indices[score_id] = np.random.choice(all_perf_keys)

            # set length
            self.length += len(self.all_piece_excerpts[score_id][self.rand_perf_indices[score_id]])

            self.piece_excerpts.extend(self.all_piece_excerpts[score_id][self.rand_perf_indices[score_id]])


def load_piece(params):

    i = params['i']
    path = params['path']
    piece_name = params['piece_name']
    spectrogram_params = params['spectrogram_params']
    scale_factor = params.get('scale_factor', 3)
    tempo_factors = params['tempo_factors']
    transpose = params.get('transpose', 0)

    org_score_res, score, coords, coord2onset = load_score(path, piece_name, scale_factor)

    performances = {}

    for tempo_factor in tempo_factors:

        spec, onsets, coords_new, interpol_fnc = load_performance(path, piece_name, spectrogram_params, coords,
                                                                  coord2onset, sf_path=params['sf_path'],
                                                                  tempo_factor=tempo_factor,
                                                                  real_perf=params['real_perf'],
                                                                  transpose=transpose)
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords = sorted(np.unique(coords_new[:, 0]))

        for c in staff_coords:

            cs_staff = coords_new[coords_new[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords)):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        interpol_c2o = interpolate.interp1d(unrolled_coords_x, onsets, kind='previous', bounds_error=False,
                                            fill_value=(onsets[0], onsets[-1]))

        performances[tempo_factor] = {'interpol_fnc': interpol_fnc,
                                      'spec': spec,
                                      'onsets': onsets,
                                      'interpol_c2o': interpol_c2o,
                                      'add_per_staff': [staff_coords, add_per_staff]
                                      }

    return i, score, piece_name, performances


def load_dataset(path, config, n_frames=40, augment=False, scale_factor=3, all_tempi=False, transpose=0, split_file=None):
    scores = {}
    piece_names = {}
    performances = {}

    params = []

    if transpose != 0:
        print(f'\033[91m Your are transposing the MIDI file by {transpose}. Are you sure you want to do this?\033[0m')

    if split_file is not None:
        with open(split_file, 'rb') as fp:
            split = yaml.load(fp, Loader=yaml.FullLoader)

        files = [os.path.join(path, 'score', f'{file}.npz') for file in split['files']]

    else:
        files = glob.glob(os.path.join(path, 'score', '*.npz'))

    for i, score_path in enumerate(tqdm.tqdm(files)):
        params.append(dict(
            i=i,
            piece_name=os.path.basename(score_path)[:-4],
            path=path,
            sf_path=config['sf_path'],
            scale_factor=scale_factor,
            spectrogram_params=config['spectrogram_params'],
            tempo_factors=config['tempo_factors'],
            real_perf=config['real_perf'],
            transpose=transpose
        ))

    print('Loading Data...')
    pool = Pool(8)
    results = list(tqdm.tqdm(pool.map(load_piece, params), total=len(params)))

    for result in results:
        i, score, piece_name, perfs = result
        scores[i] = score
        piece_names[i] = piece_name
        performances[i] = perfs

    pool.close()
    print('Done loading.')

    return ScoreAudioDataset(scores, performances, piece_names, n_frames=n_frames,
                             config=config, augment=augment, all_tempi=all_tempi)


def iterate_dataset(network, optimizer, dataset,  batch_size, seq_len, train=True, device="cpu", threshold=0.5,
                    average_stats=True, eval_center_of_mass=False, eval_only_onsets=False, clip_grads=None):

    # only necessary for the non-sequential-dataset-wrapper
    dataset.set_random_perfs()

    batch_size = min(batch_size, len(dataset))

    if train:
        network.train()
    else:
        network.eval()

    losses = []

    song_order = [i for i in range(dataset.length)]
    shuffle(song_order)

    progress_bar = tqdm.tqdm(total=len(dataset))

    current_pipeline = []
    for i in range(batch_size):
        if len(song_order) > 0:
            current_pipeline.append(dataset[song_order.pop(0)])

    indices = np.array([0 for i in range(batch_size)])
    lengths = np.array([current_pipeline[i]['inputs']['length'] for i in range(batch_size)])

    use_lstm = hasattr(network, "rnn")

    if use_lstm:
        hidden = (torch.zeros(network.rnn_layers, batch_size, network.rnn_size).to(device),
                  torch.zeros(network.rnn_layers, batch_size, network.rnn_size).to(device))
    else:
        hidden = None

    end_epoch = False

    piece_stats = {}

    while not end_epoch:

        max_seq_length = min(min(lengths - indices), seq_len)

        score_batch, perf_batch, y_batch, onsets = prepare_batch(current_pipeline, indices, max_seq_length, device)

        bs = score_batch.shape[0]*score_batch.shape[1]

        if bs > 1 or not train:

            with (dummy_context() if train else torch.no_grad()):
                model_return = network(score=score_batch, perf=perf_batch, hidden=hidden)
                pred = model_return['segmentation']
                hidden = model_return['hidden']

            loss = dice_loss(pred, y_batch, smoothing=0.)

            # perform update
            if train:
                optimizer.zero_grad()
                loss.backward()

                if clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), clip_grads)
                optimizer.step()

            piece_stats = calculate_batch_stats(pred, y_batch, piece_stats, current_pipeline, onsets,
                                                eval_center_of_mass, eval_only_onsets, threshold)

            if use_lstm:
                hidden = (hidden[0].detach(), hidden[1].detach())

            losses.append(loss.item())

        indices += max_seq_length

        pop_indices = []

        for idx, reset_state in enumerate((indices - lengths) >= 0):

            if reset_state:

                if len(song_order) > 0:
                    # new pieces can be loaded -> load new piece to the pipeline
                    current_pipeline[idx] = dataset[song_order.pop(0)]

                    # zero out hidden state
                    if use_lstm:
                        hidden[0][:, idx] = 0
                        hidden[1][:, idx] = 0

                    # reset index and lengths
                    indices[idx] = 0
                    lengths[idx] = current_pipeline[idx]['inputs']['length']

                else:
                    # no more new pieces -> remove finished piece from the pipeline
                    pop_indices.append(idx)

                progress_bar.update(1)

        for idx in sorted(pop_indices, reverse=True):
            current_pipeline.pop(idx)
            indices = np.delete(indices, idx)
            lengths = np.delete(lengths, idx)

            if use_lstm:
                h0 = torch.cat((hidden[0][:, :idx], hidden[0][:, idx + 1:]), dim=1)
                h1 = torch.cat((hidden[1][:, :idx], hidden[1][:, idx + 1:]), dim=1)
                hidden = (h0, h1)

        if len(current_pipeline) == 0:
            end_epoch = True

    stats = summarize_stats(piece_stats, average_stats, eval_center_of_mass)
    stats['loss'] = np.mean(losses)

    progress_bar.close()
    return stats


def prepare_batch(current_pipeline, indices, max_seq_length, device):

    score_batch = []
    perf_batch = []
    y_batch = []
    onsets = []

    for idx, data in enumerate(current_pipeline):
        inputs, targets = data['inputs'], data['targets']
        scores = inputs['score']
        perf = inputs['perf']
        y = targets['y']

        start_index = indices[idx]

        score_tmp = scores[start_index:start_index + max_seq_length][:, None]
        perf_tmp = perf[start_index:start_index + max_seq_length][:, None]
        y_tmp = y[start_index:start_index + max_seq_length][:, None]

        score_batch.append(score_tmp)
        perf_batch.append(perf_tmp)
        y_batch.append(y_tmp)

        if 'is_onset' in data:
            onsets.append(data['is_onset'][start_index:start_index + max_seq_length])

    score_batch = torch.from_numpy(np.concatenate(score_batch, axis=1)).to(device)
    perf_batch = torch.from_numpy(np.concatenate(perf_batch, axis=1)).to(device)
    y_batch = torch.from_numpy(np.concatenate(y_batch, axis=1)).to(device)

    return score_batch, perf_batch, y_batch, onsets


def calculate_batch_stats(pred, y_batch, piece_stats, current_pipeline, onsets, eval_center_of_mass,
                          eval_only_onsets, threshold):

    pred = pred.detach()
    gt = y_batch
    pred = pred.view(y_batch.shape[0], y_batch.shape[1], *y_batch.shape[2:])

    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    # seq_len, bs, c, h, w
    tp = ((pred == 1) & (gt == 1)).sum((2, 3, 4))
    fp = ((pred == 1) & (gt == 0)).sum((2, 3, 4))
    tn = ((pred == 0) & (gt == 0)).sum((2, 3, 4))
    fn = ((pred == 0) & (gt == 1)).sum((2, 3, 4))

    for num, piece in enumerate(current_pipeline):

        for tstep in range(y_batch.shape[0]):

            is_onset = onsets[num][tstep]

            if not eval_only_onsets or is_onset:

                fname = piece['file_name']
                if fname not in piece_stats:
                    piece_stats[fname] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

                    if eval_center_of_mass:
                        piece_stats[fname]['com_diff'] = []
                        piece_stats[fname]['frame_diff'] = []

                piece_stats[fname]['tp'] += tp[tstep][num].item()
                piece_stats[fname]['fp'] += fp[tstep][num].item()
                piece_stats[fname]['tn'] += tn[tstep][num].item()
                piece_stats[fname]['fn'] += fn[tstep][num].item()

                if eval_center_of_mass:

                    com_gt = center_of_mass(y_batch[tstep, num, 0])

                    # calculate center of mass for prediction and ground truth
                    if pred[tstep, num, 0].sum() == 0:
                        com_pred = torch.zeros_like(com_gt)
                    else:
                        com_pred = center_of_mass(pred[tstep, num, 0])

                    # compute the euclidean distance between centers
                    com_diff = (com_pred - com_gt).pow(2).sum().sqrt()

                    piece_stats[fname]['com_diff'].append(com_diff.cpu().detach().item())

                    com_gt = com_gt.cpu().numpy()
                    com_pred = com_pred.cpu().numpy()

                    staff_coords, add_per_staff = current_pipeline[num]['add_per_staff']

                    # map prediction and ground truth to closest staff y-coordinate
                    staff_id_pred = np.argwhere(
                        min(staff_coords, key=lambda x: abs(x - com_pred[0])) == staff_coords).item()
                    staff_id_gt = np.argwhere(min(staff_coords, key=lambda x: abs(x - com_gt[0])) == staff_coords).item()

                    # unroll x coord
                    x_coord_gt = com_gt[1] + add_per_staff[staff_id_gt]
                    x_coord_pred = com_pred[1] + add_per_staff[staff_id_pred]

                    # calculate difference of onset frames
                    frame_diff = abs(current_pipeline[num]['interpol_c2o'](x_coord_pred) -
                                     current_pipeline[num]['interpol_c2o'](x_coord_gt))

                    piece_stats[fname]['frame_diff'].append(frame_diff)

    return piece_stats


def summarize_stats(piece_stats, average_stats, eval_center_of_mass):
    stats = {}
    precision = {}
    recall = {}

    for key in piece_stats:
        stat = piece_stats[key]
        precision[key] = (stat['tp'] / (stat['tp'] + stat['fp']) if stat['tp'] > 0 else 0)
        recall[key] = (stat['tp'] / (stat['tp'] + stat['fn']) if stat['tp'] > 0 else 0)

    if average_stats:
        precision = np.array(list(precision.values())).mean()
        recall = np.array(list(recall.values())).mean()

    stats['precision'] = precision
    stats['recall'] = recall

    if eval_center_of_mass:
        com_diffs = {}
        frame_diffs = {'onset_diffs': []}
        for key in piece_stats:
            stat = piece_stats[key]
            com_diffs[key] = np.mean(stat['com_diff'])
            frame_diffs[key] = {'mean': np.mean(stat['frame_diff']), 'diffs': stat['frame_diff']}
            frame_diffs['onset_diffs'].extend(stat['frame_diff'])

        if average_stats:
            com_diffs = np.array(list(com_diffs.values())).mean()
            frame_diffs = np.array(list(frame_diffs['onset_diffs'])).mean()

        stats['center_of_mass_differences'] = com_diffs
        stats['frame_differences'] = frame_diffs

    return stats

