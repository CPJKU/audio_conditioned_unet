
import copy
import cv2
import os
import time
import tempfile
import torch
import yaml

import numpy as np

from madmom.io import midi as mm_midi
from scipy import interpolate

EPS = 1e-8


def dice_loss(inputs, targets, smoothing=1.):

    iflat = inputs.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smoothing) / ((iflat**2).sum() + (tflat**2).sum() + smoothing + EPS))


def merge_onsets(cur_onsets, stk_note_coords, coords2onsets):
    """ merge onsets occurring in the same frame """

    # get coordinate keys
    coord_ids = coords2onsets.keys()

    # init list of unique onsets and coordinates
    onsets, coords = [], []

    # iterate coordinates
    for i in coord_ids:
        # check if onset already exists in list
        if cur_onsets[coords2onsets[i]] not in onsets:
            coords.append(stk_note_coords[i])
            onsets.append(cur_onsets[coords2onsets[i]])

    # convert to arrays
    coords = np.asarray(coords, dtype=np.float32)
    onsets = np.asarray(onsets, dtype=np.int)

    return onsets, coords


def render_audio(midi_file_path, sound_font):
    """
    Render midi to audio
    """

    # split file name and extention
    name, extention = midi_file_path.rsplit(".", 1)

    # set file names
    audio_file = name + ".wav"

    # synthesize midi file to audio
    cmd = "fluidsynth -F %s -O s16 -T wav %s %s 1> /dev/null" % (audio_file, sound_font, midi_file_path)

    os.system(cmd)
    return audio_file


def midi_to_spec_otf(midi, spec_params: dict, sound_font_path=None) -> np.ndarray:
    """MIDI to Spectrogram (on the fly)

       Synthesizes a MIDI with fluidsynth and extracts a spectrogram.
       The spectrogram is directly returned
    """
    processor = spectrogram_processor(spec_params)

    mid_path = os.path.join(tempfile.gettempdir(), str(time.time())+'.mid')

    with open(mid_path, 'wb') as f:
        midi.save(f)
        # midi.write(mid_path)

    audio_path = render_audio(mid_path, sound_font=sound_font_path)

    # compute spectrogram
    spec = processor.process(audio_path).T

    os.remove(mid_path)
    os.remove(audio_path)

    return spec


def wav_to_spec_otf(wav_path: str, spec_params: dict) -> np.ndarray:
    processor = spectrogram_processor(spec_params)

    # compute spectrogram
    spec = processor.process(wav_path).T

    return spec


def spectrogram_processor(spec_params):
    from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, \
        LogarithmicFilterbank
    from madmom.processors import SequentialProcessor

    """Helper function for our spectrogram extraction."""
    sig_proc = SignalProcessor(num_channels=1, sample_rate=spec_params['sample_rate'])
    fsig_proc = FramedSignalProcessor(frame_size=spec_params['frame_size'], fps=spec_params['fps'])

    spec_proc = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=60, fmax=6000,
                                             norm_filters=True, unique_filters=False)

    log_proc = LogarithmicSpectrogramProcessor()

    processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc, log_proc])

    return processor


def load_song(dir, piece, spectrogram_params, sf_path, tempo_factor=1., scale_factor=3, real_perf=False):

    org_score_res, score, coords, coord2onset = load_score(dir, piece, scale_factor)

    spec, onsets, coords_new, interpol_fnc = load_performance(dir, piece, spectrogram_params, coords, coord2onset,
                                                              sf_path=sf_path, tempo_factor=tempo_factor,
                                                              real_perf=real_perf)

    return org_score_res, score, spec, interpol_fnc, onsets


def load_score(path, piece, scale_factor=3,):
    npzfile = np.load(os.path.join(path, 'score', piece + '.npz'), allow_pickle=True)

    score, coords, coord2onset = npzfile["sheet"], npzfile["coords"], npzfile['coord2onset']

    org_score_res = np.array(np.copy(score), dtype=np.float32) / 255.

    org_score_res = cv2.cvtColor(org_score_res, cv2.COLOR_GRAY2BGR)
    score = 1 - np.array(score, dtype=np.float32) / 255.
    score = cv2.resize(score, (int(score.shape[1] // scale_factor), int(score.shape[0] // scale_factor)),
                       interpolation=cv2.INTER_AREA)

    coords /= scale_factor

    return org_score_res, score, coords, coord2onset


def load_performance(path, piece, spectrogram_params, coords, coord2onset, sf_path, tempo_factor=1.,
                     real_perf=False, transpose=0):
    if real_perf:
        wav_path = os.path.join(path, 'performance', piece + f'_{tempo_factor}.wav')
        midi_path = os.path.join(path, 'performance', piece + '.mid')
    else:
        if tempo_factor == -1:
            # flag to indicate no tempo factor
            midi_path = os.path.join(path, 'performance', piece + '.mid')
        else:
            midi_path = os.path.join(path, 'performance', piece + f'_tempo_{tempo_factor}.mid')

    midi = mm_midi.MIDIFile(midi_path)

    if transpose != 0:
        notes = midi.notes
        notes[:, 1] += transpose
        midi = mm_midi.MIDIFile.from_notes(notes)

    if real_perf and tempo_factor != -1:
        spec = wav_to_spec_otf(wav_path, spectrogram_params)
    else:
        spec = midi_to_spec_otf(midi, spectrogram_params, sound_font_path=sf_path)

    spec = np.pad(spec, ((0, 0), (spectrogram_params['pad'], 0)), mode='constant')

    onsets = (midi.notes[:, 0] * spectrogram_params['fps']).astype(int)

    onsets, coords_new = merge_onsets(onsets, copy.deepcopy(coords), coord2onset[0])
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return spec, onsets, coords_new, interpol_fnc


def load_game_config(config_file: str) -> dict:
    """Load game config from YAML file."""
    with open(config_file, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def center_of_mass(input):
    """
    adapted from scipy for pytorch
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html?highlight=scipy%20ndimage%20measurements%20center_of_mass
    """
    normalizer = input.sum()

    grids = [torch.arange(input.shape[0]).float().unsqueeze(1).to(normalizer.device),
             torch.arange(input.shape[1]).float().unsqueeze(0).to(normalizer.device)]

    result = torch.cat([((input * grids[dir]).sum() / normalizer).unsqueeze(0) for dir in range(len(grids))])

    return result


class dummy_context(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def __call__(self, func):
        pass