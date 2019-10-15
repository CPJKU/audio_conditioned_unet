
import cv2
import os
import time
import tempfile

import numpy as np
import pretty_midi as pm

from scipy import interpolate

def create_ground_truth(score, targets):

    # adaptive squares
    y = np.zeros_like(score)
    for t in targets:
        start, end = t[0], t[-1]

        start[1] -= 2
        end[1] += 2

        if start[0] != end[0]:

            # split ground truth over two staffs
            upper_staff_end = t[t[:, 0] == start[0]][-1]
            lower_staff_start = t[t[:, 0] == end[0]][0]

            y[start[0] - 10:start[0] + 11, start[1]:upper_staff_end[1] + 2] = 1
            y[end[0] - 10:end[0] + 11, lower_staff_start[1] - 2:end[1]] = 1
        else:
            y[end[0] - 10:end[0] + 11, start[1]:end[1] + 1] = 1
    return y


def dice_loss(inputs, targets, smoothing=1.):

    iflat = inputs.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smoothing) / ((iflat**2).sum() + (tflat**2).sum() + smoothing))


def find_matching_patterns(midi, max_frames, pad, fps):

    notes = np.array([[int(note.start * fps) + pad,
                       int(note.end * fps) + pad, note.pitch]
                      for note in midi.instruments[0].notes])
    matches = []

    for frame in range(pad, max_frames):

        for index in range(frame, max_frames):

            pattern = notes[(notes[:, 0] > index - pad) & (notes[:, 0] <= index)][:, -1]
            target_pattern = notes[(notes[:, 0] > frame - pad) & (notes[:, 0] <= frame)][:, -1]

            if len(pattern) == len(target_pattern):

                if (pattern == target_pattern).all():
                    matches.append((frame, index))
                    matches.append((index, frame))

    return np.array(matches)


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


def midi_to_spec_otf(midi: pm.PrettyMIDI, spec_params: dict, sound_font_path=None) -> np.ndarray:
    """MIDI to Spectrogram (on the fly)

       Synthesizes a MIDI with fluidsynth and extracts a spectrogram.
       The spectrogram is directly returned
    """
    processor = spectrogram_processor(spec_params)

    def render_audio(midi_file_path, sound_font):
        """
        Render midi to audio
        """

        # split file name and extention
        name, extention = midi_file_path.rsplit(".", 1)

        # set file names
        audio_file = name + ".wav"

        # audio_file = tempfile.TemporaryFile('w+b')

        # synthesize midi file to audio
        cmd = "fluidsynth -F %s -O s16 -T wav %s %s 1> /dev/null" % (audio_file, sound_font, midi_file_path)

        os.system(cmd)
        return audio_file

    mid_path = os.path.join(tempfile.gettempdir(), str(time.time())+'.mid')

    with open(mid_path, 'wb') as f:
        midi.write(f)

    audio_path = render_audio(mid_path, sound_font=sound_font_path)

    # compute spectrogram
    spec = processor.process(audio_path).T

    os.remove(mid_path)
    os.remove(audio_path)

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


def load_song(path, spectrogram_params, sf_path, tempo_factor=1., scale_factor=3):
    npzfile = np.load(path + '.npz', allow_pickle=True)

    score, coords, coord2onset = npzfile["sheet"], npzfile["coords"], npzfile['coord2onset']

    org_score_res = np.array(np.copy(score), dtype=np.float32) / 255.

    org_score_res = cv2.cvtColor(org_score_res, cv2.COLOR_GRAY2BGR)
    score = 1 - np.array(score, dtype=np.float32) / 255.
    score = cv2.resize(score, (int(score.shape[1] // scale_factor), int(score.shape[0] // scale_factor)),
                       interpolation=cv2.INTER_AREA)

    coords /= scale_factor

    midi = pm.PrettyMIDI(path + '.mid')

    # perform tempo augment
    if tempo_factor != 1:
        for instrument in midi.instruments:
            for note in instrument.notes:
                note.start = note.start * tempo_factor
                note.end = note.end * tempo_factor

    spec = midi_to_spec_otf(midi, spectrogram_params, sound_font_path=sf_path)

    spec = np.pad(spec, ((0, 0), (spectrogram_params['pad'], 0)), mode='constant')

    onsets = (midi.get_onsets() * spectrogram_params['fps']).astype(int)
    onsets, coords = merge_onsets(onsets, coords, coord2onset[0])

    interpol_fnc = interpolate.interp1d(onsets, coords.T, kind='previous', bounds_error=False,
                                        fill_value=(coords[0, :], coords[-1, :]))

    return org_score_res, score, midi, spec, interpol_fnc, onsets

