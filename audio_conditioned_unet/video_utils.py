
import cv2
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from matplotlib.colors import ListedColormap


def write_video(images, fn_output='output.mp4', frame_rate=20, overwrite=False):
    """Takes a list of images and interprets them as frames for a video.

    Source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    """
    height, width, _ = images[0].shape

    if overwrite:
        if os.path.exists(fn_output):
            os.remove(fn_output)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fn_output, fourcc, frame_rate, (width, height))

    for cur_image in images:
        frame = cv2.resize(cur_image, (width, height))
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()

    return fn_output


def mux_video_audio(path_video, path_audio, path_output='output_audio.mp4'):
    """Use FFMPEG to mux video with audio recording."""
    from subprocess import check_call

    check_call(["ffmpeg", "-y", "-i", path_video, "-i", path_audio, "-shortest", path_output])


def fluidsynth(midi, fs=44100, sf2_path=None):
    """Synthesize using fluidsynth.
    Copied and adapted from `pretty_midi`.

    Parameters
    ----------
    midi : pm.PrettyMidi
    fs : int
        Sampling rate to synthesize at.
    sf2_path : str
        Path to a .sf2 file.
        Default ``None``, which uses the TimGM6mb.sf2 file included with
        ``pretty_midi``.
    Returns
    -------
    synthesized : np.ndarray
        Waveform of the MIDI data, synthesized at ``fs``.
    """
    # If there are no instruments, or all instruments have no notes, return
    # an empty array
    if len(midi.instruments) == 0 or all(len(i.notes) == 0 for i in midi.instruments):
        return np.array([])
    # Get synthesized waveform for each instrument
    waveforms = []
    for i in midi.instruments:
        if len(i.notes) > 0:
            waveforms.append(i.fluidsynth(fs=fs, sf2_path=sf2_path))

    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))

    # Sum all waveforms in
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform

    # Scale to [-1, 1]
    synthesized /= 2**16
    # synthesized = synthesized.astype(np.int16)

    # normalize
    synthesized /= float(np.max(np.abs(synthesized)))

    return synthesized


def create_video(observation_images, midi, piece_name, spectrogram_params, sf_path):
    fn_audio = '../videos/tmp.wav'

    perf_audio = fluidsynth(midi, fs=spectrogram_params['sample_rate'], sf2_path=sf_path)

    # let it start at the first onset position
    # perf_audio = midi_synth[int(onsets[0] * spectrogram_params['sample_rate'] / spectrogram_params['fps']):]

    # get synthesized MIDI as WAV
    sf.write(fn_audio, perf_audio, spectrogram_params['sample_rate'])

    # frame rate video is now based on the piano roll's frame rate
    path_video = write_video(observation_images, fn_output='../videos/test.mp4',
                             frame_rate=spectrogram_params['fps'], overwrite=True)

    # mux video and audio with ffmpeg
    mux_video_audio(path_video, fn_audio, path_output='../videos/{}.mp4'.format(piece_name))

    # clean up
    os.remove(fn_audio)
    os.remove(path_video)


def prepare_score_for_render(score, mask, cmap=None):

    if cmap is None:
        cmap = get_transparent_cmap(plt.get_cmap('YlOrBr'), alpha=0.5, n_steps_blend=50)
        # cmap = get_transparent_cmap(plt.get_cmap('inferno'), alpha=0.5, n_steps_blend=50)

    mask = cv2.resize(mask, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = np.array(cmap(mask), dtype=np.float32)

    img = cv2.cvtColor(score, cv2.COLOR_RGB2BGRA)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGRA)

    indices = mask[:, :, 3] > 0.
    img[indices] = cv2.addWeighted(img[indices], 0.5, mask[indices], 0.5, 0)

    return img[:, :, :3]


def prepare_spec_for_render(spec, score, scale_factor=5):
    spec_excerpt = cv2.resize(np.flipud(spec), (spec.shape[1] * scale_factor, spec.shape[0] * scale_factor))

    perf_img = np.pad(cm.viridis(spec_excerpt)[:, :, :3],
                      ((score.shape[0] // 2 - spec_excerpt.shape[0] // 2 + 1,
                        score.shape[0] // 2 - spec_excerpt.shape[0] // 2),
                       (20, 20), (0, 0)), mode="constant")

    return perf_img


def get_transparent_cmap(source_cmap, alpha, n_steps_blend):
    transp_cmap = source_cmap(np.arange(source_cmap.N))
    transp_cmap[:n_steps_blend, -1] = np.linspace(0, alpha, n_steps_blend)
    transp_cmap[n_steps_blend:, -1] = alpha
    return ListedColormap(transp_cmap)