
import cv2
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from audio_conditioned_unet.utils import render_audio
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


def create_video(observation_images, midi_path, piece_name, spectrogram_params, sf_path, path="../videos", real_perf=False):

    if not os.path.exists(path):
        os.mkdir(path)

    if real_perf:
        # midi path will be the wav path
        fn_audio = midi_path
    else:
        fn_audio = render_audio(midi_path, sf_path)

    # frame rate video is now based on the piano roll's frame rate
    path_video = write_video(observation_images, fn_output=os.path.join(path, 'test.mp4'),
                             frame_rate=spectrogram_params['fps'], overwrite=True)

    # mux video and audio with ffmpeg
    mux_video_audio(path_video, fn_audio, path_output=os.path.join(path, '{}.mp4'.format(piece_name)))

    # clean up
    if not real_perf:
        os.remove(fn_audio)

    os.remove(path_video)


def prepare_score_for_render(score, mask, cmap=None):

    if cmap is None:
        cmap = get_transparent_cmap(plt.get_cmap('YlOrBr'), alpha=0.5, n_steps_blend=50)

    if mask.shape[0] != score.shape[0] or mask.shape[1] != score.shape[1]:

        mask = cv2.resize(mask, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = np.array(cmap(mask), dtype=np.float32)

    img = cv2.cvtColor(score, cv2.COLOR_RGB2BGRA)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGRA)

    indices = mask[:, :, 3] > 0.
    img[indices] = cv2.addWeighted(img[indices], 0.5, mask[indices], 0.5, 0)

    return img[:, :, :3], mask


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