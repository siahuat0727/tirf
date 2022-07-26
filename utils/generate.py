import os
import csv
import pickle
from copy import deepcopy
from collections import deque
from pathlib import Path
from PIL import Image
import subprocess
import numpy as np

import matplotlib.pyplot as plt
import cv2

from .readers import select_reader


class SmallVideo:
    # TODO rename size to width
    def __init__(self, info, frame_size, fps, size=32, n_frame_before=50,
                 n_frame_after=10, name='sample', dir_='.', for_anotation=False):
        self.dir = dir_
        self.center = info['x'], info['y']
        self.size = size
        self.for_anotation = for_anotation

        # Frame slice
        self.n_frame_after = n_frame_after
        # self.start_frame = max(0, info['frame_end'] - n_frame_before)
        self.start_frame = info['frame_end'] - n_frame_before
        self.end_frame = info['frame_end'] + n_frame_after
        self.total_frame = n_frame_before + n_frame_after + 1


        self.x_min = info['x_min']
        self.x_max = info['x_max']
        self.y_min = info['y_min']
        self.y_max = info['y_max']
        self.n_points = info['n_points']
        self.time = info['frame'] / fps
        self.coor = info['x'], info['y'], info['frame_end']
        self.fps = fps

        self.intensity = []
        self.image_count = 0

    def set_name(self, name):
        self.name = name  # TODO: better name

        n_points = ','.join(map(str, self.n_points))
        title = f"{name}: {self.time:.2f}s {self.coor} ({n_points} points)"

        max_len = 35
        self.title = '\n'.join(
            title[i:i+max_len]
            for i in range(0, len(title), max_len)
        )

        self.path = os.path.join(self.dir, self.name)
        Path(self.path).mkdir(parents=True, exist_ok=True)

        if self.start_frame < 0:
            self.add_padding_image(-self.start_frame)
            self.start_frame = 0

        # TODO open writer when in active list
        # self.video = cv2.VideoWriter()
        # ok = self.video.open(f'{self.path}.mov',
        #                              # cv2.VideoWriter_fourcc(*'XVID'),
        #                              # cv2.VideoWriter_fourcc(*'MPEG'),
        #                              cv2.VideoWriter_fourcc(*'mp4v'),
        #                              self.fps, self.size, True)

    def record(self, frame, cc_frame):
        def draw_rect(image, x_lo, x_hi, y_lo, y_hi):
            x_lo = max(0, x_lo-1)
            y_lo = max(0, y_lo-1)
            x_hi = min(image.shape[0]-1, x_hi+1)
            y_hi = min(image.shape[1]-1, y_hi+1)
            image = deepcopy(image)
            image[x_lo, y_lo] = 255
            image[x_hi, y_hi] = 255
            return image

        def get_two_ends(center, size):
            return center - size//2, center + size//2

        def crop_image(image, center, size: int):
            """
            Generate a croped numpy image centered on the center with size.
            If the size is out of bound, add zero padding to ensure center is
            in the image center.
            """
            x_lo, x_hi = get_two_ends(center[0], size)
            y_lo, y_hi = get_two_ends(center[1], size)
            x_lo_pad = max(0, x_lo) - x_lo
            x_hi_pad = x_hi - min(image.shape[0]-1, x_hi)
            y_lo_pad = max(0, y_lo) - y_lo
            y_hi_pad = y_hi - min(image.shape[1]-1, y_hi)
            image = image[max(0, x_lo):min(image.shape[0]-1, x_hi), max(0, y_lo):min(image.shape[1]-1, y_hi)]
            image = np.pad(image, ((x_lo_pad, x_hi_pad), (y_lo_pad, y_hi_pad)), 'constant')
            assert image.shape == (size, size), f'{image.shape} != {size, size}'
            return image


        def get_concatenated_image(frame, cc_frame):
            image = draw_rect(frame, self.x_min, self.x_max,
                              self.y_min, self.y_max)
            image_cc = draw_rect(cc_frame, self.x_min,
                                 self.x_max, self.y_min, self.y_max)
            image = crop_image(image, self.center, self.size)
            image_cc = crop_image(image_cc, self.center, self.size)
            return np.concatenate((image, image_cc), axis=1)

        if self.for_anotation:
            image = get_concatenated_image(frame, cc_frame)
        else:
            image = crop_image(frame, self.center, self.size)

        self.save_image(image)

        self.intensity.append(
            frame[self.x_min: self.x_max+1, self.y_min: self.y_max+1].mean())

    def add_padding_image(self, n_padding_frame):
        """ Add n_padding_frame of zero padding image to the video. """
        for _ in range(n_padding_frame):
            self.save_image(np.zeros((self.size, self.size), dtype=np.uint8))

    def save_image(self, image):
        save_image(f'{self.path}/{self.image_count:04d}.png', image)
        self.image_count += 1

    def is_start(self, frame_i):
        return frame_i == self.start_frame

    def is_end(self, frame_i):
        return frame_i == self.end_frame

    def terminate(self):
        # self.video.release()
        # print(f'Save video {self.path}.avi')

        self.add_padding_image(self.total_frame - self.image_count)

        cmd = f'ffmpeg -r 10 -i {self.path}/%04d.png -vcodec libx264 {self.path}.mp4'

        # subprocess.call(cmd, shell=True)

        return {
            'x': self.coor[0],
            'y': self.coor[1],
            'z': self.coor[2],
            'title': self.title,
            'intensity': self.intensity,
            'n_frame_after': self.n_frame_after,
            'n_points': self.n_points,
        }


def save_image(img_path, image):
    Image.fromarray(image).save(img_path)

def save_videos(infos, cc, args, out_dir):

    def get_videos(frame_size):
        videos = list(sorted([
            SmallVideo(info, frame_size, args.fps, size=args.crop_size,
                       name=str(i),
                       n_frame_before=args.n_frame_before,
                       n_frame_after=args.n_frame_after,
                       dir_=out_dir / 'frames',
                       for_anotation=args.for_anotation)
            for i, info in enumerate(infos)
        ], key=lambda obj: obj.start_frame))

        for i, vid in enumerate(videos):
            vid.set_name(str(i))

        return videos

    active = []
    plots = []

    reader = select_reader(args.input_type)
    generator = reader(args.input)
    videos = None

    cc = cc.astype(np.uint8)

    for frame_i, (frame, cc_frame) in enumerate(zip(generator, cc)):
        if videos is None:
            videos = get_videos(frame.shape)[::-1]

        if not (videos or active):
            break

        while videos and videos[-1].is_start(frame_i):
            active.append(videos.pop())

        # print(f'\r{frame_i} ', end='')

        if not active:
            continue

        for video in active:
            video.record(frame, cc_frame)
            if video.is_end(frame_i):  # TODO off-by-one?
                plots.append(video.terminate())

        active = [video for video in active if not video.is_end(frame_i)]

    return plots


def thresholding_algo(y, lag, threshold, influence, min_std=0.0, reverse=False):
    """
    min_std: Minimum std
    """
    if reverse:
        y = y[::-1]
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        stdFilter[i] = max(min_std, stdFilter[i])

    if reverse:
        signals = signals[::-1]
        avgFilter = avgFilter[::-1]
        stdFilter = stdFilter[::-1]

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))


def ax_plot(ax, x_axis, intensity, intensity_diff, low_bound, high_bound, signal, title, is_candidate, lag=5, **_kwargs):
    ax.set_ylim(-11, 180)
    ax.plot(x_axis, intensity)
    ax.plot(x_axis[:-lag], high_bound[:-lag], color="green", lw=1)
    ax.plot(x_axis[:-lag], low_bound[:-lag], color="green", lw=1)
    ax.plot(x_axis[:-1], intensity_diff, color="cyan", lw=1)

    ax.step(x_axis[:-lag], 10*signal[:-lag], color="red", lw=1)

    intensity_diff = (intensity_diff > 8).astype(int)
    ax.step(x_axis[:-1], 5*intensity_diff + 1, color="black", lw=1)

    ax.plot(x_axis, intensity)
    if is_candidate:
        title = f'Good {title}'

    ax.title.set_text(title)
    print(f'Plot subgraph {title}')


def data_to_plot_info(data, threshold, lag, influence):
    title = data['title']
    intensity = data['intensity']
    n_frame_after = data['n_frame_after']

    result = thresholding_algo(intensity,
                               lag=lag,
                               threshold=threshold,
                               influence=influence,
                               min_std=1.5,
                               reverse=True)

    x_axis = list(range(len(intensity)))
    low_bound = result["avgFilter"] - threshold * result["stdFilter"]
    high_bound = result["avgFilter"] + threshold * result["stdFilter"]
    signal = result["signals"]

    # TODO check off-by-one
    is_candidate = np.sum(result["signals"][-n_frame_after-3:-n_frame_after+3] == 1) > 0

    intensity_diff = np.asarray(intensity[:-1]) - np.asarray(intensity[1:])
    is_drastically_drop = np.sum(intensity_diff[-n_frame_after-2:-n_frame_after+3] > 8) > 0

    return {
        'x': data['x'],
        'y': data['y'],
        'z': data['z'],
        'x_axis': x_axis,
        'intensity': intensity,
        'intensity_diff': intensity_diff,
        'event_intensity_diff': intensity_diff[-n_frame_after-2:-n_frame_after+3],
        'low_bound': low_bound,
        'high_bound': high_bound,
        'signal': signal,
        'title': title,
        'is_candidate': is_candidate,
        'is_drastically_drop': is_drastically_drop,
        'n_points': data['n_points'],
    }


def plot_graph(row, col, infos, lag):

    assert len(infos) <= row*col
    plt.clf()
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(row, col, hspace=0.5, wspace=0)
    axs = gs.subplots(sharey=True)

    for ax, info in zip(axs.flat, infos):
        ax_plot(ax, **info, lag=lag)

    fig.suptitle('Title')
    return fig


def plot_graphs(infos, lag, row=4, col=4, dir_='.'):
    for i, info_i in enumerate(range(0, len(infos), row*col)):
        fig = plot_graph(row, col, infos[info_i: info_i+row*col], lag)
        path = str(dir_ / f'{i}.png')
        plt.savefig(path)
        print(path)


def plot(args, datas, dir_, row=4, col=4, threshold=5, lag=5, influence=0.9):
    info_dir = dir_ / 'infos'
    dir_all = info_dir / 'all'
    dir_all.mkdir(parents=True, exist_ok=True)

    dir_candidate = info_dir / 'candidate'
    dir_candidate.mkdir(parents=True, exist_ok=True)

    dir_good_candidate = info_dir / 'good_candidate'
    dir_good_candidate.mkdir(parents=True, exist_ok=True)

    all_infos = [
        data_to_plot_info(data, threshold, lag, influence)
        for data in datas
    ]

    def remove_top_right(info):
        return not (info['x'] < args.rm_x and info['y'] > args.rm_y)

    def remove_top_left(info):
        return not (info['x'] < args.rm_x and info['y'] < args.rm_y)

    def remove_custom(infos, remove):
        return list(filter(remove_top_right, infos))

    if args.remove is not None:
        assert args.rm_x is not None
        assert args.rm_y is not None
        remove_func = {
            'top_left': remove_top_left,
            'top_right': remove_top_right,
        }.get(args.remove)
        all_infos = remove_custom(all_infos, remove_func)

    plot_graphs(all_infos, lag, dir_=dir_all)

    candidate_infos = [
        info
        for info in all_infos
        if info['is_candidate']
    ]
    plot_graphs(candidate_infos, lag, dir_=dir_candidate)

    drastic_drop_candidate_infos = [
        info
        for info in candidate_infos
        if info['is_drastically_drop']
    ]
    plot_graphs(drastic_drop_candidate_infos, lag, dir_=dir_good_candidate)

    candidate_infos = enrich_neighbor_info(candidate_infos, all_infos)
    generate_dataset(candidate_infos, dir_=dir_)

    print(f'Number of all candidates: {len(all_infos)}')
    print(f'Number of good candidates: {len(candidate_infos)}')
    print(f'Number of good candidate with drastically drop: {len(drastic_drop_candidate_infos)}')
    # print(f'Number of super candidates: {len(super_candidate_info)}')


def generate_dataset(candidate_infos, dir_):

    def generate_video(info, name):
        frame_dir = dir_ / 'frames' / name
        cmd = f'ffmpeg -r 10 -i "{frame_dir}/%04d.png" -vcodec libx264 "{dir_ / name}.mp4"'
        subprocess.call(cmd, shell=True)

    # CSV writer
    with open(dir_ / 'label.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'label', 'intensity_diff', 'n_neighbor', 'x,y,z', 'n_points'])
        for info in candidate_infos:
            name = info['title'].split(':')[0]
            generate_video(info, name)
            writer.writerow([
                name,
                '',
                ', '.join(f'{v:.2f}' for v in info['event_intensity_diff']),
                info['n_neighbor'],
                ','.join([str(info['x']), str(info['y']), str(info['z'])]),
                sum(info['n_points']),
            ])


def enrich_neighbor_info(good_infos, all_infos):
    xs = np.array([info['x'] for info in all_infos])
    ys = np.array([info['y'] for info in all_infos])
    zs = np.array([info['z'] for info in all_infos])

    def count_event_around(info):
        x, y, z = info['x'], info['y'], info['z']
        z_range = 10

        xy_dis = np.sqrt(((xs - x)**2) + ((ys - y)**2))
        inside_z = (zs < z+z_range) & (zs > z-z_range)
        inside_xy = (xy_dis >= 3) & (xy_dis < 20)
        print(np.sum(inside_z & inside_xy),
              ' '.join(info['title'].split('\n')))
        print()
        return np.sum(inside_z & inside_xy)

    for info in good_infos:
        info['n_neighbor'] = count_event_around(info)
    return good_infos


def generate(args, out_dir):
    assert args.fps is not None and args.fps > 0

    info_dir = out_dir / 'infos'
    with open(info_dir / args.pkl, 'rb') as f:
        infos = pickle.load(f)
    with open(info_dir / args.cc, 'rb') as f:
        cc = pickle.load(f)
    datas = save_videos(infos, cc, args, out_dir)
    plot(args, datas, out_dir)
