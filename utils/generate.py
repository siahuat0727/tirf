import os
import pickle
from copy import deepcopy
from collections import deque
from pathlib import Path
from PIL import Image
import subprocess
import numpy as np


import matplotlib.pyplot as plt
import cv2

from .utils import select_reader

class SmallVideo:
    def __init__(self, info, frame_size, fps, size=32, n_frame=200,
                 buf=40, name='sample', dir_='.'):
        self.dir = dir_
        self.x_start = max(0, info['x'] - size//2)
        self.y_start = max(0, info['y'] - size//2)

        x_max, y_max = frame_size
        self.x_end = min(self.x_start + size, x_max)
        self.y_end = min(self.y_start + size, y_max)

        # Extend if the event is longer then n_frame
        self.start_frame = max(
            0, min(info['frame_start'] - buf, info['frame'] - n_frame//2))
        self.end_frame = max(info['frame_end'] + buf,
                             info['frame'] + n_frame//2)
        self.x_min = info['x_min']
        self.x_max = info['x_max']
        self.y_min = info['y_min']
        self.y_max = info['y_max']
        self.n_points = info['n_points']
        self.time = info['frame'] / fps
        self.coor = info['x'], info['y']
        self.fps = fps
        self.size = (self.x_end-self.x_start, self.y_end - self.y_start)  # TODO check out of box?

        self.intensity = []
        self.frame_i = 0

    def set_name(self, name):
        self.name = name  # TODO: better name

        n_points = ','.join(map(str, self.n_points))
        self.title = f"{name}: {self.time:.2f}s {self.coor} ({n_points} points)"

        # TODO open writer when in active list
        self.path = os.path.join(self.dir, self.name)
        self.video = cv2.VideoWriter()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
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

        def save_image():
            Path(self.path).mkdir(parents=True, exist_ok=True)
            img_path = f'{self.path}/{self.frame_i:04d}.png'
            self.frame_i += 1
            image = draw_rect(frame, self.x_min, self.x_max, self.y_min, self.y_max)
            image_cc = draw_rect(cc_frame, self.x_min, self.x_max, self.y_min, self.y_max)

            image = image[self.x_start:self.x_end, self.y_start:self.y_end]
            image_cc = image_cc[self.x_start:self.x_end, self.y_start:self.y_end]
            image = np.concatenate((image, image_cc), axis=1)
            Image.fromarray(image).save(img_path)

        save_image()

        # self.video.write(
        #     frame[self.x_start:self.x_end, self.y_start:self.y_end])

        self.intensity.append(
            frame[self.x_min: self.x_max+1, self.y_min: self.y_max+1].mean())

    def is_start(self, frame_i):
        return frame_i == self.start_frame

    def is_end(self, frame_i):
        return frame_i == self.end_frame

    def terminate(self):
        # self.video.release()
        # print(f'Save video {self.path}.avi')

        cmd = f'ffmpeg -r 10 -i {self.path}/%04d.png -vcodec libx264 {self.path}.mp4'

        subprocess.call(cmd, shell=True)

        return self.title, self.start_frame, self.intensity


def save_videos(infos, cc, args):

    assert args.fps is not None and args.fps > 0

    def get_videos(frame_size):
        videos = list(sorted([
            SmallVideo(info, frame_size, args.fps, name=f'{i}', n_frame=args.n_frame, dir_=args.output)
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


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
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

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


def plot_graph(row, col, plots, i, dir_):
    threshold = 5
    lag = 10
    assert len(plots) <= row*col
    plt.clf()
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(row, col, hspace=0.5, wspace=0)
    axs = gs.subplots(sharey=True)
    for ax, data in zip(axs.flat, plots):
        name, start_frame, intensity = data

        result = thresholding_algo(intensity[::-1],
                                   lag=lag,
                                   threshold=threshold,
                                   influence=0.4)
        x_axis = list(range(start_frame, start_frame+len(intensity)))
        low_bound = result["avgFilter"] - threshold * result["stdFilter"]
        high_bound = result["avgFilter"] + threshold * result["stdFilter"]

        ax.plot(x_axis, intensity)
        ax.plot(x_axis[:-lag], high_bound[lag:][::-1], color="green", lw=1)
        ax.plot(x_axis[:-lag], low_bound[lag:][::-1], color="green", lw=1)
        ax.step(x_axis[:-lag], 10*result["signals"][lag:][::-1], color="red", lw=1)
        ax.title.set_text(name)
        print(f'Plot subgraph {name}')

    fig.suptitle('Title')
    path = os.path.join(dir_, f'{i}.png')
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(path)
    print(path)


def plot_graphs(plots, dir_, row=4, col=4):
    plots = [
        plots[i: i+row*col]
        for i in range(0, len(plots), row*col)
    ]
    for i, data in enumerate(plots):
        plot_graph(row, col, data, i, dir_)


def generate(args):
    with open(args.pkl, 'rb') as f:
        infos = pickle.load(f)
    with open(args.cc, 'rb') as f:
        cc = pickle.load(f)
    plots = save_videos(infos, cc, args)
    plot_graphs(plots, args.output)
