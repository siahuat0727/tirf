import os
import pickle
from collections import deque

import matplotlib.pyplot as plt
import cv2

from utils import Tirf

class SmallVideo:
    def __init__(self, info, fps, size=32, n_frame=200,
                 buf=40, name='sample', dir_='.'):
        self.dir = dir_
        self.x_start = info['x'] - size//2
        self.y_start = info['y'] - size//2
        self.x_end = self.x_start + size
        self.y_end = self.y_start + size

        # Extend if the event is longer then n_frame
        self.start_frame = max(
            0, min(info['frame_start'] - buf, info['frame'] - n_frame//2))
        self.end_frame = max(info['frame_end'] + buf,
                             info['frame'] + n_frame//2)
        self.x_min = info['x_min']
        self.x_max = info['x_max']
        self.y_min = info['y_min']
        self.y_max = info['y_max']
        self.time = info['frame'] / fps
        self.coor = info['x'], info['y']
        self.fps = fps
        self.size = (size, size)  # TODO check out of box?

        self.intensity = []

    def set_name(self, name):
        self.name = name  # TODO: better name
        self.title = f"{name}: {self.time:.2f}s {self.coor}"

        # TODO open writer when in active list
        self.path = os.path.join(self.dir, self.name)
        self.video = cv2.VideoWriter(f'{self.path}.avi',
                                     cv2.VideoWriter_fourcc(*'XVID'),
                                     self.fps, self.size)

    def record(self, frame):
        self.video.write(
            frame[self.x_start:self.x_end, self.y_start:self.y_end])
        self.intensity.append(
            frame[self.x_min: self.x_max+1, self.y_min: self.y_max+1].mean())

    def is_start(self, frame_i):
        return frame_i == self.start_frame

    def is_end(self, frame_i):
        return frame_i == self.end_frame

    def terminate(self):
        self.video.release()
        print(f'Save video {self.path}.avi')
        return self.title, self.start_frame, self.intensity


def save_videos(infos, args):

    tirf = Tirf(args.video)
    # tirf = Tirf(args.video, args.x, args.y) ?

    videos = list(sorted([
        SmallVideo(info, tirf.fps, name=f'{i}', dir_=args.output)
        for i, info in enumerate(infos)
    ], key=lambda obj: obj.start_frame))

    for i, vid in enumerate(videos):
        vid.set_name(str(i))

    videos = deque(videos)

    active = []
    plots = []

    for frame_i, frame in enumerate(tirf.readall()):
        if not (videos or active):
            break

        while videos and videos[0].is_start(frame_i):
            active.append(videos.popleft())

        print(f'\r{frame_i} ', end='')

        if not active:
            continue

        for video in active:
            video.record(frame)
            if video.is_end(frame_i):  # TODO off-by-one?
                plots.append(video.terminate())

        active = [video for video in active if not video.is_end(frame_i)]

    return plots


def plot_graph(row, col, plots, i, dir_):
    assert len(plots) <= row*col
    plt.clf()
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(row, col, hspace=0.5, wspace=0)
    axs = gs.subplots(sharey=True)
    for ax, data in zip(axs.flat, plots):
        name, start_frame, intensity = data
        ax.plot(range(start_frame, start_frame+len(intensity)), intensity)
        ax.title.set_text(name)
        print(f'Plot subgraph {name}')

    fig.suptitle('Title')
    path = os.path.join(dir_, f'{i}.png')
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
    plots = save_videos(infos, args)
    plot_graphs(plots, args.output)
