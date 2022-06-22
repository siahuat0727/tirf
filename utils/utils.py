from itertools import count

import cv2
import numpy as np
from nd2reader import ND2Reader as _ND2Reader

# TODO refactor readers! just simply return a list of images

class VideoReader:
    def __init__(self, path, reverse=False, x=None, y=None):
        self.path = path
        self.cap = None
        self.reverse = reverse

    @property
    def fps(self):
        return self.cap.get(5)

    def __next__(self):
        if self.image_i == len(self.images):
            raise StopIteration()
        res = self.images[self.image_i]
        self.image_i += 1
        return res

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened(), \
            f'Check if the video path is correct: {path}'
        self.images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            if len(frame.shape) == 3:
                assert frame.shape[2] == 3, frame.shape
                frame = frame[:, :, 0]
            self.images.append(frame)
        if self.reverse:
            self.images = self.images[::-1]
        self.image_i = 0
        return self


class ND2Reader:
    def __init__(self, path, reverse=False):
        self.path = path
        self.images = None
        self.image_i = None
        self.reverse = reverse

    def __next__(self):
        if self.image_i == len(self.images):
            raise StopIteration()
        res = self.images[self.image_i]
        self.image_i += 1
        return res

    def __iter__(self):

        with _ND2Reader(self.path) as reader:
            images = np.array(list(reader))
        if self.reverse:
            images = images[::-1]
        self.images = ((images / np.max(images)) * 255).astype(np.uint8)
        self.image_i = 0
        return self



def select_reader(in_type):
    return {
        'video': VideoReader,
        'nd2': ND2Reader,
        # 'images': ImagesReader,
    }[in_type]
