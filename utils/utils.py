from itertools import count

import cv2
import numpy as np
from nd2reader import ND2Reader as _ND2Reader

class VideoReader:
    def __init__(self, path, x=None, y=None):
        self.path = path
        self.cap = None

    @property
    def fps(self):
        return self.cap.get(5)

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration()
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break
        return frame

    def __iter__(self):
        print('Playing video... (press \'q\' to exit)')
        self.cap = cv2.VideoCapture(path)
        assert self.cap.isOpened(), \
            f'Check if the video path is correct: {path}'


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
