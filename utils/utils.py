import cv2
from itertools import count


class Tirf:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        assert self.cap.isOpened(), \
            f'Check if the video path is correct: {path}'

    @property
    def fps(self):
        return self.cap.get(5)

    def readall(self):
        print('Playing video... (press \'q\' to exit)')
        for i in count(0):  # Infinite loop
            ret, frame = self.cap.read()
            if not ret:
                break

            if i == 0:
                assert frame.ndim == 3, \
                    f'Espected video in 3 chans but {frame.ndim}'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            yield i, frame

        self.cap.release()
