import cv2
from itertools import count


class Tirf:
    def __init__(self, path, x=None, y=None):
        self.cap = cv2.VideoCapture(path)
        assert self.cap.isOpened(), \
            f'Check if the video path is correct: {path}'
        self.x = x
        self.y = y

    @property
    def fps(self):
        return self.cap.get(5)

    def readall(self):

        def _readall():
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

                yield frame
            self.cap.release()

        if self.x is None and self.y is None:
            return _readall()

        assert len(self.x) == len(self.y) == 2
        x0, x1 = self.x
        y0, y1 = self.y
        return (frame[x0:x1, y0:y1] for frame in _readall())
