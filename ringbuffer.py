import numpy as np
import cv2

from config import WIDTH, HEIGHT


class Ringbuffer:
    def __init__(self, num_frames):
        if num_frames > 42 * 10:
            raise ValueError('Too many frames for RAM')
        self.num_frames = num_frames
        self.current = 0
        self.data = [None] * self.num_frames

    def store_next_frame(self, frame):
        '''
        :frame: will be written into next frame
        '''
        self.current += 1
        self.current %= self.num_frames
        if isinstance(frame, np.ndarray):
            self.data[self.current] = cv2.imencode('.jpg', frame)[1]

    def __iter__(self):
        for frame_i in list(range(self.current, self.num_frames)) + \
                list(range(0, self.current)):
            yield cv2.imdecode(self.data[frame_i], cv2.IMREAD_COLOR)






