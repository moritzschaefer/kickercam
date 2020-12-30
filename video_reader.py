import cv2


BUFFER_LENGTH = 2500

class VideoReader:
    def __init__(self, fn):
        self.fn = fn
        self.cap = cv2.VideoCapture(fn)
        # Check if camera opened successfully
        assert self.cap.isOpened(), "Error opening video stream or file"
        self.ringbuffer = [None] * BUFFER_LENGTH
        self.next_frame = 0
        self.show_again = 0

    def jump_back(self, n):
        '''
        show the n last frames before showing capture frames again.
        Return whether it was possible to jump back n frames
        '''

        ng0 = min(n, self.next_frame)  # 'n greater 0' we can maximum reshow as many frames as we already showed
        new_show_again = min(BUFFER_LENGTH, self.show_again + ng0) # we can only reshow as many frames as we have buffer length

        if new_show_again != n + self.show_again:
            print('cant jump back soo much')
            ret = False
        else:
            ret = True

        self.next_frame -= (new_show_again - self.show_again)
        self.show_again = new_show_again

        return ret

    def read_next(self):
        'get the next frame'
        if self.show_again > 0:
            frame = self.ringbuffer[self.next_frame % BUFFER_LENGTH].copy()
            self.show_again -= 1
        else:
            ret, frame = self.cap.read()
            self.ringbuffer[self.next_frame % BUFFER_LENGTH] = frame.copy()
            if not ret:
                raise StopIteration

        self.next_frame += 1

        return frame

    def is_opened(self):
        return self.cap.isOpened()

