import cv2

BUFFER_LENGTH = 2500

class VideoReader:
    def __init__(self, fn, buffer_length=BUFFER_LENGTH):
        self.fn = fn
        self.cap = cv2.VideoCapture(fn, cv2.CAP_FFMPEG)
        # Check if camera opened successfully
        assert self.cap.isOpened(), "Error opening video stream or file"
        self.ringbuffer = [None] * buffer_length
        self.next_frame = 0
        self.show_again = 0
        self.buffer_length = buffer_length

    def jump_back(self, n):
        '''
        show the n last frames before showing capture frames again.
        Return whether it was possible to jump back n frames
        '''

        ng0 = min(n, self.next_frame)  # 'n greater 0' we can maximum reshow as many frames as we already showed
        new_show_again = min(self.buffer_length, self.show_again + ng0) # we can only reshow as many frames as we have buffer length

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
            frame = self.ringbuffer[self.next_frame % self.buffer_length].copy()
            self.show_again -= 1
        else:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration

            if self.buffer_length > 0:
                self.ringbuffer[self.next_frame % self.buffer_length] = frame.copy()

        self.next_frame += 1

        return frame

    def is_opened(self):
        return self.cap.isOpened()

    def __del__(self):
        self.cap.release()
        print("Video Reader is freed")
