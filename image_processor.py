import io
import threading
from PIL import Image
from time import sleep, time
import numpy as np

from goal_detector import GoalDetector
from classifier import Classifier


class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.gd = GoalDetector()
        self.classifier = Classifier('trainingdata')
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    image = Image.open(self.stream)
                    hsv = np.array(image.convert('HSV'))
                    obs, goal_diff = self.gd.step(hsv)
                    for i, obstacles in enumerate(obs):
                        for obstacle in obstacles:
                            print('obstacle detected')
                            if self.classifier.predict(obstacle, hsv, self.gd.goal_rects[i]):
                                print('goal detected!!')
                                self.owner.detected = True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)

class ProcessOutput(object):
    def __init__(self):
        self.done = False
        self.detected = False
        # Construct a pool of 3 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(3)]
        self.processor = None
        self.i = 0

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        try:
            while True:
                with self.lock:
                    proc = self.pool.pop()
                proc.terminated = True
                proc.join()

        except IndexError:
            pass # pool is empty
