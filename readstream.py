#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Moritz
"""
import io 
from collections import deque
import numpy as np
import cv2
import scipy
from image_processor import ProcessOutput
from goal_detector import GoalDetector


def main():
    po = ProcessOutput()
    gd = GoalDetector()
    i = 0
    videopath = './match2.h264'
    camera = cv2.VideoCapture(videopath)
    print(np.shape(camera))
    (grabbed, frame) = camera.read()

    while grabbed:
        print("iteration", i)

        is_success, buf = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 99])
        io_buf = io.BytesIO(buf)

        #frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gd.step(hsv)

        po.write(io_buf.getvalue())

        cv2.imshow('image', hsv)
        cv2.waitKey(1)

        (grabbed, frame) = camera.read()
        i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
