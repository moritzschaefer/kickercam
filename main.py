#!/usr/bin/python3 -u
"""
Runs kickercam and goal detection etc

"""
import random
import picamera
import cv2
import ipdb
import numpy as np
from time import sleep
from signal import pause

from image_processor import ProcessOutput

ONLY_PREVIEW = True

def main():
    with picamera.PiCamera() as camera:
        camera.resolution = (1600, 1200)
        camera.framerate = 30
        camera.start_preview()
        sleep(1)

        if ONLY_PREVIEW:
            pause()

        stream = picamera.PiCameraCircularIO(camera, seconds=7)
        camera.start_recording(stream,'h264')  # TODO use mjpeg!
        output = ProcessOutput()
        camera.start_recording(output,'mjpeg', splitter_port=2)

        try:
            while not output.done:
                camera.wait_recording(1)
                if output.detected:
                    output.detected = False
                    camera.stop_recording(splitter_port=2)
                    camera.stop_recording()
                    stream.copy_to('motion.h264')
                    camera.stop_preview()
                    cv2.namedWindow('replay', cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty('replay',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                    cap = cv2.VideoCapture('motion.h264')
                    try:
                        while(cap.isOpened()):
                            ret, frame = cap.read()
                            cv2.imshow('replay',frame)
                            cv2.waitKey(1)
                        cap.release()
                    except cv2.error:
                        cap.release()
                    finally:
                        cv2.destroyAllWindows()
                    camera.start_preview()
                    camera.start_recording(stream,'h264')
                    camera.start_recording(output,'mjpeg', splitter_port=2)
        finally:
            camera.stop_recording()
            camera.stop_recording(splitter_port=2)

            camera.stop_preview()


main()
