'''
Load all frames from an encoded video file and store them as compressed numpy file
'''

import argparse
import sys

import cv2
import numpy as np

from video_reader import VideoReader

def get_greyscale_hsv_filtered(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_b = np.array([0,0,50])
    upper_b = np.array([255,100,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_b, upper_b)
    # Bitwise-AND mask and original image
    frame = cv2.bitwise_and(frame,frame, mask= mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('videofile')
    ap.add_argument('outputfile')
    ap.add_argument('--target_height', default=144)
    ap.add_argument('--target_width', default=256)
    ap.add_argument('--hsv',  action='store_true', default=False)

    num_frames = 120 * 60 * 5
    args = ap.parse_args(sys.argv[1:])
    data = np.ndarray(shape=(num_frames, 3 + args.hsv, args.target_width, args.target_height), dtype=np.uint8)

    vr = VideoReader(args.videofile, buffer_length=0)

    i = 0
    while vr.is_opened():
        try:
            frame = vr.read_next()
        except StopIteration:
            break
        scaled_frame = cv2.resize(frame, (args.target_width, args.target_height))
        if args.hsv:
            grey = np.expand_dims(np.transpose(get_greyscale_hsv_filtered(scaled_frame)), 0)
            data[i] = np.concatenate([scaled_frame.T, grey], axis=0)
        else:
            data[i] = np.transpose(scaled_frame)

        i += 1

    np.savez_compressed(args.outputfile, data[:i])
