'''
Load all frames from an encoded video file and store them as compressed numpy file
'''

import argparse
import sys

import cv2
import numpy as np

from video_reader import VideoReader


def post_process_frame(frame, use_rgb=True, use_gray=False, target_width=256, target_height=144, mean=None):
    scaled_frame = cv2.resize(frame, (target_width, target_height))
    if use_gray:
        hsv = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2HSV)
        lower_b = np.array([0, 0, 50])
        upper_b = np.array([255, 100, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_b, upper_b)
        # Bitwise-AND mask and original image
        masked_frame = cv2.bitwise_and(scaled_frame, scaled_frame, mask=mask)
        gray = np.expand_dims(cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY).T,0)

    if use_rgb and use_gray:
        result = np.concatenate([scaled_frame.T, gray], axis=0)
    elif use_gray:
        result = gray
    else:
        result = scaled_frame.T
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('videofile')
    ap.add_argument('outputfile')
    ap.add_argument('--target_height', default=144)
    ap.add_argument('--target_width', default=256)
    ap.add_argument('--gray',  action='store_true', default=False)
    ap.add_argument('--rgb', action='store_false', default=True)
    num_frames = 120 * 60 * 6
    args = ap.parse_args(sys.argv[1:])

    data = np.ndarray(shape=(num_frames, 1 * args.gray + 3 * (args.rgb), args.target_width, args.target_height), dtype=np.uint8)

    vr = VideoReader(args.videofile, buffer_length=0)

    i = 0
    while vr.is_opened():
        try:
            frame = vr.read_next()
        except StopIteration:
            break
        data[i] = post_process_frame(frame,args.rgb,args.gray,args.target_width, args.target_height)
        i += 1

    np.savez_compressed(args.outputfile, data[:i])
