'''
Load all frames from an encoded video file and store them as compressed numpy file
'''

import argparse
import sys

import cv2
import numpy as np

from video_reader import VideoReader

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('videofile')
    ap.add_argument('outputfile')
    ap.add_argument('--target_height', default=144)
    ap.add_argument('--target_width', default=256)

    num_frames = 120 * 60 * 5
    args = ap.parse_args(sys.argv[1:])
    data = np.ndarray(shape=(num_frames, 3, args.target_width, args.target_height), dtype=np.uint8)

    vr = VideoReader(args.videofile, buffer_length=0)

    i = 0
    while vr.is_opened():
        try:
            frame = vr.read_next()
        except StopIteration:
            break
        data[i] = np.transpose(cv2.resize(frame, (args.target_width, args.target_height)))
        i += 1

    np.savez_compressed(args.outputfile, data[:i])
