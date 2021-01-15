#/usr/bin/env python
'''
Load all frames from an encoded video file and store them as compressed numpy
file
'''

import argparse
import sys

import cv2
import numpy as np

from .preprocessing import process_frame
from .video_reader import VideoReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videofile')
    ap.add_argument('outputfile')
    ap.add_argument('--target_height', default=144)
    ap.add_argument('--target_width', default=256)
    ap.add_argument('--gray',  action='store_true', default=False)
    ap.add_argument('--rgb', action='store_false', default=True)
    num_frames = 120 * 60 * 6
    args = ap.parse_args(sys.argv[1:])

    data = np.ndarray(shape=(num_frames, 1 * args.gray + 3 * (args.rgb),
                             args.target_width, args.target_height),
                      dtype=np.uint8)

    vr = VideoReader(args.videofile, buffer_length=0)

    i = 0
    while vr.is_opened():
        try:
            frame = vr.read_next()
        except StopIteration:
            break
        data[i] = process_frame(frame, args.rgb, args.gray,
                                (3, args.target_width, args.target_height))
        i += 1

    np.savez_compressed(args.outputfile, data[:i])


if __name__ == '__main__':
    main()
